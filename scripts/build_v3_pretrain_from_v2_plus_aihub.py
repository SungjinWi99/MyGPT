"""Build a v3 pretraining dataset by extending an existing v2 dataset with AIHub.

This script is intended for Colab. It copies an immutable prepared v2 pretrain
dataset, converts selected AIHub source data into additional Parquet shards, and
updates profile/manifest metadata in the copied v3 dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import random
import shutil
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_pipeline.adapters import (
    iter_aihub_commonsense_sentence_zip,
    iter_aihub_korean_llm_zip,
)
from src.dataset_pipeline.builder import (
    PARQUET_SCHEMA,
    Profile,
    SQLiteDeduper,
    _profile_dict,
    deterministic_split,
    normalize_text,
    sha256_file,
)
from src.dataset_pipeline.prepare_pretrain import (
    _aihub_commonsense_sentence_paths,
    _aihub_korean_llm_paths,
    _file_inventory,
)
from src.dataset_pipeline.schema import RejectedRecord, SourceDocument


DEFAULT_DATASET_ROOT = Path("/content/drive/MyDrive/KTB/MyGPT/datasets/pretrain")
DEFAULT_AIHUB_ROOT = Path("/content/drive/MyDrive/KTB/datasets/AIHUB")
PROFILE_TOTAL_KEYS = (
    "raw_records",
    "accepted_records",
    "accepted_bytes_utf8",
    "accepted_tokens",
    "train_records",
    "validation_records",
)


class ParquetAppendWriter:
    def __init__(self, root: Path, *, shard_rows: int, prefix: str = "aihub"):
        self.root = root
        self.shard_rows = shard_rows
        self.prefix = prefix
        self.buffers: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "validation": [],
        }
        self.shard_indexes = {"train": 0, "validation": 0}
        self.files: list[Path] = []

    def add(self, row: dict[str, Any]) -> None:
        split = row["split"]
        self.buffers[split].append(row)
        if len(self.buffers[split]) >= self.shard_rows:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        split_dir = self.root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        path = split_dir / f"{self.prefix}-part-{self.shard_indexes[split]:05d}.parquet"
        table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
        pq.write_table(
            table,
            path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        self.files.append(path)
        self.shard_indexes[split] += 1
        self.buffers[split] = []

    def close(self) -> list[Path]:
        self.flush("train")
        self.flush("validation")
        return self.files


class AIHubAppender:
    def __init__(
        self,
        *,
        tokenizer: Any,
        deduper: SQLiteDeduper,
        output_parquet_root: Path,
        validation_fraction: float,
        shard_rows: int,
        tokenize_batch_size: int,
        sample_size: int,
        max_accepted_per_source: int | None,
    ):
        self.tokenizer = tokenizer
        self.deduper = deduper
        self.writer = ParquetAppendWriter(
            output_parquet_root,
            shard_rows=shard_rows,
            prefix="aihub",
        )
        self.validation_fraction = validation_fraction
        self.tokenize_batch_size = tokenize_batch_size
        self.sample_size = sample_size
        self.max_accepted_per_source = max_accepted_per_source
        self.profiles: dict[str, Profile] = {}
        self.rngs: dict[str, random.Random] = {}
        self.samples: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        self.pending: list[tuple[str, SourceDocument, bytes, str]] = []

    def profile(self, source: str) -> Profile:
        if source not in self.profiles:
            self.profiles[source] = Profile()
            seed = int.from_bytes(hashlib.sha256(source.encode()).digest()[:8], "big")
            self.rngs[source] = random.Random(seed)
            self.samples[source] = []
        return self.profiles[source]

    def add_sample(self, source: str, score: int, row: dict[str, Any]) -> None:
        heap = self.samples[source]
        item = (-score, row)
        if len(heap) < self.sample_size:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def flush_pending(self) -> None:
        if not self.pending:
            return
        texts = [document.text for _, document, _, _ in self.pending]
        encodings = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )

        for pending, token_ids in zip(self.pending, encodings["input_ids"]):
            stream_name, document, document_hash, split = pending
            token_count = len(token_ids)
            row = {
                "id": f"{document.source}:{document.source_id}",
                "source": document.source,
                "source_id": document.source_id,
                "text": document.text,
                "title": document.title,
                "url": document.url,
                "license": document.license,
                "language": document.language,
                "year": document.year,
                "corpus": document.corpus,
                "metadata_json": json.dumps(
                    document.metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
                "document_sha256": document_hash.hex(),
                "token_count": token_count,
                "split": split,
            }
            self.writer.add(row)

            profile = self.profile(stream_name)
            profile.accept(
                document.text,
                token_count,
                split,
                self.rngs[stream_name],
                reservoir_size=100_000,
                upstream_source=str(document.metadata.get("upstream_source") or "")
                or None,
            )
            sample_row = {
                key: row[key]
                for key in (
                    "id",
                    "source",
                    "title",
                    "url",
                    "language",
                    "year",
                    "corpus",
                    "token_count",
                    "split",
                )
            }
            sample_row["text_preview"] = document.text[:1000]
            self.add_sample(
                stream_name,
                int.from_bytes(document_hash[:8], "big"),
                sample_row,
            )

        self.pending = []

    def process(
        self,
        stream_name: str,
        records: Iterable[SourceDocument | RejectedRecord],
    ) -> None:
        profile = self.profile(stream_name)
        progress = tqdm(records, desc=f"AIHub {stream_name}", unit="records")
        for record in progress:
            if (
                self.max_accepted_per_source is not None
                and profile.accepted_records
                + sum(1 for name, *_ in self.pending if name == stream_name)
                >= self.max_accepted_per_source
            ):
                break

            profile.raw_records += 1
            if isinstance(record, RejectedRecord):
                profile.reject(record.reason)
                continue

            text = normalize_text(record.text)
            if not text:
                profile.reject("empty_after_normalization")
                continue

            document_hash = hashlib.sha256(text.encode("utf-8")).digest()
            if not self.deduper.add(document_hash):
                profile.reject("exact_duplicate")
                continue

            record.text = text
            split = deterministic_split(document_hash, self.validation_fraction)
            self.pending.append((stream_name, record, document_hash, split))
            if len(self.pending) >= self.tokenize_batch_size:
                self.flush_pending()
        self.flush_pending()

    def close(self) -> list[Path]:
        self.flush_pending()
        return self.writer.close()

    def profile_payload(self) -> dict[str, Any]:
        return {
            name: _profile_dict(profile)
            for name, profile in sorted(self.profiles.items())
        }

    def sample_rows(self) -> list[dict[str, Any]]:
        rows = []
        for source, heap in sorted(self.samples.items()):
            for _, row in sorted(heap, reverse=True):
                rows.append({"profile_source": source, **row})
        return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create v3 pretrain dataset by copying v2 and appending AIHub data."
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "v2",
        help="Existing prepared pretrain dataset directory.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "v3",
        help="New immutable output dataset directory.",
    )
    parser.add_argument(
        "--aihub-root",
        type=Path,
        default=DEFAULT_AIHUB_ROOT,
        help="Google Drive AIHub root directory.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/content/mygpt_dataset_work"),
    )
    parser.add_argument("--include-validation-source", action="store_true")
    parser.add_argument("--tokenizer-name")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--shard-rows", type=int, default=100_000)
    parser.add_argument("--tokenize-batch-size", type=int, default=128)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--max-accepted-per-source", type=int)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parquet_paths(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "parquet").glob("*/*.parquet"))


def seed_existing_hashes(
    deduper: SQLiteDeduper,
    dataset_dir: Path,
    *,
    batch_size: int = 50_000,
) -> int:
    if deduper.connection is None:
        raise RuntimeError("Deduper is closed")

    inserted = 0
    batch: list[tuple[bytes]] = []
    paths = parquet_paths(dataset_dir)
    for path in tqdm(paths, desc="Reading existing v2 hashes", unit="files"):
        parquet_file = pq.ParquetFile(path)
        if "document_sha256" not in parquet_file.schema.names:
            raise ValueError(f"Missing document_sha256 column in {path}")
        for record_batch in parquet_file.iter_batches(columns=["document_sha256"]):
            for row in record_batch.to_pylist():
                value = row.get("document_sha256")
                if not value:
                    continue
                batch.append((bytes.fromhex(value),))
                if len(batch) >= batch_size:
                    deduper.connection.executemany(
                        "INSERT OR IGNORE INTO document_hashes(hash) VALUES (?)",
                        batch,
                    )
                    inserted += len(batch)
                    batch = []
    if batch:
        deduper.connection.executemany(
            "INSERT OR IGNORE INTO document_hashes(hash) VALUES (?)",
            batch,
        )
        inserted += len(batch)
    deduper.connection.commit()
    return inserted


def copy_parquet_files(files: list[Path], source_root: Path, output_dataset: Path) -> None:
    for path in files:
        relative = path.relative_to(source_root)
        destination = output_dataset / "parquet" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def merge_counter_dict(
    left: dict[str, int] | None,
    right: dict[str, int] | None,
) -> dict[str, int]:
    counter = Counter(left or {})
    counter.update(right or {})
    return dict(sorted(counter.items()))


def merge_profiles(
    existing_profile: dict[str, Any],
    added_sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sources = dict(existing_profile.get("sources") or {})
    for source, profile in added_sources.items():
        if source not in sources:
            sources[source] = profile
            continue

        original = sources[source]
        merged = dict(original)
        for key in PROFILE_TOTAL_KEYS:
            merged[key] = int(original.get(key, 0)) + int(profile.get(key, 0))
        merged["rejected"] = merge_counter_dict(
            original.get("rejected"),
            profile.get("rejected"),
        )
        merged["character_counts"] = merge_counter_dict(
            original.get("character_counts"),
            profile.get("character_counts"),
        )
        merged["accepted_upstream_sources"] = merge_counter_dict(
            original.get("accepted_upstream_sources"),
            profile.get("accepted_upstream_sources"),
        )
        character_total = merged["character_counts"].get("total", 0)
        merged["character_ratios"] = {
            key: (
                merged["character_counts"].get(key, 0) / character_total
                if character_total
                else 0.0
            )
            for key in ("hangul", "hanja", "latin", "digit", "replacement")
        }
        sources[source] = merged

    total = {
        key: sum(int(source_profile.get(key, 0)) for source_profile in sources.values())
        for key in PROFILE_TOTAL_KEYS
    }
    return {"sources": sources, "total": total}


def artifact_inventory(dataset_dir: Path) -> list[dict[str, Any]]:
    artifact_paths = parquet_paths(dataset_dir)
    for name in ("profile.json", "samples.jsonl"):
        path = dataset_dir / name
        if path.exists():
            artifact_paths.append(path)
    return [
        {
            "path": str(path.relative_to(dataset_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def append_samples(samples_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with samples_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    source_dataset = args.source_dataset
    output_dataset = args.output_dataset
    staging_dataset = output_dataset.parent / f".{output_dataset.name}.building"
    build_key = hashlib.sha256(str(output_dataset.resolve()).encode()).hexdigest()[:16]
    build_state_dir = args.work_dir / f".{build_key}.aihub_append"
    aihub_parquet_root = build_state_dir / "parquet"

    if not source_dataset.exists():
        raise FileNotFoundError(f"Missing source dataset: {source_dataset}")
    if output_dataset.exists():
        raise FileExistsError(f"Output dataset already exists: {output_dataset}")
    if staging_dataset.exists():
        raise FileExistsError(f"Incomplete staging dataset exists: {staging_dataset}")
    if build_state_dir.exists():
        raise FileExistsError(f"Incomplete local build state exists: {build_state_dir}")
    if not args.aihub_root.exists():
        raise FileNotFoundError(f"Missing AIHub root: {args.aihub_root}")

    source_manifest = load_json(source_dataset / "manifest.json")
    source_profile = load_json(source_dataset / "profile.json")
    split_policy = source_manifest.get("split_policy") or {}
    validation_fraction = (
        args.validation_fraction
        if args.validation_fraction is not None
        else float(split_policy.get("validation_fraction", 0.005))
    )

    tokenizer_config = source_manifest.get("tokenizer") or {}
    tokenizer_name = args.tokenizer_name or tokenizer_config.get("name")
    tokenizer_revision = args.tokenizer_revision or tokenizer_config.get("revision")
    if not tokenizer_name:
        raise ValueError("Tokenizer name is missing. Pass --tokenizer-name.")

    aihub_llm_paths = _aihub_korean_llm_paths(
        args.aihub_root,
        include_validation_source=args.include_validation_source,
        include_rlhf=False,
    )
    aihub_commonsense_paths = _aihub_commonsense_sentence_paths(
        args.aihub_root,
        include_validation_source=args.include_validation_source,
    )
    if not aihub_llm_paths and not aihub_commonsense_paths:
        raise FileNotFoundError(f"No selected AIHub source zip files under {args.aihub_root}")

    print(f"Source dataset: {source_dataset}")
    print(f"Output dataset: {output_dataset}")
    print(f"AIHub Korean LLM zips: {len(aihub_llm_paths)}")
    print(f"AIHub commonsense sentence zips: {len(aihub_commonsense_paths)}")
    print(f"Tokenizer: {tokenizer_name} @ {tokenizer_revision}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    build_state_dir.mkdir(parents=True)
    deduper = SQLiteDeduper(build_state_dir / "exact_dedup.sqlite3")

    try:
        seeded = seed_existing_hashes(deduper, source_dataset)
        print(f"Seeded existing v2 document hashes: {seeded}")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=tokenizer_revision,
        )
        appender = AIHubAppender(
            tokenizer=tokenizer,
            deduper=deduper,
            output_parquet_root=aihub_parquet_root,
            validation_fraction=validation_fraction,
            shard_rows=args.shard_rows,
            tokenize_batch_size=args.tokenize_batch_size,
            sample_size=args.sample_size,
            max_accepted_per_source=args.max_accepted_per_source,
        )

        if aihub_llm_paths:
            appender.process(
                "aihub_korean_llm",
                iter_aihub_korean_llm_zip(aihub_llm_paths),
            )
        if aihub_commonsense_paths:
            appender.process(
                "aihub_commonsense_sentence_generation",
                iter_aihub_commonsense_sentence_zip(aihub_commonsense_paths),
            )
        added_parquet_files = appender.close()
        added_profile_sources = appender.profile_payload()
        deduper.close()

        print(f"Added Parquet shards: {len(added_parquet_files)}")
        print("Copying v2 dataset to staging v3 directory...")
        shutil.copytree(source_dataset, staging_dataset)
        copy_parquet_files(added_parquet_files, aihub_parquet_root, staging_dataset)

        merged_profile = merge_profiles(source_profile, added_profile_sources)
        write_json(staging_dataset / "profile.json", merged_profile)
        append_samples(staging_dataset / "samples.jsonl", appender.sample_rows())

        manifest = load_json(staging_dataset / "manifest.json")
        manifest["profile"] = merged_profile
        manifest.setdefault("sources", {})
        if aihub_llm_paths:
            manifest["sources"]["aihub_korean_llm"] = {
                "dataset": "121.한국어 성능이 개선된 초거대AI 언어모델 개발 및 데이터",
                "local_root": str(args.aihub_root),
                "file_count": len(aihub_llm_paths),
                "downloaded_bytes": sum(path.stat().st_size for path in aihub_llm_paths),
                "include_validation_source": args.include_validation_source,
                "include_rlhf": False,
                "license": "AIHub approved-use terms",
                "raw_files": _file_inventory(aihub_llm_paths, args.aihub_root),
            }
        if aihub_commonsense_paths:
            manifest["sources"]["aihub_commonsense_sentence_generation"] = {
                "dataset": "048.일반상식 문장 생성 데이터",
                "local_root": str(args.aihub_root),
                "file_count": len(aihub_commonsense_paths),
                "downloaded_bytes": sum(
                    path.stat().st_size for path in aihub_commonsense_paths
                ),
                "include_validation_source": args.include_validation_source,
                "license": "AIHub approved-use terms",
                "raw_files": _file_inventory(aihub_commonsense_paths, args.aihub_root),
            }
        manifest["derived_from"] = {
            "dataset_path": str(source_dataset),
            "manifest_sha256": sha256_file(source_dataset / "manifest.json"),
            "profile_sha256": sha256_file(source_dataset / "profile.json"),
        }
        manifest["extension"] = {
            "method": "copy_source_dataset_and_append_aihub_parquet_shards",
            "added_sources": sorted(added_profile_sources),
            "validation_fraction": validation_fraction,
        }
        manifest["artifacts"] = artifact_inventory(staging_dataset)
        write_json(staging_dataset / "manifest.json", manifest)

        os.replace(staging_dataset, output_dataset)
        shutil.rmtree(build_state_dir)
        print(f"Dataset build complete: {output_dataset}")
        print(json.dumps(merged_profile["total"], ensure_ascii=False, indent=2))
    except Exception:
        deduper.close()
        raise


if __name__ == "__main__":
    main()
