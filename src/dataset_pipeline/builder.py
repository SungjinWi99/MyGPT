import hashlib
import heapq
import json
import os
import random
import re
import shutil
import sqlite3
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.dataset_pipeline.schema import RejectedRecord, SourceDocument


_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", flags=re.MULTILINE)
_HANGUL_RE = re.compile(r"[가-힣]")
_HANJA_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"[0-9]")

PARQUET_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("text", pa.string()),
        ("title", pa.string()),
        ("url", pa.string()),
        ("license", pa.string()),
        ("language", pa.string()),
        ("year", pa.int32()),
        ("corpus", pa.string()),
        ("metadata_json", pa.string()),
        ("document_sha256", pa.string()),
        ("token_count", pa.int64()),
        ("split", pa.string()),
    ]
)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _TRAILING_SPACE_RE.sub("", normalized)
    normalized = _BLANK_LINES_RE.sub("\n\n", normalized)
    return normalized.strip()


def deterministic_split(document_hash: bytes, validation_fraction: float) -> str:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    bucket = int.from_bytes(document_hash[:8], byteorder="big") / 2**64
    return "validation" if bucket < validation_fraction else "train"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class SQLiteDeduper:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection: sqlite3.Connection | None = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS document_hashes "
            "(hash BLOB PRIMARY KEY) WITHOUT ROWID"
        )

    def add(self, document_hash: bytes) -> bool:
        if self.connection is None:
            raise RuntimeError("Deduper is closed")
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO document_hashes(hash) VALUES (?)",
            (document_hash,),
        )
        return cursor.rowcount == 1

    def close(self) -> None:
        if self.connection is None:
            return
        self.connection.commit()
        self.connection.close()
        self.connection = None


class ParquetShardWriter:
    def __init__(self, root: Path, shard_rows: int):
        self.root = root
        self.shard_rows = shard_rows
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
            self._flush(split)

    def _flush(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return

        split_dir = self.root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        path = split_dir / f"part-{self.shard_indexes[split]:05d}.parquet"
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
        self._flush("train")
        self._flush("validation")
        return self.files


@dataclass
class Profile:
    raw_records: int = 0
    accepted_records: int = 0
    accepted_bytes_utf8: int = 0
    accepted_tokens: int = 0
    train_records: int = 0
    validation_records: int = 0
    rejected: Counter = field(default_factory=Counter)
    characters: Counter = field(default_factory=Counter)
    upstream_sources: Counter = field(default_factory=Counter)
    length_sample: list[int] = field(default_factory=list)
    token_length_sample: list[int] = field(default_factory=list)
    reservoir_seen: int = 0

    def reject(self, reason: str) -> None:
        self.rejected[reason] += 1

    def accept(
        self,
        text: str,
        token_count: int,
        split: str,
        rng: random.Random,
        reservoir_size: int,
        upstream_source: str | None = None,
    ) -> None:
        self.accepted_records += 1
        self.accepted_bytes_utf8 += len(text.encode("utf-8"))
        self.accepted_tokens += token_count
        self.train_records += split == "train"
        self.validation_records += split == "validation"
        self.characters["hangul"] += len(_HANGUL_RE.findall(text))
        self.characters["hanja"] += len(_HANJA_RE.findall(text))
        self.characters["latin"] += len(_LATIN_RE.findall(text))
        self.characters["digit"] += len(_DIGIT_RE.findall(text))
        self.characters["replacement"] += text.count("\ufffd")
        self.characters["total"] += len(text)
        if upstream_source:
            self.upstream_sources[upstream_source] += 1

        self.reservoir_seen += 1
        if len(self.length_sample) < reservoir_size:
            self.length_sample.append(len(text))
            self.token_length_sample.append(token_count)
            return

        index = rng.randrange(self.reservoir_seen)
        if index < reservoir_size:
            self.length_sample[index] = len(text)
            self.token_length_sample[index] = token_count


def _percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    sorted_values = sorted(values)
    result = {}
    for percentile in (1, 10, 25, 50, 75, 90, 95, 99):
        index = round((len(sorted_values) - 1) * percentile / 100)
        result[f"p{percentile}"] = sorted_values[index]
    return result


def _profile_dict(profile: Profile) -> dict[str, Any]:
    character_total = profile.characters.get("total", 0)
    ratios = {}
    for key in ("hangul", "hanja", "latin", "digit", "replacement"):
        count = profile.characters.get(key, 0)
        ratios[key] = count / character_total if character_total else 0.0

    return {
        "raw_records": profile.raw_records,
        "accepted_records": profile.accepted_records,
        "accepted_bytes_utf8": profile.accepted_bytes_utf8,
        "accepted_tokens": profile.accepted_tokens,
        "train_records": profile.train_records,
        "validation_records": profile.validation_records,
        "rejected": dict(sorted(profile.rejected.items())),
        "character_counts": dict(sorted(profile.characters.items())),
        "character_ratios": ratios,
        "accepted_upstream_sources": dict(sorted(profile.upstream_sources.items())),
        "document_char_percentiles": _percentiles(profile.length_sample),
        "document_token_percentiles": _percentiles(profile.token_length_sample),
        "percentile_sample_size": len(profile.length_sample),
    }


class DatasetBuilder:
    def __init__(
        self,
        *,
        output_dir: Path,
        work_dir: Path,
        tokenizer: Any,
        tokenizer_info: dict[str, Any],
        source_config: dict[str, Any],
        shard_rows: int = 100_000,
        tokenize_batch_size: int = 128,
        validation_fraction: float = 0.005,
        reservoir_size: int = 100_000,
        sample_size: int = 20,
        max_accepted_per_source: int | None = None,
    ):
        self.output_dir = output_dir
        self.staging_dir = output_dir.parent / f".{output_dir.name}.building"
        self.work_dir = work_dir
        self.tokenizer = tokenizer
        self.tokenizer_info = tokenizer_info
        self.source_config = source_config
        self.shard_rows = shard_rows
        self.tokenize_batch_size = tokenize_batch_size
        self.validation_fraction = validation_fraction
        self.reservoir_size = reservoir_size
        self.sample_size = sample_size
        self.max_accepted_per_source = max_accepted_per_source
        self.profiles: dict[str, Profile] = {}
        self.samples: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        self.rngs: dict[str, random.Random] = {}
        self.pending: list[tuple[str, SourceDocument, bytes, str]] = []

    def _profile(self, name: str) -> Profile:
        if name not in self.profiles:
            self.profiles[name] = Profile()
            seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
            self.rngs[name] = random.Random(seed)
            self.samples[name] = []
        return self.profiles[name]

    def _sample(self, source: str, score: int, row: dict[str, Any]) -> None:
        heap = self.samples[source]
        item = (-score, row)
        if len(heap) < self.sample_size:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def _flush_pending(
        self,
        writer: ParquetShardWriter,
    ) -> None:
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
            document_hash_hex = document_hash.hex()
            row_id = f"{document.source}:{document.source_id}"
            row = {
                "id": row_id,
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
                "document_sha256": document_hash_hex,
                "token_count": token_count,
                "split": split,
            }
            writer.add(row)

            profile = self._profile(stream_name)
            profile.accept(
                document.text,
                token_count,
                split,
                self.rngs[stream_name],
                self.reservoir_size,
                str(document.metadata.get("upstream_source") or "") or None,
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
            self._sample(
                stream_name,
                int.from_bytes(document_hash[:8], "big"),
                sample_row,
            )

        self.pending = []

    def build(
        self,
        streams: list[tuple[str, Iterable[SourceDocument | RejectedRecord]]],
    ) -> Path:
        if self.output_dir.exists():
            raise FileExistsError(
                f"Output directory already exists and dataset versions are immutable: "
                f"{self.output_dir}"
            )
        if self.staging_dir.exists():
            raise FileExistsError(
                f"Incomplete staging directory exists: {self.staging_dir}. "
                "Inspect or remove it before retrying."
            )

        self.work_dir.mkdir(parents=True, exist_ok=True)
        build_key = hashlib.sha256(
            str(self.output_dir.resolve()).encode("utf-8")
        ).hexdigest()[:16]
        build_state_dir = self.work_dir / f".{build_key}.building"
        if build_state_dir.exists():
            raise FileExistsError(
                f"Incomplete local build state exists: {build_state_dir}. "
                "Inspect or remove it before retrying."
            )
        self.staging_dir.mkdir(parents=True)
        deduper = SQLiteDeduper(build_state_dir / "exact_dedup.sqlite3")
        writer = ParquetShardWriter(self.staging_dir / "parquet", self.shard_rows)

        try:
            for stream_name, records in streams:
                profile = self._profile(stream_name)
                for record in records:
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
                    if not deduper.add(document_hash):
                        profile.reject("exact_duplicate")
                        continue

                    record.text = text
                    split = deterministic_split(
                        document_hash,
                        self.validation_fraction,
                    )
                    self.pending.append(
                        (stream_name, record, document_hash, split)
                    )
                    if len(self.pending) >= self.tokenize_batch_size:
                        self._flush_pending(writer)

                self._flush_pending(writer)

            files = writer.close()
            deduper.close()
            shutil.rmtree(build_state_dir)
            self._write_outputs(files)
            os.replace(self.staging_dir, self.output_dir)
            return self.output_dir
        except Exception:
            deduper.close()
            raise

    def _write_outputs(self, files: list[Path]) -> None:
        profile_payload = {
            "sources": {
                name: _profile_dict(profile)
                for name, profile in sorted(self.profiles.items())
            }
        }
        profile_payload["total"] = {
            key: sum(source[key] for source in profile_payload["sources"].values())
            for key in (
                "raw_records",
                "accepted_records",
                "accepted_bytes_utf8",
                "accepted_tokens",
                "train_records",
                "validation_records",
            )
        }

        profile_path = self.staging_dir / "profile.json"
        profile_path.write_text(
            json.dumps(profile_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        samples_path = self.staging_dir / "samples.jsonl"
        with samples_path.open("w", encoding="utf-8") as handle:
            for source, heap in sorted(self.samples.items()):
                for _, row in sorted(heap, reverse=True):
                    handle.write(
                        json.dumps(
                            {"profile_source": source, **row},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        artifact_files = files + [profile_path, samples_path]
        manifest = {
            "schema_version": 1,
            "dataset_type": "pretrain",
            "canonical_format": "parquet",
            "split_policy": {
                "method": "sha256_normalized_text",
                "train_fraction": 1.0 - self.validation_fraction,
                "validation_fraction": self.validation_fraction,
            },
            "deduplication": {
                "method": "exact_sha256_after_normalization",
                "scope": "all_selected_sources",
            },
            "normalization": {
                "unicode": "NFC",
                "line_endings": "LF",
                "trailing_whitespace": "removed",
                "maximum_consecutive_newlines": 2,
            },
            "tokenizer": self.tokenizer_info,
            "sources": self.source_config,
            "profile": profile_payload,
            "artifacts": [
                {
                    "path": str(path.relative_to(self.staging_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in artifact_files
            ],
        }
        (self.staging_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
