import argparse
import hashlib
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import requests
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer

from src.dataset_pipeline.adapters import (
    DEFAULT_WEBTEXT_UPSTREAM_SOURCES,
    iter_aihub_commonsense_sentence_zip,
    iter_aihub_korean_llm_zip,
    iter_historical_jsonl,
    iter_korean_webtext_parquet,
    iter_nikl_corpus,
    iter_wikimedia_dump,
)
from src.dataset_pipeline.builder import DatasetBuilder, sha256_file


DEFAULT_WIKIMEDIA_URL = (
    "https://dumps.wikimedia.org/kowiki/20260601/"
    "kowiki-20260601-pages-articles.xml.bz2"
)
DEFAULT_HISTORICAL_REPO = "seyoungsong/Open-Korean-Historical-Corpus"
DEFAULT_WEBTEXT_REPO = "HAERAE-HUB/KOREAN-WEBTEXT"
DEFAULT_TOKENIZER = "skt/kogpt2-base-v2"
DEFAULT_SOURCES = ("wikimedia", "webtext")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the versioned MyGPT Korean pretraining dataset."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/content/mygpt_dataset_work"),
    )
    parser.add_argument(
        "--raw-cache-dir",
        type=Path,
        help="Persistent cache for downloaded raw sources. Defaults to WORK_DIR/raw.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("wikimedia", "webtext", "historical", "nikl", "aihub"),
        default=DEFAULT_SOURCES,
    )
    parser.add_argument(
        "--wikimedia-dump-url",
        default=DEFAULT_WIKIMEDIA_URL,
    )
    parser.add_argument("--wikimedia-dump-path", type=Path)
    parser.add_argument(
        "--historical-repo",
        default=DEFAULT_HISTORICAL_REPO,
    )
    parser.add_argument("--historical-revision")
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument(
        "--historical-allow-pattern",
        action="append",
        dest="historical_allow_patterns",
    )
    parser.add_argument(
        "--historical-copyright",
        default="Public Domain",
        help='Exact copyright value to keep, or "ANY" to disable this filter.',
    )
    parser.add_argument(
        "--webtext-repo",
        default=DEFAULT_WEBTEXT_REPO,
    )
    parser.add_argument("--webtext-revision")
    parser.add_argument("--webtext-root", type=Path)
    parser.add_argument(
        "--webtext-allow-pattern",
        action="append",
        dest="webtext_allow_patterns",
    )
    parser.add_argument(
        "--webtext-upstream-sources",
        nargs="+",
        default=DEFAULT_WEBTEXT_UPSTREAM_SOURCES,
        help='Allowed upstream sources, or "ANY" to disable source filtering.',
    )
    parser.add_argument(
        "--webtext-keep-obvious-spam",
        action="store_true",
        help="Disable the conservative adult, gambling, trading, and repetition filter.",
    )
    parser.add_argument("--nikl-root", type=Path)
    parser.add_argument(
        "--nikl-corpora",
        nargs="+",
        default=("WRITTEN", "NEWSPAPER"),
    )
    parser.add_argument("--aihub-root", type=Path)
    parser.add_argument(
        "--aihub-include-validation-source",
        action="store_true",
        help="Include AIHub Validation/01.원천데이터 zips in addition to Training.",
    )
    parser.add_argument(
        "--aihub-include-rlhf",
        action="store_true",
        help="Include AIHub RLHF zips. Keep disabled for plain pretraining.",
    )
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--shard-rows", type=int, default=100_000)
    parser.add_argument("--tokenize-batch-size", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.005)
    parser.add_argument("--max-accepted-per-source", type=int)
    return parser.parse_args()


def download_with_resume(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.exists() else 0
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    total_size = int(head.headers.get("Content-Length", 0))
    if total_size and existing_size == total_size:
        return destination
    if total_size and existing_size > total_size:
        existing_size = 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}

    with requests.get(url, stream=True, headers=headers, timeout=60) as response:
        response.raise_for_status()
        append = existing_size > 0 and response.status_code == 206
        mode = "ab" if append else "wb"
        with destination.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


def _wikimedia_sha1_url(dump_url: str) -> str:
    parsed = urlparse(dump_url)
    directory = parsed.path.rsplit("/", 1)[0]
    dump_date = directory.rstrip("/").rsplit("/", 1)[-1]
    return f"{parsed.scheme}://{parsed.netloc}{directory}/kowiki-{dump_date}-sha1sums.txt"


def _sha1_from_checksum_text(checksum_text: str, filename: str) -> str:
    for line in checksum_text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("*") == filename:
            return parts[0]
    raise ValueError(f"SHA1 entry not found for {filename}")


def verify_wikimedia_sha1(dump_url: str, dump_path: Path) -> str:
    response = requests.get(_wikimedia_sha1_url(dump_url), timeout=60)
    response.raise_for_status()
    expected = _sha1_from_checksum_text(response.text, dump_path.name)

    digest = hashlib.sha1()
    with dump_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"Wikimedia dump SHA1 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _resolve_dataset_revision(repo_id: str, revision: str | None) -> str:
    info = HfApi().dataset_info(repo_id, revision=revision)
    return info.sha


def _resolve_model_revision(model_id: str, revision: str | None) -> str:
    info = HfApi().model_info(model_id, revision=revision)
    return info.sha


def _historical_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("*.jsonl"))


def _webtext_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet"))


def _nikl_inventory(root: Path, corpora: tuple[str, ...] | list[str]) -> list[dict]:
    inventory = []
    for corpus in corpora:
        corpus_root = root / corpus.upper()
        if not corpus_root.exists():
            continue
        for path in sorted(corpus_root.rglob("*")):
            if path.is_file() and path.suffix.lower() == ".json":
                inventory.append(
                    {
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return inventory


def _aihub_korean_llm_paths(
    root: Path,
    *,
    include_validation_source: bool = False,
    include_rlhf: bool = False,
) -> list[Path]:
    return _aihub_source_zip_paths(
        root,
        include_validation_source=include_validation_source,
        include_rlhf=include_rlhf,
        required_path_markers=(
            "121.한국어 성능이 개선된 초거대AI 언어모델 개발 및 데이터",
            "한국어말뭉치데이터",
        ),
    )


def _aihub_commonsense_sentence_paths(
    root: Path,
    *,
    include_validation_source: bool = False,
) -> list[Path]:
    return _aihub_source_zip_paths(
        root,
        include_validation_source=include_validation_source,
        include_rlhf=False,
        required_path_markers=(
            "048.일반상식 문장 생성 데이터",
            "일반상식 문장_생성데이터",
        ),
    )


def _aihub_source_zip_paths(
    root: Path,
    *,
    include_validation_source: bool,
    include_rlhf: bool,
    required_path_markers: tuple[str, ...],
) -> list[Path]:
    paths = []
    normalized_root_parts = {
        unicodedata.normalize("NFC", part) for part in root.parts
    }
    for path in sorted(root.rglob("*.zip")):
        if not path.is_file():
            continue
        if path.stat().st_size <= 0:
            continue
        lowered_name = path.name.lower()
        if ".irx" in lowered_name:
            continue
        if not include_rlhf and "rlhf" in lowered_name:
            continue
        normalized_path = unicodedata.normalize("NFC", str(path))
        if any(marker not in normalized_path for marker in required_path_markers):
            continue

        parts = {unicodedata.normalize("NFC", part) for part in path.parts}
        is_source_data = "01.원천데이터" in parts
        is_training = "Training" in parts
        is_validation = "Validation" in parts

        if "01.원천데이터" in normalized_root_parts:
            is_source_data = True
            is_training = True
        if "Training" in normalized_root_parts:
            is_training = True
        if "Validation" in normalized_root_parts:
            is_validation = True

        if not is_source_data:
            continue
        if is_training or (include_validation_source and is_validation):
            paths.append(path)
    return paths


def _file_inventory(paths: list[Path], root: Path) -> list[dict]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir = args.raw_cache_dir or args.work_dir / "raw"
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    streams = []
    source_config = {}

    tokenizer_revision = _resolve_model_revision(
        args.tokenizer_name,
        args.tokenizer_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        revision=tokenizer_revision,
    )
    tokenizer_info = {
        "name": args.tokenizer_name,
        "revision": tokenizer_revision,
        "class": tokenizer.__class__.__name__,
        "vocab_size": tokenizer.vocab_size,
        "special_tokens_map": tokenizer.special_tokens_map,
    }

    if "wikimedia" in args.sources:
        if args.wikimedia_dump_path is not None:
            dump_path = args.wikimedia_dump_path
            if not dump_path.exists():
                raise FileNotFoundError(dump_path)
            dump_sha1 = None
        else:
            filename = args.wikimedia_dump_url.rsplit("/", 1)[-1]
            dump_path = raw_cache_dir / "wikimedia" / filename
            download_with_resume(args.wikimedia_dump_url, dump_path)
            dump_sha1 = verify_wikimedia_sha1(
                args.wikimedia_dump_url,
                dump_path,
            )

        source_config["wikimedia"] = {
            "url": args.wikimedia_dump_url,
            "local_file": str(dump_path),
            "bytes": dump_path.stat().st_size,
            "sha1": dump_sha1,
            "sha256": sha256_file(dump_path),
        }
        streams.append(("wikimedia", iter_wikimedia_dump(dump_path)))

    if "historical" in args.sources:
        historical_revision = _resolve_dataset_revision(
            args.historical_repo,
            args.historical_revision,
        )
        if args.historical_root is not None:
            historical_root = args.historical_root
        else:
            allow_patterns = args.historical_allow_patterns or ["*.jsonl"]
            historical_root = Path(
                snapshot_download(
                    repo_id=args.historical_repo,
                    repo_type="dataset",
                    revision=historical_revision,
                    local_dir=raw_cache_dir / "historical" / historical_revision,
                    allow_patterns=allow_patterns,
                )
            )
        historical_paths = _historical_paths(historical_root)
        if not historical_paths:
            raise FileNotFoundError(
                f"No JSONL files found under {historical_root}"
            )
        required_copyright = (
            None
            if args.historical_copyright.upper() == "ANY"
            else args.historical_copyright
        )
        source_config["historical"] = {
            "repo_id": args.historical_repo,
            "revision": historical_revision,
            "local_root": str(historical_root),
            "file_count": len(historical_paths),
            "language_filter": "Modern Korean",
            "copyright_filter": required_copyright,
        }
        streams.append(
            (
                "historical",
                iter_historical_jsonl(
                    historical_paths,
                    required_language="Modern Korean",
                    required_copyright=required_copyright,
                ),
            )
        )

    if "webtext" in args.sources:
        webtext_revision = _resolve_dataset_revision(
            args.webtext_repo,
            args.webtext_revision,
        )
        if args.webtext_root is not None:
            webtext_root = args.webtext_root
        else:
            allow_patterns = args.webtext_allow_patterns or ["data/*.parquet"]
            webtext_root = Path(
                snapshot_download(
                    repo_id=args.webtext_repo,
                    repo_type="dataset",
                    revision=webtext_revision,
                    local_dir=raw_cache_dir / "webtext" / webtext_revision,
                    allow_patterns=allow_patterns,
                )
            )
        webtext_paths = _webtext_paths(webtext_root)
        if not webtext_paths:
            raise FileNotFoundError(
                f"No Parquet files found under {webtext_root}"
            )
        allowed_upstream_sources = (
            None
            if len(args.webtext_upstream_sources) == 1
            and args.webtext_upstream_sources[0].upper() == "ANY"
            else list(args.webtext_upstream_sources)
        )
        source_config["webtext"] = {
            "repo_id": args.webtext_repo,
            "revision": webtext_revision,
            "local_root": str(webtext_root),
            "file_count": len(webtext_paths),
            "downloaded_bytes": sum(path.stat().st_size for path in webtext_paths),
            "license": "Not declared in the dataset card",
            "allowed_upstream_sources": allowed_upstream_sources,
            "obvious_spam_filter": not args.webtext_keep_obvious_spam,
            "upstream_processing": [
                "sentence and document quality filters",
                "exact line deduplication",
                "first 15 token deduplication",
                "last 15 token deduplication",
            ],
        }
        streams.append(
            (
                "webtext",
                iter_korean_webtext_parquet(
                    webtext_paths,
                    allowed_upstream_sources=allowed_upstream_sources,
                    reject_obvious_spam=not args.webtext_keep_obvious_spam,
                ),
            )
        )

    if "nikl" in args.sources:
        if args.nikl_root is None:
            raise ValueError("--nikl-root is required when selecting NIKL")
        if not args.nikl_root.exists():
            raise FileNotFoundError(args.nikl_root)
        nikl_corpora = [corpus.upper() for corpus in args.nikl_corpora]
        source_config["nikl"] = {
            "official_service": "https://kli.korean.go.kr/",
            "local_root": str(args.nikl_root),
            "corpora": nikl_corpora,
            "license": "NIKL approved-use terms",
            "raw_files": _nikl_inventory(args.nikl_root, nikl_corpora),
        }
        streams.append(
            (
                "nikl",
                iter_nikl_corpus(args.nikl_root, nikl_corpora),
            )
        )

    if "aihub" in args.sources:
        if args.aihub_root is None:
            raise ValueError("--aihub-root is required when selecting AIHub")
        if not args.aihub_root.exists():
            raise FileNotFoundError(args.aihub_root)
        aihub_paths = _aihub_korean_llm_paths(
            args.aihub_root,
            include_validation_source=args.aihub_include_validation_source,
            include_rlhf=args.aihub_include_rlhf,
        )
        aihub_commonsense_paths = _aihub_commonsense_sentence_paths(
            args.aihub_root,
            include_validation_source=args.aihub_include_validation_source,
        )
        if not aihub_paths and not aihub_commonsense_paths:
            raise FileNotFoundError(
                f"No usable AIHub source zip files found under {args.aihub_root}"
            )
        if aihub_paths:
            source_config["aihub_korean_llm"] = {
                "dataset": "121.한국어 성능이 개선된 초거대AI 언어모델 개발 및 데이터",
                "local_root": str(args.aihub_root),
                "file_count": len(aihub_paths),
                "downloaded_bytes": sum(path.stat().st_size for path in aihub_paths),
                "include_validation_source": args.aihub_include_validation_source,
                "include_rlhf": args.aihub_include_rlhf,
                "license": "AIHub approved-use terms",
                "raw_files": _file_inventory(aihub_paths, args.aihub_root),
            }
            streams.append(
                (
                    "aihub_korean_llm",
                    iter_aihub_korean_llm_zip(aihub_paths),
                )
            )
        if aihub_commonsense_paths:
            source_config["aihub_commonsense_sentence_generation"] = {
                "dataset": "048.일반상식 문장 생성 데이터",
                "local_root": str(args.aihub_root),
                "file_count": len(aihub_commonsense_paths),
                "downloaded_bytes": sum(
                    path.stat().st_size for path in aihub_commonsense_paths
                ),
                "include_validation_source": args.aihub_include_validation_source,
                "license": "AIHub approved-use terms",
                "raw_files": _file_inventory(
                    aihub_commonsense_paths,
                    args.aihub_root,
                ),
            }
            streams.append(
                (
                    "aihub_commonsense_sentence_generation",
                    iter_aihub_commonsense_sentence_zip(aihub_commonsense_paths),
                )
            )
        source_config["aihub"] = {
            "dataset": "AIHub pretraining source bundle",
            "local_root": str(args.aihub_root),
            "file_count": len(aihub_paths) + len(aihub_commonsense_paths),
            "downloaded_bytes": sum(
                path.stat().st_size
                for path in [*aihub_paths, *aihub_commonsense_paths]
            ),
            "include_validation_source": args.aihub_include_validation_source,
            "include_rlhf": args.aihub_include_rlhf,
            "license": "AIHub approved-use terms",
            "included_datasets": [
                "121.한국어 성능이 개선된 초거대AI 언어모델 개발 및 데이터",
                "048.일반상식 문장 생성 데이터",
            ],
            "raw_files": _file_inventory(
                [*aihub_paths, *aihub_commonsense_paths],
                args.aihub_root,
            ),
        }

    builder = DatasetBuilder(
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        tokenizer=tokenizer,
        tokenizer_info=tokenizer_info,
        source_config=source_config,
        shard_rows=args.shard_rows,
        tokenize_batch_size=args.tokenize_batch_size,
        validation_fraction=args.validation_fraction,
        max_accepted_per_source=args.max_accepted_per_source,
    )
    output_dir = builder.build(streams)
    print(f"Dataset build complete: {output_dir}")


if __name__ == "__main__":
    main()
