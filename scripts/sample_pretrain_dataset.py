#!/usr/bin/env python3
"""Randomly sample documents from a prepared MyGPT pretrain dataset."""

from __future__ import annotations

import argparse
import html
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_COLUMNS = [
    "id",
    "source",
    "source_id",
    "text",
    "title",
    "url",
    "license",
    "language",
    "year",
    "corpus",
    "metadata_json",
    "document_sha256",
    "token_count",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create random JSONL and HTML samples from a prepared pretrain dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Prepared dataset directory, e.g. /content/drive/.../datasets/pretrain/v2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where sample files will be written.",
    )
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "all"),
        default="train",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--html-preview-chars",
        type=int,
        default=2000,
        help="Maximum text characters rendered per row in the HTML preview.",
    )
    return parser.parse_args()


def parquet_paths(dataset_path: Path, split: str) -> list[Path]:
    parquet_root = dataset_path / "parquet"
    if split == "all":
        candidates = sorted(parquet_root.glob("*/*.parquet"))
    else:
        candidates = sorted((parquet_root / split).glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No parquet files found for split={split!r} under {parquet_root}"
        )
    return candidates


def safe_json_loads(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def iter_rows(paths: list[Path], batch_size: int):
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        available_columns = [
            column for column in DEFAULT_COLUMNS if column in parquet_file.schema.names
        ]
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=available_columns,
        ):
            rows = batch.to_pylist()
            for row in rows:
                row["parquet_file"] = path.name
                metadata = safe_json_loads(row.get("metadata_json"))
                row["metadata"] = metadata
                if metadata.get("upstream_source"):
                    row["upstream_source"] = metadata["upstream_source"]
                yield row


def reservoir_sample(paths: list[Path], sample_size: int, seed: int, batch_size: int):
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    seen = 0
    for row in tqdm(iter_rows(paths, batch_size), desc="Sampling documents"):
        seen += 1
        if len(sample) < sample_size:
            sample.append(row)
            continue
        replacement_index = rng.randrange(seen)
        if replacement_index < sample_size:
            sample[replacement_index] = row
    sample.sort(key=lambda row: str(row.get("id", "")))
    return sample, seen


def compact_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    text = row.get("text") or ""
    return {
        "sample_index": index,
        "id": row.get("id") or "",
        "source": row.get("source") or "",
        "upstream_source": row.get("upstream_source") or "",
        "source_id": row.get("source_id") or "",
        "title": row.get("title") or "",
        "url": row.get("url") or "",
        "license": row.get("license") or "",
        "language": row.get("language") or "",
        "year": row.get("year"),
        "corpus": row.get("corpus") or "",
        "token_count": row.get("token_count"),
        "document_sha256": row.get("document_sha256") or "",
        "split": row.get("split") or "",
        "parquet_file": row.get("parquet_file") or "",
        "metadata": row.get("metadata") or {},
        "text": text,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, rows: list[dict[str, Any]], seen: int, args: argparse.Namespace):
    source_counts = Counter(row["source"] for row in rows)
    upstream_counts = Counter(row["upstream_source"] or "(none)" for row in rows)
    token_counts = [row["token_count"] for row in rows if isinstance(row["token_count"], int)]
    payload = {
        "dataset_path": str(args.dataset_path),
        "split": args.split,
        "seed": args.seed,
        "requested_sample_size": args.sample_size,
        "actual_sample_size": len(rows),
        "documents_seen": seen,
        "source_counts": dict(sorted(source_counts.items())),
        "upstream_source_counts": dict(sorted(upstream_counts.items())),
        "token_count": {
            "min": min(token_counts) if token_counts else None,
            "max": max(token_counts) if token_counts else None,
            "mean": sum(token_counts) / len(token_counts) if token_counts else None,
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_html(path: Path, rows: list[dict[str, Any]], summary_path: Path, preview_chars: int):
    cards = []
    for row in rows:
        text = row["text"]
        clipped = text[:preview_chars]
        if len(text) > preview_chars:
            clipped += "\n\n... [HTML preview clipped; full text is in JSONL]"
        meta = [
            ("sample", row["sample_index"]),
            ("source", row["source"]),
            ("upstream", row["upstream_source"]),
            ("tokens", row["token_count"]),
            ("split", row["split"]),
            ("title", row["title"]),
            ("url", row["url"]),
        ]
        meta_html = "".join(
            f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(str(value or ''))}</dd>"
            for key, value in meta
        )
        cards.append(
            f"""
            <article class="card">
              <dl>{meta_html}</dl>
              <pre>{html.escape(clipped)}</pre>
            </article>
            """
        )

    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>MyGPT Pretrain Dataset Random Samples</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #1f2430;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 1;
      padding: 18px 24px;
      background: rgba(255, 255, 255, 0.95);
      border-bottom: 1px solid #d9dde5;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    .card {{
      margin-bottom: 20px;
      padding: 18px;
      background: white;
      border: 1px solid #e1e5ec;
      border-radius: 14px;
      box-shadow: 0 1px 2px rgba(20, 28, 45, 0.04);
    }}
    dl {{
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 6px 12px;
      margin: 0 0 14px;
      font-size: 13px;
    }}
    dt {{
      color: #667085;
      font-weight: 700;
    }}
    dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 0;
      padding: 14px;
      background: #f8fafc;
      border-radius: 10px;
      line-height: 1.55;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>MyGPT Pretrain Dataset Random Samples</h1>
    <div>Summary: {html.escape(summary_path.name)} / HTML text clipped for review speed.</div>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = parquet_paths(args.dataset_path, args.split)
    sample, seen = reservoir_sample(paths, args.sample_size, args.seed, args.batch_size)
    rows = [compact_row(row, index + 1) for index, row in enumerate(sample)]

    stem = f"pretrain_{args.split}_random_{len(rows)}_seed{args.seed}"
    jsonl_path = args.output_dir / f"{stem}.jsonl"
    summary_path = args.output_dir / f"{stem}.summary.json"
    html_path = args.output_dir / f"{stem}.html"

    write_jsonl(jsonl_path, rows)
    write_summary(summary_path, rows, seen, args)
    write_html(html_path, rows, summary_path, args.html_preview_chars)

    print(f"Documents seen: {seen}")
    print(f"Samples written: {len(rows)}")
    print(f"JSONL: {jsonl_path}")
    print(f"Summary: {summary_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
