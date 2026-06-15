import bz2
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import quote

import ijson
import mwparserfromhell
import pyarrow.parquet as pq

from src.dataset_pipeline.schema import RejectedRecord, SourceDocument


Record = SourceDocument | RejectedRecord

_WIKI_TABLE_RE = re.compile(r"\{\|.*?\|\}", flags=re.DOTALL)
_WIKI_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_WIKI_CATEGORY_RE = re.compile(
    r"^\s*\[\[(?:분류|Category|파일|File|그림|Image):.*?\]\]\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in element:
        if _local_name(child.tag) == name:
            return child
    return None


def _direct_text(element: ET.Element, name: str) -> str:
    child = _direct_child(element, name)
    return child.text if child is not None and child.text else ""


def clean_wikitext(raw_text: str) -> str:
    text = _WIKI_COMMENT_RE.sub("", raw_text)
    text = _WIKI_TABLE_RE.sub("", text)
    text = _WIKI_CATEGORY_RE.sub("", text)

    code = mwparserfromhell.parse(text)
    for tag in list(code.filter_tags(recursive=True)):
        if str(tag.tag).strip().lower() in {
            "ref",
            "references",
            "gallery",
            "math",
            "score",
            "timeline",
        }:
            try:
                code.remove(tag)
            except ValueError:
                pass

    for template in list(code.filter_templates(recursive=True)):
        try:
            code.remove(template)
        except ValueError:
            pass

    return code.strip_code(normalize=True, collapse=False)


def iter_wikimedia_dump(dump_path: Path) -> Iterator[Record]:
    source = "wikimedia_kowiki"
    opener = bz2.open if dump_path.suffix == ".bz2" else open

    with opener(dump_path, "rb") as handle:
        for _, page in ET.iterparse(handle, events=("end",)):
            if _local_name(page.tag) != "page":
                continue

            page_id = _direct_text(page, "id")
            title = _direct_text(page, "title")
            namespace = _direct_text(page, "ns")

            if namespace != "0":
                yield RejectedRecord(source=source, reason="non_article_namespace")
                page.clear()
                continue

            if _direct_child(page, "redirect") is not None:
                yield RejectedRecord(source=source, reason="redirect")
                page.clear()
                continue

            revision = _direct_child(page, "revision")
            text_element = _direct_child(revision, "text") if revision is not None else None
            raw_text = (
                text_element.text
                if text_element is not None and text_element.text is not None
                else ""
            )
            if not raw_text.strip():
                yield RejectedRecord(source=source, reason="empty_source_text")
                page.clear()
                continue

            try:
                text = clean_wikitext(raw_text)
            except Exception:
                yield RejectedRecord(source=source, reason="wikitext_parse_error")
                page.clear()
                continue

            yield SourceDocument(
                source=source,
                source_id=page_id or title,
                title=title,
                text=text,
                url=f"https://ko.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                license="CC BY-SA 3.0 and GFDL",
                language="ko",
                corpus="Korean Wikipedia",
                metadata={"namespace": 0},
            )
            page.clear()


def iter_historical_jsonl(
    paths: Iterable[Path],
    *,
    required_language: str = "Modern Korean",
    required_copyright: str | None = "Public Domain",
) -> Iterator[Record]:
    source = "open_korean_historical_corpus"

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    yield RejectedRecord(source=source, reason="malformed_json")
                    continue

                language = str(row.get("language") or "")
                if language != required_language:
                    yield RejectedRecord(source=source, reason="non_modern_korean")
                    continue

                copyright_value = str(row.get("copyright") or "")
                if (
                    required_copyright is not None
                    and copyright_value != required_copyright
                ):
                    yield RejectedRecord(source=source, reason="copyright_filter")
                    continue

                content = row.get("content")
                title = content.get("title", "") if isinstance(content, dict) else ""
                source_id = str(row.get("id") or f"{path.name}:{line_number}")
                metadata = row.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata = {
                    **metadata,
                    "script": row.get("script"),
                    "source_name": row.get("source"),
                    "format": row.get("format"),
                    "raw_file": path.name,
                }

                year = row.get("year")
                try:
                    year = int(year) if year is not None else None
                except (TypeError, ValueError):
                    year = None

                yield SourceDocument(
                    source=source,
                    source_id=source_id,
                    title=str(title or ""),
                    text=str(row.get("text") or ""),
                    url=str(row.get("url") or ""),
                    license="CC BY-NC 4.0",
                    language=language,
                    year=year,
                    corpus=str(row.get("corpus") or ""),
                    metadata=metadata,
                )


def iter_korean_webtext_parquet(
    paths: Iterable[Path],
    *,
    batch_size: int = 128,
) -> Iterator[Record]:
    source = "korean_webtext"

    for path in paths:
        parquet_file = pq.ParquetFile(path)
        required_columns = {
            "text",
            "source",
            "token_count",
            "__index_level_0__",
        }
        available_columns = set(parquet_file.schema_arrow.names)
        missing_columns = required_columns - available_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing KOREAN-WEBTEXT columns in {path}: {missing}")

        row_offset = 0
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=sorted(required_columns),
        ):
            for row in batch.to_pylist():
                text = str(row.get("text") or "")
                if not text.strip():
                    yield RejectedRecord(source=source, reason="empty_source_text")
                    row_offset += 1
                    continue

                upstream_index = row.get("__index_level_0__")
                source_id = f"{path.name}:{upstream_index}"
                if upstream_index is None:
                    source_id = f"{path.name}:row-{row_offset}"

                yield SourceDocument(
                    source=source,
                    source_id=source_id,
                    text=text,
                    license="Not declared in the dataset card",
                    language="ko",
                    corpus="KOREAN-WEBTEXT",
                    metadata={
                        "upstream_source": row.get("source"),
                        "upstream_token_count": row.get("token_count"),
                        "raw_file": path.name,
                    },
                )
                row_offset += 1


def _iter_ijson_documents(path: Path, prefix: str) -> Iterator[dict]:
    with path.open("rb") as handle:
        for document in ijson.items(handle, prefix):
            if isinstance(document, dict):
                yield document


def _iter_nikl_documents(path: Path) -> Iterator[dict]:
    with path.open("rb") as handle:
        first_character = b""
        while byte := handle.read(1):
            if not byte.isspace():
                first_character = byte
                break

    prefix = "item" if first_character == b"[" else "document.item"
    yield from _iter_ijson_documents(path, prefix)


def _nikl_text(document: dict) -> str:
    direct_text = document.get("text")
    if isinstance(direct_text, str):
        return direct_text

    paragraphs = document.get("paragraph")
    if not isinstance(paragraphs, list):
        return ""

    forms = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict):
            continue
        form = paragraph.get("form")
        if isinstance(form, str) and form.strip():
            forms.append(form)
    return "\n\n".join(forms)


def iter_nikl_corpus(nikl_root: Path, corpora: Iterable[str]) -> Iterator[Record]:
    for corpus_name in corpora:
        corpus = corpus_name.upper()
        corpus_root = nikl_root / corpus
        source = f"nikl_{corpus.lower()}"

        if not corpus_root.exists():
            yield RejectedRecord(source=source, reason="missing_corpus_directory")
            continue

        paths = sorted(
            path
            for path in corpus_root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".json"
        )
        if not paths:
            yield RejectedRecord(source=source, reason="missing_json_files")
            continue

        for path in paths:
            try:
                documents = _iter_nikl_documents(path)
                yielded = False
                for index, document in enumerate(documents):
                    yielded = True
                    metadata = document.get("metadata")
                    if not isinstance(metadata, dict):
                        metadata = {}
                    source_id = str(
                        document.get("id")
                        or document.get("document_id")
                        or f"{path.name}:{index}"
                    )
                    yield SourceDocument(
                        source=source,
                        source_id=source_id,
                        title=str(
                            document.get("title")
                            or metadata.get("title")
                            or ""
                        ),
                        text=_nikl_text(document),
                        license="NIKL approved-use terms",
                        language="ko",
                        corpus=corpus,
                        metadata={
                            **metadata,
                            "topic": document.get("topic"),
                            "original_topic": document.get("original_topic"),
                            "raw_file": str(path.relative_to(nikl_root)),
                        },
                    )
                if not yielded:
                    yield RejectedRecord(source=source, reason="unsupported_json_shape")
            except (OSError, ValueError, ijson.JSONError):
                yield RejectedRecord(source=source, reason="malformed_json")
