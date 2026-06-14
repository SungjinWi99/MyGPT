import bz2
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from src.dataset_pipeline.adapters import (
    iter_historical_jsonl,
    iter_nikl_corpus,
    iter_wikimedia_dump,
)
from src.dataset_pipeline.builder import DatasetBuilder, normalize_text
from src.dataset_pipeline.prepare_pretrain import parse_args
from src.dataset_pipeline.schema import RejectedRecord, SourceDocument


class FakeTokenizer:
    vocab_size = 256
    special_tokens_map = {}

    def __call__(self, texts, **_):
        return {"input_ids": [list(text.encode("utf-8")) for text in texts]}


class DatasetAdapterTest(unittest.TestCase):
    def test_pretrain_cli_defaults_exclude_nikl(self):
        with patch(
            "sys.argv",
            ["prepare_pretrain", "--output-dir", "/tmp/pretrain-test"],
        ):
            args = parse_args()

        self.assertEqual(args.sources, ("wikimedia", "historical"))

    def test_wikimedia_adapter_keeps_articles_and_rejects_redirects(self):
        xml = """\
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>테스트 문서</title><ns>0</ns><id>1</id>
    <revision><text>본문 {{틀}} 문장.&lt;ref&gt;출처&lt;/ref&gt;</text></revision>
  </page>
  <page>
    <title>넘겨주기</title><ns>0</ns><id>2</id>
    <redirect title="테스트 문서"/>
    <revision><text>#REDIRECT</text></revision>
  </page>
  <page>
    <title>토론:테스트</title><ns>1</ns><id>3</id>
    <revision><text>토론</text></revision>
  </page>
</mediawiki>
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "wiki.xml.bz2"
            with bz2.open(path, "wt", encoding="utf-8") as handle:
                handle.write(xml)

            records = list(iter_wikimedia_dump(path))

        documents = [record for record in records if isinstance(record, SourceDocument)]
        rejected = [record.reason for record in records if isinstance(record, RejectedRecord)]
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].source_id, "1")
        self.assertIn("본문", documents[0].text)
        self.assertNotIn("출처", documents[0].text)
        self.assertCountEqual(rejected, ["redirect", "non_article_namespace"])

    def test_historical_adapter_keeps_only_modern_public_domain(self):
        rows = [
            {
                "id": "keep",
                "text": "현대 한국어 문서",
                "language": "Modern Korean",
                "copyright": "Public Domain",
                "corpus": "News",
            },
            {
                "id": "language",
                "text": "고전 문서",
                "language": "Classical Chinese",
                "copyright": "Public Domain",
            },
            {
                "id": "copyright",
                "text": "현대 문서",
                "language": "Modern Korean",
                "copyright": "Restricted",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            records = list(iter_historical_jsonl([path]))

        documents = [record for record in records if isinstance(record, SourceDocument)]
        rejected = [record.reason for record in records if isinstance(record, RejectedRecord)]
        self.assertEqual([document.source_id for document in documents], ["keep"])
        self.assertCountEqual(rejected, ["non_modern_korean", "copyright_filter"])

    def test_nikl_adapter_reads_written_documents(self):
        payload = {
            "document": [
                {
                    "id": "NIKL-1",
                    "metadata": {"title": "문어 말뭉치"},
                    "paragraph": [
                        {"form": "첫 문단입니다."},
                        {"form": "둘째 문단입니다."},
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus_dir = root / "WRITTEN"
            corpus_dir.mkdir()
            (corpus_dir / "sample.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            records = list(iter_nikl_corpus(root, ["WRITTEN"]))

        self.assertEqual(len(records), 1)
        self.assertIsInstance(records[0], SourceDocument)
        self.assertEqual(records[0].source_id, "NIKL-1")
        self.assertEqual(records[0].text, "첫 문단입니다.\n\n둘째 문단입니다.")


class DatasetBuilderTest(unittest.TestCase):
    def test_builder_deduplicates_and_writes_manifest_and_parquet(self):
        first_stream = [
            SourceDocument(
                source="source_a",
                source_id="1",
                text="같은 문서\r\n",
                license="test",
            ),
            RejectedRecord(source="source_a", reason="adapter_rejection"),
        ]
        second_stream = [
            SourceDocument(
                source="source_b",
                source_id="2",
                text="같은 문서",
                license="test",
            ),
            SourceDocument(
                source="source_b",
                source_id="3",
                text="다른 문서",
                license="test",
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "pretrain" / "v1"
            builder = DatasetBuilder(
                output_dir=output_dir,
                work_dir=root / "work",
                tokenizer=FakeTokenizer(),
                tokenizer_info={"name": "fake", "revision": "test"},
                source_config={"source_a": {}, "source_b": {}},
                shard_rows=1,
                tokenize_batch_size=2,
                validation_fraction=0.0,
            )
            builder.build(
                [
                    ("source_a", first_stream),
                    ("source_b", second_stream),
                ]
            )

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            parquet_paths = sorted((output_dir / "parquet" / "train").glob("*.parquet"))
            rows = sum(pq.read_table(path).num_rows for path in parquet_paths)

        self.assertEqual(rows, 2)
        self.assertEqual(manifest["profile"]["total"]["accepted_records"], 2)
        self.assertEqual(
            manifest["profile"]["sources"]["source_b"]["rejected"]["exact_duplicate"],
            1,
        )
        self.assertEqual(
            manifest["profile"]["sources"]["source_a"]["rejected"]["adapter_rejection"],
            1,
        )

    def test_normalization_is_stable(self):
        self.assertEqual(
            normalize_text(" 문장  \r\n\r\n\r\n다음 문장 \t"),
            "문장\n\n다음 문장",
        )


if __name__ == "__main__":
    unittest.main()
