import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.data import PackedPretrainDataset, collate_fn


class FakeTokenizer:
    eos_token_id = 0
    name_or_path = "fake-tokenizer"

    def __call__(self, texts, **_):
        return {
            "input_ids": [
                [ord(character) % 255 + 1 for character in text]
                for text in texts
            ]
        }


class PackedPretrainDatasetTest(unittest.TestCase):
    def test_builds_packed_cache_from_pretrain_parquet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "pretrain" / "v1"
            train_dir = dataset_dir / "parquet" / "train"
            train_dir.mkdir(parents=True)
            pq.write_table(
                pa.table({"text": ["가나다라마", "바사아자차"]}),
                train_dir / "part-00000.parquet",
            )

            dataset = PackedPretrainDataset(
                dataset_dir,
                FakeTokenizer(),
                max_seq_len=4,
                split="train",
            )

            self.assertEqual(len(dataset), 2)
            first = dataset[0]
            self.assertEqual(first["input_ids"].shape, torch.Size([4]))
            self.assertEqual(first["labels"].shape, torch.Size([4]))
            self.assertTrue(torch.equal(first["input_ids"][1:], first["labels"][:-1]))
            self.assertTrue((dataset.cache_dir / "train.uint32.bin").exists())
            self.assertTrue((dataset.cache_dir / "train.meta.json").exists())

    def test_collate_stacks_fixed_length_language_model_batches(self):
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([2, 3, 4]),
            },
            {
                "input_ids": torch.tensor([5, 6, 7]),
                "labels": torch.tensor([6, 7, 8]),
            },
        ]

        collated = collate_fn(batch)

        self.assertEqual(collated["input_ids"].shape, torch.Size([2, 3]))
        self.assertEqual(collated["labels"].shape, torch.Size([2, 3]))


if __name__ == "__main__":
    unittest.main()
