import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class PackedPretrainDataset(Dataset):
    """Fixed-length autoregressive LM samples built from cleaned pretrain Parquet."""

    def __init__(
        self,
        dataset_dir: str | Path,
        tokenizer: Any,
        max_seq_len: int,
        *,
        split: str = "train",
        cache_dir: str | Path | None = None,
        rebuild_cache: bool = False,
        parquet_batch_size: int = 1024,
        tokenize_batch_size: int = 128,
        token_budget: int | None = None,
        eos_token_id: int | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tokens_per_sample = max_seq_len + 1
        self.split = split
        self.eos_token_id = self._resolve_eos_token_id(tokenizer, eos_token_id)

        if token_budget is not None and token_budget < max_seq_len:
            raise ValueError("token_budget must be at least max_seq_len")

        self.cache_dir = Path(cache_dir) if cache_dir else self._default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bin_path = self.cache_dir / f"{split}.uint32.bin"
        self.meta_path = self.cache_dir / f"{split}.meta.json"

        if rebuild_cache or not self.bin_path.exists() or not self.meta_path.exists():
            self._build_cache(
                parquet_batch_size=parquet_batch_size,
                tokenize_batch_size=tokenize_batch_size,
                token_budget=token_budget,
            )

        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self._validate_meta()
        self.num_samples = int(self.meta["num_samples"])
        self._data = np.memmap(
            self.bin_path,
            dtype=np.uint32,
            mode="r",
            shape=(self.num_samples, self.tokens_per_sample),
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.num_samples
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)

        tokens = torch.as_tensor(self._data[index].astype(np.int64), dtype=torch.long)
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }

    def _default_cache_dir(self) -> Path:
        tokenizer_name = getattr(self.tokenizer, "name_or_path", "tokenizer")
        tokenizer_name = "".join(
            character if character.isalnum() else "-"
            for character in str(tokenizer_name)
        ).strip("-")
        cache_name = f"packed-{tokenizer_name}-seq{self.max_seq_len}"
        return self.dataset_dir / "tokenized" / cache_name

    @staticmethod
    def _resolve_eos_token_id(tokenizer: Any, eos_token_id: int | None) -> int:
        if eos_token_id is not None:
            return eos_token_id
        tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
        if tokenizer_eos is None:
            raise ValueError("tokenizer.eos_token_id is required for document packing")
        return int(tokenizer_eos)

    def _validate_meta(self) -> None:
        expected = {
            "dataset_dir": str(self.dataset_dir),
            "split": self.split,
            "max_seq_len": self.max_seq_len,
            "tokens_per_sample": self.tokens_per_sample,
            "eos_token_id": self.eos_token_id,
        }
        for key, value in expected.items():
            if self.meta.get(key) != value:
                raise ValueError(
                    f"Packed cache metadata mismatch for {key}: "
                    f"expected {value!r}, found {self.meta.get(key)!r}. "
                    "Use rebuild_cache=True or choose a different cache_dir."
                )
        if self.meta["num_samples"] <= 0:
            raise ValueError(f"Packed cache has no samples: {self.meta_path}")

    def _parquet_paths(self) -> list[Path]:
        parquet_dir = self.dataset_dir / "parquet" / self.split
        if not parquet_dir.exists():
            raise FileNotFoundError(f"Missing parquet split directory: {parquet_dir}")

        paths = sorted(parquet_dir.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
        return paths

    def _build_cache(
        self,
        *,
        parquet_batch_size: int,
        tokenize_batch_size: int,
        token_budget: int | None,
    ) -> None:
        tmp_bin_path = self.bin_path.with_suffix(".bin.tmp")
        tmp_meta_path = self.meta_path.with_suffix(".json.tmp")
        max_samples = None
        if token_budget is not None:
            max_samples = token_budget // self.max_seq_len

        num_samples = 0
        source_documents = 0
        source_tokens = 0
        buffer: list[int] = []

        with tmp_bin_path.open("wb") as output:
            for texts in self._iter_text_batches(parquet_batch_size):
                for offset in range(0, len(texts), tokenize_batch_size):
                    text_batch = texts[offset : offset + tokenize_batch_size]
                    encodings = self.tokenizer(
                        text_batch,
                        add_special_tokens=False,
                        padding=False,
                        truncation=False,
                        return_attention_mask=False,
                    )

                    for token_ids in encodings["input_ids"]:
                        if not token_ids:
                            continue

                        source_documents += 1
                        source_tokens += len(token_ids)
                        buffer.extend(int(token_id) for token_id in token_ids)
                        buffer.append(self.eos_token_id)

                        while len(buffer) >= self.tokens_per_sample:
                            sample = np.asarray(
                                buffer[: self.tokens_per_sample],
                                dtype=np.uint32,
                            )
                            sample.tofile(output)
                            del buffer[: self.max_seq_len]
                            num_samples += 1

                            if max_samples is not None and num_samples >= max_samples:
                                self._finish_cache(
                                    tmp_bin_path,
                                    tmp_meta_path,
                                    num_samples,
                                    source_documents,
                                    source_tokens,
                                    token_budget,
                                )
                                return

        self._finish_cache(
            tmp_bin_path,
            tmp_meta_path,
            num_samples,
            source_documents,
            source_tokens,
            token_budget,
        )

    def _iter_text_batches(self, parquet_batch_size: int) -> list[str]:
        for path in self._parquet_paths():
            parquet_file = pq.ParquetFile(path)
            if "text" not in parquet_file.schema_arrow.names:
                raise ValueError(f"Parquet file has no text column: {path}")

            for record_batch in parquet_file.iter_batches(
                batch_size=parquet_batch_size,
                columns=["text"],
            ):
                yield record_batch.column("text").to_pylist()

    def _finish_cache(
        self,
        tmp_bin_path: Path,
        tmp_meta_path: Path,
        num_samples: int,
        source_documents: int,
        source_tokens: int,
        token_budget: int | None,
    ) -> None:
        if num_samples <= 0:
            tmp_bin_path.unlink(missing_ok=True)
            raise ValueError(
                f"No packed samples were produced from {self.dataset_dir} "
                f"split={self.split}"
            )

        meta = {
            "dataset_dir": str(self.dataset_dir),
            "split": self.split,
            "max_seq_len": self.max_seq_len,
            "tokens_per_sample": self.tokens_per_sample,
            "eos_token_id": self.eos_token_id,
            "dtype": "uint32",
            "num_samples": num_samples,
            "training_tokens": num_samples * self.max_seq_len,
            "stored_tokens": num_samples * self.tokens_per_sample,
            "source_documents_seen": source_documents,
            "source_tokens_seen": source_tokens,
            "token_budget": token_budget,
            "packing_policy": "concatenate_documents_with_eos_stride_max_seq_len",
        }
        tmp_meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_bin_path.replace(self.bin_path)
        tmp_meta_path.replace(self.meta_path)


PretrainDataset = PackedPretrainDataset


class KoAlpacaDataset(Dataset):
    def __init__(self, *_, **__):
        raise RuntimeError(
            "KoAlpacaDataset is no longer the default dataset for this project. "
            "Use PackedPretrainDataset(dataset_dir, tokenizer, max_seq_len, split=...) "
            "for the cleaned pretrain parquet dataset."
        )


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int | None = None,
) -> dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    if all(tensor.size(0) == input_ids[0].size(0) for tensor in input_ids):
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
        }

    if pad_token_id is None:
        raise ValueError("pad_token_id is required for variable-length batches")

    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        ),
    }
