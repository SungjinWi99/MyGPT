import math
import unittest

import torch
import torch.nn.functional as F

import src.model  # noqa: F401 - registers model classes in ModelFactory.
from src.config import ModelConfig, ModelFactory


class MyGPT2ModelTest(unittest.TestCase):
    def test_model_factory_accepts_dropout_and_forward_runs(self):
        config = ModelConfig(
            model_name="MyGPT2",
            d_model=64,
            vocab_size=512,
            n_decoder_blocks=2,
            n_attention_heads=4,
            dropout=0.1,
            max_seq_len=32,
        )
        model = ModelFactory.build_model_from_config(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        with torch.no_grad():
            logits = model(input_ids)

        self.assertEqual(logits.shape, (2, 16, config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(model.tie_embeddings)
        self.assertFalse(hasattr(model, "lm_head"))

    def test_model_factory_can_disable_weight_tying(self):
        config = ModelConfig(
            model_name="MyGPT2",
            d_model=64,
            vocab_size=512,
            n_decoder_blocks=2,
            n_attention_heads=4,
            dropout=0.0,
            max_seq_len=32,
            tie_embeddings=False,
        )
        model = ModelFactory.build_model_from_config(config)
        model.eval()

        self.assertFalse(model.tie_embeddings)
        self.assertTrue(hasattr(model, "lm_head"))
        self.assertNotEqual(
            model.embedding.weight.data_ptr(),
            model.lm_head.weight.data_ptr(),
        )

        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        with torch.no_grad():
            logits = model(input_ids)

        self.assertEqual(logits.shape, (2, 16, config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_gpt_style_init_keeps_initial_logits_well_scaled(self):
        torch.manual_seed(0)
        model = ModelFactory.build_model_from_config(
            ModelConfig(
                model_name="MyGPT2",
                d_model=128,
                vocab_size=4096,
                n_decoder_blocks=4,
                n_attention_heads=4,
                dropout=0.0,
                max_seq_len=64,
            )
        )
        model.eval()

        embedding_std = model.embedding.weight.detach().std().item()
        self.assertGreater(embedding_std, 0.015)
        self.assertLess(embedding_std, 0.025)

        input_ids = torch.randint(0, 4096, (2, 32))
        labels = torch.randint(0, 4096, (2, 32))
        with torch.no_grad():
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

        self.assertLess(abs(loss.item() - math.log(4096)), 1.0)


if __name__ == "__main__":
    unittest.main()
