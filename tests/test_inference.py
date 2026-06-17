import unittest

import torch

from src.inference import _apply_top_k_top_p, _strip_compile_prefix


class InferenceUtilityTest(unittest.TestCase):
    def test_strip_compile_prefix(self):
        state_dict = {
            "_orig_mod.embedding.weight": torch.ones(2, 2),
            "_orig_mod.normalize.gamma": torch.ones(2),
        }

        stripped = _strip_compile_prefix(state_dict)

        self.assertIn("embedding.weight", stripped)
        self.assertIn("normalize.gamma", stripped)
        self.assertNotIn("_orig_mod.embedding.weight", stripped)

    def test_top_k_filter_keeps_only_requested_logits(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        filtered = _apply_top_k_top_p(logits, top_k=2, top_p=1.0)

        self.assertTrue(torch.isneginf(filtered[0, 0]))
        self.assertTrue(torch.isneginf(filtered[0, 1]))
        self.assertFalse(torch.isneginf(filtered[0, 2]))
        self.assertFalse(torch.isneginf(filtered[0, 3]))


if __name__ == "__main__":
    unittest.main()
