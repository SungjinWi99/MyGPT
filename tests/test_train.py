import unittest

import torch

from src.config import OptimizerConfig, SchedulerConfig
from src.train import build_optimizer, build_scheduler


class TrainUtilityTest(unittest.TestCase):
    def test_adamw_optimizer_uses_decay_and_no_decay_groups(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.LayerNorm(8),
            torch.nn.Linear(8, 2, bias=False),
        )
        optimizer = build_optimizer(
            model,
            OptimizerConfig(
                name="adamw",
                lr=0.001,
                weight_decay=0.1,
                betas=[0.9, 0.95],
                eps=1e-8,
            ),
        )

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.1)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)

        grouped_params = sum(
            len(group["params"]) for group in optimizer.param_groups
        )
        self.assertEqual(grouped_params, len(list(model.parameters())))

    def test_cosine_scheduler_warms_up_then_decays(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=1.0)
        scheduler = build_scheduler(
            optimizer,
            SchedulerConfig(
                name="cosine",
                warmup_steps=2,
                min_lr_ratio=0.1,
            ),
            total_steps=10,
        )

        lrs = []
        for _ in range(4):
            optimizer.step()
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        self.assertLess(lrs[-1], lrs[1])


if __name__ == "__main__":
    unittest.main()
