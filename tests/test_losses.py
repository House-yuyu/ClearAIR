"""Regression tests for differentiable ClearAIR losses."""

import unittest

import torch

from clearair.losses import ICRMLoss


class ICRMLossTest(unittest.TestCase):
    def test_internal_clue_loss_backpropagates_to_restored_image(self):
        torch.manual_seed(0)
        restored = torch.rand(1, 3, 32, 32, requires_grad=True)

        loss = ICRMLoss()(restored)

        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(restored.grad)
        self.assertTrue(torch.isfinite(restored.grad).all())
        self.assertGreater(restored.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
