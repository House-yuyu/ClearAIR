"""Tests for the lightweight wrappers around frozen external backends."""

import unittest

import torch
import torch.nn as nn

from clearair.auxiliary import MLLMIQA, SemanticGuidanceUnit, TaskIdentifier


class FakeDeQA(nn.Module):
    @torch.inference_mode()
    def forward(self, images, prompt):
        del prompt
        return torch.ones(images.shape[0], 4, device=images.device)


class FakeSAM2(nn.Module):
    @torch.inference_mode()
    def forward(self, images, num_masks):
        return torch.ones(
            images.shape[0], num_masks, images.shape[-2], images.shape[-1], device=images.device
        )


class FakeDAClip(nn.Module):
    @torch.inference_mode()
    def forward(self, images):
        feature = torch.ones(images.shape[0], 8, device=images.device)
        return feature, -feature


class RealWrapperTest(unittest.TestCase):
    def test_deqa_state_is_projected_with_gradient(self):
        wrapper = MLLMIQA(hidden_dim=4, out_dim=2, dummy=False, mllm=FakeDeQA())
        output = wrapper(torch.rand(2, 3, 8, 8))
        self.assertEqual(output.shape, (2, 2))
        output.sum().backward()
        self.assertIsNotNone(wrapper.proj.weight.grad)

    def test_sam2_masks_are_forwarded(self):
        wrapper = SemanticGuidanceUnit(
            num_masks=3, mask_dropout=0.0, dummy=False, sam2=FakeSAM2()
        )
        masks = wrapper(torch.rand(2, 3, 8, 10))
        self.assertEqual(masks.shape, (2, 3, 8, 10))

    def test_daclip_embeddings_are_forwarded(self):
        wrapper = TaskIdentifier(embed_dim=8, dummy=False, da_clip=FakeDAClip())
        content, degradation = wrapper(torch.rand(2, 3, 8, 8))
        self.assertEqual(content.shape, (2, 8))
        self.assertTrue(torch.equal(degradation, -content))


if __name__ == "__main__":
    unittest.main()
