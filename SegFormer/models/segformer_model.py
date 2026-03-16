import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

import config


class SegFormerChange(nn.Module):
    def __init__(self, num_classes=7, model_name=None):
        super().__init__()

        if model_name is None:
            model_name = config.MODEL_NAME

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        # Modify first layer (3 → 6 channels)
        old_conv = self.model.segformer.encoder.patch_embeddings[0].proj

        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )

        # Copy weights smartly
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight

        self.model.segformer.encoder.patch_embeddings[0].proj = new_conv

    def forward(self, x):
        return self.model(pixel_values=x).logits