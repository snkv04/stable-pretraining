"""Poly-View Contrastive Learning.

Generalizes two-view contrastive learning to M > 2 augmented views per sample,
contrasting all views of the same instance jointly.

References:
    Shidani, Hjelm, Ramapuram, Webb, Dhekane, Busbridge. "Poly-View
    Contrastive Learning." ICLR 2024.
    https://arxiv.org/abs/2403.05490
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module


@dataclass
class PolyViewOutput(ModelOutput):
    """Output from Poly-View forward pass."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None


class PolyView(Module):
    """Poly-View contrastive learning with M views per sample.

    TODO: implement.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 2048, 256),
        temperature: float = 0.1,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()
        # TODO: build backbone, projector, and poly-view loss
        raise NotImplementedError

    def forward(self, *views: torch.Tensor) -> PolyViewOutput:
        """Forward pass.

        :param views: M augmented views, each [B, C, H, W]. A single view
            returns only the backbone embedding (eval mode).
        :return: :class:`PolyViewOutput`.
        """
        # TODO: encode + project all views, compute poly-view contrastive loss
        raise NotImplementedError
