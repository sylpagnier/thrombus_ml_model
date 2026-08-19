"""Differentiable loss functions for training/tuning the wall model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceF1Loss(nn.Module):
    """Differentiable Soft Dice / F1 loss masked to the vessel wall."""

    def __init__(self, *, beta: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.beta2 = beta ** 2
        self.eps = eps

    def forward(
        self,
        prob_pred: torch.Tensor,
        target_gt: torch.Tensor,
        wall_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute soft F1 loss on wall nodes.

        prob_pred: [N] continuous probability in [0, 1]
        target_gt: [N] binary GT clot (0 or 1)
        wall_mask: [N] boolean/float mask indicating wall nodes
        """
        p = prob_pred
        y = target_gt
        if wall_mask is not None:
            mask = wall_mask.float()
            p = p * mask
            y = y * mask

        # If GT has no clot at all on this vessel, penalize any false positive mass
        y_sum = y.sum()
        p_sum = p.sum()

        tp = (p * y).sum()
        fp = (p * (1.0 - y)).sum()
        fn = ((1.0 - p) * y).sum()

        f_beta = ((1.0 + self.beta2) * tp + self.eps) / (
            (1.0 + self.beta2) * tp + self.beta2 * fn + fp + self.eps
        )
        return 1.0 - f_beta


class CombinedWallClotLoss(nn.Module):
    """Combines Soft Dice/F1 loss with BCE and false-positive mass penalty."""

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        bce_weight: float = 0.1,
        no_clot_fp_weight: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dice_loss = SoftDiceF1Loss(eps=eps)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.no_clot_fp_weight = no_clot_fp_weight
        self.eps = eps

    def forward(
        self,
        prob_pred: torch.Tensor,
        target_gt: torch.Tensor,
        wall_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = wall_mask.float()
        p = torch.clamp(prob_pred * mask, min=1e-7, max=1.0 - 1e-7)
        y = target_gt * mask

        # Dice loss
        l_dice = self.dice_loss(p, y, mask)

        # BCE loss on wall nodes
        l_bce = F.binary_cross_entropy(p[mask > 0], y[mask > 0])

        # Extra penalty if vessel has zero GT clot but model predicts clot
        y_sum = y.sum()
        if y_sum == 0:
            l_fp = p.sum() / max(mask.sum(), 1.0)
            total = l_dice + self.no_clot_fp_weight * l_fp + self.bce_weight * l_bce
        else:
            total = self.dice_weight * l_dice + self.bce_weight * l_bce

        return {
            "loss": total,
            "dice_loss": l_dice,
            "bce_loss": l_bce,
        }
