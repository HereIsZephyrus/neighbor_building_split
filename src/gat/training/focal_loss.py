"""
Focal Loss implementation for handling class imbalance.

Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

Benefits for small datasets with imbalanced classes:
- Down-weights easy examples (well-classified)
- Focuses on hard examples (misclassified or uncertain)
- Prevents model from always predicting dominant class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)

    Args:
        alpha: Class weights tensor of shape (num_classes,). If None, no class weighting.
        gamma: Focusing parameter (>= 0). Higher gamma puts more focus on hard examples.
               - gamma=0: equivalent to CrossEntropyLoss
               - gamma=2: recommended default
        reduction: Reduction method ('mean', 'sum', or 'none')
        label_smoothing: Label smoothing factor (0.0 to 1.0)

    Example:
        >>> criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if not 0 <= label_smoothing < 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Logits tensor of shape (N, C) where C = number of classes
            targets: Ground truth labels of shape (N,) with values in [0, C-1]

        Returns:
            Focal loss scalar
        """
        # Get number of classes
        num_classes = inputs.shape[1]

        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            # Create smooth labels: (1 - ε) for true class, ε/(C-1) for others
            smooth_labels = torch.zeros_like(inputs)
            smooth_labels.fill_(self.label_smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

            # Compute cross entropy with smooth labels
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -(smooth_labels * log_probs).sum(dim=1)

            # Get predicted probability for the true class
            probs = F.softmax(inputs, dim=1)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            # Standard focal loss computation
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)  # p_t = probability of true class

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute focal loss
        focal_loss = focal_weight * ce_loss

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.

    Label smoothing prevents the model from becoming over-confident
    and improves generalization on small datasets.

    Args:
        smoothing: Smoothing factor (0.0 to 1.0)
                  - 0.0: no smoothing (standard CE)
                  - 0.1: 10% smoothing (recommended)
        weight: Class weights tensor
        reduction: Reduction method
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross entropy loss.

        Args:
            inputs: Logits tensor (N, C)
            targets: Ground truth labels (N,)

        Returns:
            Loss scalar
        """
        num_classes = inputs.shape[1]
        log_probs = F.log_softmax(inputs, dim=1)

        # Create smooth labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_probs)
            smooth_labels.fill_(self.smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Compute loss
        loss = -(smooth_labels * log_probs).sum(dim=1)

        # Apply class weights if provided
        if self.weight is not None:
            if self.weight.device != inputs.device:
                self.weight = self.weight.to(inputs.device)
            weight_t = self.weight.gather(0, targets)
            loss = loss * weight_t

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_loss_function(
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    focal_loss: bool = False,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0
) -> nn.Module:
    """
    Factory function to create appropriate loss function.

    Args:
        num_classes: Number of classes
        class_weights: Class weight tensor of shape (num_classes,)
        focal_loss: Whether to use Focal Loss
        focal_gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor

    Returns:
        Loss function module
    """
    if focal_loss:
        return FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing
        )
    elif label_smoothing > 0:
        return LabelSmoothingCrossEntropy(
            smoothing=label_smoothing,
            weight=class_weights
        )
    else:
        return nn.CrossEntropyLoss(weight=class_weights)

