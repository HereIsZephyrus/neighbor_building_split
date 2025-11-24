"""
LCZ Similarity-Aware Loss Function

Replaces standard CrossEntropyLoss with similarity-aware soft labels for LCZ classification.

Key Concept:
- LCZ classes have semantic similarity (LCZ1 closer to LCZ2 than LCZ9)
- Reduces penalty for predicting similar classes
- Improves learning on ordinal/continuous label spaces

Configuration (training_config.yaml):
    similarity_loss:
      enabled: true
      temperature: 0.1  # 0.05-0.3, controls soft label strength

LCZ Similarity Matrix:
- Rows/cols correspond to LCZ1-9 (indices 0-8)
- Values in [0,1]: 1.0 = identical class, <1.0 = different but similar
- Based on building density, height, and spatial arrangement

Example:
    LCZ1 (compact high-rise) has high similarity to LCZ2 (compact mid-rise)
    but low similarity to LCZ9 (sparse low-rise)

Usage:
    >>> # Training with similarity loss
    >>> loss_fn = create_loss_function(
    ...     num_classes=9,
    ...     class_weights=weights,
    ...     use_similarity_loss=True,
    ...     similarity_temperature=0.1
    ... )
    >>> loss = loss_fn(logits, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def create_lcz_similarity_matrix(num_classes: int = 9) -> torch.Tensor:
    """
    Create LCZ label similarity matrix based on two-dimensional semantic structure.

    LCZ Classification Structure (Density × Height):

    Density Groups:
    - LCZ1-3: Compact buildings (high density)
      * LCZ1: Compact high-rise (>10 stories)
      * LCZ2: Compact mid-rise (3-9 stories)
      * LCZ3: Compact low-rise (1-3 stories)

    - LCZ4-6: Open buildings (medium density)
      * LCZ4: Open high-rise (>10 stories)
      * LCZ5: Open mid-rise (3-9 stories)
      * LCZ6: Open low-rise (1-3 stories)

    - LCZ7-9: Sparse buildings (low density)
      * LCZ7: Lightweight low-rise
      * LCZ8: Large low-rise (industrial, commercial)
      * LCZ9: Sparsely built (scattered buildings)

    Height Groups:
    - High-rise: LCZ1, LCZ4
    - Mid-rise: LCZ2, LCZ5
    - Low-rise: LCZ3, LCZ6, LCZ7, LCZ8, LCZ9

    Similarity Design Principles (Two-dimensional):

    1. Within same density group (vertical similarity):
       - Adjacent heights: 0.70-0.80 (e.g., LCZ1↔LCZ2, LCZ2↔LCZ3)
       - Non-adjacent heights: 0.50-0.60 (e.g., LCZ1↔LCZ3)

    2. Within same height group (horizontal similarity):
       - Compact↔Open (same height): 0.45-0.55
       - Open↔Sparse (low-rise only): 0.25-0.35
       - Compact↔Sparse: 0.00 (opposite extremes)

    3. Different density AND different height:
       - Set to 0.00-0.15 (too dissimilar)

    Args:
        num_classes: Number of LCZ classes (default 9)

    Returns:
        Similarity matrix of shape (num_classes, num_classes)
    """
    if num_classes != 9:
        raise ValueError(f"LCZ similarity matrix only supports 9 classes, got {num_classes}")

    # Stricter similarity matrix considering both density and height dimensions
    # Matrix structure:
    #           Compact          Open            Sparse
    #         [1    2    3]   [4    5    6]   [7    8    9]
    # High:    1              4                -
    # Mid:     2              5                -
    # Low:     3              6                7    8    9
    # 
    # Design rationale:
    # - Same density group, adjacent height: 0.70-0.80
    # - Same density group, non-adjacent height: 0.50-0.60
    # - Same height, Compact↔Open: 0.45-0.55
    # - Same height (low), Open↔Sparse: 0.25-0.35
    # - Compact↔Sparse (any height): 0.00 (too different)
    # - Different density AND height: 0.00-0.15 (minimal similarity)
    similarity = torch.tensor([
        # LCZ1  LCZ2  LCZ3  LCZ4  LCZ5  LCZ6  LCZ7  LCZ8  LCZ9
        [1.00, 0.75, 0.55, 0.50, 0.30, 0.15, 0.00, 0.00, 0.00],  # LCZ1: Compact high-rise
        [0.75, 1.00, 0.75, 0.35, 0.50, 0.25, 0.00, 0.00, 0.00],  # LCZ2: Compact mid-rise
        [0.55, 0.75, 1.00, 0.20, 0.35, 0.45, 0.15, 0.15, 0.00],  # LCZ3: Compact low-rise
        [0.50, 0.35, 0.20, 1.00, 0.75, 0.55, 0.00, 0.00, 0.00],  # LCZ4: Open high-rise
        [0.30, 0.50, 0.35, 0.75, 1.00, 0.75, 0.15, 0.15, 0.00],  # LCZ5: Open mid-rise
        [0.15, 0.25, 0.45, 0.55, 0.75, 1.00, 0.30, 0.30, 0.15],  # LCZ6: Open low-rise
        [0.00, 0.00, 0.15, 0.00, 0.15, 0.30, 1.00, 0.70, 0.50],  # LCZ7: Lightweight low-rise
        [0.00, 0.00, 0.15, 0.00, 0.15, 0.30, 0.70, 1.00, 0.75],  # LCZ8: Large low-rise
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.15, 0.50, 0.75, 1.00],  # LCZ9: Sparsely built
    ], dtype=torch.float32)

    # Verify symmetry
    assert torch.allclose(similarity, similarity.T), "Similarity matrix must be symmetric"

    # Verify all values in [0, 1]
    assert (similarity >= 0).all() and (similarity <= 1).all(), "Similarity values must be in [0, 1]"

    # Verify diagonal is 1.0
    assert torch.allclose(similarity.diag(), torch.ones(num_classes)), "Diagonal must be 1.0"

    return similarity


class SimilarityAwareCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss with label similarity awareness for LCZ classification.

    Converts hard labels to soft labels based on a similarity matrix, reducing
    penalty for predicting similar classes (e.g., LCZ1 predicted as LCZ2).

    The soft label is computed as:
        soft_label = (1 - temperature) * one_hot + temperature * similarity[target]

    Where:
    - temperature controls the strength of similarity smoothing
    - temperature=0.0 → standard one-hot (no smoothing)
    - temperature=0.1 → mild smoothing (recommended)
    - temperature=0.3 → strong smoothing

    Args:
        similarity_matrix: Pre-computed similarity matrix (num_classes, num_classes)
        temperature: Smoothing temperature in [0, 1], controls soft label strength
        alpha: Optional class weights tensor of shape (num_classes,)
        reduction: Reduction method ('mean', 'sum', or 'none')

    Example:
        >>> similarity = create_lcz_similarity_matrix(num_classes=9)
        >>> criterion = SimilarityAwareCrossEntropyLoss(
        ...     similarity_matrix=similarity,
        ...     temperature=0.1,
        ...     alpha=class_weights
        ... )
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        similarity_matrix: torch.Tensor,
        temperature: float = 0.1,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()

        if not 0 <= temperature <= 1:
            raise ValueError(f"temperature must be in [0, 1], got {temperature}")

        self.register_buffer('similarity_matrix', similarity_matrix)
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

        # Pre-compute normalized similarity for soft targets
        # Each row sums to 1.0 (probability distribution)
        self.register_buffer('soft_targets', F.normalize(similarity_matrix, p=1, dim=1))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity-aware cross entropy loss.

        Args:
            inputs: Logits tensor of shape (N, C) where C = number of classes
            targets: Ground truth labels of shape (N,) with values in [0, C-1]

        Returns:
            Loss scalar (or tensor if reduction='none')
        """
        num_classes = inputs.size(1)

        # Get soft_targets on the correct device
        soft_targets = self.soft_targets
        if soft_targets.device != inputs.device:
            soft_targets = soft_targets.to(inputs.device)

        # Create one-hot encoding
        one_hot = torch.zeros(targets.size(0), num_classes, dtype=torch.float32, device=inputs.device)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Create soft labels using similarity matrix
        # soft_label = (1-temp) * one_hot + temp * similarity[target]
        soft_labels = (1 - self.temperature) * one_hot + \
                     self.temperature * soft_targets[targets]

        # Compute log probabilities
        log_probs = F.log_softmax(inputs, dim=1)

        # Compute cross entropy with soft labels
        # CE = -sum(soft_labels * log_probs)
        loss = -(soft_labels * log_probs).sum(dim=1)

        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def create_loss_function(
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    use_similarity_loss: bool = True,
    similarity_temperature: float = 0.1
) -> nn.Module:
    """
    Factory function to create appropriate loss function.

    Args:
        num_classes: Number of classes (must be 9 for LCZ)
        class_weights: Optional class weight tensor of shape (num_classes,)
        use_similarity_loss: Whether to use similarity-aware loss
        similarity_temperature: Temperature for soft label smoothing (0.0-0.3)

    Returns:
        Loss function module (SimilarityAwareCrossEntropyLoss or nn.CrossEntropyLoss)

    Example:
        >>> # With similarity loss
        >>> criterion = create_loss_function(
        ...     num_classes=9,
        ...     class_weights=weights,
        ...     use_similarity_loss=True,
        ...     similarity_temperature=0.1
        ... )

        >>> # Without similarity loss (standard CE)
        >>> criterion = create_loss_function(
        ...     num_classes=9,
        ...     class_weights=weights,
        ...     use_similarity_loss=False
        ... )
    """
    if use_similarity_loss:
        if num_classes != 9:
            raise ValueError(
                f"Similarity loss only supports 9 LCZ classes, got {num_classes}. "
                f"Set use_similarity_loss=False for other class counts."
            )

        similarity_matrix = create_lcz_similarity_matrix(num_classes)

        return SimilarityAwareCrossEntropyLoss(
            similarity_matrix=similarity_matrix,
            temperature=similarity_temperature,
            alpha=class_weights
        )
    else:
        # Standard cross entropy loss
        return nn.CrossEntropyLoss(weight=class_weights)


class ClusterPurityRewardLoss(nn.Module):
    """
    Loss function with cluster purity reward.

    Core idea:
    - Even if predictions are wrong, consistent errors should be rewarded
    - Encourages "consistent mistakes" rather than "random mistakes"
    - Corresponds to concentration in confusion matrix

    Purity calculation:
    - For each true class, compute entropy of its predictions
    - Lower entropy = more concentrated predictions (higher purity)
    - Example: 10 samples all predicted as class 3 → entropy=0 (perfect purity)
    - Example: 10 samples uniformly distributed → entropy=max (worst purity)

    Total loss = classification_loss - λ_purity × purity_reward

    Args:
        base_criterion: Base classification loss (e.g., SimilarityAwareCrossEntropyLoss)
        lambda_purity: Purity reward weight (recommended 0.05-0.2)
        min_samples_per_class: Minimum samples, classes with fewer samples excluded

    Example:
        >>> base_loss = create_loss_function(num_classes=9, ...)
        >>> criterion = ClusterPurityRewardLoss(
        ...     base_criterion=base_loss,
        ...     lambda_purity=0.1
        ... )
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        lambda_purity: float = 0.1,
        min_samples_per_class: int = 2
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.lambda_purity = lambda_purity
        self.min_samples_per_class = min_samples_per_class

        if lambda_purity < 0:
            raise ValueError(f"lambda_purity must be non-negative, got {lambda_purity}")

    def compute_cluster_purity(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """
        Compute cluster purity based on prediction entropy.

        For each true class:
        1. Find all samples belonging to that class
        2. Compute prediction distribution of these samples
        3. Calculate entropy of prediction distribution (lower = higher purity)

        Args:
            predictions: Predicted labels (N,)
            targets: True labels (N,)
            num_classes: Total number of classes

        Returns:
            Average purity (normalized to [0, 1], 1 = perfect purity)
        """
        unique_targets = targets.unique()
        purity_scores = []

        # Maximum entropy (uniform distribution)
        max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))

        for target_class in unique_targets:
            # Find all samples with true class = target_class
            mask = (targets == target_class)
            class_size = mask.sum().item()

            # Skip classes with too few samples
            if class_size < self.min_samples_per_class:
                continue

            # Get predictions for these samples
            class_predictions = predictions[mask]

            # Compute prediction distribution (frequency of each class)
            pred_counts = torch.bincount(
                class_predictions, 
                minlength=num_classes
            ).float()
            pred_probs = pred_counts / pred_counts.sum()

            # Calculate entropy of prediction distribution
            # Entropy = -Σ p(i) * log(p(i))
            # Lower entropy means more concentrated predictions (even if wrong)
            pred_probs = pred_probs[pred_probs > 0]  # Remove zero probabilities
            entropy = -(pred_probs * torch.log(pred_probs)).sum()

            # Normalize entropy to [0, 1]
            normalized_entropy = entropy / max_entropy

            # Purity = 1 - normalized_entropy
            purity = 1.0 - normalized_entropy
            purity_scores.append(purity)

        if len(purity_scores) == 0:
            # All classes have too few samples, return 0 reward
            return torch.tensor(0.0, device=targets.device)

        # Return average purity
        return torch.stack(purity_scores).mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss with purity reward.

        Args:
            logits: Model output (N, num_classes)
            targets: True labels (N,)

        Returns:
            Total loss = classification_loss - purity_reward
        """
        # 1. Compute base classification loss
        classification_loss = self.base_criterion(logits, targets)

        # 2. Compute cluster purity reward
        if self.lambda_purity > 0:
            num_classes = logits.size(1)
            predictions = logits.argmax(dim=1)

            # Compute purity (0-1 range, 1 = perfect purity)
            purity = self.compute_cluster_purity(predictions, targets, num_classes)

            # Purity reward: subtract from loss (higher purity = lower loss)
            purity_reward = self.lambda_purity * purity

            total_loss = classification_loss - purity_reward
        else:
            total_loss = classification_loss

        return total_loss


class PredictionDiversityLoss(nn.Module):
    """
    Prediction diversity loss: prevents model from predicting only one class.
    
    Core idea:
    1. Purity reward encourages "consistent errors" (same true class → consistent prediction)
    2. Diversity penalty punishes "over-concentration" (all predictions → few classes)
    3. Balance: intra-class consistency + inter-class discrimination
    
    Implementation:
    - Compute predicted class distribution P_pred = [p1, p2, ..., pK]
    - Compute true class distribution P_true = [q1, q2, ..., qK]
    - Penalize distribution deviation using KL divergence: KL(P_true || P_pred)
    
    Goal:
    - If true data has 3 classes, predictions should also have similar proportions
    - Prevent "predict all as majority class" from achieving high but meaningless accuracy
    
    Args:
        base_criterion: Base loss function
        lambda_diversity: Diversity loss weight (recommended 0.5-2.0)
        smoothing: Smoothing parameter to avoid zero probabilities
    
    Example:
        >>> criterion = PredictionDiversityLoss(
        ...     base_criterion=base_loss,
        ...     lambda_diversity=1.0
        ... )
        >>> loss = criterion(logits, targets)
    """
    
    def __init__(
        self,
        base_criterion: nn.Module,
        lambda_diversity: float = 1.0,
        smoothing: float = 1e-6
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.lambda_diversity = lambda_diversity
        self.smoothing = smoothing
        
        if lambda_diversity < 0:
            raise ValueError(f"lambda_diversity must be non-negative, got {lambda_diversity}")
    
    def compute_distribution_divergence(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """
        Compute KL divergence between predicted and true class distributions.
        
        KL(P_true || P_pred) = Σ P_true(i) * log(P_true(i) / P_pred(i))
        
        Args:
            predictions: Predicted labels (N,)
            targets: True labels (N,)
            num_classes: Total number of classes
        
        Returns:
            KL divergence (scalar)
        """
        # Compute true class distribution
        true_counts = torch.bincount(targets, minlength=num_classes).float()
        true_dist = true_counts / true_counts.sum()
        true_dist = true_dist + self.smoothing  # Smooth to avoid zeros
        true_dist = true_dist / true_dist.sum()  # Re-normalize
        
        # Compute predicted class distribution
        pred_counts = torch.bincount(predictions, minlength=num_classes).float()
        pred_dist = pred_counts / pred_counts.sum()
        pred_dist = pred_dist + self.smoothing  # Smooth to avoid zeros
        pred_dist = pred_dist / pred_dist.sum()  # Re-normalize
        
        # Compute KL divergence
        kl_div = (true_dist * torch.log(true_dist / pred_dist)).sum()
        
        return kl_div
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss with diversity penalty.
        
        Args:
            logits: Model output (N, num_classes)
            targets: True labels (N,)
        
        Returns:
            Total loss = base_loss + diversity_penalty
        """
        # 1. Compute base loss
        base_loss = self.base_criterion(logits, targets)
        
        # 2. Compute prediction diversity penalty
        if self.lambda_diversity > 0:
            num_classes = logits.size(1)
            predictions = logits.argmax(dim=1)
            
            # Compute KL divergence between predicted and true distributions
            diversity_penalty = self.compute_distribution_divergence(
                predictions, targets, num_classes
            )
            
            total_loss = base_loss + self.lambda_diversity * diversity_penalty
        else:
            total_loss = base_loss
        
        return total_loss


def create_loss_function_with_purity(
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    use_similarity_loss: bool = True,
    similarity_temperature: float = 0.1,
    lambda_purity: float = 0.1,
    min_samples_per_class: int = 2
) -> nn.Module:
    """
    Create loss function with cluster purity reward.
    
    Args:
        num_classes: Number of classes
        class_weights: Class weights
        use_similarity_loss: Whether to use similarity-aware loss
        similarity_temperature: Similarity temperature parameter
        lambda_purity: Purity reward weight (recommended 0.05-0.2)
        min_samples_per_class: Minimum samples threshold
    
    Returns:
        Loss function module
    
    Example:
        >>> criterion = create_loss_function_with_purity(
        ...     num_classes=9,
        ...     class_weights=weights,
        ...     use_similarity_loss=True,
        ...     lambda_purity=0.1  # Purity reward weight (medium encouragement)
        ... )
        >>> loss = criterion(logits, targets)
    """
    # 1. Create base classification loss
    base_criterion = create_loss_function(
        num_classes=num_classes,
        class_weights=class_weights,
        use_similarity_loss=use_similarity_loss,
        similarity_temperature=similarity_temperature
    )
    
    # 2. Wrap with purity reward if enabled
    if lambda_purity > 0:
        return ClusterPurityRewardLoss(
            base_criterion=base_criterion,
            lambda_purity=lambda_purity,
            min_samples_per_class=min_samples_per_class
        )
    else:
        return base_criterion


def create_balanced_loss_function(
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    use_similarity_loss: bool = True,
    similarity_temperature: float = 0.1,
    lambda_purity: float = 0.1,
    lambda_diversity: float = 1.0,
    min_samples_per_class: int = 2
) -> nn.Module:
    """
    Create balanced loss function: purity reward + diversity penalty.
    
    This loss function addresses two complementary goals:
    1. Purity reward: Encourages consistent predictions within same true class (avoid random errors)
    2. Diversity penalty: Prevents predicting all as one class (avoid majority class bias)
    
    Balanced effect:
    - Within same true class: predictions should be consistent ✓
    - Between different true classes: predictions should be discriminative ✓
    - Overall prediction distribution: should match true distribution ✓
    
    Loss hierarchy:
        Base classification loss (CrossEntropy/Similarity)
            ↓
        - λ_purity × purity (intra-class consistency reward)
            ↓
        + λ_diversity × KL(P_true||P_pred) (inter-class diversity penalty)
    
    Args:
        num_classes: Number of classes
        class_weights: Class weights for handling imbalance
        use_similarity_loss: Whether to use LCZ similarity-aware loss
        similarity_temperature: Similarity temperature (0.05-0.3)
        lambda_purity: Purity reward weight (recommended 0.05-0.2)
        lambda_diversity: Diversity penalty weight (recommended 0.5-2.0)
        min_samples_per_class: Minimum samples for purity calculation
    
    Returns:
        Loss function module
    
    Hyperparameter tuning guide:
        λ_purity=0.1, λ_diversity=0.5: Light encouragement
        λ_purity=0.1, λ_diversity=1.0: Medium (recommended) ⭐
        λ_purity=0.1, λ_diversity=2.0: Strong diversity requirement
    
    Example:
        >>> criterion = create_balanced_loss_function(
        ...     num_classes=9,
        ...     class_weights=weights,
        ...     lambda_purity=0.1,      # Purity reward
        ...     lambda_diversity=1.0     # Diversity penalty
        ... )
        >>> loss = criterion(logits, targets)
    """
    # 1. Create loss with purity reward
    loss_with_purity = create_loss_function_with_purity(
        num_classes=num_classes,
        class_weights=class_weights,
        use_similarity_loss=use_similarity_loss,
        similarity_temperature=similarity_temperature,
        lambda_purity=lambda_purity,
        min_samples_per_class=min_samples_per_class
    )
    
    # 2. Wrap with diversity penalty if enabled
    if lambda_diversity > 0:
        return PredictionDiversityLoss(
            base_criterion=loss_with_purity,
            lambda_diversity=lambda_diversity
        )
    else:
        return loss_with_purity

