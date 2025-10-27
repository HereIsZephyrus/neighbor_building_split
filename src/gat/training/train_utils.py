"""Training utilities and helper functions."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
import json

from ..utils.metrics import compute_metrics
from ..utils.logger import get_logger

logger = get_logger()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    config: Any,
    filepath: Path,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Current metrics dict
        config: Training configuration
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config.to_dict() if hasattr(config, 'to_dict') else {},
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")
    
    if is_best:
        best_path = filepath.parent / 'best_model.pth'
        torch.save(checkpoint, best_path)
        logger.info(f"Best model saved to {best_path}")


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        
    Returns:
        Dictionary with checkpoint information
    """
    logger.info(f"Loading checkpoint from {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint


def compute_loss_and_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute loss and metrics.
    
    Args:
        logits: Model output logits (N, C)
        labels: Ground truth labels (N,)
        criterion: Loss function
        mask: Optional mask for which nodes to evaluate
        
    Returns:
        Dictionary with loss and metrics
    """
    # Apply mask if provided
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    
    # Compute loss
    loss = criterion(logits, labels)
    
    # Compute metrics
    metrics = compute_metrics(logits, labels)
    metrics['loss'] = loss.item()
    
    return metrics


def log_metrics_to_tensorboard(
    writer,
    metrics: Dict[str, float],
    epoch: int,
    prefix: str = 'train'
) -> None:
    """
    Log metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        metrics: Dictionary of metrics to log
        epoch: Current epoch
        prefix: Prefix for metric names (e.g., 'train', 'val')
    """
    for key, value in metrics.items():
        writer.add_scalar(f'{prefix}/{key}', value, epoch)


def log_model_info(model: nn.Module) -> None:
    """
    Log model information.
    
    Args:
        model: Model to inspect
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary as string.
    
    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    return ', '.join([f'{k}={v:.{precision}f}' for k, v in metrics.items()])


def save_training_history(
    history: Dict[str, list],
    filepath: Path
) -> None:
    """
    Save training history to JSON file.
    
    Args:
        history: Dictionary with lists of metrics over epochs
        filepath: Path to save JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training history saved to {filepath}")


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class EarlyStopping:
    """
    Early stopping handler.
    
    Stops training when validation metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 100,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' (higher is better) or 'min' (lower is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        assert mode in ['max', 'min'], "mode must be 'max' or 'min'"
        
        logger.info(f"Early stopping: patience={patience}, mode={mode}")
    
    def __call__(self, score: float) -> bool:
        """
        Update early stopping state.
        
        Args:
            score: Current validation score
            
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                self.early_stop = True
                return True
            return False
    
    def is_better(self, score: float) -> bool:
        """Check if score is better than current best."""
        if self.best_score is None:
            return True
        
        if self.mode == 'max':
            return score > self.best_score
        else:
            return score < self.best_score

