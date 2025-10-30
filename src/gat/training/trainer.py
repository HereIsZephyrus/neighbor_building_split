"""Trainer class for GAT model.

Following pytorch-GAT training script structure.
"""

from typing import List, Optional, Dict
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

from torch.utils.tensorboard import SummaryWriter

from ..models.gat import GAT
from ..data.graph_batch_sampler import create_neighbor_loader, should_use_neighbor_sampling
from .config import GATConfig
from .smooth_loss import edge_smoothness_loss
from .train_utils import (
    save_checkpoint,
    load_checkpoint,
    log_metrics_to_tensorboard,
    log_model_info,
    set_random_seed,
    format_metrics,
    get_lr,
    EarlyStopping,
    save_training_history
)
from ..utils.logger import get_logger

logger = get_logger()


class Trainer:
    """
    Trainer for GAT model on building graphs.

    Handles:
    - Training loop with mini-batch sampling
    - Validation and early stopping
    - Model checkpointing
    - TensorBoard logging
    """

    def __init__(
        self,
        model: GAT,
        config: GATConfig,
        train_data_list: List[Data],
        val_data_list: Optional[List[Data]] = None
    ):
        """
        Initialize trainer.

        Args:
            model: GAT model
            config: Training configuration
            train_data_list: List of training graphs
            val_data_list: Optional list of validation graphs
        """
        self.model = model
        self.config = config
        self.train_data_list = train_data_list
        self.val_data_list = val_data_list

        # Move model to device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=config.patience // 2
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()  # Node classification

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode='max'  # Maximize accuracy
        )

        # TensorBoard writer
        self.writer = None
        if config.enable_tensorboard:
            self.writer = SummaryWriter(log_dir=str(config.log_dir))
            logger.info(f"TensorBoard logging to {config.log_dir}")

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_cls_loss': [],      # Classification loss component
            'train_smooth_loss': [],   # Smoothness loss component
            'val_loss': [],
            'val_acc': [],
            'val_cls_loss': [],        # Classification loss component
            'val_smooth_loss': [],     # Smoothness loss component
            'lr': []
        }

        # Log model info
        log_model_info(self.model)

    def train_epoch(self, _epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of average training metrics
        """
        self.model.train()

        total_loss = 0
        total_cls_loss = 0
        total_smooth_loss = 0
        total_correct = 0
        total_samples = 0

        # Train on each graph
        for data in self.train_data_list:
            data = data.to(self.device)

            # Check if we need neighbor sampling
            if should_use_neighbor_sampling(data, threshold=self.config.node_threshold):
                # Use mini-batch training with NeighborLoader
                loader = create_neighbor_loader(
                    data,
                    num_neighbors=self.config.num_neighbors,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=self.config.num_workers
                )

                batch_count = 0
                for batch_count, batch in enumerate(loader):
                    batch = batch.to(self.device)

                    # Forward pass
                    node_logits = self.model(batch.x, batch.edge_index)

                    # Classification loss on sampled nodes
                    # NeighborLoader provides batch.batch_size for the number of seed nodes
                    loss_cls = self.criterion(
                        node_logits[:batch.batch_size], 
                        batch.y[:batch.batch_size]
                    )

                    # Spatial smoothness loss on all nodes in batch
                    # Extract edge attributes if available
                    edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
                    loss_smooth = edge_smoothness_loss(
                        node_logits,
                        batch.edge_index,
                        edge_attr,
                        temperature=self.config.smooth_temperature
                    )

                    # Combined loss
                    loss = loss_cls + self.config.lambda_smooth * loss_smooth

                    # Backward pass
                    loss.backward()

                    # Gradient accumulation
                    if (batch_count + 1) % self.config.gradient_accumulation_steps == 0:
                        # Gradient clipping to prevent gradient explosion
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # Metrics
                    with torch.no_grad():
                        # Check for NaN in loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.error(f"NaN/Inf detected in loss! cls_loss={loss_cls.item()}, smooth_loss={loss_smooth.item()}")
                            logger.error("This may be caused by: 1) learning rate too high, 2) gradient explosion, 3) numerical instability")
                            raise ValueError("Training stopped due to NaN/Inf in loss")

                        pred = node_logits[:batch.batch_size].argmax(dim=1)
                        correct = (pred == batch.y[:batch.batch_size]).sum().item()
                        total_correct += correct
                        total_samples += batch.batch_size
                        total_loss += loss.item() * batch.batch_size
                        total_cls_loss += loss_cls.item() * batch.batch_size
                        total_smooth_loss += loss_smooth.item() * batch.batch_size

                # Final optimizer step if needed
                if batch_count % self.config.gradient_accumulation_steps != 0:
                    # Gradient clipping to prevent gradient explosion
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            else:
                # Full-graph training
                self.optimizer.zero_grad()

                # Forward pass
                node_logits = self.model(data.x, data.edge_index)

                # Classification loss
                loss_cls = self.criterion(node_logits, data.y)

                # Spatial smoothness loss
                edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
                loss_smooth = edge_smoothness_loss(
                    node_logits,
                    data.edge_index,
                    edge_attr,
                    temperature=self.config.smooth_temperature
                )

                # Combined loss
                loss = loss_cls + self.config.lambda_smooth * loss_smooth

                # Backward pass
                loss.backward()
                # Gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    # Check for NaN in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error(f"NaN/Inf detected in loss! cls_loss={loss_cls.item()}, smooth_loss={loss_smooth.item()}")
                        logger.error("This may be caused by: 1) learning rate too high, 2) gradient explosion, 3) numerical instability")
                        raise ValueError("Training stopped due to NaN/Inf in loss")

                    pred = node_logits.argmax(dim=1)
                    correct = (pred == data.y).sum().item()

                    total_correct += correct
                    total_samples += data.num_nodes
                    total_loss += loss.item() * data.num_nodes
                    total_cls_loss += loss_cls.item() * data.num_nodes
                    total_smooth_loss += loss_smooth.item() * data.num_nodes

        # Average metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        avg_cls_loss = total_cls_loss / total_samples if total_samples > 0 else 0
        avg_smooth_loss = total_smooth_loss / total_samples if total_samples > 0 else 0

        metrics = {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'cls_loss': avg_cls_loss,
            'smooth_loss': avg_smooth_loss
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_data_list is None or len(self.val_data_list) == 0:
            return {}

        self.model.eval()

        total_loss = 0
        total_cls_loss = 0
        total_smooth_loss = 0
        total_correct = 0
        total_samples = 0

        for data in self.val_data_list:
            data = data.to(self.device)

            # Forward pass
            node_logits = self.model(data.x, data.edge_index)

            # Classification loss
            loss_cls = self.criterion(node_logits, data.y)

            # Spatial smoothness loss
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
            loss_smooth = edge_smoothness_loss(
                node_logits,
                data.edge_index,
                edge_attr,
                temperature=self.config.smooth_temperature
            )

            # Combined loss
            loss = loss_cls + self.config.lambda_smooth * loss_smooth

            # Metrics
            pred = node_logits.argmax(dim=1)
            correct = (pred == data.y).sum().item()

            total_correct += correct
            total_samples += data.num_nodes
            total_loss += loss.item() * data.num_nodes
            total_cls_loss += loss_cls.item() * data.num_nodes
            total_smooth_loss += loss_smooth.item() * data.num_nodes

        # Average metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        avg_cls_loss = total_cls_loss / total_samples if total_samples > 0 else 0
        avg_smooth_loss = total_smooth_loss / total_samples if total_samples > 0 else 0

        metrics = {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'cls_loss': avg_cls_loss,
            'smooth_loss': avg_smooth_loss
        }

        return metrics

    def train(self) -> Dict[str, list]:
        """
        Main training loop.

        Returns:
            Training history dictionary
        """
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info("Train graphs: %d", len(self.train_data_list))
        if self.val_data_list:
            logger.info("Val graphs: %d", len(self.val_data_list))
        logger.info("Device: %s", self.device)
        logger.info("Epochs: %d", self.config.epochs)
        logger.info("=" * 80)

        # Set random seed
        set_random_seed(self.config.seed)

        best_val_acc = 0.0

        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            if val_metrics:
                self.scheduler.step(val_metrics['accuracy'])

            # Log metrics
            current_lr = get_lr(self.optimizer)

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_cls_loss'].append(train_metrics['cls_loss'])
            self.history['train_smooth_loss'].append(train_metrics['smooth_loss'])
            self.history['lr'].append(current_lr)

            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['val_cls_loss'].append(val_metrics['cls_loss'])
                self.history['val_smooth_loss'].append(val_metrics['smooth_loss'])

            # TensorBoard logging
            if self.writer is not None:
                log_metrics_to_tensorboard(self.writer, train_metrics, epoch, 'train')
                if val_metrics:
                    log_metrics_to_tensorboard(self.writer, val_metrics, epoch, 'val')
                self.writer.add_scalar('lr', current_lr, epoch)

            # Print progress
            if epoch % self.config.log_interval == 0 or epoch == 1:
                log_str = f"Epoch {epoch:3d}/{self.config.epochs} | " \
                         f"Train: {format_metrics(train_metrics, precision=4)}"
                if val_metrics:
                    log_str += f" | Val: {format_metrics(val_metrics, precision=4)}"
                log_str += f" | LR: {current_lr:.6f}"
                logger.info(log_str)

            # Checkpointing
            is_best = False
            if val_metrics and val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                is_best = True

            if epoch % self.config.checkpoint_interval == 0 or is_best:
                checkpoint_path = Path(self.config.checkpoint_dir) / f'{self.config.model_identifier}_checkpoint_epoch_{epoch}.pth'
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_metrics if val_metrics else train_metrics,
                    self.config,
                    checkpoint_path,
                    is_best=is_best
                )

            # Early stopping
            if val_metrics:
                if self.early_stopping(val_metrics['accuracy']):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Save final checkpoint
        final_model_dir = Path(self.config.output_root_dir) / 'models'
        final_model_dir.mkdir(parents=True, exist_ok=True)
        final_checkpoint_path = final_model_dir / f'{self.config.model_identifier}_final_model.pth'
        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            epoch,
            val_metrics if val_metrics else train_metrics,
            self.config,
            final_checkpoint_path
        )

        # Save training history
        history_path = Path(self.config.checkpoint_dir) / f'{self.config.model_identifier}_training_history.json'
        save_training_history(self.history, history_path)

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()

        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info("Best validation accuracy: %.4f", best_val_acc)
        logger.info("=" * 80)

        return self.history

    def resume_from_checkpoint(self, checkpoint_path: Path) -> int:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Epoch to resume from
        """
        checkpoint = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            device=str(self.device)
        )

        epoch = checkpoint.get('epoch', 0)
        logger.info("Resumed from epoch %d", epoch)

        return epoch

