import sys
import copy
from pathlib import Path
from datetime import datetime
import yaml
from torch.utils.tensorboard import SummaryWriter
from .models import GAT
from .training import Trainer
from .training.tensorborad_utils import log_district_visualizations_to_tensorboard
from .utils import get_logger
from .data import kfold_split

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None

logger = get_logger(__name__)

def train_cross_validation_mpi(config, dataset, args):
    """Train using K-fold cross-validation with MPI parallelization.

    Args:
        config: Training configuration
        dataset: BuildingGraphDataset
        args: Command line arguments
    """
    if not MPI_AVAILABLE:
        logger.error("MPI is not available. Install mpi4py to use cross-validation mode.")
        logger.error("Falling back to sequential cross-validation...")
        return train_cross_validation_sequential(config, dataset, args)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_folds = config.k_fold

    if rank == 0:
        logger.info("=" * 80)
        logger.info("K-Fold Cross-Validation with MPI")
        logger.info("Number of folds: %d", n_folds)
        logger.info("MPI processes: %d", size)
        logger.info("=" * 80)

        if size > n_folds:
            logger.warning("More MPI processes (%d) than folds (%d). Extra processes will be idle.", size, n_folds)

    # Prepare data
    data_list = [dataset.get(i) for i in range(len(dataset))]

    # Create TensorBoard writers for each fold (to avoid write conflicts)
    # Each rank will create a writer when processing its assigned fold
    shared_tensorboard_base_dir = None
    if config.enable_tensorboard:
        # Rank 0 creates the base directory and broadcasts the path
        if rank == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            shared_tensorboard_base_dir = Path(config.log_dir) / f"{config.model_identifier}_cv_{timestamp}"
            shared_tensorboard_base_dir.mkdir(parents=True, exist_ok=True)
            shared_tensorboard_base_dir = str(shared_tensorboard_base_dir)

        # Broadcast tensorboard base directory to all ranks
        shared_tensorboard_base_dir = comm.bcast(shared_tensorboard_base_dir, root=0)
        logger.info("Rank %d: TensorBoard base directory: %s", rank, shared_tensorboard_base_dir)

    # Each rank processes specific folds
    fold_results = []

    for fold_idx, (train_data, val_data) in enumerate(kfold_split(data_list, n_splits=n_folds, random_seed=config.seed), 1):
        # Distribute folds across MPI processes
        if (fold_idx - 1) % size != rank:
            continue  # This fold is not assigned to this rank

        logger.info("=" * 80)
        logger.info("Rank %d: Training Fold %d/%d", rank, fold_idx, n_folds)
        logger.info("=" * 80)
        logger.info("Train: %d districts, Val: %d districts", len(train_data), len(val_data))

        # Create fold-specific TensorBoard writer (避免写入冲突)
        fold_writer = None
        if shared_tensorboard_base_dir is not None:
            fold_log_dir = Path(shared_tensorboard_base_dir) / f"fold{fold_idx}"
            fold_log_dir.mkdir(parents=True, exist_ok=True)
            fold_writer = SummaryWriter(log_dir=str(fold_log_dir))
            logger.info("Rank %d: Fold %d TensorBoard logging to %s", rank, fold_idx, fold_log_dir)

        # Initialize model for this fold
        logger.info("Initializing model for fold %d...", fold_idx)
        model = GAT(
            in_features=dataset.num_features,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            negative_slope=config.negative_slope,
            add_self_loops=config.add_self_loops
        )

        # Create fold-specific config
        fold_config = copy.deepcopy(config)
        fold_config.model_identifier = f"{config.model_identifier}_fold{fold_idx}"
        fold_config.checkpoint_dir = f"{config.checkpoint_dir}/fold{fold_idx}"
        fold_config.output_dir = f"{config.output_dir}/fold{fold_idx}"
        fold_config.enable_tensorboard = False  # Use shared writer

        # Create directories (MPI-safe: use try-except to handle race condition)
        try:
            Path(fold_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass  # Another process already created it

        try:
            Path(fold_config.output_dir).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass  # Another process already created it

        # Initialize trainer
        logger.info("Rank %d: Initializing trainer for fold %d...", rank, fold_idx)
        trainer = Trainer(
            model=model,
            config=fold_config,
            train_data_list=train_data,
            val_data_list=val_data
        )

        # Train this fold
        try:
            history = trainer.train()

            best_val_acc = max(history.get('val_acc', [0])) if history.get('val_acc') else 0
            logger.info("Rank %d: Fold %d completed! Best validation accuracy: %.4f", rank, fold_idx, best_val_acc)

            fold_results.append({
                'fold': fold_idx,
                'best_val_acc': best_val_acc,
                'history': history
            })

            # Generate visualizations for this fold (each rank visualizes its own fold)
            if fold_writer is not None:
                logger.info("Rank %d: Generating visualizations for fold %d...", rank, fold_idx)
                logger.info("Rank %d: Fold writer directory: %s", rank, fold_writer.log_dir)

                # Visualize training data
                if train_data:
                    try:
                        max_visualize = getattr(fold_config, 'max_visualize_districts', 9)
                        logger.info("Rank %d: Fold %d - visualizing %d training districts", 
                                   rank, fold_idx, min(max_visualize, len(train_data)))
                        log_district_visualizations_to_tensorboard(
                            writer=fold_writer,
                            model=model,
                            data_list=train_data[:max_visualize],
                            building_path=Path(fold_config.building_path),
                            epoch=fold_idx,
                            tag=f'train',
                            max_districts=max_visualize,
                            device=fold_config.device,
                            district_path=Path(fold_config.district_path) if hasattr(fold_config, 'district_path') else None,
                            adjacency_dir=Path(fold_config.adjacency_dir) if hasattr(fold_config, 'adjacency_dir') else None,
                            enable_spectral_clustering=True
                        )
                        logger.info("Rank %d: Completed training visualizations for fold %d", rank, fold_idx)
                    except Exception as e:
                        logger.error("Rank %d: Failed to generate training visualizations for fold %d: %s", 
                                   rank, fold_idx, e, exc_info=True)

                # Visualize validation data
                if val_data:
                    try:
                        max_visualize = getattr(fold_config, 'max_visualize_districts', 9)
                        logger.info("Rank %d: Fold %d - visualizing %d validation districts", 
                                   rank, fold_idx, min(max_visualize, len(val_data)))
                        log_district_visualizations_to_tensorboard(
                            writer=fold_writer,
                            model=model,
                            data_list=val_data[:max_visualize],
                            building_path=Path(fold_config.building_path),
                            epoch=fold_idx,
                            tag=f'val',
                            max_districts=max_visualize,
                            device=fold_config.device,
                            district_path=Path(fold_config.district_path) if hasattr(fold_config, 'district_path') else None,
                            adjacency_dir=Path(fold_config.adjacency_dir) if hasattr(fold_config, 'adjacency_dir') else None,
                            enable_spectral_clustering=True
                        )
                        logger.info("Rank %d: Completed validation visualizations for fold %d", rank, fold_idx)
                    except Exception as e:
                        logger.error("Rank %d: Failed to generate validation visualizations for fold %d: %s", 
                                   rank, fold_idx, e, exc_info=True)

            # Close fold-specific writer
            if fold_writer is not None:
                fold_writer.flush()
                logger.info("Rank %d: Flushed fold %d writer", rank, fold_idx)
                fold_writer.close()
                logger.info("Rank %d: Closed fold %d writer", rank, fold_idx)

        except Exception as exc:
            logger.error("Rank %d: Training failed for fold %d: %s", rank, fold_idx, exc, exc_info=True)
            fold_results.append({
                'fold': fold_idx,
                'best_val_acc': 0.0,
                'error': str(exc)
            })

    # Gather results from all ranks
    all_fold_results = comm.gather(fold_results, root=0)

    # Process and report results (only on rank 0)
    if rank == 0:
        # Flatten the list of lists
        all_results = []
        for results in all_fold_results:
            all_results.extend(results)

        # Sort by fold number
        all_results.sort(key=lambda x: x['fold'])

        # Create a summary writer for cross-validation statistics (only on rank 0)
        summary_writer = None
        if shared_tensorboard_base_dir is not None:
            summary_log_dir = Path(shared_tensorboard_base_dir) / "summary"
            summary_log_dir.mkdir(parents=True, exist_ok=True)
            summary_writer = SummaryWriter(log_dir=str(summary_log_dir))
            logger.info("Rank 0: Cross-validation summary TensorBoard logging to %s", summary_log_dir)
            logger.info("Rank 0: Processing %d fold results for summary", len(all_results))
        else:
            logger.warning("Rank 0: shared_tensorboard_base_dir is None, skipping summary writer creation")

        # Log fold results to TensorBoard
        if summary_writer is not None:
            logger.info("Rank 0: Writing fold results to summary...")
            for result in all_results:
                if 'error' in result:
                    continue

                fold_idx = result['fold']
                history = result['history']
                best_val_acc = result['best_val_acc']

                # Log training curves
                train_losses = history.get('train_loss', [])
                train_accs = history.get('train_acc', [])
                val_losses = history.get('val_loss', [])
                val_accs = history.get('val_acc', [])

                for epoch_idx, (train_loss, train_acc) in enumerate(zip(train_losses, train_accs), 1):
                    summary_writer.add_scalar(f'fold{fold_idx}/train_loss', train_loss, epoch_idx)
                    summary_writer.add_scalar(f'fold{fold_idx}/train_acc', train_acc, epoch_idx)

                val_interval = config.val_interval
                for val_idx, (val_loss, val_acc) in enumerate(zip(val_losses, val_accs)):
                    epoch_idx = (val_idx + 1) * val_interval
                    summary_writer.add_scalar(f'fold{fold_idx}/val_loss', val_loss, epoch_idx)
                    summary_writer.add_scalar(f'fold{fold_idx}/val_acc', val_acc, epoch_idx)

                summary_writer.add_scalar('cross_validation/best_val_acc_by_fold', best_val_acc, fold_idx)

        # Report summary
        logger.info("=" * 80)
        logger.info("%d-Fold Cross-Validation Results", n_folds)
        logger.info("=" * 80)

        valid_results = [r for r in all_results if 'error' not in r]
        if valid_results:
            for result in valid_results:
                logger.info("Fold %d: Best Val Acc = %.4f", result['fold'], result['best_val_acc'])

            avg_acc = sum(r['best_val_acc'] for r in valid_results) / len(valid_results)
            std_acc = (sum((r['best_val_acc'] - avg_acc) ** 2 for r in valid_results) / len(valid_results)) ** 0.5

            logger.info("-" * 80)
            logger.info("Average Validation Accuracy: %.4f ± %.4f", avg_acc, std_acc)
            logger.info("=" * 80)

            # Log summary to TensorBoard
            if summary_writer is not None:
                summary_writer.add_scalar('cross_validation/average_val_acc', avg_acc, 0)
                summary_writer.add_scalar('cross_validation/std_val_acc', std_acc, 0)

                text_summary = "**Cross-Validation Results**\n\n"
                text_summary += f"- Number of Folds: {n_folds}\n"
                text_summary += f"- Average Val Acc: {avg_acc:.4f}\n"
                text_summary += f"- Std Dev: ±{std_acc:.4f}\n\n"
                text_summary += "**Per-Fold Results:**\n\n"
                for result in valid_results:
                    text_summary += f"- Fold {result['fold']}: {result['best_val_acc']:.4f}\n"

                summary_writer.add_text('cross_validation/summary', text_summary, 0)

                # Ensure all data is flushed before closing
                summary_writer.flush()
                logger.info("Rank 0: Flushed summary writer")
                summary_writer.close()
                logger.info("Rank 0: Closed summary writer - TensorBoard logs saved to %s/summary", 
                           shared_tensorboard_base_dir)

            # Save CV summary
            cv_summary_path = Path(config.output_dir) / f'cv_summary_{config.model_identifier}.yaml'
            cv_summary = {
                'n_folds': n_folds,
                'fold_results': [
                    {'fold': r['fold'], 'best_val_acc': float(r['best_val_acc'])} 
                    for r in valid_results
                ],
                'average_val_acc': float(avg_acc),
                'std_val_acc': float(std_acc)
            }
            with open(cv_summary_path, 'w', encoding='utf-8') as f:
                yaml.dump(cv_summary, f, default_flow_style=False, allow_unicode=True)
            logger.info("Cross-validation summary saved to: %s", cv_summary_path)
        else:
            logger.error("All folds failed to train!")
            if summary_writer is not None:
                summary_writer.close()
            sys.exit(1)


def train_cross_validation_sequential(config, dataset, args):
    """Train using K-fold cross-validation sequentially (fallback without MPI).

    Args:
        config: Training configuration
        dataset: BuildingGraphDataset
        args: Command line arguments
    """
    logger.info("=" * 80)
    logger.info("K-Fold Cross-Validation (Sequential)")
    logger.info("=" * 80)

    # Prepare data for K-fold cross-validation
    data_list = [dataset.get(i) for i in range(len(dataset))]

    # Store results for all folds
    fold_results = []
    n_folds = config.k_fold

    # Create a shared TensorBoard writer for all folds
    if config.enable_tensorboard:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tensorboard_dir = Path(config.log_dir) / f"{config.model_identifier}_cv_{timestamp}"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        shared_writer = SummaryWriter(log_dir=str(tensorboard_dir))
        logger.info("TensorBoard logging to %s", tensorboard_dir)
    else:
        shared_writer = None

    # Perform K-fold cross-validation
    for fold_idx, (train_data, val_data) in enumerate(kfold_split(data_list, n_splits=n_folds, random_seed=config.seed), 1):
        logger.info("=" * 80)
        logger.info("Training Fold %d/%d", fold_idx, n_folds)
        logger.info("=" * 80)
        logger.info("Train: %d districts, Val: %d districts", len(train_data), len(val_data))

        # Initialize model for this fold
        logger.info("Initializing model for fold %d...", fold_idx)
        model = GAT(
            in_features=dataset.num_features,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            negative_slope=config.negative_slope,
            add_self_loops=config.add_self_loops
        )

        # Create a deep copy of config for this fold to avoid modifying the original
        fold_config = copy.deepcopy(config)
        fold_config.model_identifier = f"{config.model_identifier}_fold{fold_idx}"

        # Update paths - checkpoints stored per fold, but no separate tensorboard logs
        fold_config.checkpoint_dir = f"{config.checkpoint_dir}/fold{fold_idx}"
        fold_config.output_dir = f"{config.output_dir}/fold{fold_idx}"

        # Disable individual TensorBoard writers for each fold (we use shared writer)
        fold_config.enable_tensorboard = False

        # Create directories
        Path(fold_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(fold_config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize trainer for this fold
        logger.info("Initializing trainer for fold %d...", fold_idx)
        trainer = Trainer(
            model=model,
            config=fold_config,
            train_data_list=train_data,
            val_data_list=val_data
        )

        if args.resume and fold_idx == 1:
            logger.info("Resuming from checkpoint: %s", args.resume)
            trainer.resume_from_checkpoint(Path(args.resume))

        # Train this fold
        try:
            history = trainer.train()

            best_val_acc = max(history.get('val_acc', [0])) if history.get('val_acc') else 0
            logger.info("Fold %d completed! Best validation accuracy: %.4f", fold_idx, best_val_acc)

            fold_results.append({
                'fold': fold_idx,
                'best_val_acc': best_val_acc,
                'history': history
            })

            # Log fold results to shared TensorBoard
            if shared_writer is not None:
                # Log training curves for this fold
                train_losses = history.get('train_loss', [])
                train_accs = history.get('train_acc', [])
                val_losses = history.get('val_loss', [])
                val_accs = history.get('val_acc', [])

                # Log training metrics (every epoch)
                for epoch_idx, (train_loss, train_acc) in enumerate(zip(train_losses, train_accs), 1):
                    shared_writer.add_scalar(f'fold{fold_idx}/train_loss', train_loss, epoch_idx)
                    shared_writer.add_scalar(f'fold{fold_idx}/train_acc', train_acc, epoch_idx)

                # Log validation metrics (only on validation epochs, based on val_interval)
                val_interval = fold_config.val_interval
                for val_idx, (val_loss, val_acc) in enumerate(zip(val_losses, val_accs)):
                    # Calculate the actual epoch number for this validation point
                    # Validation happens at epochs: val_interval, 2*val_interval, ..., and final epoch
                    epoch_idx = (val_idx + 1) * val_interval
                    shared_writer.add_scalar(f'fold{fold_idx}/val_loss', val_loss, epoch_idx)
                    shared_writer.add_scalar(f'fold{fold_idx}/val_acc', val_acc, epoch_idx)

                # Log best validation accuracy as a scalar
                shared_writer.add_scalar('cross_validation/best_val_acc_by_fold', best_val_acc, fold_idx)

                # Generate visualizations for this fold
                logger.info("Generating visualizations for fold %d...", fold_idx)

                # Visualize training data
                if train_data:
                    try:
                        max_visualize = getattr(fold_config, 'max_visualize_districts', 9)
                        logger.info("Fold %d - visualizing %d training districts", 
                                   fold_idx, min(max_visualize, len(train_data)))
                        log_district_visualizations_to_tensorboard(
                            writer=shared_writer,
                            model=model,
                            data_list=train_data[:max_visualize],
                            building_path=Path(fold_config.building_path),
                            epoch=fold_idx,  # Use fold index as "epoch" for organization
                            tag=f'fold{fold_idx}_train',
                            max_districts=max_visualize,
                            device=fold_config.device,
                            district_path=Path(fold_config.district_path) if hasattr(fold_config, 'district_path') else None,
                            adjacency_dir=Path(fold_config.adjacency_dir) if hasattr(fold_config, 'adjacency_dir') else None,
                            enable_spectral_clustering=True
                        )
                        logger.info("Completed training visualizations for fold %d", fold_idx)
                    except Exception as e:
                        logger.error("Failed to generate training visualizations for fold %d: %s", fold_idx, e, exc_info=True)

                # Visualize validation data
                if val_data:
                    try:
                        max_visualize = getattr(fold_config, 'max_visualize_districts', 9)
                        logger.info("Fold %d - visualizing %d validation districts", 
                                   fold_idx, min(max_visualize, len(val_data)))
                        log_district_visualizations_to_tensorboard(
                            writer=shared_writer,
                            model=model,
                            data_list=val_data[:max_visualize],
                            building_path=Path(fold_config.building_path),
                            epoch=fold_idx,  # Use fold index as "epoch" for organization
                            tag=f'fold{fold_idx}_val',
                            max_districts=max_visualize,
                            device=fold_config.device,
                            district_path=Path(fold_config.district_path) if hasattr(fold_config, 'district_path') else None,
                            adjacency_dir=Path(fold_config.adjacency_dir) if hasattr(fold_config, 'adjacency_dir') else None,
                            enable_spectral_clustering=True
                        )
                        logger.info("Completed validation visualizations for fold %d", fold_idx)
                    except Exception as e:
                        logger.error("Failed to generate validation visualizations for fold %d: %s", fold_idx, e, exc_info=True)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            if shared_writer is not None:
                shared_writer.close()
            sys.exit(0)
        except Exception as exc:
            logger.error("Training failed for fold %d: %s", fold_idx, exc, exc_info=True)
            fold_results.append({
                'fold': fold_idx,
                'best_val_acc': 0.0,
                'error': str(exc)
            })

    # Report cross-validation results
    logger.info("=" * 80)
    logger.info("%d-Fold Cross-Validation Results", n_folds)
    logger.info("=" * 80)

    valid_results = [r for r in fold_results if 'error' not in r]
    if valid_results:
        for result in valid_results:
            logger.info("Fold %d: Best Val Acc = %.4f", result['fold'], result['best_val_acc'])

        avg_acc = sum(r['best_val_acc'] for r in valid_results) / len(valid_results)
        std_acc = (sum((r['best_val_acc'] - avg_acc) ** 2 for r in valid_results) / len(valid_results)) ** 0.5

        logger.info("-" * 80)
        logger.info("Average Validation Accuracy: %.4f ± %.4f", avg_acc, std_acc)
        logger.info("=" * 80)

        # Log summary statistics to TensorBoard
        if shared_writer is not None:
            shared_writer.add_scalar('cross_validation/average_val_acc', avg_acc, 0)
            shared_writer.add_scalar('cross_validation/std_val_acc', std_acc, 0)

            # Add text summary
            text_summary = "**Cross-Validation Results**\n\n"
            text_summary += f"- Number of Folds: {n_folds}\n"
            text_summary += f"- Average Val Acc: {avg_acc:.4f}\n"
            text_summary += f"- Std Dev: ±{std_acc:.4f}\n\n"
            text_summary += "**Per-Fold Results:**\n\n"
            for result in valid_results:
                text_summary += f"- Fold {result['fold']}: {result['best_val_acc']:.4f}\n"

            shared_writer.add_text('cross_validation/summary', text_summary, 0)
            shared_writer.close()
            logger.info("TensorBoard logs saved")

        # Save cross-validation summary
        cv_summary_path = Path(config.output_root_dir) / f'cv_summary_{config.model_identifier}.yaml'
        cv_summary = {
            'n_folds': n_folds,
            'fold_results': [
                {'fold': r['fold'], 'best_val_acc': float(r['best_val_acc'])} 
                for r in valid_results
            ],
            'average_val_acc': float(avg_acc),
            'std_val_acc': float(std_acc)
        }
        with open(cv_summary_path, 'w', encoding='utf-8') as f:
            yaml.dump(cv_summary, f, default_flow_style=False, allow_unicode=True)
        logger.info("Cross-validation summary saved to: %s", cv_summary_path)
    else:
        logger.error("All folds failed to train!")
        if shared_writer is not None:
            shared_writer.close()
        sys.exit(1)
