import sys
from pathlib import Path
from .models import GAT
from .training import Trainer
from .utils import get_logger

logger = get_logger()

def train_final_model(config, dataset, args):
    """Train final model on all data without validation split.

    Args:
        config: Training configuration
        dataset: BuildingGraphDataset
        args: Command line arguments
    """
    logger.info("=" * 80)
    logger.info("Final Model Training (Full Data, No Validation)")
    logger.info("=" * 80)

    # Load all data for training
    data_list = [dataset.get(i) for i in range(len(dataset))]
    logger.info("Training on all %d districts", len(data_list))

    # Initialize model
    logger.info("Initializing model...")
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

    logger.info("Model:\n%s", model)

    # Initialize trainer (no validation data)
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        train_data_list=data_list,
        val_data_list=None  # No validation in final training
    )

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        trainer.resume_from_checkpoint(Path(args.resume))

    # Train
    try:
        history = trainer.train()

        logger.info("Training completed successfully!")
        logger.info("Final model saved to: %s", Path(config.output_root_dir) / 'models')

        return history

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)
