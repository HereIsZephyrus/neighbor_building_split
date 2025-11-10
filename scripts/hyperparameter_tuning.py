#!/usr/bin/env python3
"""
hyperparameter tuning with MPI - grid search

Usage:
    mpirun -np 64 python -m scripts.hyperparameter_tuning \
        --config src/gat/training_config.yaml \
        --adjacency-dir /path/to/adjacency \
        --building-path /path/to/buildings.shp
        --district-path /path/to/districts.shp
        --output-dir experiments/tuning_results
"""

import sys
import copy
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import yaml
import numpy as np
from sklearn.model_selection import ParameterGrid
from torch.utils.tensorboard import SummaryWriter

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None

from src.gat import GAT, BuildingGraphDataset, GATConfig, Trainer, kfold_split, get_logger, DistrictDataset, BuildingDataset

logger = get_logger()

# hyperparameter search space
PARAM_GRID = {
    'hidden_dim': [32, 64, 128],
    'num_layers': [2, 3, 4],
    'num_heads': [4, 8],
    'dropout': [0.4, 0.6],
    'lr': [1e-3, 5e-3, 1e-2],
    'weight_decay': [5e-4, 1e-3],
    'lambda_smooth': [0.1, 0.5, 1.0]
}


def generate_param_combinations() -> List[Dict[str, Any]]:
    """
    generate all hyperparameter combinations

    Returns:
        hyperparameter combinations list
    """
    param_combinations = list(ParameterGrid(PARAM_GRID))
    logger.info(f"Generated {len(param_combinations)} parameter combinations")
    return param_combinations


def create_config_with_params(
    base_config: GATConfig,
    params: Dict[str, Any],
    run_id: int
) -> GATConfig:
    """
    create new configuration with hyperparameters

    Args:
        base_config: base configuration
        params: hyperparameter dictionary
        run_id: run ID

    Returns:
        updated configuration
    """
    config = copy.deepcopy(base_config)

    # update model parameters
    for key, value in params.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # update identifier
    param_str = '_'.join([f"{k[:2]}{v}" for k, v in params.items()])
    config.model_identifier = f"tune_run{run_id}_{param_str}"
    config.enable_visualization = False

    return config


def train_single_fold_mpi(
    config: GATConfig,
    dataset: BuildingGraphDataset,
    train_data: List,
    val_data: List,
    fold_idx: int,
    rank: int
) -> Dict[str, Any]:
    """
    train single fold (MPI worker process)

    Args:
        config: training configuration
        dataset: dataset
        train_data: training data
        val_data: validation data
        fold_idx: fold index
        rank: MPI rank

    Returns:
        training result dictionary
    """
    logger.info(f"Rank {rank}: Training fold {fold_idx}...")

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

    # create fold-specific configuration
    fold_config = copy.deepcopy(config)
    fold_config.model_identifier = f"{config.model_identifier}_fold{fold_idx}"
    fold_config.checkpoint_dir = f"{config.output_root_dir}/checkpoints/fold{fold_idx}"
    fold_config.output_dir = f"{config.output_root_dir}/output_{config.model_identifier}_fold{fold_idx}"
    fold_config.enable_tensorboard = False  # use shared writer

    Path(fold_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(fold_config.output_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        config=fold_config,
        train_data_list=train_data,
        val_data_list=val_data
    )

    try:
        history = trainer.train()
        best_val_acc = max(history.get('val_acc', [0])) if history.get('val_acc') else 0

        result = {
            'fold': fold_idx,
            'best_val_acc': best_val_acc,
            'history': history,
            'success': True
        }

        logger.info(f"Rank {rank}: Fold {fold_idx} completed with val_acc={best_val_acc:.4f}")

    except Exception as e:
        logger.error(f"Rank {rank}: Fold {fold_idx} failed: {e}", exc_info=True)
        result = {
            'fold': fold_idx,
            'best_val_acc': 0.0,
            'error': str(e),
            'success': False
        }

    return result


def train_with_params_mpi(
    base_config: GATConfig,
    dataset: BuildingGraphDataset,
    params: Dict[str, Any],
    run_id: int,
    comm,
    rank: int,
    size: int
) -> Dict[str, Any]:
    """
    train with MPI parallelization for single hyperparameter combination - K-fold cross-validation

    Args:
        base_config: base configuration
        dataset: dataset
        params: hyperparameter dictionary
        run_id: run ID
        comm: MPI communicator
        rank: MPI rank
        size: MPI size

    Returns:
        cross-validation result
    """
    config = create_config_with_params(base_config, params, run_id)

    data_list = [dataset.get(i) for i in range(len(dataset))]
    n_folds = config.k_fold

    # check if there are enough processes
    if size < n_folds:
        if rank == 0:
            logger.warning(
                f"Not enough MPI processes ({size}) for {n_folds} folds. "
                f"Some folds will run sequentially."
            )

    # each process trains its assigned folds
    fold_results = []

    for fold_idx, (train_data, val_data) in enumerate(kfold_split(data_list, n_splits=n_folds, random_seed=config.seed), 1):
        if (fold_idx - 1) % size != rank:
            continue
        result = train_single_fold_mpi(
            config=config,
            dataset=dataset,
            train_data=train_data,
            val_data=val_data,
            fold_idx=fold_idx,
            rank=rank
        )
        fold_results.append(result)

    all_fold_results = comm.gather(fold_results, root=0)

    if rank == 0:
        # flatten result list
        all_results = []
        for results in all_fold_results:
            all_results.extend(results)
        all_results.sort(key=lambda x: x['fold'])
        successful_results = [r for r in all_results if r.get('success', False)]

        if successful_results:
            val_accs = [r['best_val_acc'] for r in successful_results]
            mean_val_acc = np.mean(val_accs)
            std_val_acc = np.std(val_accs)

            cv_result = {
                'params': params,
                'run_id': run_id,
                'mean_val_acc': mean_val_acc,
                'std_val_acc': std_val_acc,
                'fold_results': all_results,
                'n_successful_folds': len(successful_results),
                'n_total_folds': n_folds
            }

            logger.info(
                f"Run {run_id} completed: mean_val_acc={mean_val_acc:.4f} ± {std_val_acc:.4f}"
            )
        else:
            cv_result = {
                'params': params,
                'run_id': run_id,
                'mean_val_acc': 0.0,
                'std_val_acc': 0.0,
                'fold_results': all_results,
                'n_successful_folds': 0,
                'n_total_folds': n_folds,
                'error': 'All folds failed'
            }
            logger.error(f"Run {run_id} failed: all folds failed")

        return cv_result
    else:
        return None


def save_best_params(
    results: List[Dict[str, Any]],
    output_dir: Path
) -> None:
    """
    save best hyperparameter configuration

    Args:
        results: all running results
        output_dir: output directory
    """
    # sort to get best results
    successful_results = [r for r in results if r.get('n_successful_folds', 0) > 0]

    if not successful_results:
        logger.error("No successful runs to save")
        return

    # sort by mean validation accuracy
    successful_results.sort(key=lambda x: x['mean_val_acc'], reverse=True)

    # save top 5
    top_k = min(5, len(successful_results))

    logger.info("=" * 80)
    logger.info(f"Top {top_k} Hyperparameter Configurations")
    logger.info("=" * 80)

    for i, result in enumerate(successful_results[:top_k], 1):
        logger.info(
            f"Rank {i}: Val Acc = {result['mean_val_acc']:.4f} ± {result['std_val_acc']:.4f}"
        )
        logger.info(f"  Params: {result['params']}")

    # save best parameters to YAML
    best_result = successful_results[0]
    best_params_path = output_dir / 'best_params.yaml'

    best_params_data = {
        'best_params': best_result['params'],
        'mean_val_acc': float(best_result['mean_val_acc']),
        'std_val_acc': float(best_result['std_val_acc']),
        'n_successful_folds': int(best_result['n_successful_folds']),
        'run_id': int(best_result['run_id'])
    }

    with open(best_params_path, 'w', encoding='utf-8') as f:
        yaml.dump(best_params_data, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Best parameters saved to: {best_params_path}")

    # save all results
    all_results_path = output_dir / 'all_results.yaml'
    all_results_data = []

    for result in successful_results:
        all_results_data.append({
            'run_id': int(result['run_id']),
            'params': result['params'],
            'mean_val_acc': float(result['mean_val_acc']),
            'std_val_acc': float(result['std_val_acc']),
            'n_successful_folds': int(result['n_successful_folds'])
        })

    with open(all_results_path, 'w', encoding='utf-8') as f:
        yaml.dump(all_results_data, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"All results saved to: {all_results_path}")


def log_to_tensorboard_hparams(
    results: List[Dict[str, Any]],
    log_dir: Path
) -> None:
    """
    log hyperparameter tuning results to TensorBoard

    Args:
        results: all running results
        log_dir: TensorBoard log directory
    """
    writer = SummaryWriter(log_dir=str(log_dir))

    successful_results = [r for r in results if r.get('n_successful_folds', 0) > 0]

    for result in successful_results:
        hparams = {
            f'hp/{k}': v for k, v in result['params'].items()
        }
        metrics = {
            'hparam/mean_val_acc': result['mean_val_acc'],
            'hparam/std_val_acc': result['std_val_acc']
        }

        # log to TensorBoard
        writer.add_hparams(
            hparam_dict=hparams,
            metric_dict=metrics,
            run_name=f"run_{result['run_id']}"
        )

    writer.close()
    logger.info(f"TensorBoard HParams logged to: {log_dir}")


def evaluate_hyperparameters_mpi(
    base_config: GATConfig,
    dataset: BuildingGraphDataset
) -> None:
    """
    evaluate hyperparameters with MPI parallelization (main function)

    Args:
        base_config: base configuration
        dataset: dataset
    """
    if not MPI_AVAILABLE:
        logger.error("MPI is not available. Install mpi4py to use hyperparameter tuning.")
        sys.exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_folds = base_config.k_fold

    if rank == 0:
        logger.info("=" * 80)
        logger.info("Hyperparameter Tuning with MPI")
        logger.info("=" * 80)
        logger.info(f"MPI processes: {size}")
        logger.info(f"K-fold: {n_folds}")
        logger.info(f"Processes per hyperparameter run: {n_folds}")
        logger.info(f"Parallel runs: {size // n_folds}")
        logger.info("=" * 80)
        param_combinations = generate_param_combinations()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(base_config.output_root_dir) / f'hyperparameter_tuning_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)

        log_dir = output_dir / 'tensorboard'
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {output_dir}")
        logger.info(f"TensorBoard logs: {log_dir}")
    else:
        param_combinations = None
        output_dir = None
        log_dir = None

    # broadcast parameter combinations and directories
    param_combinations = comm.bcast(param_combinations, root=0)
    output_dir = comm.bcast(output_dir, root=0)
    parallel_runs = size // n_folds

    if rank == 0:
        logger.info(f"Total parameter combinations: {len(param_combinations)}")
        logger.info(f"Will process in batches of {parallel_runs} runs")

    all_results = []

    # batch process hyperparameter combinations
    for batch_start in range(0, len(param_combinations), parallel_runs):
        batch_end = min(batch_start + parallel_runs, len(param_combinations))
        batch_size = batch_end - batch_start

        if rank == 0:
            logger.info(f"\nProcessing batch {batch_start//parallel_runs + 1}: "
                       f"runs {batch_start} to {batch_end-1}")
        run_group = rank // n_folds

        if run_group < batch_size:
            run_id = batch_start + run_group
            params = param_combinations[run_id]

            color = run_group
            sub_comm = comm.Split(color, rank)
            sub_rank = sub_comm.Get_rank()
            sub_size = sub_comm.Get_size()
            result = train_with_params_mpi(
                base_config=base_config,
                dataset=dataset,
                params=params,
                run_id=run_id,
                comm=sub_comm,
                rank=sub_rank,
                size=sub_size
            )

            sub_comm.Free()
        else:
            result = None

        # collect batch results
        batch_results = comm.gather(result, root=0)

        if rank == 0:
            batch_results = [r for r in batch_results if r is not None]
            all_results.extend(batch_results)

    # summarize and save results
    if rank == 0:
        logger.info("\n" + "=" * 80)
        logger.info("Hyperparameter Tuning Completed")
        logger.info("=" * 80)
        logger.info(f"Total runs completed: {len(all_results)}")
        save_best_params(all_results, output_dir)
        log_to_tensorboard_hparams(all_results, log_dir)

        logger.info("=" * 80)
        logger.info("All results saved!")
        logger.info("=" * 80)


def main():
    """main function"""
    parser = argparse.ArgumentParser(description='GAT Hyperparameter Tuning with MPI')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config YAML file')
    parser.add_argument('--adjacency-dir', type=str, required=True,
                       help='Directory containing adjacency matrices')
    parser.add_argument('--building-path', type=str, required=True,
                       help='Path to building shapefile')
    parser.add_argument('--district-path', type=str, required=True,
                       help='Path to district shapefile')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for results')

    args = parser.parse_args()
    if MPI_AVAILABLE:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    if rank == 0:
        logger.info("Loading configuration and dataset...")

    # load configuration
    resource_paths = {
        'adjacency_dir': args.adjacency_dir,
        'building_path': args.building_path,
        'district_path': args.district_path,
        'output_root_dir': args.output_dir,
        'model_identifier': 'hyperparameter_tuning'
    }

    config = GATConfig.from_yaml(Path(args.config), resource_paths)

    # load dataset
    dataset = BuildingGraphDataset(
        adjacency_dir=args.adjacency_dir,
        district_dataset=DistrictDataset(args.district_path),
        building_dataset=BuildingDataset(args.building_path),
        dataset_dir=args.output_dir
    )

    if rank == 0:
        logger.info(f"Dataset loaded: {len(dataset)} districts")
        logger.info(f"Starting hyperparameter tuning...")
    evaluate_hyperparameters_mpi(config, dataset)

    if rank == 0:
        logger.info("Hyperparameter tuning finished!")


if __name__ == '__main__':
    main()
