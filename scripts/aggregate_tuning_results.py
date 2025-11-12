#!/usr/bin/env python3
"""
Aggregate results from hyperparameter tuning jobs.

This script scans the output directory for all completed tuning runs,
aggregates the cross-validation results, and generates a summary report
with the best hyperparameter configurations.

Usage:
    python scripts/aggregate_tuning_results.py \
        --tuning-dir experiments/tuning \
        --output-file experiments/tuning/best_params.yaml \
        --top-k 10
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def load_run_results(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Load results from all completed runs.

    Args:
        results_dir: Directory containing run result subdirectories

    Returns:
        List of result dictionaries
    """
    results = []

    # Find all run directories
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])

    print(f"Scanning {len(run_dirs)} run directories...")

    for run_dir in run_dirs:
        summary_path = run_dir / 'run_summary.yaml'

        if not summary_path.exists():
            print(f"Skipping {run_dir.name}: no run_summary.yaml found")
            continue

        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = yaml.safe_load(f)

            # Check if run completed successfully
            if summary.get('status') != 'completed':
                print(f"Skipping {run_dir.name}: status = {summary.get('status')}")
                continue

            # Check if we have CV results
            if 'mean_val_acc' not in summary or summary['mean_val_acc'] is None:
                print(f"Skipping {run_dir.name}: no mean_val_acc found")
                continue

            results.append(summary)

        except Exception as e:
            print(f"Error loading {run_dir.name}: {e}")
            continue

    print(f"Successfully loaded {len(results)} completed runs")

    return results


def generate_summary_report(
    results: List[Dict[str, Any]],
    output_path: Path,
    top_k: int = 10
) -> None:
    """
    Generate summary report with best hyperparameter configurations.

    Args:
        results: List of result dictionaries
        output_path: Path to save summary report
        top_k: Number of top configurations to save
    """
    if not results:
        print("No results to summarize!")
        return

    # Sort by mean validation accuracy
    results_sorted = sorted(
        results,
        key=lambda x: x.get('mean_val_acc', 0),
        reverse=True
    )

    print("\n" + "=" * 80)
    print("Hyperparameter Tuning Results Summary")
    print("=" * 80)
    print(f"Total completed runs: {len(results)}")
    print(f"Showing top {min(top_k, len(results))} configurations")
    print("=" * 80)

    # Display top configurations
    for rank, result in enumerate(results_sorted[:top_k], 1):
        mean_acc = result.get('mean_val_acc', 0)
        std_acc = result.get('std_val_acc', 0)
        run_id = result.get('run_id', 'unknown')
        params = result.get('hyperparameters', {})

        print(f"\nRank {rank}: Run {run_id}")
        print(f"  Val Acc: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Parameters:")
        for key, value in params.items():
            if value is not None:
                print(f"    {key}: {value}")

    print("\n" + "=" * 80)

    # Save best parameters
    best_result = results_sorted[0]
    best_params_data = {
        'best_params': best_result.get('hyperparameters'),
        'mean_val_acc': float(best_result.get('mean_val_acc', 0)),
        'std_val_acc': float(best_result.get('std_val_acc', 0)),
        'n_folds': int(best_result.get('n_folds', 0)),
        'run_id': int(best_result.get('run_id', 0)),
        'timestamp': best_result.get('timestamp')
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(best_params_data, f, default_flow_style=False, allow_unicode=True)

    print(f"Best parameters saved to: {output_path}")

    # Save all results
    all_results_path = output_path.parent / 'all_results.yaml'
    all_results_data = []

    for result in results_sorted:
        all_results_data.append({
            'run_id': int(result.get('run_id', 0)),
            'hyperparameters': result.get('hyperparameters'),
            'mean_val_acc': float(result.get('mean_val_acc', 0)),
            'std_val_acc': float(result.get('std_val_acc', 0)),
            'n_folds': int(result.get('n_folds', 0)),
            'timestamp': result.get('timestamp')
        })

    with open(all_results_path, 'w', encoding='utf-8') as f:
        yaml.dump(all_results_data, f, default_flow_style=False, allow_unicode=True)

    print(f"All results saved to: {all_results_path}")


def log_to_tensorboard(
    results: List[Dict[str, Any]],
    log_dir: Path
) -> None:
    """
    Log hyperparameter tuning results to TensorBoard.

    Args:
        results: List of result dictionaries
        log_dir: TensorBoard log directory
    """
    if not results:
        print("No results to log to TensorBoard")
        return

    print("\n" + "=" * 80)
    print("Logging results to TensorBoard...")
    print("=" * 80)

    writer = SummaryWriter(log_dir=str(log_dir))

    for result in results:
        run_id = result.get('run_id', 0)
        params = result.get('hyperparameters', {})
        mean_acc = result.get('mean_val_acc', 0)
        std_acc = result.get('std_val_acc', 0)

        # Create hparams dictionary
        hparams = {}
        for key, value in params.items():
            if value is not None:
                hparams[f'hp/{key}'] = value

        # Create metrics dictionary
        metrics = {
            'hparam/mean_val_acc': mean_acc,
            'hparam/std_val_acc': std_acc
        }

        # Log to TensorBoard
        try:
            writer.add_hparams(
                hparam_dict=hparams,
                metric_dict=metrics,
                run_name=f"run_{run_id:04d}"
            )
        except Exception as e:
            print(f"Failed to log run {run_id} to TensorBoard: {e}")

    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"  View with: tensorboard --logdir {log_dir}")


def generate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics from all results.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with statistics
    """
    if not results:
        return {}

    mean_accs = [r.get('mean_val_acc', 0) for r in results]

    stats = {
        'total_runs': len(results),
        'best_val_acc': float(np.max(mean_accs)),
        'worst_val_acc': float(np.min(mean_accs)),
        'avg_val_acc': float(np.mean(mean_accs)),
        'median_val_acc': float(np.median(mean_accs)),
        'std_val_acc': float(np.std(mean_accs)),
        'generated_at': datetime.now().isoformat()
    }

    print("\n" + "=" * 80)
    print("Overall Statistics")
    print("=" * 80)
    print(f"Total runs: {stats['total_runs']}")
    print(f"Best validation accuracy: {stats['best_val_acc']:.4f}")
    print(f"Worst validation accuracy: {stats['worst_val_acc']:.4f}")
    print(f"Average validation accuracy: {stats['avg_val_acc']:.4f}")
    print(f"Median validation accuracy: {stats['median_val_acc']:.4f}")
    print(f"Standard deviation: {stats['std_val_acc']:.4f}")
    print("=" * 80)

    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Aggregate hyperparameter tuning results'
    )
    parser.add_argument(
        '--tuning-dir',
        type=str,
        required=True,
        help='Directory containing tuning results'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file for best parameters (default: <tuning-dir>/best_params.yaml)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top configurations to display (default: 10)'
    )
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Generate TensorBoard logs for hyperparameter comparison'
    )

    args = parser.parse_args()

    tuning_dir = Path(args.tuning_dir)
    results_dir = tuning_dir / 'results'

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Load all results
    results = load_run_results(results_dir)

    if not results:
        print("No completed runs found!")
        return

    # Generate statistics
    stats = generate_statistics(results)

    # Save statistics
    stats_path = tuning_dir / 'statistics.yaml'
    with open(stats_path, 'w', encoding='utf-8') as f:
        yaml.dump(stats, f, default_flow_style=False, allow_unicode=True)
    print(f"Statistics saved to: {stats_path}")

    # Determine output path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = tuning_dir / 'best_params.yaml'

    # Generate summary report
    generate_summary_report(results, output_path, args.top_k)

    # Generate TensorBoard logs if requested
    if args.tensorboard:
        log_dir = tuning_dir / 'tensorboard'
        log_to_tensorboard(results, log_dir)

    print("\n" + "=" * 80)
    print("Aggregation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
