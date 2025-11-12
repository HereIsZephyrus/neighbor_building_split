#!/usr/bin/env python3
"""
Single hyperparameter configuration training script.

This script wraps the main GAT training module to handle a single hyperparameter
configuration and generate a summary file for aggregation.

Usage:
    mpirun -np 5 python -m scripts.train_single_hyperparam \
        --config config.yaml \
        --adjacency-dir /path/to/adjacency \
        --building-path /path/to/buildings.shp \
        --district-path /path/to/districts.shp \
        --output-dir /path/to/output \
        --run-id 0
"""

import argparse
import sys
import yaml
from pathlib import Path
from datetime import datetime
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gat.train import main as train_main


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train GAT model with single hyperparameter configuration'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config YAML file'
    )
    parser.add_argument(
        '--adjacency-dir',
        type=str,
        required=True,
        help='Directory containing adjacency matrices'
    )
    parser.add_argument(
        '--building-path',
        type=str,
        required=True,
        help='Path to building shapefile'
    )
    parser.add_argument(
        '--district-path',
        type=str,
        required=True,
        help='Path to district shapefile'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for this run'
    )
    parser.add_argument(
        '--run-id',
        type=int,
        required=True,
        help='Run identifier'
    )
    
    return parser.parse_args()


def collect_fold_results(output_dir: Path) -> dict:
    """
    Collect results from all folds after training completes.
    
    Args:
        output_dir: Output directory containing fold results
        
    Returns:
        Dictionary with aggregated results
    """
    results = {
        'fold_results': [],
        'mean_val_acc': None,
        'mean_val_f1': None,
        'mean_val_weighted_f1': None,
        'mean_val_macro_f1': None,
        'std_val_acc': None,
        'std_val_f1': None,
        'std_val_weighted_f1': None,
        'std_val_macro_f1': None,
    }
    
    try:
        # Look for tensorboard event files or checkpoint files to extract metrics
        # The actual implementation depends on what src.gat outputs
        
        # Try to find metrics from logs or output files
        # For now, we'll look for any metrics files
        metrics_files = list(output_dir.glob('**/metrics*.yaml'))
        metrics_files.extend(list(output_dir.glob('**/fold*/best_metrics.yaml')))
        
        if not metrics_files:
            # If no metrics files found, try to parse from tensorboard logs
            print(f"Warning: No metrics files found in {output_dir}")
            return results
        
        # Collect metrics from each fold
        val_accs = []
        val_f1s = []
        val_weighted_f1s = []
        val_macro_f1s = []
        
        for metrics_file in metrics_files:
            try:
                with open(metrics_file, 'r') as f:
                    metrics = yaml.safe_load(f)
                
                if metrics:
                    fold_result = {}
                    if 'val_acc' in metrics:
                        val_accs.append(metrics['val_acc'])
                        fold_result['val_acc'] = metrics['val_acc']
                    if 'val_f1' in metrics:
                        val_f1s.append(metrics['val_f1'])
                        fold_result['val_f1'] = metrics['val_f1']
                    if 'val_weighted_f1' in metrics:
                        val_weighted_f1s.append(metrics['val_weighted_f1'])
                        fold_result['val_weighted_f1'] = metrics['val_weighted_f1']
                    if 'val_macro_f1' in metrics:
                        val_macro_f1s.append(metrics['val_macro_f1'])
                        fold_result['val_macro_f1'] = metrics['val_macro_f1']
                    
                    if fold_result:
                        results['fold_results'].append(fold_result)
            
            except Exception as e:
                print(f"Warning: Could not load metrics from {metrics_file}: {e}")
                continue
        
        # Calculate means and stds
        if val_accs:
            import numpy as np
            results['mean_val_acc'] = float(np.mean(val_accs))
            results['std_val_acc'] = float(np.std(val_accs))
        
        if val_f1s:
            import numpy as np
            results['mean_val_f1'] = float(np.mean(val_f1s))
            results['std_val_f1'] = float(np.std(val_f1s))
        
        if val_weighted_f1s:
            import numpy as np
            results['mean_val_weighted_f1'] = float(np.mean(val_weighted_f1s))
            results['std_val_weighted_f1'] = float(np.std(val_weighted_f1s))
        
        if val_macro_f1s:
            import numpy as np
            results['mean_val_macro_f1'] = float(np.mean(val_macro_f1s))
            results['std_val_macro_f1'] = float(np.std(val_macro_f1s))
    
    except Exception as e:
        print(f"Warning: Error collecting fold results: {e}")
        traceback.print_exc()
    
    return results


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Hyperparameter Tuning Run {args.run_id}")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Load config to extract hyperparameters
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Prepare arguments for src.gat.train
    train_args = argparse.Namespace(
        adjacency_dir=args.adjacency_dir,
        sample_buildings=args.building_path,
        sample_districts=args.district_path,
        output_root_dir=args.output_dir,
        model_identifier=f"run{args.run_id:04d}",
        config=args.config,
        resume=None,
        mode='cv'  # Cross-validation mode
    )
    
    # Record start time
    start_time = datetime.now()
    
    # Run training
    success = False
    error_message = None
    
    try:
        print("\nStarting training...")
        train_main(train_args)
        success = True
        print("\n✓ Training completed successfully!")
        
    except Exception as e:
        success = False
        error_message = str(e)
        print(f"\n✗ Training failed: {e}")
        traceback.print_exc()
    
    # Record end time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Collect results from folds
    print("\nCollecting results from folds...")
    fold_results = collect_fold_results(output_dir)
    
    # Generate summary file
    summary = {
        'run_id': args.run_id,
        'status': 'completed' if success else 'failed',
        'config_file': args.config,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration,
        'hyperparameters': {
            'model': config_dict.get('model', {}),
            'training': {
                k: v for k, v in config_dict.get('training', {}).items()
                if k in ['lr', 'weight_decay', 'lambda_smooth', 'dropout']
            },
            'spectral_clustering': {
                k: v for k, v in config_dict.get('spectral_clustering', {}).items()
                if k in ['embedding_weight', 'feature_weight', 'distance_weight']
            }
        },
        **fold_results
    }
    
    if error_message:
        summary['error'] = error_message
    
    # Save summary
    summary_path = output_dir / 'run_summary.yaml'
    print(f"\nSaving summary to {summary_path}")
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)
    
    print("=" * 80)
    print(f"Run {args.run_id} Summary:")
    print(f"  Status: {summary['status']}")
    print(f"  Duration: {duration:.1f}s")
    if fold_results['mean_val_acc'] is not None:
        print(f"  Mean Val Accuracy: {fold_results['mean_val_acc']:.4f} ± {fold_results['std_val_acc']:.4f}")
    if fold_results['mean_val_weighted_f1'] is not None:
        print(f"  Mean Val Weighted F1: {fold_results['mean_val_weighted_f1']:.4f} ± {fold_results['std_val_weighted_f1']:.4f}")
    print("=" * 80)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

