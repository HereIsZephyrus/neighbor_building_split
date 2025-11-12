#!/usr/bin/env python3
"""
Generate LSF job scripts for hyperparameter tuning.

This script generates individual LSF bsub scripts for each hyperparameter
combination, allowing parallel execution of hyperparameter tuning jobs
on an HPC cluster.

Usage:
    python scripts/generate_tuning_jobs.py \
        --base-config src/gat/training_config.yaml \
        --adjacency-dir /path/to/adjacency \
        --building-path /path/to/buildings.shp \
        --district-path /path/to/districts.shp \
        --output-dir experiments/tuning \
        --submit  # Optional: auto-submit jobs
"""

import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import yaml
from sklearn.model_selection import ParameterGrid


class HyperparameterGenerator:
    """
    Generator for hyperparameter tuning jobs.

    Creates individual LSF job scripts for each hyperparameter combination,
    enabling independent parallel execution on HPC clusters.
    """

    # Hyperparameter search space
    PARAM_GRID = {
        'hidden_dim': [32, 64],
        'num_layers': [2, 3],
        'num_heads': [4, 8],
        'dropout': [0.3, 0.4, 0.5, 0.6],
        'lr': [5e-4, 1e-3, 2e-3, 5e-3],
        'weight_decay': [5e-4, 1e-3, 2e-3, 5e-3],
        'lambda_smooth': [0.1, 0.2, 0.5, 1.0],
        'weight_components': [ # embedding_weight, feature_weight, distance_weight
            # Balanced weights
            [0.3, 0.3, 0.4],
            # Emphasize embedding
            [0.5, 0.25, 0.25],
            [0.4, 0.3, 0.3],
            # Emphasize feature
            [0.25, 0.5, 0.25],
            [0.3, 0.4, 0.3],
            # Emphasize distance
            [0.25, 0.25, 0.5],
            [0.3, 0.3, 0.4],
            # Mixed emphasis
            [0.4, 0.4, 0.2],
            [0.2, 0.4, 0.4],
            [0.4, 0.2, 0.4]
        ]
    }

    def __init__(
        self,
        base_config_path: Path,
        adjacency_dir: str,
        building_path: str,
        district_path: str,
        output_dir: Path,
        n_cores: int = 5,
        memory_per_core: int = 4096,
        time_limit: str = "24:00",
        queue: str = "normal"
    ):
        """
        Initialize the hyperparameter generator.

        Args:
            base_config_path: Path to base training config YAML
            adjacency_dir: Directory containing adjacency matrices
            building_path: Path to building shapefile
            district_path: Path to district shapefile
            output_dir: Output directory for job scripts and results
            n_cores: Number of cores per job (should match k_fold)
            memory_per_core: Memory per core in MB
            time_limit: Time limit per job (HH:MM)
            queue: LSF queue name
        """
        self.base_config_path = base_config_path
        self.adjacency_dir = adjacency_dir
        self.building_path = building_path
        self.district_path = district_path
        self.output_dir = output_dir
        self.n_cores = n_cores
        self.memory_per_core = memory_per_core
        self.time_limit = time_limit
        self.queue = queue

        # Load base configuration
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)

        # Create output directories
        self.jobs_dir = output_dir / 'job_scripts'
        self.configs_dir = output_dir / 'configs'
        self.results_dir = output_dir / 'results'
        self.logs_dir = output_dir / 'logs'

        for dir_path in [self.jobs_dir, self.configs_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Generate parameter combinations
        self.param_combinations = list(ParameterGrid(self.PARAM_GRID))

    @staticmethod
    def _format_param_value(_key: str, value: Any) -> str:
        """
        Format parameter value for filename generation.
        
        Args:
            _key: Parameter name (reserved for future use)
            value: Parameter value
            
        Returns:
            Formatted string representation
        """
        if isinstance(value, list):
            # For weight_components like [0.3, 0.3, 0.4], extract just the numbers
            # Convert to string like "334" (e.g., 0.3 -> 3, 0.33 -> 3, 0.25 -> 2, 0.6 -> 6)
            formatted_values = []
            for v in value:
                if isinstance(v, float):
                    # Extract first digit after decimal point
                    # e.g., 0.3 -> 3, 0.33 -> 3, 0.25 -> 2, 0.6 -> 6
                    int_val = int(round(v * 10))
                    formatted_values.append(str(int_val))
                else:
                    formatted_values.append(str(v))
            return ''.join(formatted_values)
        elif isinstance(value, float):
            # Format float values more compactly
            if value < 0.01:
                # Scientific notation for very small values (e.g., 5e-4 -> 5e-4)
                return f"{value:.0e}".replace('-0', '-')
            else:
                # Regular format (e.g., 0.3 -> 0.3)
                return str(value)
        else:
            return str(value)

    def generate_all_jobs(self) -> List[Path]:
        """
        Generate all job scripts for hyperparameter combinations.

        Returns:
            List of paths to generated job scripts
        """
        job_scripts = []

        print("=" * 80)
        print(f"Generating LSF job scripts for hyperparameter tuning")
        print("=" * 80)
        print(f"Total parameter combinations: {len(self.param_combinations)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Cores per job: {self.n_cores}")
        print(f"Memory per core: {self.memory_per_core} MB")
        print(f"Time limit: {self.time_limit}")
        print(f"Queue: {self.queue}")
        print("=" * 80)

        for run_id, params in enumerate(self.param_combinations):
            job_script = self._generate_single_job(run_id, params)
            job_scripts.append(job_script)

            if (run_id + 1) % 10 == 0:
                print(f"Generated {run_id + 1}/{len(self.param_combinations)} job scripts...")

        print(f"✓ All {len(job_scripts)} job scripts generated successfully!")

        # Generate tracking file
        self._generate_tracking_file()

        return job_scripts

    def _generate_single_job(self, run_id: int, params: Dict[str, Any]) -> Path:
        """
        Generate a single LSF job script for given parameters.

        Args:
            run_id: Unique run identifier
            params: Hyperparameter dictionary

        Returns:
            Path to generated job script
        """
        # Create a short parameter identifier
        param_str = '_'.join([
            f"{k[:2]}{self._format_param_value(k, v)}" 
            for k, v in params.items()
        ])
        job_name = f"tune_run{run_id:04d}_tcb"

        # Generate config file with these parameters
        config_path = self._generate_config_file(run_id, params, param_str)

        # Create result directory for this run
        run_result_dir = self.results_dir / job_name
        run_result_dir.mkdir(parents=True, exist_ok=True)

        # Generate LSF script
        job_script_path = self.jobs_dir / f"{job_name}.sh"

        script_content = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -n {self.n_cores}
#BSUB -W {self.time_limit}
#BSUB -M {self.memory_per_core}
#BSUB -R "span[ptile={self.n_cores}]"
#BSUB -R "rusage[mem={self.memory_per_core}]"
#BSUB -m "node4 node5 node6"
#BSUB -q {self.queue}
#BSUB -o {self.logs_dir}/{job_name}_%J.out
#BSUB -e {self.logs_dir}/{job_name}_%J.err

# Configuration
RUN_ID={run_id}
CONFIG_FILE="{config_path}"
ADJACENCY_DIR="{self.adjacency_dir}"
BUILDING_PATH="{self.building_path}"
DISTRICT_PATH="{self.district_path}"
OUTPUT_DIR="{run_result_dir}"

# Print job information
echo "========================================================================"
echo "Hyperparameter Tuning - Run $RUN_ID"
echo "========================================================================"
echo "Job ID: $LSB_JOBID"
echo "Job Name: {job_name}"
echo "Started at: $(date)"
echo "Running on host: $(hostname)"
echo "Parameters:"
"""

        # Add parameter information
        for key, value in params.items():
            script_content += f'echo "  {key}: {value}"\n'

        script_content += f"""
echo "========================================================================"

# Setup environment
cd /data/users/guxh01/2026_tcb/neighbor_building_split || exit 1
source ~/.bashrc
conda activate build_adj || exit 1

# Run training with MPI (K-fold cross-validation)
echo "Starting training..."
mpirun -np {self.n_cores} python -m scripts.train_single_hyperparam \\
    --config "$CONFIG_FILE" \\
    --adjacency-dir "$ADJACENCY_DIR" \\
    --building-path "$BUILDING_PATH" \\
    --district-path "$DISTRICT_PATH" \\
    --output-dir "$OUTPUT_DIR" \\
    --run-id $RUN_ID

EXIT_CODE=$?

echo "========================================================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================================================"

exit $EXIT_CODE
"""

        # Write script file
        with open(job_script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # Make executable
        job_script_path.chmod(0o755)

        return job_script_path

    def _generate_config_file(
        self,
        run_id: int,
        params: Dict[str, Any],
        param_str: str
    ) -> Path:
        """
        Generate configuration file with specific hyperparameters.

        Args:
            run_id: Run identifier
            params: Hyperparameter dictionary
            param_str: Short parameter string identifier

        Returns:
            Path to generated config file
        """
        # Deep copy base config
        import copy
        config = copy.deepcopy(self.base_config)

        # Update model parameters
        if 'hidden_dim' in params:
            config['model']['hidden_dim'] = params['hidden_dim']
        if 'num_layers' in params:
            config['model']['num_layers'] = params['num_layers']
        if 'num_heads' in params:
            config['model']['num_heads'] = params['num_heads']
        if 'dropout' in params:
            config['model']['dropout'] = params['dropout']

        # Update training parameters
        if 'lr' in params:
            config['training']['lr'] = params['lr']
        if 'weight_decay' in params:
            config['training']['weight_decay'] = params['weight_decay']
        if 'lambda_smooth' in params:
            config['training']['lambda_smooth'] = params['lambda_smooth']
        
        # Update weight components for spectral clustering if present
        if 'weight_components' in params:
            weights = params['weight_components']
            # These weights are used in spectral clustering
            if 'spectral_clustering' not in config:
                config['spectral_clustering'] = {}
            config['spectral_clustering']['embedding_weight'] = weights[0]
            config['spectral_clustering']['feature_weight'] = weights[1]
            config['spectral_clustering']['distance_weight'] = weights[2]

        # Disable checkpoint saving for HPC tuning
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['enable_checkpoint_saving'] = False
        config['logging']['enable_final_visualization'] = True

        # Save config file
        config_path = self.configs_dir / f"config_run{run_id:04d}_{param_str}.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return config_path

    def _generate_tracking_file(self) -> None:
        """Generate a tracking file listing all runs and their parameters."""
        tracking_path = self.output_dir / 'job_tracking.yaml'

        tracking_data = {
            'generated_at': datetime.now().isoformat(),
            'total_jobs': len(self.param_combinations),
            'base_config': str(self.base_config_path),
            'resources': {
                'cores_per_job': self.n_cores,
                'memory_per_core_mb': self.memory_per_core,
                'time_limit': self.time_limit,
                'queue': self.queue
            },
            'jobs': []
        }

        for run_id, params in enumerate(self.param_combinations):
            param_str = '_'.join([
                f"{k[:2]}{self._format_param_value(k, v)}" 
                for k, v in params.items()
            ])
            job_name = f"tcb_tune_run{run_id:04d}"

            tracking_data['jobs'].append({
                'run_id': run_id,
                'job_name': job_name,
                'script': str(self.jobs_dir / f"{job_name}.sh"),
                'config': str(self.configs_dir / f"config_run{run_id:04d}_{param_str}.yaml"),
                'output_dir': str(self.results_dir / job_name),
                'parameters': params
            })

        with open(tracking_path, 'w', encoding='utf-8') as f:
            yaml.dump(tracking_data, f, default_flow_style=False, allow_unicode=True)

        print(f"✓ Job tracking file saved to: {tracking_path}")

    def submit_all_jobs(self) -> List[str]:
        """
        Submit all generated job scripts to LSF.

        Returns:
            List of LSF job IDs
        """
        job_ids = []

        print("\n" + "=" * 80)
        print("Submitting jobs to LSF...")
        print("=" * 80)

        for run_id in range(len(self.param_combinations)):
            job_name = f"tune_run{run_id:04d}"
            job_script = self.jobs_dir / f"{job_name}.sh"

            try:
                result = subprocess.run(
                    ['bsub', '<', str(job_script)],
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=True
                )

                # Parse job ID from bsub output
                # Typical output: "Job <12345> is submitted to queue <normal>."
                output = result.stdout.strip()
                if 'Job <' in output:
                    job_id = output.split('Job <')[1].split('>')[0]
                    job_ids.append(job_id)
                    print(f"✓ Submitted {job_name}: Job ID {job_id}")
                else:
                    print(f"✗ Failed to parse job ID for {job_name}: {output}")

            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to submit {job_name}: {e}")
                print(f"  Error output: {e.stderr}")

        print("=" * 80)
        print(f"✓ Submitted {len(job_ids)}/{len(self.param_combinations)} jobs successfully!")
        print("=" * 80)

        # Save job IDs
        job_ids_path = self.output_dir / 'submitted_job_ids.txt'
        with open(job_ids_path, 'w', encoding='utf-8') as f:
            for job_id in job_ids:
                f.write(f"{job_id}\n")

        print(f"✓ Job IDs saved to: {job_ids_path}")

        return job_ids


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate LSF job scripts for hyperparameter tuning'
    )
    parser.add_argument(
        '--base-config',
        type=str,
        required=True,
        help='Path to base training config YAML file'
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
        help='Output directory for job scripts and results'
    )
    parser.add_argument(
        '--n-cores',
        type=int,
        default=5,
        help='Number of cores per job (default: 5, should match k_fold)'
    )
    parser.add_argument(
        '--memory-per-core',
        type=int,
        default=4096,
        help='Memory per core in MB (default: 4096)'
    )
    parser.add_argument(
        '--time-limit',
        type=str,
        default='24:00',
        help='Time limit per job in HH:MM format (default: 24:00)'
    )
    parser.add_argument(
        '--queue',
        type=str,
        default='normal',
        help='LSF queue name (default: normal)'
    )
    parser.add_argument(
        '--submit',
        action='store_true',
        help='Automatically submit all jobs after generation'
    )

    args = parser.parse_args()

    # Create generator
    generator = HyperparameterGenerator(
        base_config_path=Path(args.base_config),
        adjacency_dir=args.adjacency_dir,
        building_path=args.building_path,
        district_path=args.district_path,
        output_dir=Path(args.output_dir),
        n_cores=args.n_cores,
        memory_per_core=args.memory_per_core,
        time_limit=args.time_limit,
        queue=args.queue
    )

    # Generate all job scripts
    job_scripts = generator.generate_all_jobs()

    print(f"\n✓ Successfully generated {len(job_scripts)} job scripts!")
    print(f"\nTo submit jobs manually, run:")
    print(f"  cd {generator.jobs_dir}")
    print(f"  for script in *.sh; do bsub < $script; done")

    # Submit jobs if requested
    if args.submit:
        job_ids = generator.submit_all_jobs()
        print(f"\n✓ All jobs submitted! Monitor with: bjobs")
    else:
        print(f"\nTo auto-submit all jobs, run with --submit flag")


if __name__ == '__main__':
    main()

