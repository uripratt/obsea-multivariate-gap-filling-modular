#!/usr/bin/env python
"""
Script 2: Simulate artificial gaps for validation
- Load preprocessed data
- Create various gap patterns (random, clustered)
- Save gapped data and ground truth
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logger
from src.data import load_obsea_data, GapSimulator
import pandas as pd
import logging

logger = setup_logger(
    name="gap_simulation",
    log_file="results/logs/02_gap_simulation.log",
    level=logging.INFO
)

def main():
    """Main gap simulation routine."""
    logger.info("="*80)
    logger.info("ARTIFICIAL GAP SIMULATION")
    logger.info("="*80)
    
    # Load configurations
    data_config = load_config("configs/data_config.yaml")
    gap_config = load_config("configs/gap_simulation.yaml")
    
    # Create simulator
    simulator = GapSimulator(random_seed=gap_config['random_seed'])
    
    # Load validation and test sets
    processed_dir = Path(data_config['data']['processed_dir'])
    output_dir = Path(gap_config['simulation']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_name in ['val', 'test']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset_name.upper()} set")
        logger.info(f"{'='*60}")
        
        # Load data
        dataset_file = processed_dir / data_config['data'][f'{dataset_name}_file']
        df = load_obsea_data(str(dataset_file))
        
        # Create gaps for each pattern
        for pattern_idx, pattern in enumerate(gap_config['gap_patterns']):
            pattern_name = pattern['name']
            logger.info(f"\nPattern {pattern_idx + 1}: {pattern_name}")
            
            # Simulate gaps
            df_gapped, ground_truth = simulator.simulate_gaps_from_config(
                df.copy(), pattern
            )
            
            # Get statistics
            stats = simulator.get_gap_statistics(ground_truth)
            
            # Log statistics
            for var, var_stats in stats.items():
                logger.info(f"  {var}:")
                logger.info(f"    Gaps created: {var_stats['num_gaps']}")
                logger.info(f"    Total masked points: {var_stats['total_masked']}")
                logger.info(f"    Gap length range: {var_stats['min_gap_length']}-{var_stats['max_gap_length']}")
                logger.info(f"    Mean gap length: {var_stats['mean_gap_length']:.1f}")
            
            # Save
            output_prefix = output_dir / f"{dataset_name}_{pattern_name}"
            df_gapped.to_csv(f"{output_prefix}_gapped.csv")
            ground_truth.to_csv(f"{output_prefix}_truth.csv")
            
            logger.info(f"  Saved to: {output_prefix}_*.csv")
    
    logger.info("\n" + "="*80)
    logger.info("GAP SIMULATION COMPLETE!")
    logger.info("="*80)
    
    print("\n✓ Gap simulation complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Patterns created: {len(gap_config['gap_patterns'])}")
    print(f"  Datasets: validation, test")

if __name__ == "__main__":
    main()
