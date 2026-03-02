"""
Gap simulation for validation - creates artificial gaps in observed data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class GapSimulator:
    """
    Create artificial gaps in time series data for validation.
    
    Supports multiple gap patterns:
    - Random gaps
    - Clustered gaps (simulating sensor failures)
    - Seasonal/periodic gaps
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize gap simulator.
        
        Parameters
        ----------
        random_seed : int
            Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def create_random_gaps(
        self,
        df: pd.DataFrame,
        variables: List[str],
        num_gaps: int,
        min_length: str = '1H',
        max_length: str = '6H',
        only_mask_existing: bool = True,
        min_gap_distance: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create random gaps in the data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        variables : list of str
            Variables to create gaps in
        num_gaps : int
            Number of gaps to create
        min_length : str
            Minimum gap length (pandas timedelta string)
        max_length : str
            Maximum gap length (pandas timedelta string)
        only_mask_existing : bool
            Only create gaps where data exists
        min_gap_distance : str, optional
            Minimum distance between gaps
            
        Returns
        -------
        pd.DataFrame
            Dataframe with gaps (NaN)
        pd.DataFrame
            Ground truth (original values)
        """
        df_gapped = df.copy()
        ground_truth = pd.DataFrame(index=df.index)
        
        # Convert length strings to number of time steps
        freq = pd.infer_freq(df.index) or '30min'
        min_steps = int(pd.Timedelta(min_length) / pd.Timedelta(freq))
        max_steps = int(pd.Timedelta(max_length) / pd.Timedelta(freq))
        
        if min_gap_distance:
            min_distance_steps = int(pd.Timedelta(min_gap_distance) / pd.Timedelta(freq))
        else:
            min_distance_steps = 0
        
        logger.info(f"Creating {num_gaps} random gaps with length {min_length}-{max_length}")
        
        gap_positions = []
        
        for i in range(num_gaps):
            # Find valid position
            max_attempts = 1000
            for attempt in range(max_attempts):
                # Random gap length
                gap_length = np.random.randint(min_steps, max_steps + 1)
                
                # Random start position
                max_start = len(df) - gap_length
                if max_start <= 0:
                    break
                    
                gap_start = np.random.randint(0, max_start)
                gap_end = gap_start + gap_length
                
                # Check if position is valid
                valid = True
                
                # Check minimum distance from other gaps
                if min_distance_steps > 0:
                    for prev_start, prev_end in gap_positions:
                        if not (gap_end + min_distance_steps < prev_start or
                                gap_start > prev_end + min_distance_steps):
                            valid = False
                            break
                
                # Check if data exists at this position
                if only_mask_existing and valid:
                    for var in variables:
                        if df.iloc[gap_start:gap_end][var].notna().sum() == 0:
                            valid = False
                            break
                
                if valid:
                    gap_positions.append((gap_start, gap_end))
                    break
            
            else:
                logger.warning(f"Could not place gap {i+1} after {max_attempts} attempts")
                continue
            
            # Create gap
            gap_start, gap_end = gap_positions[-1]
            for var in variables:
                # Save ground truth
                ground_truth.loc[df.index[gap_start:gap_end], var] = \
                    df.iloc[gap_start:gap_end][var].values
                
                # Create gap
                df_gapped.iloc[gap_start:gap_end, df_gapped.columns.get_loc(var)] = np.nan
        
        logger.info(f"Created {len(gap_positions)} gaps successfully")
        
        return df_gapped, ground_truth
    
    def create_clustered_gaps(
        self,
        df: pd.DataFrame,
        variables: List[str],
        num_clusters: int,
        gaps_per_cluster: int,
        min_length: str = '2H',
        max_length: str = '12H',
        cluster_spread: str = '7D',
        only_mask_existing: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create clustered gaps (simulating realistic sensor failure patterns).
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        variables : list of str
            Variables to create gaps in
        num_clusters : int
            Number of gap clusters
        gaps_per_cluster : int
            Number of gaps in each cluster
        min_length : str
            Minimum gap length
        max_length : str
            Maximum gap length
        cluster_spread : str
            Maximum time span of each cluster
        only_mask_existing : bool
            Only create gaps where data exists
            
        Returns
        -------
        pd.DataFrame
            Dataframe with gaps
        pd.DataFrame
            Ground truth
        """
        df_gapped = df.copy()
        ground_truth = pd.DataFrame(index=df.index)
        
        freq = pd.infer_freq(df.index) or '30min'
        min_steps = int(pd.Timedelta(min_length) / pd.Timedelta(freq))
        max_steps = int(pd.Timedelta(max_length) / pd.Timedelta(freq))
        spread_steps = int(pd.Timedelta(cluster_spread) / pd.Timedelta(freq))
        
        logger.info(
            f"Creating {num_clusters} gap clusters with {gaps_per_cluster} gaps each"
        )
        
        for cluster_idx in range(num_clusters):
            # Random cluster center
            cluster_center = np.random.randint(spread_steps, len(df) - spread_steps)
            
            # Create gaps within cluster
            for gap_idx in range(gaps_per_cluster):
                # Random position within cluster
                gap_length = np.random.randint(min_steps, max_steps + 1)
                offset_from_center = np.random.randint(-spread_steps//2, spread_steps//2)
                gap_start = cluster_center + offset_from_center
                gap_end = gap_start + gap_length
                
                # Ensure valid bounds
                if gap_start < 0 or gap_end >= len(df):
                    continue
                
                # Create gap
                for var in variables:
                    # Save ground truth
                    ground_truth.loc[df.index[gap_start:gap_end], var] = \
                        df.iloc[gap_start:gap_end][var].values
                    
                    # Create gap
                    df_gapped.iloc[gap_start:gap_end, df_gapped.columns.get_loc(var)] = np.nan
        
        return df_gapped, ground_truth
    
    def simulate_gaps_from_config(
        self,
        df: pd.DataFrame,
        gap_config: Dict,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulate gaps based on configuration.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        gap_config : dict
            Gap configuration dictionary
            
        Returns
        -------
        pd.DataFrame
            Dataframe with gaps
        pd.DataFrame
            Ground truth
        """
        gap_type = gap_config.get('type', 'random')
        
        if gap_type == 'random':
            return self.create_random_gaps(
                df=df,
                variables=gap_config['target_vars'],
                num_gaps=gap_config['num_gaps'],
                min_length=gap_config['min_length'],
                max_length=gap_config['max_length'],
                only_mask_existing=gap_config.get('only_mask_existing', True),
                min_gap_distance=gap_config.get('min_gap_distance'),
            )
        
        elif gap_type == 'clustered':
            return self.create_clustered_gaps(
                df=df,
                variables=gap_config['target_vars'],
                num_clusters=gap_config['num_clusters'],
                gaps_per_cluster=gap_config['gaps_per_cluster'],
                min_length=gap_config['min_length'],
                max_length=gap_config['max_length'],
                cluster_spread=gap_config['cluster_spread'],
                only_mask_existing=gap_config.get('only_mask_existing', True),
            )
        
        else:
            raise ValueError(f"Unknown gap type: {gap_type}")
    
    def get_gap_statistics(
        self,
        ground_truth: pd.DataFrame,
    ) -> Dict:
        """
        Calculate statistics about created gaps.
        
        Parameters
        ----------
        ground_truth : pd.DataFrame
            Ground truth dataframe with gap values
            
        Returns
        -------
        dict
            Gap statistics
        """
        stats = {}
        
        for var in ground_truth.columns:
            gap_mask = ground_truth[var].notna()
            
            if gap_mask.sum() == 0:
                continue
            
            # Find gap lengths
            gap_lengths = []
            in_gap = False
            current_length = 0
            
            for val in gap_mask:
                if val:  # In gap
                    if not in_gap:
                        in_gap = True
                        current_length = 1
                    else:
                        current_length += 1
                else:  # Not in gap
                    if in_gap:
                        gap_lengths.append(current_length)
                        in_gap = False
                        current_length = 0
            
            if in_gap:
                gap_lengths.append(current_length)
            
            stats[var] = {
                'num_gaps': len(gap_lengths),
                'total_masked': gap_mask.sum(),
                'min_gap_length': min(gap_lengths) if gap_lengths else 0,
                'max_gap_length': max(gap_lengths) if gap_lengths else 0,
                'mean_gap_length': np.mean(gap_lengths) if gap_lengths else 0,
            }
        
        return stats
