import pandas as pd
import logging
from pathlib import Path

from obsea_pipeline.config.settings import CONFIG

logger = logging.getLogger(__name__)

def load_instrument_data(instrument: str) -> pd.DataFrame:
    """
    Load data for a specific instrument from CSV file (Fallback to the new STA API).
    """
    # Assuming run from scripts directory or package root
    base_path = Path.cwd() 
    file_path = base_path / CONFIG['data_paths'][instrument]
    
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Parse TIME column
    if 'TIME' in df.columns:
        df['TIME'] = pd.to_datetime(df['TIME'])
        df.set_index('TIME', inplace=True)
    elif 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    elif 'Unnamed: 0' in df.columns:
        df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
        df.set_index('Unnamed: 0', inplace=True)
        df.index.name = 'Timestamp'
        
    df.sort_index(inplace=True)
    
    # Get relevant variables for this instrument based on dict mapping
    vars_to_keep = []
    for var in CONFIG['variables'].get(instrument, []):
        if var in df.columns:
            vars_to_keep.append(var)
            if f'{var}_QC' in df.columns:
                vars_to_keep.append(f'{var}_QC')
            if f'{var}_STD' in df.columns:
                vars_to_keep.append(f'{var}_STD')
    
    return df[vars_to_keep] if vars_to_keep else df

def load_all_data() -> dict:
    """
    Load all configured instrument data from local storage.
    """
    data = {}
    logger.info("Loading baseline data from local CSVs...")
    
    for instrument in CONFIG['data_paths'].keys():
        logger.info(f"Loading {instrument}...")
        df = load_instrument_data(instrument)
        if not df.empty:
            data[instrument] = df
            logger.info(f"  ✓ {instrument}: {len(df):,} records.")
            
    return data
