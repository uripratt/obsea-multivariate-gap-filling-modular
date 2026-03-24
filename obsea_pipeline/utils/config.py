"""
Configuration utilities for loading and validating YAML config files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML config file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def load_all_configs(config_dir: str = "configs") -> Dict[str, Dict[str, Any]]:
    """
    Load all configuration files from the config directory.
    
    Parameters
    ----------
    config_dir : str
        Directory containing config files
        
    Returns
    -------
    dict
        Dictionary mapping config names to their contents
    """
    config_path = Path(config_dir)
    configs = {}
    
    for config_file in config_path.glob("*.yaml"):
        config_name = config_file.stem
        configs[config_name] = load_config(str(config_file))
    
    logger.info(f"Loaded {len(configs)} configuration files")
    return configs


def get_nested_value(config: Dict[str, Any], key_path: str, default: Optional[Any] = None) -> Any:
    """
    Get a value from a nested dictionary using dot notation.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    key_path : str
        Dot-separated path to the value (e.g., "model.hidden_size")
    default : any, optional
        Default value if key not found
        
    Returns
    -------
    any
        Value at the specified path
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Parameters
    ----------
    *configs : dict
        Configuration dictionaries to merge
        
    Returns
    -------
    dict
        Merged configuration
    """
    merged = {}
    
    for config in configs:
        merged = _deep_merge(merged, config)
    
    return merged


def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    
    Parameters
    ----------
    dict1 : dict
        First dictionary
    dict2 : dict
        Second dictionary (takes precedence)
        
    Returns
    -------
    dict
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result
