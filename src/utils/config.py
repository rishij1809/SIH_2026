"""
Configuration manager for Project Vigilance.
Loads YAML configuration files into python dictionaries with type checking and fallback defaults.
"""

from pathlib import Path
from typing import Any, Dict, Union
import yaml

from src.utils.logger import setup_logger

logger = setup_logger("ConfigManager")


def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """Loads configuration file from specified path."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found at: {path.resolve()}")
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.debug(f"Loaded config successfully from {path}")
        return config
    except Exception as e:
        logger.error(f"Failed to parse configuration file {path}: {str(e)}")
        raise e
