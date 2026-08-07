"""
YAML and JSON configuration loader for model and training specifications.
"""

import json
import os
from typing import Tuple, Dict, Any, Union
import yaml

from slm.config.model_config import ModelConfig
from slm.config.train_config import TrainConfig


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Loads raw configuration dictionary from YAML file.

    Args:
        file_path: Path to .yaml or .yml file.

    Returns:
        Dictionary of raw settings.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    
    return config_data if config_data is not None else {}


def load_json_config(file_path: str) -> Dict[str, Any]:
    """
    Loads raw configuration dictionary from JSON file.

    Args:
        file_path: Path to .json file.

    Returns:
        Dictionary of raw settings.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(file_path: str) -> Tuple[ModelConfig, TrainConfig]:
    """
    Parses configuration file (.yaml or .json) into ModelConfig and TrainConfig objects.

    Args:
        file_path: Path to YAML or JSON config file.

    Returns:
        Tuple of (ModelConfig, TrainConfig).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".yaml", ".yml"):
        raw_data = load_yaml_config(file_path)
    elif ext == ".json":
        raw_data = load_json_config(file_path)
    else:
        raise ValueError(f"Unsupported config extension: {ext}. Use .yaml, .yml, or .json")

    model_dict = raw_data.get("model", raw_data)
    train_dict = raw_data.get("training", raw_data)

    model_config = ModelConfig.from_dict(model_dict)
    train_config = TrainConfig.from_dict(train_dict)

    return model_config, train_config


def save_config(model_config: ModelConfig, train_config: TrainConfig, file_path: str) -> None:
    """
    Saves ModelConfig and TrainConfig instances to YAML or JSON file.

    Args:
        model_config: Model parameters.
        train_config: Training parameters.
        file_path: Destination path.
    """
    combined_dict = {
        "model": model_config.to_dict(),
        "training": train_config.to_dict()
    }
    
    ext = os.path.splitext(file_path)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            yaml.dump(combined_dict, f, default_flow_style=False, sort_keys=False)
        elif ext == ".json":
            json.dump(combined_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format extension: {ext}")
