from pathlib import Path
from typing import Dict, List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

__all__ = ["get_configs", "list_configs", "get_config"]


def get_configs(folder: Optional[str] = None) -> Dict[str, DictConfig]:
    config_dir = Path(__file__).parent
    if folder:
        config_dir = config_dir / folder
    config_names = [file.name.replace(".yaml", "") for file in config_dir.glob("*.yaml")]
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.2"):
        outputs = {name: compose(config_name=name) for name in config_names}
    return outputs


def list_configs(folder: Optional[str] = None) -> List[str]:
    config_dir = Path(__file__).parent
    if folder:
        config_dir = config_dir / folder
    config_names = [file.name.replace(".yaml", "") for file in config_dir.glob("*.yaml")]
    return config_names


def get_config(config_name: str, folder: Optional[str] = None, overrides: Optional[List[str]] = None) -> DictConfig:
    config_dir = Path(__file__).parent
    if folder:
        config_dir = config_dir / folder
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.2"):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg
