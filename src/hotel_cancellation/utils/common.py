import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from hotel_cancellation.exception import ConfigError, ModelError


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its content as a dict."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as file:
        content = yaml.safe_load(file)
    if not isinstance(content, dict):
        raise ConfigError(f"Config file is empty or not a mapping: {path}")
    return content


def _to_builtin(value: Any) -> Any:
    """Turn numpy / pandas / Path values into plain Python types so they can go into JSON."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path | pd.Timestamp):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=_to_builtin)


def read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def save_object(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_object(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise ModelError(f"Saved object not found: {path}")
    return joblib.load(path)


def split_tail(df: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-sorted frame into (head, tail), where tail holds the last `fraction` rows."""
    if not 0 < fraction < 1:
        raise ConfigError(f"Split fraction must be between 0 and 1, got {fraction}")
    cut = int(round(len(df) * (1 - fraction)))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
