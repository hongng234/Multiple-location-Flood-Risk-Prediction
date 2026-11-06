"""Config loading and output-directory setup."""
from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Read the YAML config file into a plain dict.

    What: single place where settings are loaded, so every module reads the same values.
    How: open the file as UTF-8 (Vietnamese names are safe) and parse it with `yaml.safe_load`,
    which only builds plain Python types (no arbitrary object execution).

    Args:
        path: path to the YAML file (default: config.yaml in the working directory).

    Returns:
        Nested dict, e.g. config["data"]["locations"].
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict) -> None:
    """Create every folder the pipeline writes to (derived from config paths, not hard-coded).

    What: avoids "No such file or directory" errors when saving CSVs, figures or the model.
    How: collect the parent folder of every output path in the config, then create each one.
    Deriving folders from the config lets tests redirect all outputs to a temporary directory.
    """
    # Parent folders of every file the pipeline writes + the figure/metric folders themselves
    dirs = [
        Path(config["data"]["raw_path"]).parent,
        Path(config["data"]["processed_path"]).parent,
        Path(config["output"]["figures_dir"]),
        Path(config["output"]["metrics_dir"]),
        Path(config["output"]["model_path"]).parent,
    ]
    # Create each folder; parents=True builds missing intermediate folders, exist_ok=True makes reruns safe
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
