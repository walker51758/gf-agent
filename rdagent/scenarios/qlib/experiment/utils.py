"""Describe the factor data folder for LLM prompts."""

import re
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS


def get_file_desc(path: Path) -> str:
    template = Environment(undefined=StrictUndefined).from_string(
        "# {{ name }}\n\n## File Type\n{{ kind }}\n\n## Content Overview\n{{ content }}"
    )
    if path.suffix == ".md":
        return template.render(name=path.name, kind="Markdown Documentation", content=path.read_text(encoding="utf-8"))
    if path.suffix == ".h5":
        frame = pd.read_hdf(path)
        content = f"Index: {frame.index.names}\n\nColumns:\n{frame.dtypes.to_string()}"
        return template.render(name=path.name, kind="HDF5 Data File", content=content)
    raise NotImplementedError(f"Unsupported factor data file: {path.name}")


def get_data_folder_intro(fname_reg: str = ".*", flags: int = 0, variable_mapping=None) -> str:
    del variable_mapping
    folder = Path(FACTOR_COSTEER_SETTINGS.data_folder_debug)
    folder.mkdir(parents=True, exist_ok=True)
    descriptions = [get_file_desc(path) for path in folder.iterdir() if re.match(fname_reg, path.name, flags)]
    return "\n----------------- file splitter -------------\n".join(descriptions)
