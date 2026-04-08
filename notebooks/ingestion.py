import sys
from pathlib import Path
# carregar as configs
ROOT_DIR = Path(__file__).resolve().parent.parent
SECRETS_PATH = ROOT_DIR / 'secrets.env'
CONFIG_DIR = ROOT_DIR / 'config'
PIPELINE_CONFIG = CONFIG_DIR / 'pipeline.yaml'
DATA_CONFIG = CONFIG_DIR / 'data.yaml'
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]

# adicionar o root e o config no meu path do sistema
for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.insert(0, _p)

data_config = load_yaml(DATA_CONFIG)

rootDir = load_config.get("rootDir", "data")
print(f"Root directory: {rootDir}")
# verificar se tem as credencias
# verificar se tem algum arquivo faltando
# baixar os datasets
df_1 = download_data(rootDir, "dataset_1.csv")
df_2 = download_data(rootDir, "dataset_2.csv")

merged_df = merge_dataframes(df_1, df_2)

# juntar os dataset
# salvar os dataset raw em uma pasta
# salvar raw como parquet (sera q vale a pena?)

#####################################################
# Funcoes Auxiliares
import yaml
from pathlib import Path
from typing import Any

def load_yaml(path: Path) -> dict[str, Any]:
    """carrega os yamls
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            f"Expected location: {path.resolve()}"
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}

import logging
import sys
from pathlib import Path
from typing import Any

def get_logger(name: str, logging_config: dict[str, Any]) -> logging.Logger:
    """
    Build and return a configured logger.

    Args:
        name:           Module __name__ — becomes the logger hierarchy name.
        logging_config: The 'logging' section from pipeline.yaml.

    Returns:
        Configured logging.Logger instance.

    Note:
        Calling this multiple times with the same name is safe — Python's
        logging module returns the same logger instance and handlers are
        checked for duplicates before adding.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level_str = logging_config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    fmt = logging_config.get(
        "format", "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    datefmt = logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S")
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler — always added
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — added only when log_to_file is True
    if logging_config.get("log_to_file", False):
        log_file = logging_config.get("log_file", "pipeline.log")
        _add_file_handler(logger, log_file, formatter, level)

    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False

    return logger