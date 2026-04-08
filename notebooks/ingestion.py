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

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.ingestion import ingest_csv_to_parquet
from src.downloader import check_kaggle_credentials, list_remote_files, download_dataset


data_config = load_yaml(DATA_CONFIG)
pipeline_config = load_yaml(PIPELINE_CONFIG)
logger_config = pipeline_config.get("logging", {})

raw_dir = ROOT_DIR / pipeline_config.get('paths').get('raw_data_dir')
raw_dir.mkdir(parents=True, exist_ok=True)

rootDir = data_config.get("paths", "raw_data_dir")
logger = get_logger(
    name='ingestao',
    logging_config=logger_config
)
logger.info(f"Root directory: {rootDir}")
logger.info("NOTEBOOK INGESTAO")
logger.info("Verificando credenciais kaggle")

if check_kaggle_credentials(SECRETS_PATH):
    logger.info("Credenciais encontradas. Pronto para baixar os datasets.")
else:
    logger.error("Credenciais do Kaggle não encontradas. Verifique o arquivo secrets.env.")

logger.info( "Baixando Dataset...")
download_dataset(
    dataset=data_config["kaggle"]["dataset"],
    expected_files=data_config["kaggle"]["expected_files"],
    destination_dir=raw_dir,
    skip_if_exists=True,
    force=False,
    logging_config=logger_config
)
logger.info("Dataset Baixado com Sucesso!")
