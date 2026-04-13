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
downloaded = download_dataset(
    dataset=data_config["kaggle"]["dataset"],
    expected_files=data_config["kaggle"]["expected_files"],
    destination_dir=raw_dir,
    skip_if_exists=pipeline_config["execution"]["skip_download_if_exists"],
    logging_config=logger_config
)
logger.info("Dataset Baixado com Sucesso!")



logger.info('Arquivos prontos: %d', len(downloaded))

# verificar o conteúdo do diretório raw
for f in sorted(raw_dir.glob('*.csv')):
    logger.info('  %s (%.1f KB)', f.name, f.stat().st_size / 1024)

# %%
# definir o caminho de saída do Parquet
processed_dir = ROOT_DIR / pipeline_config['paths']['processed_data_dir']
output_path = processed_dir / pipeline_config['paths']['output_filename']

logger.info('Saída: %s', output_path)

# obtendo configurações de processamento
compression = data_config.get('ingest').get('compression', 'snappy')
chunk_size = data_config.get('ingest').get('chunk_size_rows', 50_000)
validate = data_config.get('ingest').get('validate_schema')
required_cols = data_config.get('schema').get('required_columns')
skip_ingest = pipeline_config.get('execution').get('skip_ingest_if_exists')
force_ingest = pipeline_config.get('execution').get('force_ingest')

result_path = ingest_csv_to_parquet(
    raw_dir=raw_dir,
    output_path=output_path,
    compression=compression,
    chunk_size_rows=chunk_size,
    validate_schema=validate,
    skip_if_exists=skip_ingest,
    logging_config=logger_config
)