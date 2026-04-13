# %%
# ─────────────────────────────────────────────────────────────────────────────
# Aula MLOps — Pré-processamento e Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
#
# Este script é a TERCEIRA etapa do pipeline de dados.
# Entrada : data/processed/house_price.parquet  ← gerado por qualidade_walkthrough.py
# Saída   : data/features/house_price_features.parquet
#
# Conceito central: SEPARAÇÃO entre política e mecanismo
#   • Política  → config/preprocessing.yaml  (O QUÊ transformar e com quais parâmetros)
#   • Mecanismo → src/preprocessing.py       (COMO executar cada transformação)
#
# Para ajustar qualquer transformação (ex: mudar threshold de flag, adicionar
# uma nova feature de razão), edite apenas o YAML. O código não muda.
#
# ─────────────────────────────────────────────────────────────────────────────
# Configuração do Ambiente
import sys
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq


# Definições de caminhos — mesmo padrão dos outros walkthroughs
ROOT_DIR   = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / 'config'
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]
prep_yaml_path = CONFIG_DIR / 'preprocessing.yaml'

for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml

# ── Importa todos os transformadores do módulo src/preprocessing.py ──────────
from src.preprocessing import (
    LogTransformer,
)

# %%
config = load_yaml(CONFIG_DIR / 'pipeline.yaml')
preprocessing = load_yaml(prep_yaml_path)
config.update(preprocessing)  # Mescla as configs (pipeline + preprocessing)

# Configura o logger
log_config = config.get('logging')
logger = get_logger(
    name='preprocessamento',
    logging_config=log_config
)

logger.info('=== Pré-processamento e Feature Engineering ===')
logger.info('Config carregada: pipeline.yaml + data.yaml + preprocessing.yaml')

# ─────────────────────────────────────────────────────────────────────────────
# Inspeciona a configuração carregada
#
# Ponto de ensino: o aluno vê exatamente o que foi lido do YAML.
# Qualquer mudança no preprocessing.yaml aparece aqui sem alterar o código.
# ─────────────────────────────────────────────────────────────────────────────

# %%
prep_config    = config.get('preprocessing', {})
output_dir  = ROOT_DIR / prep_config.get('output_dir', 'data/features')
output_path = output_dir / prep_config.get('output_filename', 'wine-quality-features.parquet')
compression = prep_config.get('compression', 'snappy')

logger.info('Saída      : %s', output_path)
logger.info('Compressão : %s', compression)

# %%
# Lista as transformações configuradas no YAML
logger.info('Imputações configuradas : %d', len(config.get('imputation', [])))

# ─────────────────────────────────────────────────────────────────────────────
# Inspeciona a configuração carregada
#
# Ponto de ensino: o aluno vê exatamente o que foi lido do YAML.
# Qualquer mudança no preprocessing.yaml aparece aqui sem alterar o código.
# ─────────────────────────────────────────────────────────────────────────────

# %%
prep_config    = config.get('preprocessing', {})
output_dir  = ROOT_DIR / prep_config.get('output_dir', 'data/features')
output_path = output_dir / prep_config.get('output_filename', 'wine_quality_features.parquet')
compression = prep_config.get('compression', 'snappy')

logger.info('Saída      : %s', output_path)
logger.info('Compressão : %s', compression)

# %%
# Lista as transformações configuradas no YAML
logger.info('Imputações configuradas : %d', len(config.get('imputation', [])))
logger.info('Features de razão       : %d', len(config.get('ratio_features', [])))
logger.info('Colunas log1p           : %d', len(config.get('log_transform', {}).get('columns', [])))
logger.info('Features polinomiais    : %d', len(config.get('polynomial_features', [])))
logger.info('Configurações de encoding: %s', config.get('categorical_encoding', {}))
# logger.info('Features para seleção    : %d', len(config.get('feature_selection', {}).get('features_to_keep', []))) TODO

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 1 — Carregamento do Dataset
# ─────────────────────────────────────────────────────────────────────────────

# %%
# Caminho do Parquet gerado pela etapa de ingestão
processed_dir = ROOT_DIR / config['paths']['processed_data_dir']
parquet_path  = processed_dir / config['paths']['output_filename']

logger.info('Lendo: %s', parquet_path)

if not parquet_path.exists():
    raise FileNotFoundError(
        f"Arquivo Parquet não encontrado: {parquet_path}\n"
        "Execute ingestao.py e qualidade.py antes deste script."
    )

# %%
# Inspeciona o schema sem carregar os dados (leitura de metadados é barata)
schema = pq.read_schema(str(parquet_path))
logger.info('Schema original (%d colunas):', len(schema))
for field in schema:
    logger.info('  %-25s %s', field.name, field.type)

# %%
# Carrega o DataFrame completo
df = pq.read_table(str(parquet_path)).to_pandas()
logger.info('Shape original: %s', df.shape)

# %%
# Visão rápida antes das transformações
logger.info(df.head())

logger.info("Transformando type em isWhite")

df["isWhite"] = (df["type"] == "white").astype(int)
df.drop(columns=["type"], inplace=True)

# Estatísticas descritivas e nulos — baseline antes do pré-processamento
logger.info('Valores ausentes por coluna:')
for col, n_null in df.isna().sum()[df.isna().sum() > 0].items():
    logger.info('  %-25s %d (%.2f%%)', col, n_null, 100 * n_null / len(df))

# %%
logger.info(df.describe())

logger.info("Limpando os nomes das colunas, tirando espaco por _")
df.columns = df.columns.str.replace(' ', '_')
logger.info(df.describe())


# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 4 — Features de Razão TODO
#
# Totais absolutos dependem do tamanho do bloco censitário:
#   - Um bairro com 10.000 cômodos e 2.000 domicílios é diferente de um com
#     100 cômodos e 20 domicílios — mas ambos têm 5 cômodos/domicílio.
#
# EDA mostrou que:
#   bedrooms_per_room       r = -0.256 (vs total_bedrooms r = +0.050)
#   rooms_per_household     r = +0.152 (vs total_rooms    r = +0.134)
#   population_per_household r = -0.247 (vs population    r = -0.025)
#
# Razões são MUITO mais informativas que totais absolutos!
# ─────────────────────────────────────────────────────────────────────────────

# # %%
# ratio_cfg = config.get('ratio_features', [])
# logger.info('─' * 60)
# logger.info('SEÇÃO 4: Features de Razão')

# # %%
# ratio_transformer = RatioFeatureTransformer(ratio_cfg, logger=logger)
# df = ratio_transformer.transform(df)

# # %%
# # Estatísticas das novas features
# new_ratio_cols = [spec['name'] for spec in ratio_cfg]
# logger.info(df[new_ratio_cols].describe())

# # %%
# # Correlação com o target — confirma que razões são mais informativas
# logger.info('Correlação das razões com median_house_value:')
# for col in new_ratio_cols:
#     corr = df[col].corr(df['median_house_value'])
#     logger.info('  %-30s r = %.3f', col, corr)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 5 — Transformação Logarítmica (log1p)
#
# Features com skewness > 1.5 têm cauda longa que distorce modelos lineares
# e aumenta a influência de outliers extremos.
#
# log1p(x) = log(1+x):
#   - Seguro para x=0 (evita log(0) = -Inf)
#   - Comprime a cauda direita, reduzindo a assimetria
#   - Melhora a linearidade com o target
#
# Colunas originais são mantidas para referência.
# Colunas transformadas recebem prefixo 'log_'.
# ─────────────────────────────────────────────────────────────────────────────

# %%
log_cols = config.get('log_transform', {}).get('columns', [])
logger.info('─' * 60)
logger.info('SEÇÃO 5: Transformação Logarítmica (log1p)')

# %%
log_transformer = LogTransformer(log_cols, logger=logger)
df = log_transformer.transform(df)

# %%
# Comparação de skewness: antes vs depois
logger.info('Comparação de assimetria (skewness):')
for col in log_cols:
    if col in df.columns:
        log_col = f'log_{col}'
        skew_raw = df[col].dropna().skew()
        skew_log = df[log_col].dropna().skew() if log_col in df.columns else float('nan')
        logger.info('  %-30s  raw: %+.2f  → log: %+.2f', col, skew_raw, skew_log)

# %%
# Colunas log criadas
log_created = [f'log_{c}' for c in log_cols if f'log_{c}' in df.columns]
df[log_created].head()


# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 7 — Features Polinomiais e Interações
#
# A relação entre renda e preço não é perfeitamente linear:
#   - Renda alta tem retorno decrescente sobre o preço → median_income²
#
# Interação income × age captura o efeito de bairros ricos E antigos:
#   - Bairros históricos no Bay Area têm prêmio tanto de renda quanto de idade
#   - EDA: median_income_x_housing_median_age tem r = +0.589 com o target!
#   - Esse é um dos features engineered mais preditivos do dataset.
# ─────────────────────────────────────────────────────────────────────────────

# %%
#TODO
# poly_cfg = config.get('polynomial_features', [])
# logger.info('─' * 60)
# logger.info('SEÇÃO 7: Features Polinomiais e Interações')

# # %%
# poly_transformer = PolynomialFeatureTransformer(poly_cfg, logger=logger)
# df = poly_transformer.transform(df)

# # %%
# # Correlação das features polinomiais com o target
# poly_cols = [spec['name'] for spec in poly_cfg]
# logger.info('Correlação das features polinomiais com median_house_value:')
# for col in poly_cols:
#     if col in df.columns:
#         corr = df[col].corr(df['median_house_value'])
#         logger.info('  %-40s r = %.3f', col, corr)

# # %%
# logger.info(df[poly_cols].describe())

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 11 — Persistência do Resultado
#
# Salva o dataset processado em Parquet (formato colunar, comprimido).
# Este arquivo é a ENTRADA da etapa de modelagem.
#
# O relatório pode ser:
#   • Versionado no repositório (rastreabilidade de dados)
#   • Carregado diretamente pelo script de treinamento do modelo
#   • Comparado entre execuções para detectar drift de features
# ─────────────────────────────────────────────────────────────────────────────

# %%
# Cria o diretório de saída se não existir
output_dir.mkdir(parents=True, exist_ok=True)
logger.info('─' * 60)
logger.info('SEÇÃO 11: Persistência')
logger.info('Diretório de saída: %s', output_dir)

# %%
# Salva em Parquet
df.to_parquet(str(output_path), compression=compression, index=False)
size_mb = output_path.stat().st_size / (1024 ** 2)
logger.info('Arquivo salvo: %s (%.2f MB)', output_path, size_mb)

# %%
# Validação pós-escrita: lê o schema sem carregar os dados
schema_out = pq.read_schema(str(output_path))
logger.info('Schema de saída (%d colunas):', len(schema_out))
for field in schema_out:
    logger.info('  %-35s %s', field.name, field.type)

# %%
# Lê uma amostra para confirmação visual
df_check = pd.read_parquet(str(output_path))
logger.info('Verificação pós-leitura — shape: %s', df_check.shape)
logger.info(df_check.head())