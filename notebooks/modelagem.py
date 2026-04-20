# %%
# ─────────────────────────────────────────────────────────────────────────────
# Aula MLOps — Experimentação e Modelagem com MLFlow + Optuna
# ─────────────────────────────────────────────────────────────────────────────
#
# Este script é a QUARTA etapa do pipeline de dados.
# Entrada : data/features/house_price_features.parquet  ← preprocessamento.py
# Saída   : mlruns/          (servidor MLFlow — experimentos, runs, artefatos)
#           outputs/modeling/ (plots PNG salvos localmente antes de logar)
#
# Conceito central: EXPERIMENTAÇÃO RASTREÁVEL
#   • Política  → config/modeling.yaml  (modelos, search spaces, CV, artefatos)
#   • Mecanismo → este script           (laços de treino, Optuna, MLFlow logging)
#
# O que é rastreado no MLFlow:
#   1. Baseline CV    → cada modelo com parâmetros padrão; métricas por fold
#   2. Optuna Trials  → cada combinação de hiperparâmetros (runs aninhados)
#   3. Ensembles      → Stacking e Voting construídos sobre o top-3 individual
#   4. Melhor Modelo  → artefatos completos (6 plots + análise de resíduo)
#   5. Holdout        → métricas finais em dados nunca vistos durante a busca
#
# Modelos testados:
#   Lineares    : LinearRegression, Ridge, Lasso
#   Árvore      : DecisionTreeRegressor
#   Vizinhança  : KNeighborsRegressor
#   Kernel      : SVR  (com subsampling nos trials — O(n²))
#   Ensemble    : RandomForestRegressor, GradientBoostingRegressor
#   Boosting    : XGBRegressor, LGBMRegressor
#   Avançados   : StackingRegressor (top-3), VotingRegressor (top-3)
# ─────────────────────────────────────────────────────────────────────────────

# %%
# Configuração do Ambiente
import sys
import json
import time
from sklearn.metrics import precision_score, recall_score
import yaml
import warnings
import importlib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns


from pathlib import Path
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# Importações — rastreamento e otimização
import optuna
import mlflow
import mlflow.sklearn

# Importações — scikit-learn
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split, learning_curve


# Definições de caminhos — mesmo padrão dos outros walkthroughs
ROOT_DIR   = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / 'config'
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]

for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.modeling import aggregate_fold_metrics, build_pipeline, default_reducer_params, run_cv, suggest_param
matplotlib.use('Agg')           # backend não-interativo: salva em arquivo sem abrir janela
# Importações do projeto
from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.preprocessing import GroupMedianImputer, StandardScalerTransformer
from src.feature_reducer import FeatureReducer
from sklearn.pipeline import Pipeline as SklearnPipeline

# Suprime warnings verbosos de libs externas (XGBoost, LightGBM, sklearn deprecations)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Configuração geral do pipeline
config = load_yaml(CONFIG_DIR / 'pipeline.yaml')
modeling_cfg = load_yaml(CONFIG_DIR / 'modeling.yaml')
preprocessing_cfg = load_yaml(CONFIG_DIR / 'preprocessing.yaml')
config.update(modeling_cfg)  # mescla modeling.yaml no config geral do pipeline
config.update(preprocessing_cfg)  # mescla preprocessing.yaml para acesso unificado

# Cria o logger
log_cfg = config.get('logging')
logger  = get_logger('modelagem', log_cfg)

logger.info('=== Experimentação e Modelagem — MLFlow + Optuna ===')
logger.info('Config carregada: pipeline.yaml + modeling.yaml')

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Configuração do MLFlow
#
# tracking_uri local gera uma pasta mlruns/ no diretório de trabalho.
# Para inspecionar os resultados, execute no terminal:
#   mlflow ui --backend-store-uri mlruns
# e acesse http://localhost:5000
# ─────────────────────────────────────────────────────────────────────────────

# %%
modeling_cfg    = config.get('modeling', {})
tracking_uri    = modeling_cfg.get('tracking_uri', 'mlruns')
experiment_name = modeling_cfg.get('experiment_name', 'wine-quality-experiments')
SEED            = modeling_cfg.get('random_seed', 42)
pipe_cfg        = config.get('pipeline', {})
feature_reduction_config    = config.get('feature_reduction', {})
optuna_cfg      = config.get('optuna', {})
_global_n_trials = optuna_cfg.get('default_trials', 50)

# Path.as_uri() converte o caminho absoluto para file:///E:/... no Windows,
# evitando que o MLFlow interprete a letra do drive (E:) como URI scheme.
print(f"Configuração do MLFlow: tracking_uri={tracking_uri} (resolvido para {Path(tracking_uri).resolve().as_uri()})")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

logger.info('MLFlow tracking URI  : %s', tracking_uri)
logger.info('MLFlow experiment    : %s', experiment_name)
logger.info('Random seed          : %d', SEED)

def objectiveModel(trial ):
    # parâmetros do modelo vindos do search_space do YAML
     with mlflow.start_run(nested=True, run_name=f"{model_name}_trial_{trial.number}") as child_run:
        model_params = {}
        for param_name, spec in (model_cfg.get('search_space') or {}).items():
            model_params[param_name] = suggest_param(trial, param_name, spec)

        # exemplo de otimização do feature reducer se estiver configurado no YAML
        reducer_params = default_reducer_params(_reduction_method, _reduction_method_config)
        if feature_reduction_config.get('search_space'):
            red_search = feature_reduction_config['search_space']
            if 'method' in red_search:
                reducer_params['method'] = suggest_param(trial, 'reducer_method', red_search['method'])
            method = reducer_params['method']
            method_cfg = feature_reduction_config.get(method, {})
            for param_name, spec in (method_cfg.get('search_space') or {}).items():
                reducer_params[param_name] = suggest_param(trial, f"{method}_{param_name}", spec)

        pipeline = build_pipeline(
            model_cfg=model_cfg,
            model_params=model_params,
            reducer_params=reducer_params,
            pipe_cfg=pipe_cfg,
        )

        fold_metrics = run_cv(pipeline, X_train, y_train, cv)
        agg = aggregate_fold_metrics(fold_metrics)

        primary = config.get('metrics', {}).get('primary', 'precision')
        mlflow.log_params(model_params)
        mlflow.log_params({f"reducer_{k}": v for k, v in reducer_params.items()})
        mlflow.set_tag('model_class', f"{model_cfg['module']}.{model_cfg['class']}")
        mlflow.set_tag('reducer_method', _reduction_method)

        # Executa CV (clone() do pipeline garante isolação entre folds)
        fold_metrics = run_cv(pipeline, X_train, y_train, cv)

        # Agrega e loga métricas consolidadas
        agg = aggregate_fold_metrics(fold_metrics)

        mlflow.log_metrics(agg)
        mlflow.log_metric('training_time_s', time.time() - t0)
        trial.set_user_attr("run_id", child_run.info.run_id)

        return agg[f'cv_{primary}_mean']
# %%


# %%
# Inspeciona o que foi carregado do YAML
cv_cfg      = config.get('cv', {})
holdout_cfg = config.get('holdout', {})
models_cfg  = config.get('models', {})
ensembles_cfg = config.get('ensembles', {})
artifacts_cfg = config.get('artifacts', {})

logger.info('CV              : %s (%d folds)', cv_cfg.get('strategy'), cv_cfg.get('n_splits'))
logger.info('Holdout         : %.0f%%', holdout_cfg.get('test_size', 0.2) * 100)
logger.info('Modelos         : %d configurados', len(models_cfg))
enabled_models = [k for k, v in models_cfg.items() if v.get('enabled', True)]
logger.info('  Habilitados   : %s', enabled_models)
logger.info('Imputadores     : %d step(s)', len(pipe_cfg.get('imputation', [])))
logger.info('Scaling cols    : %d', len(pipe_cfg.get('scaling', {}).get('columns', [])))
logger.info('Feature reducer : method=%s', feature_reduction_config.get('method', 'none'))


# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 1 — Carregar Features
# ─────────────────────────────────────────────────────────────────────────────

# %%
features_dir  = ROOT_DIR / config.get('preprocessing', {}).get('output_dir', 'data/features')
features_file = features_dir / config.get('preprocessing', {}).get('output_filename', 'wine_quality.parquet')

logger.info('─' * 60)
logger.info('SEÇÃO 1: Carregar Features')
logger.info('Lendo: %s', features_file)

if not features_file.exists():
    raise FileNotFoundError(
        f"Arquivo de features não encontrado: {features_file}\n"
        "Execute preprocessamento.py antes deste script."
    )


schema = pq.read_schema(str(features_file))
logger.info('Schema (%d colunas):', len(schema))
for field in schema:
    logger.info('  %-35s %s', field.name, field.type)


df = pq.read_table(str(features_file)).to_pandas()
logger.info('Shape: %s', df.shape)

sel_cfg    = config.get('feature_selection', {})
target_col = sel_cfg.get('target', 'isRecommended')

feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].copy()
y = df[target_col]

# XGBoost rejeita nomes de colunas com '[', ']' ou '<' (ex: op_<1H OCEAN).
# Sanitizamos aqui, uma vez, para que todos os modelos downstream usem nomes limpos.
_rename_map = {
    c: c.replace('<', 'lt_').replace('[', '(').replace(']', ')')
    for c in X.columns
    if any(ch in c for ch in ('<', '[', ']'))
}
if _rename_map:
    X = X.rename(columns=_rename_map)
    logger.info('Colunas renomeadas para compatibilidade com XGBoost: %s', _rename_map)

logger.info('Features : %d colunas', len(feature_cols))
logger.info('Target   : %s  (min=%.0f, max=%.0f, média=%.0f)',
            target_col, y.min(), y.max(), y.mean())

logger.info(y.describe())

n_nulls = X.isna().sum().sum()
if n_nulls > 0:
    logger.warning('ATENÇÃO: %d valores nulos encontrados nas features!', n_nulls)
else:
    logger.info('Sem valores nulos nas features ✓')


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 2 — Divisão Treino / Holdout
# ─────────────────────────────────────────────────────────────────────────────

# %%
n_bins    = holdout_cfg.get('stratify_bins', 10)
test_size = holdout_cfg.get('test_size', 0.2)

# Cria bins de quantis do target para estratificação
y_bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')

logger.info('─' * 60)
logger.info('SEÇÃO 2: Divisão Treino / Holdout')
logger.info('Test size: %.0f%%  |  Bins de estratificação: %d', test_size * 100, n_bins)

# %%
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y,
    test_size=test_size,
    random_state=SEED,
    stratify=y_bins,
)

logger.info('Treino  : %d amostras (%.1f%%)', len(X_train), 100 * len(X_train) / len(X))
logger.info('Holdout : %d amostras (%.1f%%)', len(X_holdout), 100 * len(X_holdout) / len(X))
logger.info('Target no treino  — média: %.3f | std: %.3f', y_train.mean(), y_train.std())
logger.info('Target no holdout — média: %.3f | std: %.3f', y_holdout.mean(), y_holdout.std())

# %%
# Verifica que o holdout reflete a distribuição original (sem grandes desvios)
logger.info('Distribuição por quantil (treino vs holdout):')
train_dist   = pd.qcut(y_train,   q=5, labels=False, duplicates='drop').value_counts(normalize=True).sort_index()
holdout_dist = pd.qcut(y_holdout, q=5, labels=False, duplicates='drop').value_counts(normalize=True).sort_index()
dist_df = pd.DataFrame({'treino': train_dist, 'holdout': holdout_dist})
logger.info('\n%s', dist_df.round(3).to_string())

# # %%
# # ─────────────────────────────────────────────────────────────────────────────
# # SEÇÃO 3 — Registro de Modelos e Configuração do Pipeline
# # ─────────────────────────────────────────────────────────────────────────────

# %%
logger.info('─' * 60)
logger.info('SEÇÃO 3: Registro de Modelos e Configuração do Pipeline')

rows = []
for name, cfg in models_cfg.items():
    enabled = cfg.get('enabled', True)
    rows.append({
        'modelo'       : name,
        'habilitado'   : '✓' if enabled else '✗',
        'classe'       : f"{cfg['module']}.{cfg['class']}",
        'optuna_trials': cfg.get('optuna_trials', _global_n_trials),
        'params_default': len(cfg.get('default_params') or {}),
        'search_space' : len(cfg.get('search_space') or {}),
    })

models_table = pd.DataFrame(rows)
logger.info('\n%s', models_table.to_string(index=False))

# %%
# Lê configuração da redução de features e prepara os parâmetros padrão
_reduction_method     = feature_reduction_config.get('method', 'none')
_reduction_method_config = feature_reduction_config.get(_reduction_method, {})



logger.info('Feature reducer: method=%s  params=%s', _reduction_method, default_reducer_params(_reduction_method, _reduction_method_config))

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 4 — Baseline: Cross-Validation com Parâmetros Padrão

# ─────────────────────────────────────────────────────────────────────────────

# %%
# Cria o objeto de CV a partir do config
n_splits = cv_cfg.get('n_splits', 5)
shuffle  = cv_cfg.get('shuffle', True)

cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=SEED)
logger.info('─' * 60)
logger.info('SEÇÃO 4: Baseline CV  (%s, %d folds)', cv_cfg.get('strategy', 'kfold'), n_splits)


all_results: dict[str, dict] = {}

# %%
# Loop de baseline — um MLFlow run por modelo
for model_name, model_cfg in models_cfg.items():
    if not model_cfg.get('enabled', True):
        logger.info('[SKIP] %s (desabilitado)', model_name)
        continue

    logger.info('  [BASELINE] %-25s ...', model_name)
    pipeline = build_pipeline(
        model_cfg   = model_cfg,
        model_params = None,
        reducer_params = default_reducer_params(_reduction_method, _reduction_method_config),
        pipe_cfg    = pipe_cfg,
    )
    t0 = time.time()

    logger.info('[MLFLOW] Iniciando run para modelo: %s', model_name)
    with mlflow.start_run(
        run_name=f'study_{model_name}',
        tags={'stage': 'model_training', 'model': model_name},
    ) as run:
        with mlflow.start_run(nested=True, run_name="default_param") as child_run:
            default_params = {
                str(k): (str(v) if v is None else v)
                for k, v in (model_cfg.get('default_params') or {}).items()
            }
            default_params['reducer_method'] = _reduction_method
            mlflow.log_params(default_params)
            mlflow.set_tag('model_class', f"{model_cfg['module']}.{model_cfg['class']}")
            mlflow.set_tag('reducer_method', _reduction_method)
            logger.info('Reducer params (baseline): %s', default_reducer_params(_reduction_method, _reduction_method_config))
            # Executa CV (clone() do pipeline garante isolação entre folds)
            fold_metrics = run_cv(pipeline, X_train, y_train, cv)

            # Agrega e loga métricas consolidadas
            agg = aggregate_fold_metrics(fold_metrics)
            mlflow.log_metrics(agg)
            mlflow.log_metric('training_time_s', time.time() - t0)

        all_results[model_name] = {
            **agg,
            'fold_metrics'  : fold_metrics,
            'model_cfg'     : model_cfg,
            'best_params'   : dict(model_cfg.get('default_params') or {}),
            'reducer_params': default_reducer_params(_reduction_method, _reduction_method_config),
            'tuned'         : False,
        }
        logger.info(
            '    CV Accuracy: %.2f  | Precision: %.4f |  Recall: %.4f  | F1 %.3f | %.1fs',
            agg['cv_accuracy_mean'],agg['cv_precision_mean'], agg['cv_recall_mean'], agg['cv_f1_mean'], time.time() - t0,
        )
        
        #--------------------------------------------------------
        study = optuna.create_study(
        direction='maximize',
        study_name=f'optuna_{model_name}',
        )   

        n_trials = model_cfg.get('optuna_trials', _global_n_trials)
        study.optimize(objectiveModel, n_trials=n_trials)
        best_params = study.best_params

        method = best_params['reducer_method']
        best_reducer_params = {
        key[len(method) + 1:]: value
        for key, value in best_params.items()
        if key.startswith(f'{method}_')
        }
        logger.info('Best params (Optuna): %s', best_params)
        if method:
            prefix = f'{method}_'
            best_models_params = {
                k: v
                for k, v in best_params.items()
                if k != 'reducer_method' and not k.startswith(prefix)
            }


        mlflow.log_params(best_models_params)
        mlflow.log_params({f"reducer_{k}": v for k, v in best_reducer_params.items()})
        mlflow.set_tag('model_class', f"{model_cfg['module']}.{model_cfg['class']}")
        mlflow.set_tag('reducer_method', _reduction_method)
        best_pipeline = build_pipeline(
            model_cfg=model_cfg,
            model_params=best_models_params,
            reducer_params={'method': method, **best_reducer_params},
            pipe_cfg=pipe_cfg,
        )
        t0 = time.time()
        fold_metrics = run_cv(best_pipeline, X_train, y_train, cv)

        # Agrega e loga métricas consolidadas
        agg = aggregate_fold_metrics(fold_metrics)
        mlflow.log_metrics(agg)
        mlflow.log_metric('training_time_s', time.time() - t0)

        # Compara métricas do modelo tunado vs baseline (parâmetros padrão)
        baseline_agg = {k: v for k, v in all_results[model_name].items() if k in agg}
        delta_metrics = {f'delta_{k}': agg[k] - baseline_agg[k] for k in baseline_agg}
        mlflow.log_metrics(delta_metrics)
        logger.info('Comparação tunado vs baseline (%s):', model_name)
        for metric_key, delta_val in delta_metrics.items():
            base_val = baseline_agg[metric_key.replace('delta_', '')]
            tuned_val = agg[metric_key.replace('delta_', '')]
            logger.info('  %-35s baseline=%.4f  tunado=%.4f  delta=%+.4f', metric_key, base_val, tuned_val, delta_val)

        best_pipeline.fit(X_train, y_train)
        y_pred = best_pipeline.predict(X_holdout)
        logger.info('Best Model: ')
        logger.info(
            '    CV Accuracy: %.2f  | Precision: %.4f |  Recall: %.4f  | F1 %.3f | %.1fs',
            agg['cv_accuracy_mean'], agg['cv_precision_mean'], agg['cv_recall_mean'], agg['cv_f1_mean'], time.time() - t0,
        )

        logger.info('Holdout precision: %.4f', precision_score(y_holdout, y_pred, average='macro', zero_division=0))
        # Salvar o melhor modelo global
        primary_metric = config.get('metrics', {}).get('primary', 'precision')
        score = agg.get(f'cv_{primary_metric}_mean', None)
        if score is not None:
            if 'best_global_score' not in globals() or best_global_score is None or score > best_global_score:
                best_global_score = score
                best_global_pipeline = best_pipeline
                best_global_model_name = model_name
                best_global_params = best_models_params.copy()
                best_run_id = child_run.info.run_id  
        if best_run_id := study.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id)
        mlflow.end_run()

# Após o loop, registrar o melhor modelo global
if 'best_global_pipeline' in globals() and best_global_pipeline is not None:
    logger.info(f'Registrando o melhor modelo global: {best_global_model_name} (score={best_global_score:.4f})')
    with mlflow.start_run(run_name="best_model_registration") as registration_run:
        mlflow.set_tag("origin_model", best_global_model_name)
        mlflow.set_tag("origin_run_id", best_run_id)
        mlflow.log_param("best_global_score", best_global_score)
        mlflow.sklearn.log_model(
            best_global_pipeline,
            artifact_path="best_model",
            registered_model_name="best_optuna_model"
        )
        logger.info('Modelo global registrado no MLflow (run_id=%s, origin_run_id=%s).', registration_run.info.run_id, best_run_id)
