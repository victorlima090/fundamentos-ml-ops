from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Configuração do Ambiente
import sys
import json
import time
from sklearn.preprocessing import RobustScaler
import yaml
import warnings
import importlib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
matplotlib.use('Agg')           # backend não-interativo: salva em arquivo sem abrir janela
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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor, VotingRegressor


from src.preprocessing import GroupMedianImputer, StandardScalerTransformer
from src.feature_reducer import FeatureReducer
from sklearn.pipeline import Pipeline as SklearnPipeline


# %%
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas de classificação usadas neste pipeline."""
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(
        precision_score(y_true, y_pred, average='macro', zero_division=0)
    )
    recall = float(
        recall_score(y_true, y_pred, average='macro', zero_division=0)
    )
    f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# %%
def _run_cv(model, X: pd.DataFrame, y: pd.Series, cv: KFold) -> list[dict]:
    """
    Executa Cross-Validation e retorna métricas por fold.

    Clona o modelo em cada fold para evitar contaminação de estado entre folds.
    Retorna lista de dicts: [{fold, accuracy, precision, recall, f1}, ...].
    """
    fold_metrics = []
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = m.predict(X.iloc[val_idx])
        metrics = _compute_metrics(y.iloc[val_idx].values, y_pred)
        metrics['fold'] = fold_i + 1
        fold_metrics.append(metrics)
    return fold_metrics

# %%
def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Agrega métricas de todos os folds em média ± desvio padrão."""
    df = pd.DataFrame(fold_metrics)
    result = {}
    for col in ['accuracy', 'precision', 'recall', 'f1']:
        result[f'cv_{col}_mean'] = float(df[col].mean())
        result[f'cv_{col}_std']  = float(df[col].std())
    return result

# %%
def _suggest_param(trial: optuna.Trial, name: str, spec: dict):
    """
    Constrói uma sugestão Optuna a partir de um spec do search_space do YAML.

    Tipos suportados:
        log_float  → suggest_float(..., log=True)
        float      → suggest_float(...)
        int        → suggest_int(...)
        categorical→ suggest_categorical(...)
    """
    ptype = spec['type']
    # Garante tipos numéricos — PyYAML pode parsear notação científica (1.0e-4)
    # como string em algumas versões; float()/int() normalizam sem custo.
    if ptype == 'log_float':
        return trial.suggest_float(name, float(spec['low']), float(spec['high']), log=True)
    elif ptype == 'float':
        return trial.suggest_float(name, float(spec['low']), float(spec['high']))
    elif ptype == 'int':
        return trial.suggest_int(name, int(spec['low']), int(spec['high']))
    elif ptype == 'categorical':
        return trial.suggest_categorical(name, spec['choices'])
    else:
        raise ValueError(f'Tipo de search_space desconhecido: {ptype!r}')

# %%
def _build_model(model_cfg: dict, extra_params: dict | None = None):
    """
    Instancia um modelo usando importlib a partir do config (module + class).

    Mescla default_params com extra_params (extra_params sobrescreve o default).
    Permite instanciar qualquer modelo sklearn-compatível sem hardcode.
    """
    module    = importlib.import_module(model_cfg['module'])
    cls       = getattr(module, model_cfg['class'])
    params    = dict(model_cfg.get('default_params') or {})
    if extra_params:
        params.update(extra_params)
    return cls(**params)

# %%
def _build_pipeline(
    model_cfg: dict,
    model_params: dict | None,
    reducer_params: dict | None,
    pipe_cfg: dict,
) -> SklearnPipeline:
    """
    Constrói um sklearn Pipeline leak-free para um modelo:

        GroupMedianImputer(s)       ← um por entrada em pipe_cfg['imputation']
        StandardScalerTransformer   ← colunas de pipe_cfg['scaling']['columns']
        FeatureReducer              ← method e params de reducer_params
        estimator                   ← instanciado via _build_model

    Por que usar Pipeline?
    - fit() em cada fold de CV chama fit() em TODOS os steps, usando apenas
      os índices de treino daquele fold.  Nenhum dado de validação/holdout
      vaza para o imputador ou scaler.
    - clone() (usado em _run_cv) preserva os hiperparâmetros sem o estado
      aprendido, garantindo isolação entre folds.

    Parâmetros
    ----------
    model_cfg      : dict do models section em modeling.yaml
    model_params   : parâmetros extras do Optuna (sobrescrevem default_params)
    reducer_params : parâmetros para FeatureReducer (method + kwargs do método ativo)
    pipe_cfg       : dict de pipeline section em modeling.yaml
    """
    steps = []

    # ── Imputação (stateful — aprende medianas só no treino) ──────────────────
    #TODO ver se ainda precisa, axo q nao
    # for imp_spec in pipe_cfg.get('imputation', []):
    #     step_name = f"imputer_{imp_spec['column'].replace('/', '_')}"
    #     steps.append((
    #         step_name,
    #         GroupMedianImputer(
    #             group_col=imp_spec['group_by'],
    #             target_col=imp_spec['column'],
    #         ),
    #     ))
    # TODO ver se precisa disso
    # # ── Escalonamento (stateful — aprende μ/σ só no treino) ──────────────────
    # scale_cols = pipe_cfg.get('scaling', {}).get('columns', [])
    # if scale_cols:
    #     steps.append(('scaler', StandardScalerTransformer(columns=scale_cols)))
    steps.append(('robustScaler', RobustScaler()))

    # ── Redução de features (opcional, tunable pelo Optuna) ──────────────────
    reducer_kw = reducer_params or {}
    steps.append(('reducer', FeatureReducer(**reducer_kw)))

    # ── Estimador final ───────────────────────────────────────────────────────
    steps.append(('estimator', _build_model(model_cfg, model_params)))

    return SklearnPipeline(steps)

# %%
def _get_feature_importance(
    model, feature_names: list[str],
    X_val: pd.DataFrame, y_val: pd.Series,
) -> pd.Series:
    """
    Extrai importância de features do modelo treinado.

    Suporta sklearn Pipeline: extrai o estimador final via named_steps['estimator']
    e usa os nomes de features pós-redução de named_steps['reducer'] quando
    disponível.

    Prioridade:
        1. feature_importances_ (árvores, ensembles baseados em árvores)
        2. coef_ (modelos lineares — usa valor absoluto)
        3. permutation_importance (fallback model-agnóstico: SVR, KNN, ensembles mistos)
    """
    # Desempacota Pipeline para obter estimador + nomes de features pós-redução
    if isinstance(model, SklearnPipeline):
        estimator = model.named_steps['estimator']
        reducer   = model.named_steps.get('reducer')
        if reducer is not None and reducer.selected_features is not None:
            # RFE mantém nomes originais; PCA/kPCA usa 'pc_0', 'pc_1', ...
            imp_feature_names = reducer.selected_features
        else:
            imp_feature_names = feature_names
    else:
        estimator = model
        imp_feature_names = feature_names

    if hasattr(estimator, 'feature_importances_'):
        return pd.Series(estimator.feature_importances_, index=imp_feature_names)
    elif hasattr(estimator, 'coef_'):
        coef = np.abs(estimator.coef_)
        if coef.ndim > 1:
            coef = coef.flatten()
        return pd.Series(coef, index=imp_feature_names)
    else:
        # Permutation importance sobre o PIPELINE completo (inclui transformações)
        # — usa X_val original para que o pipeline transforme consistentemente
        sample_size = min(2000, len(X_val))
        idx = np.random.default_rng(42).choice(len(X_val), sample_size, replace=False)
        r = permutation_importance(
            model, X_val.iloc[idx], y_val.iloc[idx],
            n_repeats=5, random_state=42, n_jobs=-1,
        )
        return pd.Series(r.importances_mean, index=feature_names)