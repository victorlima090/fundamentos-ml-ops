"""
pages/1_Predicao.py — Interface Streamlit: predição de recomendação de vinho.

O usuário insere as features químicas base e o tipo do vinho (red/white).
Ao submeter:
    1. As features derivadas (razões e logs) são calculadas localmente.
    2. O modelo é carregado diretamente do banco SQLite do MLflow (sem servidor REST).
    3. A classe prevista é exibida como recomendação (0 = não recomendado, 1 = recomendado).
"""
from __future__ import annotations

import sys
from pathlib import Path
import math

import streamlit as st
import pandas as pd



# ── Bootstrap de paths ────────────────────────────────────────────────────────
_PAGE_DIR     = Path(__file__).resolve().parent   # pages/
_APP_DIR      = _PAGE_DIR.parent                  # production_app/
_PROJECT_ROOT = _APP_DIR.parent                   # demo_projeto/

for _p in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.config_loader import load_yaml
from src.utils.logger import get_logger
# Após montar features_df, comparar com uma linha real do parquet
import pandas as pd
from production_app.utils.pipeline_utils import obter_parquet_features

from src.preprocessing import FeatureSelector, LogTransformer, RatioFeatureTransformer
from utils.pipeline_utils import obter_nomes_colunas_features, preprocessar_entradas
from utils.model_utils import (
    carregar_modelo,
    prever_individual,
)
ROOT_DIR   = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT_DIR / 'config'
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]
prep_yaml_path = CONFIG_DIR / 'preprocessing.yaml'

config = load_yaml(CONFIG_DIR / 'pipeline.yaml')
preprocessing = load_yaml(prep_yaml_path)
config.update(preprocessing)  # Mescla as configs (pipeline + preprocessing)
ratio_cfg = config.get('ratio_features', [])

# Configura o logger
log_config = config.get('logging')
logger = get_logger(
    name='preprocessamento',
    logging_config=log_config
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predição de Recomendação de Vinho",
    page_icon="🍷",
    layout="wide",
)

st.title("Predição de Recomendação de Vinho")
st.markdown(
    """
    Informe as features físico-químicas do vinho abaixo.
    A aplicação calcula as **features derivadas** (razões e logs),
    alinha com as colunas esperadas no treino e carrega o modelo
    diretamente do **banco SQLite do MLflow**.
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configuração do banco MLflow
# ─────────────────────────────────────────────────────────────────────────────
_MLFLOW_DB_PATH = (_PROJECT_ROOT / "mlruns.db").resolve()
_URI_PADRAO = f"sqlite:///{_MLFLOW_DB_PATH.as_posix()}"

with st.sidebar:
    st.header("⚙️ Configurações MLflow")
    db_uri = st.text_input(
        "URI do banco SQLite",
        value=_URI_PADRAO,
        help=(
            "URI do banco SQLite gerado por modelagem.py.\n"
            "Exemplos:\n"
            f"  {_URI_PADRAO}\n"
            "  sqlite:////caminho/absoluto/mlruns.db"
        ),
    )

    st.divider()
    st.markdown(
        """
        **Como gerar o banco:**
        ```bash
        cd fundamentos-ml-ops
        python notebooks/ingestion.py
        python notebooks/preprocessing.py
        python notebooks/modelagem.py
        ```
        Depois execute a aplicação:
        ```bash
        streamlit run production_app/app.py
        ```
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
# Cache do modelo (recarregado apenas quando o URI muda)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Carregando modelo do MLflow...")
def _modelo_em_cache(uri: str):
    """Carrega e armazena em cache o modelo MLflow para evitar recarregamentos."""
    return carregar_modelo(uri)


# ─────────────────────────────────────────────────────────────────────────────
# Defaults: linha do parquet com maior probabilidade entre isRecommended == 1
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "fixed_acidity": 7.0, "volatile_acidity": 0.30, "citric_acid": 0.35,
    "residual_sugar": 5.0, "chlorides": 0.045, "free_sulfur_dioxide": 30.0,
    "total_sulfur_dioxide": 115.0, "density": 0.996, "pH": 3.20,
    "sulphates": 0.50, "alcohol": 10.5, "isWhite": 1,
}
try:
    _df_pq = obter_parquet_features()
    _cls1 = _df_pq[_df_pq["isRecommended"] == 1].drop(columns=["isRecommended"])
    if not _cls1.empty:
        _probs_init = _modelo_em_cache(db_uri).predict_proba(_cls1)[:, 1]
        _melhor_init = _cls1.iloc[_probs_init.argmax()]
        for _c in list(_DEFAULTS.keys()):
            if _c in _melhor_init.index:
                _DEFAULTS[_c] = float(_melhor_init[_c])
except Exception:
    pass  # mantém os defaults fixos se o modelo ainda não estiver treinado


# ─────────────────────────────────────────────────────────────────────────────
# Formulário de entrada — features base de vinho
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Features do vinho")
st.caption(
    "As features derivadas (razões e logs) são calculadas automaticamente."
)

col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.number_input(
        "Fixed acidity",
        min_value=0.0,
        max_value=None,
        value=_DEFAULTS["fixed_acidity"],
        step=0.1,
        format="%.2f",
    )
    volatile_acidity = st.number_input(
        "Volatile acidity",
        min_value=0.0,
        max_value=None,
        value=_DEFAULTS["volatile_acidity"],
        step=0.01,
        format="%.3f",
    )
    citric_acid = st.number_input(
        "Citric acid",
        min_value=0.0,
        max_value=None,
        value=_DEFAULTS["citric_acid"],
        step=0.01,
        format="%.3f",
    )
    wine_type = st.selectbox(
        "Tipo",
        options=["red", "white"],
        index=0 if _DEFAULTS["isWhite"] == 0 else 1,
    )

with col2:
    residual_sugar = st.number_input(
        "Residual sugar",
        min_value=0.0,
        max_value=None,
        value=_DEFAULTS["residual_sugar"],
        step=0.1,
        format="%.2f",
    )
    chlorides = st.number_input(
        "Chlorides",
        min_value=0.0,
        max_value=None,
        value=_DEFAULTS["chlorides"],
        step=0.001,
        format="%.4f",
    )
    free_sulfur_dioxide = st.number_input(
        "Free sulfur dioxide",
        min_value=0.0,
        max_value=None,
        value=_DEFAULTS["free_sulfur_dioxide"],
        step=1.0,
        format="%.1f",
    )
    total_sulfur_dioxide = st.number_input(
        "Total sulfur dioxide",
        min_value=0.0,
        max_value=None,
        value=_DEFAULTS["total_sulfur_dioxide"],
        step=1.0,
        format="%.1f",
    )

with col3:
    density = st.number_input(
        "Density",
        min_value=0.0,
        max_value=None,
        value=_DEFAULTS["density"],
        step=0.0001,
        format="%.4f",
    )
    ph = st.number_input(
        "pH",
        min_value=0.0,
        max_value=None,
        value=_DEFAULTS["pH"],
        step=0.01,
        format="%.2f",
    )
    sulphates = st.number_input(
        "Sulphates",
        min_value=0.0,
        max_value=None,
        value=_DEFAULTS["sulphates"],
        step=0.01,
        format="%.2f",
    )
    alcohol = st.number_input(
        "Alcohol",
        min_value=0.0,
        max_value=None,
        value=_DEFAULTS["alcohol"],
        step=0.1,
        format="%.2f",
    )


def _safe_div(a: float, b: float) -> float:
    """Divisão segura para evitar inf quando o denominador é zero."""
    if math.isclose(b, 0.0, abs_tol=1e-12):
        return 0.0
    return float(a / b)


def _montar_features_modelo() -> pd.DataFrame:
    """Monta um DataFrame com as colunas esperadas pelo modelo de classificação."""
    base = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": ph,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "isWhite": 1 if wine_type == "white" else 0,
    }
    print("Features base para inferência:", base)
    df = preprocessar_entradas(base)
    # ratio_transformer = RatioFeatureTransformer(ratio_cfg, logger=logger)
    # df = ratio_transformer.transform(df)
    
    # log_cols = config.get('log_transform', {}).get('columns', [])


    # log_transformer = LogTransformer(log_cols, logger=logger)
    # df = log_transformer.transform(df)

    # sel_cfg = config.get('feature_selection', {})
    # features_to_keep = sel_cfg.get('features_to_keep', [])
    # features_to_keep = features_to_keep + [sel_cfg.get('target', 'isRecommended')]  # Garante que o target é mantido

    # selector = FeatureSelector(features_to_keep, logger=logger)
    # df = selector.fit_transform(df)

    # colunas_modelo = obter_nomes_colunas_features()
    # return pd.DataFrame([df]).reindex(columns=colunas_modelo, fill_value=0.0)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Predição
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
btn_prever = st.button(
    "🔮 Calcular Recomendação",
    type="primary",
    use_container_width=True,
)

if btn_prever:
    # ── Passo 1: montagem das features para inferência ───────────────────────
    with st.spinner("Preparando features para inferência..."):
        try:
            features_df = _montar_features_modelo()
            # Garante que a ordem das colunas é idêntica à do parquet de treino
            from utils.pipeline_utils import obter_parquet_features as _get_pq
            _colunas_treino = [c for c in _get_pq().columns if c != "isRecommended"]
            features_df = features_df.reindex(columns=_colunas_treino, fill_value=0.0)
            pipeline_ok = True
        except Exception as exc:
            st.error(f"❌ Erro ao preparar as features: {exc}")
            pipeline_ok = False

    # ── Passo 2: carregamento do modelo e predição ────────────────────────────
    if pipeline_ok:
        with st.spinner("Carregando modelo e realizando predição..."):
            try:
                
                modelo = _modelo_em_cache(db_uri)
                print("Modelo carregado com sucesso do MLflow.", features_df["alcohol"].values)
                y_hat  = prever_individual(features_df, modelo)
                predicao_ok = True
            except Exception as exc:
                st.error(
                    f"❌ Erro ao carregar o modelo ou realizar predição: {exc}\n\n"
                    f"Verifique se o banco SQLite está em: `{db_uri}`"
                )
                predicao_ok = False

    # ── Diagnóstico: verifica se features variam entre chamadas ──────────────
    if pipeline_ok:
        from utils.pipeline_utils import obter_parquet_features
        df_real = obter_parquet_features()
        linha_ref = df_real.drop(columns=["isRecommended"]).iloc[[0]]

        colunas_form    = set(features_df.columns.tolist())
        colunas_parquet = set(linha_ref.columns.tolist())
        apenas_form     = colunas_form - colunas_parquet
        apenas_parquet  = colunas_parquet - colunas_form

        ordem_igual = features_df.columns.tolist() == linha_ref.columns.tolist()

        # with st.expander("🛠️ Diagnóstico de features (debug)", expanded=True):
        #     st.markdown(f"**Colunas no formulário:** {len(colunas_form)}  |  **Colunas no parquet:** {len(colunas_parquet)}")
        #     if apenas_form:
        #         st.error(f"Colunas APENAS no formulário: {sorted(apenas_form)}")
        #     if apenas_parquet:
        #         st.error(f"Colunas APENAS no parquet: {sorted(apenas_parquet)}")
        #     if not apenas_form and not apenas_parquet:
        #         st.success("Colunas idênticas ✓")

        #     if ordem_igual:
        #         st.success("Ordem das colunas idêntica ✓")
        #     else:
        #         st.error("Ordem das colunas DIFERENTE — valores chegam nas features erradas!")
        #         ordem_df = pd.DataFrame({
        #             "pos": range(len(features_df.columns)),
        #             "formulario": features_df.columns.tolist(),
        #             "parquet":    linha_ref.columns.tolist(),
        #             "igual":      [a == b for a, b in zip(features_df.columns, linha_ref.columns)],
        #         })
        #         st.dataframe(ordem_df[~ordem_df["igual"]], use_container_width=True)

        #     nans = features_df.isna().sum().sum()
        #     if nans > 0:
        #         st.error(f"NaN encontrados: {nans}")
        #     else:
        #         st.success("Sem NaN ✓")

        #     comparacao = features_df.T.rename(columns={0: "formulario"}).join(
        #         linha_ref.T.rename(columns={0: "parquet_ref"}), how="outer"
        #     )
        #     comparacao["diferenca"] = (comparacao["formulario"] - comparacao["parquet_ref"]).abs()
        #     st.dataframe(comparacao.style.format("{:.6f}"), use_container_width=True, height=600)

        #     # Teste de sensibilidade: perturba cada coluna +10% e verifica se a probabilidade muda
        #     st.markdown("**Teste de sensibilidade do modelo:**")
        #     prob_base = modelo.predict_proba(features_df)[0, 1]
        #     sensibilidade = []
        #     for col in features_df.columns:
        #         df_perturb = features_df.copy()
        #         df_perturb[col] = df_perturb[col] * 1.5 + 1.0
        #         prob_perturb = modelo.predict_proba(df_perturb)[0, 1]
        #         sensibilidade.append({"feature": col, "prob_base": prob_base, "prob_perturbada": prob_perturb, "delta": abs(prob_perturb - prob_base)})
        #     sens_df = pd.DataFrame(sensibilidade).sort_values("delta", ascending=False)
        #     st.dataframe(sens_df.style.format({"prob_base": "{:.4f}", "prob_perturbada": "{:.4f}", "delta": "{:.4f}"}), use_container_width=True)
        #     if sens_df["delta"].max() < 0.001:
        #         st.error("⚠️ Modelo completamente insensível a perturbações — problema está no modelo treinado (PCA + desbalanceamento). Retreine com threshold menor ou class_weight=balanced.")

    # ── Passo 4: exibição dos resultados ──────────────────────────────────────
    if pipeline_ok and predicao_ok:
        st.divider()
        st.subheader("📊 Resultado da Predição")

        classe_prevista = 1 if float(y_hat) >= 0.5 else 0
        rotulo = "Recomendado" if classe_prevista == 1 else "Não recomendado"
        delta = "Classe 1" if classe_prevista == 1 else "Classe 0"

        col_res1, col_res2 = st.columns([2, 1])

        with col_res1:
            st.metric(
                label="Recomendação prevista",
                value=rotulo,
                delta=delta,
                help="Classe prevista pelo modelo carregado do MLflow.",
            )
            st.progress(float(y_hat))

        with col_res2:
            st.metric("Probabilidade (classe 1)", f"{float(y_hat):.4f}")
            st.caption("Limiar aplicado na página: 0.5")

        # ── Inspeção das features engenheiradas ───────────────────────────────
        with st.expander("🔍 Inspecionar features engenheiradas"):
            st.caption(
                f"{len(features_df.columns)} features enviadas ao modelo "
                f"(entrada base + derivadas → {len(features_df.columns)} colunas finais)"
            )
            st.dataframe(
                features_df.T.rename(columns={0: "valor"}).style.format("{:.4f}"),
                use_container_width=True,
                height=600,
            )



# df_real = obter_parquet_features()
# linha_qualidade_9 = df_real[df_real['isRecommended'] == 1].iloc[0:1].drop(columns=['isRecommended'])
# prob_real = prever_individual(linha_qualidade_9, _modelo_em_cache(db_uri))
# st.write(f"Prob de linha REAL do parquet (classe 1): {prob_real:.4f}")
