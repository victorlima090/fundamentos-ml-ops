"""
pages/2_Monitoramento.py — Dashboard Streamlit: monitoramento de modelo em produção.

Simula monitoramento de inferência em lote para classificação binária:
  1. Carrega N amostras do parquet de features.
  2. Realiza predição em lote via modelo MLflow local (sem servidor REST),
     obtendo a probabilidade da classe 1 (isRecommended) para cada amostra.
  3. Divide as predições em K lotes sequenciais para simular ingestão temporal.
  4. Calcula métricas por lote: Acurácia, F1, ROC-AUC, probabilidade média.
  5. Visualiza:
       - Séries temporais de cada métrica com média móvel (janela configurável).
       - Distribuição de probabilidades por classe real (classe 0 vs classe 1).
       - Curva ROC global.
       - Tabela de métricas por lote (expansível).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# ── Bootstrap de paths ────────────────────────────────────────────────────────
_PAGE_DIR     = Path(__file__).resolve().parent   # pages/
_APP_DIR      = _PAGE_DIR.parent                  # production_app/
_PROJECT_ROOT = _APP_DIR.parent                   # demo_projeto/

for _p in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.pipeline_utils import (
    obter_parquet_features,
    _TARGET_COL,
)
from utils.model_utils import carregar_modelo, prever_lote

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monitoramento do Modelo",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Dashboard de Monitoramento do Modelo")
st.markdown(
    """
    Simula **monitoramento de produção em lote** para classificação binária de vinho.
    Amostra N pontos do parquet de features, obtém a **probabilidade da classe 1**
    (vinho recomendado) para cada um e calcula métricas ao longo de K lotes sequenciais.
    Use para detectar drift de performance ou degradação do modelo ao longo do tempo.
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configurações
# ─────────────────────────────────────────────────────────────────────────────
_URI_PADRAO = f"sqlite:///{_PROJECT_ROOT / 'mlruns.db'}"

with st.sidebar:
    st.header("⚙️ Configurações")
    db_uri = st.text_input(
        "URI do banco SQLite",
        value=_URI_PADRAO,
        help="URI do banco SQLite gerado por modelagem.py.",
    )
    n_amostras = st.slider(
        "Total de amostras",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Número de pontos a amostrar do parquet de features.",
    )
    n_lotes = st.slider(
        "Número de lotes",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Divide o total de amostras em N lotes sequenciais.",
    )
    limiar = st.slider(
        "Limiar de classificação",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probabilidade mínima para classificar como classe 1 (recomendado).",
    )
    janela_movel = st.slider(
        "Janela de média móvel",
        min_value=2,
        max_value=10,
        value=3,
        help="Tamanho da janela para a média móvel nos gráficos de série temporal.",
    )
    semente = st.number_input("Semente aleatória", value=42, step=1)

# ─────────────────────────────────────────────────────────────────────────────
# Cache do modelo
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Carregando modelo do MLflow...")
def _modelo_em_cache(uri: str):
    """Carrega e armazena em cache o modelo MLflow para evitar recarregamentos."""
    return carregar_modelo(uri)


# ─────────────────────────────────────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def _calcular_metricas(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calcula Acurácia, F1, ROC-AUC e probabilidade média da classe 1."""
    y_pred = (y_proba >= threshold).astype(int)
    metricas: dict[str, float] = {
        "acuracia":   float(accuracy_score(y_true, y_pred)),
        "f1":         float(f1_score(y_true, y_pred, zero_division=0)),
        "prob_media": float(y_proba.mean()),
    }
    if len(np.unique(y_true)) >= 2:
        metricas["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metricas["roc_auc"] = float("nan")
    return metricas


def _plotar_serie_temporal(
    ax: plt.Axes,
    lotes: list[int],
    valores: list[float],
    movel: pd.Series,
    nome_metrica: str,
    cor: str,
) -> None:
    """Plota métrica por lote + média móvel em um eixo Matplotlib."""
    ax.plot(lotes, valores, "o-", color=cor, alpha=0.5, linewidth=1.5,
            markersize=4, label="Por lote")
    ax.plot(lotes, movel, "-", color=cor, linewidth=2.5,
            label=f"Média móvel (j={janela_movel})")
    ax.set_xlabel("Lote #", fontsize=9)
    ax.set_title(nome_metrica, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    arr_movel = movel.values.astype(float)
    desvio = np.nanstd(valores)
    ax.fill_between(lotes, arr_movel - desvio, arr_movel + desvio,
                    color=cor, alpha=0.08)


# ─────────────────────────────────────────────────────────────────────────────
# Execução do monitoramento
# ─────────────────────────────────────────────────────────────────────────────
btn_executar = st.button(
    "▶️ Executar Análise de Monitoramento",
    type="primary",
    use_container_width=True,
)

if btn_executar:

    # ── Carregamento dos dados de features ────────────────────────────────────
    with st.spinner("Carregando parquet de features..."):
        try:
            df_features = obter_parquet_features()
        except Exception as exc:
            st.error(f"❌ Erro ao carregar o parquet de features: {exc}")
            st.stop()

    # ── Amostragem e separação X / y ──────────────────────────────────────────
    rng = np.random.default_rng(int(semente))
    idx_amostra = rng.choice(
        len(df_features),
        size=min(n_amostras, len(df_features)),
        replace=False,
    )
    df_amostra = df_features.iloc[idx_amostra].reset_index(drop=True)

    y_true_total = df_amostra[_TARGET_COL].values.astype(int)
    X_amostra = df_amostra.drop(columns=[_TARGET_COL]).copy()

    # ── Predição em lote via modelo MLflow ────────────────────────────────────
    with st.spinner(f"Calculando probabilidade classe 1 ({len(X_amostra)} linhas)..."):
        try:
            modelo = _modelo_em_cache(db_uri)
            y_proba_total = np.array(prever_lote(X_amostra, modelo))
        except Exception as exc:
            st.error(f"❌ Erro na predição em lote: {exc}")
            st.stop()

    # ── Métricas por lote ─────────────────────────────────────────────────────
    tamanho_lote = len(X_amostra) // n_lotes
    resto        = len(X_amostra) % n_lotes

    metricas_lotes: list[dict] = []
    for i in range(n_lotes):
        inicio = i * tamanho_lote
        fim    = inicio + tamanho_lote + (1 if i < resto else 0)
        if fim > len(X_amostra):
            break
        m = _calcular_metricas(
            y_true_total[inicio:fim],
            y_proba_total[inicio:fim],
            limiar,
        )
        m["lote"] = i + 1
        metricas_lotes.append(m)

    df_metricas = pd.DataFrame(metricas_lotes).set_index("lote")
    df_movel    = df_metricas.rolling(window=janela_movel, min_periods=1).mean()
    geral       = _calcular_metricas(y_true_total, y_proba_total, limiar)

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 1 — KPIs gerais
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader(f"Métricas Gerais ({len(X_amostra)} amostras  |  limiar = {limiar:.2f})")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Acurácia",    f"{geral['acuracia']:.4f}")
    kpi2.metric("F1-Score",    f"{geral['f1']:.4f}")
    kpi3.metric("ROC-AUC",     f"{geral['roc_auc']:.4f}")
    kpi4.metric("Prob. Média", f"{geral['prob_media']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 2 — Séries temporais por lote
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📈 Métricas por Lote ao Longo do Tempo")
    st.caption(
        f"{len(metricas_lotes)} lotes × ~{tamanho_lote} amostras cada  |  "
        f"Janela de média móvel = {janela_movel} lotes"
    )

    lotes  = df_metricas.index.tolist()
    paleta = {
        "acuracia":   "#27ae60",
        "f1":         "#e67e22",
        "roc_auc":    "#2980b9",
        "prob_media": "#9b59b6",
    }
    rotulos = {
        "acuracia":   "Acurácia",
        "f1":         "F1-Score",
        "roc_auc":    "ROC-AUC",
        "prob_media": "Prob. Média (classe 1)",
    }

    fig_ts, eixos_ts = plt.subplots(2, 2, figsize=(14, 7), tight_layout=True)
    fig_ts.patch.set_facecolor("#0e1117")

    for metrica, ax in zip(paleta, eixos_ts.flat):
        ax.set_facecolor("#1a1a2e")
        _plotar_serie_temporal(
            ax=ax,
            lotes=lotes,
            valores=df_metricas[metrica].tolist(),
            movel=df_movel[metrica],
            nome_metrica=rotulos[metrica],
            cor=paleta[metrica],
        )
        for borda in ax.spines.values():
            borda.set_edgecolor("#333")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    st.pyplot(fig_ts, use_container_width=True)
    plt.close(fig_ts)

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 3 — Distribuição de probabilidades + Curva ROC
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📊 Distribuição de Probabilidades e Curva ROC")

    fig_dist, (ax_prob, ax_roc) = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)
    fig_dist.patch.set_facecolor("#0e1117")

    # --- Distribuição de probabilidades por classe real ----------------------
    ax_prob.set_facecolor("#1a1a2e")
    proba_cls0 = y_proba_total[y_true_total == 0]
    proba_cls1 = y_proba_total[y_true_total == 1]
    bins = np.linspace(0, 1, 31)
    ax_prob.hist(proba_cls0, bins=bins, alpha=0.6, color="#e74c3c", label="Classe 0 (não recom.)")
    ax_prob.hist(proba_cls1, bins=bins, alpha=0.6, color="#27ae60", label="Classe 1 (recom.)")
    ax_prob.axvline(limiar, color="white", linestyle="--", linewidth=1.5,
                    label=f"Limiar = {limiar:.2f}")
    ax_prob.set_title("Distribuição de Probabilidades (classe 1)", color="white",
                      fontsize=11, fontweight="bold")
    ax_prob.set_xlabel("Probabilidade prevista", color="white")
    ax_prob.set_ylabel("Contagem", color="white")
    ax_prob.tick_params(colors="white")
    for borda in ax_prob.spines.values():
        borda.set_edgecolor("#333")
    ax_prob.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    # --- Curva ROC -----------------------------------------------------------
    ax_roc.set_facecolor("#1a1a2e")
    if len(np.unique(y_true_total)) >= 2:
        fpr, tpr, _ = roc_curve(y_true_total, y_proba_total)
        ax_roc.plot(fpr, tpr, color="#2980b9", linewidth=2,
                    label=f"ROC-AUC = {geral['roc_auc']:.4f}")
    ax_roc.plot([0, 1], [0, 1], "r--", linewidth=1, label="Classificador aleatório")
    ax_roc.set_title("Curva ROC", color="white", fontsize=11, fontweight="bold")
    ax_roc.set_xlabel("Taxa de Falsos Positivos", color="white")
    ax_roc.set_ylabel("Taxa de Verdadeiros Positivos", color="white")
    ax_roc.tick_params(colors="white")
    for borda in ax_roc.spines.values():
        borda.set_edgecolor("#333")
    ax_roc.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    st.pyplot(fig_dist, use_container_width=True)
    plt.close(fig_dist)

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 4 — Tabela de métricas por lote
    # ═══════════════════════════════════════════════════════════════════════════
    with st.expander("📋 Tabela de métricas por lote"):
        df_exibir = df_metricas.copy()
        df_exibir["acuracia"]   = df_exibir["acuracia"].map("{:.4f}".format)
        df_exibir["f1"]         = df_exibir["f1"].map("{:.4f}".format)
        df_exibir["roc_auc"]    = df_exibir["roc_auc"].map("{:.4f}".format)
        df_exibir["prob_media"] = df_exibir["prob_media"].map("{:.4f}".format)
        df_exibir.columns = ["Acurácia", "F1-Score", "ROC-AUC", "Prob. Média"]
        st.dataframe(df_exibir, use_container_width=True)
