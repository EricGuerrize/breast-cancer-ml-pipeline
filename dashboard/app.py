"""Dashboard Streamlit — exploração e predição (educacional)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Garante import do pacote em ambientes sem install editável
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from breast_cancer_ml.config import (  # noqa: E402
    FEATURE_NAMES,
    METRICS_DIR,
    MODELS_DIR,
    TARGET_LABELS,
)
from breast_cancer_ml.ingestion.load import load_from_sklearn  # noqa: E402
from breast_cancer_ml.utils.io import load_json, load_model  # noqa: E402

st.set_page_config(
    page_title="Breast Cancer ML · Eric Guerrize",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS: atmosfera sutil, tipografia distinta (evita roxo/creme genéricos)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #dce9f5 0%, transparent 55%),
                    radial-gradient(900px 500px at 100% 0%, #e8f0ea 0%, transparent 50%),
                    linear-gradient(180deg, #f7fafc 0%, #eef3f7 100%);
    }
    h1, h2, h3 { letter-spacing: -0.02em; color: #0f2a3d !important; }
    .brand {
        font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.14em;
        color: #2a6f7c; font-weight: 700; margin-bottom: 0.25rem;
    }
    .mono { font-family: 'IBM Plex Mono', monospace; }
    .ethics {
        border-left: 3px solid #c45c26; padding: 0.75rem 1rem;
        background: rgba(196, 92, 38, 0.08); margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_from_sklearn()


@st.cache_resource(show_spinner=False)
def get_model():
    path = MODELS_DIR / "best_model.joblib"
    if not path.exists():
        # Treina sob demanda se artefato ausente
        from breast_cancer_ml.training.pipeline import train
        from breast_cancer_ml.validation.schema import validate_dataframe

        train(validate_dataframe(get_dataset()), persist=True)
    return load_model(MODELS_DIR / "best_model.joblib")


@st.cache_data(show_spinner=False)
def get_metrics() -> dict | None:
    path = METRICS_DIR / "metrics.json"
    if path.exists():
        return load_json(path)
    return None


def sidebar_nav() -> str:
    st.sidebar.markdown("### Eric Guerrize")
    st.sidebar.caption("Data Engineer · ML Pipeline")
    page = st.sidebar.radio(
        "Navegação",
        ["Predição", "Explorar dataset", "Métricas & explicação"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div class="ethics"><strong>Aviso ético:</strong> ferramenta educacional. '
        "Não é dispositivo médico e não substitui diagnóstico profissional.</div>",
        unsafe_allow_html=True,
    )
    return page


def page_predict(model) -> None:
    st.markdown('<p class="brand">Breast Cancer ML Pipeline</p>', unsafe_allow_html=True)
    st.title("Predição diagnóstica (educacional)")
    st.write(
        "Informe as 30 features do núcleo celular (mesmo schema UCI / sklearn) "
        "ou carregue um CSV com uma linha."
    )

    df = get_dataset()
    means = df[FEATURE_NAMES].mean()

    uploaded = st.file_uploader("CSV opcional (colunas = features)", type=["csv"])
    values: dict[str, float] = {}

    if uploaded is not None:
        up = pd.read_csv(uploaded)
        for name in FEATURE_NAMES:
            if name in up.columns:
                values[name] = float(up.iloc[0][name])
            else:
                values[name] = float(means[name])
        st.success("CSV carregado — features ausentes preenchidas com a média do dataset.")
    else:
        st.caption("Ajuste os sliders (valores iniciais = média do dataset).")
        cols = st.columns(3)
        for i, name in enumerate(FEATURE_NAMES):
            lo = float(df[name].quantile(0.01))
            hi = float(df[name].quantile(0.99))
            if hi <= lo:
                hi = lo + 1.0
            with cols[i % 3]:
                values[name] = st.slider(
                    name,
                    min_value=round(lo, 4),
                    max_value=round(hi, 4),
                    value=round(float(means[name]), 4),
                    key=f"feat_{i}",
                )

    if st.button("Predizer", type="primary"):
        row = pd.DataFrame([{n: values[n] for n in FEATURE_NAMES}])
        pred = int(model.predict(row)[0])
        proba = model.predict_proba(row)[0]
        label = TARGET_LABELS[pred]
        conf = float(proba[pred])
        st.markdown(f"### Resultado: **{label}** (classe `{pred}`)")
        st.progress(conf, text=f"Confiança do modelo: {conf:.1%}")
        st.caption(
            "Encoding sklearn: 0 = malignant (maligno), 1 = benign (benigno). "
            "Calibração de probabilidade é aproximada — ver seção de métricas."
        )


def page_explore() -> None:
    st.markdown('<p class="brand">Breast Cancer ML Pipeline</p>', unsafe_allow_html=True)
    st.title("Explorar dataset")
    df = get_dataset()
    st.write(
        f"**{len(df)}** amostras · **{len(FEATURE_NAMES)}** features · "
        "fonte: `sklearn.datasets.load_breast_cancer` "
        "(equivalente UCI / Kaggle Wisconsin Diagnostic)."
    )
    st.dataframe(df.head(20), use_container_width=True)

    counts = df["target"].map(TARGET_LABELS).value_counts()
    st.bar_chart(counts)

    feat = st.selectbox("Distribuição de feature", FEATURE_NAMES, index=0)
    hist = (
        df[[feat, "target"]]
        .assign(classe=lambda d: d["target"].map(TARGET_LABELS))
        .groupby("classe", as_index=False)[feat]
        .mean()
        .set_index("classe")
    )
    st.caption(f"Média de `{feat}` por classe")
    st.bar_chart(hist)


def page_metrics(model) -> None:
    st.markdown('<p class="brand">Breast Cancer ML Pipeline</p>', unsafe_allow_html=True)
    st.title("Métricas & explicação")
    metrics = get_metrics()
    if metrics:
        best = metrics.get("best_model")
        st.subheader(f"Melhor modelo: `{best}`")
        m = metrics["models"][best]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['accuracy']:.3f}")
        c2.metric("Precision", f"{m['precision']:.3f}")
        c3.metric("Recall", f"{m['recall']:.3f}")
        c4.metric("F1", f"{m['f1']:.3f}")
        c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        st.write("Matriz de confusão:", m["confusion_matrix"])
        cal = metrics.get("calibration", {}).get(best, {})
        if cal:
            st.info(cal.get("note", ""))
        imp = m.get("feature_importance") or {}
    else:
        st.warning("Métricas ainda não geradas. Rode `make train` ou `bcm demo`.")
        imp = {}
        # Fallback: importância do modelo em memória
        clf = model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            vals = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            vals = np.abs(clf.coef_).ravel()
        else:
            vals = np.zeros(len(FEATURE_NAMES))
        imp = dict(
            sorted(
                zip(FEATURE_NAMES, vals, strict=True),
                key=lambda x: -x[1],
            )
        )

    st.subheader("Importância de features")
    top = dict(list(imp.items())[:15])
    st.bar_chart(pd.Series(top, name="importância"))

    with st.expander("SHAP (opcional — pode ser lento)"):
        if st.button("Calcular SHAP (amostra)"):
            try:
                import shap

                df = get_dataset()
                sample = df[FEATURE_NAMES].sample(n=min(80, len(df)), random_state=42)
                # Explica o classificador após o scaler do Pipeline
                scaled = model.named_steps["scaler"].transform(sample)
                clf = model.named_steps["clf"]
                explainer = shap.Explainer(clf.predict_proba, scaled)
                shap_values = explainer(scaled)
                # Classe positiva (benign=1) — índice 1
                vals = shap_values.values
                if vals.ndim == 3:
                    vals = vals[:, :, 1]
                mean_abs = np.abs(vals).mean(axis=0)
                shap_imp = (
                    pd.Series(mean_abs, index=FEATURE_NAMES).sort_values(ascending=False).head(15)
                )
                st.bar_chart(shap_imp)
            except Exception as exc:  # noqa: BLE001
                st.error(f"SHAP indisponível: {exc}")


def main() -> None:
    page = sidebar_nav()
    model = get_model()
    if page == "Predição":
        page_predict(model)
    elif page == "Explorar dataset":
        page_explore()
    else:
        page_metrics(model)


if __name__ == "__main__":
    main()
