
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BAYES_CSV = DATA_DIR / "bayesian_offensive_test_predictions.csv"


@st.cache_data
def _load_bayesian_predictions() -> pd.DataFrame:
    return pd.read_csv(BAYES_CSV)


st.set_page_config(
    page_title="NFL Draft — Bayesian model",
    layout="wide",
    initial_sidebar_state="expanded",
)

page = st.sidebar.radio("Page", ["Home", "Bayesian results"])

if page == "Home":
    st.title("NFL draft prediction — Bayesian logistic model")
    st.caption("DS 4420 · Andrew Thomas Lotocki & Alexander Erik Rees")
    st.markdown(
        """
This project predicts whether a college football player will be drafted using
combine measurements and college production stats, for quarterbacks, running backs, and wide receivers.

### Method

We fit a Bayesian logistic regression with a Gaussian prior on coefficients.
Posterior exploration uses Metropolis–Hastings MCMC, logit likelihood, random walk proposals, burn-in
and thinning, then it gives the posterior mean draft probability for each player.
        """
    )

else:
    st.title("Bayesian model")
    st.caption(
        "Posterior mean P(drafted) from MH MCMC; threshold slider "
        "without refitting."
    )

    df = _load_bayesian_predictions()
    positions = sorted(df["position"].unique().tolist())
    pos = st.selectbox("Position", positions, index=0)
    sub = df[df["position"] == pos].sort_values("test_row").reset_index(drop=True)
    sub = sub.assign(
        y_true=pd.to_numeric(sub["y_true"], errors="coerce").astype("Int64"),
        test_row=pd.to_numeric(sub["test_row"], errors="coerce").astype(int),
        p_drafted=pd.to_numeric(sub["p_drafted"], errors="coerce"),
    )
    if sub["y_true"].isna().any() or sub["p_drafted"].isna().any():
        st.warning("Dropped rows with missing y_true or p_drafted.")
        sub = sub.dropna(subset=["y_true", "p_drafted"])

    default_thr = float(sub["prob_threshold_default"].iloc[0])
    thr = st.slider(
        "Classify as drafted if P(drafted) ≥",
        min_value=0.05,
        max_value=0.95,
        value=min(max(default_thr, 0.05), 0.95),
        step=0.05,
    )

    y_true = sub["y_true"].astype(int).to_numpy()
    probs = sub["p_drafted"].to_numpy(dtype=float)
    y_pred = (probs >= thr).astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test n", len(sub))
    c2.metric("Accuracy @ threshold", f"{accuracy_score(y_true, y_pred):.3f}")
    c3.metric("F1 (drafted)", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    c4.metric("True drafted / predicted drafted", f"{cm[1, 1]}/{cm[1].sum()}")

    st.subheader("Interactive plot")
    st.caption(
        "Each row is one player in the set, the x axis shows the posterior mean predicted by our model for a player to be drafted. The dashed line is the threshold you set in choosing whether to classify a player as drafted or not."
        "The players are sorted by the posterior mean predicted by our model for a player to be drafted, from highest to lowest."
    )
    viz = sub.assign(
        player=sub["player"].astype(str),
        actually_drafted=sub["y_true"].astype(int).map({0: "Not drafted", 1: "Drafted"}),
    )
    player_order = viz.sort_values("p_drafted", ascending=False)["player"].tolist()

    pts = (
        alt.Chart(viz)
        .mark_circle(size=90, stroke="black", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "p_drafted:Q",
                title="Posterior mean P(drafted)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "player:N",
                sort=player_order,
                title="Player (holdout set)",
            ),
            color=alt.Color(
                "actually_drafted:N",
                title="Actual outcome",
                scale=alt.Scale(domain=["Not drafted", "Drafted"], range=["#c0392b", "#2980b9"]),
            ),
            tooltip=[
                alt.Tooltip("player", title="Player"),
                alt.Tooltip("p_drafted", title="P(drafted)", format=".3f"),
                alt.Tooltip("actually_drafted", title="Actual"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"thr": [thr]}))
        .mark_rule(strokeDash=[6, 4], color="#555")
        .encode(x="thr:Q")
    )
    h = min(900, 120 + 14 * len(viz))
    chart = (
        (pts + rule)
        .properties(
            title=f"{pos}: posterior draft probability by player",
            width="container",
            height=h,
        )
        .configure_axis(labelLimit=400)
    )
    st.altair_chart(chart, use_container_width=True)
