"""Predictive Maintenance AI Platform — Streamlit Dashboard.

Tabs
----
1. Overview          — project summary, architecture, dataset stats
2. Benchmark         — full metrics table, comparison plots, cost analysis
3. Live Prediction   — interactive sensor input → real-time failure probability
4. Error Analysis    — FP/FN breakdown by machine type, degradation, proximity
5. Interpretability  — feature importance, gradient saliency, sensor rankings

Usage
-----
streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Make src importable ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def _load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def _load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_resource
def _load_predictor(ckpt_path: str, prep_path: str):
    """Load LSTM predictor (cached across reruns)."""
    from src.api.predictor import MaintenancePredictor
    try:
        return MaintenancePredictor(checkpoint_path=ckpt_path, preprocessor_path=prep_path)
    except Exception as exc:
        return exc


def _info_box(msg: str) -> None:
    st.info(f" {msg}", icon="ℹ️")


def _missing(resource: str) -> None:
    st.warning(
        f"**{resource}** not found. Run the training/benchmark pipeline first.",
        icon="⚠️",
    )


def _metric_card(label: str, value: str, delta: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — dataset selector
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Predictive Maintenance AI")
    st.markdown("---")

    dataset = st.selectbox(
        "Dataset",
        options=["cmapss", "synthetic"],
        index=0,
        help="Switch between the NASA CMAPSS real dataset and the synthetic benchmark.",
    )
    cmapss_subset = "FD001"
    if dataset == "cmapss":
        cmapss_subset = st.selectbox(
            "CMAPSS Subset",
            options=["FD001", "FD002", "FD003", "FD004"],
            index=0,
        )

    st.markdown("---")
    st.caption("Platform v1.0 · Staff ML Portfolio")

# ── Derived paths ─────────────────────────────────────────────────────────────
if dataset == "cmapss":
    _bench_dir   = Path(f"reports/benchmark/cmapss/{cmapss_subset}")
    _ckpt_base   = Path(f"models/checkpoints/cmapss/{cmapss_subset}")
    _base_base   = Path(f"models/baselines/cmapss/{cmapss_subset}")
    _report_base = Path(f"reports/cmapss/{cmapss_subset}")
    _err_dir     = Path(f"reports/error_analysis/cmapss/{cmapss_subset}")
    _interp_dir  = Path(f"reports/interpretability/cmapss/{cmapss_subset}")
else:
    _bench_dir   = Path("reports/benchmark")
    _ckpt_base   = Path("models/checkpoints")
    _base_base   = Path("models/baselines")
    _report_base = Path("reports")
    _err_dir     = Path("reports/error_analysis")
    _interp_dir  = Path("reports/interpretability")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_overview, tab_bench, tab_pred, tab_error, tab_interp = st.tabs([
    " Overview",
    " Benchmark",
    " Live Prediction",
    " Error Analysis",
    " Interpretability",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.header("Predictive Maintenance AI Platform")
    st.markdown(
        """
        **Binary classification**: predict whether a machine will fail within the next
        **30 time-steps** so maintenance crews can intervene before unplanned downtime occurs.

        | Aspect | Details |
        |---|---|
        | **Problem type** | Binary classification (failure-imminent horizon) |
        | **Input** | Sliding window of sensor readings (T × F) |
        | **Output** | P(failure within next 30 steps) |
        | **Models** | LSTM, 1D-CNN, Random Forest, Logistic Regression |
        | **Evaluation** | F1, ROC-AUC, PR-AUC, MCC, Brier Score, ECE, Expected Cost |
        | **Class imbalance** | BCEWithLogitsLoss + auto pos_weight · 3–5% positive rate |
        | **Calibration** | Platt Scaling, Isotonic Regression, ECE analysis |
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Datasets")
        st.markdown(
            """
            **Synthetic** — 200 machines, 5 sensors, variable failure patterns
            * Temperature, Vibration, Pressure, RPM, Current
            * 3 machine types × 3 degradation profiles
            * Failure position randomised (not always at end of series)

            **NASA CMAPSS** — real turbofan engine degradation data
            * 4 sub-datasets: FD001 (single condition) → FD004 (multi-condition + fault mode)
            * 21 sensors, 3 operating settings, up to 362 engines
            * Ground-truth RUL provided for test split
            """
        )

    with col2:
        st.subheader("Architecture")
        st.markdown(
            """
            **LSTM Classifier**
            * Bidirectional · 2 layers · hidden=128 · dropout=0.3
            * Linear(256→1) + sigmoid for probability output
            * ~400k parameters

            **Temporal CNN Classifier**
            * 4 residual blocks · 64 channels · kernel=7
            * GlobalAvgPool → Linear(64→1)
            * ~180k parameters

            **Baselines** · Random Forest (300 trees) · Logistic Regression (L2)
            """
        )

    # Dataset stats
    st.subheader("Dataset Statistics")
    if dataset == "synthetic":
        df = _load_csv("data/synthetic/sensor_data.csv")
        if df is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total rows", f"{len(df):,}")
            c2.metric("Machines", str(df["machine_id"].nunique()))
            c3.metric("Positive rate", f"{df['failure_imminent'].mean():.2%}")
            c4.metric("Sensor columns", "5")
            if "machine_type" in df.columns:
                st.markdown("**Machine type distribution**")
                st.dataframe(df["machine_type"].value_counts().rename("count").to_frame(), use_container_width=False)
        else:
            _missing("data/synthetic/sensor_data.csv")
    else:
        st.markdown(f"Selected CMAPSS subset: **{cmapss_subset}**")
        cmapss_info = {
            "FD001": ("100 train engines", "100 test engines", "1 operating condition", "1 fault mode"),
            "FD002": ("260 train engines", "259 test engines", "6 operating conditions", "1 fault mode"),
            "FD003": ("100 train engines", "100 test engines", "1 operating condition", "2 fault modes"),
            "FD004": ("249 train engines", "248 test engines", "6 operating conditions", "2 fault modes"),
        }
        info = cmapss_info[cmapss_subset]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train engines", info[0])
        c2.metric("Test engines", info[1])
        c3.metric("Op. conditions", info[2])
        c4.metric("Fault modes", info[3])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

with tab_bench:
    st.header("Model Benchmark Results")

    bench = _load_json(str(_bench_dir / "benchmark_results.json"))

    if bench is None:
        _missing(str(_bench_dir / "benchmark_results.json"))
        st.markdown(
            "Run the full benchmark with:\n"
            "```\n"
            f"python scripts/run_full_benchmark.py --dataset {dataset}"
            + (f" --cmapss-subset {cmapss_subset}" if dataset == "cmapss" else "")
            + "\n```"
        )
    else:
        metrics_dict: dict = bench.get("metrics", {})

        # ── Top KPIs ─────────────────────────────────────────────────────────
        st.subheader("Test Set KPIs")
        best_model = max(metrics_dict, key=lambda m: metrics_dict[m].get("f1", 0))
        bm = metrics_dict[best_model]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best Model", best_model)
        c2.metric("F1", f"{bm.get('f1', 0):.4f}")
        c3.metric("ROC-AUC", f"{bm.get('roc_auc', 0):.4f}")
        c4.metric("PR-AUC", f"{bm.get('pr_auc', 0):.4f}")
        c5.metric("MCC", f"{bm.get('mcc', 0):.4f}")

        # ── Metrics table ─────────────────────────────────────────────────────
        st.subheader("Full Metrics Table")
        display_cols = ["f1", "roc_auc", "pr_auc", "recall", "precision", "mcc",
                        "brier_score", "ece", "expected_cost_optimal_threshold"]
        rows = []
        for model_name, m in metrics_dict.items():
            row = {"Model": model_name}
            for col in display_cols:
                v = m.get(col, 0)
                row[col] = f"${v:,.0f}" if col == "expected_cost_optimal_threshold" else round(float(v), 4)
            rows.append(row)

        if rows:
            df_metrics = pd.DataFrame(rows).set_index("Model")
            # Highlight best per column (higher=better except cost+brier+ece)
            lower_better = {"brier_score", "ece", "expected_cost_optimal_threshold"}
            st.dataframe(df_metrics, use_container_width=True)

        # ── Bootstrap CIs ─────────────────────────────────────────────────────
        ci = bench.get("bootstrap_ci", {})
        if ci:
            st.subheader("Bootstrap 95% Confidence Intervals (n=500)")
            ci_rows = []
            for model_name, model_ci in ci.items():
                for metric_name, vals in model_ci.items():
                    if metric_name in {"f1", "roc_auc", "pr_auc"}:
                        ci_rows.append({
                            "Model": model_name,
                            "Metric": metric_name,
                            "Mean": round(vals.get("mean", 0), 4),
                            "Lower": round(vals.get("lower", 0), 4),
                            "Upper": round(vals.get("upper", 0), 4),
                        })
            if ci_rows:
                st.dataframe(pd.DataFrame(ci_rows), use_container_width=True)

        # ── Visual artifacts ──────────────────────────────────────────────────
        st.subheader("Plots")
        plot_files = {
            "ROC Curves (all models)": _bench_dir / "roc_all_models.png",
            "PR Curves (all models)": _bench_dir / "pr_all_models.png",
            "Calibration": _bench_dir / "calibration_all_models.png",
            "Model Comparison": _bench_dir / "comparison_bar.png",
            "Cost Curves": _bench_dir / "cost_curve_all_models.png",
        }
        available = {k: v for k, v in plot_files.items() if v.exists()}
        if available:
            cols = st.columns(min(3, len(available)))
            for i, (title, path) in enumerate(available.items()):
                with cols[i % len(cols)]:
                    st.image(str(path), caption=title, use_container_width=True)
        else:
            _info_box("No plot images found in the benchmark report directory.")

        # ── Per-model confusion matrices ──────────────────────────────────────
        st.subheader("Confusion Matrices")
        cm_files = list(_bench_dir.glob("*_confusion_matrix.png"))
        if cm_files:
            cols = st.columns(min(4, len(cm_files)))
            for i, path in enumerate(cm_files):
                with cols[i % len(cols)]:
                    st.image(str(path), caption=path.stem, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

with tab_pred:
    st.header("Live Failure Probability Prediction")
    st.markdown(
        "Enter current sensor readings below. The LSTM model will estimate the "
        "probability of failure within the next **30 time-steps**."
    )

    # ── Load predictor ────────────────────────────────────────────────────────
    ckpt_path = str(_ckpt_base / "lstm" / "best_model.pt")
    prep_path = str(_ckpt_base / "lstm" / "preprocessor.joblib")

    if not Path(ckpt_path).exists():
        # Fall back to old layout
        ckpt_path = str(_ckpt_base / "best_model.pt")
        prep_path = str(_ckpt_base / "preprocessor.joblib")

    model_ready = Path(ckpt_path).exists() and Path(prep_path).exists()

    if not model_ready:
        _missing(f"LSTM checkpoint ({ckpt_path})")
        st.markdown(
            "Train the LSTM model first:\n"
            "```\n"
            f"python scripts/train_neural_model.py --model-type lstm --dataset {dataset}"
            + (f" --cmapss-subset {cmapss_subset}" if dataset == "cmapss" else "")
            + "\n```"
        )
    else:
        predictor = _load_predictor(ckpt_path, prep_path)
        if isinstance(predictor, Exception):
            st.error(f"Failed to load model: {predictor}")
        else:
            # ── Sensor inputs ─────────────────────────────────────────────────
            st.subheader("Sensor Readings")

            # Determine sensor columns from preprocessor
            sensor_cols = predictor._preprocessor.sensor_columns  # type: ignore[union-attr]
            window_size = predictor._preprocessor.window_size  # type: ignore[union-attr]

            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.markdown(f"**Window size**: {window_size} time-steps  \n"
                            f"**Sensors**: {', '.join(sensor_cols)}")
                threshold = st.slider(
                    "Decision threshold", min_value=0.05, max_value=0.95,
                    value=0.5, step=0.01,
                )

            with col_right:
                # Provide sensible defaults for synthetic sensors
                defaults = {
                    "temperature": 75.0, "vibration": 0.5, "pressure": 100.0,
                    "rpm": 1500.0, "current": 12.0,
                }
                reading: dict[str, float] = {}
                for col in sensor_cols:
                    reading[col] = st.number_input(
                        label=col,
                        value=float(defaults.get(col, 0.0)),
                        format="%.4f",
                        key=f"sensor_{col}",
                    )

            if st.button("Predict", type="primary"):
                readings = [reading] * window_size  # repeat to fill window
                try:
                    result = predictor.predict(readings, threshold=threshold)
                    prob = result["failure_probability"]
                    imminent = result["failure_imminent"]

                    st.markdown("---")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Failure Probability", f"{prob:.4f}")
                    r2.metric("Decision", "FAILURE IMMINENT" if imminent else "Normal", delta=None)
                    r3.metric("Threshold", f"{threshold:.2f}")

                    if imminent:
                        st.error(
                            f" **Failure imminent!** "
                            f"Probability {prob:.2%} exceeds threshold {threshold:.0%}. "
                            f"Schedule maintenance immediately.",
                            icon="",
                        )
                    else:
                        st.success(
                            f" **Normal operation.** "
                            f"Failure probability {prob:.2%} is below threshold {threshold:.0%}.",
                            icon="",
                        )

                    # Probability gauge
                    import plotly.graph_objects as go  # type: ignore[import]
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={"text": "Failure Probability (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "crimson" if imminent else "steelblue"},
                            "steps": [
                                {"range": [0, threshold * 100], "color": "lightgreen"},
                                {"range": [threshold * 100, 100], "color": "lightyellow"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": threshold * 100,
                            },
                        },
                        number={"suffix": "%", "valueformat": ".2f"},
                    ))
                    fig.update_layout(height=300, margin={"t": 30, "b": 0, "l": 0, "r": 0})
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

            # ── Batch inference from CSV upload ───────────────────────────────
            st.markdown("---")
            st.subheader("Batch Inference (CSV Upload)")
            st.markdown(
                f"Upload a CSV with columns: `{', '.join(sensor_cols)}`  \n"
                "Each row is one time-step. The last `window_size` rows will be used."
            )
            uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])
            if uploaded:
                try:
                    batch_df = pd.read_csv(uploaded)
                    missing_cols = [c for c in sensor_cols if c not in batch_df.columns]
                    if missing_cols:
                        st.error(f"Missing columns: {missing_cols}")
                    else:
                        rows_needed = window_size
                        if len(batch_df) < rows_needed:
                            st.warning(
                                f"CSV has {len(batch_df)} rows; need {rows_needed}. "
                                "Padding by repeating the last row."
                            )
                            pad = pd.concat(
                                [batch_df.iloc[[-1]]] * (rows_needed - len(batch_df)),
                                ignore_index=True,
                            )
                            batch_df = pd.concat([batch_df, pad], ignore_index=True)

                        window_df = batch_df[sensor_cols].tail(rows_needed)
                        batch_readings = window_df.to_dict(orient="records")
                        result = predictor.predict(batch_readings, threshold=threshold)
                        prob = result["failure_probability"]
                        imminent = result["failure_imminent"]
                        st.metric("Failure Probability", f"{prob:.4f}")
                        if imminent:
                            st.error(" Failure imminent! Schedule maintenance.", icon="")
                        else:
                            st.success(" Normal operation.", icon="")
                except Exception as exc:
                    st.error(f"Error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab_error:
    st.header("FP / FN Error Analysis")
    st.markdown(
        "Deep-dive into **False Positives** (unnecessary maintenance alerts) and "
        "**False Negatives** (missed failures) stratified by machine type, "
        "degradation stage, and proximity to failure."
    )

    err_json = _load_json(str(_err_dir / "error_analysis.json"))

    if err_json is None:
        _missing(str(_err_dir / "error_analysis.json"))
        st.markdown(
            "Run the error analysis script:\n"
            "```\n"
            f"python scripts/run_error_analysis.py --dataset {dataset}"
            + (f" --cmapss-subset {cmapss_subset}" if dataset == "cmapss" else "")
            + "\n```"
        )
    else:
        # ── Summary stats ─────────────────────────────────────────────────────
        for model_name, model_data in err_json.items():
            with st.expander(f"**{model_name}**", expanded=(model_name == list(err_json.keys())[0])):
                summary = model_data.get("summary", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total FP", str(summary.get("total_fp", "—")))
                c2.metric("Total FN", str(summary.get("total_fn", "—")))
                c3.metric("FP Rate", f"{summary.get('fp_rate', 0):.3f}")
                c4.metric("FN Rate", f"{summary.get('fn_rate', 0):.3f}")

                # Machine type breakdown
                mt = model_data.get("by_machine_type")
                if mt:
                    st.markdown("**By Machine Type**")
                    st.dataframe(pd.DataFrame(mt).T if isinstance(mt, dict) else pd.DataFrame(mt),
                                 use_container_width=True)

                # Degradation stage breakdown
                ds = model_data.get("by_degradation_stage")
                if ds:
                    st.markdown("**By Degradation Stage**")
                    st.dataframe(pd.DataFrame(ds).T if isinstance(ds, dict) else pd.DataFrame(ds),
                                 use_container_width=True)

                # Proximity breakdown
                prox = model_data.get("by_proximity")
                if prox:
                    st.markdown("**By Proximity to Failure (cycles before failure)**")
                    st.dataframe(pd.DataFrame(prox).T if isinstance(prox, dict) else pd.DataFrame(prox),
                                 use_container_width=True)

    # ── Plot images ───────────────────────────────────────────────────────────
    err_plots = list(_err_dir.glob("*.png")) if _err_dir.exists() else []
    if err_plots:
        st.subheader("Error Analysis Plots")
        cols = st.columns(min(3, len(err_plots)))
        for i, path in enumerate(err_plots):
            with cols[i % len(cols)]:
                st.image(str(path), caption=path.stem.replace("_", " ").title(),
                         use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — INTERPRETABILITY
# ══════════════════════════════════════════════════════════════════════════════

with tab_interp:
    st.header("Model Interpretability")
    st.markdown(
        "Understand **what the models learned**: sensor-level feature importance, "
        "temporal gradient saliency, and permutation importance."
    )

    interp_json = _load_json(str(_interp_dir / "interpretability.json"))

    if interp_json is None:
        _missing(str(_interp_dir / "interpretability.json"))
        st.markdown(
            "Run the interpretability script:\n"
            "```\n"
            f"python scripts/run_interpretability.py --dataset {dataset}"
            + (f" --cmapss-subset {cmapss_subset}" if dataset == "cmapss" else "")
            + "\n```"
        )
    else:
        for model_name, model_data in interp_json.items():
            with st.expander(f"**{model_name}**", expanded=True):
                # Sensor importance ranking
                si = model_data.get("sensor_importance")
                if si:
                    st.markdown("**Sensor Importance Ranking**")
                    si_df = pd.DataFrame.from_dict(si, orient="index", columns=["importance"])
                    si_df = si_df.sort_values("importance", ascending=False)
                    st.bar_chart(si_df)

                # Temporal importance
                ti = model_data.get("temporal_importance")
                if ti:
                    st.markdown("**Temporal Importance (timesteps)**")
                    ti_df = pd.DataFrame({"timestep": range(len(ti)), "importance": ti})
                    st.line_chart(ti_df.set_index("timestep"))

    # ── Plot images ───────────────────────────────────────────────────────────
    interp_plots = list(_interp_dir.glob("*.png")) if _interp_dir.exists() else []
    if interp_plots:
        st.subheader("Interpretability Plots")
        cols = st.columns(min(3, len(interp_plots)))
        for i, path in enumerate(interp_plots):
            with cols[i % len(cols)]:
                st.image(str(path), caption=path.stem.replace("_", " ").title(),
                         use_container_width=True)

    # ── Method explanations ───────────────────────────────────────────────────
    st.subheader("Methods")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Gradient Saliency (LSTM / CNN)**
            * Enable gradients on input tensor
            * Forward pass through model (not predict_proba)
            * `loss.backward()` on positive-class logit
            * Accumulate |grad| over batches → normalize [0,1]
            * Reshape (T, F) → sensor importance = mean over T; temporal = mean over F
            """
        )
    with col2:
        st.markdown(
            """
            **Permutation Importance (all models)**
            * Baseline F1 on test windows
            * For each sensor: shuffle that feature across all windows
            * Re-evaluate F1 — drop = sensor importance
            * Repeat for each timestep position for temporal importance

            **Intrinsic Feature Importance (RF / LR)**
            * RF: mean impurity decrease across all trees
            * LR: |coefficient| per feature across all time-steps
            * Both reshaped (T × F) → same sensor/temporal breakdown
            """
        )
