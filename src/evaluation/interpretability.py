"""Feature importance and model interpretability for predictive maintenance.

Two complementary approaches are implemented:

1. **Intrinsic importance** (sklearn baselines only)
   - Random Forest: ``feature_importances_`` from sklearn, reshaped to
     (window_size, n_features) so temporal position is visible.
   - Logistic Regression: absolute coefficient magnitudes, same reshape.

2. **Gradient-based input saliency** (neural models: LSTM, CNN)
   Computes dL/dX — the gradient of the scalar output (sum of logits over
   the test batch) with respect to the raw input tensor.  The magnitude
   |dL/dX| at each (timestep, feature) position indicates how much a
   small perturbation there would change the prediction.
   Averaged across the test batch, this gives an approximate heatmap of
   "what the model is attending to".

   This approach is:
   ✓ Fast (single backward pass)
   ✓ Exact for the given model weights
   ✓ Position-aware (unlike permutation importance)
   ✗ Input-gradient methods can be noisy for ReLU/GELU networks
   ✗ Captures local linear sensitivity, not global importance

3. **Permutation importance** (model-agnostic, all models)
   Permutes one sensor across all test windows and measures the drop in F1.
   This is slower but more robust: it tests whether the model *actually uses*
   that feature for classification.

   Interpretation: importance_i = F1_baseline − F1_with_sensor_i_permuted
   Higher = more important.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Intrinsic importance for sklearn baselines
# ──────────────────────────────────────────────────────────────────────────────

def baseline_feature_importance(
    model,
    window_size: int,
    n_features: int,
    sensor_cols: list[str],
) -> np.ndarray:
    """Extract and reshape sklearn model's intrinsic feature importance.

    For Random Forest: ``model.feature_importances_``
    For Logistic Regression: ``|model.coef_[0]|``

    The flattened (window_size * n_features,) importance vector is reshaped
    to (window_size, n_features) so temporal structure is preserved.

    Args:
        model: ``RandomForestBaseline`` or ``LogisticRegressionBaseline`` instance.
        window_size: Number of timesteps per window.
        n_features: Number of sensor features.
        sensor_cols: Sensor column names (length must equal n_features).

    Returns:
        Importance matrix of shape (window_size, n_features).
        Columns correspond to sensor_cols, rows to timestep position (0=oldest).
    """
    # Random Forest
    if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
        importances = model.model.feature_importances_

    # Logistic Regression pipeline
    elif hasattr(model, "pipeline"):
        clf = model.pipeline.named_steps.get("clf")
        if clf is None:
            raise ValueError("LogisticRegression pipeline does not have a 'clf' step.")
        importances = np.abs(clf.coef_[0])

    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    expected = window_size * n_features
    if len(importances) != expected:
        raise ValueError(
            f"Importance vector length {len(importances)} ≠ window_size × n_features "
            f"({window_size} × {n_features} = {expected})."
        )

    # Normalise
    importances = importances / (importances.sum() + 1e-10)
    return importances.reshape(window_size, n_features)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient-based saliency for neural models
# ──────────────────────────────────────────────────────────────────────────────

def neural_gradient_saliency(
    model: torch.nn.Module,
    X_test: np.ndarray,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Compute mean absolute input gradient for a neural model.

    The saliency map S[t, f] = E_batch[ |dOutput/dX[t, f]| ]
    shows on average how sensitive the model's output is to changes
    at timestep t for feature f.

    Args:
        model: PyTorch model (LSTMClassifier or TemporalCNNClassifier).
        X_test: Test input array, shape (N, window_size, n_features).
        device: Computation device.
        batch_size: Mini-batch size for gradient accumulation.

    Returns:
        Saliency matrix of shape (window_size, n_features).
    """
    model = model.to(device)
    model.eval()

    # Re-enable gradients through forward pass
    accumulated_grad = np.zeros(X_test.shape[1:], dtype=np.float64)  # (T, F)
    n_batches = 0

    for start in range(0, len(X_test), batch_size):
        batch = X_test[start : start + batch_size]
        x = torch.from_numpy(batch.astype(np.float32)).to(device)
        x.requires_grad_(True)

        # Forward pass — must use model.forward() so gradients flow
        # (predict_proba wraps in torch.no_grad)
        logits = model.forward(x)

        # Scalar loss: sum of logits (gradient flows uniformly)
        loss = logits.sum()
        loss.backward()

        grad = x.grad.detach().cpu().abs().numpy()   # (batch, T, F)
        accumulated_grad += grad.mean(axis=0)         # average over batch
        n_batches += 1

        # Clear grad for next iteration
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

    saliency = accumulated_grad / max(n_batches, 1)

    # Normalise to [0, 1] for interpretability
    saliency = saliency / (saliency.max() + 1e-10)
    return saliency.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Permutation importance (model-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

def sensor_permutation_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensor_cols: list[str],
    threshold: float = 0.5,
    n_repeats: int = 5,
    random_seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Measure how much performance drops when each sensor is permuted.

    For each sensor column, the values are randomly shuffled across all
    test samples and all timesteps within that sensor.  The baseline F1
    is compared to the permuted F1.  Importance = F1_base − F1_permuted.

    This method answers: "if we removed this sensor entirely, how much
    worse would predictions become?"

    Args:
        predict_fn: Function mapping (N, T, F) array → (N,) probability array.
        X_test: Test input array, shape (N, window_size, n_features).
        y_test: Ground-truth labels.
        sensor_cols: Sensor names (length n_features).
        threshold: Decision threshold for F1 computation.
        n_repeats: Number of permutation repeats (average for stability).
        random_seed: RNG seed.

    Returns:
        Dict mapping sensor_name → {
            "mean_importance": float,
            "std_importance": float,
            "mean_permuted_f1": float,
            "baseline_f1": float,
        }
    """
    rng = np.random.default_rng(random_seed)
    n_features = X_test.shape[2]

    if len(sensor_cols) != n_features:
        raise ValueError(
            f"sensor_cols length ({len(sensor_cols)}) ≠ X_test n_features ({n_features})"
        )

    # Baseline score
    baseline_proba = predict_fn(X_test)
    baseline_pred = (baseline_proba >= threshold).astype(int)
    baseline_f1 = float(f1_score(y_test, baseline_pred, zero_division=0))
    logger.info(f"  Baseline F1: {baseline_f1:.4f}")

    results: dict[str, dict[str, float]] = {}

    for feat_idx, sensor_name in enumerate(sensor_cols):
        repeat_importances: list[float] = []
        repeat_f1s: list[float] = []

        for _ in range(n_repeats):
            X_perm = X_test.copy()
            # Permute this sensor across all samples and timesteps
            n_samples = X_perm.shape[0]
            perm_idx = rng.permutation(n_samples)
            X_perm[:, :, feat_idx] = X_perm[perm_idx, :, feat_idx]

            proba_perm = predict_fn(X_perm)
            pred_perm = (proba_perm >= threshold).astype(int)
            f1_perm = float(f1_score(y_test, pred_perm, zero_division=0))

            importance = baseline_f1 - f1_perm
            repeat_importances.append(importance)
            repeat_f1s.append(f1_perm)

        mean_imp = float(np.mean(repeat_importances))
        std_imp = float(np.std(repeat_importances))
        logger.info(
            f"  {sensor_name:<15}: importance={mean_imp:+.4f} ± {std_imp:.4f}  "
            f"(permuted F1={np.mean(repeat_f1s):.4f})"
        )
        results[sensor_name] = {
            "mean_importance": mean_imp,
            "std_importance": std_imp,
            "mean_permuted_f1": float(np.mean(repeat_f1s)),
            "baseline_f1": baseline_f1,
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Temporal permutation importance
# ──────────────────────────────────────────────────────────────────────────────

def timestep_permutation_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    aggregate_bins: int = 10,
    random_seed: int = 42,
) -> dict[str, list]:
    """Measure importance of each temporal position in the window.

    Permutes one timestep position at a time (across all features and
    all test samples) and measures the resulting F1 drop.

    Computing this for each of the T timestep positions is expensive.
    If T > 50, timesteps are grouped into ``aggregate_bins`` contiguous
    bins and the bin-average importance is returned.

    Args:
        predict_fn: Function mapping (N, T, F) → (N,) probabilities.
        X_test: Shape (N, T, F).
        y_test: Labels.
        threshold: Decision threshold.
        aggregate_bins: Number of temporal bins (≤ T).
        random_seed: Seed.

    Returns:
        Dict with lists of length ``aggregate_bins``:
          "bin_centers", "importances", "bin_labels"
    """
    rng = np.random.default_rng(random_seed)
    T = X_test.shape[1]

    baseline_proba = predict_fn(X_test)
    baseline_f1 = float(f1_score(
        y_test, (baseline_proba >= threshold).astype(int), zero_division=0
    ))

    # Build bin edges
    n_bins = min(aggregate_bins, T)
    bin_edges = np.linspace(0, T, n_bins + 1, dtype=int)

    bin_labels: list[str] = []
    importances: list[float] = []
    bin_centers: list[float] = []

    for i in range(n_bins):
        t_start = int(bin_edges[i])
        t_end = int(bin_edges[i + 1])

        X_perm = X_test.copy()
        n_samples = X_perm.shape[0]
        perm_idx = rng.permutation(n_samples)
        X_perm[:, t_start:t_end, :] = X_perm[perm_idx, t_start:t_end, :]

        proba_perm = predict_fn(X_perm)
        f1_perm = float(f1_score(
            y_test, (proba_perm >= threshold).astype(int), zero_division=0
        ))
        importance = baseline_f1 - f1_perm

        bin_labels.append(f"t={t_start}–{t_end-1}")
        bin_centers.append(float((t_start + t_end) / 2))
        importances.append(importance)

    return {
        "bin_labels": bin_labels,
        "bin_centers": bin_centers,
        "importances": importances,
        "baseline_f1": baseline_f1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: combined importance dict
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_importances(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensor_cols: list[str],
    threshold: float = 0.5,
    model_type: str = "neural",
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Compute all available importances for a given model.

    Args:
        model: Trained model (sklearn baseline or PyTorch neural model).
        X_test: Test array (N, T, F).
        y_test: Test labels.
        sensor_cols: Sensor feature names.
        threshold: Decision threshold.
        model_type: "rf", "lr", "lstm", "cnn".
        device: PyTorch device (for neural models only).

    Returns:
        Dict with keys depending on model_type:
          - "saliency_matrix": (T, F) ndarray (neural only)
          - "intrinsic_matrix": (T, F) ndarray (baseline only)
          - "permutation": per-sensor permutation importance
          - "temporal_permutation": temporal position importance
    """
    window_size = X_test.shape[1]
    n_features = X_test.shape[2]
    result: dict[str, object] = {}

    # Intrinsic importance (sklearn only)
    if model_type in ("rf", "lr"):
        try:
            result["intrinsic_matrix"] = baseline_feature_importance(
                model, window_size, n_features, sensor_cols
            )
        except Exception as e:
            logger.warning(f"Intrinsic importance failed: {e}")

    # Gradient saliency (neural only)
    if model_type in ("lstm", "cnn"):
        logger.info(f"Computing gradient saliency for {model_type.upper()}…")
        try:
            result["saliency_matrix"] = neural_gradient_saliency(
                model, X_test, device=device
            )
        except Exception as e:
            logger.warning(f"Gradient saliency failed: {e}")

    # Permutation importance (all models)
    logger.info("Computing sensor permutation importance…")

    if model_type in ("lstm", "cnn"):
        model_nn = model.to(device) if isinstance(model, torch.nn.Module) else model
        model_nn.eval()

        def predict_fn(X: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return model_nn.predict_proba(
                    torch.from_numpy(X.astype(np.float32)).to(device)
                ).cpu().numpy()
    else:
        def predict_fn(X: np.ndarray) -> np.ndarray:  # type: ignore[misc]
            return model.predict_proba(X)

    result["permutation"] = sensor_permutation_importance(
        predict_fn, X_test, y_test, sensor_cols, threshold=threshold
    )

    logger.info("Computing temporal permutation importance…")
    result["temporal_permutation"] = timestep_permutation_importance(
        predict_fn, X_test, y_test, threshold=threshold
    )

    return result
