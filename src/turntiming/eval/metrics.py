"""Metrics that reflect how a voice agent actually feels.

Accuracy is close to useless here. The two failure modes are not symmetric:
speaking when the user had not finished is an interruption, and interruptions
are what make an agent unbearable. So the headline numbers are precision,
recall, F1 and -- separately -- the **false-interruption rate**:

    FIR = FP / (FP + TN)

i.e. of all the moments the user was *still going*, how often would the agent
have barged in. Precision answers "when it spoke, was it right?"; FIR answers
"how often did it talk over me?", and only the second is what users complain
about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Signed onset error (predicted - actual), in seconds.
TIMING_BANDS: tuple[tuple[str, float, float], ...] = (
    ("interruptive", -np.inf, -0.20),
    ("natural", -0.20, 0.20),
    ("slightly_late", 0.20, 0.60),
    ("sluggish", 0.60, np.inf),
)


@dataclass
class TurnEndMetrics:
    n: int
    positive_rate: float
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_interruption_rate: float
    tp: int
    fp: int
    tn: int
    fn: int

    def as_row(self) -> str:
        return (
            f"n={self.n:<6d} acc={self.accuracy:.3f} P={self.precision:.3f} "
            f"R={self.recall:.3f} F1={self.f1:.3f} FIR={self.false_interruption_rate:.3f}"
        )


@dataclass
class OnsetMetrics:
    n: int
    mae_ms: float
    median_ae_ms: float
    bands: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> str:
        bands = " ".join(f"{name}={share:.2f}" for name, share in self.bands.items())
        return f"n={self.n:<6d} MAE={self.mae_ms:6.1f}ms median={self.median_ae_ms:6.1f}ms  {bands}"


def turn_end_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> TurnEndMetrics:
    """Classification metrics at one operating point."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return TurnEndMetrics(
        n=len(y_true),
        positive_rate=float(y_true.mean()) if len(y_true) else 0.0,
        threshold=threshold,
        accuracy=(tp + tn) / len(y_true) if len(y_true) else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        false_interruption_rate=fp / (fp + tn) if fp + tn else 0.0,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def onset_metrics(
    y_true_s: np.ndarray, y_pred_s: np.ndarray, mask: np.ndarray | None = None
) -> OnsetMetrics:
    """Response-onset regression error, reported in milliseconds.

    ``mask`` selects which rows to score -- use it to drop censored targets,
    where the true onset is only known to be "further away than the horizon".
    """
    y_true_s = np.asarray(y_true_s, dtype=float)
    y_pred_s = np.asarray(y_pred_s, dtype=float)
    if mask is not None:
        y_true_s, y_pred_s = y_true_s[mask], y_pred_s[mask]
    if len(y_true_s) == 0:
        return OnsetMetrics(n=0, mae_ms=float("nan"), median_ae_ms=float("nan"))

    error = y_pred_s - y_true_s
    bands = {
        name: float(np.mean((error >= low) & (error < high)))
        for name, low, high in TIMING_BANDS
    }
    return OnsetMetrics(
        n=len(y_true_s),
        mae_ms=float(np.mean(np.abs(error)) * 1000.0),
        median_ae_ms=float(np.median(np.abs(error)) * 1000.0),
        bands=bands,
    )


def threshold_for_false_interruption_rate(
    y_true: np.ndarray, y_prob: np.ndarray, max_fir: float = 0.05
) -> float:
    """Best threshold whose false-interruption rate stays within budget.

    This is the operating point a real agent ships at: pick an interruption
    budget you can live with, then take all the recall it buys you. Among the
    thresholds that fit the budget, take the most recall, then the most
    precision -- so a model that is already inside its budget is never pushed
    to a worse operating point just because a lower threshold also fits.
    """
    candidates = np.unique(np.concatenate([np.asarray(y_prob), [0.0, 1.0 + 1e-6]]))
    best, best_key = 1.0 + 1e-6, (-1.0, -1.0)
    for threshold in np.sort(candidates):
        metrics = turn_end_metrics(y_true, y_prob, float(threshold))
        if metrics.false_interruption_rate > max_fir:
            continue
        key = (metrics.recall, metrics.precision)
        if key > best_key:
            best, best_key = float(threshold), key
    return best


def report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    onset_true_s: np.ndarray | None = None,
    onset_pred_s: np.ndarray | None = None,
    onset_mask: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    threshold: float = 0.5,
    max_fir: float = 0.05,
    title: str = "",
) -> str:
    """Human-readable evaluation block, including the per-offset breakdown."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    lines = []
    if title:
        lines.append(title)

    lines.append(f"  turn-end @{threshold:.2f}      {turn_end_metrics(y_true, y_prob, threshold).as_row()}")

    budget_threshold = threshold_for_false_interruption_rate(y_true, y_prob, max_fir)
    budget = turn_end_metrics(y_true, y_prob, budget_threshold)
    lines.append(
        f"  turn-end @FIR<={max_fir:.0%}   {budget.as_row()}  (threshold={budget_threshold:.3f})"
    )

    if offsets is not None:
        offsets = np.asarray(offsets)
        for offset in np.unique(offsets):
            sel = offsets == offset
            row = turn_end_metrics(y_true[sel], y_prob[sel], threshold)
            lines.append(
                f"    offset {float(offset):>4.2f}s      {row.as_row()} base_rate={row.positive_rate:.3f}"
            )

    if onset_true_s is not None and onset_pred_s is not None:
        lines.append(f"  response onset      {onset_metrics(onset_true_s, onset_pred_s, onset_mask).as_row()}")

    return "\n".join(lines)
