"""E0: the dumbest baselines worth beating.

Every voice agent on the market endpoints on silence: "quiet for T milliseconds,
therefore done talking". That is the hypothesis under test, so it gets to be the
baseline, in the strongest form a pause duration can take:

``MajorityBaseline``
    Always predict the commoner class. The true floor.
``PauseThresholdBaseline``
    The VAD rule, with the threshold *and the direction* fitted.
``PauseRateBaseline``
    A lookup table from pause duration to the observed turn-end rate. This is
    all the information a pause duration contains, with none of the threshold's
    discretisation loss, and unlike a threshold it emits real probabilities --
    so it can be held to an interruption budget.

The onset side gets an equally dumb partner: predict the training median, always.

Run it::

    python -m turntiming.baselines.pause --data data/ami.jsonl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from turntiming.data.schema import read_jsonl
from turntiming.data.split import SPLIT_KEYS, group_keys, group_split
from turntiming.eval.metrics import report, turn_end_metrics


@dataclass
class PauseThresholdBaseline:
    """Predict "turn ended" from how long the silence has lasted.

    ``polarity`` is fitted rather than assumed. ``longer`` is the conventional
    VAD rule (more silence means more finished); ``shorter`` is its opposite,
    and on real conversation it is frequently the better of the two, because a
    speaker who is merely thinking mid-turn out-waits a partner who is simply
    taking their next turn.
    """

    threshold_s: float = 0.5
    polarity: str = "longer"

    def decision_score(self, pause_s: np.ndarray) -> np.ndarray:
        """Monotone score in [0, 1]; 0.5 is the decision boundary."""
        pause_s = np.asarray(pause_s, dtype=float)
        margin = pause_s - self.threshold_s
        if self.polarity == "shorter":
            margin = -margin
        return 1.0 / (1.0 + np.exp(-margin / 0.1))

    def predict(self, pause_s: np.ndarray) -> np.ndarray:
        return (self.decision_score(pause_s) >= 0.5).astype(int)

    @classmethod
    def fit(
        cls,
        pause_s: np.ndarray,
        y: np.ndarray,
        grid: np.ndarray | None = None,
        objective: str = "youden",
    ) -> PauseThresholdBaseline:
        """Pick the threshold and polarity that score best on this data.

        The default objective is Youden's J (TPR - FPR), not F1. When one class
        is commoner, F1 is maximised by predicting that class everywhere, and a
        baseline that quietly degenerates into the majority class tells you
        nothing about whether pauses carry information. J scores that predictor
        at exactly 0, so any lift it reports is real.
        """
        pause_s = np.asarray(pause_s, dtype=float)
        if grid is None:
            grid = np.unique(np.round(np.concatenate([pause_s, pause_s + 1e-6]), 4))

        def score(metrics) -> float:
            if objective == "f1":
                return metrics.f1
            if objective == "accuracy":
                return metrics.accuracy
            return metrics.recall - metrics.false_interruption_rate

        best, best_score = cls(), -np.inf
        for polarity in ("longer", "shorter"):
            for threshold in grid:
                candidate = cls(float(threshold), polarity)
                value = score(turn_end_metrics(y, candidate.decision_score(pause_s)))
                if value > best_score:
                    best, best_score = candidate, value
        return best


@dataclass
class PauseRateBaseline:
    """P(turn ended | pause so far), read straight off the training data.

    Bins are the distinct observation offsets when there are few of them,
    quantiles otherwise. It cannot represent anything except "how long has it
    been quiet", which is the point: whatever a timing model gains over this, it
    gained from the audio.
    """

    edges: np.ndarray
    rates: np.ndarray
    centers: np.ndarray
    prior: float = 0.5

    @classmethod
    def fit(cls, pause_s: np.ndarray, y: np.ndarray, max_bins: int = 12) -> PauseRateBaseline:
        pause_s = np.asarray(pause_s, dtype=float)
        y = np.asarray(y, dtype=float)
        distinct = np.unique(pause_s)
        if len(distinct) <= max_bins:
            edges = np.concatenate([[-np.inf], (distinct[1:] + distinct[:-1]) / 2, [np.inf]])
        else:
            inner = np.quantile(pause_s, np.linspace(0, 1, max_bins + 1)[1:-1])
            edges = np.concatenate([[-np.inf], np.unique(inner), [np.inf]])

        prior = float(y.mean()) if len(y) else 0.5
        index = np.digitize(pause_s, edges[1:-1])
        rates, centers = [], []
        for b in range(len(edges) - 1):
            in_bin = index == b
            rates.append(float(y[in_bin].mean()) if np.any(in_bin) else prior)
            centers.append(float(pause_s[in_bin].mean()) if np.any(in_bin) else np.nan)
        return cls(edges=edges, rates=np.array(rates), centers=np.array(centers), prior=prior)

    def decision_score(self, pause_s: np.ndarray) -> np.ndarray:
        index = np.digitize(np.asarray(pause_s, dtype=float), self.edges[1:-1])
        return self.rates[np.clip(index, 0, len(self.rates) - 1)]


@dataclass
class ConstantOnsetBaseline:
    """Always predict the same response delay."""

    onset_s: float = 0.5

    @classmethod
    def fit(cls, onset_s: np.ndarray) -> ConstantOnsetBaseline:
        return cls(float(np.median(np.asarray(onset_s, dtype=float))))

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.onset_s, dtype=float)


@dataclass
class MajorityBaseline:
    """Always predict the majority class. The true floor."""

    label: int = 0

    @classmethod
    def fit(cls, y: np.ndarray) -> MajorityBaseline:
        return cls(int(np.mean(np.asarray(y)) >= 0.5))

    def decision_score(self, n: int) -> np.ndarray:
        return np.full(n, float(self.label), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="E0 pause-duration baselines")
    parser.add_argument("--data", required=True, help="examples JSONL from scripts/ingest.py")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-fir", type=float, default=0.05)
    parser.add_argument("--split-by", choices=SPLIT_KEYS, default="conversation")
    parser.add_argument("--objective", choices=("youden", "f1", "accuracy"), default="youden",
                        help="what the pause threshold is fitted to maximise")
    args = parser.parse_args()

    rows = list(read_jsonl(args.data))
    if not rows:
        raise SystemExit(f"No examples in {args.data}")

    y = np.array([r.turn_end for r in rows])
    pause = np.array([r.pause_duration_s for r in rows])
    onset = np.array([r.time_to_next_speaker_s for r in rows])
    uncensored = np.array([not r.onset_censored for r in rows])
    offsets = np.array([r.observation_offset_s for r in rows])

    keys = group_keys([r.to_dict() for r in rows], args.split_by)
    train_idx, val_idx = group_split(keys, args.val_fraction, args.seed)

    majority = MajorityBaseline.fit(y[train_idx])
    pause_model = PauseThresholdBaseline.fit(pause[train_idx], y[train_idx], objective=args.objective)
    rate_model = PauseRateBaseline.fit(pause[train_idx], y[train_idx])
    onset_model = ConstantOnsetBaseline.fit(onset[train_idx][uncensored[train_idx]])

    print(f"examples={len(rows)}  train={len(train_idx)}  val={len(val_idx)}  split_by={args.split_by}")
    print(f"turn_end base rate={y.mean():.3f}  censored onsets={int((~uncensored).sum())}\n")

    print(
        report(
            y[val_idx],
            majority.decision_score(len(val_idx)),
            title=f"E0a majority class (always {majority.label})",
            max_fir=args.max_fir,
        )
    )
    print()
    print(
        report(
            y[val_idx],
            pause_model.decision_score(pause[val_idx]),
            onset_true_s=onset[val_idx],
            onset_pred_s=onset_model.predict(len(val_idx)),
            onset_mask=uncensored[val_idx],
            offsets=offsets[val_idx],
            max_fir=args.max_fir,
            title=(
                f"E0b pause threshold ({pause_model.polarity} than "
                f"{pause_model.threshold_s * 1000:.0f}ms) + constant onset "
                f"({onset_model.onset_s * 1000:.0f}ms)"
            ),
        )
    )
    print()
    print(
        report(
            y[val_idx],
            rate_model.decision_score(pause[val_idx]),
            offsets=offsets[val_idx],
            max_fir=args.max_fir,
            title="E0c pause -> turn-end rate lookup",
        )
    )
    curve = "  ".join(
        f"{center * 1000:.0f}ms -> {rate:.3f}"
        for center, rate in zip(rate_model.centers, rate_model.rates, strict=True)
        if np.isfinite(center)
    )
    print(f"  P(turn ended | silence so far):  {curve}")


if __name__ == "__main__":
    main()
