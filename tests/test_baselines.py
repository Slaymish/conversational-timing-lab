import numpy as np
import pytest

from turntiming.baselines.pause import (
    ConstantOnsetBaseline,
    MajorityBaseline,
    PauseRateBaseline,
    PauseThresholdBaseline,
)


def test_pause_baseline_learns_conventional_polarity() -> None:
    """Long pause -> turn ended. The rule every VAD endpointer assumes."""
    pause = np.array([0.1, 0.1, 0.2, 0.2, 0.9, 0.9, 1.2, 1.2])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = PauseThresholdBaseline.fit(pause, y)
    assert model.polarity == "longer"
    assert np.array_equal(model.predict(pause), y)


def test_pause_baseline_learns_inverted_polarity() -> None:
    """Real conversation often runs the other way; the baseline must be allowed to."""
    pause = np.array([0.1, 0.1, 0.2, 0.2, 0.9, 0.9, 1.2, 1.2])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    model = PauseThresholdBaseline.fit(pause, y)
    assert model.polarity == "shorter"
    assert np.array_equal(model.predict(pause), y)


def test_constant_onset_baseline_uses_median() -> None:
    model = ConstantOnsetBaseline.fit(np.array([0.2, 0.4, 0.6]))
    assert model.onset_s == 0.4
    assert np.all(model.predict(3) == 0.4)


def test_majority_baseline() -> None:
    assert MajorityBaseline.fit(np.array([0, 0, 0, 1])).label == 0
    assert MajorityBaseline.fit(np.array([1, 1, 1, 0])).label == 1


def test_pause_threshold_fit_does_not_degenerate_to_majority() -> None:
    """With an 80/20 split, F1 prefers 'always yes'; Youden must find the signal."""
    pause = np.array([0.1] * 50 + [1.0] * 50)
    y = np.array([1] * 45 + [0] * 5 + [1] * 35 + [0] * 15)

    youden = PauseThresholdBaseline.fit(pause, y, objective="youden")
    assert youden.polarity == "shorter"
    assert youden.predict(np.array([0.1, 1.0])).tolist() == [1, 0]

    f1_fitted = PauseThresholdBaseline.fit(pause, y, objective="f1")
    assert f1_fitted.predict(pause).sum() == len(pause)  # degenerates, as expected


def test_pause_rate_baseline_recovers_the_conditional_rate() -> None:
    pause = np.array([0.1] * 100 + [1.0] * 100)
    y = np.array([1] * 80 + [0] * 20 + [1] * 30 + [0] * 70)
    model = PauseRateBaseline.fit(pause, y)
    scores = model.decision_score(np.array([0.1, 1.0]))
    assert scores[0] == pytest.approx(0.8)
    assert scores[1] == pytest.approx(0.3)


def test_pause_rate_baseline_bins_continuous_pauses() -> None:
    rng = np.random.default_rng(0)
    pause = rng.random(500)
    y = (pause < 0.5).astype(int)
    model = PauseRateBaseline.fit(pause, y, max_bins=10)
    assert model.decision_score(np.array([0.05]))[0] > model.decision_score(np.array([0.95]))[0]
