import numpy as np

from turntiming.eval.metrics import (
    onset_metrics,
    threshold_for_false_interruption_rate,
    turn_end_metrics,
)


def test_confusion_matrix_and_false_interruption_rate() -> None:
    #                    tp   fn   fp   tn   tn
    y_true = np.array([1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.2])
    m = turn_end_metrics(y_true, y_prob)

    assert (m.tp, m.fn, m.fp, m.tn) == (1, 1, 1, 2)
    assert m.precision == 0.5
    assert m.recall == 0.5
    # One barge-in across three moments the speaker was still going.
    assert m.false_interruption_rate == 1 / 3


def test_false_interruption_rate_differs_from_precision() -> None:
    """With few positives, precision can crater while interruptions stay rare."""
    y_true = np.array([1] + [0] * 99)
    y_prob = np.array([1.0, 1.0] + [0.0] * 98)
    m = turn_end_metrics(y_true, y_prob)
    assert m.precision == 0.5
    assert m.false_interruption_rate == 1 / 99


def test_threshold_respects_interruption_budget() -> None:
    y_true = np.array([1] * 50 + [0] * 50)
    y_prob = np.concatenate([np.linspace(0.5, 1.0, 50), np.linspace(0.0, 0.6, 50)])
    threshold = threshold_for_false_interruption_rate(y_true, y_prob, max_fir=0.05)
    assert turn_end_metrics(y_true, y_prob, threshold).false_interruption_rate <= 0.05


def test_threshold_keeps_the_best_point_inside_budget() -> None:
    """A model already inside budget must not be pushed to a worse threshold."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.9, 0.1, 0.1])
    threshold = threshold_for_false_interruption_rate(y_true, y_prob, max_fir=0.05)
    m = turn_end_metrics(y_true, y_prob, threshold)
    assert (m.recall, m.precision) == (1.0, 1.0)


def test_onset_metrics_report_milliseconds_and_bands() -> None:
    m = onset_metrics(np.array([1.0, 1.0]), np.array([1.5, 0.5]))
    assert m.mae_ms == 500.0
    assert m.bands["slightly_late"] == 0.5
    assert m.bands["interruptive"] == 0.5


def test_onset_mask_drops_censored_rows() -> None:
    m = onset_metrics(
        np.array([1.0, 10.0]), np.array([1.0, 2.0]), mask=np.array([True, False])
    )
    assert m.n == 1
    assert m.mae_ms == 0.0
