"""Test suite for run_evaluation.py: verifies dynamic acceptance matrix and failure detection."""

import pytest
from run_evaluation import (
    calculate_holdout_metrics,
    compute_confusion_matrix,
    compute_ece,
    evaluate_acceptance,
    load_json,
)
import numpy as np


def test_acceptance_gate_pass_and_fail():
    lora = load_json("metrics_lora_holdout.json")
    xgb = load_json("metrics_xgb_holdout.json")

    # Default metrics should pass all 6 checks
    checks, all_passed = evaluate_acceptance(lora, xgb, fail_threshold_test=False)
    assert all_passed is True
    assert len(checks) == 6
    assert all(c[4] for c in checks)

    # When a strict threshold fails (e.g. 99% recall), the gate must return False (NO-GO)
    checks_fail, fail_passed = evaluate_acceptance(lora, xgb, fail_threshold_test=True)
    assert fail_passed is False
    assert any(not c[4] for c in checks_fail)


def test_confusion_matrix_calculation():
    # 10 samples: 3 positives, 7 negatives
    probs = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    labels = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])

    cm = compute_confusion_matrix(probs, labels, threshold=0.5)
    # Predicted positives: items 0, 1, 2 (probs >= 0.5) -> labels are 1, 1, 0 -> TP=2, FP=1
    # Predicted negatives: items 3..9 -> labels are 1, 0, 0, 0, 0, 0, 0 -> FN=1, TN=6
    assert cm["tp"] == 2
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tn"] == 6
    assert cm["precision"] == pytest.approx(2 / 3)
    assert cm["recall"] == pytest.approx(2 / 3)
    assert cm["false_positive_rate"] == pytest.approx(1 / 7)


def test_ece_calculation():
    probs = np.array([0.9, 0.1])
    labels = np.array([1, 0])
    ece_val = compute_ece(probs, labels, bins=10)
    assert ece_val == pytest.approx(0.1, abs=1e-3)
