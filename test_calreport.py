"""Tests for the train-time calibration (math-check) report."""
import json
import math

import numpy as np
import pytest

import pipeline


def _probs_from_rows(rows):
    """Rows of [lead, important, background, noise] weights -> softmax probs."""
    out = []
    for r in rows:
        m = max(r)
        e = [math.exp(x - m) for x in r]
        s = sum(e)
        out.append([x / s for x in e])
    return np.array(out)


def test_calreport_writes_buckets_ece_and_deciles(tmp_path):
    names = ["investigation_lead", "important", "background", "noise"]
    # 6 items confidently relevant + 6 confidently noise
    rows = (
        [[2.0, 1.0, -1.0, -2.0]] * 6
        + [[-2.0, -1.0, 0.5, 2.0]] * 6
    )
    probs = _probs_from_rows(rows)
    y = ["investigation_lead"] * 3 + ["important"] * 3 + ["noise"] * 6
    model = tmp_path / "model_pX_vY.joblib"
    model.write_text("x")

    pipeline._write_calibration_report(probs, names, y, str(model))

    path = pipeline.calibration_report_path(str(model))
    assert path.exists()
    rep = json.loads(path.read_text())
    assert rep["holdout_n"] == 12
    # every relevant item sits in one high bucket, noise in one low bucket
    hi = [b for b in rep["reliability"] if b["range"][0] >= 0.8]
    lo = [b for b in rep["reliability"] if b["range"][1] <= 0.2]
    assert hi and hi[0]["n"] == 6 and hi[0]["obs"] == 1.0
    assert lo and lo[0]["n"] == 6 and lo[0]["obs"] == 0.0
    # honest extremes -> small ECE
    assert rep["ece"] < 0.10
    assert len(rep["composite_deciles"]) == 5
    assert rep["observed_relevant_rate"] == 0.5


def test_calreport_miscalibrated_has_larger_ece(tmp_path):
    names = ["investigation_lead", "important", "background", "noise"]
    rows = [[1.5, 1.0, -1.0, -1.5]] * 8
    probs = _probs_from_rows(rows)
    y = ["noise"] * 8  # model claims relevant, truth says noise
    model = tmp_path / "m.joblib"
    model.write_text("x")

    pipeline._write_calibration_report(probs, names, y, str(model))
    rep = json.loads(pipeline.calibration_report_path(str(model)).read_text())
    assert rep["ece"] > 0.4


def test_calreport_never_raises(tmp_path):
    # wrong class list / ragged input must not raise (telemetry guard)
    pipeline._write_calibration_report(
        np.array([[0.25] * 4]),
        ["a", "b", "c", "d"],
        ["a"],
        str(tmp_path / "missing_dir/m.joblib"),
    )


def test_report_path_sibling_of_model():
    assert str(pipeline.calibration_report_path("/x/model_p1_v2.joblib")).endswith(
        "model_p1_v2.calreport.json"
    )
