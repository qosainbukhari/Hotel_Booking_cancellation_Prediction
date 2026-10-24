import numpy as np
import pandas as pd

from hotel_cancellation.monitoring.drift import build_reference_profile, detect_drift


def _frame(n: int, shift: float = 0.0, city_share: float = 0.5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "lead_time": rng.normal(100 + shift, 30, n),
            "previous_cancellations": rng.choice([0, 1], n, p=[0.9, 0.1]),
            "hotel": rng.choice(["City", "Resort"], n, p=[city_share, 1 - city_share]),
        }
    )


def _profile(df):
    return build_reference_profile(df, ["lead_time", "previous_cancellations"], ["hotel"])


def test_same_distribution_has_no_drift():
    report = detect_drift(_profile(_frame(5000)), _frame(5000, seed=1))
    assert report["drifted_features"] == []
    assert not report["dataset_drift"]


def test_shifted_numeric_and_categorical_drift():
    current = _frame(5000, shift=80, city_share=0.95, seed=1)
    report = detect_drift(_profile(_frame(5000)), current, drift_share_threshold=0.5)
    assert set(report["drifted_features"]) == {"lead_time", "hotel"}
    assert report["dataset_drift"]


def test_zero_heavy_feature_is_still_monitored():
    current = _frame(5000, seed=1)
    current["previous_cancellations"] = 1
    report = detect_drift(_profile(_frame(5000)), current)
    assert report["features"]["previous_cancellations"]["drifted"]


def test_unseen_categories_count_as_other():
    current = _frame(1000, seed=1)
    current["hotel"] = "Hostel"
    report = detect_drift(_profile(_frame(1000)), current)
    assert report["features"]["hotel"]["drifted"]
