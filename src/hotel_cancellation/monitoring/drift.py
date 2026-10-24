"""Data drift detection with the Population Stability Index (PSI).

A reference profile (bin edges and proportions per feature) is built from the training data
and stored next to each registered model, so new data can be checked against the
distribution the deployed model was trained on without keeping the training set around.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OTHER = "__other__"
EPSILON = 1e-4
MAX_CATEGORIES = 30


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, EPSILON, None)
    actual = np.clip(actual, EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _bin_shares(values: pd.Series, edges: list[float]) -> np.ndarray:
    # Bins are (-inf, e0], (e0, e1], ..., (e_last, inf).
    bins = np.digitize(values.to_numpy(dtype=float), edges, right=True)
    return np.bincount(bins, minlength=len(edges) + 1) / max(len(values), 1)


def build_reference_profile(
    df: pd.DataFrame, numerical: list[str], categorical: list[str], n_bins: int = 10
) -> dict[str, Any]:
    profile: dict[str, Any] = {"n_rows": len(df), "numerical": {}, "categorical": {}}

    for column in numerical:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        quantiles = np.quantile(values, np.linspace(0, 1, n_bins + 1))
        # Keep the lowest edge so zero-heavy features still get a "<= 0" bin.
        edges = np.unique(quantiles)[:-1].tolist()
        profile["numerical"][column] = {
            "edges": edges,
            "shares": _bin_shares(values, edges).tolist(),
        }

    for column in categorical:
        shares = df[column].astype(str).value_counts(normalize=True)
        top = shares.head(MAX_CATEGORIES)
        result = top.to_dict()
        result[OTHER] = float(max(1.0 - top.sum(), 0.0))
        profile["categorical"][column] = result

    return profile


def detect_drift(
    profile: dict[str, Any],
    current: pd.DataFrame,
    psi_threshold: float = 0.2,
    drift_share_threshold: float = 0.3,
) -> dict[str, Any]:
    """Compare `current` against a reference profile and report PSI per feature."""
    columns: dict[str, dict[str, Any]] = {}

    for column, ref in profile["numerical"].items():
        values = pd.to_numeric(current[column], errors="coerce").dropna()
        if values.empty:
            continue
        psi = _psi(np.array(ref["shares"]), _bin_shares(values, ref["edges"]))
        columns[column] = {"type": "numerical", "psi": round(psi, 4)}

    for column, ref in profile["categorical"].items():
        values = current[column].astype(str)
        known = [c for c in ref if c != OTHER]
        values = values.where(values.isin(known), OTHER)
        actual = values.value_counts(normalize=True)
        categories = list(ref)
        psi = _psi(
            np.array([ref[c] for c in categories]),
            np.array([actual.get(c, 0.0) for c in categories]),
        )
        columns[column] = {"type": "categorical", "psi": round(psi, 4)}

    for result in columns.values():
        result["drifted"] = result["psi"] > psi_threshold

    drifted = sorted(c for c, r in columns.items() if r["drifted"])
    share = len(drifted) / len(columns) if columns else 0.0
    return {
        "n_rows": len(current),
        "n_features": len(columns),
        "drifted_features": drifted,
        "drift_share": round(share, 4),
        "dataset_drift": share > drift_share_threshold,
        "psi_threshold": psi_threshold,
        "drift_share_threshold": drift_share_threshold,
        "features": columns,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: check a CSV of new bookings for drift against the production model's profile."""
    from hotel_cancellation.components.data_transformation import FeatureEngineer
    from hotel_cancellation.components.model_registry import ModelRegistry
    from hotel_cancellation.config.configuration import ConfigurationManager
    from hotel_cancellation.utils.common import read_json, write_json

    parser = argparse.ArgumentParser(description="Check new booking data for drift.")
    parser.add_argument("current", type=Path, help="CSV file with new raw bookings")
    parser.add_argument("--profile", type=Path, help="Reference profile JSON (default: production)")
    parser.add_argument("--output", type=Path, help="Where to write the JSON report")
    args = parser.parse_args(argv)

    config = ConfigurationManager()
    validation = config.get_data_validation_config()
    if args.profile:
        profile_path = args.profile
    else:
        registry = ModelRegistry(config.get_model_registry_config())
        model_dir = registry.production_dir()
        if model_dir is None:
            print("No production model registered; pass --profile.", file=sys.stderr)
            return 2
        profile_path = model_dir / "reference_profile.json"

    current = FeatureEngineer().transform(pd.read_csv(args.current))
    report = detect_drift(
        read_json(profile_path),
        current,
        validation.psi_threshold,
        validation.drift_share_threshold,
    )
    if args.output:
        write_json(args.output, report)

    print(f"Rows checked: {report['n_rows']}")
    for column, result in sorted(report["features"].items(), key=lambda kv: -kv[1]["psi"]):
        flag = "DRIFT" if result["drifted"] else "ok"
        print(f"  {column:<32} PSI={result['psi']:.4f}  {flag}")
    print(
        f"Drifted features: {len(report['drifted_features'])}/{report['n_features']} "
        f"-> dataset drift: {report['dataset_drift']}"
    )
    return 1 if report["dataset_drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
