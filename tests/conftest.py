from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from hotel_cancellation.config.configuration import ConfigurationManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]  # fmt: skip


def make_bookings(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Synthetic raw bookings with the real column layout and a learnable cancellation signal."""
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2016-01-01") + pd.to_timedelta(rng.integers(0, 540, n), unit="D")
    deposit = rng.choice(["No Deposit", "Non Refund", "Refundable"], n, p=[0.8, 0.15, 0.05])
    lead_time = rng.integers(0, 400, n)
    special_requests = rng.integers(0, 4, n)
    logit = -1.0 + 3.0 * (deposit == "Non Refund") + lead_time / 150 - 0.6 * special_requests
    canceled = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame(
        {
            "hotel": rng.choice(["City Hotel", "Resort Hotel"], n),
            "is_canceled": canceled,
            "lead_time": lead_time,
            "arrival_date_year": dates.year,
            "arrival_date_month": [MONTHS[m - 1] for m in dates.month],
            "arrival_date_week_number": dates.isocalendar().week.to_numpy(),
            "arrival_date_day_of_month": dates.day,
            "stays_in_weekend_nights": rng.integers(0, 3, n),
            "stays_in_week_nights": rng.integers(0, 6, n),
            "adults": rng.integers(1, 4, n),
            "children": rng.choice([0.0, 1.0, np.nan], n, p=[0.85, 0.14, 0.01]),
            "babies": rng.choice([0, 1], n, p=[0.97, 0.03]),
            "meal": rng.choice(["BB", "HB", "SC"], n),
            "country": rng.choice(["PRT", "GBR", "FRA", "ESP", None], n),
            "market_segment": rng.choice(["Online TA", "Offline TA/TO", "Direct", "Groups"], n),
            "distribution_channel": rng.choice(["TA/TO", "Direct"], n),
            "is_repeated_guest": rng.choice([0, 1], n, p=[0.95, 0.05]),
            "previous_cancellations": rng.choice([0, 1], n, p=[0.9, 0.1]),
            "previous_bookings_not_canceled": rng.choice([0, 1, 2], n, p=[0.9, 0.07, 0.03]),
            "reserved_room_type": rng.choice(["A", "D", "E"], n),
            "assigned_room_type": rng.choice(["A", "D", "E"], n),
            "booking_changes": rng.integers(0, 3, n),
            "deposit_type": deposit,
            "agent": rng.choice([9.0, 240.0, np.nan], n),
            "company": rng.choice([np.nan, 40.0], n, p=[0.9, 0.1]),
            "days_in_waiting_list": rng.choice([0, 10], n, p=[0.95, 0.05]),
            "customer_type": rng.choice(["Transient", "Transient-Party", "Contract"], n),
            "adr": rng.uniform(40, 250, n).round(2),
            "required_car_parking_spaces": rng.choice([0, 1], n, p=[0.9, 0.1]),
            "total_of_special_requests": special_requests,
            "reservation_status": np.where(canceled == 1, "Canceled", "Check-Out"),
            "reservation_status_date": dates.strftime("%Y-%m-%d"),
        }
    )


@pytest.fixture
def bookings() -> pd.DataFrame:
    return make_bookings()


@pytest.fixture
def schema() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "configs" / "schema.yaml").read_text())


@pytest.fixture
def config_manager(tmp_path: Path, bookings: pd.DataFrame) -> ConfigurationManager:
    """A ConfigurationManager whose data and artifacts all live in a temp directory."""
    raw_path = tmp_path / "data" / "hotel_bookings.csv"
    raw_path.parent.mkdir(parents=True)
    bookings.to_csv(raw_path, index=False)

    artifacts = tmp_path / "artifacts"
    config = {
        "artifacts_root": str(artifacts),
        "data_ingestion": {"raw_data_path": str(raw_path), "root_dir": str(artifacts / "ing")},
        **{
            stage: {"root_dir": str(artifacts / stage)}
            for stage in (
                "data_validation",
                "data_transformation",
                "model_trainer",
                "model_evaluation",
                "model_registry",
            )
        },
    }
    params = yaml.safe_load((PROJECT_ROOT / "configs" / "params.yaml").read_text())
    params["model"].update(n_estimators=100, max_depth=3, n_jobs=1)
    params["training"]["early_stopping_rounds"] = 20
    params["preprocessing"]["min_category_frequency"] = 5
    params["evaluation"]["min_roc_auc"] = 0.6
    params["mlflow"]["enabled"] = False

    paths = {}
    for name, content in (("config", config), ("params", params)):
        paths[name] = tmp_path / f"{name}.yaml"
        paths[name].write_text(yaml.safe_dump(content))
    return ConfigurationManager(
        config_path=paths["config"],
        params_path=paths["params"],
        schema_path=PROJECT_ROOT / "configs" / "schema.yaml",
    )
