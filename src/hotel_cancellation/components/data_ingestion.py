import pandas as pd

from hotel_cancellation.entity.artifact_entity import DataIngestionArtifact
from hotel_cancellation.entity.config_entity import DataIngestionConfig
from hotel_cancellation.exception import DataValidationError
from hotel_cancellation.logger import get_logger
from hotel_cancellation.utils.common import split_tail

logger = get_logger(__name__)

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}  # fmt: skip


def arrival_dates(df: pd.DataFrame) -> pd.Series:
    """Build the arrival date from the year / month-name / day columns (NaT when invalid)."""
    return pd.to_datetime(
        pd.DataFrame(
            {
                "year": df["arrival_date_year"],
                "month": df["arrival_date_month"].map(MONTHS),
                "day": df["arrival_date_day_of_month"],
            }
        ),
        errors="coerce",
    )


class DataIngestion:
    """Load the raw bookings, drop leaking columns and make a temporal train / test split.

    The split is by arrival date (oldest bookings for training, newest for testing) because
    the model is used to score future bookings; a random split would overstate its accuracy.
    """

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def run(self) -> DataIngestionArtifact:
        logger.info("Reading raw data from %s", self.config.raw_data_path)
        if not self.config.raw_data_path.exists():
            raise DataValidationError(f"Raw data not found: {self.config.raw_data_path}")
        df = pd.read_csv(self.config.raw_data_path)
        if df.empty:
            raise DataValidationError("Raw data file is empty")

        df = df.drop(columns=[c for c in self.config.leakage_columns if c in df.columns])

        dates = arrival_dates(df)
        if dates.isna().any():
            logger.warning("Dropping %d rows with an invalid arrival date", dates.isna().sum())
        df = df.loc[dates.notna()]
        order = dates.loc[dates.notna()].sort_values(kind="stable").index
        df = df.loc[order].reset_index(drop=True)

        train, test = split_tail(df, self.config.test_size)
        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(self.config.train_path, index=False)
        test.to_csv(self.config.test_path, index=False)

        test_start = arrival_dates(test).min().date()
        logger.info(
            "Saved %d train rows and %d test rows (test arrivals from %s)",
            len(train),
            len(test),
            test_start,
        )
        return DataIngestionArtifact(
            train_path=self.config.train_path, test_path=self.config.test_path
        )
