"""REST API and web UI for the production cancellation model.

Run from the project root:  uv run uvicorn api.main:app --reload
Then open http://127.0.0.1:8000 for the UI or /docs for the API reference.
"""

import io
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from hotel_cancellation import __version__
from hotel_cancellation.components.data_transformation import FeatureEngineer
from hotel_cancellation.components.data_validation import validate_dataframe
from hotel_cancellation.components.model_evaluation import classification_metrics
from hotel_cancellation.config.configuration import ConfigurationManager
from hotel_cancellation.entity.config_entity import DataValidationConfig
from hotel_cancellation.exception import HotelCancellationError
from hotel_cancellation.logger import get_logger
from hotel_cancellation.monitoring.drift import detect_drift
from hotel_cancellation.pipeline.prediction_pipeline import PredictionPipeline

logger = get_logger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
MAX_BATCH_SIZE = 1000
MAX_CSV_ROWS = 100_000
# Columns echoed back for each row of a scored CSV so the UI can show which booking it was.
PREVIEW_COLUMNS = [
    "hotel",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "lead_time",
    "adr",
    "deposit_type",
    "market_segment",
]

Month = Literal[
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip
NonNegative = Annotated[int, Field(ge=0)]


class Booking(BaseModel):
    """One booking, with the same fields as a row of the raw data (minus the outcome)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "hotel": "City Hotel",
                "lead_time": 120,
                "arrival_date_year": 2017,
                "arrival_date_month": "August",
                "arrival_date_week_number": 33,
                "arrival_date_day_of_month": 15,
                "stays_in_weekend_nights": 1,
                "stays_in_week_nights": 3,
                "adults": 2,
                "children": 0,
                "babies": 0,
                "meal": "BB",
                "country": "PRT",
                "market_segment": "Online TA",
                "distribution_channel": "TA/TO",
                "is_repeated_guest": 0,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 0,
                "reserved_room_type": "A",
                "deposit_type": "No Deposit",
                "agent": 9,
                "company": None,
                "days_in_waiting_list": 0,
                "customer_type": "Transient",
                "adr": 110.5,
                "required_car_parking_spaces": 0,
                "total_of_special_requests": 1,
            }
        },
    )

    hotel: Literal["City Hotel", "Resort Hotel"]
    lead_time: NonNegative = Field(description="Days between booking and arrival")
    arrival_date_year: Annotated[int, Field(ge=2000, le=2100)]
    arrival_date_month: Month
    arrival_date_week_number: Annotated[int, Field(ge=1, le=53)]
    arrival_date_day_of_month: Annotated[int, Field(ge=1, le=31)]
    stays_in_weekend_nights: NonNegative
    stays_in_week_nights: NonNegative
    adults: NonNegative
    children: NonNegative | None = 0
    babies: NonNegative = 0
    meal: Literal["BB", "FB", "HB", "SC", "Undefined"]
    country: str | None = Field(default=None, description="ISO 3166-1 alpha-3 code")
    market_segment: Literal[
        "Aviation", "Complementary", "Corporate", "Direct", "Groups",
        "Offline TA/TO", "Online TA", "Undefined",
    ]  # fmt: skip
    distribution_channel: Literal["Corporate", "Direct", "GDS", "TA/TO", "Undefined"]
    is_repeated_guest: Literal[0, 1] = 0
    previous_cancellations: NonNegative = 0
    previous_bookings_not_canceled: NonNegative = 0
    reserved_room_type: str = Field(min_length=1, max_length=2)
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]
    agent: int | None = Field(default=None, description="Travel agency id, if any")
    company: int | None = Field(default=None, description="Company id, if any")
    days_in_waiting_list: NonNegative = 0
    customer_type: Literal["Contract", "Group", "Transient", "Transient-Party"]
    adr: Annotated[float, Field(ge=0, description="Average daily rate")]
    required_car_parking_spaces: NonNegative = 0
    total_of_special_requests: NonNegative = 0


class FeatureContribution(BaseModel):
    feature: str
    value: str | float | int | None
    contribution: float = Field(description="Effect on the log-odds of cancelling")


class Prediction(BaseModel):
    cancel_probability: float
    will_cancel: bool
    threshold: float
    model_version: str
    explanation: list[FeatureContribution] | None = None


class BatchRequest(BaseModel):
    bookings: list[Booking] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


def load_predictor() -> PredictionPipeline | None:
    try:
        predictor = PredictionPipeline()
    except HotelCancellationError as error:
        logger.warning("No model loaded: %s", error)
        return None
    logger.info("Loaded production model %s", predictor.version)
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = load_predictor()
    yield


app = FastAPI(
    title="Hotel Cancellation API",
    description="Predicts whether a hotel booking will be cancelled.",
    version=__version__,
    lifespan=lifespan,
)


def get_predictor(request: Request) -> PredictionPipeline:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="No model loaded; train a model first")
    return predictor


Predictor = Annotated[PredictionPipeline, Depends(get_predictor)]


def _score(
    predictor: PredictionPipeline, bookings: list[Booking], explain: bool, top_k: int = 5
) -> list[Prediction]:
    df = pd.DataFrame([booking.model_dump() for booking in bookings])
    results = predictor.predict(df)
    explanations = predictor.explain(df, top_k) if explain else [None] * len(df)
    return [
        Prediction(
            cancel_probability=round(float(row.cancel_probability), 4),
            will_cancel=bool(row.will_cancel),
            threshold=round(predictor.threshold, 4),
            model_version=predictor.version,
            explanation=explanation,
        )
        for row, explanation in zip(results.itertuples(), explanations, strict=True)
    ]


@app.get("/health")
def health(request: Request) -> dict:
    predictor = getattr(request.app.state, "predictor", None)
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "model_version": predictor.version if predictor else None,
    }


@app.get("/model")
def model_info(predictor: Predictor) -> dict:
    metadata = predictor.metadata
    return {
        "version": predictor.version,
        "trained_at": metadata.get("trained_at"),
        "threshold": predictor.threshold,
        "test_metrics": metadata.get("test_metrics"),
        "n_features": len(metadata.get("feature_names", [])),
    }


@app.post("/model/reload")
def reload_model(request: Request) -> dict:
    """Load the current production model from the registry (e.g. after retraining)."""
    predictor = load_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="No production model in the registry")
    request.app.state.predictor = predictor
    return {"model_version": predictor.version}


@app.post("/predict", response_model=Prediction)
def predict(
    booking: Booking,
    predictor: Predictor,
    explain: Annotated[bool, Query(description="Include top feature contributions")] = False,
    top_k: Annotated[int, Query(ge=1, le=20, description="Factors to explain")] = 5,
) -> Prediction:
    return _score(predictor, [booking], explain, top_k)[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(
    request: BatchRequest,
    predictor: Predictor,
    explain: Annotated[bool, Query(description="Include top feature contributions")] = False,
    top_k: Annotated[int, Query(ge=1, le=20, description="Factors to explain")] = 5,
) -> BatchResponse:
    return BatchResponse(
        model_version=predictor.version,
        predictions=_score(predictor, request.bookings, explain, top_k),
    )


@lru_cache
def validation_settings() -> DataValidationConfig:
    return ConfigurationManager().get_data_validation_config()


async def read_upload(
    file: Annotated[UploadFile, File(description="CSV of raw bookings")],
    settings: Annotated[DataValidationConfig, Depends(validation_settings)],
) -> pd.DataFrame:
    """Parse an uploaded CSV and check it has every column the model needs."""
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=422, detail=f"Could not read the CSV: {error}") from error
    if df.empty:
        raise HTTPException(status_code=422, detail="The CSV has no rows")
    if len(df) > MAX_CSV_ROWS:
        raise HTTPException(status_code=413, detail=f"At most {MAX_CSV_ROWS} rows per file")

    # The outcome column is optional here: when present it is used to report live metrics.
    columns = {k: v for k, v in settings.columns.items() if k != settings.target}
    result = validate_dataframe(df, columns, settings.target, settings.max_invalid_fraction)
    if not result["is_valid"]:
        raise HTTPException(status_code=422, detail={"errors": result["errors"]})
    df.attrs["warnings"] = result["warnings"]
    return df


Upload = Annotated[pd.DataFrame, Depends(read_upload)]


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> JSON-safe records (NaN becomes null)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


@app.get("/model/versions")
def model_versions(predictor: Predictor) -> dict:
    """Every registered version with its test metrics, and which one is in production."""
    return predictor.registry.index()


@app.post("/predict/csv")
def predict_csv(
    df: Upload,
    predictor: Predictor,
    settings: Annotated[DataValidationConfig, Depends(validation_settings)],
    output: Annotated[Literal["json", "csv"], Query(description="json summary or csv")] = "json",
):
    """Score every booking in a CSV.

    `output=csv` returns the uploaded file with `cancel_probability` and `will_cancel` added.
    With JSON output, metrics are included when the file has an `is_canceled` column.
    """
    results = predictor.predict(df)

    if output == "csv":
        buffer = io.StringIO()
        pd.concat([df, results], axis=1).to_csv(buffer, index=False)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    rows = df[[c for c in PREVIEW_COLUMNS if c in df.columns]].copy()
    rows.insert(0, "row", np.arange(1, len(df) + 1))
    rows["cancel_probability"] = results["cancel_probability"].astype(float).round(4)
    rows["will_cancel"] = results["will_cancel"]

    metrics = None
    target = settings.target
    if target in df.columns and df[target].isin([0, 1]).all():
        rows["actual"] = df[target]
        if df[target].nunique() == 2:
            metrics = classification_metrics(
                df[target].to_numpy(), results["cancel_probability"].to_numpy(), predictor.threshold
            )

    probabilities = results["cancel_probability"].to_numpy()
    return {
        "model_version": predictor.version,
        "threshold": predictor.threshold,
        "n_rows": len(df),
        "n_predicted_cancel": int(results["will_cancel"].sum()),
        "mean_probability": float(probabilities.mean()),
        "histogram": np.histogram(probabilities, bins=20, range=(0, 1))[0].tolist(),
        "metrics": metrics,
        "warnings": df.attrs.get("warnings", []),
        "rows": _records(rows),
    }


@app.post("/monitoring/drift")
def drift_report(
    df: Upload,
    predictor: Predictor,
    settings: Annotated[DataValidationConfig, Depends(validation_settings)],
) -> dict:
    """PSI drift of an uploaded CSV against the production model's training data."""
    report = detect_drift(
        predictor.reference_profile,
        FeatureEngineer().transform(df),
        settings.psi_threshold,
        settings.drift_share_threshold,
    )
    return {"model_version": predictor.version, **report}


# The web UI. Mounted last so the API routes above take precedence.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
