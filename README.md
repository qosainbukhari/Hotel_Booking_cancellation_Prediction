# Hotel Cancellation Prediction

Predicts whether a hotel booking will be cancelled, using the
[Hotel Booking Demand](https://www.sciencedirect.com/science/article/pii/S2352340918315191) dataset
(119,390 bookings from a city hotel and a resort hotel in Portugal, July 2015 – August 2017).

The project covers the whole lifecycle: data validation, training, evaluation against the live
model, a versioned model registry, drift monitoring, a REST API and a web UI. It also ships with
Docker and CI.

![Single booking prediction with explanation](docs/images/predict.png)

## Quick start

```bash
uv sync                      # install
# put the dataset at data/raw/hotel_bookings.csv
uv run hc-train              # train, evaluate, register the model
uv run uvicorn api.main:app  # open http://127.0.0.1:8000
```

Or use `make install train serve`.

## Results

On the held-out test set (the most recent 20% of arrivals, 23,878 bookings from 22 Apr 2017):

| Metric    | Value |
|-----------|-------|
| ROC-AUC   | 0.875 |
| PR-AUC    | 0.834 |
| F1        | 0.738 |
| Precision | 0.766 |
| Recall    | 0.711 |
| Accuracy  | 0.793 |

The decision threshold (≈0.29) is tuned for F1 on a validation slice, not on the test set.

These numbers are lower than the ~0.95 AUC often quoted for this dataset. That is on purpose:

- **No leakage.** `reservation_status` and `reservation_status_date` encode the answer directly,
  and `assigned_room_type` and `booking_changes` keep changing up to check-in or cancellation.
  All four are dropped as soon as the data is loaded.
- **Temporal split.** The model is trained on older bookings and tested on newer ones, as it
  would be used in practice. A random split mixes future bookings into training and
  overstates accuracy.

## Project layout

```
configs/
  config.yaml          file locations
  params.yaml          split sizes, model hyper-parameters, acceptance rules, drift thresholds
  schema.yaml          expected columns, value rules, leakage columns, model features
src/hotel_cancellation/
  components/
    data_ingestion.py       load raw CSV, drop leakage columns, temporal train/test split
    data_validation.py      schema checks, reference profile, train→test drift report
    data_transformation.py  feature engineering + encoding (sklearn Pipeline)
    model_trainer.py        XGBoost with early stopping, F1-optimal threshold
    model_evaluation.py     test metrics, comparison with the production model
    model_registry.py       file-based registry with versions and a production pointer
  monitoring/drift.py       PSI drift detection (library + `hc-drift` CLI)
  pipeline/
    training_pipeline.py    runs all stages (`hc-train`)
    prediction_pipeline.py  loads the production model, predicts and explains
  config/configuration.py   turns YAML into typed config objects
  entity/                   dataclasses for stage configs and outputs
api/main.py                 FastAPI service (JSON API + serves the web UI)
frontend/                   web UI: plain HTML/CSS/JS, no build step, no CDN
notebooks/
  project_walkthrough.ipynb EDA, leakage/split experiment, test curves, explanations, drift
tests/                      unit, API and end-to-end tests (synthetic data, no CSV needed)
Dockerfile, docker-compose.yml, Makefile, .github/workflows/ci.yml
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Put the dataset at `data/raw/hotel_bookings.csv` (the `data/` folder is not in Git).

## Train

```bash
uv run hc-train
```

The pipeline runs six stages:

1. **Ingestion:** reads the CSV, drops the leakage columns, sorts by arrival date and keeps the
   newest 20% for testing.
2. **Validation:** checks every column against `schema.yaml`. Missing columns, wrong types or a
   bad target fail the run. Rows that break a range or allowed-value rule only cause a warning
   until they exceed `max_invalid_fraction` (1%). It also saves a reference profile of the
   training data and reports train→test drift.
3. **Transformation:** derives features (`total_nights`, `total_guests`, `adr_per_guest`,
   `arrival_weekday`, `has_company`, …) and one-hot encodes categories. Categories seen fewer
   than 50 times are grouped into one bucket, so new countries or agents don't break scoring.
4. **Training:** XGBoost, early-stopped on the newest 15% of the training rows. The same slice is
   used to pick the decision threshold.
5. **Evaluation:** scores the candidate on the test set. If a production model exists, it is
   scored on the same test set.
6. **Registry:** the candidate is promoted only if its ROC-AUC is ≥ `min_roc_auc` (0.80) and
   beats production by at least `min_improvement` (0.002).

Outputs go to `artifacts/`:

- `data_validation/validation_report.json`
- `model_trainer/metadata.json`
- `model_evaluation/metrics.json`
- `model_registry/versions/vN/`

Each registered version holds the model, its metadata and its reference profile. Runs are also
logged to MLflow (`sqlite:///mlruns/mlflow.db`; set `mlflow.enabled: false` in `params.yaml` to
turn this off). To browse them:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

The saved model is a single sklearn `Pipeline` that takes raw booking rows. Training and serving
share the same feature code, so they cannot drift apart.

## Web UI and API

```bash
uv run uvicorn api.main:app --reload
```

- **http://127.0.0.1:8000** is the web UI.
- **http://127.0.0.1:8000/docs** is the interactive API reference.

The UI has four tabs:

| Tab | What it does |
|-----|--------------|
| **Single booking** | Fill in a booking (or load a low/high-risk preset) to get the cancellation probability on a dial, with the decision threshold marked. Also shows the eight factors that pushed the risk up or down. |
| **Batch scoring** | Upload a CSV to get summary tiles, the risk distribution and the riskiest bookings, plus a download of the full file with predictions added. If the file has `is_canceled`, it also shows live ROC-AUC, precision, recall and F1. |
| **Drift monitor** | Upload recent bookings to get PSI per feature against the production model's training data, plus an overall drift verdict. |
| **Model** | Shows the current version, its test metrics, the full version history, and a button to load a newly promoted model without restarting. |

The form's dropdown options come from the API's OpenAPI schema, so the UI and the API's
validation always match.

![Batch scoring](docs/images/batch.png)

### Endpoints

| Method | Path                | Purpose                                                       |
|--------|---------------------|---------------------------------------------------------------|
| GET    | `/health`           | Liveness, and whether a model is loaded                       |
| GET    | `/model`            | Version, training date, threshold, test metrics               |
| GET    | `/model/versions`   | Registry history and which version is in production           |
| POST   | `/model/reload`     | Load the current production model after retraining            |
| POST   | `/predict`          | Score one booking (`?explain=true&top_k=8` adds top drivers)  |
| POST   | `/predict/batch`    | Score up to 1,000 bookings sent as JSON                       |
| POST   | `/predict/csv`      | Score an uploaded CSV (≤100k rows); `?output=csv` returns the file with predictions |
| POST   | `/monitoring/drift` | PSI drift report for an uploaded CSV                          |

```bash
curl -X POST 'http://127.0.0.1:8000/predict?explain=true' \
  -H 'content-type: application/json' \
  -d '{"hotel":"City Hotel","lead_time":120,"arrival_date_year":2017,
       "arrival_date_month":"August","arrival_date_week_number":33,
       "arrival_date_day_of_month":15,"stays_in_weekend_nights":1,"stays_in_week_nights":3,
       "adults":2,"children":0,"babies":0,"meal":"BB","country":"PRT",
       "market_segment":"Online TA","distribution_channel":"TA/TO","reserved_room_type":"A",
       "deposit_type":"No Deposit","agent":9,"customer_type":"Transient","adr":110.5,
       "total_of_special_requests":1}'

curl -F file=@artifacts/data_ingestion/test.csv 'http://127.0.0.1:8000/predict/csv?output=csv' \
  -o predictions.csv
```

```json
{
  "cancel_probability": 0.2646,
  "will_cancel": false,
  "threshold": 0.2947,
  "model_version": "v1",
  "explanation": [
    {"feature": "total_of_special_requests", "value": 1, "contribution": -0.6157},
    {"feature": "country", "value": "PRT", "contribution": 0.4447},
    {"feature": "deposit_type", "value": "No Deposit", "contribution": -0.4234},
    {"feature": "lead_time", "value": 120, "contribution": 0.2652},
    {"feature": "market_segment", "value": "Online TA", "contribution": 0.159}
  ]
}
```

Explanations are XGBoost's built-in TreeSHAP values in log-odds. One-hot columns are summed back
into their source feature. Positive values push towards cancellation.

The API rejects invalid input with a 422 error: unknown categories in fixed sets, negative
counts, leakage fields such as `reservation_status`, and CSVs with missing columns.

## Monitor drift

![Drift monitor](docs/images/drift.png)

Use the **Drift monitor** tab, `POST /monitoring/drift`, or the CLI to compare a CSV of new
bookings with the production model's training distribution:

```bash
uv run hc-drift path/to/new_bookings.csv --output drift_report.json
```

Each feature gets a Population Stability Index (PSI). A feature counts as drifted above 0.2.
The dataset counts as drifted when more than 30% of features drift; in that case the command
exits with code 1, so it can gate a scheduled retraining job.

The train→test check already flags `arrival_month`, `arrival_date_week_number`, `adr`,
`adr_per_guest`, `agent` and `lead_time`. That is expected: the test window is spring–summer
2017, so seasonality and price changes show up. It is also the main reason test AUC is lower
than validation AUC (0.875 vs 0.911).

## Notebook

`notebooks/project_walkthrough.ipynb` walks through the project with code and charts. It is saved
with its outputs, so you can read it without running anything:

1. **The data:** cancellation rates by hotel, deposit, market segment, lead time and month.
2. **Leakage and the split:** the same model trained three ways.
3. **The production model:** ROC and precision/recall curves, confusion matrix, calibration.
4. **What drives predictions:** global TreeSHAP importance and one booking explained.
5. **Drift:** PSI between the training and test periods.

The leakage experiment shows what each choice is worth:

| Setup | Test ROC-AUC |
|-------|-------------:|
| Random split + leaky columns | 0.954 |
| Random split, no leakage | 0.948 |
| Time split, no leakage (this project) | 0.882 |

The calibration plot shows that on the test period, probabilities run low: bookings cancel more
often than predicted, because of seasonal drift.

To re-run it after `uv run hc-train`:

```bash
uv run --with jupyterlab jupyter lab notebooks/   # or open it in VS Code with the .venv kernel
```

It reads the raw data and the trained model but never writes to `artifacts/`.

## Docker

The image serves the API and UI, with runtime dependencies only. The model registry is mounted
from the host, so you can retrain without rebuilding:

```bash
uv run hc-train
docker compose up --build        # http://localhost:8000
```

After retraining on the host, press **Reload from registry** in the Model tab, or call
`POST /model/reload`.

## Make targets

| Target | Does |
|--------|------|
| `make install` | `uv sync` |
| `make train` | run the training pipeline |
| `make serve` | API + UI with auto-reload |
| `make test` | tests with coverage |
| `make lint` / `make format` | ruff check / fix |
| `make drift FILE=…` | drift check from the CLI |
| `make mlflow` | browse training runs |
| `make docker-up` | build and run the container |

## Test and lint

```bash
make test    # or: uv run pytest
make lint
```

CI (`.github/workflows/ci.yml`) runs lint and tests on every push and pull request. It installs
only the runtime and dev dependencies; MLflow, Optuna and the other training extras aren't
needed. Without MLflow installed, `hc-train` still runs and simply skips tracking.

The tests use synthetic bookings, so they run without the dataset. They cover:

- schema validation
- feature engineering
- the temporal split
- drift detection
- an end-to-end training run, including the "no improvement → keep production" rule
- the API, including CSV upload, drift and the served UI

Coverage is 88%.
