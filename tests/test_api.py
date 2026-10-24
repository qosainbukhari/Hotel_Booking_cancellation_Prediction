import pytest
from fastapi.testclient import TestClient

from api.main import Booking, app, get_predictor, validation_settings
from hotel_cancellation.pipeline.prediction_pipeline import PredictionPipeline
from hotel_cancellation.pipeline.training_pipeline import TrainingPipeline
from tests.conftest import make_bookings, make_config

EXAMPLE = Booking.model_config["json_schema_extra"]["example"]


@pytest.fixture
def csv_bytes(bookings):
    return bookings.head(300).to_csv(index=False).encode()


@pytest.fixture(scope="module")
def trained_config(tmp_path_factory):
    """Train once for the whole module; the API tests only read the serving bundle."""
    config_manager = make_config(tmp_path_factory.mktemp("api"), make_bookings())
    TrainingPipeline(config_manager).run()
    return config_manager


@pytest.fixture
def client(trained_config):
    config_manager = trained_config
    predictor = PredictionPipeline(config_manager.get_model_registry_config().serving_dir)
    settings = config_manager.get_data_validation_config()
    app.dependency_overrides[get_predictor] = lambda: predictor
    app.dependency_overrides[validation_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info(client):
    body = client.get("/model").json()
    assert body["version"] == "1"
    assert body["costs"] == {"false_negative": 100.0, "false_positive": 20.0}
    assert 0 < body["threshold"] < 1


def test_predict_single(client):
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["cancel_probability"] <= 1
    assert body["model_version"] == "1"
    assert body["explanation"] is None


def test_predict_with_explanation(client):
    body = client.post("/predict", params={"explain": True}, json=EXAMPLE).json()
    assert len(body["explanation"]) == 5


def test_predict_batch(client):
    other = {**EXAMPLE, "deposit_type": "Non Refund", "agent": None}
    response = client.post("/predict/batch", json={"bookings": [EXAMPLE, other]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"hotel": "Motel"},
        {"lead_time": -1},
        {"arrival_date_month": "Smarch"},
        {"reservation_status": "Canceled"},  # leaking fields are rejected
        {"country": "PRT"},
        {"required_car_parking_spaces": 1},
    ],
)
def test_invalid_booking_is_rejected(client, change):
    assert client.post("/predict", json={**EXAMPLE, **change}).status_code == 422


def test_predict_without_model_returns_503():
    with TestClient(app) as test_client:
        test_client.app.state.predictor = None
        assert test_client.post("/predict", json=EXAMPLE).status_code == 503


def test_frontend_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Cancellation Radar" in response.text
    assert client.get("/app.js").status_code == 200


def test_model_versions(client):
    body = client.get("/model/versions").json()
    assert body["production"] == "1"
    assert [v["version"] for v in body["versions"]] == ["1"]


def test_predict_csv_with_labels_reports_metrics(client, csv_bytes):
    response = client.post("/predict/csv", files={"file": ("b.csv", csv_bytes, "text/csv")})
    assert response.status_code == 200
    body = response.json()
    assert body["n_rows"] == 300 == len(body["rows"])
    assert sum(body["histogram"]) == 300
    assert 0 <= body["metrics"]["roc_auc"] <= 1
    assert "cost_per_booking" in body["metrics"]
    assert {"row", "cancel_probability", "will_cancel", "actual"} <= body["rows"][0].keys()


def test_predict_csv_download_appends_predictions(client, csv_bytes):
    response = client.post(
        "/predict/csv", params={"output": "csv"}, files={"file": ("b.csv", csv_bytes)}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    header = response.text.splitlines()[0].split(",")
    assert header[-2:] == ["cancel_probability", "will_cancel"]
    assert len(response.text.strip().splitlines()) == 301


def test_predict_csv_rejects_missing_columns(client):
    response = client.post("/predict/csv", files={"file": ("b.csv", b"a,b\n1,2\n")})
    assert response.status_code == 422
    assert "Missing columns" in response.json()["detail"]["errors"][0]


def test_drift_report(client, csv_bytes):
    response = client.post("/monitoring/drift", files={"file": ("b.csv", csv_bytes)})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "1"
    assert body["n_features"] > 0 and "dataset_drift" in body


def test_predict_csv_returns_riskiest_rows_with_whole_file_counts(client, bookings):
    csv_bytes = bookings.to_csv(index=False).encode()  # 1,500 rows
    body = client.post("/predict/csv", files={"file": ("b.csv", csv_bytes)}).json()

    assert body["n_rows"] == 1500
    assert len(body["rows"]) == 500
    assert body["n_actual_cancel"] == int(bookings["is_canceled"].sum())
    probabilities = [row["cancel_probability"] for row in body["rows"]]
    assert probabilities == sorted(probabilities, reverse=True)
