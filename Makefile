.PHONY: install train serve test lint format drift mlflow docker-up clean

install:  ## Install all dependencies
	uv sync

train:  ## Run the full training pipeline (ingest -> register)
	uv run hc-train

serve:  ## Start the API + web UI on http://127.0.0.1:8000
	uv run uvicorn api.main:app --reload

test:  ## Run the test suite with coverage
	uv run pytest --cov=hotel_cancellation --cov=api --cov-report=term-missing

lint:  ## Check code style
	uv run ruff check . && uv run ruff format --check .

format:  ## Auto-format code
	uv run ruff check --fix . && uv run ruff format .

drift:  ## Check a CSV for drift: make drift FILE=path/to/bookings.csv
	uv run hc-drift $(FILE)

mlflow:  ## Browse logged training runs
	uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

docker-up:  ## Build and run the serving container (train first)
	docker compose up --build

clean:  ## Remove generated artifacts and logs (keeps the raw data)
	rm -rf artifacts logs mlruns .pytest_cache .ruff_cache .coverage
