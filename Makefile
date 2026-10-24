.PHONY: install train repro registry serve test lint format drift mlflow docker-up vercel clean

install:  ## Install all dependencies
	uv sync

train:  ## Run the full training pipeline (ingest -> register)
	uv run hc-train

repro:  ## Re-run only the DVC stages whose code, params or data changed
	uv run dvc repro

registry:  ## List model versions in the MLflow registry
	uv run hc-registry list

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

docker-up:  ## Build and run the serving container (train first: the bundle is baked in)
	docker compose up --build

vercel:  ## Deploy API + UI to Vercel (needs Node.js and a one-time `npx vercel login`)
	uv run python scripts/build_vercel.py
	cd build/cancellation-radar && npx --yes vercel deploy --prod --yes

clean:  ## Remove generated artifacts and logs (keeps the raw data)
	rm -rf artifacts logs mlruns .pytest_cache .ruff_cache .coverage
