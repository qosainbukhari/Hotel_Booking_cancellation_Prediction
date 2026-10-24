"""Stage a self-contained Vercel deployment of the API + web UI.

Vercel installs whatever the project's manifest resolves. The repo's pyproject.toml pulls in the
dev and train groups by default (MLflow, DVC, Evidently, ...), which would blow the 500 MB
function limit, so the deployment gets its own folder with:

    index.py            Vercel entrypoint (exposes the FastAPI `app`)
    requirements.txt    runtime packages only, pinned to the versions the model was trained with
    .python-version     3.12 (Vercel supports 3.12+)
    vercel.json         function settings
    api/ frontend/ configs/ src/hotel_cancellation/ artifacts/serving/

Usage:
    uv run hc-train                                # produces artifacts/serving
    uv run python scripts/build_vercel.py          # writes build/cancellation-radar/
    cd build/cancellation-radar && npx vercel deploy --prod
(or simply `make vercel`)
"""

import argparse
import json
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The Vercel CLI names the project after this folder.
DEFAULT_OUT = ROOT / "build" / "cancellation-radar"
PYTHON_VERSION = "3.12"

# Packages the API imports at runtime. Exact versions match the training environment so the
# pickled model loads the same way it was saved.
RUNTIME_PACKAGES = [
    "fastapi",
    "pydantic",
    "python-multipart",
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "xgboost-cpu",
    "joblib",
    "pyyaml",
    "pandera",
]

ENTRYPOINT = '''"""Vercel entrypoint: the FastAPI app (JSON API + web UI)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from api.main import app  # noqa: E402, F401
'''

VERCEL_CONFIG = {
    "$schema": "https://openapi.vercel.sh/vercel.json",
    "functions": {"index.py": {"maxDuration": 60}},
}


def build(out: Path) -> None:
    serving = ROOT / "artifacts" / "serving"
    if not (serving / "model.joblib").exists():
        sys.exit("No serving bundle in artifacts/serving. Run `uv run hc-train` first.")

    # Keep the CLI's project link (.vercel/) so redeploys update the same Vercel project.
    link = out / ".vercel"
    saved_link = None
    if link.exists():
        saved_link = out.parent / ".vercel-link"
        shutil.rmtree(saved_link, ignore_errors=True)
        shutil.move(link, saved_link)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if saved_link is not None:
        shutil.move(saved_link, link)

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ("api", "frontend", "configs"):
        shutil.copytree(ROOT / name, out / name, ignore=ignore)
    shutil.copytree(
        ROOT / "src" / "hotel_cancellation", out / "src" / "hotel_cancellation", ignore=ignore
    )
    shutil.copytree(serving, out / "artifacts" / "serving")

    (out / "index.py").write_text(ENTRYPOINT)
    (out / ".python-version").write_text(PYTHON_VERSION + "\n")
    (out / "vercel.json").write_text(json.dumps(VERCEL_CONFIG, indent=2) + "\n")
    requirements = [f"{name}=={version(name)}" for name in RUNTIME_PACKAGES]
    (out / "requirements.txt").write_text("\n".join(requirements) + "\n")

    model_version = json.loads((serving / "metadata.json").read_text())["version"]
    print(f"Staged model v{model_version} for Vercel in {out}")
    print("Runtime packages:\n  " + "\n  ".join(requirements))


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage the Vercel deployment folder.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output folder")
    build(parser.parse_args().out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
