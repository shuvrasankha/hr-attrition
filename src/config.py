"""Central configuration and path constants for the pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
RUNS_DIR = ROOT_DIR / "runs"

for d in (BRONZE_DIR, SILVER_DIR, GOLD_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# LLM config — GitHub Models API (OpenAI-compatible endpoint).
# If GITHUB_TOKEN is not set, agents fall back to deterministic
# template-based text generation so the pipeline still runs end-to-end offline.
GITHUB_MODELS_URL = "https://models.inference.ai.azure.com"
GITHUB_MODEL = "gpt-4o"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# Business definition defaults (used as the *default* proposed rule; the user
# can edit/override these at the approval checkpoint).
DEFAULT_TOP_PERFORMER_MIN_RATING = 4
DEFAULT_TOP_PERFORMER_LOOKBACK_CYCLES = 2
