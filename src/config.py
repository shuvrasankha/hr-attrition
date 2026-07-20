"""Central configuration and path constants for the pipeline."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
RUNS_DIR = ROOT_DIR / "runs"

for d in (BRONZE_DIR, SILVER_DIR, GOLD_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# LLM config. If ANTHROPIC_API_KEY is not set, agents fall back to deterministic
# template-based text generation so the pipeline still runs end-to-end offline.
ANTHROPIC_MODEL = "claude-sonnet-4-6"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Business definition defaults (used as the *default* proposed rule; the user
# can edit/override these at the approval checkpoint).
DEFAULT_TOP_PERFORMER_MIN_RATING = 4
DEFAULT_TOP_PERFORMER_LOOKBACK_CYCLES = 2
