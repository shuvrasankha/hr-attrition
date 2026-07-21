"""Ingestion Agent: lands raw source data unmodified into the Bronze zone
and captures ingestion metadata."""
from datetime import datetime

import pandas as pd

from src.config import BRONZE_DIR
from src.state import PipelineState

REQUIRED_COLUMNS = {
    "employee_id": "Unique employee identifier",
    "department": "Department name",
    "team": "Team name",
    "performance_rating": "Numeric performance rating (1-5)",
    "compensation": "Annual salary",
    "attrition_flag": "Yes / No",
    "hire_date": "Employment start date",
    "overtime_hours_per_month": "Monthly overtime hours",
    "engagement_score": "Employee engagement score",
}


def validate_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (missing_required, extra_columns, found_required)."""
    found = [c for c in REQUIRED_COLUMNS if c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return missing, extra, found


def run(state: PipelineState, source_df: pd.DataFrame, source_filename: str) -> pd.DataFrame:
    state.source_filename = source_filename
    bronze_path = BRONZE_DIR / f"{state.run_id}_bronze.csv"
    source_df.to_csv(bronze_path, index=False)

    state.bronze_rows = len(source_df)
    state.status = "ingested"
    state.add_log(
        agent="IngestionAgent",
        action="ingest",
        detail=(
            f"Ingested '{source_filename}' -> {bronze_path.name}. "
            f"{len(source_df)} rows, {len(source_df.columns)} columns, "
            f"schema: {list(source_df.columns)}. Ingested at {datetime.utcnow().isoformat()}Z."
        ),
    )
    return source_df
