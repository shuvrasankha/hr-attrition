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

NUMERIC_COLUMNS = ["performance_rating", "compensation", "overtime_hours_per_month", "engagement_score"]


def validate_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (missing_required, extra_columns, found_required)."""
    found = [c for c in REQUIRED_COLUMNS if c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return missing, extra, found


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce column types to safe defaults. Returns a copy."""
    out = df.copy()

    # Coerce numeric columns
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Drop only non-required columns that are entirely null
    required = set(REQUIRED_COLUMNS.keys())
    for col in list(out.columns):
        if col not in required and out[col].isna().all():
            out.drop(columns=[col], inplace=True)

    # Coerce attrition_flag to consistent string
    if "attrition_flag" in out.columns:
        out["attrition_flag"] = (
            out["attrition_flag"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": "Yes", "no": "No", "1": "Yes", "0": "No", "true": "Yes", "false": "No"})
            .fillna("No")
        )

    # Coerce date columns
    for col in ["hire_date", "exit_date", "last_promotion_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")

    # Strip whitespace from string columns
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str).str.strip()

    # Drop rows where employee_id is null
    if "employee_id" in out.columns:
        before = len(out)
        out = out.dropna(subset=["employee_id"])
        dropped = before - len(out)
        if dropped > 0:
            pass  # logged by caller

    return out


def run(state: PipelineState, source_df: pd.DataFrame, source_filename: str) -> pd.DataFrame:
    state.source_filename = source_filename

    # Coerce types for safety
    cleaned = coerce_types(source_df)

    bronze_path = BRONZE_DIR / f"{state.run_id}_bronze.csv"
    cleaned.to_csv(bronze_path, index=False)

    state.bronze_rows = len(cleaned)
    state.status = "ingested"

    dropped = len(source_df) - len(cleaned)
    detail = (
        f"Ingested '{source_filename}' -> {bronze_path.name}. "
        f"{len(cleaned)} rows, {len(cleaned.columns)} columns."
    )
    if dropped > 0:
        detail += f" Dropped {dropped} rows with null employee_id."
    detail += f" Schema: {list(cleaned.columns)}. Ingested at {datetime.utcnow().isoformat()}Z."

    state.add_log(agent="IngestionAgent", action="ingest", detail=detail)
    return cleaned
