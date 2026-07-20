"""Ingestion Agent: lands raw source data unmodified into the Bronze zone
and captures ingestion metadata."""
from datetime import datetime

import pandas as pd

from src.config import BRONZE_DIR
from src.state import PipelineState


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
