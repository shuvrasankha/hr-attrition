"""QA / Validation Agent: reconciles row counts and key metrics across
Bronze -> Silver -> Gold to catch pipeline errors before they reach the UI.
Runs as an optional check after each major step."""
import pandas as pd

from src.state import PipelineState


def validate(state: PipelineState, bronze: pd.DataFrame, silver: pd.DataFrame,
             silver_enriched: pd.DataFrame, gold: pd.DataFrame) -> dict:
    report = {"checks": [], "passed": True}

    def _check(name: str, ok: bool, detail: str):
        report["checks"].append({"name": name, "passed": ok, "detail": detail})
        if not ok:
            report["passed"] = False

    # Bronze checks
    if bronze is not None and not bronze.empty:
        _check("bronze_has_rows", len(bronze) > 0, f"Bronze has {len(bronze)} rows.")
        _check("bronze_has_employee_id", "employee_id" in bronze.columns,
               f"employee_id column present: {'employee_id' in bronze.columns}.")
    else:
        _check("bronze_has_rows", False, "Bronze is empty or None.")

    # Silver checks
    if silver is not None and not silver.empty:
        if bronze is not None and not bronze.empty:
            _check("silver_row_count", len(silver) <= len(bronze),
                   f"Silver ({len(silver)}) <= Bronze ({len(bronze)}) after dedup/cleaning.")
        if "employee_id" in silver.columns:
            _check("silver_no_duplicates", silver["employee_id"].duplicated().sum() == 0,
                   f"Duplicate employee_ids in Silver: {silver['employee_id'].duplicated().sum()}.")
        if "hire_date" in silver.columns:
            _check("silver_dates_parsed", not silver["hire_date"].astype(str).str.contains("/", na=False).any(),
                   "No US-format dates remaining in hire_date.")
    elif silver is not None:
        _check("silver_row_count", True, f"Silver has {len(silver)} rows (empty but valid).")

    # Silver enriched checks
    if silver_enriched is not None and not silver_enriched.empty:
        _check("enriched_has_top_performer", "is_top_performer" in silver_enriched.columns,
               f"is_top_performer column present: {'is_top_performer' in silver_enriched.columns}.")
        _check("enriched_has_tenure", "tenure_years" in silver_enriched.columns,
               f"tenure_years column present: {'tenure_years' in silver_enriched.columns}.")
        if silver is not None and not silver.empty:
            _check("enriched_row_count", len(silver_enriched) == len(silver),
                   f"Enriched ({len(silver_enriched)}) == Silver ({len(silver)}).")

    # Gold checks
    if gold is not None and not gold.empty:
        _check("gold_has_depts", len(gold) >= 1,
               f"Gold has {len(gold)} department rows.")
        if "top_performer_attrition_rate" in gold.columns:
            rates = pd.to_numeric(gold["top_performer_attrition_rate"], errors="coerce").dropna()
            _check("gold_rates_bounded", rates.between(0, 100).all() if len(rates) > 0 else True,
                   "All top-performer attrition rates are between 0-100%.")
    elif gold is not None:
        _check("gold_has_depts", True, f"Gold has {len(gold)} rows (empty but valid).")

    state.add_log(
        agent="QAAgent",
        action="validate",
        detail=f"{'All' if report['passed'] else 'Some'} checks passed: {len(report['checks'])} total, "
               f"{sum(c['passed'] for c in report['checks'])} passed.",
    )
    return report
