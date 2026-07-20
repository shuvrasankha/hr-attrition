"""Data Quality Agent: profiles the Bronze dataset, detects issues, and
PROPOSES cleaning rules. It does not apply anything itself — the
Transformation Agent applies only rules the human has approved."""
import pandas as pd

from src.llm import llm_complete
from src.state import PipelineState

EXPECTED_DATE_COLS = ["hire_date", "exit_date", "last_promotion_date"]
NUMERIC_COLS = ["performance_rating", "compensation", "engagement_score", "overtime_hours_per_month"]


def _detect_mixed_casing(df: pd.DataFrame, col: str) -> list:
    variants = df[col].dropna().unique().tolist()
    normalized = {}
    for v in variants:
        key = str(v).strip().lower()
        normalized.setdefault(key, set()).add(v)
    return [list(v) for v in normalized.values() if len(v) > 1]


def _detect_mixed_date_formats(df: pd.DataFrame, col: str) -> bool:
    sample = df[col].dropna().astype(str).head(200)
    formats_seen = set()
    for v in sample:
        if "-" in v and v.split("-")[0].isdigit() and len(v.split("-")[0]) == 4:
            formats_seen.add("ISO (YYYY-MM-DD)")
        elif "/" in v:
            formats_seen.add("US (MM/DD/YYYY)")
        elif "-" in v:
            formats_seen.add("EU (DD-MM-YYYY)")
    return len(formats_seen) > 1


def profile(df: pd.DataFrame) -> dict:
    findings = {
        "row_count": len(df),
        "duplicate_ids": int(df.duplicated(subset=["employee_id"]).sum()) if "employee_id" in df else 0,
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum() > 0},
        "casing_issues": {},
        "mixed_date_formats": [],
    }
    for c in ["department", "team"]:
        if c in df.columns:
            issues = _detect_mixed_casing(df, c)
            if issues:
                findings["casing_issues"][c] = issues
    for c in EXPECTED_DATE_COLS:
        if c in df.columns and _detect_mixed_date_formats(df, c):
            findings["mixed_date_formats"].append(c)
    return findings


def propose_rules(state: PipelineState, df: pd.DataFrame) -> list:
    findings = profile(df)
    rules = []
    rid = 0

    if findings["duplicate_ids"] > 0:
        rid += 1
        rules.append({
            "id": f"dq-{rid}",
            "column": "employee_id",
            "issue": f"{findings['duplicate_ids']} duplicate employee_id rows detected (repeated export).",
            "proposed_action": "Deduplicate: keep the last occurrence per employee_id.",
            "rule_type": "dedupe",
            "approved": True,  # default proposal is pre-checked; user can uncheck/edit
        })

    for col, count in findings["nulls"].items():
        rid += 1
        if col in NUMERIC_COLS:
            action = f"Impute missing '{col}' with the department median."
            rtype = "impute_median"
        else:
            action = f"Flag rows with missing '{col}' as 'Unknown' rather than dropping."
            rtype = "fill_unknown"
        rules.append({
            "id": f"dq-{rid}",
            "column": col,
            "issue": f"{count} missing values in '{col}'.",
            "proposed_action": action,
            "rule_type": rtype,
            "approved": True,
        })

    for col, variants in findings["casing_issues"].items():
        rid += 1
        example = variants[0]
        rules.append({
            "id": f"dq-{rid}",
            "column": col,
            "issue": f"Inconsistent casing/spacing in '{col}', e.g. {example}.",
            "proposed_action": f"Standardize '{col}' to Title Case and trim whitespace.",
            "rule_type": "standardize_text",
            "approved": True,
        })

    for col in findings["mixed_date_formats"]:
        rid += 1
        rules.append({
            "id": f"dq-{rid}",
            "column": col,
            "issue": f"Mixed date formats detected in '{col}' (ISO/US/EU).",
            "proposed_action": f"Parse '{col}' to a standard ISO (YYYY-MM-DD) date.",
            "rule_type": "standardize_date",
            "approved": True,
        })

    summary_fallback = (
        f"Profiled {findings['row_count']} Bronze rows and found "
        f"{findings['duplicate_ids']} duplicate IDs, "
        f"{len(findings['nulls'])} columns with missing values, "
        f"{len(findings['casing_issues'])} columns with inconsistent text casing, "
        f"and {len(findings['mixed_date_formats'])} columns with mixed date formats. "
        f"Proposing {len(rules)} cleaning rule(s) for review."
    )
    narrative = llm_complete(
        system="You are a meticulous data quality analyst. Summarize findings in 2-3 plain-English sentences for a non-technical HR stakeholder.",
        prompt=f"Data quality findings: {findings}\nProposed rules: {rules}",
        fallback=summary_fallback,
    )

    state.dq_rules = rules
    state.status = "dq_proposed"
    state.add_log(agent="DataQualityAgent", action="propose_rules", detail=narrative)
    return rules
