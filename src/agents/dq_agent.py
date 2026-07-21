"""Data Quality Agent: profiles the Bronze dataset, asks the AI to propose
cleaning rules in natural language, then parses them into structured rules.
The human reviews and approves before the Transformation Agent applies them."""
import json

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
        "columns": list(df.columns),
        "duplicate_ids": int(df.duplicated(subset=["employee_id"]).sum()) if "employee_id" in df else 0,
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum() > 0},
        "casing_issues": {},
        "mixed_date_formats": [],
        "sample_rows": df.head(5).to_dict(orient="records"),
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


def _build_hardcoded_rules(findings: dict) -> list:
    """Fallback rules if the AI is unavailable."""
    rules = []
    rid = 0

    if findings["duplicate_ids"] > 0:
        rid += 1
        rules.append({
            "id": f"dq-{rid}",
            "column": "employee_id",
            "issue": f"{findings['duplicate_ids']} duplicate employee_id rows detected.",
            "proposed_action": "Deduplicate: keep the last occurrence per employee_id.",
            "rule_type": "dedupe",
            "approved": True,
        })

    for col, count in findings["nulls"].items():
        rid += 1
        if col in NUMERIC_COLS:
            action = f"Impute missing '{col}' with the department median."
            rtype = "impute_median"
        else:
            action = f"Flag rows with missing '{col}' as 'Unknown'."
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
        rules.append({
            "id": f"dq-{rid}",
            "column": col,
            "issue": f"Inconsistent casing in '{col}', e.g. {variants[0]}.",
            "proposed_action": f"Standardize '{col}' to Title Case and trim whitespace.",
            "rule_type": "standardize_text",
            "approved": True,
        })

    for col in findings["mixed_date_formats"]:
        rid += 1
        rules.append({
            "id": f"dq-{rid}",
            "column": col,
            "issue": f"Mixed date formats in '{col}'.",
            "proposed_action": f"Parse '{col}' to ISO (YYYY-MM-DD) format.",
            "rule_type": "standardize_date",
            "approved": True,
        })

    return rules


def _ask_ai_for_rules(findings: dict, hardcoded: list) -> list:
    """Ask the AI to review the data profile and suggest cleaning rules."""
    example = {
        "id": "dq-1",
        "column": "employee_id",
        "issue": "24 duplicate employee_id rows detected (repeated export).",
        "proposed_action": "Deduplicate: keep the last occurrence per employee_id.",
        "rule_type": "dedupe",
        "approved": True
    }
    system = (
        "You are a data quality engineer reviewing a raw HR dataset. "
        "Given the data profile below, propose cleaning rules as a JSON array. "
        "Each rule MUST have these exact fields: "
        "id (string like 'dq-1', 'dq-2'), column (string), "
        "issue (plain English description of the problem), "
        "proposed_action (plain English description of the fix), "
        "rule_type (one of: dedupe, impute_median, fill_unknown, standardize_text, standardize_date), "
        "approved (boolean, always true). "
        "Only propose rules for real issues. Do NOT duplicate rules. "
        "Return ONLY the JSON array, nothing else."
    )
    prompt = (
        f"Data profile:\n"
        f"- Rows: {findings['row_count']}\n"
        f"- Columns: {findings['columns']}\n"
        f"- Duplicate employee_ids: {findings['duplicate_ids']}\n"
        f"- Nulls: {findings['nulls']}\n"
        f"- Casing issues: {findings['casing_issues']}\n"
        f"- Mixed date formats: {findings['mixed_date_formats']}\n"
        f"- Sample rows: {findings['sample_rows']}\n\n"
        f"Example rule format:\n{json.dumps(example, indent=2)}\n\n"
        f"Now propose rules for this data."
    )

    response = llm_complete(
        system=system,
        prompt=prompt,
        max_tokens=800,
        fallback="",
    )

    if not response:
        return hardcoded

    try:
        # Extract JSON from the response (handle markdown code blocks)
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        ai_rules = json.loads(text.strip())
        if isinstance(ai_rules, list) and len(ai_rules) > 0:
            # Ensure all rules have required fields
            for r in ai_rules:
                r.setdefault("approved", True)
                r.setdefault("rule_type", "fill_unknown")
            return ai_rules
    except (json.JSONDecodeError, IndexError, KeyError):
        pass

    return hardcoded


def propose_rules(state: PipelineState, df: pd.DataFrame) -> list:
    findings = profile(df)
    hardcoded = _build_hardcoded_rules(findings)

    # Ask the AI for its own suggestions
    ai_rules = _ask_ai_for_rules(findings, hardcoded)

    summary_fallback = (
        f"Profiled {findings['row_count']} Bronze rows. "
        f"Found {findings['duplicate_ids']} duplicate IDs, "
        f"{len(findings['nulls'])} columns with missing values, "
        f"{len(findings['casing_issues'])} casing issues, "
        f"{len(findings['mixed_date_formats'])} date format issues. "
        f"Proposing {len(ai_rules)} cleaning rule(s) for review."
    )

    narrative = llm_complete(
        system="You are a data quality analyst. Explain the proposed cleaning rules in 2-3 plain-English sentences for an HR stakeholder.",
        prompt=f"Findings: {json.dumps(findings, default=str)}\nRules proposed: {json.dumps(ai_rules, default=str)}",
        fallback=summary_fallback,
    )

    state.dq_rules = ai_rules
    state.status = "dq_proposed"
    state.add_log(agent="DataQualityAgent", action="propose_rules", detail=narrative)
    return ai_rules
