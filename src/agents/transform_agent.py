"""Transformation Agent:
  1. Applies human-APPROVED Data Quality rules to Bronze -> Silver.
  2. Asks the AI to propose business/domain rules for Silver -> Gold,
     then applies human-approved versions to enrich Silver.
"""
import json

import numpy as np
import pandas as pd

from src.config import SILVER_DIR, DEFAULT_TOP_PERFORMER_MIN_RATING
from src.llm import llm_complete
from src.state import PipelineState


def apply_dq_rules(state: PipelineState, df: pd.DataFrame, rules: list) -> pd.DataFrame:
    out = df.copy()
    applied = []

    for rule in rules:
        if not rule.get("approved"):
            continue
        col = rule["column"]
        rtype = rule["rule_type"]

        if rtype == "dedupe":
            out = out.drop_duplicates(subset=["employee_id"], keep="last")
        elif rtype == "impute_median" and col in out.columns:
            median = pd.to_numeric(out[col], errors="coerce").median()
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(median)
        elif rtype == "fill_unknown" and col in out.columns:
            out[col] = out[col].fillna("Unknown")
        elif rtype == "standardize_text" and col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.title()
        elif rtype == "standardize_date" and col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")

        applied.append(f"{rule['id']} ({rtype} on {col})")

    if "attrition_flag" in out.columns:
        out["attrition_flag"] = out["attrition_flag"].astype(str).str.strip().str.title()
        out["attrition_flag"] = out["attrition_flag"].map({"Yes": True, "No": False}).fillna(False)

    silver_path = SILVER_DIR / f"{state.run_id}_silver.csv"
    out.to_csv(silver_path, index=False)

    state.silver_rows = len(out)
    state.dq_rules_approved = True
    state.status = "silver_ready"
    state.add_log(
        agent="TransformationAgent",
        action="apply_dq_rules",
        detail=f"Applied {len(applied)} approved rule(s): {applied}. Silver now has {len(out)} rows -> {silver_path.name}.",
    )
    return out


def _build_default_rules() -> list:
    """Fallback business rules if AI is unavailable."""
    return [
        {
            "id": "biz-1",
            "name": "top_performer_definition",
            "description": f"Define 'top performer' as performance_rating >= {DEFAULT_TOP_PERFORMER_MIN_RATING}.",
            "param": {"min_rating": DEFAULT_TOP_PERFORMER_MIN_RATING},
            "approved": True,
        },
        {
            "id": "biz-2",
            "name": "tenure_calculation",
            "description": "Compute tenure_years as (exit_date or today) minus hire_date, in years.",
            "param": {},
            "approved": True,
        },
        {
            "id": "biz-3",
            "name": "high_overtime_threshold",
            "description": "Flag employees with overtime_hours_per_month > 15 as 'high overtime' for risk-factor analysis.",
            "param": {"threshold_hours": 15},
            "approved": True,
        },
    ]


def _ask_ai_for_rules(silver_df: pd.DataFrame, defaults: list) -> list:
    """Ask the AI to analyze the Silver data and suggest business rules."""
    columns = list(silver_df.columns)
    dtypes = {c: str(silver_df[c].dtype) for c in columns}
    sample = silver_df.head(3).to_dict(orient="records")

    system = (
        "You are a People Analytics lead designing business rules for an attrition analysis pipeline. "
        "Given the Silver dataset schema and sample data below, propose business rules as a JSON array. "
        "Each rule must have: id, name, description, param (dict of parameters), approved (boolean). "
        "Rules should define: how to identify top performers, how to calculate tenure, "
        "and any other enrichment columns useful for attrition analysis. "
        "Return ONLY the JSON array, no other text."
    )
    prompt = (
        f"Silver dataset:\n"
        f"- Rows: {len(silver_df)}\n"
        f"- Columns: {columns}\n"
        f"- Column types: {dtypes}\n"
        f"- Sample rows: {sample}\n\n"
        f"Here is a reference format:\n"
        f"{json.dumps(defaults, indent=2)}\n\n"
        f"Propose business rules as JSON."
    )

    response = llm_complete(system=system, prompt=prompt, max_tokens=800, fallback="")

    if not response:
        return defaults

    try:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        ai_rules = json.loads(text.strip())
        if isinstance(ai_rules, list) and len(ai_rules) > 0:
            for r in ai_rules:
                r.setdefault("approved", True)
                r.setdefault("param", {})
            return ai_rules
    except (json.JSONDecodeError, IndexError, KeyError):
        pass

    return defaults


def propose_business_rules(state: PipelineState, silver_df: pd.DataFrame) -> list:
    defaults = _build_default_rules()
    rules = _ask_ai_for_rules(silver_df, defaults)

    fallback = (
        f"Proposing {len(rules)} business rules based on the Silver dataset columns: "
        + ", ".join(silver_df.columns) + ". "
        "These rules define how to identify top performers, calculate tenure, and flag risk factors."
    )
    narrative = llm_complete(
        system="You are a People Analytics lead explaining proposed business definitions to an HR stakeholder in 2-3 plain sentences.",
        prompt=f"Silver columns: {list(silver_df.columns)}\nBusiness rules: {json.dumps(rules, default=str)}",
        fallback=fallback,
    )

    state.business_rules = rules
    state.status = "rules_proposed"
    state.add_log(agent="TransformationAgent", action="propose_business_rules", detail=narrative)
    return rules


def apply_business_rules(state: PipelineState, silver_df: pd.DataFrame, rules: list) -> pd.DataFrame:
    out = silver_df.copy()
    rule_map = {r["name"]: r for r in rules if r.get("approved")}

    min_rating = rule_map.get("top_performer_definition", {}).get("param", {}).get("min_rating", DEFAULT_TOP_PERFORMER_MIN_RATING)
    out["performance_rating"] = pd.to_numeric(out["performance_rating"], errors="coerce")
    out["is_top_performer"] = out["performance_rating"] >= min_rating

    if "tenure_calculation" in rule_map:
        hire = pd.to_datetime(out["hire_date"], errors="coerce")
        exit_ = pd.to_datetime(out["exit_date"], errors="coerce")
        end = exit_.fillna(pd.Timestamp("2026-06-30"))
        out["tenure_years"] = ((end - hire).dt.days / 365.25).round(2)

    thresh = rule_map.get("high_overtime_threshold", {}).get("param", {}).get("threshold_hours", 15)
    out["overtime_hours_per_month"] = pd.to_numeric(out["overtime_hours_per_month"], errors="coerce")
    out["high_overtime"] = out["overtime_hours_per_month"] > thresh

    silver_path = SILVER_DIR / f"{state.run_id}_silver_enriched.csv"
    out.to_csv(silver_path, index=False)

    state.business_rules_approved = True
    state.status = "rules_approved"
    state.add_log(
        agent="TransformationAgent",
        action="apply_business_rules",
        detail=f"Applied {len(rule_map)} approved business rule(s). Enriched Silver saved -> {silver_path.name}.",
    )
    return out
