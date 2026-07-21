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
    if df is None or df.empty:
        raise ValueError("Cannot apply DQ rules: Bronze DataFrame is empty.")

    out = df.copy()
    applied = []

    for rule in rules:
        if not rule.get("approved"):
            continue
        col = rule.get("column", "")
        rtype = rule.get("rule_type", "")

        try:
            if rtype == "dedupe" and "employee_id" in out.columns:
                before = len(out)
                out = out.drop_duplicates(subset=["employee_id"], keep="last")
                applied.append(f"{rule['id']} (dedupe: {before - len(out)} rows removed)")
            elif rtype == "impute_median" and col in out.columns:
                median = pd.to_numeric(out[col], errors="coerce").median()
                if pd.notna(median):
                    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(median)
                    applied.append(f"{rule['id']} (impute_median on {col}: filled with {median})")
            elif rtype == "fill_unknown" and col in out.columns:
                out[col] = out[col].fillna("Unknown")
                applied.append(f"{rule['id']} (fill_unknown on {col})")
            elif rtype == "standardize_text" and col in out.columns:
                out[col] = out[col].astype(str).str.strip().str.title()
                applied.append(f"{rule['id']} (standardize_text on {col})")
            elif rtype == "standardize_date" and col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
                applied.append(f"{rule['id']} (standardize_date on {col})")
        except Exception:
            continue

    # Normalize attrition_flag to boolean
    if "attrition_flag" in out.columns:
        out["attrition_flag"] = (
            out["attrition_flag"]
            .astype(str).str.strip().str.lower()
            .map({"yes": True, "no": False, "1": True, "0": False, "true": True, "false": False})
            .fillna(False)
        )

    silver_path = SILVER_DIR / f"{state.run_id}_silver.csv"
    out.to_csv(silver_path, index=False)

    state.silver_rows = len(out)
    state.dq_rules_approved = True
    state.status = "silver_ready"
    state.add_log(
        agent="TransformationAgent",
        action="apply_dq_rules",
        detail=f"Applied {len(applied)} approved rule(s). Silver now has {len(out)} rows -> {silver_path.name}.",
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
    if silver_df is None or silver_df.empty:
        return defaults

    columns = list(silver_df.columns)
    dtypes = {c: str(silver_df[c].dtype) for c in columns}
    sample = silver_df.head(3).to_dict(orient="records")

    example = {
        "id": "biz-1",
        "name": "top_performer_definition",
        "description": "Define 'top performer' as performance_rating >= 4.",
        "param": {"min_rating": 4},
        "approved": True
    }
    system = (
        "You are a People Analytics lead designing business rules for an attrition analysis pipeline. "
        "Given the Silver dataset schema and sample data below, propose business rules as a JSON array. "
        "Each rule MUST have these exact fields: "
        "id (string like 'biz-1'), name (snake_case identifier), "
        "description (plain English), param (dict of parameters), approved (boolean, always true). "
        "Rules should define: how to identify top performers, how to calculate tenure, "
        "and enrichment columns useful for attrition analysis. Do NOT duplicate rules. "
        "Return ONLY the JSON array, nothing else."
    )
    prompt = (
        f"Silver dataset:\n"
        f"- Rows: {len(silver_df)}\n"
        f"- Columns: {columns}\n"
        f"- Column types: {dtypes}\n"
        f"- Sample rows: {sample}\n\n"
        f"Example rule format:\n{json.dumps(example, indent=2)}\n\n"
        f"Now propose business rules for this data."
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
                r.setdefault("name", "unnamed_rule")
                r.setdefault("description", "")
            return ai_rules
    except (json.JSONDecodeError, IndexError, KeyError):
        pass

    return defaults


def propose_business_rules(state: PipelineState, silver_df: pd.DataFrame) -> list:
    defaults = _build_default_rules()

    if silver_df is None or silver_df.empty:
        state.business_rules = defaults
        state.status = "rules_proposed"
        state.add_log(agent="TransformationAgent", action="propose_business_rules",
                      detail="Silver empty, using default business rules.")
        return defaults

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
    if silver_df is None or silver_df.empty:
        raise ValueError("Cannot apply business rules: Silver DataFrame is empty.")

    out = silver_df.copy()
    rule_map = {r["name"]: r for r in rules if r.get("approved") and r.get("name")}

    # is_top_performer
    min_rating = (
        rule_map.get("top_performer_definition", {})
        .get("param", {})
        .get("min_rating", DEFAULT_TOP_PERFORMER_MIN_RATING)
    )
    if "performance_rating" in out.columns:
        out["performance_rating"] = pd.to_numeric(out["performance_rating"], errors="coerce")
        out["is_top_performer"] = out["performance_rating"] >= min_rating
    else:
        out["is_top_performer"] = False

    # tenure_years
    if "hire_date" in out.columns:
        hire = pd.to_datetime(out["hire_date"], errors="coerce")
        if "exit_date" in out.columns:
            exit_ = pd.to_datetime(out["exit_date"], errors="coerce")
        else:
            exit_ = pd.Series([pd.NaT] * len(out), index=out.index)
        end = exit_.fillna(pd.Timestamp.now())
        days = (end - hire).dt.days
        out["tenure_years"] = (days / 365.25).round(2)
        out["tenure_years"] = out["tenure_years"].clip(lower=0).fillna(0)
    else:
        out["tenure_years"] = 0.0

    # high_overtime
    thresh = (
        rule_map.get("high_overtime_threshold", {})
        .get("param", {})
        .get("threshold_hours", 15)
    )
    if "overtime_hours_per_month" in out.columns:
        out["overtime_hours_per_month"] = pd.to_numeric(out["overtime_hours_per_month"], errors="coerce").fillna(0)
        out["high_overtime"] = out["overtime_hours_per_month"] > thresh
    else:
        out["high_overtime"] = False

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
