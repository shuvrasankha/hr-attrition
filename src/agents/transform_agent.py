"""Transformation Agent:
  1. Applies human-APPROVED Data Quality rules to Bronze -> Silver.
  2. Proposes business/domain rules (e.g. 'top performer' definition,
     attrition-risk grouping) for the Silver -> Gold step, again gated
     on human approval before Analytics Agent aggregates.
"""
from datetime import datetime

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

    # Always-on structural normalization (not a "judgment call", so not gated)
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


def propose_business_rules(state: PipelineState, silver_df: pd.DataFrame) -> list:
    rules = [
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
    fallback = (
        "Proposing 3 business rules for review: how a 'top performer' is defined, "
        "how tenure is calculated, and the overtime threshold used to flag burnout risk. "
        "These directly affect which employees count toward the attrition metrics in the Gold layer."
    )
    narrative = llm_complete(
        system="You are a People Analytics lead explaining proposed business definitions to an HR stakeholder in 2-3 plain sentences.",
        prompt=f"Proposed business rules: {rules}",
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
