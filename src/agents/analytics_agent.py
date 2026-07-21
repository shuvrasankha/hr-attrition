"""Analytics / Insight Agent: aggregates the enriched Silver dataset into
Gold-layer, department and team-level attrition metrics, ranks the departments
by top-performer attrition rate, infers likely contributing factors,
computes a composite risk score, and generates a natural-language executive narrative."""
import pandas as pd

from src.config import GOLD_DIR
from src.llm import llm_complete
from src.state import PipelineState


def _safe_mean(series: pd.Series):
    """Return mean of series, or None if empty/all-NaN."""
    if series.empty or series.isna().all():
        return None
    val = series.mean()
    return round(val, 2) if pd.notna(val) else None


def _compute_dept_metrics(group: pd.DataFrame) -> pd.Series:
    required = ["is_top_performer", "attrition_flag", "compensation",
                 "overtime_hours_per_month", "engagement_score"]
    for col in required:
        if col not in group.columns:
            return pd.Series({
                "headcount": len(group), "attritions": 0, "attrition_rate": 0.0,
                "top_performers": 0, "top_performer_attritions": 0,
                "top_performer_attrition_rate": 0.0, "avg_compensation": None,
                "avg_compensation_leavers_top": None, "avg_compensation_stayers_top": None,
                "comp_gap": None, "avg_overtime_leavers_top": None,
                "avg_overtime_stayers_top": None, "overtime_gap": None,
                "avg_engagement_leavers_top": None,
            })

    top_mask = group["is_top_performer"].fillna(False).astype(bool)
    attrition = group["attrition_flag"].fillna(False).astype(bool)
    left_top = top_mask & attrition
    stay_top = top_mask & ~attrition

    comp = pd.to_numeric(group["compensation"], errors="coerce")
    ot = pd.to_numeric(group["overtime_hours_per_month"], errors="coerce")
    eng = pd.to_numeric(group["engagement_score"], errors="coerce")

    avg_comp_leavers = _safe_mean(comp[left_top])
    avg_comp_stayers = _safe_mean(comp[stay_top])
    avg_ot_leavers = _safe_mean(ot[left_top])
    avg_ot_stayers = _safe_mean(ot[stay_top])
    avg_eng_leavers = _safe_mean(eng[left_top])

    tp_count = int(top_mask.sum())
    tp_attritions = int(left_top.sum())
    tp_rate = round(tp_attritions / max(tp_count, 1) * 100, 1)

    comp_gap = None
    if avg_comp_leavers is not None and avg_comp_stayers is not None:
        comp_gap = round(avg_comp_stayers - avg_comp_leavers, 0)

    ot_gap = None
    if avg_ot_leavers is not None and avg_ot_stayers is not None:
        ot_gap = round(avg_ot_leavers - avg_ot_stayers, 1)

    return pd.Series({
        "headcount": len(group),
        "attritions": int(attrition.sum()),
        "attrition_rate": round(attrition.mean() * 100, 1),
        "top_performers": tp_count,
        "top_performer_attritions": tp_attritions,
        "top_performer_attrition_rate": tp_rate,
        "avg_compensation": _safe_mean(comp),
        "avg_compensation_leavers_top": avg_comp_leavers,
        "avg_compensation_stayers_top": avg_comp_stayers,
        "comp_gap": comp_gap,
        "avg_overtime_leavers_top": avg_ot_leavers,
        "avg_overtime_stayers_top": avg_ot_stayers,
        "overtime_gap": ot_gap,
        "avg_engagement_leavers_top": avg_eng_leavers,
    })


def _identify_driver(row: pd.Series) -> str:
    factors = []
    if row.get("comp_gap") is not None and row["comp_gap"] > 2500:
        factors.append(f"compensation gap (~${row['comp_gap']:,.0f} below staying top performers)")
    if row.get("overtime_gap") is not None and row["overtime_gap"] > 3:
        factors.append("elevated overtime before exit")
    if row.get("avg_engagement_leavers_top") is not None and row["avg_engagement_leavers_top"] < 3.0:
        factors.append("low engagement score prior to exit")
    return "; ".join(factors) if factors else "no dominant single factor identified"


def _risk_score(row: pd.Series) -> float:
    tp_rate = row.get("top_performer_attrition_rate", 0) or 0
    score = min(tp_rate / 5.0, 8.0)
    rate_factor = tp_rate / 100.0
    bonus = 0.0
    if row.get("comp_gap") is not None and row["comp_gap"] > 2500:
        bonus += 0.75 * rate_factor
    if row.get("overtime_gap") is not None and row["overtime_gap"] > 3:
        bonus += 0.5 * rate_factor
    if row.get("avg_engagement_leavers_top") is not None and row["avg_engagement_leavers_top"] < 3.0:
        bonus += 0.5 * rate_factor
    score += bonus
    return round(min(score, 10.0), 1)


def _risk_label(score: float) -> str:
    if score >= 6.0:
        return "CRITICAL"
    elif score >= 4.0:
        return "HIGH"
    elif score >= 2.5:
        return "MEDIUM"
    return "LOW"


def build_gold(state: PipelineState, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        state.gold_rows = 0
        state.status = "gold_ready"
        return pd.DataFrame()

    if "department" not in df.columns:
        raise ValueError("Cannot build Gold: 'department' column missing from Silver enriched data.")

    dept_gold = df.groupby("department").apply(_compute_dept_metrics, include_groups=False).reset_index()
    dept_gold = dept_gold.sort_values("top_performer_attrition_rate", ascending=False).reset_index(drop=True)
    dept_gold["risk_rank"] = dept_gold.index + 1
    dept_gold["likely_driver"] = dept_gold.apply(_identify_driver, axis=1)
    dept_gold["risk_score"] = dept_gold.apply(_risk_score, axis=1)
    dept_gold["risk_label"] = dept_gold["risk_score"].apply(_risk_label)

    gold_path = GOLD_DIR / f"{state.run_id}_gold.csv"
    dept_gold.to_csv(gold_path, index=False)

    state.gold_rows = len(dept_gold)
    state.status = "gold_ready"
    state.add_log(
        agent="AnalyticsAgent",
        action="build_gold",
        detail=f"Aggregated {len(df)} Silver rows into {len(dept_gold)} department-level Gold rows -> {gold_path.name}.",
    )
    return dept_gold


def build_team_gold(state: PipelineState, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "department" not in df.columns or "team" not in df.columns:
        return pd.DataFrame()

    team_gold = df.groupby(["department", "team"]).apply(_compute_dept_metrics, include_groups=False).reset_index()
    team_gold = team_gold.sort_values("top_performer_attrition_rate", ascending=False).reset_index(drop=True)
    team_gold["likely_driver"] = team_gold.apply(_identify_driver, axis=1)
    team_gold["risk_score"] = team_gold.apply(_risk_score, axis=1)
    team_gold["risk_label"] = team_gold["risk_score"].apply(_risk_label)

    team_path = GOLD_DIR / f"{state.run_id}_gold_team.csv"
    team_gold.to_csv(team_path, index=False)

    state.add_log(
        agent="AnalyticsAgent",
        action="build_team_gold",
        detail=f"Aggregated into {len(team_gold)} team-level rows -> {team_path.name}.",
    )
    return team_gold


def generate_narrative(state: PipelineState, gold: pd.DataFrame, team_gold: pd.DataFrame) -> str:
    if gold is None or gold.empty:
        narrative = "No Gold data available to generate a narrative."
        state.narrative = narrative
        state.status = "complete"
        return narrative

    top_risk = gold.iloc[0]
    worst_team = team_gold.iloc[0] if team_gold is not None and not team_gold.empty else None

    dept_ranking = ", ".join(
        f"{r['department']} ({r['top_performer_attrition_rate']}%)" for _, r in gold.iterrows()
    )

    team_line = ""
    if worst_team is not None and worst_team.get("department") != top_risk.get("department"):
        team_line = (
            f" The worst-affected team is **{worst_team['team']}** in {worst_team['department']} "
            f"at {worst_team['top_performer_attrition_rate']}% top-performer attrition."
        )

    fallback = (
        f"**{top_risk['department']}** has the highest top-performer attrition rate at "
        f"{top_risk['top_performer_attrition_rate']}% ({int(top_risk['top_performer_attritions'])} of "
        f"{int(top_risk['top_performers'])} top performers), driven primarily by {top_risk['likely_driver']}."
        f"{team_line}"
        f"\n\nDepartments ranked by risk: {dept_ranking}. "
        "Recommend prioritizing a retention review for the top-ranked department(s) before the next review cycle."
    )
    narrative = llm_complete(
        system=(
            "You are a People Analytics lead writing a short executive summary for HR leadership. "
            "Be direct, specific with numbers, and action-oriented. 4-6 sentences max. "
            "Call out the top 1-2 teams (not just departments) most at risk."
        ),
        prompt=f"Department-level Gold metrics:\n{gold.to_dict(orient='records')}\n\n"
               f"Team-level Gold metrics:\n{team_gold.to_dict(orient='records') if team_gold is not None and not team_gold.empty else 'No team data'}\n\n"
               f"Answer: which teams/departments are losing their highest-performing employees, and why?",
        max_tokens=500,
        fallback=fallback,
    )
    state.narrative = narrative
    state.status = "complete"
    state.add_log(agent="AnalyticsAgent", action="generate_narrative", detail="Executive narrative generated.")
    return narrative
