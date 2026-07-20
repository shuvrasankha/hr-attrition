"""Analytics / Insight Agent: aggregates the enriched Silver dataset into
Gold-layer, department and team-level attrition metrics, ranks the departments
by top-performer attrition rate, infers likely contributing factors,
computes a composite risk score, and generates a natural-language executive narrative."""
import pandas as pd

from src.config import GOLD_DIR
from src.llm import llm_complete
from src.state import PipelineState


def _compute_dept_metrics(group: pd.DataFrame) -> pd.Series:
    top_mask = group["is_top_performer"]
    left_top = top_mask & group["attrition_flag"]
    stay_top = top_mask & ~group["attrition_flag"]

    avg_comp_leavers = round(group.loc[left_top, "compensation"].mean(), 0) if left_top.any() else None
    avg_comp_stayers = round(group.loc[stay_top, "compensation"].mean(), 0) if stay_top.any() else None
    avg_ot_leavers = round(group.loc[left_top, "overtime_hours_per_month"].mean(), 1) if left_top.any() else None
    avg_ot_stayers = round(group.loc[stay_top, "overtime_hours_per_month"].mean(), 1) if stay_top.any() else None
    avg_eng_leavers = round(group.loc[left_top, "engagement_score"].mean(), 2) if left_top.any() else None

    return pd.Series({
        "headcount": len(group),
        "attritions": int(group["attrition_flag"].sum()),
        "attrition_rate": round(group["attrition_flag"].mean() * 100, 1),
        "top_performers": int(top_mask.sum()),
        "top_performer_attritions": int(left_top.sum()),
        "top_performer_attrition_rate": round(left_top.sum() / max(top_mask.sum(), 1) * 100, 1),
        "avg_compensation": round(group["compensation"].mean(), 0),
        "avg_compensation_leavers_top": avg_comp_leavers,
        "avg_compensation_stayers_top": avg_comp_stayers,
        "comp_gap": round(avg_comp_stayers - avg_comp_leavers, 0) if avg_comp_leavers is not None and avg_comp_stayers is not None else None,
        "avg_overtime_leavers_top": avg_ot_leavers,
        "avg_overtime_stayers_top": avg_ot_stayers,
        "overtime_gap": round(avg_ot_leavers - avg_ot_stayers, 1) if avg_ot_leavers is not None and avg_ot_stayers is not None else None,
        "avg_engagement_leavers_top": avg_eng_leavers,
    })


def _identify_driver(row: pd.Series) -> str:
    factors = []
    if row["comp_gap"] is not None and row["comp_gap"] > 2500:
        factors.append(f"compensation gap (~${row['comp_gap']:,.0f} below staying top performers)")
    if row["overtime_gap"] is not None and row["overtime_gap"] > 3:
        factors.append("elevated overtime before exit")
    if row["avg_engagement_leavers_top"] is not None and row["avg_engagement_leavers_top"] < 3.0:
        factors.append("low engagement score prior to exit")
    return "; ".join(factors) if factors else "no dominant single factor identified"


def _risk_score(row: pd.Series) -> float:
    tp_rate = row["top_performer_attrition_rate"]
    score = min(tp_rate / 5.0, 8.0)
    rate_factor = tp_rate / 100.0
    bonus = 0.0
    if row["comp_gap"] is not None and row["comp_gap"] > 2500:
        bonus += 0.75 * rate_factor
    if row["overtime_gap"] is not None and row["overtime_gap"] > 3:
        bonus += 0.5 * rate_factor
    if row["avg_engagement_leavers_top"] is not None and row["avg_engagement_leavers_top"] < 3.0:
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
    top_risk = gold.iloc[0]
    worst_team = team_gold.iloc[0]

    dept_ranking = ", ".join(f"{r['department']} ({r['top_performer_attrition_rate']}%)" for _, r in gold.iterrows())

    team_line = ""
    if worst_team["department"] != top_risk["department"]:
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
               f"Team-level Gold metrics:\n{team_gold.to_dict(orient='records')}\n\n"
               f"Answer: which teams/departments are losing their highest-performing employees, and why?",
        max_tokens=500,
        fallback=fallback,
    )
    state.narrative = narrative
    state.status = "complete"
    state.add_log(agent="AnalyticsAgent", action="generate_narrative", detail="Executive narrative generated.")
    return narrative
