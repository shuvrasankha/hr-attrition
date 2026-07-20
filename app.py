import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src import data_gen
from src.agents.orchestrator import Orchestrator
from src.state import PipelineState

st.set_page_config(page_title="HR Attrition — Agentic Medallion Pipeline", layout="wide")

STEPS = ["Upload / Ingest", "Data Quality Approval", "Business Rules Approval", "Gold Insights"]


def init_session():
    if "orch" not in st.session_state:
        st.session_state.orch = None
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "qa_report" not in st.session_state:
        st.session_state.qa_report = None


def sidebar():
    st.sidebar.title("Medallion Pipeline")
    st.sidebar.caption("Bronze → Silver → Gold, agent-orchestrated")

    from src.llm import is_llm_enabled
    if is_llm_enabled():
        st.sidebar.success("AI: ON (GitHub Models — gpt-4o)")
    else:
        st.sidebar.warning("AI: OFF (using fallback text)")

    page = st.sidebar.radio("Navigate", ["Run Pipeline", "Run History"])

    if st.session_state.orch:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Current run status**")
        state = st.session_state.orch.state
        st.sidebar.code(f"run_id: {state.run_id}\nstatus: {state.status}")
        for i, s in enumerate(STEPS):
            marker = "✅" if i < st.session_state.step else ("▶️" if i == st.session_state.step else "⬜")
            st.sidebar.write(f"{marker} {s}")

        if st.session_state.orch.bronze_df is not None:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Layer row counts**")
            st.sidebar.write(f"Bronze: {state.bronze_rows}")
            st.sidebar.write(f"Silver: {state.silver_rows or '—'}")
            st.sidebar.write(f"Gold: {state.gold_rows or '—'}")
    return page


def _render_dq_approval(orch: Orchestrator):
    st.subheader("Step 2 · Data Quality Agent — proposed cleaning rules")
    st.write(f"Bronze layer: **{orch.state.bronze_rows} rows** ingested from `{orch.state.source_filename}`.")

    with st.expander("Preview Bronze data (first 20 rows)"):
        st.dataframe(orch.bronze_df.head(20), use_container_width=True)

    st.markdown("### Agent Findings")
    st.info("The Data Quality Agent profiled the Bronze data and found issues with duplicates, missing values, inconsistent casing, and mixed date formats. Review each proposed fix below.")

    if not orch.state.dq_rules:
        with st.spinner("Data Quality Agent profiling Bronze data..."):
            orch.propose_dq_rules()

    st.markdown("### Proposed Cleaning Rules")
    st.write("Uncheck a rule to skip it. Edit the action text to customize the fix.")
    edited_rules = []
    for rule in orch.state.dq_rules:
        with st.container(border=True):
            c1, c2 = st.columns([0.06, 0.94])
            with c1:
                approved = st.checkbox("Apply", value=rule.get("approved", True), key=f"dq_{rule['id']}")
            with c2:
                st.markdown(f"**{rule['issue']}**")
                new_action = st.text_input(
                    "Proposed action",
                    value=rule["proposed_action"],
                    key=f"dq_action_{rule['id']}",
                    label_visibility="collapsed",
                )
            r = dict(rule)
            r["approved"] = approved
            r["proposed_action"] = new_action
            edited_rules.append(r)

    if st.button("Approve rules & build Silver layer", type="primary"):
        orch.apply_dq_rules(edited_rules)

        with st.spinner("Running QA validation..."):
            qa_report = orch.run_qa_validation()
        st.session_state.qa_report = qa_report

        st.session_state.step = 2
        st.rerun()


def _render_biz_approval(orch: Orchestrator):
    st.subheader("Step 3 · Transformation Agent — proposed business rules")
    st.write(f"Silver layer: **{orch.state.silver_rows} rows** after cleaning.")

    if st.session_state.get("qa_report"):
        qa = st.session_state.qa_report
        if qa["passed"]:
            st.success(f"QA Validation passed ({sum(c['passed'] for c in qa['checks'])}/{len(qa['checks'])} checks)")
        else:
            st.warning(f"QA Validation issues ({sum(c['passed'] for c in qa['checks'])}/{len(qa['checks'])} checks passed)")
        with st.expander("QA Validation Details"):
            for c in qa["checks"]:
                icon = "✅" if c["passed"] else "❌"
                st.write(f"{icon} **{c['name']}**: {c['detail']}")

    with st.expander("Preview Silver data (first 20 rows)"):
        st.dataframe(orch.silver_df.head(20), use_container_width=True)

    st.markdown("### Agent Findings")
    st.info("The Transformation Agent proposes these business definitions which directly affect which employees are counted as 'top performers' and how attrition metrics are calculated.")

    if not orch.state.business_rules:
        with st.spinner("Transformation Agent drafting business rule proposals..."):
            orch.propose_business_rules()

    st.markdown("### Proposed Business Rules")
    st.write("Uncheck a rule to skip it. Edit the description to customize the definition.")
    edited_rules = []
    for rule in orch.state.business_rules:
        with st.container(border=True):
            c1, c2 = st.columns([0.06, 0.94])
            with c1:
                approved = st.checkbox("Apply", value=rule.get("approved", True), key=f"biz_{rule['id']}")
            with c2:
                st.markdown(f"**{rule['name'].replace('_', ' ').title()}**")
                new_desc = st.text_input(
                    "Description",
                    value=rule["description"],
                    key=f"biz_desc_{rule['id']}",
                    label_visibility="collapsed",
                )
            r = dict(rule)
            r["approved"] = approved
            r["description"] = new_desc
            edited_rules.append(r)

    if st.button("Approve rules & build Gold layer", type="primary"):
        orch.apply_business_rules(edited_rules)
        orch.build_gold_and_narrative()
        st.session_state.step = 3
        st.rerun()


def _render_gold_insights(orch: Orchestrator):
    st.subheader("Step 4 · Analytics Agent — Gold layer insights")

    st.success(orch.state.narrative)

    gold = orch.gold_df
    team_gold = orch.team_gold_df

    # --- Risk summary cards ---
    st.markdown("### Risk Overview")
    cols = st.columns(min(len(gold), 5))
    for i, (_, row) in enumerate(gold.iterrows()):
        with cols[i]:
            label = row["risk_label"]
            color = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(label, "⚪")
            st.metric(
                label=f"{color} {row['department']}",
                value=f"{row['top_performer_attrition_rate']}%",
                delta=f"Risk: {label} ({row['risk_score']}/10)",
                delta_color="inverse" if label in ("CRITICAL", "HIGH") else "off",
            )

    st.markdown("---")

    # --- Charts ---
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top-performer attrition rate by department**")
        chart_df = gold.set_index("department")[["top_performer_attrition_rate"]]
        st.bar_chart(chart_df, color="#e74c3c")
    with c2:
        st.markdown("**Overall attrition rate by department**")
        chart_df2 = gold.set_index("department")[["attrition_rate"]]
        st.bar_chart(chart_df2, color="#3498db")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Compensation gap: top-performer leavers vs stayers**")
        comp_data = gold[["department", "comp_gap"]].dropna().set_index("department")
        if not comp_data.empty:
            st.bar_chart(comp_data, color="#e67e22")
        else:
            st.info("No compensation gap data available.")
    with c4:
        st.markdown("**Risk score by department**")
        risk_data = gold.set_index("department")[["risk_score"]]
        st.bar_chart(risk_data, color="#8e44ad")

    # --- Department ranking table ---
    st.markdown("### Department Risk Ranking")
    display_cols = [
        "risk_rank", "department", "headcount", "attrition_rate",
        "top_performers", "top_performer_attritions", "top_performer_attrition_rate",
        "comp_gap", "overtime_gap", "risk_score", "risk_label", "likely_driver",
    ]
    existing_cols = [c for c in display_cols if c in gold.columns]
    st.dataframe(gold[existing_cols], use_container_width=True, hide_index=True)

    # --- Team-level drill-down ---
    st.markdown("---")
    st.subheader("Team-Level Drill-Down")
    st.write("Top 10 teams by attrition risk (cross-department view):")
    team_display = team_gold.head(10)
    team_cols = ["department", "team", "headcount", "top_performers", "top_performer_attritions",
                  "top_performer_attrition_rate", "comp_gap", "risk_score", "risk_label", "likely_driver"]
    existing_team_cols = [c for c in team_cols if c in team_display.columns]
    st.dataframe(team_display[existing_team_cols], use_container_width=True, hide_index=True)

    # --- Data lineage view ---
    st.markdown("---")
    st.subheader("Data Lineage")
    st.write("Trace which raw rows contributed to each Gold aggregate:")
    with st.expander("View data lineage"):
        selected_dept = st.selectbox("Select a department to trace", gold["department"].tolist())
        if selected_dept and orch.silver_enriched_df is not None:
            dept_silver = orch.silver_enriched_df[orch.silver_enriched_df["department"] == selected_dept]
            st.markdown(f"**{selected_dept}**: {len(dept_silver)} Silver rows")
            st.markdown(f"- {int(dept_silver['is_top_performer'].sum())} top performers")
            st.markdown(f"- {int(dept_silver['attrition_flag'].sum())} total attritions")
            st.markdown(f"- {int((dept_silver['is_top_performer'] & dept_silver['attrition_flag']).sum())} top-performer attritions")

            st.markdown("**Sample top-performer leavers (Silver → Gold trace):**")
            trace_df = dept_silver[dept_silver["is_top_performer"] & dept_silver["attrition_flag"]][
                ["employee_id", "team", "performance_rating", "compensation", "tenure_years", "high_overtime"]
            ].head(10)
            st.dataframe(trace_df, use_container_width=True, hide_index=True)

    # --- Full tables ---
    with st.expander("Full Gold-layer table"):
        st.dataframe(gold, use_container_width=True)
    with st.expander("Team-level Gold table"):
        st.dataframe(team_gold, use_container_width=True)

    # --- Audit trail ---
    with st.expander("Agent run log (audit trail)"):
        for entry in orch.state.log:
            st.markdown(f"`{entry['timestamp']}` **{entry['agent']}** — {entry['action']}  \n{entry['detail']}")

    if st.button("Start a new run"):
        st.session_state.orch = None
        st.session_state.step = 0
        st.session_state.qa_report = None
        st.rerun()


def page_run():
    init_session()
    st.title("HR / Attrition — Agentic Medallion Pipeline")
    st.caption("Which teams and departments are losing their highest-performing employees, and why?")

    step = st.session_state.step

    if step == 0:
        st.subheader("Step 1 · Ingestion Agent")
        st.write("Upload an HR CSV, or use the built-in synthetic sample dataset to demo the full pipeline.")
        col1, col2 = st.columns(2)
        with col1:
            uploaded = st.file_uploader("Upload CSV(s)", type=["csv"], accept_multiple_files=True)
        with col2:
            use_sample = st.button("Use synthetic sample dataset", type="primary")

        if uploaded or use_sample:
            if uploaded:
                frames = [pd.read_csv(f) for f in uploaded]
                df = pd.concat(frames, ignore_index=True)
                fname = ", ".join(f.name for f in uploaded)
            else:
                df = data_gen.generate()
                fname = "hr_attrition_synthetic.csv"

            orch = Orchestrator()
            orch.ingest(df, fname)
            st.session_state.orch = orch
            st.session_state.step = 1
            st.rerun()

    elif step == 1:
        _render_dq_approval(st.session_state.orch)

    elif step == 2:
        _render_biz_approval(st.session_state.orch)

    elif step == 3:
        _render_gold_insights(st.session_state.orch)


def page_history():
    st.title("Run History / Audit Trail")
    runs = PipelineState.list_runs()
    if not runs:
        st.write("No runs yet.")
        return
    for path in runs:
        state = PipelineState.load(path.stem)
        status_icon = "✅" if state.status == "complete" else "⏳"
        with st.expander(f"{status_icon} {state.run_id} — {state.status} — {state.created_at}"):
            st.write(f"**Source:** `{state.source_filename}`")
            c1, c2, c3 = st.columns(3)
            c1.metric("Bronze", state.bronze_rows)
            c2.metric("Silver", state.silver_rows or "—")
            c3.metric("Gold", state.gold_rows or "—")

            if state.dq_rules:
                st.markdown("**DQ Rules Approved:**")
                for r in state.dq_rules:
                    icon = "✅" if r.get("approved") else "⬜"
                    st.write(f"{icon} {r['id']}: {r['issue']}")

            if state.business_rules:
                st.markdown("**Business Rules Approved:**")
                for r in state.business_rules:
                    icon = "✅" if r.get("approved") else "⬜"
                    st.write(f"{icon} {r['id']}: {r['description']}")

            if state.narrative:
                st.info(state.narrative)

            st.markdown("**Agent log:**")
            for entry in state.log:
                st.markdown(f"`{entry['timestamp']}` **{entry['agent']}** — {entry['action']}  \n{entry['detail']}")


def main():
    init_session()
    page = sidebar()
    if page == "Run Pipeline":
        page_run()
    else:
        page_history()


if __name__ == "__main__":
    main()
