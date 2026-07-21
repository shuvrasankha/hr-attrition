import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src.agents.orchestrator import Orchestrator
from src.agents.ingestion_agent import validate_columns, REQUIRED_COLUMNS
from src.state import PipelineState

st.set_page_config(
    page_title="HR Attrition Pipeline",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { padding-top: 1rem; }
    .pipeline-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white; padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
    }
    .pipeline-header h1 { color: white !important; margin-bottom: 0.25rem; font-size: 1.6rem; }
    .pipeline-header p { color: #a0aec0 !important; margin: 0; font-size: 0.95rem; }
    .step-badge {
        display: inline-block; background: #e2e8f0; color: #2d3748;
        padding: 0.2rem 0.75rem; border-radius: 9999px; font-size: 0.8rem;
        font-weight: 600; margin-bottom: 0.75rem;
    }
    .step-badge.active { background: #3182ce; color: white; }
    .step-badge.done { background: #38a169; color: white; }
    div[data-testid="stMetric"] {
        background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stMetric"] label { font-size: 0.8rem !important; }
    .rule-card {
        background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 1rem 1.2rem; margin-bottom: 0.5rem;
    }
    .sidebar-status { font-size: 0.82rem; }
    .narrative-box {
        background: #ebf8ff; border-left: 4px solid #3182ce;
        padding: 1rem 1.25rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
    }
    .narrative-box p { margin: 0; color: #2d3748; line-height: 1.6; }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

STEPS = ["Upload", "Data Quality", "Business Rules", "Gold Insights"]


def init_session():
    defaults = {
        "orch": None, "step": 0, "qa_report": None,
        "uploaded_dfs": None, "uploaded_names": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def sidebar():
    st.sidebar.markdown("### :material/database: Medallion Pipeline")
    st.sidebar.caption("Bronze \u2192 Silver \u2192 Gold")

    page = st.sidebar.radio("Navigate", ["Run Pipeline", "Run History"], label_visibility="collapsed")

    if st.session_state.orch:
        st.sidebar.divider()
        state = st.session_state.orch.state

        steps_html = ""
        for i, s in enumerate(STEPS):
            if i < st.session_state.step:
                steps_html += f'<span class="step-badge done">\u2713 {s}</span> '
            elif i == st.session_state.step:
                steps_html += f'<span class="step-badge active">\u25b6 {s}</span> '
            else:
                steps_html += f'<span class="step-badge">{s}</span> '
        st.sidebar.markdown(steps_html, unsafe_allow_html=True)

        st.sidebar.caption(f"Run `{state.run_id[:8]}` \u00b7 Status: **{state.status.replace('_', ' ').title()}**")

        if st.session_state.orch.bronze_df is not None:
            st.sidebar.divider()
            c1, c2, c3 = st.sidebar.columns(3)
            c1.metric("Bronze", state.bronze_rows)
            c2.metric("Silver", state.silver_rows or "\u2014")
            c3.metric("Gold", state.gold_rows or "\u2014")

        st.sidebar.divider()
        if st.sidebar.button(":material/refresh: Start over", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    return page


def _render_dq_approval(orch: Orchestrator):
    st.markdown('<span class="step-badge active">Step 2</span>', unsafe_allow_html=True)
    st.markdown("### Data Quality Agent")
    st.caption(f"Bronze layer: **{orch.state.bronze_rows} rows** ingested from `{orch.state.source_filename}`")

    if orch.bronze_df is not None and not orch.bronze_df.empty:
        with st.expander(":material/preview: Preview Bronze data (first 20 rows)"):
            st.dataframe(orch.bronze_df.head(20), use_container_width=True, hide_index=True)
    else:
        st.warning("Bronze data is empty. Cannot proceed.")
        return

    if not orch.state.dq_rules:
        with st.spinner("Data Quality Agent profiling Bronze data..."):
            try:
                orch.propose_dq_rules()
            except Exception as e:
                st.error(f"Error during DQ profiling: {e}")
                return

    if not orch.state.dq_rules:
        st.info("No cleaning rules proposed — data looks clean.")
        if st.button("Continue to Business Rules", type="primary", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
        return

    st.info(
        "The DQ Agent profiled the data and found issues with duplicates, missing values, "
        "inconsistent casing, and mixed date formats. Review each proposed fix below."
    )

    edited_rules = []
    for rule in orch.state.dq_rules:
        with st.container(border=True):
            c1, c2 = st.columns([0.06, 0.94])
            with c1:
                approved = st.checkbox("Apply", value=rule.get("approved", True), key=f"dq_{rule['id']}")
            with c2:
                st.markdown(f"**{rule['issue']}**")
                new_action = st.text_input(
                    "Proposed action", value=rule["proposed_action"],
                    key=f"dq_action_{rule['id']}", label_visibility="collapsed",
                )
            r = dict(rule)
            r["approved"] = approved
            r["proposed_action"] = new_action
            edited_rules.append(r)

    st.markdown("")
    if st.button(":material/check_circle: Approve rules & build Silver layer", type="primary", use_container_width=True):
        try:
            orch.apply_dq_rules(edited_rules)
        except Exception as e:
            st.error(f"Error building Silver layer: {e}")
            return
        with st.spinner("Running QA validation..."):
            qa_report = orch.run_qa_validation()
        st.session_state.qa_report = qa_report
        st.session_state.step = 2
        st.rerun()


def _render_biz_approval(orch: Orchestrator):
    st.markdown('<span class="step-badge active">Step 3</span>', unsafe_allow_html=True)
    st.markdown("### Transformation Agent")
    st.caption(f"Silver layer: **{orch.state.silver_rows} rows** after cleaning")

    if st.session_state.get("qa_report"):
        qa = st.session_state.qa_report
        if qa["passed"]:
            st.success(f"QA passed ({sum(c['passed'] for c in qa['checks'])}/{len(qa['checks'])} checks)")
        else:
            st.warning(f"QA issues ({sum(c['passed'] for c in qa['checks'])}/{len(qa['checks'])} checks passed)")
        with st.expander(":material/verified: QA Validation Details"):
            for c in qa["checks"]:
                icon = ":material/check_circle:" if c["passed"] else ":material/cancel:"
                st.markdown(f"{icon} **{c['name']}**  \n{c['detail']}")

    if orch.silver_df is not None and not orch.silver_df.empty:
        with st.expander(":material/preview: Preview Silver data (first 20 rows)"):
            st.dataframe(orch.silver_df.head(20), use_container_width=True, hide_index=True)
    else:
        st.warning("Silver data is empty. Cannot proceed.")
        return

    if not orch.state.business_rules:
        with st.spinner("Transformation Agent drafting business rule proposals..."):
            try:
                orch.propose_business_rules()
            except Exception as e:
                st.error(f"Error during business rule proposal: {e}")
                return

    if not orch.state.business_rules:
        st.info("No business rules proposed.")
        if st.button("Continue to Gold", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
        return

    st.info(
        "The Transformation Agent proposes business definitions that determine "
        "which employees are counted as top performers and how attrition metrics are calculated."
    )

    edited_rules = []
    for rule in orch.state.business_rules:
        with st.container(border=True):
            c1, c2 = st.columns([0.06, 0.94])
            with c1:
                approved = st.checkbox("Apply", value=rule.get("approved", True), key=f"biz_{rule['id']}")
            with c2:
                st.markdown(f"**{rule['name'].replace('_', ' ').title()}**")
                new_desc = st.text_input(
                    "Description", value=rule["description"],
                    key=f"biz_desc_{rule['id']}", label_visibility="collapsed",
                )
            r = dict(rule)
            r["approved"] = approved
            r["description"] = new_desc
            edited_rules.append(r)

    st.markdown("")
    if st.button(":material/check_circle: Approve rules & build Gold layer", type="primary", use_container_width=True):
        try:
            orch.apply_business_rules(edited_rules)
            with st.spinner("Building Gold layer..."):
                orch.build_gold_and_narrative()
            st.session_state.step = 3
            st.rerun()
        except Exception as e:
            st.error(f"Error building Gold layer: {e}")


def _render_gold_insights(orch: Orchestrator):
    gold = orch.gold_df
    team_gold = orch.team_gold_df

    if gold is None or (hasattr(gold, 'empty') and gold.empty):
        st.error("Gold layer is empty or could not be generated.")
        if orch.silver_enriched_df is not None:
            st.write(f"Silver enriched rows: {len(orch.silver_enriched_df)}")
        st.write(f"State status: {orch.state.status}")
        if orch.state.log:
            st.markdown("**Last agent log:**")
            last_log = orch.state.log[-1]
            st.write(f"`{last_log['timestamp']}` **{last_log['agent']}** — {last_log['action']}")
            st.write(last_log['detail'])
        return

    st.markdown('<span class="step-badge done">Complete</span>', unsafe_allow_html=True)
    st.markdown("### Gold Layer Insights")

    st.markdown(f'<div class="narrative-box"><p>{orch.state.narrative}</p></div>', unsafe_allow_html=True)

    st.markdown("#### Risk Overview")
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

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top-performer attrition rate by department**")
        st.bar_chart(gold.set_index("department")[["top_performer_attrition_rate"]], color="#e74c3c")
    with c2:
        st.markdown("**Overall attrition rate by department**")
        st.bar_chart(gold.set_index("department")[["attrition_rate"]], color="#3498db")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Compensation gap: leavers vs stayers**")
        comp_data = gold[["department", "comp_gap"]].dropna().set_index("department")
        if not comp_data.empty:
            st.bar_chart(comp_data, color="#e67e22")
        else:
            st.info("No compensation gap data.")
    with c4:
        st.markdown("**Risk score by department**")
        st.bar_chart(gold.set_index("department")[["risk_score"]], color="#8e44ad")

    st.divider()
    st.markdown("#### Department Risk Ranking")
    display_cols = [
        "risk_rank", "department", "headcount", "attrition_rate",
        "top_performers", "top_performer_attritions", "top_performer_attrition_rate",
        "comp_gap", "overtime_gap", "risk_score", "risk_label", "likely_driver",
    ]
    existing_cols = [c for c in display_cols if c in gold.columns]
    st.dataframe(gold[existing_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Team-Level Drill-Down")
    st.caption("Top 10 teams by attrition risk (cross-department view)")
    if team_gold is not None and not team_gold.empty:
        team_display = team_gold.head(10)
        team_cols = [
            "department", "team", "headcount", "top_performers", "top_performer_attritions",
            "top_performer_attrition_rate", "comp_gap", "risk_score", "risk_label", "likely_driver",
        ]
        existing_team_cols = [c for c in team_cols if c in team_display.columns]
        st.dataframe(team_display[existing_team_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No team-level data available.")

    st.divider()
    st.markdown("#### Data Lineage")
    with st.expander(":material/account_tree: Trace raw rows to Gold aggregates"):
        selected_dept = st.selectbox("Select a department", gold["department"].tolist())
        if selected_dept and orch.silver_enriched_df is not None:
            dept_silver = orch.silver_enriched_df[orch.silver_enriched_df["department"] == selected_dept]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Silver rows", len(dept_silver))
            m2.metric("Top performers", int(dept_silver["is_top_performer"].sum()) if "is_top_performer" in dept_silver.columns else 0)
            m3.metric("Total attritions", int(dept_silver["attrition_flag"].sum()) if "attrition_flag" in dept_silver.columns else 0)
            tp_attritions = int((dept_silver["is_top_performer"] & dept_silver["attrition_flag"]).sum()) if "is_top_performer" in dept_silver.columns and "attrition_flag" in dept_silver.columns else 0
            m4.metric("TP attritions", tp_attritions)

            trace_cols = ["employee_id", "team", "performance_rating", "compensation", "tenure_years", "high_overtime"]
            available_trace_cols = [c for c in trace_cols if c in dept_silver.columns]
            if available_trace_cols:
                st.markdown("**Sample top-performer leavers:**")
                mask = dept_silver["is_top_performer"] & dept_silver["attrition_flag"] if "is_top_performer" in dept_silver.columns and "attrition_flag" in dept_silver.columns else pd.Series([False] * len(dept_silver), index=dept_silver.index)
                trace_df = dept_silver[mask][available_trace_cols].head(10)
                st.dataframe(trace_df, use_container_width=True, hide_index=True)

    with st.expander(":material/table_rows: Full Gold tables"):
        st.markdown("**Department-level**")
        st.dataframe(gold, use_container_width=True, hide_index=True)
        if team_gold is not None and not team_gold.empty:
            st.markdown("**Team-level**")
            st.dataframe(team_gold, use_container_width=True, hide_index=True)

    with st.expander(":material/list_alt: Agent run log (audit trail)"):
        if orch.state.log:
            for entry in orch.state.log:
                st.markdown(
                    f"`{entry['timestamp']}` **{entry['agent']}** \u2014 {entry['action']}  \n"
                    f"{entry['detail']}"
                )
        else:
            st.info("No log entries yet.")


def page_run():
    init_session()

    st.markdown("""
    <div class="pipeline-header">
        <h1>HR Attrition Pipeline</h1>
        <p>Which teams and departments are losing their highest-performing employees, and why?</p>
    </div>
    """, unsafe_allow_html=True)

    step = st.session_state.step

    if step == 0:
        st.markdown('<span class="step-badge active">Step 1</span>', unsafe_allow_html=True)
        st.markdown("### Upload your data")
        st.caption("Upload one or more HR CSV files to begin the pipeline.")

        uploaded = st.file_uploader(
            "Upload CSV(s)", type=["csv"], accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded:
            if not isinstance(uploaded, list):
                uploaded = [uploaded]
            st.session_state.uploaded_dfs = []
            st.session_state.uploaded_names = []
            for f in uploaded:
                try:
                    df_preview = pd.read_csv(f)
                    if df_preview.empty:
                        st.warning(f"**{f.name}** is empty — skipping.")
                        continue
                    st.session_state.uploaded_dfs.append(df_preview)
                    st.session_state.uploaded_names.append(f.name)
                    st.success(f"**{f.name}** \u2014 {len(df_preview)} rows, {len(df_preview.columns)} columns")
                    st.dataframe(df_preview.head(5), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Error reading **{f.name}**: {e}")

            # Column validation
            combined = pd.concat(st.session_state.uploaded_dfs, ignore_index=True)
            missing, extra, found = validate_columns(combined)

            st.markdown("#### Column check")
            if found:
                st.success(f"Found: {', '.join(found)}")
            if extra:
                st.info(f"Extra (will be ignored): {', '.join(extra)}")
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
                st.caption("Required columns and their purpose:")
                for col in missing:
                    st.write(f"- **{col}** \u2014 {REQUIRED_COLUMNS[col]}")

        has_files = bool(st.session_state.get("uploaded_dfs"))
        can_process = has_files and not missing if 'missing' in dir() else has_files

        if can_process:
            st.markdown("")
            if st.button(":material/upload: Process uploaded files", type="primary", use_container_width=True):
                try:
                    frames = st.session_state.uploaded_dfs
                    df = pd.concat(frames, ignore_index=True)
                    fname = ", ".join(st.session_state.uploaded_names)
                    orch = Orchestrator()
                    orch.ingest(df, fname)
                    st.session_state.orch = orch
                    st.session_state.step = 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Error combining files: {e}")

    elif step == 1:
        _render_dq_approval(st.session_state.orch)

    elif step == 2:
        _render_biz_approval(st.session_state.orch)

    elif step == 3:
        _render_gold_insights(st.session_state.orch)


def page_history():
    st.markdown("""
    <div class="pipeline-header">
        <h1>Run History</h1>
        <p>Audit trail of all pipeline executions</p>
    </div>
    """, unsafe_allow_html=True)

    runs = PipelineState.list_runs()
    if not runs:
        st.info("No runs yet. Upload a file to get started.")
        return

    for path in runs:
        try:
            state = PipelineState.load(path.stem)
        except Exception:
            continue
        done = state.status == "complete"
        icon = ":material/check_circle:" if done else ":material/pending:"
        label = state.status.replace("_", " ").title()
        with st.expander(f"{state.run_id[:8]} \u00b7 {label} \u00b7 {state.created_at}"):
            st.caption(f"Source: `{state.source_filename}`")
            c1, c2, c3 = st.columns(3)
            c1.metric("Bronze", state.bronze_rows)
            c2.metric("Silver", state.silver_rows or "\u2014")
            c3.metric("Gold", state.gold_rows or "\u2014")

            if state.dq_rules:
                st.markdown("**DQ Rules:**")
                for r in state.dq_rules:
                    icon = "\u2705" if r.get("approved") else "\u2b1c"
                    st.write(f"{icon} {r['id']}: {r['issue']}")

            if state.business_rules:
                st.markdown("**Business Rules:**")
                for r in state.business_rules:
                    icon = "\u2705" if r.get("approved") else "\u2b1c"
                    st.write(f"{icon} {r['id']}: {r['description']}")

            if state.narrative:
                st.info(state.narrative)

            st.markdown("**Agent log:**")
            for entry in state.log:
                st.markdown(
                    f"`{entry['timestamp']}` **{entry['agent']}** \u2014 {entry['action']}  \n"
                    f"{entry['detail']}"
                )


def main():
    init_session()
    page = sidebar()
    if page == "Run Pipeline":
        page_run()
    else:
        page_history()


if __name__ == "__main__":
    main()
