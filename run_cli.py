"""
End-to-end CLI test runner (no UI). Auto-approves the default proposed
rules at each checkpoint so the whole pipeline can be smoke-tested from
the command line: `python run_cli.py`
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import data_gen
from src.agents.orchestrator import Orchestrator
from src.llm import is_llm_enabled


def main():
    llm_status = "ON (HuggingFace — Llama-3.1)" if is_llm_enabled() else "OFF (using fallback text)"
    print(f"=== AI Status: {llm_status} ===\n")
    df = data_gen.generate()
    print(f"Generated synthetic dataset: {len(df)} rows")

    orch = Orchestrator()
    orch.ingest(df, "hr_attrition_raw.csv (synthetic)")
    print(f"\n[Bronze] {orch.state.bronze_rows} rows ingested. run_id={orch.state.run_id}")

    dq_rules = orch.propose_dq_rules()
    print(f"\n[DQ Agent] proposed {len(dq_rules)} rules:")
    for r in dq_rules:
        print(f"  - {r['id']}: {r['issue']} -> {r['proposed_action']}")

    orch.apply_dq_rules(dq_rules)
    qa1 = orch.run_qa_validation()
    print(f"\n[Silver] {orch.state.silver_rows} rows after cleaning.")
    print(f"[QA] {'All checks passed' if qa1['passed'] else 'SOME CHECKS FAILED'} ({sum(c['passed'] for c in qa1['checks'])}/{len(qa1['checks'])})")

    biz_rules = orch.propose_business_rules()
    print(f"\n[Transformation Agent] proposed {len(biz_rules)} business rules:")
    for r in biz_rules:
        print(f"  - {r['id']}: {r['description']}")

    orch.apply_business_rules(biz_rules)
    print("\n[Silver enriched] business rules applied.")

    gold, narrative = orch.build_gold_and_narrative()
    qa2 = orch.run_qa_validation()
    print(f"\n[QA] {'All checks passed' if qa2['passed'] else 'SOME CHECKS FAILED'} ({sum(c['passed'] for c in qa2['checks'])}/{len(qa2['checks'])})")

    print(f"\n[Gold - Department] {len(gold)} rows:\n")
    print(gold[[
        "department", "risk_rank", "headcount", "attrition_rate",
        "top_performer_attrition_rate", "comp_gap", "overtime_gap",
        "risk_score", "risk_label", "likely_driver"
    ]].to_string(index=False))

    print(f"\n[Gold - Team] {len(orch.team_gold_df)} rows:\n")
    print(orch.team_gold_df[[
        "department", "team", "headcount", "top_performers",
        "top_performer_attritions", "top_performer_attrition_rate",
        "risk_score", "risk_label", "likely_driver"
    ]].head(10).to_string(index=False))

    print("\n[Narrative]\n" + narrative)
    print(f"\nRun saved: runs/{orch.state.run_id}.json")


if __name__ == "__main__":
    main()
