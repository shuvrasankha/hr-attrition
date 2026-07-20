"""Orchestrator Agent: sequences the other agents, exposes explicit
checkpoint functions the UI calls after a human approves/edits proposed
rules. Keeps pipeline state in one place and persists it after every step
so runs are resumable and auditable."""
import pandas as pd

from src.agents import ingestion_agent, dq_agent, transform_agent, analytics_agent, qa_agent
from src.state import PipelineState


class Orchestrator:
    def __init__(self, state: PipelineState = None):
        self.state = state or PipelineState()
        self.bronze_df: pd.DataFrame | None = None
        self.silver_df: pd.DataFrame | None = None
        self.silver_enriched_df: pd.DataFrame | None = None
        self.gold_df: pd.DataFrame | None = None
        self.team_gold_df: pd.DataFrame | None = None

    def ingest(self, source_df: pd.DataFrame, filename: str):
        self.bronze_df = ingestion_agent.run(self.state, source_df, filename)
        self.state.save()
        return self.bronze_df

    def propose_dq_rules(self):
        rules = dq_agent.propose_rules(self.state, self.bronze_df)
        self.state.save()
        return rules

    def apply_dq_rules(self, approved_rules: list):
        self.state.dq_rules = approved_rules
        self.silver_df = transform_agent.apply_dq_rules(self.state, self.bronze_df, approved_rules)
        self.state.save()
        return self.silver_df

    def propose_business_rules(self):
        rules = transform_agent.propose_business_rules(self.state, self.silver_df)
        self.state.save()
        return rules

    def apply_business_rules(self, approved_rules: list):
        self.state.business_rules = approved_rules
        self.silver_enriched_df = transform_agent.apply_business_rules(self.state, self.silver_df, approved_rules)
        self.state.save()
        return self.silver_enriched_df

    def build_gold_and_narrative(self):
        self.gold_df = analytics_agent.build_gold(self.state, self.silver_enriched_df)
        self.team_gold_df = analytics_agent.build_team_gold(self.state, self.silver_enriched_df)
        narrative = analytics_agent.generate_narrative(self.state, self.gold_df, self.team_gold_df)
        self.state.save()
        return self.gold_df, narrative

    def run_qa_validation(self) -> dict:
        report = qa_agent.validate(self.state, self.bronze_df, self.silver_df, self.silver_enriched_df, self.gold_df)
        self.state.save()
        return report
