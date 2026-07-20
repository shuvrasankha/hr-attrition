"""Shared pipeline run state — passed between agents and persisted to /runs
as JSON for the audit trail / run history UI."""
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional

from src.config import RUNS_DIR


@dataclass
class AgentLogEntry:
    agent: str
    action: str
    detail: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PipelineState:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    source_filename: str = ""
    status: str = "created"  # created -> ingested -> dq_proposed -> dq_approved -> silver_ready
                              # -> rules_proposed -> rules_approved -> gold_ready -> complete
    bronze_rows: int = 0
    silver_rows: int = 0
    gold_rows: int = 0
    dq_rules: list = field(default_factory=list)          # proposed by DQ Agent
    dq_rules_approved: bool = False
    business_rules: list = field(default_factory=list)     # proposed by Transformation Agent
    business_rules_approved: bool = False
    narrative: str = ""
    log: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add_log(self, agent: str, action: str, detail: str):
        self.log.append(asdict(AgentLogEntry(agent=agent, action=action, detail=detail)))

    def save(self):
        path = RUNS_DIR / f"{self.run_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        return path

    @staticmethod
    def load(run_id: str) -> "PipelineState":
        path = RUNS_DIR / f"{run_id}.json"
        with open(path) as f:
            data = json.load(f)
        return PipelineState(**data)

    @staticmethod
    def list_runs():
        return sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
