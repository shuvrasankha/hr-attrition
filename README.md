# HR Attrition — Agentic Medallion Pipeline

## Business Problem
Identify which teams and departments are losing their **highest-performing**
employees, and understand the likely reasons why — so HR can act before more
key talent leaves. Overall attrition is a vanity metric; this pipeline
specifically isolates attrition *among top performers*, which is the number
that should worry leadership.

## Architecture

**Medallion layers**
- **Bronze** — raw ingested CSV, untouched, with ingestion metadata.
- **Silver** — cleaned/conformed data (deduped, standardized casing and
  dates, imputed nulls) plus enriched business fields (`is_top_performer`,
  `tenure_years`, `high_overtime`).
- **Gold** — department-level and team-level aggregated attrition metrics,
  risk ranking, and likely contributing factors, plus an LLM-generated
  executive narrative.

**Agents** (`src/agents/`)
| Agent | File | Responsibility |
|---|---|---|
| Orchestrator | `orchestrator.py` | Sequences all agents, exposes checkpoint methods the UI calls, persists state after every step. |
| Ingestion Agent | `ingestion_agent.py` | Lands raw CSV into Bronze + metadata. |
| Data Quality Agent | `dq_agent.py` | Profiles Bronze (nulls, dupes, mixed casing, mixed date formats), **proposes** cleaning rules — never applies them itself. |
| Transformation Agent | `transform_agent.py` | Applies human-**approved** DQ rules to build Silver; separately proposes business rules (top-performer definition, tenure calc, overtime threshold), applies human-approved versions to enrich Silver. |
| Analytics Agent | `analytics_agent.py` | Aggregates enriched Silver into Gold (department + team level), ranks departments by top-performer attrition rate, infers likely drivers (comp gap, overtime, engagement), generates the executive narrative. |
| QA/Validation Agent | `qa_agent.py` | Reconciles row counts and key metrics across Bronze → Silver → Gold to catch pipeline errors before they reach the UI. |

**Human-in-the-loop checkpoints**
1. After Data Quality Agent profiles Bronze → user reviews/edits/unchecks
   each proposed cleaning rule in the UI before Silver is built.
2. After Transformation Agent proposes business rules → user reviews/edits
   before Gold aggregation runs.

All approvals, agent actions, and row counts at each stage are logged to
`state.log` and persisted per run under `/runs/<run_id>.json` — this is the
audit trail / run history shown in the UI.

## Tech Stack
- **Language:** Python
- **Orchestration:** custom lightweight orchestrator (`Orchestrator` class) —
  chosen over a heavier framework (LangGraph/CrewAI) to keep the checkpoint
  hand-off explicit and easy to follow for grading; each agent is a plain
  function taking/returning a `pandas.DataFrame` and the shared `PipelineState`.
- **Data processing:** pandas / numpy
- **Storage:** local filesystem, one folder per medallion layer (`data/bronze`,
  `data/silver`, `data/gold`), CSV per run keyed by `run_id`.
- **LLM:** HuggingFace Inference API (`microsoft/Phi-3.5-mini-instruct`),
  used for turning structured findings into plain-English rule proposals
  and the executive narrative. If `HUGGINGFACE_TOKEN` is not set, every
  LLM call falls back to a deterministic templated summary so the pipeline
  still runs fully offline (see `src/llm.py`).
- **UI:** Streamlit (`app.py`) — upload/run, two approval screens, Gold
  insights dashboard with charts, team-level drill-down, and a run-history/audit page.

## Setup & Run Instructions

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# optional — enables live LLM narratives instead of the offline fallback
export HUGGINGFACE_TOKEN=hf_...

# UI
streamlit run app.py

# or, headless end-to-end smoke test (auto-approves default rules)
python run_cli.py
```

There's no dataset required to try it — the app has a **"Use synthetic
sample dataset"** button that generates a realistic, deliberately messy HR
CSV (1,200+ rows) with intentional attrition signal baked in (Sales loses
top performers to compensation gaps; Engineering loses them to overtime),
so the pipeline has something real to surface on first run.

## Sample Output
Running `run_cli.py` against the synthetic dataset produces a Gold-layer
ranking such as:

```
 department  risk_rank  headcount  attrition_rate  top_performer_attrition_rate  comp_gap  overtime_gap  likely_driver
Engineering          1      240.0            13.8                          31.9   -1031.0           8.7  elevated overtime before exit; low engagement
      Sales          2      240.0            14.2                          23.5    2735.0           1.9  compensation gap (~$2,735 below staying top performers)
    Support          3      240.0            13.8                          10.8      430.0          -1.0  low engagement score prior to exit
  Marketing          4      240.0             9.6                           9.6     -498.0          -1.9  no dominant single factor identified
    Finance          5      240.0            11.7                           7.1    2702.0           3.7  comp gap; elevated overtime; low engagement
```

With team-level drill-down surfacing the highest-risk teams:

```
 department        team  top_performer_attrition_rate
Engineering    Data Eng                          41.2
      Sales  Mid-Market                          38.9
Engineering Product Eng                          31.2
```

## Design Decisions & Tradeoffs
- **Proposal vs. apply separation:** every agent that makes a judgment call
  (which rows are duplicates, how nulls should be handled, what counts as a
  "top performer") only *proposes* — a separate apply step, gated on
  approval, does the actual mutation. This makes the human checkpoint a real
  gate rather than a rubber stamp.
- **Offline-safe LLM calls:** all narrative/summary generation degrades
  gracefully to deterministic text if no API key is present, so the pipeline
  is always demoable without depending on network/API availability.
- **CSV-per-layer-per-run rather than a database:** keeps the solution
  inspectable (you can open any Bronze/Silver/Gold file directly) and keeps
  the audit trail (`/runs/*.json`) simple to reason about for a project of
  this scope.
- **QA Agent as validation gate:** catches row-count drift, duplicate IDs in
  Silver, and metric bounds before results reach the UI — a lightweight
  safeguard rather than a full data contracts framework.

## Demo Video
_Add link here after recording (4–5 min, voice-over): executive summary →
architecture walkthrough → live UI walkthrough of a full run including an
approval step._
