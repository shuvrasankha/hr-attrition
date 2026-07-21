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
  dates, imputed nulls).
- **Silver Enriched** — Silver plus business-rule-derived columns
  (`is_top_performer`, `tenure_years`, `high_overtime`).
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
`state.log` and persisted per run under `runs/<run_id>.json` — this is the
audit trail / run history shown in the UI.

## AI Integration
Both the DQ Agent and Transformation Agent use an LLM to propose rules:

- **DQ Agent** sends a data profile (row counts, nulls, duplicates, casing
  issues, date formats, sample rows) to the LLM and asks it to propose
  cleaning rules as structured JSON.
- **Transformation Agent** sends the Silver schema, column types, and sample
  rows to the LLM and asks it to propose business rules (top-performer
  definition, tenure calculation, overtime thresholds) as structured JSON.
- **Analytics Agent** uses the LLM to generate the executive narrative from
  the Gold metrics.

If the LLM is unavailable or returns unparseable JSON, every agent falls
back to deterministic hardcoded rules — the pipeline always completes.

## Tech Stack
- **Language:** Python
- **Orchestration:** custom lightweight orchestrator (`Orchestrator` class) —
  each agent is a plain function taking/returning a `pandas.DataFrame` and
  the shared `PipelineState`.
- **Data processing:** pandas / numpy
- **Storage:** local filesystem, one folder per medallion layer (`data/bronze`,
  `data/silver`, `data/gold`), CSV per run keyed by `run_id`.
- **LLM:** HuggingFace Inference API (`meta-llama/Llama-3.1-8B-Instruct`),
  used for generating rule proposals and the executive narrative. If
  `HUGGINGFACE_TOKEN` is not set, every LLM call falls back to a
  deterministic templated summary so the pipeline runs fully offline
  (see `src/llm.py`).
- **UI:** Streamlit (`app.py`) — upload CSV(s), two human-approval screens
  for DQ and business rules, Gold insights dashboard with charts, team-level
  drill-down, data lineage, and a run-history/audit page.

## Setup & Run Instructions

### Mac / Linux
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# optional — enables live LLM rule proposals and narratives
echo "HUGGINGFACE_TOKEN=hf_your_token_here" > .env

# run the UI
streamlit run app.py
```

### Windows
```cmd
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# optional — enables live LLM rule proposals and narratives
echo HUGGINGFACE_TOKEN=hf_your_token_here > .env

# run the UI
streamlit run app.py
```

### CLI smoke test
```bash
python run_cli.py
```

### Getting a HuggingFace Token
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token (read access is sufficient)
4. Paste it in `.env` as `HUGGINGFACE_TOKEN=hf_...`

## Upload Format
Upload one or more CSV files containing HR data. Required columns:
- `employee_id` — unique identifier
- `department` — department name
- `team` — team name within the department
- `performance_rating` — numeric rating (1–5)
- `compensation` — annual salary
- `attrition_flag` — "Yes" or "No"
- `hire_date` — employment start date
- `overtime_hours_per_month` — numeric
- `engagement_score` — numeric

Optional columns: `exit_date`, `last_promotion_date`, `satisfaction_score`,
`training_hours`, `distance_from_home`.

The pipeline works with any CSV matching this schema. To try it without your
own data, generate a synthetic dataset with `python -c "from src.data_gen import
generate; generate().to_csv('sample.csv', index=False)"` and upload that file.

## Sample Output
Running the pipeline produces a Gold-layer ranking such as:

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
  only *proposes* — a separate apply step, gated on approval, does the actual
  mutation. This makes the human checkpoint a real gate rather than a rubber
  stamp.
- **AI-assisted rule generation:** the LLM analyzes actual data profiles
  and proposes rules, rather than relying solely on hardcoded logic. The
  pipeline still works fully offline via deterministic fallbacks.
- **Multi-file upload:** multiple CSVs are concatenated before processing,
  so the pipeline handles split exports or department-level files.
- **Offline-safe LLM calls:** all narrative/summary generation degrades
  gracefully to deterministic text if no API key is present, so the pipeline
  is always demoable without depending on network/API availability.
- **CSV-per-layer-per-run rather than a database:** keeps the solution
  inspectable (you can open any Bronze/Silver/Gold file directly) and keeps
  the audit trail (`runs/*.json`) simple to reason about for a project of
  this scope.
- **QA Agent as validation gate:** catches row-count drift, duplicate IDs in
  Silver, and metric bounds before results reach the UI — a lightweight
  safeguard rather than a full data contracts framework.
