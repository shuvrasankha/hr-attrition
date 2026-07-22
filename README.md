# HR Attrition — Agentic Medallion Pipeline

An AI-powered data pipeline that ingests HR CSV data and answers: **Which teams and departments are losing their highest-performing employees, and why?**

Built with a Bronze → Silver → Gold medallion architecture, six specialized agents, and human-in-the-loop approval checkpoints.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/shuvrasankha/hr-attrition.git
cd hr-attrition

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Enable AI rule proposals
#    Get a free token from https://huggingface.co/settings/tokens
echo "HUGGINGFACE_TOKEN=hf_your_token_here" > .env

# 5. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**. Upload any HR CSV and follow the guided pipeline.

> **No data?** Generate a sample file:
> ```bash
> python -c "from src.data_gen import generate; generate().to_csv('sample.csv', index=False)"
> ```
> Then upload `sample.csv` in the app.

## What It Does

```
Upload CSV → Ingest (Bronze) → Clean (Silver) → Enrich (Silver+) → Aggregate (Gold) → Insights
                  ↑ DQ approval              ↑ Business rules approval
```

1. **Upload** — drag and drop one or more HR CSVs. The app validates required columns instantly.
2. **Data Quality Approval** — the DQ Agent profiles your data (duplicates, nulls, casing, date formats) and proposes cleaning rules. You review, edit, or uncheck before Silver is built.
3. **Business Rules Approval** — the Transformation Agent proposes how to define "top performer", calculate tenure, and flag risk factors. You review before Gold is built.
4. **Gold Insights** — risk-ranked departments, team-level drill-down, compensation gap charts, data lineage trace, and an AI-generated executive narrative.

Every step is logged for the audit trail under `runs/`.

## Required Data Format

Upload CSV(s) with these columns:

| Column | Type | Description |
|---|---|---|
| `employee_id` | string | Unique identifier |
| `department` | string | Department name |
| `team` | string | Team within department |
| `performance_rating` | numeric | Rating 1–5 |
| `compensation` | numeric | Annual salary |
| `attrition_flag` | string | "Yes" or "No" |
| `hire_date` | date | Start date (any format) |
| `overtime_hours_per_month` | numeric | Monthly OT hours |
| `engagement_score` | numeric | Engagement score |

**Optional columns:** `exit_date`, `last_promotion_date`, `satisfaction_score`, `training_hours`, `distance_from_home`

The pipeline is fault-tolerant: missing optional columns get safe defaults, wrong types are coerced, and all-null non-required columns are dropped.

## Architecture

### Medallion Layers

| Layer | Description | Stored at |
|---|---|---|
| **Bronze** | Raw ingested CSV, untouched | `data/bronze/` |
| **Silver** | Cleaned (deduped, standardized, imputed) | `data/silver/` |
| **Silver Enriched** | Silver + business-rule columns (`is_top_performer`, `tenure_years`, `high_overtime`) | `data/silver/` |
| **Gold** | Department + team aggregated metrics, risk scores, drivers | `data/gold/` |

### Agents

| Agent | File | What it does |
|---|---|---|
| **Ingestion** | `src/agents/ingestion_agent.py` | Lands raw CSV into Bronze, coerces types, validates columns |
| **Data Quality** | `src/agents/dq_agent.py` | Profiles Bronze, asks AI to propose cleaning rules |
| **Transformation** | `src/agents/transform_agent.py` | Applies DQ rules → Silver; proposes business rules → Silver Enriched |
| **Analytics** | `src/agents/analytics_agent.py` | Aggregates Gold, computes risk scores, identifies drivers, generates narrative |
| **QA** | `src/agents/qa_agent.py` | Validates row counts and metrics across layers |
| **Orchestrator** | `src/agents/orchestrator.py` | Sequences agents, manages state, exposes UI checkpoints |

### Human-in-the-Loop

Two approval gates ensure you stay in control:

1. **DQ Rules** — edit proposed cleaning actions, uncheck rules you don't want
2. **Business Rules** — edit top-performer threshold, tenure logic, overtime flag

### AI Integration

When `HUGGINGFACE_TOKEN` is set, agents send data profiles to **Llama-3.1-8B-Instruct** via HuggingFace Inference API and receive structured JSON rule proposals. The AI sees:

- DQ Agent: row counts, nulls, duplicates, casing issues, date formats, sample rows
- Transformation Agent: schema, column types, sample rows
- Analytics Agent: Gold metrics → executive narrative

Without a token, deterministic fallback rules are used — the pipeline always completes.

## Project Structure

```
hr-attrition/
├── app.py                      # Streamlit UI — upload, approval screens, Gold dashboard
├── run_cli.py                  # CLI smoke test (headless, auto-approves all rules)
├── requirements.txt            # Python dependencies
├── setup.sh                    # Mac/Linux one-click setup
├── setup.bat                   # Windows one-click setup
├── .env                        # HUGGINGFACE_TOKEN (not committed)
├── .gitignore                  # Ignores .env, __pycache__, generated data/runs
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Paths, LLM config, business definition defaults
│   ├── llm.py                  # HuggingFace wrapper with retry, backoff, fallback
│   ├── state.py                # PipelineState dataclass — persisted as JSON per run
│   ├── data_gen.py             # Synthetic HR dataset generator (1,200+ rows)
│   │
│   └── agents/
│       ├── __init__.py
│       ├── ingestion_agent.py  # CSV → Bronze, type coercion, column validation
│       ├── dq_agent.py         # Profiles Bronze, AI proposes cleaning rules
│       ├── transform_agent.py  # Applies DQ rules → Silver; AI proposes business rules → Silver Enriched
│       ├── analytics_agent.py  # Aggregates Gold, risk scores, drivers, executive narrative
│       ├── qa_agent.py         # Validates row counts and metrics across layers
│       └── orchestrator.py     # Sequences agents, manages state, UI checkpoints
│
├── data/                       # Generated layer CSVs (per run, gitignored)
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
└── runs/                       # Run history JSON — audit trail (gitignored)
```

## Fail-Safe Design

The pipeline handles edge cases gracefully:

- **Empty files** — rejected with clear error message
- **Missing required columns** — listed with descriptions before processing starts
- **Wrong data types** — numeric columns coerced with `errors="coerce"`, dates parsed flexibly
- **All-null columns** — required columns kept with NaN defaults, non-required dropped
- **Division by zero** — `_safe_mean()` and `max(count, 1)` guards throughout
- **Empty DataFrames** — each agent returns safe defaults instead of crashing
- **LLM failures** — 2 retries with linear backoff, then silent fallback to hardcoded rules
- **JSON parse errors** — AI responses wrapped in try/except, fallback to defaults
- **Missing enrichment columns** — `is_top_performer`, `tenure_years`, `high_overtime` get safe defaults

## Tech Stack

- **Python** with pandas, numpy
- **Streamlit** for the interactive UI
- **HuggingFace Inference API** (Llama-3.1-8B-Instruct) for AI rule proposals
- **Custom orchestrator** — lightweight, no external framework dependencies

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the activated venv |
| `HUGGINGFACE_TOKEN not set` | Create `.env` with `HUGGINGFACE_TOKEN=hf_...` (pipeline works without it using fallback rules) |
| Gold layer fails | Check that your CSV has `department`, `is_top_performer`, `attrition_flag`, `compensation` columns |
| Streamlit won't start | Try `streamlit run app.py --server.port 8502` (port 8501 may be in use) |
| Empty Gold layer | Ensure your data has at least 2 departments and some attrition (`attrition_flag = Yes`) |

## License

This project was built as an academic demonstration of agentic data pipeline design.
