"""
Generates a synthetic, deliberately messy HR/attrition CSV so the pipeline
can be run and demoed end-to-end without a real dataset.

Signal baked into the data (so Gold-layer insights have something real to find):
  - Sales: below-market compensation drives high-performer attrition.
  - Engineering: heavy overtime drives high-performer attrition.
  - Support / Finance / Marketing: attrition roughly average, mostly
    driven by tenure/performance mix rather than comp or overtime.

Messiness baked in (so the Data Quality Agent has real work to do):
  - Inconsistent department casing ("Sales", "sales", "SALES")
  - Mixed date formats (ISO, US, slashes)
  - Duplicate employee_id rows (simulating repeated exports)
  - Missing values in performance_rating, compensation, engagement_score
"""
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

DEPARTMENTS = {
    "Sales": {"teams": ["Enterprise", "SMB", "Mid-Market"], "base_comp": 68000, "overtime_bias": 0.15, "comp_penalty": 0.82},
    "Engineering": {"teams": ["Platform", "Product Eng", "Data Eng"], "base_comp": 105000, "overtime_bias": 0.55, "comp_penalty": 1.0},
    "Support": {"teams": ["Tier 1", "Tier 2"], "base_comp": 52000, "overtime_bias": 0.20, "comp_penalty": 0.97},
    "Finance": {"teams": ["FP&A", "Accounting"], "base_comp": 78000, "overtime_bias": 0.10, "comp_penalty": 1.0},
    "Marketing": {"teams": ["Growth", "Brand"], "base_comp": 71000, "overtime_bias": 0.12, "comp_penalty": 0.98},
}

CASING_VARIANTS = {
    "Sales": ["Sales", "sales", "SALES"],
    "Engineering": ["Engineering", "engineering", "Engineering "],
    "Support": ["Support", "support"],
    "Finance": ["Finance", "FINANCE"],
    "Marketing": ["Marketing", "marketing"],
}

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]


def _random_date(start_year=2018, end_year=2026):
    start = date(start_year, 1, 1)
    end = date(end_year, 6, 30)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def _fmt_date(d):
    fmt = random.choice(DATE_FORMATS)
    return d.strftime(fmt)


def generate(n_employees: int = 1200, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    emp_id = 10000

    for dept, cfg in DEPARTMENTS.items():
        n_dept = int(n_employees / len(DEPARTMENTS))
        for _ in range(n_dept):
            emp_id += 1
            team = random.choice(cfg["teams"])
            hire_date = _random_date(2018, 2025)
            tenure_days_cap = (date(2026, 6, 30) - hire_date).days
            performance = np.clip(np.random.normal(3.2, 0.9), 1, 5)
            is_top_performer = performance >= 4

            overtime_hours = np.random.exponential(5) * (2 if random.random() < cfg["overtime_bias"] else 0.5)
            comp_multiplier = cfg["comp_penalty"] * (1 + (performance - 3) * 0.05)
            compensation = int(cfg["base_comp"] * comp_multiplier * np.random.normal(1, 0.08))
            engagement = np.clip(np.random.normal(3.4, 1.0), 1, 5)

            # attrition probability driven by department-specific stressor
            attrition_prob = 0.10
            if is_top_performer:
                if dept == "Sales":
                    attrition_prob += 0.35 * (1 - comp_multiplier)  # underpaid top performers leave
                if dept == "Engineering":
                    attrition_prob += 0.03 * overtime_hours  # overworked top performers leave
                attrition_prob += max(0, (3.2 - engagement)) * 0.05
            attrition_prob = np.clip(attrition_prob, 0.02, 0.85)

            attrited = random.random() < attrition_prob
            exit_date = None
            if attrited and tenure_days_cap > 60:
                exit_date = hire_date + timedelta(days=random.randint(60, tenure_days_cap))

            last_promo = None
            if random.random() < 0.4 and tenure_days_cap > 365:
                last_promo = hire_date + timedelta(days=random.randint(180, tenure_days_cap))

            dept_display = random.choice(CASING_VARIANTS[dept])

            row = {
                "employee_id": emp_id,
                "department": dept_display,
                "team": team,
                "job_role": f"{dept} Associate" if performance < 3.7 else f"Senior {dept} Specialist",
                "hire_date": _fmt_date(hire_date),
                "exit_date": _fmt_date(exit_date) if exit_date else "",
                "attrition_flag": "Yes" if exit_date else "No",
                "performance_rating": round(performance, 1),
                "last_promotion_date": _fmt_date(last_promo) if last_promo else "",
                "compensation": compensation,
                "engagement_score": round(engagement, 1),
                "overtime_hours_per_month": round(overtime_hours, 1),
                "manager_id": 9000 + hash((dept, team)) % 40,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # --- inject messiness ---
    # 1. Missing values
    for col, frac in [("performance_rating", 0.03), ("compensation", 0.02), ("engagement_score", 0.04)]:
        idx = df.sample(frac=frac, random_state=seed).index
        df.loc[idx, col] = np.nan

    # 2. Duplicate rows (simulate repeated export of same employee)
    dupes = df.sample(frac=0.02, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    # 3. Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    from config import BRONZE_DIR
    out = generate()
    path = BRONZE_DIR / "hr_attrition_raw.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} rows to {path}")
