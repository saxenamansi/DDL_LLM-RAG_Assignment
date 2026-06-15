"""
Evaluator.py
============
Automated accuracy scoring for RAG model outputs against GroundTruth.py.

Q1 — numeric extraction: parse numbers from model answer, compare to ground
     truth values with a relative tolerance (default 2%). Reports per-field
     and per-company accuracy.

Q2 — CCUS presence: check whether each company is identified as having CCUS.
     Binary per company (mentioned / not mentioned).

Q3 — net-zero year: extract year from model answer per company, compare to
     ground truth year.

Usage
-----
    from Evaluator import evaluate_all
    scores = evaluate_all(df_results)   # df_results from run_all_models()
    print(scores)

Or run directly:
    python Evaluator.py <path_to_model_answers_txt>
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from GroundTruth import CCUS_GT, EMISSIONS_GT, NET_ZERO_GT


# ── Q1 helpers ─────────────────────────────────────────────────────────────────

# Ground truth values keyed as (company, calendar_year, field)
# We normalise all values to a common unit for comparison:
#   Scope 1/2/3 → tonnes CO2(e)
#   Energy      → GJ

_UNIT_CONVERSIONS = {
    # JFE: MtCO2e → t,  PJ → GJ
    "JFE Holdings": {
        "scope1": 1e6,   # MtCO2e → tCO2e
        "scope2": 1e6,
        "scope3": 1e3,   # ktCO2e → tCO2e
        "energy": 1e6,   # PJ → GJ  (1 PJ = 1e6 GJ)
    },
    # JSW: TonCO2 already in tonnes, GJ already GJ
    "JSW Steel": {
        "scope1": 1.0,
        "scope2": 1.0,
        "scope3": 1.0,
        "energy": 1.0,
    },
    # POSCO: tCO2e already tonnes, GJ already GJ
    "POSCO": {
        "scope1": 1.0,
        "scope2": 1.0,
        "scope3": 1.0,
        "energy": 1.0,
    },
    # Tata: MtCO2e → t,  TJ → GJ
    "Tata Steel UK": {
        "scope1": 1e6,
        "scope2": 1e6,
        "scope3": 1e6,
        "energy": 1e3,   # TJ → GJ  (1 TJ = 1000 GJ)
    },
}

# Build lookup: (company, calendar_year, field) → normalised value in tonnes/GJ
def _build_gt_lookup() -> Dict[Tuple, float]:
    lookup = {}
    field_map = {
        "JFE Holdings":  {
            "scope1": "scope1_MtCO2e", "scope2": "scope2_MtCO2e",
            "scope3": "scope3_ktCO2e", "energy": "total_energy_PJ",
        },
        "JSW Steel": {
            "scope1": "scope1_TonCO2", "scope2": "scope2_TonCO2",
            "scope3": "scope3_TonCO2", "energy": "total_energy_GJ",
        },
        "POSCO": {
            "scope1": "scope1_tCO2e", "scope2": "scope2_tCO2e",
            "scope3": "scope3_tCO2e", "energy": "total_energy_GJ",
        },
        "Tata Steel UK": {
            "scope1": "scope1_MtCO2e", "scope2": "scope2_MtCO2e",
            "scope3": "scope3_MtCO2e", "energy": "total_energy_TJ",
        },
    }
    for row in EMISSIONS_GT:
        company = row["company"]
        year    = row["calendar_year"]
        conv    = _UNIT_CONVERSIONS[company]
        fmap    = field_map[company]
        for field in ("scope1", "scope2", "scope3", "energy"):
            raw = row.get(fmap[field])
            if raw is not None:
                lookup[(company, year, field)] = float(raw) * conv[field]
    return lookup

GT_LOOKUP = _build_gt_lookup()

COMPANIES = ["JFE Holdings", "JSW Steel", "POSCO", "Tata Steel UK"]
FIELDS    = ["scope1", "scope2", "scope3", "energy"]
YEARS     = [2023, 2024, 2025]


def _extract_numbers(text: str) -> List[float]:
    """
    Extract all numeric values from a string.
    Handles: 1,234,567  /  1.23  /  52,106,566.00  /  Indian lakh 5,31,00,751

    Excludes:
    - Citation bracket numbers like [1], [2], [13]
    - Pure integers <= 10 (scope numbers, list indices, page refs)
    - Years (4-digit 19xx/20xx)
    Returns raw floats — caller decides which to compare.
    """
    # Remove citation brackets [N] first
    cleaned = re.sub(r"\[\d+\]", "", text)
    # Collapse commas used as thousands separators (handles Indian lakh too)
    cleaned = re.sub(r"(\d),(\d)", r"\1\2", cleaned)
    nums = []
    for m in re.findall(r"\b\d+(?:\.\d+)?\b", cleaned):
        val = float(m)
        # Skip small integers (scope indices, list numbers) and years
        if val <= 10:
            continue
        if 1900 <= val <= 2100:
            continue
        nums.append(val)
    return nums


def _find_value_near_keyword(answer: str, keywords: List[str],
                              company: str) -> Optional[float]:
    """
    Look for a number near a keyword in the answer text.
    Searches a window of ~200 chars after each keyword hit.
    """
    answer_lower = answer.lower()
    for kw in keywords:
        idx = answer_lower.find(kw.lower())
        if idx == -1:
            continue
        window = answer[idx: idx + 200]
        nums = _extract_numbers(window)
        if nums:
            return nums[0]
    return None


# Keywords the model might use when reporting each field for each company
_FIELD_KEYWORDS = {
    ("JFE Holdings", "scope1"):  ["scope 1", "scope1"],
    ("JFE Holdings", "scope2"):  ["scope 2", "scope2"],
    ("JFE Holdings", "scope3"):  ["scope 3", "scope3"],
    ("JFE Holdings", "energy"):  ["energy", "pj", "petajoule"],
    ("JSW Steel",    "scope1"):  ["scope 1", "scope1"],
    ("JSW Steel",    "scope2"):  ["scope 2", "scope2"],
    ("JSW Steel",    "scope3"):  ["scope 3", "scope3"],
    ("JSW Steel",    "energy"):  ["energy", "gj", "gigajoule"],
    ("POSCO",        "scope1"):  ["scope 1", "scope1", "direct"],
    ("POSCO",        "scope2"):  ["scope 2", "scope2", "indirect"],
    ("POSCO",        "scope3"):  ["scope 3", "scope3"],
    ("POSCO",        "energy"):  ["energy", "gj", "gigajoule"],
    ("Tata Steel UK","scope1"):  ["scope 1", "scope1"],
    ("Tata Steel UK","scope2"):  ["scope 2", "scope2"],
    ("Tata Steel UK","scope3"):  ["scope 3", "scope3"],
    ("Tata Steel UK","energy"):  ["energy", "tj", "terajoule"],
}


def score_q1(answer: str, tol: float = 0.02) -> Dict:
    """
    Score Q1 answer against ground truth.
    tol = relative tolerance (0.02 = within 2%).

    Returns dict with per-(company, year, field) correctness and summary stats.
    """
    results = {}
    correct = 0
    total   = 0
    skipped = 0   # ground truth is None (year not reported)

    for company in COMPANIES:
        # Find the company section in the answer
        company_idx = answer.lower().find(company.lower())
        if company_idx == -1:
            # try short names
            short = company.split()[0]
            company_idx = answer.lower().find(short.lower())
        company_section = answer[company_idx:company_idx + 2000] if company_idx != -1 else answer

        for year in YEARS:
            for field in FIELDS:
                gt_val = GT_LOOKUP.get((company, year, field))
                if gt_val is None:
                    skipped += 1
                    results[(company, year, field)] = "N/A"
                    continue

                total += 1
                # Find year mention near company section
                year_strs = [str(year), str(year)[-2:], f"FY{year}", f"FY{str(year)[-2:]}"]
                year_idx = -1
                for ys in year_strs:
                    yi = company_section.find(ys)
                    if yi != -1:
                        year_idx = yi
                        break

                search_section = company_section[max(0, year_idx - 50): year_idx + 500] if year_idx != -1 else company_section
                kws = _FIELD_KEYWORDS.get((company, field), [f"scope {field[-1]}"])
                extracted = _find_value_near_keyword(search_section, kws, company)

                if extracted is None:
                    results[(company, year, field)] = "MISSING"
                    continue

                # Normalise extracted value to same unit as GT (tonnes / GJ)
                # The model might report in the original units — try both
                conv = _UNIT_CONVERSIONS[company][field]
                extracted_norm = extracted * conv   # if model reported in original units
                extracted_raw  = extracted          # if model reported in normalised units

                match_norm = abs(extracted_norm - gt_val) / gt_val <= tol
                match_raw  = abs(extracted_raw  - gt_val) / gt_val <= tol

                if match_norm or match_raw:
                    results[(company, year, field)] = "✓"
                    correct += 1
                else:
                    results[(company, year, field)] = f"✗ (got {extracted:.2e}, expected {gt_val:.2e})"

    pct = round(100 * correct / total, 1) if total > 0 else 0.0
    return {
        "detail":   results,
        "correct":  correct,
        "total":    total,
        "skipped":  skipped,
        "accuracy": pct,
    }


# ── Q2 helpers ─────────────────────────────────────────────────────────────────

# Ground truth: all 4 companies have CCUS
Q2_GT = {row["company"]: True for row in CCUS_GT}

# Keywords that indicate a company has CCUS — any hit = identified
_CCUS_KEYWORDS = ["ccus", "ccs", "ccu", "carbon capture", "co2 capture",
                  "carbon utilis", "carbon utiliz", "capture and utilis",
                  "capture and utiliz"]


def score_q2(answer: str) -> Dict:
    """
    Score Q2: for each company, did the model identify them as having CCUS?
    Returns per-company boolean and overall accuracy.
    """
    answer_lower = answer.lower()
    results  = {}
    correct  = 0

    for company in COMPANIES:
        expected = Q2_GT.get(company, False)
        # Find company mention
        company_idx = answer_lower.find(company.lower())
        if company_idx == -1:
            short = company.split()[0].lower()
            company_idx = answer_lower.find(short)

        if company_idx == -1:
            identified = False
        else:
            # Check 500-char window after company mention for CCUS keywords
            window = answer_lower[company_idx: company_idx + 500]
            identified = any(kw in window for kw in _CCUS_KEYWORDS)

        match = (identified == expected)
        results[company] = "✓" if match else "✗ (not identified)"
        if match:
            correct += 1

    pct = round(100 * correct / len(COMPANIES), 1)
    return {
        "detail":   results,
        "correct":  correct,
        "total":    len(COMPANIES),
        "accuracy": pct,
    }


# ── Q3 helpers ─────────────────────────────────────────────────────────────────

Q3_GT = {row["company"]: row["net_zero_year"] for row in NET_ZERO_GT}


def _extract_year(text: str, context_window: int = 300) -> Optional[int]:
    """Extract the first 4-digit year (2030–2060 range) from text."""
    years = re.findall(r"\b(20[3-6]\d)\b", text)
    return int(years[0]) if years else None


def score_q3(answer: str) -> Dict:
    """
    Score Q3: for each company, did the model extract the correct net-zero year?
    """
    answer_lower = answer.lower()
    results  = {}
    correct  = 0

    for company in COMPANIES:
        expected_year = Q3_GT[company]

        company_idx = answer_lower.find(company.lower())
        if company_idx == -1:
            short = company.split()[0].lower()
            company_idx = answer_lower.find(short)

        if company_idx == -1:
            results[company] = f"✗ company not found (expected {expected_year})"
            continue

        window = answer[company_idx: company_idx + 400]
        extracted_year = _extract_year(window)

        if extracted_year == expected_year:
            results[company] = "✓"
            correct += 1
        elif extracted_year is None:
            results[company] = f"✗ no year found (expected {expected_year})"
        else:
            results[company] = f"✗ got {extracted_year} (expected {expected_year})"

    pct = round(100 * correct / len(COMPANIES), 1)
    return {
        "detail":   results,
        "correct":  correct,
        "total":    len(COMPANIES),
        "accuracy": pct,
    }


# ── Main evaluation entry point ─────────────────────────────────────────────────

def evaluate_all(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Run all three scorers over every row in df_results.
    df_results must have columns: model, question_label, answer.

    Returns a summary DataFrame with one row per (model, question).
    """
    rows = []
    for _, row in df_results.iterrows():
        model    = row["model"]
        q_label  = row["question_label"]
        answer   = row["answer"]

        if q_label == "Q1_emissions_energy":
            s = score_q1(answer)
            detail_str = "; ".join(
                f"{'/'.join(str(x) for x in k)}: {v}"
                for k, v in s["detail"].items() if v not in ("✓", "N/A")
            )
        elif q_label == "Q2_ccus":
            s = score_q2(answer)
            detail_str = "; ".join(f"{k}: {v}" for k, v in s["detail"].items() if v != "✓")
        elif q_label == "Q3_net_zero":
            s = score_q3(answer)
            detail_str = "; ".join(f"{k}: {v}" for k, v in s["detail"].items() if v != "✓")
        else:
            continue

        rows.append({
            "model":          model,
            "question":       q_label,
            "correct":        s["correct"],
            "total":          s["total"],
            "accuracy_pct":   s["accuracy"],
            "failures":       detail_str or "none",
        })

    return pd.DataFrame(rows)


def print_report(scores: pd.DataFrame) -> None:
    """Pretty-print the evaluation report."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    for model, grp in scores.groupby("model"):
        print(f"\n── {model} " + "─" * 40)
        for _, row in grp.iterrows():
            print(f"  {row['question']:<25} {row['accuracy_pct']:>5.1f}%  "
                  f"({row['correct']}/{row['total']})")
            if row["failures"] != "none":
                # Print failures indented, wrapped
                for failure in row["failures"].split("; "):
                    print(f"      ✗ {failure}")

    print("\n── Overall by model " + "─" * 40)
    overall = (scores.groupby("model")
               .apply(lambda g: round(100 * g["correct"].sum() / g["total"].sum(), 1))
               .reset_index(name="overall_pct"))
    print(overall.to_string(index=False))
    print()


# ── CLI usage ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick CLI check against a single model answer .txt file.
    Usage: python Evaluator.py mistral_7b_answers.txt
    Parses the Q1/Q2/Q3 answer blocks from the txt format and scores each.
    """
    if len(sys.argv) < 2:
        print("Usage: python Evaluator.py <model_answers.txt>")
        sys.exit(1)

    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")

    # Parse answer blocks
    q_blocks = {}
    for q_label in ("Q1_emissions_energy", "Q2_ccus", "Q3_net_zero"):
        pattern = rf"── {q_label}.*?Answer:\n(.*?)(?=──|$)"
        m = re.search(pattern, text, re.DOTALL)
        if m:
            q_blocks[q_label] = m.group(1).strip()

    model_name = path.stem
    print(f"\nScoring: {model_name}")
    print("=" * 50)

    scorers = {
        "Q1_emissions_energy": score_q1,
        "Q2_ccus":             score_q2,
        "Q3_net_zero":         score_q3,
    }

    total_correct = 0
    total_items   = 0

    for q_label, scorer in scorers.items():
        if q_label not in q_blocks:
            print(f"  {q_label}: NOT FOUND IN FILE")
            continue
        s = scorer(q_blocks[q_label])
        print(f"\n{q_label}: {s['accuracy']}% ({s['correct']}/{s['total']})")
        for k, v in s["detail"].items():
            key_str = "/".join(str(x) for x in k) if isinstance(k, tuple) else k
            print(f"  {key_str:<40} {v}")
        total_correct += s["correct"]
        total_items   += s["total"]

    overall = round(100 * total_correct / total_items, 1) if total_items else 0
    print(f"\nOverall: {overall}% ({total_correct}/{total_items})")