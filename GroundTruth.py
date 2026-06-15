"""
GroundTruth.py
==============
Pre-extracted ground truth answers for all three assignment questions.
Values taken line-by-line from the source PDFs — these are the reference
answers the RAG system should reproduce.

Assignment asks for 2023, 2024, and 2025.
Availability by company:
  JFE Holdings  — FY2023 (Apr22-Mar23), FY2024 (Apr23-Mar24) available.
                  FY2025 (Apr24-Mar25) NOT yet reported (report covers to FY2024).
  JSW Steel     — FY2023-24, FY2024-25 available. No 2025 calendar year data.
  POSCO         — 2023, 2024 (calendar year) available. 2025 not yet reported.
  Tata Steel UK — FY23, FY24, FY25 (Apr24-Mar25) all available.

Page numbers are PDF-absolute (pdfplumber 1-based).
JFE note: printed footer page = PDF page − 2.

Run directly to print and save all tables:
    python GroundTruth.py
"""

import pandas as pd


# ── Q1: Emissions & Energy ────────────────────────────────────────────────────

EMISSIONS_GT = [
    # ── JFE Holdings ──────────────────────────────────────────────────────────
    # FY = April–March. All-group figures.
    # Scope 1+2 combined table: PDF p.259 | Scope 2 alone: PDF p.260
    # Scope 3: PDF p.261 (unit: ktCO2e, not MtCO2e — ~22 Mt when converted)
    # Energy: PDF p.263
    {
        "company": "JFE Holdings", "reported_year": "FY2023", "calendar_year": 2023,
        "scope1_MtCO2e": 47.4,   "scope1_page": 259,
        "scope2_MtCO2e": 7.2,    "scope2_page": 260,
        "scope3_ktCO2e": 22701,  "scope3_page": 261,
        "total_energy_PJ": 612,  "energy_page": 263,
    },
    {
        "company": "JFE Holdings", "reported_year": "FY2024", "calendar_year": 2024,
        "scope1_MtCO2e": 44.0,   "scope1_page": 259,
        "scope2_MtCO2e": 7.0,    "scope2_page": 260,
        "scope3_ktCO2e": 21894,  "scope3_page": 261,
        "total_energy_PJ": 567,  "energy_page": 263,
    },
    {
        "company": "JFE Holdings", "reported_year": "FY2025", "calendar_year": 2025,
        "scope1_MtCO2e": None,   "scope1_page": None,
        "scope2_MtCO2e": None,   "scope2_page": None,
        "scope3_ktCO2e": None,   "scope3_page": None,
        "total_energy_PJ": None, "energy_page": None,
        # FY2025 ends March 2026 — not yet reported at time of this report.
    },

    # ── JSW Steel ─────────────────────────────────────────────────────────────
    # FY = April–March. Standalone basis. Indian lakh-crore notation converted.
    # Energy: PDF p.13 | Scope 1+2: PDF p.14 | Scope 3: PDF p.15
    # FY2023-24 ≈ calendar 2023 | FY2024-25 ≈ calendar 2024
    # 2025 calendar year = FY2025-26, not yet reported.
    {
        "company": "JSW Steel", "reported_year": "FY2023-24", "calendar_year": 2023,
        "scope1_TonCO2": 52_106_566,       "scope1_page": 14,
        "scope2_TonCO2": 1_061_079,        "scope2_page": 14,
        "scope3_TonCO2": 6_967_897,        "scope3_page": 15,
        "total_energy_GJ": 517_690_735,    "energy_page": 13,
    },
    {
        "company": "JSW Steel", "reported_year": "FY2024-25", "calendar_year": 2024,
        "scope1_TonCO2": 53_100_751.63,    "scope1_page": 14,
        "scope2_TonCO2": 1_653_056.65,     "scope2_page": 14,
        "scope3_TonCO2": 8_693_479,        "scope3_page": 15,
        "total_energy_GJ": 529_209_356.78, "energy_page": 13,
    },
    {
        "company": "JSW Steel", "reported_year": "FY2025-26", "calendar_year": 2025,
        "scope1_TonCO2": None,    "scope1_page": None,
        "scope2_TonCO2": None,    "scope2_page": None,
        "scope3_TonCO2": None,    "scope3_page": None,
        "total_energy_GJ": None,  "energy_page": None,
        # FY2025-26 ends March 2026 — not yet reported.
    },

    # ── POSCO ─────────────────────────────────────────────────────────────────
    # Calendar year. Domestic integrated mills (Pohang + Gwangyang).
    # Scope 3 boundary: 2023 = Pohang+Gwangyang only; 2024 = all sites.
    # Scope 1+2+3: PDF p.139 | Energy breakdown: PDF p.154
    {
        "company": "POSCO", "reported_year": "2023", "calendar_year": 2023,
        "scope1_tCO2e": 70_588_012,        "scope1_page": 139,
        "scope2_tCO2e": 1_383_895,         "scope2_page": 139,
        "scope3_tCO2e": 7_419_787,         "scope3_page": 139,
        "total_energy_GJ": 354_002_733,    "energy_page": 154,
    },
    {
        "company": "POSCO", "reported_year": "2024", "calendar_year": 2024,
        "scope1_tCO2e": 69_665_353,        "scope1_page": 139,
        "scope2_tCO2e": 1_399_825,         "scope2_page": 139,
        "scope3_tCO2e": 7_216_788,         "scope3_page": 139,
        "total_energy_GJ": 359_242_804,    "energy_page": 154,
    },
    {
        "company": "POSCO", "reported_year": "2025", "calendar_year": 2025,
        "scope1_tCO2e": None,     "scope1_page": None,
        "scope2_tCO2e": None,     "scope2_page": None,
        "scope3_tCO2e": None,     "scope3_page": None,
        "total_energy_GJ": None,  "energy_page": None,
        # 2025 calendar year data not yet published.
    },

    # ── Tata Steel UK ─────────────────────────────────────────────────────────
    # FY = April–March. All fields on PDF p.52.
    # FY25 Scope 1 drop: BF/BOF closure at Port Talbot (July 2024).
    # FY25 Scope 3 rise: EAF import-and-reroll shifts emissions upstream.
    # FY23 ≈ calendar 2023 | FY24 ≈ calendar 2024 | FY25 ≈ calendar 2025
    {
        "company": "Tata Steel UK", "reported_year": "FY23", "calendar_year": 2023,
        "scope1_MtCO2e": 6.1,  "scope1_page": 52,
        "scope2_MtCO2e": 0.1,  "scope2_page": 52,
        "scope3_MtCO2e": 1.7,  "scope3_page": 52,
        "total_energy_TJ": 71_848, "energy_page": 52,
    },
    {
        "company": "Tata Steel UK", "reported_year": "FY24", "calendar_year": 2024,
        "scope1_MtCO2e": 5.9,  "scope1_page": 52,
        "scope2_MtCO2e": 0.1,  "scope2_page": 52,
        "scope3_MtCO2e": 1.8,  "scope3_page": 52,
        "total_energy_TJ": 71_234, "energy_page": 52,
    },
    {
        "company": "Tata Steel UK", "reported_year": "FY25", "calendar_year": 2025,
        "scope1_MtCO2e": 2.4,  "scope1_page": 52,
        "scope2_MtCO2e": 0.1,  "scope2_page": 52,
        "scope3_MtCO2e": 4.4,  "scope3_page": 52,
        "total_energy_TJ": 35_174, "energy_page": 52,
    },
]


# ── Q2: CCUS Technology ────────────────────────────────────────────────────────

CCUS_GT = [
    {
        "company": "JFE Holdings",
        "technology": "CR Blast Furnace + CCU (Carbon-Recycling Blast Furnace)",
        "type": "CCU",
        "status": "under development / demonstration",
        "description": (
            "JFE is developing a carbon-recycling blast furnace (CR-BF) combined with CCU, "
            "which captures CO2 from the blast furnace, reuses it as a reducing agent within the "
            "furnace, and converts remaining CO2 into chemicals such as methanol. "
            "Funded under NEDO's Green Innovation Fund (GREINS) project."
        ),
        "source": "JFE Holdings_Sustainability_2025.pdf",
        "source_pages": "pp.68-72, 98, 119",
    },
    {
        "company": "JSW Steel",
        "technology": "CO2 Capture & Utilisation Unit (CCU) — 100 TPD",
        "type": "CCU",
        "status": "operational",
        "description": (
            "JSW Steel is operating a 100 tonnes-per-day CCU facility where CO2 is captured "
            "from steelmaking off-gases, purified, and sold to the food & beverage industry. "
            "Early-stage commercial deployment with plans to scale up CCUS in conjunction "
            "with BF-BOF operations."
        ),
        "source": "JSW Steel_Sustainability_2025.pdf",
        "source_pages": "p.14",
    },
    {
        "company": "POSCO",
        "technology": "CO2 injection into coke ovens (CCU demonstration)",
        "type": "CCU",
        "status": "piloted (demonstrated at Pohang works, Jan 2024)",
        "description": (
            "POSCO, in collaboration with RIST, demonstrated a process where captured CO2 from "
            "steelmaking is injected into coke ovens, boosting COG (Coke Oven Gas) calorific value "
            "by ~7%. Demonstrated at Pohang Steelworks in January 2024. "
            "Selected as one of the Ministry of Trade, Industry and Energy's Top 10 R&D technologies of 2024."
        ),
        "source": "POSCO_Sustainability_2024.pdf",
        "source_pages": "p.13",
    },
    {
        "company": "Tata Steel UK",
        "technology": "Flue2Chem / COZMOS / SUSTAIN (carbon capture and utilisation research)",
        "type": "CCU",
        "status": "research / collaborative pilot",
        "description": (
            "Tata Steel UK participated in the Flue2Chem (InnovateUK) and COZMOS (EU Horizon) "
            "programmes, collaborating with academic and supply-chain partners to demonstrate "
            "carbon capture and utilisation pathways. Additionally, the SUSTAIN programme "
            "(£35m partnership with Swansea, Sheffield and Warwick Universities) covers "
            "sustainable steelmaking technologies including CCS/CCU."
        ),
        "source": "Tata Steel_Sustainability_2025.pdf",
        "source_pages": "pp.12, 51",
    },
]


# ── Q3: Net-Zero Targets ───────────────────────────────────────────────────────

NET_ZERO_GT = [
    {
        "company": "JFE Holdings",
        "net_zero_year": 2050,
        "commitment_label": "Carbon Neutral by 2050",
        "interim_2030_target": "≥30% reduction in GHG emissions by FY2030 vs FY2013; CO2 intensity ≤2.0 t-CO2/t-steel by 2030",
        "scope_coverage": "Scope 1 + 2 (JFE Steel); Scope 1+2+3 across JFE Group long-term",
        "validation_body": "JFE Group Environmental Vision for 2050 (self-declared); JGreeX certified by ClassNK",
        "source_pages": "pp.3-5, 68, 99, 115-116",
        "source": "JFE Holdings_Sustainability_2025.pdf",
    },
    {
        "company": "JSW Steel",
        "net_zero_year": 2050,
        "commitment_label": "Net Neutral by 2050",
        "interim_2030_target": "1.95 tCO2/tcs by 2030 (42% reduction from 2005 baseline); aligned with IEA Sustainable Development Scenario",
        "scope_coverage": "Scope 1 + 2 (standalone basis)",
        "validation_body": "Self-declared; Bureau Veritas assurance on BRSR Core KPIs",
        "source_pages": "pp.4-5, 14",
        "source": "JSW Steel_Sustainability_2025.pdf",
    },
    {
        "company": "POSCO",
        "net_zero_year": 2050,
        "commitment_label": "Carbon Neutral by 2050",
        "interim_2030_target": "HyREX demo-plant operational verification by 2030; EAF expansion at Gwangyang by 2026 targeting up to 75% carbon reduction vs BF route",
        "scope_coverage": "Scope 1 + 2 (primary); Scope 3 expanding",
        "validation_body": "Self-declared; GHG verified by independent third party (Scope 1+2)",
        "source_pages": "pp.13-15, 19-20",
        "source": "POSCO_Sustainability_2024.pdf",
    },
    {
        "company": "Tata Steel UK",
        "net_zero_year": 2045,
        "commitment_label": "reducing CO2 emissions to reach net zero by 2045",
        "interim_2030_target": "≥90% reduction in Scope 1 CO2 emissions by 2030 vs 2018 at Port Talbot (via EAF transition); previously stated 30% reduction by 2030",
        "scope_coverage": "Scope 1 (primary); Scope 1+2+3 for full value chain long-term",
        "validation_body": "worldsteel Sustainability Champion (8 consecutive years); ResponsibleSteel member; SBTi referenced",
        "source_pages": "pp.3, 13, 28, 44, 48",
        "source": "Tata Steel_Sustainability_2025.pdf",
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_dataframes():
    """Return all three ground truth tables as DataFrames."""
    return (
        pd.DataFrame(EMISSIONS_GT),
        pd.DataFrame(CCUS_GT),
        pd.DataFrame(NET_ZERO_GT),
    )


def save_all(out_dir="."):
    """Save all ground truth tables as CSV and JSON."""
    from pathlib import Path
    out = Path(out_dir)

    df_emissions, df_ccus, df_nz = get_dataframes()

    for df, stem in [
        (df_emissions, "ground_truth_emissions"),
        (df_ccus,      "ccus_gt"),
        (df_nz,        "net_zero_gt"),
    ]:
        df.to_csv(out / f"{stem}.csv", index=False)
        df.to_json(out / f"{stem}.json", orient="records", indent=2)
        print(f"Saved {stem}.csv / .json")


if __name__ == "__main__":
    df_emissions, df_ccus, df_nz = get_dataframes()

    print("=" * 60)
    print("Q1 — Emissions & Energy (2023, 2024, 2025)")
    print("=" * 60)
    print(df_emissions.to_string(index=False))

    print("\n" + "=" * 60)
    print("Q2 — CCUS Technology")
    print("=" * 60)
    print(df_ccus[["company", "technology", "status", "source_pages"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("Q3 — Net-Zero Targets")
    print("=" * 60)
    print(df_nz[["company", "net_zero_year", "commitment_label", "interim_2030_target"]].to_string(index=False))

    save_all()