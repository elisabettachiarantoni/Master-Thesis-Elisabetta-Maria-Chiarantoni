# -*- coding: utf-8 -*-
"""
OWSA on INMB (CHF per person) at 100% adherence
- Scenarios: Ticino (50 ng), Fribourg 90/10, 80/20, 50/50 (75 ng)
- Varies screening test performance + consult-before-FIT cost (Ticino/Fri)
- Produces combined tornado plot across all scenarios
- Interactive version (plots enabled, aligned with ows_tornado_combined_legend_bl aesthetic)
- Updated to use run_noscreening for baseline costs/QALYs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- direct function imports ----------
from File_0_No_Screening import run_noscreening
from File_1_Ticino import run_ticino
from File_2_Fribourg import run_fribourg

# Set seed for reproducibility
np.random.seed(1)

# Population size (must match simulation files)
population_size = 100000

# ============================================================
# Config
# ============================================================
WTP = 100_000.0  # CHF per QALY

# Parameter ranges (low, high)
RANGES = {
    "FIT sens (adv adenomas)": {
        "50ng": (0.33, 0.47),
        "75ng": (0.25, 0.34),
        "var_name": "fit_sensitivity_polyps",
    },
    "FIT sens (CRC)": {
        "50ng": (0.84, 0.95),
        "75ng": (0.63, 0.92),
        "var_name": "fit_sensitivity_cancer",
    },
    "FIT specificity": {
        "50ng": (0.86, 0.93),
        "75ng": (0.91, 0.95),
        "var_name": "fit_specificity",
    },
    "Colono sens (large adenomas 10–20mm)": {
        "all": (0.70, 0.90),
        "var_name": "colonoscopy_sensitivity_polyps",
    },
    "Colono sens (CRC)": {
        "all": (0.97, 0.995),
        "var_name": "colonoscopy_sensitivity_cancer",
    },
    "Colono specificity": {
        "all": (0.995, 1.0),
        "var_name": "colonoscopy_specificity",
    },
    "Cost of FIT": {
        "all": (46.0 * 0.8, 46.0* 1.2),
        "var_name": "c_fit_kit",
    },
    "Cost of Colonoscopy": {
        "all": (559.30 * 0.8, 559.30 * 1.2),
        "var_name": "c_colon",
    },
}

def _threshold_key(scenario_name: str) -> str:
    return "50ng" if "Ticino" in scenario_name else "75ng"

def _is_ticino(scenario_name: str) -> bool:
    return "Ticino" in scenario_name

# --- Define scenarios as callables with their static kwargs ---
SCENARIOS = {
    "Ticino (50 ng)": {
        "fn": run_ticino,
        "kwargs": {},
        "thr": "50ng",
        "key": "ticino",  # key for extracting results from return dict
    },
    "Fribourg 80/20 (75 ng)": {
        "fn": run_fribourg,
        "kwargs": {"FIT_choice": 0.80, "colon_choice": 0.20},
        "thr": "75ng",
        "key": "fribourg",
    },
    "Fribourg 20/80 (75 ng)": {
        "fn": run_fribourg,
        "kwargs": {"FIT_choice": 0.20, "colon_choice": 0.80},
        "thr": "75ng",
        "key": "fribourg",
    },
    "Fribourg 50/50 (75 ng)": {
        "fn": run_fribourg,
        "kwargs": {"FIT_choice": 0.50, "colon_choice": 0.50},
        "thr": "75ng",
        "key": "fribourg",
    },
}

# ============================================================
# Helpers
# ============================================================
def extract_totals_from_dataframes(result_dict, key_prefix, population_size):
    """
    Extract total cost and QALY from dataframe results.
    Args:
        result_dict: Dictionary containing 'costs' and 'qaly' DataFrames
        key_prefix: Either 'no_screening' or screening scenario name ('ticino', 'fribourg')
        population_size: Size of the population
    Returns:
        Dictionary with cost, qaly, and N
    """
    costs_df = result_dict[key_prefix]["costs"]
    qaly_df = result_dict[key_prefix]["qaly"]
    
    total_cost = costs_df["Cost"].sum()
    total_qaly = qaly_df["Qaly"].sum()
    
    return {
        "cost": total_cost,
        "qaly": total_qaly,
        "N": population_size
    }

def compute_inmb(scr_data, no_scr_data):
    """Compute INMB given screening and no-screening data."""
    dC = (scr_data["cost"] - no_scr_data["cost"]) / scr_data["N"]
    dE = (scr_data["qaly"] - no_scr_data["qaly"]) / scr_data["N"]
    return WTP * dE - dC, dC, dE

# ============================================================
# OWSA driver
# ============================================================
rows = []

for scen_name, meta in SCENARIOS.items():
    fn = meta["fn"]
    base_kwargs = dict(meta.get("kwargs", {}))
    thr = meta["thr"]
    scr_key = meta["key"]

    print(f"\n=== Running base for {scen_name} ===")
    
    # Get no-screening baseline (only needs to run once per scenario)
    no_scr_result = run_noscreening()
    no_scr_data = extract_totals_from_dataframes(no_scr_result, "no_screening", population_size)
    
    # Get screening baseline
    base_scr_result = fn(**base_kwargs)
    base_scr_data = extract_totals_from_dataframes(base_scr_result, scr_key, population_size)
    base_inmb, base_dC, base_dE = compute_inmb(base_scr_data, no_scr_data)

    tests = []

    # FIT-dependent by threshold
    for label in ["FIT sens (adv adenomas)", "FIT sens (CRC)", "FIT specificity"]:
        low, high = RANGES[label][thr]
        tests.append((label, RANGES[label]["var_name"], low, high))

    # Colonoscopy params (all scenarios)
    for label in ["Colono sens (large adenomas 10–20mm)", "Colono sens (CRC)", "Colono specificity", "Cost of FIT", "Cost of Colonoscopy"]:
        low, high = RANGES[label]["all"]
        tests.append((label, RANGES[label]["var_name"], low, high))

    for label, var_name, lo_val, hi_val in tests:
        print(f"  - {label}: low={lo_val}, high={hi_val}")
        
        # Low value
        low_kwargs = dict(base_kwargs)
        low_kwargs[var_name] = lo_val
        low_scr_result = fn(**low_kwargs)
        low_scr_data = extract_totals_from_dataframes(low_scr_result, scr_key, population_size)
        low_inmb, _, _ = compute_inmb(low_scr_data, no_scr_data)
        
        # High value
        high_kwargs = dict(base_kwargs)
        high_kwargs[var_name] = hi_val
        high_scr_result = fn(**high_kwargs)
        high_scr_data = extract_totals_from_dataframes(high_scr_result, scr_key, population_size)
        high_inmb, _, _ = compute_inmb(high_scr_data, no_scr_data)

        rows.append({
            "Scenario": scen_name,
            "Parameter": label,
            "VarName": var_name,
            "Base_INMB": base_inmb,
            "Low_INMB": low_inmb,
            "High_INMB": high_inmb,
        })

# ============================================================
# Results table and tornado plot
# ============================================================
df = pd.DataFrame(rows)
df.to_csv("owsa_inmb_results.csv", index=False)
print("\nSaved: owsa_inmb_results.csv")

# Order parameters by max spread across all scenarios
agg = (df.assign(Spread=lambda d: (d[["Low_INMB","High_INMB"]].max(axis=1)
                                   - d[["Low_INMB","High_INMB"]].min(axis=1)))
         .groupby("Parameter", as_index=False)["Spread"].max()
         .sort_values("Spread", ascending=False))
ordered_params = agg["Parameter"].tolist()

scen_list = list(SCENARIOS.keys())
offsets = np.linspace(-0.3, 0.3, len(scen_list))
colors = plt.cm.Set2.colors

plt.figure(figsize=(12, max(6, 0.6 * len(ordered_params))))
y_base = np.arange(len(ordered_params))[::-1]
handles, labels = [], []

for pi, param in enumerate(ordered_params):
    base_y = y_base[pi]
    if pi == 0:
        plt.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    for si, scen in enumerate(scen_list):
        dsub = df[(df["Parameter"] == param) & (df["Scenario"] == scen)]
        if dsub.empty:
            continue
        lo = float(dsub["Low_INMB"].values[0])
        hi = float(dsub["High_INMB"].values[0])
        a, b = (lo, hi) if lo <= hi else (hi, lo)
        y = base_y + offsets[si]
        h, = plt.plot([a, b], [y, y],
                      lw=8, solid_capstyle="butt",
                      color=colors[si % len(colors)])
        if pi == 0:
            handles.append(h); labels.append(scen)

plt.gca().set_yticks(y_base)
plt.gca().set_yticklabels(ordered_params)
plt.xlabel("INMB (CHF per person) — low → high", fontsize=12)
plt.title("Combined Tornado Plot", fontsize=14)
plt.grid(True, axis="x", linestyle=":", alpha=0.5)

# Legend bottom-left inside the plot
plt.legend(handles, labels, title="Scenario",
           loc="lower left", frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig("owsa_tornado_combined_legend_bl.png", dpi=300)
plt.show()
print("Saved: owsa_tornado_combined_legend_bl.png")