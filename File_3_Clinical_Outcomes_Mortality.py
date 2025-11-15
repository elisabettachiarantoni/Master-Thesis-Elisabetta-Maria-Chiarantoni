# -*- coding: utf-8 -*-
"""

Multi-simulation summary builder.
- File 0: baseline (no screening)
- File 1: Ticino program
- File 2: Fribourg program (parameterized: FIT_choice, colon_choice) allowed {20/80, 50/50, 80/20}
- Produces a combined summary table and a combined Δ-vs-baseline table
- Saves to CSV
"""

import pandas as pd
import numpy as np

# ---------- imports ----------
# File 0: baseline (no screening)
from File_0_No_Screening import run_noscreening

# File 1: Ticino program
from File_1_Ticino import run_ticino

# File 2: Fribourg program (parameterized)
from File_2_Fribourg import run_fribourg

# Set seed for reproducibility
np.random.seed(1)

# ---------- helpers ----------
def pct_reduction(new, ref):
    return 100 * (1 - new / ref) if ref and ref > 0 else None

def dominance_label(delta_cost, delta_qaly):
    if delta_qaly is None or delta_cost is None:
        return "N/A"
    if delta_qaly > 0 and delta_cost < 0:
        return "Dominates"
    if delta_qaly < 0 and delta_cost > 0:
        return "Dominated"
    return None  # fall back to ICER

def _deaths_by_cause(df):
    # robust to missing columns (treat missing as 0)
    e = int(df.get("From Early Cancer", pd.Series([0])).sum())
    l = int(df.get("From Late Cancer", pd.Series([0])).sum())
    o = int(df.get("Other Causes", pd.Series([0])).sum())
    return e, l, o, e + l + o

def _population_from_cohort(cohort_df):
    # sum the first row across all numeric cols except 'Year'
    row0 = cohort_df.iloc[0]
    if "Year" in row0.index:
        row0 = row0.drop("Year")
    return int(pd.to_numeric(row0, errors="coerce").fillna(0).sum())

def _arm_metrics(arm):
    pop = _population_from_cohort(arm["cohort"])
    e, l, o, t = _deaths_by_cause(arm["deaths"])
    crc = e + l
    qpp = arm["qaly"]["Qaly"].sum() / pop
    cpp = arm["costs"]["Cost"].sum() / pop
    return dict(pop=pop, e=e, l=l, o=o, t=t, crc=crc, qpp=qpp, cpp=cpp)

_ALLOWED_FR_CHOICES = {(0.20, 0.80), (0.50, 0.50), (0.80, 0.20)}
def _validate_fribourg_choices(FIT_choice, colon_choice):
    pair = (float(FIT_choice), float(colon_choice))
    if pair not in _ALLOWED_FR_CHOICES:
        allowed = ", ".join([f"{a}/{b}" for (a, b) in sorted(_ALLOWED_FR_CHOICES)])
        raise ValueError(f"Invalid Fribourg split {pair}. Allowed combos are: {allowed}.")
    return pair

# ---------- main builder ----------
def build_multi_sim_tables(sim_specs):
    """
    Build tables using a single common baseline taken from the FIRST spec in sim_specs.
    Only that first spec contributes the 'no_screening' row to the main table.
    All reductions/ICERs are computed vs this common baseline.

    sim_specs: list of dicts with keys:
      - label: str
      - run_func: callable
      - params: dict (optional), passed to run_func(**params)
    """
    all_main_rows, all_delta_rows = [], []

    # --- 1) Load common baseline from the FIRST spec ---
    first_spec = sim_specs[0]
    res0 = _run_spec(first_spec)
    if "no_screening" not in res0:
        raise KeyError(f"{first_spec['label']}: expected key 'no_screening' in results.")
    common_base = res0["no_screening"]
    common_base_m = _arm_metrics(common_base)

    # Add the SINGLE baseline row (from the first spec only)
    all_main_rows.append({
        "Simulation": first_spec["label"],
        "Strategy": "no_screening",
        "Early-cancer deaths": common_base_m["e"],
        "Late-cancer deaths": common_base_m["l"],
        "Early/Late ratio": (round(common_base_m["e"]/common_base_m["l"], 3) if common_base_m["l"]>0 else None),
        "Total CRC deaths": common_base_m["crc"],
        "% reduction CRC deaths vs baseline": None,
        "Other deaths": common_base_m["o"],
        "Total deaths": common_base_m["t"],
        "QALYs/person": round(common_base_m["qpp"], 4),
        "Cost/person (CHF)": round(common_base_m["cpp"], 0),
        "ICER vs baseline (CHF/QALY)": "N/A"
    })

    # --- 2) Loop all sims, but skip adding their 'no_screening' row again ---
    for idx, spec in enumerate(sim_specs):
        results = res0 if idx == 0 else _run_spec(spec)

        for arm_name, arm in results.items():
            # Skip malformed arms
            if not isinstance(arm, dict) or "cohort" not in arm or "deaths" not in arm:
                continue

            # Skip duplicate 'no_screening' rows for non-first specs
            if arm_name == "no_screening" and idx != 0:
                continue

            m = _arm_metrics(arm)

            ratio = round(m["e"] / m["l"], 3) if m["l"] > 0 else None
            crc_red = pct_reduction(m["crc"], common_base_m["crc"]) if arm_name != "no_screening" else None

            d_qaly = m["qpp"] - common_base_m["qpp"]
            d_cost = m["cpp"] - common_base_m["cpp"]
            icer = (d_cost / d_qaly) if (d_qaly is not None and abs(d_qaly) > 0) else None
            dom = dominance_label(d_cost, d_qaly)

            all_main_rows.append({
                "Simulation": spec["label"],
                "Strategy": arm_name,
                "Early-cancer deaths": m["e"],
                "Late-cancer deaths": m["l"],
                "Early/Late ratio": ratio,
                "Total CRC deaths": m["crc"],
                "% reduction CRC deaths vs baseline": (round(crc_red, 1) if crc_red is not None else None),
                "Other deaths": m["o"],
                "Total deaths": m["t"],
                "QALYs/person": round(m["qpp"], 4),
                "Cost/person (CHF)": round(m["cpp"], 0),
                "ICER vs baseline (CHF/QALY)": (
                    "Dominates" if dom == "Dominates"
                    else ("Dominated" if dom == "Dominated"
                          else (round(icer, 0) if icer is not None else "N/A"))
                )
            })

            if arm_name != "no_screening":
                all_delta_rows.append({
                    "Comparison": f"{spec['label']} — {arm_name} vs common baseline",
                    "Δ Early-cancer deaths": m["e"] - common_base_m["e"],
                    "Δ Late-cancer deaths": m["l"] - common_base_m["l"],
                    "Δ Total CRC deaths": m["crc"] - common_base_m["crc"],
                    "Δ Other deaths": m["o"] - common_base_m["o"],
                    "Δ Total deaths": m["t"] - common_base_m["t"],
                    "Δ QALYs/person": round(d_qaly, 4),
                    "Δ Cost/person (CHF)": round(d_cost, 0),
                    "ICER (CHF/QALY)": (
                        "Dominates" if dom == "Dominates"
                        else ("Dominated" if dom == "Dominated"
                              else (round(icer, 0) if icer is not None else "N/A"))
                    )
                })

    main_df = pd.DataFrame(all_main_rows).set_index(["Simulation", "Strategy"])
    delta_df = pd.DataFrame(all_delta_rows).set_index("Comparison")

    main_cols = [
        "Early-cancer deaths", "Late-cancer deaths", "Early/Late ratio",
        "Total CRC deaths", "% reduction CRC deaths vs baseline",
        "Other deaths", "Total deaths",
        "QALYs/person", "Cost/person (CHF)",
        "ICER vs baseline (CHF/QALY)"
    ]
    delta_cols = [
        "Δ Early-cancer deaths", "Δ Late-cancer deaths",
        "Δ Total CRC deaths", "Δ Other deaths", "Δ Total deaths",
        "Δ QALYs/person", "Δ Cost/person (CHF)", "ICER (CHF/QALY)"
    ]

    main_df = main_df[main_cols]
    delta_df = delta_df[delta_cols]
    return main_df, delta_df


def _run_spec(spec):
    """Run a spec entry with optional validation for Fribourg parameters."""
    run_func = spec["run_func"]
    params = dict(spec.get("params", {}))

    # If this is the Fribourg function with FIT/colon split, validate combos.
    if run_func is run_fribourg:
        FIT = params.get("FIT_choice")
        COL = params.get("colon_choice")
        if FIT is None or COL is None:
            raise ValueError("run_fribourg requires params: FIT_choice, colon_choice.")
        _validate_fribourg_choices(FIT, COL)

    return run_func(**params)


# ---------- run as script ----------
if __name__ == "__main__":
    # Spec list:
    #   1) Baseline: File 0 (no screening)
    #   2) Ticino:   File 1
    #   3) Fribourg: File 2 — three allowed splits
    SIM_SPECS = [
        {"label": "Baseline — No screening (File 0)", "run_func": run_noscreening, "params": {}},
        {"label": "Ticino program (File 1)",          "run_func": run_ticino,        "params": {}},
        {"label": "Fribourg 20/80 (File 2)",          "run_func": run_fribourg,      "params": {"FIT_choice": 0.20, "colon_choice": 0.80}},
        {"label": "Fribourg 50/50 (File 2)",          "run_func": run_fribourg,      "params": {"FIT_choice": 0.50, "colon_choice": 0.50}},
        {"label": "Fribourg 80/20 (File 2)",          "run_func": run_fribourg,      "params": {"FIT_choice": 0.80, "colon_choice": 0.20}},
    ]

    main, delta = build_multi_sim_tables(SIM_SPECS)

    print("\n=== Combined Comprehensive Summary Table ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(main.to_string())

    print("\n=== Combined Δ vs Baseline (per simulation) ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(delta.to_string())

    main.to_csv("summary_table_ALL_simulations.csv")
    delta.to_csv("summary_table_ALL_deltas.csv")
    print("\nSaved: summary_table_ALL_simulations.csv, summary_table_ALL_deltas.csv")
