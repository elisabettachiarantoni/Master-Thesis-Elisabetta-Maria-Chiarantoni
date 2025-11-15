# -*- coding: utf-8 -*-
"""
Unified CRC incidence & detection summary table.

Now uses:
- run_noscreening() from File_0_No_Screening
- run_ticino() from File_1_Ticino
- run_fribourg(FIT_choice, colon_choice) from File_2_Fribourg

Computes stage shares, detection rates, and CRC prevented vs. no screening.
"""

import pandas as pd
from File_0_No_Screening import run_noscreening
from File_1_Ticino import run_ticino
from File_2_Fribourg import run_fribourg


# ---------- helpers ----------
def _safe_pct(numer, denom):
    return 100.0 * numer / denom if denom and denom > 0 else 0.0


def _extract_incidence_block(results, *, use_no_screening: bool, pretty_name: str):
    """
    Given a simulation results dict, return a row dict with incidence/detection.
    - If use_no_screening=True, pull the 'no_screening' arm.
    - Else, pull the screening arm.
    """
    arm_key = "no_screening" if use_no_screening else [k for k in results.keys() if k != "no_screening"][0]
    inc = results[arm_key]["incidence"]

    total_crc = inc["ever_early"] + inc["ever_late"]
    row = {
        "Early incidence (ever)": inc["ever_early"],
        "Late incidence (ever)": inc["ever_late"],
        "Total CRC incidence": total_crc,
        "Early detected": inc["detected_early"],
        "Late detected": inc["detected_late"],
        "Detected before late stage": inc["detected_early"],
    }
    return pretty_name, row


# ---------- main ----------
def compute_crc_incidence_all():
    rows = {}

    # 1) No Screening
    res0 = run_noscreening()
    name, row = _extract_incidence_block(res0, use_no_screening=True, pretty_name="No Screening")
    rows[name] = row

    # 2) Ticino (screening arm)
    res_ticino = run_ticino()
    name, row = _extract_incidence_block(res_ticino, use_no_screening=False, pretty_name="Ticino FIT")
    rows[name] = row

    # 3) Fribourg (three splits)
    fribourg_scenarios = [
        (0.20, 0.80, "Fribourg FIT (20/80)"),
        (0.50, 0.50, "Fribourg FIT (50/50)"),
        (0.80, 0.20, "Fribourg FIT (80/20)"),
    ]

    for fit, colon, label in fribourg_scenarios:
        try:
            res_fri = run_fribourg(FIT_choice=fit, colon_choice=colon)
            name, row = _extract_incidence_block(res_fri, use_no_screening=False, pretty_name=label)
            rows[name] = row
        except Exception as e:
            rows[label] = {col: 0 for col in [
                "Early incidence (ever)", "Late incidence (ever)", "Total CRC incidence",
                "Early detected", "Late detected", "Detected before late stage"
            ]}
            print(f"[WARN] Skipped Fribourg {label}: {e}")

    # --- Build DataFrame ---
    df = pd.DataFrame.from_dict(rows, orient="index")

    # --- Compute metrics ---
    df["Early % of total CRC"] = [_safe_pct(e, t) for e, t in zip(df["Early incidence (ever)"], df["Total CRC incidence"])]
    df["Late % of total CRC"] = [_safe_pct(l, t) for l, t in zip(df["Late incidence (ever)"], df["Total CRC incidence"])]
    df["Detected before late stage (%)"] = [_safe_pct(d, t) for d, t in zip(df["Detected before late stage"], df["Total CRC incidence"])]
    df["Early detection rate (% of early)"] = [_safe_pct(d, e) for d, e in zip(df["Early detected"], df["Early incidence (ever)"])]

    # --- CRC prevented vs. No Screening (%)
    baseline = df.loc["No Screening", "Total CRC incidence"]
    df["CRC prevented vs No Screening (%)"] = [
        0.0 if name == "No Screening" else _safe_pct(baseline - val, baseline)
        for name, val in zip(df.index, df["Total CRC incidence"])
    ]

    # --- Order columns ---
    cols = [
        "Total CRC incidence",
        "Early incidence (ever)",
        "Late incidence (ever)",
        "Early % of total CRC",
        "Late % of total CRC",
        "Early detected",
        "Detected before late stage (%)",
        "Early detection rate (% of early)",
        "CRC prevented vs No Screening (%)",
    ]
    df = df[cols].round(1)

    return df


if __name__ == "__main__":
    df = compute_crc_incidence_all()
    print("\n" + "=" * 100)
    print("CRC Incidence & Detection Summary (All Scenarios)")
    print("=" * 100)
    print(df.to_string())
    df.to_csv("crc_incidence_summary_all.csv")
    print("Saved: crc_incidence_summary_all.csv")
