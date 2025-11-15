# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 13:46:20 2025

@author: elisa
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as mtick
import numpy as np

# ---------- Explicit imports of functions ----------
from File_0_No_Screening import run_noscreening       # baseline (no screening)
from File_1_Ticino import run_ticino                    # Ticino program
from File_2_Fribourg import run_fribourg                # Fribourg program with params

# Set seed for reproducibility
np.random.seed(1)

# ---------- Run sims ----------
r_base = run_noscreening()                             # expects dict with 'no_screening'
r_tic  = run_ticino()                                   # expects dict with 'ticino' (and optionally 'no_screening')
r_fri_20_80 = run_fribourg(FIT_choice=0.20, colon_choice=0.80)  # expects dict with 'fribourg'
r_fri_50_50 = run_fribourg(FIT_choice=0.50, colon_choice=0.50)
r_fri_80_20 = run_fribourg(FIT_choice=0.80, colon_choice=0.20)

# Use File 0's "no_screening" as the common baseline
base = r_base["no_screening"]

programs = {
    "Ticino Screening": r_tic["ticino"],
    "Fribourg 20/80":   r_fri_20_80["fribourg"],
    "Fribourg 50/50":   r_fri_50_50["fribourg"],
    "Fribourg 80/20":   r_fri_80_20["fribourg"],
}

def format_cycles(ax=None):
    ax = ax or plt.gca()
    ax.set_xlim(0, 14)  # 15 cycles total
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.grid(True)
    return ax

# -----------------------------
# Deaths over time (stock)
# -----------------------------
plt.figure(figsize=(9,6))
plt.plot(base["cohort"].index, base["cohort"]["Death"], label="No Screening", lw=2)
for label, arm in programs.items():
    plt.plot(arm["cohort"].index, arm["cohort"]["Death"], label=label, lw=2)
plt.title("Deaths Over Time")
plt.xlabel("Cycle (2-year intervals)")
plt.ylabel("People")
plt.legend()
format_cycles()
plt.tight_layout()
plt.show()

# -----------------------------
# Per cycle QALYs
# -----------------------------
plt.figure(figsize=(9,6))
plt.plot(base["qaly"].index, base["qaly"]["Qaly"], label="No Screening", lw=2)
for label, arm in programs.items():
    plt.plot(arm["qaly"].index, arm["qaly"]["Qaly"], label=label, lw=2)

plt.title("QALYs per Cycle")
plt.xlabel("Cycle (2-year intervals)")
plt.ylabel("QALYs (thousands)")

ax = format_cycles()
ax.ticklabel_format(style='plain', axis='y', useOffset=False)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e3:,.0f}k'))

plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Cumulative QALYs
# -----------------------------
plt.figure(figsize=(9,6))
plt.plot(base["qaly"].index, base["qaly"]["Qaly"].cumsum(), label="No Screening", lw=2)
for label, arm in programs.items():
    plt.plot(arm["qaly"].index, arm["qaly"]["Qaly"].cumsum(), label=label, lw=2)

plt.title("Cumulative QALYs over Time")
plt.xlabel("Cycle (2-year intervals)")
plt.ylabel("Cumulative QALYs (thousands)")

ax = format_cycles()
ax.ticklabel_format(style='plain', axis='y', useOffset=False)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e3:,.0f}k'))

plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Per cycle Costs
# -----------------------------
fig, ax = plt.subplots(figsize=(9,6))
ax.plot(base["costs"].index, base["costs"]["Cost"], label="No Screening", lw=2)
for label, arm in programs.items():
    ax.plot(arm["costs"].index, arm["costs"]["Cost"], label=label, lw=2)

ax.set_title("Costs per Cycle")
ax.set_xlabel("Cycle (2-year intervals)")
ax.set_ylabel("Costs (CHF)")
ax.ticklabel_format(style='plain', axis='y', useOffset=False)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:,.0f}M'))
format_cycles(ax)
ax.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Cumulative Costs
# -----------------------------
plt.figure(figsize=(9,6))
plt.plot(base["costs"].index, base["costs"]["Cost"].cumsum(), label="No Screening", lw=2)
for label, arm in programs.items():
    plt.plot(arm["costs"].index, arm["costs"]["Cost"].cumsum(), label=label, lw=2)

plt.title("Cumulative Costs over Time")
plt.xlabel("Cycle (2-year intervals)")
plt.ylabel("Cumulative Costs (Million CHF)")

ax = format_cycles()
ax.ticklabel_format(style='plain', axis='y', useOffset=False)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:,.0f}M'))

plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Early Cancer cases over time
# -----------------------------
plt.figure(figsize=(9,6))
plt.plot(base["cohort"].index, base["cohort"]["Early Cancer"], label="No Screening", lw=2)
for label, arm in programs.items():
    plt.plot(arm["cohort"].index, arm["cohort"]["Early Cancer"], label=label, lw=2)

plt.title("Early Cancer Cases Over Time")
plt.xlabel("Cycle (2-year intervals)")
plt.ylabel("People")
format_cycles()
plt.legend()
plt.tight_layout()
plt.show()


