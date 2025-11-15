"""
Colorectal screening microsimulation with corrected fixed/conditional costs
- Two-year cycles, 30 years total, 3% discounting (half-cycle)
- Baseline (no screening) and four screening programs:
    * Ticino (FIT-only 100/0)
    * Fribourg 20/80 (FIT/Colonoscopy)
    * Fribourg 80/20 (FIT/Colonoscopy)
    * Fribourg 50/50 (FIT/Colonoscopy)
- Uses standard NumPy RNG; each arm draws independently (no CRN, no deterministic utilities)
"""

import numpy as np
import matplotlib.pyplot as plt
from math import log, exp, gamma

# ------------------------------------------------------------
# Global config
# ------------------------------------------------------------
years_per_cycle = 2
n_cycles = 15
population_size = 100000

disc_rate = 0.03
WTP = 100_000.0  # CHF per QALY
adh_points = np.arange(0.0, 1.1, 0.1)
adh_pct = (adh_points * 100).astype(int)

# ------------------------------------------------------------
# Fixed costs / prices (CHF)
# ------------------------------------------------------------
c_fit_kit = 46.0
c_colon = 559.30
c_consult_positive_fit = 24.0
c_prep = 50.0
c_complication = 8320.0
c_polypectomy = 91.75
c_adm_doctor = 25.80
c_EC_first = 28197.0
c_EC_subseq = 3202.0
c_LC_first = 37685.0
c_LC_subseq = 4275.0
c_admin = 4.85  # remains fixed per cycle per eligible person
c_consult_before_FIT_Tic = 70.70   # charged only when screening actually occurs
c_consult_before_FIT_Fri = 94.50   # charged only when screening actually occurs

# ------------------------------------------------------------
# Epidemiology / model inputs
# ------------------------------------------------------------
perc_polpys = 11.8
perc_EC = 0.04

population_polyps = int((population_size / 100) * perc_polpys)
population_EC = int((population_size / 100) * perc_EC)
population_normal = population_size - (population_polyps + population_EC)

complication_prob = 0.002

# Recurrence / surveillance
rec_polyps = 0.39  # 3 years
rec_EC_1 = 0.015
rec_EC_3 = 0.09
rec_EC_5 = 0.11
rec_LC_1 = 0.085
rec_LC_3 = 0.28
rec_LC_5 = 0.30

# Survival (5y)
survival_EC = 0.93
survival_LC = 0.13

# Spontaneous clinical discovery (no screening)
discover_EC = 0.21
discover_LC = 0.28

# Weibull for dynamic Early->Late progression
CYCLE_MONTHS = 24.0
DT_v = 26.0
k_doublings = 2.6
T_prog_mean = k_doublings * DT_v  # 67.6 months
p0 = 0.20                         # baseline E->L over first 24 months
p_death = 0.20

def _mean_given_alpha(alpha, CYCLE_MONTHS, p0):
    """
    For a proposed shape 'alpha', compute:
      - lambda_ so that P(progression in the first cycle) = p0
      - the implied mean time to progression E[T]
    Returns (mean_time, lambda_)
    """
    lam = CYCLE_MONTHS / (-log(1 - p0))**(1/alpha)
    return lam * gamma(1 + 1/alpha), lam

# Solve alpha for mean progression time and first-cycle p0
lo, hi = 0.6, 3.0
for _ in range(60):
    mid = (lo + hi) / 2
    mean_mid, lam_mid = _mean_given_alpha(mid, CYCLE_MONTHS, p0)
    if mean_mid > T_prog_mean:
        lo = mid
    else:
        hi = mid
alpha = (lo + hi) / 2
mean_alpha, lambda_ = _mean_given_alpha(alpha, CYCLE_MONTHS, p0)

def p_E2L_interval(m_cycles_since_first_miss: int, CYCLE_MONTHS, p_death=0.20) -> float:
    """Conditional prob of progressing EARLY->LATE in the *next* cycle,
    given m full cycles have already elapsed since the FIRST false negative."""
    t0 = m_cycles_since_first_miss * CYCLE_MONTHS
    t1 = (m_cycles_since_first_miss + 1) * CYCLE_MONTHS
    H0 = (t0 / lambda_)**alpha
    H1 = (t1 / lambda_)**alpha
    p_cond = 1.0 - exp(-(H1 - H0))
    return min(max(0.0, p_cond), 1.0 - p_death)

def disc_factor(cycle, years_per_cycle, r):
    return 1.0 / ((1.0 + r) ** (cycle * years_per_cycle))

# ------------------------------------------------------------
# State definitions and utilities
# ------------------------------------------------------------
states_base = ['Normal', 'Polyps', 'Early Cancer', 'Late Cancer', 'Death', 'Cancer Survivors']
NORMAL, POLYPS, EARLY, LATE, DEATH, CANCER_SURVIVORS = range(len(states_base))

states_scr = ['Normal', 'Polyps', 'Early Cancer', 'Late Cancer', 'Death',
              'Cancer Survivors', 'Outside of Cycle']
S_NORMAL, S_POLYPS, S_EARLY, S_LATE, S_DEATH, S_CANCER_SURVIVORS, S_OUTSIDE = range(len(states_scr))

# Utilities (QALY weights)
utility = np.zeros(len(states_scr))
utility[S_NORMAL] = 1.0
utility[S_POLYPS] = 0.88
utility[S_EARLY] = 0.74
utility[S_LATE] = 0.40
utility[S_DEATH] = 0.0
utility[S_CANCER_SURVIVORS] = 0.82

# Transition matrices
T_base = np.zeros((len(states_base), len(states_base)))
T_base[NORMAL] = [0.864, 0.13, 0.0, 0.0, 0.006, 0.0]
T_base[POLYPS] = [0.0, 0.824, 0.17, 0.0, 0.006, 0.0]
T_base[EARLY]  = [0.0, 0.0, 0.60, 0.20, 0.20, 0.0]
T_base[LATE]   = [0.0, 0.0, 0.0, 0.65, 0.35, 0.0]
T_base[DEATH]  = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
T_base[CANCER_SURVIVORS] = [0.0, 0.0, 0.0, 0.0, 0.006, 0.994]

T_scr = np.zeros((len(states_scr), len(states_scr)))
T_scr[S_NORMAL] = [0.864, 0.13, 0.0, 0.0, 0.006, 0.0, 0.0]
T_scr[S_POLYPS] = [0.0, 0.824, 0.17, 0.0, 0.006, 0.0, 0.0]
T_scr[S_EARLY]  = [0.0, 0.0, 0.60, 0.20, 0.20, 0.0, 0.0]
T_scr[S_LATE]   = [0.0, 0.0, 0.0, 0.65, 0.35, 0.0, 0.0]
T_scr[S_DEATH]  = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
T_scr[S_CANCER_SURVIVORS] = [0.0, 0.0, 0.0, 0.0, 0.006, 0.994, 0.0]
T_scr[S_OUTSIDE] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # absorbing

# Test characteristics (use Ticino parameters)
fit_sensitivity_polyps = 0.40
fit_sensitivity_cancer = 0.91
fit_specificity = 0.90
colonoscopy_sensitivity_polyps = 0.80
colonoscopy_sensitivity_cancer = 0.98
colonoscopy_specificity = 1.0

# Program choices
FIT_choice_Tic = 1.00  # Ticino: FIT-only
colon_choice_Tic = 0.00

# Fribourg splits
FIT_choice_Fri_20 = 0.20
colon_choice_Fri_80 = 0.80

FIT_choice_Fri_80 = 0.80
colon_choice_Fri_20 = 0.20

FIT_choice_Fri_50 = 0.50
colon_choice_Fri_50 = 0.50

# ------------------------------------------------------------
# Initialization helpers
# ------------------------------------------------------------
def initial_states(rng: np.random.Generator):
    base = np.array([0]*population_normal + [1]*population_polyps + [2]*population_EC, dtype=np.int8)
    # Random permutation per arm/run (no shared seeds, no CRN)
    order = rng.permutation(population_size)
    return base[order]

# ------------------------------------------------------------
# Baseline simulation (no screening)
# ------------------------------------------------------------
def run_baseline():
    rng = np.random.default_rng()  # independent stream per call
    individual_states = initial_states(rng)

    surveillance_due = np.full(population_size, np.inf)
    surgery_cycle = np.full(population_size, -1.0)

    miss_active = np.zeros(population_size, dtype=bool)
    miss_start_cycle = np.full(population_size, -1, dtype=int)

    total_cost = 0.0
    total_qaly = 0.0

    for cycle in range(n_cycles):
        cycle_cost = 0.0
        cycle_qaly = 0.0

        for i in range(population_size):
            s = individual_states[i]

            if not miss_active[i]:
                miss_active[i] = True
                miss_start_cycle[i] = cycle

            # Natural history with dynamic E->L
            row = T_base.copy()
            if s == EARLY and miss_active[i]:
                m = cycle - miss_start_cycle[i]
                p_prog = p_E2L_interval(m, CYCLE_MONTHS, p_death)
                stay = max(0.0, 1.0 - p_prog - p_death)
                row[EARLY][EARLY] = stay
                row[EARLY][LATE] = p_prog

            # draw transition
            u = rng.random()
            cumsum = np.cumsum(row[s])
            new_state = int(np.searchsorted(cumsum, u, side="right"))
            if new_state != s:
                miss_active[i] = False
                miss_start_cycle[i] = -1
            s = new_state

            # Symptomatic detection + treatment
            if s == EARLY:
                for half in range(2):
                    t = cycle + 0.5*half
                    if surgery_cycle[i] != -1:
                        alive_or_past5 = (rng.random() < survival_EC and (t - surgery_cycle[i]) < 5) or ((t - surgery_cycle[i]) >= 5)
                        if alive_or_past5:
                            if t == surveillance_due[i]:
                                cycle_cost += (c_colon + c_prep + c_adm_doctor)
                                if rng.random() < complication_prob:
                                    cycle_cost += c_complication
                                dt = t - surgery_cycle[i]
                                if dt == 0.5:
                                    if rng.random() <= rec_EC_1:
                                        cycle_cost += c_EC_first
                                        surveillance_due[i] = t + 0.5
                                        surgery_cycle[i] = t
                                        cycle_qaly += utility[S_EARLY]
                                    else:
                                        cycle_cost += c_EC_subseq
                                        surveillance_due[i] = t + 1.5
                                        cycle_qaly += utility[S_EARLY]
                                elif dt == 2.0:
                                    if rng.random() <= rec_EC_3:
                                        cycle_cost += c_EC_first
                                        surveillance_due[i] = t + 0.5
                                        surgery_cycle[i] = t
                                        cycle_qaly += utility[S_EARLY]
                                    else:
                                        surveillance_due[i] = t + 2.5
                                        cycle_qaly += utility[S_EARLY]
                                elif dt == 4.5:
                                    if rng.random() <= rec_EC_5:
                                        cycle_cost += c_EC_first
                                        surveillance_due[i] = t + 0.5
                                        surgery_cycle[i] = t
                                        cycle_qaly += utility[S_EARLY]
                                    else:
                                        s = S_CANCER_SURVIVORS
                                        surveillance_due[i] = np.inf
                                        surgery_cycle[i] = -1
                                        cycle_qaly += utility[S_CANCER_SURVIVORS] * (2.0 if half == 0 else 1.0)
                            else:
                                cycle_qaly += utility[S_EARLY]
                        else:
                            s = S_DEATH
                            surveillance_due[i] = np.inf
                            surgery_cycle[i] = -1

                    elif rng.random() < discover_EC:
                        cycle_cost += (c_colon + c_prep + c_adm_doctor)
                        if rng.random() < complication_prob:
                            cycle_cost += c_complication
                        surgery_cycle[i] = cycle
                        surveillance_due[i] = cycle + 0.5
                        cycle_cost += c_EC_first
                    cycle_qaly += utility[S_EARLY]

            elif s == LATE:
                for half in range(2):
                    t = cycle + 0.5*half
                    if surgery_cycle[i] != -1:
                        alive_or_past5 = (rng.random() < survival_LC and (t - surgery_cycle[i]) < 5) or ((t - surgery_cycle[i]) >= 5)
                        if alive_or_past5:
                            if t == surveillance_due[i]:
                                cycle_cost += (c_colon + c_prep + c_adm_doctor)
                                if rng.random() < complication_prob:
                                    cycle_cost += c_complication
                                dt = t - surgery_cycle[i]
                                if dt == 0.5:
                                    if rng.random() <= rec_LC_1:
                                        cycle_cost += c_LC_first
                                        surveillance_due[i] = t + 0.5
                                        surgery_cycle[i] = t
                                        cycle_qaly += utility[S_LATE]
                                    else:
                                        cycle_cost += c_LC_subseq
                                        surveillance_due[i] = t + 1.5
                                        cycle_qaly += utility[S_LATE]
                                elif dt == 2.0:
                                    if rng.random() <= rec_LC_3:
                                        cycle_cost += c_LC_first
                                        surveillance_due[i] = t + 0.5
                                        surgery_cycle[i] = t
                                        cycle_qaly += utility[S_LATE]
                                    else:
                                        surveillance_due[i] = t + 2.5
                                        cycle_qaly += utility[S_LATE]
                                elif dt == 4.5:
                                    if rng.random() <= rec_LC_5:
                                        cycle_cost += c_LC_first
                                        surveillance_due[i] = t + 0.5
                                        surgery_cycle[i] = t
                                        cycle_qaly += utility[S_LATE]
                                    else:
                                        s = S_CANCER_SURVIVORS
                                        surveillance_due[i] = np.inf
                                        surgery_cycle[i] = -1
                                        cycle_qaly += utility[S_CANCER_SURVIVORS] * (2.0 if half == 0 else 1.0)
                            else:
                                cycle_qaly += utility[S_LATE]
                        else:
                            s = S_DEATH
                            surveillance_due[i] = np.inf
                            surgery_cycle[i] = -1

                    elif rng.random() < discover_LC:
                        cycle_cost += (c_colon + c_prep + c_adm_doctor)
                        if rng.random() < complication_prob:
                            cycle_cost += c_complication
                        surgery_cycle[i] = cycle
                        surveillance_due[i] = cycle + 0.5
                        cycle_cost += c_LC_first
                    cycle_qaly += utility[S_LATE]
            else:
                cycle_qaly += utility[s] * 2.0 if s != DEATH else 0.0

            individual_states[i] = s

        df = disc_factor(cycle + 0.5, years_per_cycle, disc_rate)
        total_cost += df * cycle_cost
        total_qaly += df * cycle_qaly

    return total_cost, total_qaly

# ------------------------------------------------------------
# Screening simulation (shared core; program differences via params)
# ------------------------------------------------------------
def run_screening(adherence: float,
                  fit_choice: float,
                  colon_choice: float,
                  consult_before_fit_cost: float,
                  program_name: str):
    rng = np.random.default_rng()  # independent stream per call
    individual_states = initial_states(rng).astype(np.int8)
    screening_eligible = np.ones(population_size, dtype=bool)
    screening_period = True
    next_fit_due = np.zeros(population_size)  # due from cycle 0

    miss_active = np.zeros(population_size, dtype=bool)
    miss_start_cycle = np.full(population_size, -1, dtype=int)
    stationary = np.zeros(population_size)

    outside_track = np.zeros(population_size, dtype=np.int8)  # 0=not, 1=post-polypectomy, 2=post-EC, 3=post-LC
    clinical_state = np.copy(individual_states)
    surveillance_due = np.full(population_size, np.inf)
    surgery_cycle = np.full(population_size, -1.0)

    total_cost = 0.0
    total_qaly = 0.0

    for cycle in range(n_cycles):
        if cycle > 10:
            screening_period = False

        cycle_cost = 0.0
        cycle_qaly = 0.0

        for i in range(population_size):
            s = individual_states[i]
            counted = False

            # Natural history (unless stationary due to recent negative colonoscopy)
            if stationary[i] <= cycle:
                if not miss_active[i]:
                    miss_active[i] = True
                    miss_start_cycle[i] = cycle

                row = T_scr.copy()
                if s == S_EARLY and miss_active[i]:
                    m = cycle - miss_start_cycle[i]
                    p_prog = p_E2L_interval(m, CYCLE_MONTHS, p_death)
                    stay = max(0.0, 1.0 - p_prog - p_death)
                    row[S_EARLY][S_EARLY] = stay
                    row[S_EARLY][S_LATE] = p_prog

                u = rng.random()
                cumsum = np.cumsum(row[s])
                new_state = int(np.searchsorted(cumsum, u, side="right"))
                if new_state != s:
                    miss_active[i] = False
                    miss_start_cycle[i] = -1
                s = new_state

            # Absorbing states → not eligible
            if s in (S_DEATH, S_OUTSIDE):
                screening_eligible[i] = False
            else:
                # FIXED admin cost (program overhead) always charged
                cycle_cost += c_admin * 2.0

            # Symptomatic discovery (mirrors baseline)
            if s == S_EARLY:
                for _ in range(2):
                    if surgery_cycle[i] == -1 and s not in (S_DEATH, S_OUTSIDE) and rng.random() < discover_EC:
                        cycle_cost += (c_colon + c_prep + c_adm_doctor)
                        if rng.random() < complication_prob:
                            cycle_cost += c_complication
                        surgery_cycle[i] = cycle
                        surveillance_due[i] = cycle + 0.5
                        cycle_cost += c_EC_first
                        outside_track[i] = 2
                        clinical_state[i] = S_EARLY
                        screening_eligible[i] = False
                        miss_active[i] = False
                        miss_start_cycle[i] = -1
                        s = S_OUTSIDE

            elif s == S_LATE:
                for _ in range(2):
                    if surgery_cycle[i] == -1 and s not in (S_DEATH, S_OUTSIDE) and rng.random() < discover_LC:
                        cycle_cost += (c_colon + c_prep + c_adm_doctor)
                        if rng.random() < complication_prob:
                            cycle_cost += c_complication
                        surgery_cycle[i] = cycle
                        surveillance_due[i] = cycle + 0.5
                        cycle_cost += c_LC_first
                        outside_track[i] = 3
                        clinical_state[i] = S_LATE
                        screening_eligible[i] = False
                        miss_active[i] = False
                        miss_start_cycle[i] = -1
                        s = S_OUTSIDE

            # --- Organized screening (only during screening_period & when due) ---
            if screening_eligible[i] and screening_period and (next_fit_due[i] <= cycle) and s not in (S_DEATH, S_OUTSIDE):
                # Adherence
                if rng.random() < adherence:
                    # Person engages with program → charge pre-FIT consult once
                    cycle_cost += consult_before_fit_cost  

                    # Program choice (FIT vs upfront colon)
                    if rng.random() < fit_choice:
                        # FIT pathway
                        cycle_cost += c_fit_kit

                        if s == S_POLYPS:
                            if rng.random() < fit_sensitivity_polyps:
                                # positive FIT → colonoscopy
                                cycle_cost += (c_colon + c_prep + c_adm_doctor + c_consult_positive_fit)
                                if rng.random() < complication_prob:
                                    cycle_cost += c_complication
                                if rng.random() < colonoscopy_sensitivity_polyps:
                                    s = S_OUTSIDE
                                    cycle_cost += c_polypectomy
                                    surgery_cycle[i] = cycle
                                    outside_track[i] = 1
                                    clinical_state[i] = S_NORMAL
                                    surveillance_due[i] = cycle + 1.5
                                    screening_eligible[i] = False
                                    miss_active[i] = False
                                    miss_start_cycle[i] = -1
                                else:
                                    next_fit_due[i] = cycle + 5  # 10y
                            else:
                                # FIT false negative
                                next_fit_due[i] = cycle + 1      # 2y
                                if s == S_EARLY and not miss_active[i]:
                                    miss_active[i] = True
                                    miss_start_cycle[i] = cycle

                        elif s in (S_EARLY, S_LATE):
                            if rng.random() < fit_sensitivity_cancer:
                                cycle_cost += (c_colon + c_prep + c_adm_doctor + c_consult_positive_fit)
                                if rng.random() < complication_prob:
                                    cycle_cost += c_complication
                                if rng.random() < colonoscopy_sensitivity_cancer:
                                    surgery_cycle[i] = cycle
                                    if s == S_EARLY:
                                        cycle_cost += c_EC_first
                                        outside_track[i] = 2
                                        clinical_state[i] = S_EARLY
                                    else:
                                        cycle_cost += c_LC_first
                                        outside_track[i] = 3
                                        clinical_state[i] = S_LATE
                                    surveillance_due[i] = cycle + 0.5
                                    screening_eligible[i] = False
                                    miss_active[i] = False
                                    miss_start_cycle[i] = -1
                                    s = S_OUTSIDE
                                else:
                                    next_fit_due[i] = cycle + 5
                            else:
                                next_fit_due[i] = cycle + 1
                                if s == S_EARLY and not miss_active[i]:
                                    miss_active[i] = True
                                    miss_start_cycle[i] = cycle
                        else:
                            # NORMAL (and others)
                            if rng.random() > fit_specificity:
                                cycle_cost += (c_colon + c_prep + c_adm_doctor + c_consult_positive_fit)
                                if rng.random() < complication_prob:
                                    cycle_cost += c_complication
                                if rng.random() < colonoscopy_specificity:
                                    next_fit_due[i] = cycle + 5
                                    stationary[i] = cycle + 5
                            else:
                                next_fit_due[i] = cycle + 1
                    else:
                        # Upfront colonoscopy pathway
                        cycle_cost += (c_colon + c_prep + c_adm_doctor + c_consult_positive_fit)
                        if rng.random() < complication_prob:
                            cycle_cost += c_complication

                        if s == S_POLYPS:
                            if rng.random() < colonoscopy_sensitivity_polyps:
                                s = S_OUTSIDE
                                cycle_cost += c_polypectomy
                                surgery_cycle[i] = cycle
                                outside_track[i] = 1
                                clinical_state[i] = S_NORMAL
                                surveillance_due[i] = cycle + 1.5
                                screening_eligible[i] = False
                                miss_active[i] = False
                                miss_start_cycle[i] = -1
                            else:
                                next_fit_due[i] = cycle + 5
                        elif s in (S_EARLY, S_LATE):
                            if rng.random() < colonoscopy_sensitivity_cancer:
                                surgery_cycle[i] = cycle
                                if s == S_EARLY:
                                    cycle_cost += c_EC_first
                                    outside_track[i] = 2
                                    clinical_state[i] = S_EARLY
                                else:
                                    cycle_cost += c_LC_first
                                    outside_track[i] = 3
                                    clinical_state[i] = S_LATE
                                surveillance_due[i] = cycle + 0.5
                                screening_eligible[i] = False
                                miss_active[i] = False
                                miss_start_cycle[i] = -1
                                s = S_OUTSIDE
                            else:
                                next_fit_due[i] = cycle + 5
                        else:
                            # Normal with upfront colonoscopy → set stationary if negative
                            if rng.random() < colonoscopy_specificity:
                                next_fit_due[i] = cycle + 5
                                stationary[i] = cycle + 5

            # --- OUTSIDE dynamics (treatment/surveillance) ---
            if s == S_OUTSIDE:
                for half in range(2):
                    t = cycle + 0.5*half
                    if outside_track[i] == 1:  # post-polypectomy
                        if t == surveillance_due[i]:
                            cycle_cost += (c_colon + c_prep + c_adm_doctor)
                            if rng.random() < complication_prob:
                                cycle_cost += c_complication
                            if rng.random() <= rec_polyps:
                                cycle_cost += c_polypectomy
                                surgery_cycle[i] = t
                                surveillance_due[i] = t + 1.0
                                cycle_qaly += utility[S_POLYPS]
                            else:
                                if (t - surgery_cycle[i]) == 1.5:
                                    surveillance_due[i] = t + 1.0
                                    cycle_qaly += utility[S_POLYPS]
                                else:
                                    s = S_NORMAL
                                    outside_track[i] = 0
                                    stationary[i] = cycle + 5
                                    screening_eligible[i] = True
                                    surveillance_due[i] = np.inf
                                    surgery_cycle[i] = -1
                                    next_fit_due[i] = cycle + 5
                                    cycle_qaly += utility[S_NORMAL] * (2.0 if half == 0 else 1.0)
                                    counted = True
                        else:
                            cycle_qaly += utility[S_POLYPS]

                    elif outside_track[i] == 2:  # early cancer tx
                        alive_or_past5 = (rng.random() < survival_EC and (t - surgery_cycle[i]) < 5) or ((t - surgery_cycle[i]) >= 5)
                        if alive_or_past5:
                            if t == surveillance_due[i]:
                                cycle_cost += (c_colon + c_prep + c_adm_doctor)
                                if rng.random() < complication_prob:
                                    cycle_cost += c_complication
                                dt = t - surgery_cycle[i]
                                if dt == 0.5:
                                    if rng.random() <= rec_EC_1:
                                        cycle_cost += c_EC_first
                                        surveillance_due[i] = t + 0.5
                                        surgery_cycle[i] = t
                                        cycle_qaly += utility[S_EARLY]
                                    else:
                                        cycle_cost += c_EC_subseq
                                        surveillance_due[i] = t + 1.5
                                        cycle_qaly += utility[S_EARLY]
                                elif dt == 2.0:
                                    if rng.random() <= rec_EC_3:
                                        cycle_cost += c_EC_first
                                        surveillance_due[i] = t + 0.5
                                        surgery_cycle[i] = t
                                        cycle_qaly += utility[S_EARLY]
                                    else:
                                        surveillance_due[i] = t + 2.5
                                        cycle_qaly += utility[S_EARLY]
                                elif dt == 4.5:
                                    if rng.random() <= rec_EC_5:
                                        cycle_cost += c_EC_first
                                        surveillance_due[i] = t + 0.5
                                        surgery_cycle[i] = t
                                        cycle_qaly += utility[S_EARLY]
                                    else:
                                        s = S_CANCER_SURVIVORS
                                        outside_track[i] = 0
                                        surveillance_due[i] = np.inf
                                        surgery_cycle[i] = -1
                                        cycle_qaly += utility[S_CANCER_SURVIVORS] * (2.0 if half == 0 else 1.0)
                                        counted = True
                            else:
                                cycle_qaly += utility[S_EARLY]
                        else:
                            s = S_DEATH
                            outside_track[i] = 0
                            surveillance_due[i] = np.inf
                            surgery_cycle[i] = -1

                    elif outside_track[i] == 3:  # late cancer tx
                        alive_or_past5 = (rng.random() < survival_LC and (t - surgery_cycle[i]) < 5) or ((t - surgery_cycle[i]) >= 5)
                        if alive_or_past5:
                            if t == surveillance_due[i]:
                                cycle_cost += (c_colon + c_prep + c_adm_doctor)
                                if rng.random() < complication_prob:
                                    cycle_cost += c_complication
                                dt = t - surgery_cycle[i]
                                if dt == 0.5:
                                    if rng.random() <= rec_LC_1:
                                        cycle_cost += c_LC_first
                                        surveillance_due[i] = t + 0.5
                                        surgery_cycle[i] = t
                                        cycle_qaly += utility[S_LATE]
                                    else:
                                        cycle_cost += c_LC_subseq
                                        surveillance_due[i] = t + 1.5
                                        cycle_qaly += utility[S_LATE]
                                elif dt == 2.0:
                                    if rng.random() <= rec_LC_3:
                                        cycle_cost += c_LC_first
                                        surveillance_due[i] = t + 0.5
                                        surgery_cycle[i] = t
                                        cycle_qaly += utility[S_LATE]
                                    else:
                                        surveillance_due[i] = t + 2.5
                                        cycle_qaly += utility[S_LATE]
                                elif dt == 4.5:
                                    if rng.random() <= rec_LC_5:
                                        cycle_cost += c_LC_first
                                        surveillance_due[i] = t + 0.5
                                        surgery_cycle[i] = t
                                        cycle_qaly += utility[S_LATE]
                                    else:
                                        s = S_CANCER_SURVIVORS
                                        outside_track[i] = 0
                                        surveillance_due[i] = np.inf
                                        surgery_cycle[i] = -1
                                        cycle_qaly += utility[S_CANCER_SURVIVORS] * (2.0 if half == 0 else 1.0)
                                        counted = True
                            else:
                                cycle_qaly += utility[S_LATE]
                        else:
                            s = S_DEATH
                            outside_track[i] = 0
                            surveillance_due[i] = np.inf
                            surgery_cycle[i] = -1

            # QALYs for people not in OUTSIDE this cycle (and not already counted)
            if s != S_OUTSIDE and not counted:
                if s in (S_EARLY, S_LATE):
                    pass  # already added per half-cycle above
                else:
                    cycle_qaly += utility[s] * (0.0 if s == S_DEATH else 2.0)

            individual_states[i] = s

        df = disc_factor(cycle + 0.5, years_per_cycle, disc_rate)
        total_cost += df * cycle_cost
        total_qaly += df * cycle_qaly

    return total_cost, total_qaly

# ------------------------------------------------------------
# Run scenarios across adherence (no shared seeds, no CRN)
# ------------------------------------------------------------
inmb_Tic = np.zeros_like(adh_points)
inmb_Fri20 = np.zeros_like(adh_points)
inmb_Fri80 = np.zeros_like(adh_points)
inmb_Fri50 = np.zeros_like(adh_points)

dC_Tic = np.zeros_like(adh_points)
dC_Fri20 = np.zeros_like(adh_points)
dC_Fri80 = np.zeros_like(adh_points)
dC_Fri50 = np.zeros_like(adh_points)

dE_Tic = np.zeros_like(adh_points)
dE_Fri20 = np.zeros_like(adh_points)
dE_Fri80 = np.zeros_like(adh_points)
dE_Fri50 = np.zeros_like(adh_points)

base_cost, base_qaly = run_baseline()

for k, adh in enumerate(adh_points):
    # Ticino 100/0
    tic_cost, tic_qaly = run_screening(
        adherence=adh,
        fit_choice=FIT_choice_Tic,
        colon_choice=colon_choice_Tic,
        consult_before_fit_cost=c_consult_before_FIT_Tic,
        program_name="Ticino",
    )

    # Fribourg 20/80
    fri20_cost, fri20_qaly = run_screening(
        adherence=adh,
        fit_choice=FIT_choice_Fri_20,
        colon_choice=colon_choice_Fri_80,
        consult_before_fit_cost=c_consult_before_FIT_Fri,
        program_name="Fribourg (20/80)",
    )

    # Fribourg 80/20
    fri80_cost, fri80_qaly = run_screening(
        adherence=adh,
        fit_choice=FIT_choice_Fri_80,
        colon_choice=colon_choice_Fri_20,
        consult_before_fit_cost=c_consult_before_FIT_Fri,
        program_name="Fribourg (80/20)",
    )

    # Fribourg 50/50
    fri50_cost, fri50_qaly = run_screening(
        adherence=adh,
        fit_choice=FIT_choice_Fri_50,
        colon_choice=colon_choice_Fri_50,
        consult_before_fit_cost=c_consult_before_FIT_Fri,
        program_name="Fribourg (50/50)",
    )

    # Per-person deltas vs baseline
    dC_Tic[k] = (tic_cost - base_cost) / population_size
    dE_Tic[k] = (tic_qaly - base_qaly) / population_size
    inmb_Tic[k] = WTP * dE_Tic[k] - dC_Tic[k]

    dC_Fri20[k] = (fri20_cost - base_cost) / population_size
    dE_Fri20[k] = (fri20_qaly - base_qaly) / population_size
    inmb_Fri20[k] = WTP * dE_Fri20[k] - dC_Fri20[k]

    dC_Fri80[k] = (fri80_cost - base_cost) / population_size
    dE_Fri80[k] = (fri80_qaly - base_qaly) / population_size
    inmb_Fri80[k] = WTP * dE_Fri80[k] - dC_Fri80[k]

    dC_Fri50[k] = (fri50_cost - base_cost) / population_size
    dE_Fri50[k] = (fri50_qaly - base_qaly) / population_size
    inmb_Fri50[k] = WTP * dE_Fri50[k] - dC_Fri50[k]

# ------------------------------------------------------------
# Plots (4 lines each)
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(adh_pct, inmb_Tic, marker="o", label="Ticino vs No screening")
plt.plot(adh_pct, inmb_Fri20, marker="^", label="Fribourg 20/80 vs No screening")
plt.plot(adh_pct, inmb_Fri80, marker="s", label="Fribourg 80/20 vs No screening")
plt.plot(adh_pct, inmb_Fri50, marker="D", label="Fribourg 50/50 vs No screening")
plt.axhline(0, linestyle="--", color="red", label="INMB = 0 (WTP = 100k CHF/QALY)")
plt.xlabel("Adherence rate (%)")
plt.ylabel("Incremental Net Monetary Benefit (CHF per person)")
plt.title("INMB vs Adherence")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(adh_pct, dC_Tic, marker="o", label="Ticino – ΔCost")
plt.plot(adh_pct, dC_Fri20, marker="^", label="Fribourg 20/80 – ΔCost")
plt.plot(adh_pct, dC_Fri80, marker="s", label="Fribourg 80/20 – ΔCost")
plt.plot(adh_pct, dC_Fri50, marker="D", label="Fribourg 50/50 – ΔCost")
plt.axhline(0, linestyle="--", color="gray")
plt.xlabel("Adherence rate (%)")
plt.ylabel("ΔCost vs No screening (CHF per person)")
plt.title("Incremental Cost vs Adherence")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(adh_pct, dE_Tic, marker="o", label="Ticino – ΔQALY")
plt.plot(adh_pct, dE_Fri20, marker="^", label="Fribourg 20/80 – ΔQALY")
plt.plot(adh_pct, dE_Fri80, marker="s", label="Fribourg 80/20 – ΔQALY")
plt.plot(adh_pct, dE_Fri50, marker="D", label="Fribourg 50/50 – ΔQALY")
plt.axhline(0, linestyle="--", color="gray")
plt.xlabel("Adherence rate (%)")
plt.ylabel("ΔQALY vs No screening (per person)")
plt.title("Incremental QALYs vs Adherence")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

