
"""
Simulation CRC Screening in Ticino 

Cohort simulation over 15 two-year cycles (30y) with:
- FIT screening cadence and confirmatory colonoscopy,
- an 'Outside of Cycle' treatment/surveillance workflow,
- dynamic Weibull Early→Late progression (time-since-miss),
- costs, complications, recurrences, and discounted QALYs.

Outputs: per-cycle cohort counts, deaths (cause/state), costs, QALYs, and incidence.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, exp, gamma
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _mean_given_alpha(alpha, CYCLE_MONTHS, p0):
    """
    For a proposed shape 'alpha', compute:
      - lambda_ so that P(progression in the first cycle) = p0
      - the implied mean time to progression E[T]
    Returns (mean_time, lambda_)
    """
    lam = CYCLE_MONTHS / (-log(1 - p0))**(1/alpha)
    return lam * gamma(1 + 1/alpha), lam

def p_E2L_interval(m_cycles_since_first_miss: int, CYCLE_MONTHS, p_death = 0.20) -> float:
    """Conditional prob of progressing EARLY->LATE in the *next* cycle,
    given m full cycles have already elapsed since the FIRST false negative."""
    t0 = m_cycles_since_first_miss * CYCLE_MONTHS
    t1 = (m_cycles_since_first_miss + 1) * CYCLE_MONTHS
    # Weibull cumulative hazards H(t) = (t/λ)^α
    H0 = (t0 / lambda_)**alpha
    H1 = (t1 / lambda_)**alpha
    p_cond = 1.0 - exp(-(H1 - H0))     # == 1 - S(t1)/S(t0)
    # Competing risk with per-cycle death ~0.20 (simple additive combo):
    return min(max(0.0, p_cond), 1.0 - p_death)

def disc_factor(cycle, years_per_cycle, r):
    return 1.0 / ((1.0 + r) ** (cycle * years_per_cycle))

# Cause-of-death label from the state the person dies in
def _death_cause_from_state(state_idx, NORMAL, POLYPS, EARLY, LATE):
    if state_idx == POLYPS:
        return "From Polyps"
    elif state_idx == EARLY:
        return "From Early Cancer"
    elif state_idx == LATE:
        return "From Late Cancer"
    else:
        return "Other Causes"
    
# ------------------------------------------------------------
# Simulation horizon & Initial population
# ------------------------------------------------------------
years_per_cycle = 2
n_cycles = 15
population_size = 100000 
perc_polpys = 11.8
perc_EC = 0.04
population_polyps = int(( population_size /100 ) * perc_polpys)
population_EC = int(( population_size /100 ) * perc_EC)
population_normal = population_size - ( population_polyps+ population_EC )

# ------------------------------------------------------------
# Cost inputs CHF
# ------------------------------------------------------------
disc_rate = 0.03 
c_consult_positive_fit = 24.0
c_prep = 50.0
c_complication = 8320.0
c_polypectomy = 91.75   
c_adm_doctor = 25.80
c_EC_first = 28197.0 #EC = Early Cancer
c_EC_subseq = 3202.0
c_LC_first = 37685.0 #LC = Late Cancer
c_LC_subseq = 4275.0  
c_admin = 4.85    
c_consult_before_FIT_Tic = 70.70
c_consult_before_FIT_Fri = 94.50    

# ------------------------------------------------------------
# Clinical inputs: complications, recurrences, 5-year survival 
# ------------------------------------------------------------
#Complication probability
complication_prob = 0.002           
# Recurrence Probabilities 
rec_polyps = 0.39 # 3 years
rec_EC_1 = 0.015 #recurrence after 1y from surgery (guideline: colonoscopy after 1 year)
rec_EC_3 = 0.09  #recurrance after 3y from last colonoscopy (4 years from surgery)
rec_EC_5 = 0.11  #recurrance after 5y from last colonoscopy (9 years from surgery)
rec_LC_1 = 0.085
rec_LC_3 = 0.28
rec_LC_5 = 0.30

# Survival rate (5 year)
survival_EC = 0.93
survival_LC = 0.13

# Inputs Weibull problem
CYCLE_MONTHS = 24.0
DT_v = 26.0
k_doublings = 2.6
T_prog_mean = k_doublings * DT_v          # 67.6 months
p0 = 0.20                                 # baseline E->L in first cycle
p_death = 0.20                            # early cancer -> death

# simple bisection for alpha
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

# =============================================================================
# Initialize function 
# =============================================================================

def run_ticino ( fit_sensitivity_polyps = 0.40, fit_sensitivity_cancer = 0.91, fit_specificity = 0.90, colonoscopy_sensitivity_polyps = 0.80, colonoscopy_sensitivity_cancer = 0.98, colonoscopy_specificity = 1.0, c_fit_kit = 46.0, c_colon = 559.30):
    
    # Set seed for reproducibility
    np.random.seed(1)

    # ===============================================================
    # TICINO SCREENING — WITH 'OUTSIDE OF CYCLE' 
    # ===============================================================
    
    #Parameters
    states_scr = ['Normal', 'Polyps', 'Early Cancer', 'Late Cancer', 'Death', 'Cancer Survivors', 'Outside of Cycle']
    NORMAL, POLYPS, EARLY, LATE, DEATH, CANCER_SURVIVORS, OUTSIDE = range(len(states_scr))
    n_states_scr = len(states_scr)
    # Run baseline cohort model
    initial_cohort_base = np.array([population_normal, population_polyps, population_EC, 0, 0, 0, 0])
    
    # Utilities weights for QALY
    utility = np.zeros(6)
    utility[NORMAL]= 1
    utility[POLYPS] = 0.88
    utility[EARLY]= 0.74
    utility[LATE] = 0.40
    utility[DEATH] = 0
    utility[CANCER_SURVIVORS] = 0.82
    
    # Probabilities to discover cancer without screening 
    discover_EC = 0.21
    discover_LC = 0.28
        
    # Transition matrix
    T_scr = np.zeros((n_states_scr, n_states_scr))
    T_scr[NORMAL] = [0.864, 0.13, 0.0, 0.0, 0.006, 0.0, 0.0]
    T_scr[POLYPS] = [0.0, 0.824, 0.17, 0.0, 0.006, 0.0, 0.0]
    T_scr[EARLY]  = [0.0, 0.0, 0.60, 0.20, 0.20, 0.0, 0.0]   
    T_scr[LATE]   = [0.0, 0.0, 0.0, 0.65, 0.35, 0.0, 0.0]
    T_scr[DEATH]  = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    T_scr[CANCER_SURVIVORS]= [0.0, 0.0, 0.0, 0.0, 0.006, 0.994, 0.0]
    T_scr[OUTSIDE]= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]      # absorbing
    
    screening_eligible = np.ones(population_size, dtype=bool)
    screening_period = True
    next_fit_due = np.zeros(population_size)       # FIT scheduling: due at cycle 0; OUTSIDE or Death disables screening_eligible
    # Build the array
    individual_states = np.array(
        [0] * population_normal +
        [1] * population_polyps +
        [2] * population_EC
    )
    
    # Shuffle so states are randomly distributed
    np.random.shuffle(individual_states)
    # False-negative tracking: enables time-dependent Early→Late progression
    # When cancer is missed by screening, progression risk increases with time
    miss_active = np.zeros(population_size, dtype=bool)           # currently in a missed run? Currently experiencing a false-negative?
    miss_start_cycle = np.full(population_size, -1, dtype=int)    # Cycle index when the FIRST false-negative occurred
    stationary = np.zeros(population_size)                        # person that has had neg colonoscopy does not move from Normal state for 10y
    
    # OUTSIDE support
    outside_track = np.zeros(population_size, dtype=np.int8)  # 0=not outside, 1=post-polypectomy, 2=post-EC, 3=post-LC
    clinical_state = np.copy(individual_states)               # shadow clinical state while OUTSIDE
    surveillance_due = np.full(population_size, np.inf)       # post-polypectomy colonoscopy schedule (in cycles)
    surgery_cycle = np.full(population_size, -1.0)            # save the date of the surgery (polypectomy or cancer removal)
    
    # Costs
    total_cost_screening = 0
    total_qaly_screening = 0 
    cycle_cost_screening_array = []
    cycle_qaly_screening_array = []
    
    #Start Cycle 
    cycle_results = [initial_cohort_base.copy()]
    #Death tracker for CRC and other causes
    death_tracker = np.full(population_size, -1)
    
    # per-cycle deaths by cause for baseline ----
    deaths_by_cause_scr = {
        "Other Causes": np.zeros(n_cycles, dtype=int),
        "From Polyps": np.zeros(n_cycles, dtype=int),
        "From Early Cancer": np.zeros(n_cycles, dtype=int),
        "From Late Cancer": np.zeros(n_cycles, dtype=int),
    }
    
    deaths_by_state_scr = {
        "Normal": np.zeros(n_cycles, dtype=int),
        "Polyps": np.zeros(n_cycles, dtype=int),
        "Early": np.zeros(n_cycles, dtype=int),
        "Late": np.zeros(n_cycles, dtype=int),
        "Cancer Survivors": np.zeros(n_cycles, dtype=int),
    }
    
    # CRC incidence trackers
    ever_early_scr = np.zeros(population_size, dtype=bool)
    detected_early_scr = np.zeros(population_size, dtype=bool)
    ever_late_scr = np.zeros(population_size, dtype=bool)
    detected_late_scr = np.zeros(population_size, dtype=bool)
    
    #START
    for cycle in range(n_cycles):
        state_counts = np.zeros(n_states_scr, dtype=int)
        cycle_cost = 0 
        cycle_qaly = 0
        
        if cycle > 10: 
            screening_period = False
    
        for i in range(population_size):
            state = individual_states[i] 
            counted = False
            
            # Natural history: update Early row dynamically if in a missed run
            # Note: when OUTSIDE or after negative colonoscopy (stationary>cycle), hold natural history
            if stationary[i] <= cycle: 
                if state == EARLY and miss_active[i]:
                    m = cycle - miss_start_cycle[i]
                    p_prog = p_E2L_interval(m, CYCLE_MONTHS, p_death)
                    stay = max(0.0, 1.0 - p_prog - p_death)
                    row = T_scr.copy()
                    row[EARLY][2] = stay
                    row[EARLY][3]  = p_prog
                else: 
                    row = T_scr.copy()
                new_state = np.random.choice(n_states_scr, p = row[state])
                if new_state == EARLY and not ever_early_scr[i]:
                        ever_early_scr[i] = True
                if new_state == LATE and not ever_late_scr[i]:
                        ever_late_scr[i] = True
                if new_state == state: 
                    state = new_state
                else: 
                    miss_active[i] = False; miss_start_cycle[i] = -1
                    
                    if new_state == DEATH: # per-cycle death cause count 
                        if state != NORMAL and np.random.rand() < ((T_scr[state][4] - T_scr[NORMAL][4])) / T_scr[state][4]: 
                            cause_label = _death_cause_from_state(state, NORMAL, POLYPS, EARLY, LATE)
                            deaths_by_cause_scr[cause_label][cycle] += 1 
                        else: 
                            cause_label = _death_cause_from_state(NORMAL, NORMAL, POLYPS, EARLY, LATE) 
                            deaths_by_cause_scr[cause_label][cycle] += 1
                        
                        # Attribute cause of death
                        if state == POLYPS:
                            deaths_by_state_scr["Polyps"][cycle] += 1
                        elif state == EARLY:
                            deaths_by_state_scr["Early"][cycle] += 1                   
                        elif state == LATE:
                            deaths_by_state_scr["Late"][cycle] += 1
                        elif state == CANCER_SURVIVORS:
                            deaths_by_state_scr["Cancer Survivors"][cycle] += 1
                        else:
                            deaths_by_state_scr["Normal"][cycle] += 1
    
                    state = new_state
     
            # Absorbing states
            if state in (DEATH, OUTSIDE):
                screening_eligible[i] = False
            else: 
                cycle_cost += c_admin*2
                
                if np.random.rand() < 0.5 :
                    cycle_cost += c_consult_before_FIT_Tic
    
            # --- 1) USUAL-CARE DETECTION (before screening actions in this cycle) ---
            if state == EARLY:
                # if not already in treatment
                if surgery_cycle[i] == -1 and state not in (DEATH, OUTSIDE):
                    if np.random.rand() < discover_EC:
                        cycle_cost += (c_colon + c_prep + c_adm_doctor)
                        if np.random.rand() < complication_prob:
                            cycle_cost += c_complication
                        surgery_cycle[i] = cycle
                        surveillance_due[i] = cycle + 0.5
                        cycle_cost += c_EC_first
                        outside_track[i] = 2
                        clinical_state[i] = EARLY
                        screening_eligible[i] = False
                        miss_active[i] = False; miss_start_cycle[i] = -1
                        state = OUTSIDE
                        detected_early_scr[i] = True
            
            elif state == LATE:
                if surgery_cycle[i] == -1 and state not in (DEATH, OUTSIDE):
                    if np.random.rand() < discover_LC:
                        cycle_cost += (c_colon + c_prep + c_adm_doctor)
                        if np.random.rand() < complication_prob:
                            cycle_cost += c_complication
                        surgery_cycle[i] = cycle
                        surveillance_due[i] = cycle + 0.5
                        cycle_cost += c_LC_first
                        outside_track[i] = 3
                        clinical_state[i] = LATE
                        screening_eligible[i] = False
                        miss_active[i] = False; miss_start_cycle[i] = -1
                        state = OUTSIDE
                        detected_late_scr[i] = True
                        
            # --- 2) SCREENING workflow (FIT ± colonoscopy) ---
            if screening_eligible[i] and screening_period and next_fit_due[i] <= cycle and state not in (DEATH, OUTSIDE):
                cycle_cost += ( c_fit_kit )
                
                if state == POLYPS:
                    if np.random.rand() < fit_sensitivity_polyps:
                        # positive FIT -> colonoscopy
                        cycle_cost += (c_colon + c_prep + c_adm_doctor + c_consult_positive_fit)
                        if np.random.rand() < complication_prob:
                            cycle_cost += c_complication
                        if np.random.rand() < colonoscopy_sensitivity_polyps:
                            state = OUTSIDE
                            cycle_cost += c_polypectomy
                            surgery_cycle[i] = cycle
                            outside_track[i] = 1
                            clinical_state[i] = NORMAL
                            surveillance_due[i] = cycle + 1.5  # 3y surveillance after polypectomy
                            screening_eligible[i] = False
                            miss_active[i] = False; miss_start_cycle[i] = -1
                        else:
                            # negative colonoscopy after positive FIT
                            next_fit_due[i] = cycle + 5   # next FIT in 10y (5 cycles)
                            if not miss_active[i]:
                                miss_active[i] = True
                                miss_start_cycle[i] = cycle
                    else:
                        # FIT false negative
                        next_fit_due[i] = cycle + 1      # next FIT in 2y (1 cycle)
                        if not miss_active[i]:
                            miss_active[i] = True
                            miss_start_cycle[i] = cycle
    
                elif state in (EARLY, LATE):
                    if np.random.rand() < fit_sensitivity_cancer:
                        # positive FIT -> colonoscopy
                        cycle_cost += (c_colon + c_prep + c_adm_doctor + c_consult_positive_fit)
                        if np.random.rand() < complication_prob:
                            cycle_cost += c_complication
                        if np.random.rand() < colonoscopy_sensitivity_cancer:
                            surgery_cycle[i] = cycle
                            if state == EARLY: 
                                cycle_cost += c_EC_first
                                detected_early_scr[i] = True
                                outside_track[i] = 2
                                clinical_state[i] = EARLY  
                            else: 
                                cycle_cost += c_LC_first
                                detected_late_scr[i] = True
                                outside_track[i] = 3
                                clinical_state[i] = LATE
                            surveillance_due[i] = cycle + 0.5
                            screening_eligible[i] = False
                            miss_active[i] = False; miss_start_cycle[i] = -1
                            state = OUTSIDE 
                        else:
                            # colonoscopy false negative (miss)
                            next_fit_due[i] = cycle + 5   # 10y
                            if state == EARLY:
                                if not miss_active[i]:
                                    miss_active[i] = True
                                    miss_start_cycle[i] = cycle
                    else:
                        # FIT false negative (miss)
                        next_fit_due[i] = cycle + 1       # 2y
                        if state == EARLY:
                            if not miss_active[i]:
                                miss_active[i] = True
                                miss_start_cycle[i] = cycle
    
                else:  # NORMAL (and other)
                    # FIT false positive?
                    if np.random.rand() > fit_specificity:
                        cycle_cost += (c_colon + c_prep + c_adm_doctor + c_consult_positive_fit)
                        # colonoscopy negative → next FIT in 10y; mark 10y 'stationary' (no natural history moves)
                        if np.random.rand() < complication_prob:
                            cycle_cost += c_complication
                        if np.random.rand() < colonoscopy_specificity:
                            
                            next_fit_due[i] = cycle + 5
                            stationary[i] = cycle + 5 
                    else:
                        next_fit_due[i] = cycle + 1
                        
            # --- 3) OUTSIDE: treatment/surveillance in two half-cycles (j=0,1) ---
            if state == OUTSIDE:
                for j in range (2):
                    if outside_track[i] == 1: #surveillance after polypectomy (3y)
                        if cycle + (j*0.5) == surveillance_due[i]:
                            cycle_cost += (c_colon + c_prep + c_adm_doctor)
                            if np.random.rand() < complication_prob:
                                cycle_cost += c_complication
                            if np.random.rand() <= rec_polyps:
                                cycle_cost += c_polypectomy
                                surgery_cycle[i] = cycle + (j*0.5)
                                surveillance_due[i] = cycle + 1.5 + (j*0.5)
                                cycle_qaly += utility[POLYPS]
                            else:
                                if cycle + (j*0.5) - surgery_cycle[i] == 1.5: 
                                    surveillance_due[i] = cycle + 2.5 + (j*0.5)
                                    cycle_qaly += utility[POLYPS]
                                else: 
                                    state = NORMAL
                                    outside_track[i] = 0 
                                    stationary[i] = cycle + 5 
                                    screening_eligible[i] = True
                                    surveillance_due[i] = np.inf
                                    surgery_cycle[i] = -1 
                                    next_fit_due[i] = cycle + 5
                                    if j== 0:
                                        cycle_qaly += utility[NORMAL] *2
                                    else: 
                                        cycle_qaly += utility[NORMAL]
                                    counted = True
                        else: 
                             cycle_qaly += utility[POLYPS]
                             
                    elif outside_track[i] == 2: # treatment cycle for early cancer
                        if (np.random.rand() < survival_EC and (cycle + (j*0.5)) - surgery_cycle[i] < 5) or (cycle + (j*0.5)) - surgery_cycle[i] >= 5:
                            if cycle + (j*0.5) == surveillance_due[i]: 
                                cycle_cost += (c_colon + c_prep + c_adm_doctor)
                                if np.random.rand() < complication_prob:
                                    cycle_cost += c_complication
                                if cycle + (j*0.5) - surgery_cycle[i] == 0.5: 
                                    if np.random.rand() <= rec_EC_1:
                                        cycle_cost += c_EC_first 
                                        surveillance_due[i] = cycle + 0.5 + (j*0.5)
                                        surgery_cycle[i] = cycle + (j*0.5)
                                        cycle_qaly += utility[EARLY]
                                    else: 
                                        cycle_cost += c_EC_subseq
                                        surveillance_due[i] = cycle + 1.5 + (j*0.5)
                                        cycle_qaly += utility[EARLY]
                                elif cycle + (j*0.5) - surgery_cycle[i] == 2: 
                                    if np.random.rand() <= rec_EC_3: 
                                        cycle_cost += c_EC_first
                                        surveillance_due[i] = cycle + 0.5 + (j*0.5)
                                        surgery_cycle[i] = cycle + (j*0.5)
                                        cycle_qaly += utility[EARLY]
                                    else: 
                                        surveillance_due[i] = cycle + 2.5 + (j*0.5)
                                        cycle_qaly += utility[EARLY]
                                elif cycle + (j*0.5) - surgery_cycle[i] == 4.5: 
                                    if np.random.rand() <= rec_EC_5: 
                                        cycle_cost += c_EC_first
                                        surveillance_due[i] = cycle + 0.5 + (j*0.5)
                                        surgery_cycle[i] = cycle + (j*0.5)
                                        cycle_qaly += utility[EARLY]
                                    else: 
                                        state = CANCER_SURVIVORS
                                        outside_track[i] = 0
                                        surveillance_due[i] = np.inf
                                        surgery_cycle[i] = -1 
                                        if j== 0:
                                            cycle_qaly += utility[CANCER_SURVIVORS] *2
                                        else: 
                                            cycle_qaly += utility[CANCER_SURVIVORS]
                                        counted = True
                            else: 
                                cycle_qaly += utility[EARLY]
                        else: 
                            # Count early-cancer death while OUTSIDE (screening) ----
                            state = DEATH
                            death_tracker[i] = 2
                            deaths_by_cause_scr["From Early Cancer"][cycle] += 1
                            deaths_by_state_scr["Early"][cycle] += 1
                            outside_track[i] = 0 
                            surveillance_due[i] = np.inf
                            surgery_cycle[i] = -1 
                                
                    elif outside_track[i] == 3: # treatment cycle for late cancer
                        if (np.random.rand() < survival_LC and (cycle + (j*0.5)) - surgery_cycle[i] < 5) or (cycle + (j*0.5)) - surgery_cycle[i] >= 5:
                            if cycle + (j*0.5) == surveillance_due[i]: 
                                cycle_cost += (c_colon + c_prep + c_adm_doctor)
                                if np.random.rand() < complication_prob:
                                    cycle_cost += c_complication
                                if cycle + (j*0.5) - surgery_cycle[i] == 0.5: 
                                    if np.random.rand() <= rec_LC_1: 
                                        cycle_cost += c_LC_first
                                        surveillance_due[i] = cycle + 0.5 + (j*0.5)
                                        surgery_cycle[i] = cycle + (j*0.5)
                                        cycle_qaly += utility[LATE]
                                    else: 
                                        cycle_cost += c_LC_subseq
                                        surveillance_due[i] = cycle + 1.5 + (j*0.5)
                                        cycle_qaly += utility[LATE]
                                elif cycle + (j*0.5) - surgery_cycle[i] == 2: 
                                    if np.random.rand() <= rec_LC_3: 
                                        cycle_cost += c_LC_first
                                        surveillance_due[i] = cycle + 0.5 + (j*0.5)
                                        surgery_cycle[i] = cycle + (j*0.5)
                                        cycle_qaly += utility[LATE]
                                    else: 
                                        surveillance_due[i] = cycle + 2.5 + (j*0.5)
                                        cycle_qaly += utility[LATE]
                                elif cycle + (j*0.5) - surgery_cycle[i] == 4.5: 
                                    if np.random.rand() <= rec_LC_5: 
                                        cycle_cost += c_LC_first
                                        surveillance_due[i] = cycle + 0.5 + (j*0.5)
                                        surgery_cycle[i] = cycle + (j*0.5)
                                        cycle_qaly += utility[LATE]
                                    else: 
                                        state = CANCER_SURVIVORS
                                        outside_track[i] = 0
                                        surveillance_due[i] = np.inf
                                        surgery_cycle[i] = -1 
                                        if j== 0:
                                            cycle_qaly += utility[CANCER_SURVIVORS] *2
                                        else: 
                                            cycle_qaly += utility[CANCER_SURVIVORS]
                                        counted = True
                            else: 
                                cycle_qaly += utility[LATE]
                        else: 
                            # Count late-cancer death while OUTSIDE (screening) ----
                            state = DEATH
                            death_tracker[i] = 3
                            deaths_by_cause_scr["From Late Cancer"][cycle] += 1
                            deaths_by_state_scr["Late"][cycle] += 1
                            outside_track[i] = 0 
                            surveillance_due[i] = np.inf
                            surgery_cycle[i] = -1 
    
            individual_states[i] = state
            state_counts[state] += 1
            if state != OUTSIDE and counted == False:            
                cycle_qaly += (utility[state] * 2)
                
        # Apply mid-cycle discount and store cycle aggregates        
        df = disc_factor(cycle + 0.5, years_per_cycle, disc_rate)
        
        total_cost_screening += df * cycle_cost
        total_qaly_screening += df * cycle_qaly
        cycle_cost_screening_array.append(df * cycle_cost)
        cycle_qaly_screening_array.append(df * cycle_qaly)
        cycle_results.append(state_counts)
        
    screening_df = pd.DataFrame(cycle_results, columns=states_scr)
    screening_df.index.name = "Cycle"
    
    costs_screening_df = pd.DataFrame(cycle_cost_screening_array, columns=["Cost"])
    costs_screening_df.index.name = "Cycle"
    
    qaly_screening_df = pd.DataFrame(cycle_qaly_screening_array, columns=["Qaly"])
    qaly_screening_df.index.name = "Cycle"
    
    
    # Death cause base
    deaths_cause_scr_df = pd.DataFrame({
        "Other Causes": deaths_by_cause_scr["Other Causes"],
        "From Polyps": deaths_by_cause_scr["From Polyps"],
        "From Early Cancer": deaths_by_cause_scr["From Early Cancer"],
        "From Late Cancer": deaths_by_cause_scr["From Late Cancer"]
    })
    deaths_cause_scr_df.index.name = "Cycle"
    deaths_cause_scr_df["Year"] = deaths_cause_scr_df.index * years_per_cycle
    
    # Death by state
    deaths_state_scr_df = pd.DataFrame({
        "Normal": deaths_by_state_scr["Normal"],
        "Polyps": deaths_by_state_scr["Polyps"],
        "Early Cancer": deaths_by_state_scr["Early"],
        "Late Cancer": deaths_by_state_scr["Late"],
        "Cancer Survivors": deaths_by_state_scr["Cancer Survivors"],
    })
    deaths_state_scr_df.index.name = "Cycle"
    deaths_state_scr_df["Year"] = deaths_state_scr_df.index * years_per_cycle

    
    # Return values
    return {
        "ticino": {
            "cohort": screening_df,
            "deaths": deaths_cause_scr_df,
            "costs": costs_screening_df,
            "qaly": qaly_screening_df,
            "incidence": {
                "ever_early": int(ever_early_scr.sum()),
                "detected_early": int(detected_early_scr.sum()),
                "ever_late": int(ever_late_scr.sum()),
                "detected_late": int(detected_late_scr.sum())
            }
        }
    }

# Print results
if __name__ == "__main__":
    
    results = run_ticino()
    
    screening_df = results["ticino"]["cohort"]
    deaths_cause_scr_df = results["ticino"]["deaths"]
    costs_screening_df = results["ticino"]["costs"]
    qaly_screening_df = results["ticino"]["qaly"]
    
    print("\n=== Ticino Simulation Complete ===")
    print(f"Total QALYs: {qaly_screening_df['Qaly'].sum():,.2f}")
    print(f"Total Cost: {costs_screening_df['Cost'].sum():,.0f} CHF")
