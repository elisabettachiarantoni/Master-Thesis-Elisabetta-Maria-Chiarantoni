
"""
Simulation No Screening 

Simulates colorectal cancer natural history over 15 two-year cycles (30 years)
without screening. Individuals transition among:
Normal, Polyps, Early Cancer, Late Cancer, Death, and Cancer Survivors.

Weibull progression (Early→Late) is dynamically updated per cycle.
Outputs discounted costs, QALYs, deaths by cause/state, and cancer incidence.
"""
import numpy as np
import pandas as pd
from math import log, exp, gamma

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

# Cause-of-death label from the state the person dies from
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
c_colon = 559.30
c_prep = 50.0
c_complication = 8320.0
c_polypectomy = 91.75   
c_adm_doctor = 25.80
c_EC_first = 28197.0 #EC = Early Cancer
c_EC_subseq = 3202.0
c_LC_first = 37685.0 #LC = Late Cancer
c_LC_subseq = 4275.0  
c_admin = 4.85      

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

def run_noscreening ():
    
    # Set seed for reproducibility
    np.random.seed(1)
    
    # =============================================================================
    # BASELINE — NO SCREENING 
    # =============================================================================
    
    #Parameters
    states_base = ['Normal', 'Polyps', 'Early Cancer', 'Late Cancer', 'Death', 'Cancer Survivors']
    NORMAL, POLYPS, EARLY, LATE, DEATH, CANCER_SURVIVORS = range(len(states_base))
    n_states_base = len(states_base)
    
    # Utilities weights for QALY
    utility = np.zeros(n_states_base)
    utility[NORMAL]= 1
    utility[POLYPS] = 0.88
    utility[EARLY]= 0.74
    utility[LATE] = 0.40
    utility[DEATH] = 0
    utility[CANCER_SURVIVORS] = 0.82
    
    # Probabilities to discover cancer without screening 
    discover_EC = 0.21
    discover_LC = 0.28
    
    # Transition matrix for baseline (2-year cycle)
    T_base = np.zeros((n_states_base, n_states_base))
    T_base[NORMAL] = [0.864, 0.13, 0.0, 0.0, 0.006, 0.0]
    T_base[POLYPS] = [0.0, 0.824, 0.17, 0.0, 0.006, 0.0]
    T_base[EARLY]  = [0.0, 0.0, 0.60, 0.20, 0.20, 0.0] 
    T_base[LATE]   = [0.0, 0.0, 0.0, 0.65, 0.35, 0.0]
    T_base[DEATH]  = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    T_base[CANCER_SURVIVORS] = [0.0, 0.0, 0.0, 0.0, 0.006, 0.994]
    
    # Run baseline cohort model
    initial_cohort_base = np.array([population_normal, population_polyps, population_EC, 0, 0, 0])
    # Build the array
    individual_states = np.array(
        [0] * population_normal +
        [1] * population_polyps +
        [2] * population_EC
    )
    
    # Cancer Treatment variables and surveillance
    outside_track = np.zeros(population_size, dtype=np.int8)  # 0=not outside, 1=post-polypectomy, 2=post-EC, 3=post-LC
    surveillance_due = np.full(population_size, np.inf)       # post-polypectomy colonoscopy schedule (in cycles)
    surgery_cycle = np.full(population_size, -1.0)            # save the date of the surgery (polypectomy or cancer removal)
    
    # Shuffle so states are randomly distributed
    np.random.shuffle(individual_states)
    # False-negative tracking: enables time-dependent Early→Late progression
    # When cancer is missed by screening, progression risk increases with time
    miss_active = np.zeros(population_size, dtype=bool)           # currently in a missed run? Currently experiencing a false-negative?
    miss_start_cycle = np.full(population_size, -1, dtype=int)    # Cycle index when the FIRST false-negative occurred
    cohort_history_base = [initial_cohort_base.copy()]
    
    # per-cycle deaths by cause for baseline 
    deaths_by_cause_base = {
        "Other Causes": np.zeros(n_cycles, dtype=int),
        "From Polyps": np.zeros(n_cycles, dtype=int),
        "From Early Cancer": np.zeros(n_cycles, dtype=int),
        "From Late Cancer": np.zeros(n_cycles, dtype=int),
    }
    
    deaths_by_state_base = {
        "Normal": np.zeros(n_cycles, dtype=int),
        "Polyps": np.zeros(n_cycles, dtype=int),
        "Early": np.zeros(n_cycles, dtype=int),
        "Late": np.zeros(n_cycles, dtype=int),
        "Cancer Survivors": np.zeros(n_cycles, dtype=int),
    }
    
    total_cost_no_screening = 0
    total_qaly_no_screening = 0 
    cycle_cost_no_screening_array = []
    cycle_qaly_no_screening_array = []
    
    # Incidence trackers 
    ever_early_base = np.zeros(population_size, dtype=bool)
    detected_early_base = np.zeros(population_size, dtype=bool)
    ever_late_base = np.zeros(population_size, dtype=bool)
    detected_late_base = np.zeros(population_size, dtype=bool)
    
    #No screening with dynamic probability for Early
    for cycle in range(n_cycles):
        state_counts = np.zeros(n_states_base, dtype=int)
        cycle_cost = 0
        cycle_qaly = 0 
    
        for i in range(population_size):
            state = individual_states[i]
            counted = False
            
            # Start counting cycles in which you are in a state
            if not miss_active[i]: 
                miss_active[i] = True
                miss_start_cycle[i] = cycle 
                
            # Dynamic probability for early cancer            
            if state == EARLY and miss_active[i]:
                m = cycle - miss_start_cycle[i]            #### elapsed missed cycles
                p_prog = p_E2L_interval(m, CYCLE_MONTHS, p_death)   #### dynamic
                stay = max(0.0, 1.0 - p_prog - p_death)
                
                row = T_base.copy()  #copy matrix and make modifications in the copy
                row[EARLY][2] = stay
                row[EARLY][3]  = p_prog
            else: 
                row = T_base.copy()
    
            # If in treatment (surgery_cycle set) and in EC/LC, lock state; otherwise draw transition
            if not ((state == EARLY or state == LATE) and surgery_cycle[i] != -1):
                new_state = np.random.choice(n_states_base, p = row[state])
                # Track first-time entry into cancer states
                if new_state == EARLY and not ever_early_base[i]:
                    ever_early_base[i] = True
                if new_state == LATE and not ever_late_base[i]:
                    ever_late_base[i] = True
            else: 
                new_state = state
    
            if new_state == state: 
                state = new_state
            else: 
                miss_active[i] = False; miss_start_cycle[i] = -1 # reset time-in-state
                
                if new_state == DEATH: # per-cycle death cause count 
                    if state != NORMAL and np.random.rand() < ((T_base[state][4] - T_base[NORMAL][4])) / T_base[state][4]: 
                        cause_label = _death_cause_from_state(state, NORMAL, POLYPS, EARLY, LATE)
                        deaths_by_cause_base[cause_label][cycle] += 1 
                    else: 
                        cause_label = _death_cause_from_state(NORMAL, NORMAL, POLYPS, EARLY, LATE) 
                        deaths_by_cause_base[cause_label][cycle] += 1
                    
                    # Attribute cause of death
                    if state == POLYPS:
                        deaths_by_state_base["Polyps"][cycle] += 1
                    elif state == EARLY:
                        deaths_by_state_base["Early"][cycle] += 1                   
                    elif state == LATE:
                        deaths_by_state_base["Late"][cycle] += 1
                    elif state == CANCER_SURVIVORS:
                        deaths_by_state_base["Cancer Survivors"][cycle] += 1
                    else:
                        deaths_by_state_base["Normal"][cycle] += 1
                state = new_state
    
            # Treatment/surveillance dynamics (can also cause death)
            # no surveillance for polyps baseline assumption
            if state == EARLY: 
                # Split the 2-year cycle into two half-cycles to allow mid-cycle surveillance/costs
                for j in range (2):
                    if surgery_cycle[i] != -1:
                        if (np.random.rand() < survival_EC and (cycle + (j*0.5)) - surgery_cycle[i] < 5) or (cycle + (j*0.5)) - surgery_cycle[i] >= 5:
                            if cycle + (j*0.5) == surveillance_due[i]: 
                                #costs colonoscopy
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
                                        surgery_cycle[i] = -1 
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
                            # count early-cancer death while in treatment
                            deaths_by_cause_base["From Early Cancer"][cycle] += 1
                            deaths_by_state_base["Early"][cycle] += 1
                            state = DEATH
                            outside_track[i] = 0 
                            surveillance_due[i] = np.inf
                            surgery_cycle[i] = -1 
                    else:
                        cycle_qaly += utility[EARLY]
                    
                        if  np.random.rand() < discover_EC:
                            cycle_cost += (c_colon + c_prep + c_adm_doctor)
                            if np.random.rand() < complication_prob:
                                cycle_cost += c_complication
                            surgery_cycle[i] = (j*0.5) + cycle  
                            surveillance_due[i] = cycle + 0.5 + (j*0.5)
                            cycle_cost += c_EC_first 
                            detected_early_base[i] = True
    
            elif state == LATE: 
                for j in range (2): 
                    if surgery_cycle[i] != -1:
                        if (np.random.rand() < survival_LC and (cycle + (j*0.5)) - surgery_cycle[i] < 5) or (cycle + (j*0.5)) - surgery_cycle[i] >= 5:
                            if cycle + (j*0.5) == surveillance_due[i]: 
                                #costs colonoscopy
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
                                    if np.random.rand() <= rec_EC_3: 
                                        cycle_cost += c_LC_first
                                        surveillance_due[i] = cycle + 0.5 + (j*0.5)
                                        surgery_cycle[i] = cycle + (j*0.5)
                                        cycle_qaly += utility[LATE]
                                    else: 
                                        surveillance_due[i] = cycle + 2.5 + (j*0.5)
                                        cycle_qaly += utility[LATE]
                                elif cycle + (j*0.5) - surgery_cycle[i] == 4.5: 
                                    if np.random.rand() <= rec_EC_5: 
                                        cycle_cost += c_LC_first
                                        surveillance_due[i] = cycle + 0.5 + (j*0.5)
                                        surgery_cycle[i] = cycle + (j*0.5)
                                        cycle_qaly += utility[LATE]
                                    else: 
                                        state = CANCER_SURVIVORS
                                        surgery_cycle[i] = -1
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
                            #count late-cancer death while in treatment
                            deaths_by_cause_base["From Late Cancer"][cycle] += 1
                            deaths_by_state_base["Late"][cycle] += 1
                            state = DEATH
                            outside_track[i] = 0 
                            surveillance_due[i] = np.inf
                            surgery_cycle[i] = -1 
                    
                    else:
                        cycle_qaly += utility[LATE]
                        
                        if np.random.rand() < discover_LC:
                            cycle_cost += (c_colon + c_prep + c_adm_doctor)
                            if np.random.rand() < complication_prob:
                                cycle_cost += c_complication
                            surgery_cycle[i] = cycle 
                            surveillance_due[i] = cycle + 0.5 + (j*0.5)
                            cycle_cost += c_LC_first 
                            detected_late_base[i] = True
    
    
            individual_states[i] = state
            state_counts[state] += 1
            if state != EARLY and state != LATE and counted == False:            
                cycle_qaly += (utility[state] * 2)
                
        # Apply mid-cycle discount and store cycle aggregates        
        df = disc_factor(cycle + 0.5, years_per_cycle, disc_rate)
           
        cohort_history_base.append(state_counts)
        total_cost_no_screening  += df * cycle_cost
        total_qaly_no_screening  += df * cycle_qaly
        cycle_cost_no_screening_array.append(df * cycle_cost)
        cycle_qaly_no_screening_array.append(df * cycle_qaly)
    
    cohort_df = pd.DataFrame(cohort_history_base, columns=states_base)
    cohort_df.index.name = "Cycle"
    cohort_df["Year"] = cohort_df.index * years_per_cycle
    
    costs_no_screening_df = pd.DataFrame(cycle_cost_no_screening_array, columns=["Cost"])
    costs_no_screening_df.index.name = "Cycle"
    costs_no_screening_df["Year"] = costs_no_screening_df.index * years_per_cycle
    
    qaly_no_screening_df = pd.DataFrame(cycle_qaly_no_screening_array, columns=["Qaly"])
    qaly_no_screening_df.index.name = "Cycle"
    qaly_no_screening_df["Year"] = qaly_no_screening_df.index * years_per_cycle
    
    # Death cause base
    deaths_cause_base_df = pd.DataFrame({
        "Other Causes": deaths_by_cause_base["Other Causes"],
        "From Polyps": deaths_by_cause_base["From Polyps"],
        "From Early Cancer": deaths_by_cause_base["From Early Cancer"],
        "From Late Cancer": deaths_by_cause_base["From Late Cancer"]
    })
    deaths_cause_base_df.index.name = "Cycle"
    deaths_cause_base_df["Year"] = deaths_cause_base_df.index * years_per_cycle
    
    # Death state base
    deaths_state_base_df = pd.DataFrame({
        "Normal": deaths_by_state_base["Normal"],
        "Polyps": deaths_by_state_base["Polyps"],
        "Early Cancer": deaths_by_state_base["Early"],
        "Late Cancer": deaths_by_state_base["Late"],
        "Cancer Survivors": deaths_by_state_base["Cancer Survivors"]
    })
    deaths_state_base_df.index.name = "Cycle"
    deaths_state_base_df["Year"] = deaths_state_base_df.index * years_per_cycle
    
    # Return values
    return {
        "no_screening": {
            "cohort": cohort_df,
            "deaths": deaths_cause_base_df,
            "costs": costs_no_screening_df,
            "qaly": qaly_no_screening_df,
            "incidence": {
                "ever_early": int(ever_early_base.sum()),
                "detected_early": int(detected_early_base.sum()),
                "ever_late": int(ever_late_base.sum()),
                "detected_late": int(detected_late_base.sum())
            }
        }
    }

# Print results
if __name__ == "__main__":
    
    results = run_noscreening()
    
    cohort_df = results["no_screening"]["cohort"]
    deaths_cause_base_df = results["no_screening"]["deaths"]
    costs_no_screening_df = results["no_screening"]["costs"]
    qaly_no_screening_df = results["no_screening"]["qaly"]
    incidence = results["no_screening"]["incidence"]
    
    print("\n=== No Screening Simulation Complete ===")
    print(f"Total QALYs: {qaly_no_screening_df['Qaly'].sum():,.2f}")
    print(f"Total Cost: {costs_no_screening_df['Cost'].sum():,.0f} CHF")





