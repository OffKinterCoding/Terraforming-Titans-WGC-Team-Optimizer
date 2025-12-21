import numpy as np
import pulp # type: ignore
from enum import Enum

import tkinter as tk
from tkinter import ttk

class JOB(Enum):
    Soldier = 1
    Nat_Scientist = 2
    Soc_Scientist = 3
    
    @classmethod
    def toEnumOption(cls, stringJob):
        match stringJob:
            case "Soldier":
                return JOB.Soldier
            case "Natural Scientist":
                return JOB.Nat_Scientist
            case "Social Scientist":
                return JOB.Soc_Scientist
        return None

class HAZARD(Enum):
    Neutral =   [1, 1, 1, 1]
    Negotiation = [0.9, 1.1, 1, 1]
    Agressive = [1.25, 0.85, 1, 1]
    Recon =     [1, 0.85, 0.9, 1.25]

probability = {
    "100%": [1,4],
    "90%": [3,27],
    "80%": [5,32],
    "70%": [7,36],
    "60%": [9,39],
    "50%": [11,42]
}

def solve_maxmin(obst_lvl,
                 shoot_lvl,
                 lib_lvl,
                 skill_leader,
                 skill_other,
                 roll_indiv = 1,
                 roll_group = 4,
                 options = [JOB.Soldier, JOB.Nat_Scientist, JOB.Soc_Scientist],
                 hazard_approach = HAZARD.Neutral,
                 verbose=False,
                 get_integer_results = False):
    # Conditional branching
    use_leader_nat_sci = all(j != JOB.Nat_Scientist for j in options)
    use_leader_soc_sci = all(j != JOB.Soc_Scientist for j in options)
    nb_soldier = options.count(JOB.Soldier)
    haz_mod = hazard_approach._value_
    
    # Define the problem
    prob = pulp.LpProblem("MaxMinProblem", pulp.LpMaximize)
    
    # Decision variables
    x11 = pulp.LpVariable("x11", lowBound=1, cat="Integer")
    x12 = pulp.LpVariable("x12", lowBound=1, cat="Integer")
    x13 = pulp.LpVariable("x13", lowBound=1, cat="Integer")
    a1 = pulp.LpVariable("a1", lowBound=1, cat="Integer")
    a2 = pulp.LpVariable("a2", lowBound=1, cat="Integer")
    a3 = pulp.LpVariable("a3", lowBound=1, cat="Integer")
    t = pulp.LpVariable("t", cat="Continuous")  # max-min varValue
    
    # Auxiliary variables for min/max
    w0 = pulp.LpVariable("w0")
    w1 = pulp.LpVariable("w1")
    
    # y values
    y11 = 1.5* x11
    y21 = a1 + 0.5 * x11

    y12 = 1.5 * x12
    y22 = a2 + 0.5 * x12

    # Row-sum constraints
    prob += (x11 + x12 + x13 == skill_leader)
    prob += (a1 + a2 + a3 == skill_other)
    prob += w0 <= y11
    prob += w0 <= y21
    prob += w1 >= y12
    prob += w1 >= y22

    # z definitions (reduced & linearized)
    z0 = (w0 * shoot_lvl + roll_indiv - 10) / 1.5
    z1 = (w1 * obst_lvl + roll_indiv - (10*haz_mod[3])) / (1.5*haz_mod[3])
    z2 = 1.5 * x13 if use_leader_nat_sci else ((a3 + 0.5 * x13) * lib_lvl + roll_indiv - 10) / 1.5
    z3 = 1.5 * x13 if use_leader_soc_sci else ((a3 + 0.5 * x13) * lib_lvl + roll_indiv - (10*haz_mod[0])) / (1.5*haz_mod[0])

    z4 = (shoot_lvl * (x11 + (3 + nb_soldier) * a1) + roll_group - (40*haz_mod[1])) / (4*haz_mod[1])
    z5 = (obst_lvl * (x12 + 3 * a2) + roll_group - (40*haz_mod[3])) / (4*haz_mod[3])
    z6 = (lib_lvl * (x13 + (3 + (3 - nb_soldier) * 0.5) * a3) + roll_group - (40*haz_mod[2])) / (4*haz_mod[2])

    # Max–min constraints
    for z in [z0, z1, z2, z3, z4, z5, z6]:
        prob += t <= z

    # Objective
    prob += t

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if verbose:
        print(f"Conditions: {options}; {hazard_approach}")
        print("status:", pulp.LpStatus[prob.status])
        print("t:", t.varValue)
        print("x1:", x11.varValue, x12.varValue, x13.varValue)
        print("a :", a1.varValue, a2.varValue, a3.varValue)
    
    
    x1_vals = np.array([x11.varValue, x12.varValue, x13.varValue])
    a_vals = np.array([a1.varValue, a2.varValue, a3.varValue])
    z_values = {
        "z0": ((a_vals[0] + 0.5 * x1_vals[0]) * shoot_lvl + roll_indiv - 10) / 1.5,
        "z1": ((a_vals[1] + 0.5 * x1_vals[1]) * obst_lvl + roll_indiv - (10 * haz_mod[3])) / (1.5 * haz_mod[3]),
        "z2": 1.5 * x1_vals[2] if use_leader_nat_sci else ((a_vals[2] + 0.5 * x1_vals[2]) * lib_lvl + roll_indiv - 10) / 1.5,
        "z3": 1.5 * x1_vals[2] if use_leader_soc_sci else ((a_vals[2] + 0.5 * x1_vals[2]) * lib_lvl + roll_indiv - (10 * haz_mod[0])) / (1.5 * haz_mod[0]),
        "z4": (shoot_lvl * (x1_vals[0] + (3 + nb_soldier) * a_vals[0]) + roll_group - (40 * haz_mod[1])) / (4 * haz_mod[1]),
        "z5": (obst_lvl * (x1_vals[1] + 3 * a_vals[1]) + roll_group - (40 * haz_mod[3])) / (4 * haz_mod[3]),
        "z6": (lib_lvl * (x1_vals[2] + (3 + (3 - nb_soldier) * 0.5) * a_vals[2]) + roll_group - (40 * haz_mod[2])) / (4 * haz_mod[2])
    }
    if get_integer_results:
        z_values = {k: int(v) for k,v in z_values.items()}
    
    return {
        "t": t.varValue,
        "x1": x1_vals.astype(int),
        "a": a_vals.astype(int),
        "z": z_values
    }

def convert_to_skill_point(points, is_leader):
    modifier = 8 if is_leader else 7
    return (points-1) * 2 + modifier

# p = "100%"
# ri, rg = probability[p]

# all_jobs = list(JOB)
# # scorecard = {f"{ji}, {jj}, {jk}": 0
# #              for ji in range(3)
# #              for jj in range(ji, 3)
# #              for jk in range(jj, 3)}
# best_distr = [-100]
# for haz in HAZARD:
#     for i in range(3):
#         potential_job_1 = all_jobs[i]
#         for j in range(i, 3):
#             potential_job_2 = all_jobs[j]
#             for k in range(j, 3):
#                 potential_job_3 = all_jobs[k]
#                 option = [potential_job_1, potential_job_2, potential_job_3]
#                 #print(option)
#                 res = solve_maxmin(obst_lvl=1.58, shoot_lvl=1.11, lib_lvl=1.01, skill_leader=convert_to_skill_point(level, True), skill_other=convert_to_skill_point(level, True), roll_indiv=ri, roll_group=rg, options=option, hazard_approach=haz, verbose=False, get_integer_results = True)
#                 #print()
                
#                 if res["t"] > best_distr[0]:
#                     best_distr = [res["t"], res, option, haz, [i,j,k]]
#                 #print(res)

# res_best = best_distr[1]
# # i_fin,j_fin,k_fin = best_distr[4]
# # scorecard[f"{i_fin}, {j_fin}, {k_fin}"] += 1

# print(f"{level}: Individual roll +{ri}; Group roll +{rg}")
# print(f"Minimum level to get {p}% success: {int(res_best["t"])}", end="\t\t")
# print("\t\tZ-results:", end="\t")
# for z_stat in res_best["z"]:
#     print(f"{res_best["z"][z_stat]}", end=", ")
# print("\nLeader stats:", end="\t\t")
# for leader_stat in res_best["x1"]:
#     print(f"{leader_stat}", end=", ")
# print("\t\t\t\tOthers' stats:", end="\t\t")
# for others_stat in res_best["a"]:
#     print(f"{others_stat}", end=", ")
# print()
# print("Respective jobs:", end="\t")
# for jobs_to_do in best_distr[2]:
#     print(f"{jobs_to_do.name}", end=", ")
# print(f"\tHazard approach:\t{best_distr[3].name}")
# print("\n")

# for job_key in scorecard:
#     print(f"{job_key}: {scorecard[job_key]}")

# print(f"""{p}%: Individual roll +{ri}; Group roll +{rg}
# Max level: {int(res_best["t"])}
# Leader stats: {res_best["x1"]}
# Others' stats: {res_best["a"]}
# Jobs: {best_distr[2]}
# Hazard approach: {best_distr[3]}
# Full results: {res_best["z"]}""")

################################
######### Global  Vars #########
################################
col_width = [10,15,10,10,10,10,10]

# Slot rows
class_options = ["Soldier", "Natural Scientist", "Social Scientist"]
class_vars = []
lvl_vars = []

################################
########## Functions ###########
################################

def validate_user_inputs():
    global class_vars, lvl_vars, top_inputs, err_msg
    users_info = {"Job1":"", "Job2": "", "Job3": "", "LeaderLvl": "", "OthersLvl": "", "Shoot": "", "Obst": "", "Lib": "", "Success": ""}
    #print(f"{type(class_vars)} {type(lvl_vars)} {type(top_inputs)}")
    all_vars = class_vars + lvl_vars + top_inputs
    
    for var, name in zip(all_vars, users_info.keys()):
        value = var.get()
        #print(f"{name}: {value}")

        # adjust conditions as needed
        if value is None or value == "":
           err_msg.set("Not all inputs filled.")
           return {} # stop immediately if anything is unset

        users_info[name] = value
    
    err_msg.set("")
    return users_info
    

def calculate_and_set():
    users_info = validate_user_inputs()
    if users_info == {}:
        return
    
    option = [JOB.toEnumOption(users_info["Job1"]), JOB.toEnumOption(users_info["Job2"]), JOB.toEnumOption(users_info["Job3"])]
    leaderLvl = convert_to_skill_point(int(users_info["LeaderLvl"]), True)
    othersLvl = convert_to_skill_point(int(users_info["OthersLvl"]), False)
    obstLvl = 1 + int(users_info["Obst"])/100
    shotLvl = 1 + int(users_info["Shoot"])/100
    librLvl = 1 + int(users_info["Lib"])/100
    ri, rg = probability[users_info["Success"]]
    
    best_distr = [-100]
    for haz in HAZARD:
        results = solve_maxmin(obst_lvl=obstLvl,
                               shoot_lvl=shotLvl,
                               lib_lvl=librLvl,
                               skill_leader=leaderLvl,
                               skill_other=othersLvl,
                               roll_indiv=ri,
                               roll_group=rg,
                               options=option,
                               hazard_approach=haz,
                               verbose=False,
                               get_integer_results = True)
        
        if results["t"] > best_distr[0]:
            best_distr = [results["t"], results, option, haz]
    
    res_best = best_distr[1]
    leader_power_var.set(res_best["x1"][0])
    leader_ath_var.set(res_best["x1"][1])
    leader_wit_var.set(res_best["x1"][2])
    
    others_power_var.set(res_best["a"][0])
    others_ath_var.set(res_best["a"][1])
    others_wit_var.set(res_best["a"][2])
    
    hazard_var.set(best_distr[3].name)
    
    max_level.set(int(res_best["t"]))
    
    
def only_digits(P):
    return P.isdigit() or P == ""   # allow empty (for backspace)

def create_entry(row, col, var, rowspan=1, sticky=""):
    entry = tk.Entry(root, textvariable=var, width=col_width[col],
                     justify="center", state="disabled",
                     disabledforeground="gray40", disabledbackground="gray80",
                     cursor="arrow")
    entry.grid(row=row, column=col, padx=5, pady=5, rowspan=rowspan, sticky=sticky)

def create_level_entry(row, lvl_var, rowspan=1, sticky=""):
    """Helper function to create a level entry."""
    lvl_entry = tk.Entry(root, textvariable=lvl_var, width=col_width[2], justify="center", validate="key", validatecommand=(vcmd, "%P"))
    lvl_entry.grid(row=row, column=2, padx=5, pady=5, rowspan=rowspan, sticky=sticky)
    lvl_vars.append(lvl_var)
    
def close_window(event):
    root.destroy()
    
################################
######### User Section #########
################################

root = tk.Tk()
root.title("Team Optimizer")
root.configure(bg="#f0f0f0")
vcmd = root.register(only_digits)

# Top labels
labels_top = ["Shooting\nRange Level", "Obstacle\nCourse Level", "Library\nLevel", "Success\nChance", "", "Hazard\nApproach", "Max Level"]
for col, text in enumerate(labels_top):
    tk.Label(root, text=text, font=("Arial", 10, "bold"), width=col_width[col]).grid(row=0, column=col, padx=5, pady=5)

# Second row: user inputs and dropdown
shoot_entry = tk.Entry(root, width=col_width[0], justify="center", validate="key", validatecommand=(vcmd, "%P"))
shoot_entry.grid(row=1, column=0, padx=5, pady=5)
obstacle_entry = tk.Entry(root, width=col_width[1], justify="center", validate="key", validatecommand=(vcmd, "%P"))
obstacle_entry.grid(row=1, column=1, padx=5, pady=5)
library_entry = tk.Entry(root, width=col_width[2], justify="center", validate="key", validatecommand=(vcmd, "%P"))
library_entry.grid(row=1, column=2, padx=5, pady=5)

success_var = tk.StringVar()
success_dropdown = ttk.Combobox(root, textvariable=success_var, width=col_width[3],
                                values=[str(i) + "%" for i in range(100,49,-10)], state="readonly")
success_dropdown.grid(row=1, column=3, padx=5, pady=5)

top_inputs = [shoot_entry, obstacle_entry, library_entry, success_var]

hazard_var = tk.StringVar()
create_entry(1, 5, hazard_var)

max_level = tk.StringVar()
create_entry(1, 6, max_level)

# Third row: Titles
labels_top = ["Slot", "Job", "Level", "", "Power", "Athletics", "Wit"]
for col, text in enumerate(labels_top):
    tk.Label(root, text=text, font=("Arial", 10, "bold"), width=col_width[col]).grid(row=2, column=col, padx=5, pady=5)

for i in range(4):
    # Slot numbers
    slot_label = tk.StringVar()
    slot_label.set(i+1)
    class_label = tk.Label(root, textvariable=slot_label, width=col_width[0])
    class_label.grid(row=i+3, column=0, padx=5, pady=5)
    #class_vars.append(slot_label)
    
    # Class dropdown
    c_var = tk.StringVar()
    if i == 0:
        c_var.set("Leader")
        class_label = tk.Label(root, textvariable=c_var, width=col_width[1], justify="center")
        class_label.grid(row=3, column=1, padx=5, pady=5)
    else:
        class_dropdown = ttk.Combobox(root, textvariable=c_var, values=class_options, width=col_width[1], justify="center", state="readonly")
        class_dropdown.grid(row=i+3, column=1, padx=5, pady=5)
        class_vars.append(c_var)

################################
### PAW that gets calculated ###
################################

# Power, Athlete, Wit (calculated)
leader_power_var = tk.StringVar()
leader_ath_var = tk.StringVar()
leader_wit_var = tk.StringVar()
create_entry(3, 4, leader_power_var)
create_entry(3, 5, leader_ath_var)
create_entry(3, 6, leader_wit_var)

# Power, Athlete, Wit (calculated)
others_power_var = tk.StringVar()
others_ath_var = tk.StringVar()
others_wit_var = tk.StringVar()
create_entry(4, 4, others_power_var, rowspan=3, sticky="ns")
create_entry(4, 5, others_ath_var, rowspan=3, sticky="ns")
create_entry(4, 6, others_wit_var, rowspan=3, sticky="ns")

################################
################################
################################

leader_lvl_var = tk.StringVar()
others_lvl_var = tk.StringVar()
create_level_entry(3, leader_lvl_var)  # Leader level
create_level_entry(4, others_lvl_var, rowspan=3, sticky="ns")  # Others' level

# Buttons
root.bind("<Return>", lambda event: calculate_and_set())
tk.Button(root, text="Calculate", command=calculate_and_set).grid(row=7, column=1, pady=10)

root.bind('<Escape>', close_window)
tk.Button(root, text="Quit", command=root.destroy).grid(row=7, column=5, pady=10)

# Error message
err_msg = tk.StringVar()
err_msg.set("")
err_msg_label = tk.Label(root, textvariable=err_msg, justify="center", fg="#DB2121")
err_msg_label.grid(row=7, column=2, columnspan=3, padx=5, pady=5)

root.mainloop()