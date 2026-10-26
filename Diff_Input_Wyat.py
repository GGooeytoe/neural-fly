# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 19:31:19 2026

@author: WyatB
"""

import os
import csv
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.modules.loss
from itertools import combinations
from sklearn.model_selection import train_test_split
import gemini_train_pi_groups
import utils

# ---------------------------------------------------------------------
# 0. Module Monkey-Patches
# ---------------------------------------------------------------------
# Reset MSELoss to standard PyTorch class
gemini_train_pi_groups.nn.MSELoss = torch.nn.modules.loss.MSELoss

# Bypass hardcoded 7D slicing inside NonDimFFNN forward pass
gemini_train_pi_groups.NonDimFFNN.forward = lambda self, x: self.net(x)

# Configuration
epochs = 100
random_seed = 42
train_folder = "./data/training"
test_folder = "./data/training-transfer"

# Hardware parameters
r_train = gemini_train_pi_groups.ROTOR_RADIUS["neural-fly"]
pwm_train = gemini_train_pi_groups.PWM_HOVER["neural-fly"]

r_test = gemini_train_pi_groups.ROTOR_RADIUS["intel"]
pwm_test = gemini_train_pi_groups.PWM_HOVER["intel"]

# ---------------------------------------------------------------------
# 1. Map 8 Grouped Raw Variables (23 Dimensions Total)
# ---------------------------------------------------------------------
# Variables kept intact as vector groups:
# p:3D, v:3D, q:4D, w:3D, T_sp:1D, q_sp:4D, hover_throttle:1D, pwm:4D
newvar = ['p', 'v', 'q', 'w', 'T_sp', 'q_sp', 'hover_throttle', 'pwm']
vardim = [3,   3,   4,   3,   1,      4,      1,               4]  # Total = 23D

var_indices = {}
curr_idx = 0
for var, dim in zip(newvar, vardim):
    var_indices[var] = list(range(curr_idx, curr_idx + dim))
    curr_idx += dim

# ---------------------------------------------------------------------
# 2. Extract Raw 23D Datasets
# ---------------------------------------------------------------------
def load_raw_23d_dataset(data_folder):
    """Loads raw 23D input features directly from experiment dictionaries."""
    raw_data = utils.load_data(folder=data_folder)
    X_list, Y_list = [], []

    for exp in raw_data:
        num_timesteps = len(exp['t'])
        for t in range(num_timesteps):
            p = exp.get('p', np.zeros((num_timesteps, 3)))[t] if 'p' in exp else np.zeros(3)
            v = exp['v'][t] if 'v' in exp else np.zeros(3)
            q = exp.get('q', np.zeros((num_timesteps, 4)))[t] if 'q' in exp else np.zeros(4)
            w = exp.get('w', np.zeros((num_timesteps, 3)))[t] if 'w' in exp else np.zeros(3)
            T_sp = np.atleast_1d(exp.get('T_sp', np.zeros((num_timesteps, 1)))[t])
            q_sp = exp.get('q_sp', np.zeros((num_timesteps, 4)))[t] if 'q_sp' in exp else np.zeros(4)
            hover_th = np.atleast_1d(exp.get('hover_throttle', np.zeros((num_timesteps, 1)))[t])
            pwm = exp['pwm'][t] if 'pwm' in exp else np.zeros(4)

            fa = exp['fa'][t]

            x_23d = np.concatenate([p, v, q, w, T_sp, q_sp, hover_th, pwm])
            X_list.append(x_23d)
            Y_list.append(fa)

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)

print("Loading raw 23D datasets...")
X_all_full, Y_raw_all = load_raw_23d_dataset(train_folder)

idx_train, idx_val = train_test_split(
    np.arange(len(X_all_full)), test_size=0.2, random_state=random_seed
)
Y_train, Y_val = Y_raw_all[idx_train], Y_raw_all[idx_val]

X_test_full, Y_test = load_raw_23d_dataset(test_folder)

assert X_all_full.shape[1] == 23, f"Expected 23D raw features, got {X_all_full.shape[1]}D"

# ---------------------------------------------------------------------
# 3. Helper Evaluation Function
# ---------------------------------------------------------------------
def evaluate_split(model, scaler_X, scaler_label, X_raw, Y_raw):
    """Evaluates physical force predictions on a dataset split."""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_scaled = scaler_X.transform(X_raw)
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        pred_scaled = model(X_tensor).cpu().numpy()
        pred_physical = scaler_label.inverse_transform(pred_scaled)

    error = pred_physical - Y_raw
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    mae = np.mean(np.abs(error), axis=0)

    eps = 1e-5
    mape = np.mean(np.abs(error / (np.abs(Y_raw) + eps)), axis=0) * 100.0
    nrmse = (rmse / (np.ptp(Y_raw, axis=0) + eps)) * 100.0  # Range-Normalized RMSE %

    return rmse, mae, mape, nrmse

orig_loader = gemini_train_pi_groups.load_and_nondimensionalize_data

# ---------------------------------------------------------------------
# 4. Generate All 255 Combinations (2^8 - 1)
# ---------------------------------------------------------------------
experiments = {}

for r in range(1, len(newvar) + 1):
    for combo in combinations(newvar, r):
        experiments[f"sub_{r}_" + "_".join(combo)] = list(combo)

# ---------------------------------------------------------------------
# 5. Execution Loop with Real-Time Saving & Failure Handling
# ---------------------------------------------------------------------
summary_csv_path = "raw_23d_grid_search_summary.csv"
summary_txt_path = "experiment_23d_log.txt"

fieldnames = [
    "Experiment", "Input_Variables", "Num_Features", "Status",
    "Train_RMSE_Fx", "Train_RMSE_Fy", "Train_RMSE_Fz",
    "Val_RMSE_Fx",   "Val_RMSE_Fy",   "Val_RMSE_Fz",
    "Test_RMSE_Fx",  "Test_RMSE_Fy",  "Test_RMSE_Fz",
    "Test_MAE_Fx",   "Test_MAE_Fy",   "Test_MAE_Fz",
    "Test_NRMSE_Fx_%", "Test_NRMSE_Fy_%", "Test_NRMSE_Fz_%"
]

if not os.path.exists(summary_csv_path):
    with open(summary_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def log_message(msg):
    print(msg)
    with open(summary_txt_path, "a") as txtfile:
        txtfile.write(msg + "\n")
        txtfile.flush()

log_message("\n" + "=" * 110)
log_message(f"STARTING GRID SEARCH ({len(experiments)} Combinations)")
log_message("=" * 110)

total_runs = len(experiments)

for run_idx, (exp_name, var_list) in enumerate(experiments.items(), start=1):
    selected_cols = []
    for v in var_list:
        selected_cols.extend(var_indices[v])

    header_msg = (
        f"\n[{run_idx}/{total_runs}] Running: {exp_name}\n"
        f"  Variables Included: {var_list}\n"
        f"  Selected Feature Columns ({len(selected_cols)}D): {selected_cols}"
    )
    log_message(header_msg)

   # Monkey-patch loader to return raw sliced X matrix with dummy C_f and Q arrays
    def sliced_raw_loader(r, pwm, folder):
        X_raw, Y_raw = load_raw_23d_dataset(folder)
        dummy_Q = np.ones((len(X_raw), 3), dtype=np.float32)
        dummy_Cf = np.ones_like(Y_raw, dtype=np.float32)
        return X_raw[:, selected_cols], dummy_Cf, dummy_Q, Y_raw

    gemini_train_pi_groups.load_and_nondimensionalize_data = sliced_raw_loader

    # Slice evaluation datasets
    X_train_sub = X_all_full[idx_train][:, selected_cols]
    X_val_sub = X_all_full[idx_val][:, selected_cols]
    X_test_sub = X_test_full[:, selected_cols]

    try:
        model_mse, scaler_X_mse, scaler_Y_mse, _ = gemini_train_pi_groups.train_model(
            rotor_radius=r_train,
            pwm_hover=pwm_train,
            save_folder_prefix=f"raw_input_{exp_name}_{epochs}ep",
            data_folder=train_folder,
            epochs=epochs,
            random_seed=random_seed
        )

        metrics = {
            "train": evaluate_split(model_mse, scaler_X_mse, scaler_Y_mse, X_train_sub, Y_train),
            "val":   evaluate_split(model_mse, scaler_X_mse, scaler_Y_mse, X_val_sub, Y_val),
            "test":  evaluate_split(model_mse, scaler_X_mse, scaler_Y_mse, X_test_sub, Y_test)
        }

        rmse_tr, mae_tr, mape_tr, nrmse_tr = metrics["train"]
        rmse_v,  mae_v,  mape_v,  nrmse_v  = metrics["val"]
        rmse_te, mae_te, mape_te, nrmse_te = metrics["test"]

        log_message(
            f"  [SUCCESS] {exp_name}\n"
            f"  Test RMSE [N]:  Fx={rmse_te[0]:.3f}, Fy={rmse_te[1]:.3f}, Fz={rmse_te[2]:.3f}\n"
            f"  Test NRMSE [%]: Fx={nrmse_te[0]:.1f}%, Fy={nrmse_te[1]:.1f}%, Fz={nrmse_te[2]:.1f}%"
        )

        row_dict = {
            "Experiment": exp_name,
            "Input_Variables": ", ".join(var_list),
            "Num_Features": len(selected_cols),
            "Status": "SUCCESS",
            "Train_RMSE_Fx": rmse_tr[0], "Train_RMSE_Fy": rmse_tr[1], "Train_RMSE_Fz": rmse_tr[2],
            "Val_RMSE_Fx":   rmse_v[0],  "Val_RMSE_Fy":   rmse_v[1],  "Val_RMSE_Fz":   rmse_v[2],
            "Test_RMSE_Fx":  rmse_te[0], "Test_RMSE_Fy":  rmse_te[1], "Test_RMSE_Fz":  rmse_te[2],
            "Test_MAE_Fx":   mae_te[0],  "Test_MAE_Fy":   mae_te[1],  "Test_MAE_Fz":   mae_te[2],
            "Test_NRMSE_Fx_%": nrmse_te[0], "Test_NRMSE_Fy_%": nrmse_te[1], "Test_NRMSE_Fz_%": nrmse_te[2]
        }

    except Exception as err:
        error_details = traceback.format_exc()
        log_message(f"  [FAILED] {exp_name}\n  Error: {str(err)}\n{error_details}")

        row_dict = {
            "Experiment": exp_name,
            "Input_Variables": ", ".join(var_list),
            "Num_Features": len(selected_cols),
            "Status": "FAILED",
            "Train_RMSE_Fx": "", "Train_RMSE_Fy": "", "Train_RMSE_Fz": "",
            "Val_RMSE_Fx": "", "Val_RMSE_Fy": "", "Val_RMSE_Fz": "",
            "Test_RMSE_Fx": "", "Test_RMSE_Fy": "", "Test_RMSE_Fz": "",
            "Test_MAE_Fx": "", "Test_MAE_Fy": "", "Test_MAE_Fz": "",
            "Test_NRMSE_Fx_%": "", "Test_NRMSE_Fy_%": "", "Test_NRMSE_Fz_%": ""
        }

    # Write & flush CSV row to disk immediately after each run
    with open(summary_csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row_dict)
        csvfile.flush()

gemini_train_pi_groups.load_and_nondimensionalize_data = orig_loader

log_message("\n" + "=" * 110)
log_message("GRID SEARCH COMPLETE")
log_message(f"Full log saved to: {summary_txt_path}")
log_message(f"CSV datasheet saved to: {summary_csv_path}")
log_message("=" * 110)