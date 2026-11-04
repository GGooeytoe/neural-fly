import os
import re
import ast
import glob
import csv
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.modules.loss
from itertools import combinations
from sklearn.model_selection import train_test_split
import gemini_train_pi_groups
import utils

# ---------------------------------------------------------------------
# 0. Dynamic Wind Speed Calculator & Helper Functions
# ---------------------------------------------------------------------
WIND_SPEED_MAP = {
    'nowind': 0.0,
    '10wind': 1.3,
    '20wind': 2.5,
    '30wind': 3.7,
    '35wind': 4.2,
    '40wind': 4.9,
    '50wind': 6.1,
    '70wind': 8.5,
    '100wind': 12.1
}

def parse_wind_speed(condition_str: str, time_sec: float = 0.0) -> float:
    """
    Computes instantaneous wind speed magnitude (m/s) based on condition string and timestamp.
    Handles dynamic time-varying wind profiles like 70p20sint = 8.5 + 2.4*sin(t).
    """
    cond_lower = str(condition_str).lower()
    
    # 1. Time-varying sinusoidal wind condition
    if '70p20sint' in cond_lower:
        return 8.5 + 2.4 * np.sin(time_sec)

    # 2. Constant wind map lookup
    for key, val in WIND_SPEED_MAP.items():
        if key in cond_lower:
            return val

    # 3. Fallback regex for numeric pattern (e.g., '10wind')
    match = re.search(r'(\d+)wind', cond_lower)
    if match:
        return float(match.group(1)) * 0.121

    return 0.0

def parse_vec(val):
    if isinstance(val, str):
        return np.array(ast.literal_eval(val), dtype=np.float32)
    return np.array(val, dtype=np.float32)

# Reset MSELoss to standard PyTorch class
gemini_train_pi_groups.nn.MSELoss = torch.nn.modules.loss.MSELoss

# Bypass hardcoded 7D slicing inside NonDimFFNN forward pass
gemini_train_pi_groups.NonDimFFNN.forward = lambda self, x: self.net(x)

# Configuration & Absolute Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
epochs = 100
random_seed = 42

train_folder = os.path.join(SCRIPT_DIR, "data", "training")
test_folder = os.path.join(SCRIPT_DIR, "data", "experiment")

# Hardware parameters
r_train = gemini_train_pi_groups.ROTOR_RADIUS["neural-fly"]
pwm_train = gemini_train_pi_groups.PWM_HOVER["neural-fly"]

r_test = gemini_train_pi_groups.ROTOR_RADIUS["intel"]
pwm_test = gemini_train_pi_groups.PWM_HOVER["intel"]

# ---------------------------------------------------------------------
# 1. Map 9 Grouped Raw Variables (26 Dimensions Total)
# ---------------------------------------------------------------------
# p:3D, v:3D, q:4D, w_ang:3D, w_wind:3D, T_sp:1D, q_sp:4D, hover_throttle:1D, pwm:4D
newvar = ['p', 'v', 'q', 'w_ang', 'w_wind', 'T_sp', 'q_sp', 'hover_throttle', 'pwm']
vardim = [3,   3,   4,   3,       3,        1,      4,      1,              4]  # Total = 26D

var_indices = {}
curr_idx = 0
for var, dim in zip(newvar, vardim):
    var_indices[var] = list(range(curr_idx, curr_idx + dim))
    curr_idx += dim

# ---------------------------------------------------------------------
# 2. Extract Raw 26D Datasets Directly From CSV Files
# ---------------------------------------------------------------------
def load_raw_26d_dataset(data_folder, file_filter=None):
    """
    Scans data_folder for CSV files, filters by filename string (e.g. 'baseline'),
    and extracts 26D raw feature inputs.
    """
    abs_folder = os.path.abspath(data_folder)
    print(f"Scanning folder: '{abs_folder}'")

    all_csvs = glob.glob(os.path.join(abs_folder, "*.csv")) + \
               glob.glob(os.path.join(abs_folder, "**", "*.csv"), recursive=True)
    
    # Exclude summary output CSV files
    csv_files = sorted(list(set([f for f in all_csvs if not f.endswith("_summary.csv")])))

    if file_filter:
        csv_files = [f for f in csv_files if file_filter.lower() in os.path.basename(f).lower()]

    print(f"Found {len(csv_files)} matching CSV experiment file(s) in '{abs_folder}'.")
    for f in csv_files:
        print(f"  -> Loading: {os.path.basename(f)}")

    X_list, Y_list = [], []

    for fpath in csv_files:
        filename = os.path.basename(fpath)
        df = pd.read_csv(fpath)
        
        timestamps = df['t'].values if 't' in df.columns else np.arange(len(df))
        num_timesteps = len(df)

        for i in range(num_timesteps):
            t_sec = timestamps[i]
            
            # Compute instantaneous wind magnitude & world vector [-v_wind(t), 0, 0]
            v_wind_mag = parse_wind_speed(filename, time_sec=t_sec)
            w_world_vec = np.array([-v_wind_mag, 0.0, 0.0], dtype=np.float32)

            # Parse rotation matrix R
            if 'R' in df.columns:
                R_val = parse_vec(df['R'].iloc[i])
                R_mat = R_val.reshape(3, 3) if R_val.size == 9 else np.eye(3, dtype=np.float32)
            else:
                R_mat = np.eye(3, dtype=np.float32)
            R_T = R_mat.T  # World -> Body rotation

            # Extract 9 variable groups
            p = parse_vec(df['p'].iloc[i]) if 'p' in df.columns else np.zeros(3, dtype=np.float32)
            v = parse_vec(df['v'].iloc[i]) if 'v' in df.columns else np.zeros(3, dtype=np.float32)
            q = parse_vec(df['q'].iloc[i]) if 'q' in df.columns else np.zeros(4, dtype=np.float32)
            w_ang = parse_vec(df['w'].iloc[i]) if 'w' in df.columns else np.zeros(3, dtype=np.float32)
            
            # Local body-frame wind vector: R^T * w_world(t)
            w_wind_local = R_T @ w_world_vec

            T_sp = np.atleast_1d(parse_vec(df['T_sp'].iloc[i])) if 'T_sp' in df.columns else np.zeros(1, dtype=np.float32)
            q_sp = parse_vec(df['q_sp'].iloc[i]) if 'q_sp' in df.columns else np.zeros(4, dtype=np.float32)
            hover_th = np.atleast_1d(parse_vec(df['hover_throttle'].iloc[i])) if 'hover_throttle' in df.columns else np.zeros(1, dtype=np.float32)
            pwm = parse_vec(df['pwm'].iloc[i]) if 'pwm' in df.columns else np.zeros(4, dtype=np.float32)

            fa = parse_vec(df['fa'].iloc[i]) if 'fa' in df.columns else np.zeros(3, dtype=np.float32)

            x_26d = np.concatenate([p, v, q, w_ang, w_wind_local, T_sp, q_sp, hover_th, pwm])
            X_list.append(x_26d)
            Y_list.append(fa)

    if len(X_list) == 0:
        print(f"[WARNING] 0 samples loaded for filter '{file_filter}' in '{abs_folder}'!")
        return np.empty((0, 26), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


# Load training data and create train/val split
print("\nLoading raw 26D training dataset...")
X_all_full, Y_raw_all = load_raw_26d_dataset(train_folder)

idx_train, idx_val = train_test_split(
    np.arange(len(X_all_full)), test_size=0.2, random_state=random_seed
)
Y_train, Y_val = Y_raw_all[idx_train], Y_raw_all[idx_val]

# Load baseline test dataset from data/experiment
print("\nLoading raw 26D baseline test dataset from 'data/experiment'...")
X_test_full, Y_test = load_raw_26d_dataset(test_folder, file_filter="baseline")

assert X_all_full.shape[1] == 26, f"Expected 26D raw features, got {X_all_full.shape[1]}D"
print(f"\nSuccessfully loaded {len(X_all_full)} training samples and {len(X_test_full)} baseline test samples.\n")

# ---------------------------------------------------------------------
# 3. Comprehensive Metric Evaluation Function
# ---------------------------------------------------------------------
def evaluate_split(model, scaler_X, scaler_label, X_raw, Y_raw):
    """Evaluates physical force predictions and calculates RMSE, MAE, MAPE %, NRMSE %."""
    if len(X_raw) == 0:
        return {
            "rmse": np.array([np.nan, np.nan, np.nan]),
            "mae": np.array([np.nan, np.nan, np.nan]),
            "mape": np.array([np.nan, np.nan, np.nan]),
            "nrmse": np.array([np.nan, np.nan, np.nan])
        }

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
    nrmse = (rmse / (np.ptp(Y_raw, axis=0) + eps)) * 100.0

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "nrmse": nrmse
    }

orig_loader = gemini_train_pi_groups.load_and_nondimensionalize_data

# ---------------------------------------------------------------------
# 4. Generate All 511 Combinations (2^9 - 1)
# ---------------------------------------------------------------------
experiments = {}

for r in range(1, len(newvar) + 1):
    for combo in combinations(newvar, r):
        experiments[f"sub_{r}_" + "_".join(combo)] = list(combo)

# ---------------------------------------------------------------------
# 5. Execution Loop with Complete Metrics across Train, Val, and Baseline Test
# ---------------------------------------------------------------------
summary_csv_path = "raw_26d_grid_search_summary.csv"
summary_txt_path = "experiment_26d_log.txt"

fieldnames = [
    "Experiment", "Input_Variables", "Num_Features", "Status",
    # Train Metrics
    "Train_RMSE_Fx", "Train_RMSE_Fy", "Train_RMSE_Fz",
    "Train_MAE_Fx", "Train_MAE_Fy", "Train_MAE_Fz",
    "Train_MAPE_Fx_%", "Train_MAPE_Fy_%", "Train_MAPE_Fz_%",
    "Train_NRMSE_Fx_%", "Train_NRMSE_Fy_%", "Train_NRMSE_Fz_%",
    # Validation Metrics
    "Val_RMSE_Fx", "Val_RMSE_Fy", "Val_RMSE_Fz",
    "Val_MAE_Fx", "Val_MAE_Fy", "Val_MAE_Fz",
    "Val_MAPE_Fx_%", "Val_MAPE_Fy_%", "Val_MAPE_Fz_%",
    "Val_NRMSE_Fx_%", "Val_NRMSE_Fy_%", "Val_NRMSE_Fz_%",
    # Baseline Test Metrics (from data/experiment)
    "Test_RMSE_Fx", "Test_RMSE_Fy", "Test_RMSE_Fz",
    "Test_MAE_Fx", "Test_MAE_Fy", "Test_MAE_Fz",
    "Test_MAPE_Fx_%", "Test_MAPE_Fy_%", "Test_MAPE_Fz_%",
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
        is_test = (folder == test_folder)
        filter_str = "baseline" if is_test else None
        X_raw, Y_raw = load_raw_26d_dataset(folder, file_filter=filter_str)
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

        tr = evaluate_split(model_mse, scaler_X_mse, scaler_Y_mse, X_train_sub, Y_train)
        va = evaluate_split(model_mse, scaler_X_mse, scaler_Y_mse, X_val_sub, Y_val)
        te = evaluate_split(model_mse, scaler_X_mse, scaler_Y_mse, X_test_sub, Y_test)

        log_message(
            f"  [SUCCESS] {exp_name}\n"
            f"  Train NRMSE [%]: Fx={tr['nrmse'][0]:.1f}%, Fy={tr['nrmse'][1]:.1f}%, Fz={tr['nrmse'][2]:.1f}%\n"
            f"  Val   NRMSE [%]: Fx={va['nrmse'][0]:.1f}%, Fy={va['nrmse'][1]:.1f}%, Fz={va['nrmse'][2]:.1f}%\n"
            f"  Test  NRMSE [%]: Fx={te['nrmse'][0]:.1f}%, Fy={te['nrmse'][1]:.1f}%, Fz={te['nrmse'][2]:.1f}%"
        )

        row_dict = {
            "Experiment": exp_name,
            "Input_Variables": ", ".join(var_list),
            "Num_Features": len(selected_cols),
            "Status": "SUCCESS",
            # Train
            "Train_RMSE_Fx": tr["rmse"][0], "Train_RMSE_Fy": tr["rmse"][1], "Train_RMSE_Fz": tr["rmse"][2],
            "Train_MAE_Fx": tr["mae"][0], "Train_MAE_Fy": tr["mae"][1], "Train_MAE_Fz": tr["mae"][2],
            "Train_MAPE_Fx_%": tr["mape"][0], "Train_MAPE_Fy_%": tr["mape"][1], "Train_MAPE_Fz_%": tr["mape"][2],
            "Train_NRMSE_Fx_%": tr["nrmse"][0], "Train_NRMSE_Fy_%": tr["nrmse"][1], "Train_NRMSE_Fz_%": tr["nrmse"][2],
            # Validation
            "Val_RMSE_Fx": va["rmse"][0], "Val_RMSE_Fy": va["rmse"][1], "Val_RMSE_Fz": va["rmse"][2],
            "Val_MAE_Fx": va["mae"][0], "Val_MAE_Fy": va["mae"][1], "Val_MAE_Fz": va["mae"][2],
            "Val_MAPE_Fx_%": va["mape"][0], "Val_MAPE_Fy_%": va["mape"][1], "Val_MAPE_Fz_%": va["mape"][2],
            "Val_NRMSE_Fx_%": va["nrmse"][0], "Val_NRMSE_Fy_%": va["nrmse"][1], "Val_NRMSE_Fz_%": va["nrmse"][2],
            # Test (data/experiment baseline files)
            "Test_RMSE_Fx": te["rmse"][0], "Test_RMSE_Fy": te["rmse"][1], "Test_RMSE_Fz": te["rmse"][2],
            "Test_MAE_Fx": te["mae"][0], "Test_MAE_Fy": te["mae"][1], "Test_MAE_Fz": te["mae"][2],
            "Test_MAPE_Fx_%": te["mape"][0], "Test_MAPE_Fy_%": te["mape"][1], "Test_MAPE_Fz_%": te["mape"][2],
            "Test_NRMSE_Fx_%": te["nrmse"][0], "Test_NRMSE_Fy_%": te["nrmse"][1], "Test_NRMSE_Fz_%": te["nrmse"][2],
        }

    except Exception as err:
        error_details = traceback.format_exc()
        log_message(f"  [FAILED] {exp_name}\n  Error: {str(err)}\n{error_details}")

        row_dict = {
            "Experiment": exp_name,
            "Input_Variables": ", ".join(var_list),
            "Num_Features": len(selected_cols),
            "Status": "FAILED",
            **{k: "" for k in fieldnames[4:]}
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