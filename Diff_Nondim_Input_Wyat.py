# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:54:35 2026

@author: WyatB
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.modules.loss
from sklearn.model_selection import train_test_split
import gemini_train_pi_groups

# Ensure MSE loss is set to original class
OriginalMSE = torch.nn.modules.loss.MSELoss
gemini_train_pi_groups.nn.MSELoss = OriginalMSE

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
# 1. Map Input Feature Slices
# ---------------------------------------------------------------------
newvar = ['p', 'v', 'q', 'w', 'T_sp', 'q_sp', 'hover_throttle', 'pwm']
vardim = [3, 3, 4, 3, 1, 4, 1, 4]
#p=3D
#v=3D
#q=4D
#w=3D
#T_sp=1D
#q_sp=4D
#hover_throttle=1D
#pwm=4D
# Dynamically map variable names to column index ranges
var_indices = {}
curr_idx = 0
for var, dim in zip(newvar, vardim):
    var_indices[var] = list(range(curr_idx, curr_idx + dim))
    curr_idx += dim

# ---------------------------------------------------------------------
# 2. Load Full Datasets (Unfiltered Ground Truth)
# ---------------------------------------------------------------------
print("Loading datasets...")

X_all_full, _, _, Y_raw_all = gemini_train_pi_groups.load_and_nondimensionalize_data(
    r_train, pwm_train, train_folder
)
idx_train, idx_val = train_test_split(
    np.arange(len(X_all_full)), test_size=0.2, random_state=random_seed
)
Y_train, Y_val = Y_raw_all[idx_train], Y_raw_all[idx_val]

X_test_full, _, _, Y_test = gemini_train_pi_groups.load_and_nondimensionalize_data(
    r_test, pwm_test, test_folder
)

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

# Store original loader function for monkey-patching
orig_loader = gemini_train_pi_groups.load_and_nondimensionalize_data

# ---------------------------------------------------------------------
# 4. Define Feature Subset Experiments
# ---------------------------------------------------------------------
# You can test individual variables or custom combinations
experiments = {
    # Individual variables
    **{f"var_{v}": [v] for v in newvar},
    
    # Example combinations:
    "kinematics_v_w": ['v', 'w'],
    "full_state_v_w_q": ['v', 'w', 'q'],
    "all_inputs": newvar
}

# ---------------------------------------------------------------------
# 5. Run Training Loop for Feature Combinations
# ---------------------------------------------------------------------
all_results = {}

for exp_name, var_list in experiments.items():
    # Calculate column indices for selected variables
    selected_cols = []
    for v in var_list:
        selected_cols.extend(var_indices[v])

    print(f"\n==================================================")
    print(f" Experiment: {exp_name} | Variables: {var_list}")
    print(f" Selected Feature Columns ({len(selected_cols)}D): {selected_cols}")
    print(f"==================================================")

    # Monkey-patch dataset loader to return only selected feature columns
    def sliced_loader(*args, **kwargs):
        X, C_f, Q, Y_raw = orig_loader(*args, **kwargs)
        return X[:, selected_cols], C_f, Q, Y_raw

    gemini_train_pi_groups.load_and_nondimensionalize_data = sliced_loader

    # Slice raw input sets for evaluation
    X_train_sub = X_all_full[idx_train][:, selected_cols]
    X_val_sub = X_all_full[idx_val][:, selected_cols]
    X_test_sub = X_test_full[:, selected_cols]

    # Train model using standard MSE Loss
    model_mse, scaler_X_mse, scaler_Y_mse, outfolder_mse = (
        gemini_train_pi_groups.train_model(
            rotor_radius=r_train,
            pwm_hover=pwm_train,
            save_folder_prefix=f"nondim_input_{exp_name}_{epochs}ep",
            data_folder=train_folder,
            epochs=epochs,
            random_seed=random_seed
        )
    )

    # Evaluate MSE Model on Train, Val, and Test splits
    metrics = {
        "train": evaluate_split(model_mse, scaler_X_mse, scaler_Y_mse, X_train_sub, Y_train),
        "val":   evaluate_split(model_mse, scaler_X_mse, scaler_Y_mse, X_val_sub, Y_val),
        "test":  evaluate_split(model_mse, scaler_X_mse, scaler_Y_mse, X_test_sub, Y_test)
    }
    all_results[exp_name] = metrics

# Restore original loader function
gemini_train_pi_groups.load_and_nondimensionalize_data = orig_loader

# ---------------------------------------------------------------------
# 6. Display Summary Comparison
# ---------------------------------------------------------------------
print("\n" + "=" * 75)
print("INPUT FEATURE COMBINATIONS COMPARISON (TEST SET)")
print("=" * 75)
print(f"{'Experiment':<20} | {'RMSE (Fx, Fy, Fz) [N]':<25} | {'NRMSE (Fx, Fy, Fz) [%]'}")
print("-" * 75)

for exp_name, metrics in all_results.items():
    rmse, mae, mape, nrmse = metrics["test"]
    rmse_str = f"{rmse[0]:.3f}, {rmse[1]:.3f}, {rmse[2]:.3f}"
    nrmse_str = f"{nrmse[0]:.1f}%, {nrmse[1]:.1f}%, {nrmse[2]:.1f}%"
    print(f"{exp_name:<20} | {rmse_str:<25} | {nrmse_str}")