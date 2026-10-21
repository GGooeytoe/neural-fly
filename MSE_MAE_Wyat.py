# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:42:47 2026

@author: WyatB
"""

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import gemini_train_pi_groups

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
# 1. Load and Split Datasets (Train, Validation, Test)
# ---------------------------------------------------------------------
print("Loading and splitting datasets...")

# Training & Validation split (replicating the exact 80/20 split used in train_model)
X_all, _, _, Y_raw_all = gemini_train_pi_groups.load_and_nondimensionalize_data(
    r_train, pwm_train, train_folder
)
idx_train, idx_val = train_test_split(
    np.arange(len(X_all)), test_size=0.2, random_state=random_seed
)

X_train, Y_train = X_all[idx_train], Y_raw_all[idx_train]
X_val, Y_val = X_all[idx_val], Y_raw_all[idx_val]

# Transfer Test dataset
X_test, _, _, Y_test = gemini_train_pi_groups.load_and_nondimensionalize_data(
    r_test, pwm_test, test_folder
)


# Helper evaluation function
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
    
    # Percentage Errors
    eps = 1e-5
    mape = np.mean(np.abs(error / (np.abs(Y_raw) + eps)), axis=0) * 100.0  # Mean Abs % Error
    nrms = (rmse / (np.abs(Y_raw, axis=0) + eps)) * 100.0                # Range Normalized RMSE %
    


    return rmse, mae, mape


# ---------------------------------------------------------------------
# PASS 1: Non-Dimensional Model Trained with MSE Loss (nn.MSELoss)
# ---------------------------------------------------------------------
print("\n>>> [1/3] Training Non-Dimensional Model using MSE Loss...")
gemini_train_pi_groups.nn.MSELoss = nn.MSELoss

model_mse, scaler_X_mse, scaler_Y_mse, outfolder_mse = (
    gemini_train_pi_groups.train_model(
        rotor_radius=r_train,
        pwm_hover=pwm_train,
        save_folder_prefix=f"nondim_mse_{epochs}ep",
        data_folder=train_folder,
        epochs=epochs,
        random_seed=random_seed
    )
)

# Evaluate MSE Model on Train, Val, and Test splits
metrics_mse = {
    "train": evaluate_split(model_mse, scaler_X_mse, scaler_Y_mse, X_train, Y_train),
    "val":   evaluate_split(model_mse, scaler_X_mse, scaler_Y_mse, X_val, Y_val),
    "test":  evaluate_split(model_mse, scaler_X_mse, scaler_Y_mse, X_test, Y_test)
}


# ---------------------------------------------------------------------
# PASS 2: Non-Dimensional Model Trained with MAE / L1 Loss (nn.L1Loss)
# ---------------------------------------------------------------------
print("\n>>> [2/3] Training Non-Dimensional Model using Absolute Error (L1/MAE) Loss...")
gemini_train_pi_groups.nn.MSELoss = nn.L1Loss

model_mae, scaler_X_mae, scaler_Y_mae, outfolder_mae = (
    gemini_train_pi_groups.train_model(
        rotor_radius=r_train,
        pwm_hover=pwm_train,
        save_folder_prefix=f"nondim_mae_{epochs}ep",
        data_folder=train_folder,
        epochs=epochs,
        random_seed=random_seed
    )
)

# Evaluate MAE Model on Train, Val, and Test splits
metrics_mae = {
    "train": evaluate_split(model_mae, scaler_X_mae, scaler_Y_mae, X_train, Y_train),
    "val":   evaluate_split(model_mae, scaler_X_mae, scaler_Y_mae, X_val, Y_val),
    "test":  evaluate_split(model_mae, scaler_X_mae, scaler_Y_mae, X_test, Y_test)
}

# ---------------------------------------------------------------------
# PASS 3: Non-Dimensional Model Trained with MHE / L1 Loss (nn.L1Loss)
# ---------------------------------------------------------------------
print("\n>>> [3/3] Training Non-Dimensional Model using Absolute Error Huber Loss...")
gemini_train_pi_groups.nn.MSELoss = lambda: nn.HuberLoss(reduction='mean', delta=1.0)

model_mhe, scaler_X_mhe, scaler_Y_mhe, outfolder_mhe = (
    gemini_train_pi_groups.train_model(
        rotor_radius=r_train,
        pwm_hover=pwm_train,
        save_folder_prefix=f"nondim_mhe_{epochs}ep",
        data_folder=train_folder,
        epochs=epochs,
        random_seed=random_seed
    )
)

# Evaluate MAE Model on Train, Val, and Test splits
metrics_mhe = {
    "train": evaluate_split(model_mhe, scaler_X_mhe, scaler_Y_mhe, X_train, Y_train),
    "val":   evaluate_split(model_mhe, scaler_X_mhe, scaler_Y_mhe, X_val, Y_val),
    "test":  evaluate_split(model_mhe, scaler_X_mhe, scaler_Y_mhe, X_test, Y_test)
}


# ---------------------------------------------------------------------
# DISPLAY RESULTS & LOG TO FILE
# ---------------------------------------------------------------------
def format_summary(title, metrics_dict):
    out = [f"\n{'='*70}\n {title}\n{'='*70}"]
    for split, (rmse, mae, mape) in metrics_dict.items():
        out.append(f"\n--- [{split.upper()} SET] ---")
        out.append(f"  RMSE  (N) -> Fx: {rmse[0]:.4f} | Fy: {rmse[1]:.4f} | Fz: {rmse[2]:.4f}")
        out.append(f"  MAE   (N) -> Fx: {mae[0]:.4f} | Fy: {mae[1]:.4f} | Fz: {mae[2]:.4f}")
        out.append(f"  MAPE  (%) -> Fx: {mape[0]:.2f}% | Fy: {mape[1]:.2f}% | Fz: {mape[2]:.2f}%")
    return "\n".join(out)

mse_summary = format_summary("NON-DIMENSIONAL MODEL (TRAINED WITH MSE LOSS)", metrics_mse)
mae_summary = format_summary("NON-DIMENSIONAL MODEL (TRAINED WITH MAE/L1 LOSS)", metrics_mae)
mhe_summary = format_summary("NON-DIMENSIONAL MODEL (TRAINED WITH HUBER LOSS)", metrics_mhe)

print(mse_summary)
print(mae_summary)
print(mhe_summary)

# Save results log to validation folder
log_path = os.path.join(outfolder_mae, "full_train_val_test_comparison.txt")
with open(log_path, "w") as fh:
    fh.write(mse_summary + "\n\n" + mae_summary+ "\n\n" + mhe_summary)

print(f"\nSaved detailed evaluation metrics to: {log_path}")