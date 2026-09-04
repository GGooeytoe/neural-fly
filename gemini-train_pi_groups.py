import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import the repository's helper module
import utils

# Physical and Geometric Constants
RHO = 1.225            # Air density (kg/m^3)
MU = 1.81e-5           # Dynamic viscosity of air (Pa*s)
ROTOR_RADIUS = 0.127   # Rotor radius R (meters, ~5 inches)
K_PWM_TO_RADS = 1.0    # TODO: what is this for the drones? Approximate PWM scale factor to rad/s (Omega = k * PWM)
EPSILON = 1e-4         # Small constant to prevent division by zero

WIND_SPEED_MAP = {
    'nowind': 0.0,
    '10wind': 1.3,
    '20wind': 2.5,
    '30wind': 3.7,
    '35wind': 4.2,
    '40wind': 4.9,
    '50wind': 6.1,
    '70wind': 8.5,
    '70p20sint': 8.5,
    '100wind': 12.1
}

def parse_wind_speed(condition_str: str) -> float:
    """Extracts wind speed magnitude in world x-direction."""
    for key, val in WIND_SPEED_MAP.items():
        if key in condition_str.lower():
            return val
    match = re.search(r'(\d+)wind', condition_str.lower())
    if match:
        return float(match.group(1)) * 0.121
    return 0.0

def load_and_nondimensionalize_data(data_folder: str = 'data/experiment'):
    """
    Loads experiment data and extracts non-dimensional parameters:
    Inputs:
      - Tip speed ratios (lambda_1, lambda_2, lambda_3, lambda_4)
      - Body Reynolds number (Re)
      - Normalized relative velocity vector direction (v_hat_x, v_hat_y, v_hat_z)
    Output:
      - Quadratic drag force coefficient vector (C_f = f_a / (rho * V_inf^2 * R^2))
    """
    print(f"Loading data from '{data_folder}'...")
    raw_data = utils.load_data(folder=data_folder)

    X_list, C_f_list, dynamic_pressures_list, raw_forces_list = [], [], [], []

    for exp in raw_data:
        v_wind_mag = parse_wind_speed(exp.get('condition', 'nowind'))
        w_world_vec = np.array([v_wind_mag, 0.0, 0.0])
        num_timesteps = len(exp['t'])

        for t in range(num_timesteps):
            R_mat = exp['R'][t]
            if R_mat.shape == (9,):
                R_mat = R_mat.reshape(3, 3)
            R_T = R_mat.T

            # Velocities & forces in local frame
            v_drone_local = R_T @ exp['v'][t]
            w_wind_local = R_T @ w_world_vec
            fa_local = R_T @ exp['fa'][t]
            pwm = exp['pwm'][t]

            # Relative free-stream velocity vector
            v_rel_local = w_wind_local - v_drone_local
            v_inf = np.linalg_norm(v_rel_local) + EPSILON

            # 1. Tip speed ratios (v_tip / V_inf)
            omega = pwm * K_PWM_TO_RADS
            v_tip = omega * ROTOR_RADIUS
            lambda_tip = v_tip / v_inf

            # 2. Body Reynolds number (Re = rho * V_inf * 2R / mu)
            reynolds_num = (RHO * v_inf * (2.0 * ROTOR_RADIUS)) / MU

            # 3. Normalized relative velocity direction vector
            v_hat = v_rel_local / v_inf

            # Feature vector: [lambda (4), Re (1), v_hat (3)] -> total 8 features
            x_nondim = np.concatenate([lambda_tip, [reynolds_num], v_hat])

            # Non-dimensional force coefficient vector C_f
            q_factor = RHO * v_inf*v_rel_local * (ROTOR_RADIUS ** 2)
            c_f = fa_local / q_factor

            X_list.append(x_nondim)
            C_f_list.append(c_f)
            dynamic_pressures_list.append(q_factor)
            raw_forces_list.append(fa_local)

    X = np.array(X_list, dtype=np.float32)
    C_f = np.array(C_f_list, dtype=np.float32)
    Q = np.array(dynamic_pressures_list, dtype=np.float32).reshape(-1, 1)
    Y_raw = np.array(raw_forces_list, dtype=np.float32)

    print(f"Dataset compiled successfully.")
    print(f"Non-dimensional Inputs Shape: {X.shape} | Force Coefficient Output Shape: {C_f.shape}")
    return X, C_f, Q, Y_raw

class NonDimFFNN(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, output_dim: int = 3):
        super(NonDimFFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_model():
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load non-dimensionalized dataset
    data_folder = './data/experiment'
    X, C_f, Q, Y_raw = load_and_nondimensionalize_data(data_folder)

    # 2. Train-Validation Split
    indices = np.arange(len(X))
    idx_train, idx_val = train_test_split(indices, test_size=0.2, random_state=42)

    X_train, X_val = X[idx_train], X[idx_val]
    Cf_train, Cf_val = C_f[idx_train], C_f[idx_val]
    Q_val = Q[idx_val]
    Y_val_raw = Y_raw[idx_val]

    # 3. Standardize Non-Dimensional Inputs and Outputs
    scaler_X = StandardScaler()
    scaler_Cf = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    Cf_train_scaled = scaler_Cf.fit_transform(Cf_train)
    Cf_val_scaled = scaler_Cf.transform(Cf_val)

    # PyTorch DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(Cf_train_scaled, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(Cf_val_scaled, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 4. Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}\n")

    model = NonDimFFNN(input_dim=8, hidden_dim=64, output_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 5. Training Loop
    epochs = 40
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # Validation Step
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                running_val_loss += loss.item() * batch_x.size(0)

        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss (Scaled C_f): {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

    # 6. Evaluation: Convert predicted drag coefficients back to physical forces (Newtons)
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        pred_Cf_scaled = model(X_val_tensor).cpu().numpy()
        pred_Cf = scaler_Cf.inverse_transform(pred_Cf_scaled)

    # Reconstruct Physical Aerodynamic Force: f_a = C_f * (rho * V_inf^2 * R^2)
    pred_Fa_reconstructed = pred_Cf * Q_val

    rmse_force = np.sqrt(np.mean((Y_val_raw - pred_Fa_reconstructed) ** 2, axis=0))
    print("\n--- Physical Aerodynamic Force Validation (Unscaled RMSE) ---")
    print(f"Force X (f_ax): {rmse_force[0]:.4f} N")
    print(f"Force Y (f_ay): {rmse_force[1]:.4f} N")
    print(f"Force Z (f_az): {rmse_force[2]:.4f} N")

    percent_error = np.sqrt(np.mean(((pred_Fa_reconstructed-Y_val_raw)/Y_val_raw)**2, axis=0))

    print("\n--- Validation Performance RMS(Error/Truth) ---")
    print(f"Force X (f_ax): {percent_error[0]:.4f}%")
    print(f"Force Y (f_ay): {percent_error[1]:.4f}%")
    print(f"Force Z (f_az): {percent_error[2]:.4f}%")

    # 7. Visualizations
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (Scaled C_f)')
    plt.title('Non-Dimensional Model Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sample_indices = np.arange(150)
    plt.plot(Y_val_raw[sample_indices, 0], 'r--', label='True f_ax')
    plt.plot(pred_Fa_reconstructed[sample_indices, 0], 'r-', label='Pred f_ax')
    plt.plot(Y_val_raw[sample_indices, 1], 'g--', label='True f_ay')
    plt.plot(pred_Fa_reconstructed[sample_indices, 1], 'g-', label='Pred f_ay')
    plt.plot(Y_val_raw[sample_indices, 2], 'b--', label='True f_az')
    plt.plot(pred_Fa_reconstructed[sample_indices, 2], 'b-', label='Pred f_az')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstructed Force (N)')
    plt.title('Reconstructed Physical Force vs Ground Truth')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('nondimensional_neural_fly_results.png')
    plt.show()

if __name__ == '__main__':
    train_model()