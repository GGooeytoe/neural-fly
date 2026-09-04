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

# Mapping condition names to fan wind speeds in world x-direction (m/s)
WIND_SPEED_MAP = {
    'nowind': 0.0,
    '10wind': 1.3,
    '20wind': 2.5,
    '30wind': 3.7,
    '35wind': 4.2,
    '40wind': 4.9,
    '50wind': 6.1,
    '70wind': 8.5,
    '70p20sint': 8.5,  # Nominal base wind speed
    '100wind': 12.1
}

def parse_wind_speed(condition_str: str) -> float:
    """Parses wind speed magnitude in world x-direction from dataset condition metadata."""
    for key, val in WIND_SPEED_MAP.items():
        if key in condition_str.lower():
            return val
    # Fallback parsing if exact string isn't matched
    match = re.search(r'(\d+)wind', condition_str.lower())
    if match:
        duty_cycle = float(match.group(1))
        return duty_cycle * 0.121  # Approximate conversion factor
    return 0.0

def load_and_preprocess_data(data_folder: str = 'data/experiment'):
    """
    Loads Neural-Fly experiment data and converts world frame variables 
    to the drone's local body frame.
    
    Returns:
        X: Inputs [local_wind (3), local_velocity (3), pwm (4)] -> Shape (N, 10)
        Y: Target [local_aerodynamic_force (3)]             -> Shape (N, 3)
    """
    print(f"Loading data from '{data_folder}'...")
    raw_data = utils.load_data(folder=data_folder)
    print(f"Loaded {len(raw_data)} experiment runs.")

    X_list, Y_list = [], []

    for exp in raw_data:
        # Get world-frame wind vector [vx, vy, vz]
        v_wind_mag = parse_wind_speed(exp.get('condition', 'nowind'))
        w_world_vec = np.array([v_wind_mag, 0.0, 0.0])

        num_timesteps = len(exp['t'])

        for t in range(num_timesteps):
            # Extract rotation matrix R (Body -> World frame)
            R = exp['R'][t]
            if R.shape == (9,):
                R = R.reshape(3, 3)
            
            # Transpose of R maps World frame -> Body (Local) frame
            R_T = R.T  

            # Extract world frame quantities & motor PWM
            v_world = exp['v'][t]          # Drone velocity in world frame (3D)
            fa_world = exp['fa'][t]        # Aerodynamic force in world frame (3D)
            pwm = exp['pwm'][t]            # Motor PWM commands (4D)

            # Transform vectors to local frame
            w_local = R_T @ w_world_vec    # Local wind speed vector (3D)
            v_local = R_T @ v_world        # Local drone velocity vector (3D)
            fa_local = R_T @ fa_world      # Local aerodynamic force vector (3D)

            # Feature vector: [wind_local (3), v_local (3), pwm (4)]
            x_feat = np.concatenate([w_local, v_local, pwm])

            X_list.append(x_feat)
            Y_list.append(fa_local)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    print(f"Dataset compiled successfully.")
    print(f"Input features shape: {X.shape} | Target force shape: {Y.shape}")
    return X, Y

# Define Simple Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 3):
        super(FeedforwardNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_model(data_folder='./data/training',epochs=40, random_seed=42):
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 1. Load and process dataset
    X, Y = load_and_preprocess_data(data_folder)

    # 2. Train-Validation Split (80% Train, 20% Validation)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # 3. Standardize inputs and targets
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_val_scaled = scaler_Y.transform(Y_val)

    # PyTorch DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(Y_train_scaled, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(Y_val_scaled, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 4. Initialize Network, Loss Function, and Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}\n")

    model = FeedforwardNN(input_dim=10, hidden_dim=64, output_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 5. Training Loop
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

        # Validation step
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
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

    # 6. Model Performance Evaluation (Unscaled Metrics)
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        preds_scaled = model(X_val_tensor).cpu().numpy()
        preds_unscaled = scaler_Y.inverse_transform(preds_scaled)

    rmse = np.sqrt(np.mean((Y_val - preds_unscaled) ** 2, axis=0))
    print("\n--- Validation Performance (Unscaled RMSE) ---")
    print(f"Force X (f_ax): {rmse[0]:.4f} N")
    print(f"Force Y (f_ay): {rmse[1]:.4f} N")
    print(f"Force Z (f_az): {rmse[2]:.4f} N")

    percent_error = np.sqrt(np.mean(((preds_unscaled-Y_val)/Y_val)**2, axis=0))

    print("\n--- Validation Performance RMS(Error/Truth) ---")
    print(f"Force X (f_ax): {percent_error[0]:.4f}%")
    print(f"Force Y (f_ay): {percent_error[1]:.4f}%")
    print(f"Force Z (f_az): {percent_error[2]:.4f}%")

    # 7. Visualization
    plt.figure(figsize=(12, 5))

    # Loss Curve Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (Normalized)')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    # Force Prediction Comparison Plot (First 150 validation samples)
    plt.subplot(1, 2, 2)
    sample_indices = np.arange(150)
    plt.plot(Y_val[sample_indices, 0], 'r--', label='True f_ax')
    plt.plot(preds_unscaled[sample_indices, 0], 'r-', label='Pred f_ax')
    plt.plot(Y_val[sample_indices, 1], 'g--', label='True f_ay')
    plt.plot(preds_unscaled[sample_indices, 1], 'g-', label='Pred f_ay')
    plt.plot(Y_val[sample_indices, 2], 'b--', label='True f_az')
    plt.plot(preds_unscaled[sample_indices, 2], 'b-', label='Pred f_az')
    plt.xlabel('Sample Index')
    plt.ylabel('Force (N)')
    plt.title('Predicted vs Ground Truth Local Aerodynamic Forces')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('neural_fly_ffnn_results.png')
    plt.show()

if __name__ == '__main__':
    train_model()