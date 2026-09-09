import torch
import gemini_train_pi_groups
import numpy as np
from matplotlib import pyplot
import pickle

def load_model(pickle_path,model_obj):
    with open(pickle_path,"rb") as fh:
        data_dict=pickle.load(fh)
    key_matches=model_obj.load_state_dict(data_dict["state_dict"])
    scaler_X=data_dict["scaler_X"]
    scaler_label=data_dict["scaler_label"]
    return key_matches,scaler_X,scaler_label
def validate_model(model,scaler_X,scaler_label,X_val,Y_val_raw,device):
    X_scaled=scaler_X.transform(X_val)
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        pred_f_scaled = model(X_val_tensor).cpu().numpy()
        pred_Fa_reconstructed = scaler_label.inverse_transform(pred_f_scaled)

    # Reconstruct Physical Aerodynamic Force: f_a = C_f * (rho * V_inf^2 * R^2)
    # pred_Fa_reconstructed = np.array([pred_Cf[i] * Q_val[i] for i in range(len(pred_Cf))])

    mean_force_err = np.mean(np.abs(Y_val_raw - pred_Fa_reconstructed), axis=0)
    print("\n--- Physical Aerodynamic Force Validation (Mean Absolute) ---")
    print(f"Force X (f_ax): {mean_force_err[0]:.4f} N")
    print(f"Force Y (f_ay): {mean_force_err[1]:.4f} N")
    print(f"Force Z (f_az): {mean_force_err[2]:.4f} N")

    percent_error = np.mean(np.abs(pred_Fa_reconstructed-Y_val_raw)/np.abs(Y_val_raw), axis=0)

    print("\n--- Validation Performance Mean Absolute(Error)/Absolute(Truth) ---")
    print(f"Force X (f_ax): {percent_error[0]:.4f}%")
    print(f"Force Y (f_ay): {percent_error[1]:.4f}%")
    print(f"Force Z (f_az): {percent_error[2]:.4f}%")
    return pred_Fa_reconstructed,mean_force_err,percent_error
if __name__=="__main__":
    random_seed=42
    nondim_model_path="train_pi_groups/train100epochs_and_save20260907_140431/model.pkl"
    dim_model_path="train_dimensional/train100epochs_and_save20260907_174222/model.pkl"
    val_folder="data/training"
    test_folder="data/training-transfer"
    val_drone="neural-fly"
    test_drone="intel"

    val_rotor_radius=gemini_train_pi_groups.ROTOR_RADIUS[val_drone]
    val_pwm_hover=gemini_train_pi_groups.PWM_HOVER[val_drone]
    test_rotor_radius=gemini_train_pi_groups.ROTOR_RADIUS[test_drone]
    test_pwm_hover=gemini_train_pi_groups.PWM_HOVER[test_drone]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nondim_model=gemini_train_pi_groups.NonDimFFNN(input_dim=7, hidden_dim=64, output_dim=3).to(device)
    nondim_key_matchs,nondim_scaler_X,nondim_scaler_label=load_model(nondim_model_path,nondim_model)

    dim_model = gemini_train_pi_groups.FeedforwardNN(input_dim=10, hidden_dim=64, output_dim=3).to(device)
    dim_key_matchs,dim_scaler_X,dim_scaler_label=load_model(dim_model_path,dim_model)

    print("Test Nondim model")
    print("Validation set")
    X_nondim, _,_, Y_nondim = gemini_train_pi_groups.load_and_nondimensionalize_data(val_rotor_radius,val_pwm_hover,val_folder)
    #redo train val split
    indices = np.arange(len(X_nondim))
    idx_train, idx_val = gemini_train_pi_groups.train_test_split(indices, test_size=0.2, random_state=random_seed)
    X_nondim_train, X_nondim_val = X_nondim[idx_train], X_nondim[idx_val]
    Y_nondim_val = Y_nondim[idx_val]
    validate_model(nondim_model,nondim_scaler_X,nondim_scaler_label,X_nondim_val,Y_nondim_val,device)
    print("Test set")
    X_nondim_test, _,_, Y_nondim_test = gemini_train_pi_groups.load_and_nondimensionalize_data(test_rotor_radius,test_pwm_hover,test_folder)
    validate_model(nondim_model,nondim_scaler_X,nondim_scaler_label,X_nondim_test,Y_nondim_test,device)

    print("Test dim model")
    print("Validation set")
    X_dim, Y_dim = gemini_train_pi_groups.load_and_preprocess_data(val_folder)
    X_dim_train, X_dim_val, Y_dim_train, Y_dim_val = gemini_train_pi_groups.train_test_split(X_dim, Y_dim, test_size=0.2, random_state=random_seed)
    validate_model(dim_model,dim_scaler_X,dim_scaler_label,X_dim_val,Y_dim_val,device)
    print("Test set")
    X_dim_test, Y_dim_test = gemini_train_pi_groups.load_and_preprocess_data(test_folder)
    validate_model(dim_model,dim_scaler_X,dim_scaler_label,X_dim_test,Y_dim_test,device)

