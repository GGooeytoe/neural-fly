import torch
import gemini_train_pi_groups
from matplotlib import pyplot
import pickle

def plot_predicted_versus_actual(X,Y,model,scaler_X,scaler_label,axes):
    X_scaled=scaler_X.transform(X)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_Fa_reconstructed,rmse_force,percent_error=gemini_train_pi_groups.validate_model(model,scaler_label,X_scaled,Y,device)
    dims=["x","y","z"]
    for i,ax in enumerate(axes):
        ax.plot(Y[:,i],"o",label="Observed")
        ax.plot(pred_Fa_reconstructed[:,i],"x:",label="Predicted")
        ax.set_ylabel(f"F_{dims[i]} (N)")
        ax.set_xlabel("Sample")
        ax.legend()

def plot_model_on_dataset(model,scaler_X,scaler_label,data_folder,rotor_radius,pwm_hover,axes):
    X, C_f, Q, Y_raw = gemini_train_pi_groups.load_and_nondimensionalize_data(rotor_radius,pwm_hover,data_folder)
    plot_predicted_versus_actual(X,Y_raw,model,scaler_X,scaler_label,axes)

def plot_dimensional_model_on_dataset(model,scaler_X,scaler_label,data_folder,axes):
    X, Y_raw = gemini_train_pi_groups.load_and_preprocess_data(data_folder)
    plot_predicted_versus_actual(X,Y_raw,model,scaler_X,scaler_label,axes)

def load_model(pickle_path,model_obj):
    with open(pickle_path,"rb") as fh:
        data_dict=pickle.load(fh)
    key_matches=model_obj.load_state_dict(data_dict["state_dict"])
    scaler_X=data_dict["scaler_X"]
    scaler_label=data_dict["scaler_label"]
    return key_matches,scaler_X,scaler_label

if __name__=="__main__":
    nondim_model_path="train_pi_groups/train100epochs_and_save20260907_140431/model.pkl"
    dim_model_path="train_dimensional/train100epochs_and_save20260907_161346/model.pkl"
    data_folder="data/training-transfer"
    data_drone="intel"
    rotor_radius=gemini_train_pi_groups.ROTOR_RADIUS[data_drone]
    pwm_hover=gemini_train_pi_groups.PWM_HOVER[data_drone]
    import time
    timestr=time.strftime("%Y%m%d_%H%M%S")
    figname=f"nondim_versus_dim_100epochs_transfer_to_{data_drone}_{timestr}.png"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nondim_model=gemini_train_pi_groups.NonDimFFNN(input_dim=7, hidden_dim=64, output_dim=3).to(device)
    nondim_key_matchs,nondim_scaler_X,nondim_scaler_label=load_model(nondim_model_path,nondim_model)

    dim_model = gemini_train_pi_groups.FeedforwardNN(input_dim=10, hidden_dim=64, output_dim=3).to(device)
    dim_key_matchs,dim_scaler_X,dim_scaler_label=load_model(dim_model_path,dim_model)

    fig1=pyplot.figure()
    axes1=fig1.subplots(3,2)
    plot_model_on_dataset(nondim_model,nondim_scaler_X,nondim_scaler_label,data_folder,rotor_radius,pwm_hover,axes1[:,0])
    plot_dimensional_model_on_dataset(dim_model,dim_scaler_X,dim_scaler_label,data_folder)
    fig1.set_size_inches(6.5,6)
    fig1.tight_layout()
    fig1.savefig(figname)
    fig1.show()
    