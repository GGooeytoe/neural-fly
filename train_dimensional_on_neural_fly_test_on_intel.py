import gemini_train_pi_groups
import os
model,scaler_X,scaler_label,outfolder=gemini_train_pi_groups.train_dimensional_model(save_folder_prefix="train_and_save",data_folder="./data/training")
pred,rmse,percent=gemini_train_pi_groups.test_dimensional_model_on_dataset(model,scaler_X,scaler_label,"./data/training-transfer/")

with open(os.path.join(outfolder,"test_on_intel.txt"),"wt") as fh:
    fh.write("--- Physical Aerodynamic Force Validation (Unscaled RMSE) ---\n")
    fh.write(f"Force X (f_ax): {rmse[0]:.4f} N\n")
    fh.write(f"Force Y (f_ay): {rmse[1]:.4f} N\n")
    fh.write(f"Force Z (f_az): {rmse[2]:.4f} N\n")
    fh.write("\n--- Validation Performance RMS(Error/Truth) ---\n")
    fh.write(f"Force X (f_ax): {percent[0]:.4f}%\n")
    fh.write(f"Force Y (f_ay): {percent[1]:.4f}%\n")
    fh.write(f"Force Z (f_az): {percent[2]:.4f}%\n")