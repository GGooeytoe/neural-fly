import gemini_train_pi_groups
model,scaler_X,scaler_label=gemini_train_pi_groups.train_model(rotor_radius=gemini_train_pi_groups.ROTOR_RADIUS["neural-fly"],save_folder_prefix="train_and_save",data_folder="./data/training")
pred,rmse,percent=gemini_train_pi_groups.test_model_on_dataset(model,scaler_X,scaler_label,"./data/training-transfer/",gemini_train_pi_groups.ROTOR_RADIUS["intel"])