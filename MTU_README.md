README.md explains the code and data provided as part of the Neural-Fly paper.
This file describes the files added by MTU.

# gemini_train_pi_groups.py
This is a module defining functions for parsing neural-fly provided data and using it to train and evaluate neural networks for predicting the residual aerodynamic force on a drone. It can also be run as a script to train a nondimensionalized neural network on the data from the neural-fly custom drone, but other scripts are more useful for doing so.

# gemini_train_dimensional.py
Do not use. Old script for training a dimensional neural network. Superceded by the train_dimensional_model function in gemeini_train_pi_groups.py

# train_nondim.py
Incomplete script that was attempting to non-dimensionalize the architecture used in the Neural-Fly paper

# utils.py
This is mostly the provided utility module from the original neural-fly code, but adds functions to create timestamped folders and to automatically write README files that record what git commit was present when a network was trained and any settings for the training. These functions are used by the training functions in gemini_train_pi_groups.py to better document what exactly was run when saving a trained network.

# train_on_A_test_on_B.py
Scripts named like this use functions defined in gemini_train_pi_groups.py to train a nondimensionalized neural network on data from drone A, then test how it performs on data from drone B. Note that they use RMS errors which means the printed error values can look quite large if a few points were bad.

# train_dimensional_on_A_test_on_B.py
As train_on_A_test_on_B.py, but training a dimensional neural network instead.

# evaluate_trained_models_using_mean_not_RMS.py
Script to load a dimensional and a nondimensional model from specified save files, then compute their accuracy using mean absolute errors instead of RMS.

# plot_predicted_vs_actual_forces.py
Script to load a dimensional and a nondimensional model from specified save files, then make plots showing their predicted forces and the actual forces.

# nondim_versus_dim_neural_fly_and_intel_plot.py
Script that plots velocity versus force and Reynold's number versus drag coefficient for both neural fly and intel data sets. This shows that the pattern is more similar between the drones when nondimensionalized.