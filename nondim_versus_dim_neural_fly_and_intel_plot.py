from matplotlib import pyplot
import numpy as np
import utils
import gemini_train_pi_groups


def load_and_nondimensionalize_data(rotor_radius,pwm_hover,data_folder: str = 'data/experiment'):
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
        v_wind_mag = gemini_train_pi_groups.parse_wind_speed(exp.get('condition', 'nowind'))
        w_world_vec = np.array([-v_wind_mag, 0.0, 0.0])#from figure 3A in the neural fly paper, the wind is in the -x direction
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
            v_inf = np.linalg.norm(v_rel_local) + gemini_train_pi_groups.EPSILON

            # 1. Tip speed ratios (v_tip / V_inf)
            omega = pwm * gemini_train_pi_groups.K_PWM_TO_RADS
            v_tip = omega * rotor_radius
            lambda_tip = v_tip / v_inf

            # 2. Body Reynolds number (Re = rho * V_rel * 2R / mu)
            reynolds_num = (gemini_train_pi_groups.RHO * v_rel_local * (2.0 * rotor_radius)) / gemini_train_pi_groups.MU

            # 3. Normalized relative velocity direction vector
            v_hat = v_rel_local / v_rel_local
            w_hat = w_wind_local / v_rel_local

            # Non-dimensional force coefficient vector C_f
            q_factor = gemini_train_pi_groups.RHO * v_inf*v_rel_local * (rotor_radius ** 2)

            # Feature vector: [pwm/pwm_hover (4), Re (3), q_factor (3)] -> total 10 features
            x_nondim = np.concatenate([pwm/pwm_hover,reynolds_num])#,q_factor])


            c_f = fa_local / q_factor

            X_list.append(x_nondim)
            C_f_list.append(c_f)
            dynamic_pressures_list.append(q_factor)
            raw_forces_list.append(fa_local)

    X = np.array(X_list, dtype=np.float32)
    C_f = np.array(C_f_list, dtype=np.float32)
    Q = np.array(dynamic_pressures_list, dtype=np.float32)
    Y_raw = np.array(raw_forces_list, dtype=np.float32)

    print(f"Dataset compiled successfully.")
    print(f"Non-dimensional Inputs Shape: {X.shape} | Force Coefficient Output Shape: {C_f.shape}")
    return X, C_f, Q, Y_raw

X, C_f, Q, Y_raw=load_and_nondimensionalize_data(gemini_train_pi_groups.ROTOR_RADIUS["neural-fly"],gemini_train_pi_groups.PWM_HOVER["neural-fly"],"data/training")
Xtf, C_ftf, Qtf, Y_rawtf=load_and_nondimensionalize_data(gemini_train_pi_groups.ROTOR_RADIUS["intel"],gemini_train_pi_groups.PWM_HOVER["intel"],"data/training-transfer/")
pyplot.ion()

#compare using nondimensional numbers

#on separate plots
fig=pyplot.figure()
axes=fig.subplots(3,2)
def plot_Cs_vs_Res(Re,C_f,axes):
    for i,l in enumerate(["x","y","z"]):
        plot_C_vs_Re(Re,C_f,i,l,axes[i])
def plot_C_vs_Re(Re,C_f,i,letter,ax):
    ax.loglog(np.abs(Re[:,i]),np.abs(C_f[:,i]),"o")
    ax.set_xlabel(f"|Re_{letter}|")
    ax.set_ylabel(f"|C_{letter}|")
Re=X[:,4:7]
Retf=Xtf[:,4:7]
plot_Cs_vs_Res(Re,C_f,axes[:,0])
plot_Cs_vs_Res(Retf,C_ftf,axes[:,1])
axes[0,1].set_title("Intel ReadyToFly Drone")
axes[0,0].set_title("NeuralFly Drone")

#and using raw velocity and force
def plot_fs_vs_vs(Re,f,axes,rotor_radius):
    for i,l in enumerate(["x","y","z"]):
        plot_f_vs_v(Re,f,i,l,axes[i],rotor_radius)

def plot_f_vs_v(Re,f,i,letter,ax,rotor_radius):
    v=Re*gemini_train_pi_groups.MU/2/rotor_radius/gemini_train_pi_groups.RHO
    ax.plot(np.abs(v[:,i]),np.abs(f[:,i]),"o")
    ax.set_xlabel(f"v_{letter} (m/s)")
    ax.set_ylabel(f"F_{letter} (N)")
fig2=pyplot.figure()
axes2=fig2.subplots(3,2)
plot_fs_vs_vs(Re,Y_raw,axes2[:,0],gemini_train_pi_groups.ROTOR_RADIUS["neural-fly"])
plot_fs_vs_vs(Retf,Y_rawtf,axes2[:,1],gemini_train_pi_groups.ROTOR_RADIUS["intel"])
axes2[0,0].set_title("NeuralFly Drone")
axes2[0,1].set_title("Intel ReadyToFly Drone")

#overlaid
fig3=pyplot.figure()
axes3=fig3.subplots(3,2)
plot_Cs_vs_Res(Re,C_f,axes3[:,0])
plot_Cs_vs_Res(Retf,C_ftf,axes3[:,0])
axes3[0,0].legend(["NeuralFly","Intel"])

plot_fs_vs_vs(Re,Y_raw,axes3[:,1],gemini_train_pi_groups.ROTOR_RADIUS["neural-fly"])
plot_fs_vs_vs(Retf,Y_rawtf,axes3[:,1],gemini_train_pi_groups.ROTOR_RADIUS["intel"])
fig3.tight_layout()