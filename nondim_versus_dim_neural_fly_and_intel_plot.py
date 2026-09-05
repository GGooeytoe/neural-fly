from matplotlib import pyplot
import numpy as np
import gemini_train_pi_groups
X, C_f, Q, Y_raw=gemini_train_pi_groups.load_and_nondimensionalize_data(gemini_train_pi_groups.ROTOR_RADIUS["neural-fly"],"data/training")
Xtf, C_ftf, Qtf, Y_rawtf=gemini_train_pi_groups.load_and_nondimensionalize_data(gemini_train_pi_groups.ROTOR_RADIUS["intel"],"data/training-transfer/")
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
    ax.set_xlabel(f"log10(Re_{letter})")
    ax.set_ylabel(f"log10(C_{letter})")
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
    ax.set_ylabel(f"Residual Force in {letter} (N)")
fig2=pyplot.figure()
axes2=fig2.subplots(3,2)
plot_fs_vs_vs(Re,Y_raw,axes2[:,0],gemini_train_pi_groups.ROTOR_RADIUS["neural-fly"])
plot_fs_vs_vs(Retf,Y_rawtf,axes2[:,1],gemini_train_pi_groups.ROTOR_RADIUS["intel"])
axes2[0,0].set_title("NeuralFly Drone")
axes2[0,1].set_title("Intel ReadyToFly Drone")

#overlaid
fig3=pyplot.figure()
axes3=fig3.subplots(3,1)
plot_Cs_vs_Res(Re,C_f,axes3)
plot_Cs_vs_Res(Retf,C_ftf,axes3)
axes3[0].legend(["NeuralFly","Intel"])

fig4=pyplot.figure()
axes4=fig4.subplots(3,1)
plot_fs_vs_vs(Re,Y_raw,axes4,gemini_train_pi_groups.ROTOR_RADIUS["neural-fly"])
plot_fs_vs_vs(Retf,Y_rawtf,axes4,gemini_train_pi_groups.ROTOR_RADIUS["intel"])
axes4[0].legend(["NeuralFly","Intel"])
