#
# Plot Array Deployment Scenario Comparison Data
#
# Author:   Leah Kiner
# Creation Date:  Jan 15 2024
#

import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Basilisk.utilities import RigidBodyKinematics as rbk
from matplotlib import collections as mc



def run():

    # Load gimbal tip & tilt data
    path_to_file = "/Users/leahkiner/Desktop/GNCBreck2024/DataForPlotting/timespan_data.txt"
    timespan = np.loadtxt(path_to_file)

    path_to_file = "/Users/leahkiner/Desktop/GNCBreck2024/DataForPlotting/omega_BN_B_rot_rigid_data.txt"
    omega_BN_B_RotRigid = np.loadtxt(path_to_file)

    path_to_file = "/Users/leahkiner/Desktop/GNCBreck2024/DataForPlotting/omega_BN_B_rot_lanyard_data.txt"
    omega_BN_B_RotLanyard = np.loadtxt(path_to_file)

    path_to_file = "/Users/leahkiner/Desktop/GNCBreck2024/DataForPlotting/omega_BN_B_teles_rigid_data.txt"
    omega_BN_B_TelesRigid = np.loadtxt(path_to_file)

    path_to_file = "/Users/leahkiner/Desktop/GNCBreck2024/DataForPlotting/omega_BN_B_teles_lanyard_data.txt"
    omega_BN_B_TelesLanyard = np.loadtxt(path_to_file)

    omega_BN_BNorm_RotRigid = np.linalg.norm(omega_BN_B_RotRigid, axis=1)
    omega_BN_BNorm_RotLanyard = np.linalg.norm(omega_BN_B_RotLanyard, axis=1)
    omega_BN_BNorm_TelesRigid = np.linalg.norm(omega_BN_B_TelesRigid, axis=1)
    omega_BN_BNorm_TelesLanyard = np.linalg.norm(omega_BN_B_TelesLanyard, axis=1)

    # Plot scenario comparison
    plt.figure()
    plt.clf()
    plt.plot(timespan, omega_BN_BNorm_RotRigid, label=r'Rot Rigid')
    plt.plot(timespan, omega_BN_BNorm_RotLanyard, label=r'Rot Lanyard')
    plt.plot(timespan, omega_BN_BNorm_TelesRigid, label=r'Teles Rigid')
    plt.plot(timespan, omega_BN_BNorm_TelesLanyard, label=r'Teles Lanyard')
    plt.title('Hub Angular Velocity Norm $|{}^\mathcal{B} \omega_{\mathcal{B}/\mathcal{N}}|$', fontsize=16)
    plt.ylabel(r'(deg/s)', fontsize=14)
    plt.xlabel(r'(min)', fontsize=14)
    plt.legend(loc='upper right', prop={'size': 12})
    plt.grid(True)

    plt.show()
    plt.close("all")

if __name__ == "__main__":
    run()