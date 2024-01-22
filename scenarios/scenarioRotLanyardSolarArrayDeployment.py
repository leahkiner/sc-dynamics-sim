#
#   Rotational Solar Array Deployment Scenario
#   Author:             Leah Kiner
#   Creation Date:      Jan 2, 2024
#

import pytest
import inspect
import os
import numpy as np
import pandas as pd
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport
import matplotlib
import matplotlib.pyplot as plt
from Basilisk.fswAlgorithms import prescribedRot1DOF, prescribedTrans
from Basilisk.simulation import spacecraft, prescribedMotionStateEffector, gravityEffector
from Basilisk.utilities import macros, RigidBodyKinematics as rbk
from Basilisk.architecture import messaging

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
splitPath = path.split('simulation')

def run(show_plots):

    simProcessName = "simProcess"
    dynTaskName = "dynTask"
    fswTaskName = "fswTask"

    # Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    dynTimeStep = 0.01  # [s]
    fswTimeStep = 0.1  # [s]
    dataRecStep = 1.0  # [s]
    dynProcessRate = macros.sec2nano(dynTimeStep)  # [ns]
    fswProcessRate = macros.sec2nano(fswTimeStep)  # [ns]
    dataRecRate = macros.sec2nano(dataRecStep)  # [ns]
    simProc = scSim.CreateNewProcess(simProcessName)
    simProc.addTask(scSim.CreateNewTask(dynTaskName, dynProcessRate))
    simProc.addTask(scSim.CreateNewTask(fswTaskName, fswProcessRate))

    # Add the spacecraft module to test file
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraftBody"

    # Define the mass properties of the rigid spacecraft hub
    # Calculate array element inertias (approximate as rectangular prisms)
    massHub = 800  # [kg]
    lengthHub = 2.0  # [m]
    widthHub = 1.0  # [m]
    depthHub = 1.0  # [m]
    IHub_11 = (1/12) * massHub * (lengthHub * lengthHub + depthHub * depthHub)  # [kg m^2]
    IHub_22 = (1/12) * massHub * (lengthHub * lengthHub + widthHub * widthHub)  # [kg m^2]
    IHub_33 = (1/12) * massHub * (widthHub * widthHub + depthHub * depthHub)  # [kg m^2]
    scObject.hub.mHub = massHub  # kg
    scObject.hub.r_BcB_B = [0.0, 0.0, 0.0]  # [m]
    scObject.hub.IHubPntBc_B = [[IHub_11, 0.0, 0.0],
                                [0.0, IHub_22, 0.0],
                                [0.0, 0.0, IHub_33]]

    # Set the initial inertial hub states
    scObject.hub.r_CN_NInit = [0.0, 0.0, 0.0]
    scObject.hub.v_CN_NInit = [0.0, 0.0, 0.0]
    scObject.hub.omega_BN_BInit = [0.0, 0.0, 0.0]
    scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]

    # Add the scObject to the runtime call list
    scSim.AddModelToTask(dynTaskName, scObject)

    # Define the state effector kinematic properties
    # DCMs representing the attitude of the B frame with respect to the solar array body frames
    dcm_BS1 = np.array([[0,  1,  0],
                        [0,  0, -1],
                        [-1,  0,  0]])
    dcm_BS2 = np.array([[0, -1,  0],
                        [0,  0, -1],
                        [1,  0,  0]])

    # Solar array body frame origins with respect to point B expressed in B frame components
    rArray1SB_B = np.array([0.5, 0.0, 0.5])   # [m]
    rArray2SB_B = np.array([-0.5, 0.0, 0.5])   # [m]

    r_MS_S = [0.0, 0.0, 0.0]  # [m]  Use if Mount frame origin is at SADA frame origin

    r_M1S1_B = dcm_BS1 @ r_MS_S  # [m]
    r_M2S2_B = dcm_BS2 @ r_MS_S  # [m]

    r_M1B_B = r_M1S1_B + rArray1SB_B  # [m]
    r_M2B_B = r_M2S2_B + rArray2SB_B  # [m]

    radiusArray = 3.0  # [m]
    tRamp = 1.0  # [s]
    rotAxis_M = np.array([0.0, 1.0, 0.0])

    # Calculate array element inertias (approximate as rectangular prisms)
    massElement = 8.0  # [kg]
    l = radiusArray  # [m]
    w = 2 * radiusArray * np.cos(72 * macros.D2R)  # [m]
    d = 0.01  # [m]
    IElement_11 = (1/12) * massElement * (l * l + d * d)  # [kg m^2]
    IElement_22 = (1/12) * massElement * (l * l + w * w)  # [kg m^2]
    IElement_33 = (1/12) * massElement * (w * w + d * d)  # [kg m^2]

    # Rotation 1 initial parameters
    array1ThetaInit1 = 0.0 * macros.D2R  # [rad]
    array2ThetaInit1 = 0.0 * macros.D2R  # [rad]
    prv_FM1Init1 = array1ThetaInit1 * rotAxis_M
    prv_FM2Init1 = array2ThetaInit1 * rotAxis_M
    sigma_FM1Init1 = rbk.PRV2MRP(prv_FM1Init1)
    sigma_FM2Init1 = rbk.PRV2MRP(prv_FM2Init1)
    r_FM1_M1Init1 = [0.0, 0.0, 0.0] #[radiusArray * np.cos(macros.D2R * 72.0), 0.0, radiusArray * np.sin(macros.D2R * 72.0)]  # [m]
    r_FM2_M2Init1 = [0.0, 0.0, 0.0] #[- radiusArray * np.cos(macros.D2R * 72.0), 0.0, radiusArray * np.sin(macros.D2R * 72.0)]  # [m]

    # Rotation 2 initial parameters
    array1ThetaInit2 = 90.0 * macros.D2R  # [rad]
    array2ThetaInit2 = -90.0 * macros.D2R  # [rad]
    prv_FM1Init2 = array1ThetaInit2 * rotAxis_M
    prv_FM2Init2 = array2ThetaInit2 * rotAxis_M
    sigma_FM1Init2 = rbk.PRV2MRP(prv_FM1Init2)
    sigma_FM2Init2 = rbk.PRV2MRP(prv_FM2Init2)
    r_FM1_M1Init2 = [radiusArray, 0.0, 0.0]  #[(2/3) * radiusArray * np.cos(macros.D2R * 18.0), 0.0, (2/3) * radiusArray * np.sin(macros.D2R * 18.0)]  # [m]  Use if Mount frame origin is at array centroid
    r_FM2_M2Init2 = [-radiusArray, 0.0, 0.0]  #[- (2/3) * radiusArray * np.cos(macros.D2R * 18.0), 0.0, - (2/3) * radiusArray * np.sin(macros.D2R * 18.0)]  # [m]  Use if Mount frame origin is at array centroid

    # Create the solar array elements
    numArrayElements = 10
    array1ElementList = list()
    array2ElementList = list()
    for i in range(numArrayElements):
        array1ElementList.append(prescribedMotionStateEffector.PrescribedMotionStateEffector())
        array2ElementList.append(prescribedMotionStateEffector.PrescribedMotionStateEffector())
        array1ElementList[i].mass = massElement  # [kg]
        array2ElementList[i].mass = massElement  # [kg]
        array1ElementList[i].IPntFc_F = [[IElement_11, 0.0, 0.0], [0.0, IElement_22, 0.0], [0.0, 0.0, IElement_33]]  # [kg m^2]
        array2ElementList[i].IPntFc_F = [[IElement_11, 0.0, 0.0], [0.0, IElement_22, 0.0], [0.0, 0.0, IElement_33]]  # [kg m^2]
        array1ElementList[i].r_MB_B = r_M1B_B  # [m]
        array2ElementList[i].r_MB_B = r_M2B_B  # [m]
        array1ElementList[i].r_FcF_F = [- 0.5 * w, 0.0, 0.5 * radiusArray]  # [m] For rectangular wedge
        array2ElementList[i].r_FcF_F = [0.5 * w, 0.0, 0.5 * radiusArray]  # [m] For rectangular wedge
        # array1ElementList[i].r_FcF_F = [- radiusArray * np.cos(72 * macros.D2R), 0.0, (1/3) * radiusArray * np.sin(72 * macros.D2R)]  # [m] For triangular wedge
        # array2ElementList[i].r_FcF_F = [radiusArray * np.cos(72 * macros.D2R), 0.0, (1/3) * radiusArray * np.sin(72 * macros.D2R)]  # [m] For triangular wedge
        array1ElementList[i].r_FM_M = r_FM1_M1Init1  # [m]
        array2ElementList[i].r_FM_M = r_FM2_M2Init1  # [m]
        array1ElementList[i].rPrime_FM_M = np.array([0.0, 0.0, 0.0])  # [m/s]
        array2ElementList[i].rPrime_FM_M = np.array([0.0, 0.0, 0.0])  # [m/s]
        array1ElementList[i].rPrimePrime_FM_M = np.array([0.0, 0.0, 0.0])  # [m/s^2]
        array2ElementList[i].rPrimePrime_FM_M = np.array([0.0, 0.0, 0.0])  # [m/s^2]
        array1ElementList[i].omega_FM_F = np.array([0.0, 0.0, 0.0])  # [rad/s]
        array2ElementList[i].omega_FM_F = np.array([0.0, 0.0, 0.0])  # [rad/s]
        array1ElementList[i].omegaPrime_FM_F = np.array([0.0, 0.0, 0.0])  # [rad/s^2]
        array2ElementList[i].omegaPrime_FM_F = np.array([0.0, 0.0, 0.0])  # [rad/s^2]
        array1ElementList[i].sigma_FM = sigma_FM1Init1
        array2ElementList[i].sigma_FM = sigma_FM2Init1
        array1ElementList[i].omega_MB_B = [0.0, 0.0, 0.0]  # [rad/s]
        array2ElementList[i].omega_MB_B = [0.0, 0.0, 0.0]  # [rad/s]
        array1ElementList[i].omegaPrime_MB_B = [0.0, 0.0, 0.0]  # [rad/s^2]
        array2ElementList[i].omegaPrime_MB_B = [0.0, 0.0, 0.0]  # [rad/s^2]
        array1ElementList[i].sigma_MB = [0.0, 0.0, 0.0]
        array2ElementList[i].sigma_MB = [0.0, 0.0, 0.0]
        array1ElementList[i].ModelTag = "array1Element" + str(i + 1)
        array2ElementList[i].ModelTag = "array2Element" + str(i + 1)

        # Add array elements to spacecraft
        scObject.addStateEffector(array1ElementList[i])
        scObject.addStateEffector(array2ElementList[i])

        # Add the array elements to runtime call list
        scSim.AddModelToTask(dynTaskName, array1ElementList[i])
        scSim.AddModelToTask(dynTaskName, array2ElementList[i])

    # Create the array element reference angle messages
    array1ElementRotMessageList1 = list()
    array2ElementRotMessageList1 = list()
    for i in range(numArrayElements):
        array1ElementMessageData = messaging.HingedRigidBodyMsgPayload()
        array2ElementMessageData = messaging.HingedRigidBodyMsgPayload()
        array1ElementMessageData.theta = array1ThetaInit2  # [rad]
        array2ElementMessageData.theta = array2ThetaInit1  # [rad]
        array1ElementMessageData.thetaDot = 0.0  # [rad/s]
        array2ElementMessageData.thetaDot = 0.0  # [rad/s]
        array1ElementRotMessageList1.append(messaging.HingedRigidBodyMsg().write(array1ElementMessageData))
        array2ElementRotMessageList1.append(messaging.HingedRigidBodyMsg().write(array2ElementMessageData))

    # Create stand-alone element translational state messages
    array1ElementTransMotionMessageData = messaging.PrescribedTransMotionMsgPayload()
    array2ElementTransMotionMessageData = messaging.PrescribedTransMotionMsgPayload()
    array1ElementTransMotionMessageData.r_FM_M = r_FM1_M1Init1  # [m]
    array2ElementTransMotionMessageData.r_FM_M = r_FM2_M2Init1  # [m]
    array1ElementTransMotionMessageData.rPrime_FM_M = np.array([0.0, 0.0, 0.0])  # [m/s]
    array2ElementTransMotionMessageData.rPrime_FM_M = np.array([0.0, 0.0, 0.0])  # [m/s]
    array1ElementTransMotionMessageData.rPrimePrime_FM_M = np.array([0.0, 0.0, 0.0])  # [m/s^2]
    array2ElementTransMotionMessageData.rPrimePrime_FM_M = np.array([0.0, 0.0, 0.0])  # [m/s^2]
    array1ElementTransMotionMessage = messaging.PrescribedTransMotionMsg().write(array1ElementTransMotionMessageData)
    array2ElementTransMotionMessage = messaging.PrescribedTransMotionMsg().write(array2ElementTransMotionMessageData)

    # Initialize the prescribedRot1DOF module configuration data
    tDeploy1 = 5 * 60  # [s]
    tCoast1 = tDeploy1 - (2 * tRamp)

    array1MaxRotAccelList1 = []
    array2MaxRotAccelList1 = []
    for j in range(2):
        for i in range(numArrayElements):
            if j == 0:
                thetaInit = array1ThetaInit1  # [rad]
                thetaRef = array1ThetaInit2  # [rad]
            else:
                thetaInit = array2ThetaInit1  # [rad]
                thetaRef = array2ThetaInit1  # [rad]

            # Determine the angle and angle rate at the end of the ramp segment/start of the coast segment
            if (thetaInit < thetaRef):
                thetaDDotMax = (thetaRef - thetaInit) / ((tCoast1 * tRamp) + (tRamp * tRamp))
            else:
                thetaDDotMax = (thetaRef - thetaInit) / - ((tCoast1 * tRamp) + (tRamp * tRamp))

            if j == 0:
                array1MaxRotAccelList1.append(thetaDDotMax)
            else:
                array2MaxRotAccelList1.append(thetaDDotMax)

    array1PrescribedElementRotList = list()
    array2PrescribedElementRotList = list()
    for i in range(numArrayElements):
        array1PrescribedElementRotList.append(prescribedRot1DOF.prescribedRot1DOF())
        array2PrescribedElementRotList.append(prescribedRot1DOF.prescribedRot1DOF())
        array1PrescribedElementRotList[i].coastOption = True
        array2PrescribedElementRotList[i].coastOption = True
        array1PrescribedElementRotList[i].tRamp = tRamp  # [s]
        array2PrescribedElementRotList[i].tRamp = tRamp  # [s]
        array1PrescribedElementRotList[i].rotAxis_M = rotAxis_M
        array2PrescribedElementRotList[i].rotAxis_M = rotAxis_M
        array1PrescribedElementRotList[i].thetaDDotMax = array1MaxRotAccelList1[i]  # [rad/s^2]
        array2PrescribedElementRotList[i].thetaDDotMax = array2MaxRotAccelList1[i]  # [rad/s^2]
        array1PrescribedElementRotList[i].thetaInit = array1ThetaInit1  # [rad]
        array2PrescribedElementRotList[i].thetaInit = array2ThetaInit1  # [rad]
        array1PrescribedElementRotList[i].ModelTag = "prescribedRot1DOFArray1Element" + str(i + 1)
        array2PrescribedElementRotList[i].ModelTag = "prescribedRot1DOFArray2Element" + str(i + 1)

        # Add the prescribedRot1DOF modules to runtime call list
        scSim.AddModelToTask(fswTaskName, array1PrescribedElementRotList[i])
        scSim.AddModelToTask(fswTaskName, array2PrescribedElementRotList[i])

        # Connect the rotation angle messages to the rotational profilers
        array1PrescribedElementRotList[i].spinningBodyInMsg.subscribeTo(array1ElementRotMessageList1[i])
        array2PrescribedElementRotList[i].spinningBodyInMsg.subscribeTo(array2ElementRotMessageList1[i])

        # Connect the rotational profilers to the array element effectors
        array1ElementList[i].prescribedRotMotionInMsg.subscribeTo(array1PrescribedElementRotList[i].prescribedRotMotionOutMsg)
        array2ElementList[i].prescribedRotMotionInMsg.subscribeTo(array2PrescribedElementRotList[i].prescribedRotMotionOutMsg)
        array1ElementList[i].prescribedTransMotionInMsg.subscribeTo(array1ElementTransMotionMessage)
        array2ElementList[i].prescribedTransMotionInMsg.subscribeTo(array2ElementTransMotionMessage)

    # Add energy and momentum variables to log
    scObjectLog = scObject.logger(["totRotAngMomPntC_N", "totRotEnergy"], dataRecRate)
    scSim.AddModelToTask(fswTaskName, scObjectLog)

    # Add other states to log
    scStateData = scObject.scStateOutMsg.recorder(dataRecRate)
    array1Element1PrescribedDataLog = array1PrescribedElementRotList[0].spinningBodyOutMsg.recorder(dataRecRate)
    array1Element2PrescribedDataLog = array1PrescribedElementRotList[1].spinningBodyOutMsg.recorder(dataRecRate)
    array1Element3PrescribedDataLog = array1PrescribedElementRotList[2].spinningBodyOutMsg.recorder(dataRecRate)
    array1Element4PrescribedDataLog = array1PrescribedElementRotList[3].spinningBodyOutMsg.recorder(dataRecRate)
    array1Element5PrescribedDataLog = array1PrescribedElementRotList[4].spinningBodyOutMsg.recorder(dataRecRate)
    array1Element6PrescribedDataLog = array1PrescribedElementRotList[5].spinningBodyOutMsg.recorder(dataRecRate)
    array1Element7PrescribedDataLog = array1PrescribedElementRotList[6].spinningBodyOutMsg.recorder(dataRecRate)
    array1Element8PrescribedDataLog = array1PrescribedElementRotList[7].spinningBodyOutMsg.recorder(dataRecRate)
    array1Element9PrescribedDataLog = array1PrescribedElementRotList[8].spinningBodyOutMsg.recorder(dataRecRate)
    array1Element10PrescribedDataLog = array1PrescribedElementRotList[9].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element1PrescribedDataLog = array2PrescribedElementRotList[0].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element2PrescribedDataLog = array2PrescribedElementRotList[1].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element3PrescribedDataLog = array2PrescribedElementRotList[2].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element4PrescribedDataLog = array2PrescribedElementRotList[3].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element5PrescribedDataLog = array2PrescribedElementRotList[4].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element6PrescribedDataLog = array2PrescribedElementRotList[5].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element7PrescribedDataLog = array2PrescribedElementRotList[6].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element8PrescribedDataLog = array2PrescribedElementRotList[7].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element9PrescribedDataLog = array2PrescribedElementRotList[8].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element10PrescribedDataLog = array2PrescribedElementRotList[9].spinningBodyOutMsg.recorder(dataRecRate)

    scSim.AddModelToTask(fswTaskName, scStateData)
    scSim.AddModelToTask(fswTaskName, array1Element1PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element2PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element3PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element4PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element5PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element6PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element7PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element8PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element9PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element10PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element1PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element2PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element3PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element4PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element5PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element6PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element7PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element8PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element9PrescribedDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element10PrescribedDataLog)

    # Initialize the simulation
    scSim.InitializeSimulation()

    # Set the simulation time
    simTime1 = tDeploy1 + 10  # [s]
    scSim.ConfigureStopTime(macros.sec2nano(simTime1))

    # Begin the simulation
    scSim.ExecuteSimulation()

    # Initialize the prescribedRot1DOF module configuration data
    tDeploy2 = 20 * 60  # [s]
    tCoast2 = tDeploy2 - (2 * tRamp)

    array1MaxRotAccelList2 = []
    for i in range(numArrayElements):
        thetaInit = array1ThetaInit2  # [rad]
        thetaRef = (36 * i * macros.D2R) + array1ThetaInit2  # [rad]

        # Determine the angle and angle rate at the end of the ramp segment/start of the coast segment
        if (thetaInit < thetaRef):
            thetaDDotMax = (thetaRef - thetaInit) / ((tCoast2 * tRamp) + (tRamp * tRamp))
        else:
            thetaDDotMax = (thetaRef - thetaInit) / - ((tCoast2 * tRamp) + (tRamp * tRamp))

        array1MaxRotAccelList2.append(thetaDDotMax)

    # Create stand-alone element translational state messages
    array1ElementTransMotionMessageData = messaging.PrescribedTransMotionMsgPayload()
    array1ElementTransMotionMessageData.r_FM_M = r_FM1_M1Init2  # [m]
    array1ElementTransMotionMessageData.rPrime_FM_M = np.array([0.0, 0.0, 0.0])  # [m/s]
    array1ElementTransMotionMessageData.rPrimePrime_FM_M = np.array([0.0, 0.0, 0.0])  # [m/s^2]
    array1ElementTransMotionMessage = messaging.PrescribedTransMotionMsg().write(array1ElementTransMotionMessageData)

    # Create the array element reference angle messages
    array1ElementRotMessageList2 = list()
    for i in range(numArrayElements):
        array1ElementList[i].prescribedTransMotionInMsg.subscribeTo(array1ElementTransMotionMessage)
        array1ElementList[i].r_FcF_F = [- 0.5 * w, 0.0, - 0.5 * radiusArray]  # [m] For rectangular wedge
        # array1ElementList[i].r_FcF_F = [- (2/3) * radiusArray * np.cos(72 * macros.D2R) * np.sin(18 * macros.D2R), 0.0, - (2/3) * radiusArray * np.cos(72 * macros.D2R) * np.cos(18 * macros.D2R)]  # [m] For triangular wedge
        array1ElementList[i].r_FM_M = r_FM1_M1Init2  # [m]
        array1ElementList[i].sigma_FM = sigma_FM1Init2

        array1PrescribedElementRotList[i].thetaInit = array1ThetaInit2  # [rad]
        array1PrescribedElementRotList[i].thetaDDotMax = array1MaxRotAccelList2[i]  # [rad/s^2]

        array1ElementMessageData = messaging.HingedRigidBodyMsgPayload()
        array1ElementMessageData.theta = (36 * i * macros.D2R) + array1ThetaInit2  # [rad]
        array1ElementMessageData.thetaDot = 0.0  # [rad/s]
        array1ElementRotMessageList2.append(messaging.HingedRigidBodyMsg().write(array1ElementMessageData))

        # Connect the rotation angle messages to the rotational profilers
        array1PrescribedElementRotList[i].spinningBodyInMsg.subscribeTo(array1ElementRotMessageList2[i])

    # Set the simulation time
    simTime2 = tDeploy2 + 10  # [s]
    scSim.ConfigureStopTime(macros.sec2nano(simTime1 + simTime2))

    # Begin the simulation
    scSim.ExecuteSimulation()

    # Initialize the prescribedRot1DOF module configuration data
    tDeploy3 = 5 * 60  # [s]
    tCoast3 = tDeploy3 - (2 * tRamp)

    array2MaxRotAccelList2 = []
    for i in range(numArrayElements):
        thetaInit = array2ThetaInit1  # [rad]
        thetaRef = array2ThetaInit2  # [rad]

        # Determine the angle and angle rate at the end of the ramp segment/start of the coast segment
        thetaDDotMax = np.abs(thetaRef - thetaInit) / ((tCoast3 * tRamp) + (tRamp * tRamp))

        array2MaxRotAccelList2.append(thetaDDotMax)

    # Create the array element reference angle messages
    array2ElementRotMessageList2 = list()
    for i in range(numArrayElements):
        array2PrescribedElementRotList[i].thetaDDotMax = array2MaxRotAccelList2[i]  # [rad/s^2]

        array2ElementMessageData = messaging.HingedRigidBodyMsgPayload()
        array2ElementMessageData.theta = array2ThetaInit2  # [rad]
        array2ElementMessageData.thetaDot = 0.0  # [rad/s]
        array2ElementRotMessageList2.append(messaging.HingedRigidBodyMsg().write(array2ElementMessageData))

        # Connect the rotation angle messages to the rotational profilers
        array2PrescribedElementRotList[i].spinningBodyInMsg.subscribeTo(array2ElementRotMessageList2[i])
    # Set the simulation time
    simTime3 = tDeploy3 + 10  # [s]
    scSim.ConfigureStopTime(macros.sec2nano(simTime1 + simTime2 + simTime3))

    # Begin the simulation
    scSim.ExecuteSimulation()

    # Initialize the prescribedRot1DOF module configuration data
    tDeploy4 = 20 * 60  # [s]
    tCoast4 = tDeploy4 - (2 * tRamp)

    array2MaxRotAccelList3 = []
    for i in range(numArrayElements):
        thetaInit = array2ThetaInit2  # [rad]
        thetaRef = - (36 * i * macros.D2R) + array2ThetaInit2  # [rad]

        # Determine the angle and angle rate at the end of the ramp segment/start of the coast segment
        if (thetaInit < thetaRef):
            thetaDDotMax = (thetaRef - thetaInit) / ((tCoast4 * tRamp) + (tRamp * tRamp))
        else:
            thetaDDotMax = (thetaRef - thetaInit) / - ((tCoast4 * tRamp) + (tRamp * tRamp))

        array2MaxRotAccelList3.append(thetaDDotMax)

    # Create stand-alone element translational state messages
    array2ElementTransMotionMessageData = messaging.PrescribedTransMotionMsgPayload()
    array2ElementTransMotionMessageData.r_FM_M = r_FM2_M2Init2  # [m]
    array2ElementTransMotionMessageData.rPrime_FM_M = np.array([0.0, 0.0, 0.0])  # [m/s]
    array2ElementTransMotionMessageData.rPrimePrime_FM_M = np.array([0.0, 0.0, 0.0])  # [m/s^2]
    array2ElementTransMotionMessage = messaging.PrescribedTransMotionMsg().write(array2ElementTransMotionMessageData)

    # Create the array element reference angle messages
    array2ElementRotMessageList3 = list()
    for i in range(numArrayElements):
        array2ElementList[i].prescribedTransMotionInMsg.subscribeTo(array2ElementTransMotionMessage)
        array2ElementList[i].r_FcF_F = [0.5 * w, 0.0, - 0.5 * radiusArray]  # [m] For rectangular wedge
        # array2ElementList[i].r_FcF_F = [(2/3) * radiusArray * np.cos(72 * macros.D2R) * np.sin(18 * macros.D2R), 0.0, - (2/3) * radiusArray * np.cos(72 * macros.D2R) * np.cos(18 * macros.D2R)]  # [m] For triangular wedge
        array2ElementList[i].r_FM_M = r_FM2_M2Init2  # [m]
        array2ElementList[i].sigma_FM = sigma_FM2Init2

        array2PrescribedElementRotList[i].thetaInit = array2ThetaInit2  # [rad]
        array2PrescribedElementRotList[i].thetaDDotMax = array2MaxRotAccelList3[i]  # [rad/s^2]

        array2ElementMessageData = messaging.HingedRigidBodyMsgPayload()
        array2ElementMessageData.theta = - (36 * i * macros.D2R) + array2ThetaInit2  # [rad]
        array2ElementMessageData.thetaDot = 0.0  # [rad/s]
        array2ElementRotMessageList3.append(messaging.HingedRigidBodyMsg().write(array2ElementMessageData))

        # Connect the rotation angle messages to the rotational profilers
        array2PrescribedElementRotList[i].spinningBodyInMsg.subscribeTo(array2ElementRotMessageList3[i])

    # Set the simulation time
    simTime4 = tDeploy4 + 10  # [s]
    scSim.ConfigureStopTime(macros.sec2nano(simTime1 + simTime2 + simTime3 + simTime4))

    # Begin the simulation
    scSim.ExecuteSimulation()

    # Extract the logged data
    rotAngMom_N = unitTestSupport.addTimeColumn(scObjectLog.times(), scObjectLog.totRotAngMomPntC_N)
    rotEnergy = unitTestSupport.addTimeColumn(scObjectLog.times(), scObjectLog.totRotEnergy)
    timespan = scStateData.times() * macros.NANO2MIN  # [min]
    r_BN_N = scStateData.r_BN_N  # [m]
    omega_BN_B = scStateData.omega_BN_B * macros.R2D  # [deg/s]
    sigma_BN = scStateData.sigma_BN

    theta_array1Element1 = array1Element1PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array1Element2 = array1Element2PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array1Element3 = array1Element3PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array1Element4 = array1Element4PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array1Element5 = array1Element5PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array1Element6 = array1Element6PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array1Element7 = array1Element7PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array1Element8 = array1Element8PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array1Element9 = array1Element9PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array1Element10 = array1Element10PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array2Element1 = array2Element1PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array2Element2 = array2Element2PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array2Element3 = array2Element3PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array2Element4 = array2Element4PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array2Element5 = array2Element5PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array2Element6 = array2Element6PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array2Element7 = array2Element7PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array2Element8 = array2Element8PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array2Element9 = array2Element9PrescribedDataLog.theta * macros.R2D  # [deg]
    theta_array2Element10 = array2Element10PrescribedDataLog.theta * macros.R2D  # [deg]
    thetaDot_array1Element1 = array1Element1PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array1Element2 = array1Element2PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array1Element3 = array1Element3PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array1Element4 = array1Element4PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array1Element5 = array1Element5PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array1Element6 = array1Element6PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array1Element7 = array1Element7PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array1Element8 = array1Element8PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array1Element9 = array1Element9PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array1Element10 = array1Element10PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element1 = array2Element1PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element2 = array2Element2PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element3 = array2Element3PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element4 = array2Element4PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element5 = array2Element5PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element6 = array2Element6PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element7 = array2Element7PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element8 = array2Element8PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element9 = array2Element9PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element10 = array2Element10PrescribedDataLog.thetaDot * macros.R2D  # [deg/s]

    # Write timespan data to a text file
    timespan_data_file = open(r"/Users/leahkiner/Desktop/timespan_data.txt", "w+")
    np.savetxt(timespan_data_file, timespan)
    timespan_data_file.close()

    # Write hub angular velocity data to a text file for deployment scenario comparison
    omega_BN_B_rot_lanyard_data_file = open(r"/Users/leahkiner/Desktop/GNCBreck2024/DataForPlotting/omega_BN_B_rot_lanyard_data.txt", "w+")
    np.savetxt(omega_BN_B_rot_lanyard_data_file, omega_BN_B)
    omega_BN_B_rot_lanyard_data_file.close()

    plt.close("all")

    # Plot array 1 element angles
    plt.figure()
    plt.clf()
    plt.plot(timespan, theta_array1Element1, label=r'$\theta_1$')
    plt.plot(timespan, theta_array1Element2, label=r'$\theta_2$')
    plt.plot(timespan, theta_array1Element3, label=r'$\theta_3$')
    plt.plot(timespan, theta_array1Element4, label=r'$\theta_4$')
    plt.plot(timespan, theta_array1Element5, label=r'$\theta_5$')
    plt.plot(timespan, theta_array1Element6, label=r'$\theta_6$')
    plt.plot(timespan, theta_array1Element7, label=r'$\theta_7$')
    plt.plot(timespan, theta_array1Element8, label=r'$\theta_8$')
    plt.plot(timespan, theta_array1Element9, label=r'$\theta_9$')
    plt.plot(timespan, theta_array1Element10, label=r'$\theta_{10}$')
    # plt.title(r'Array 1 Element Angles', fontsize=16)
    plt.ylabel('(deg)', fontsize=14)
    plt.xlabel('Time (min)', fontsize=14)
    plt.legend(loc='upper left', prop={'size': 12})
    plt.grid(True)

    # Plot array 2 element angles
    plt.figure()
    plt.clf()
    plt.plot(timespan, theta_array2Element1, label=r'$\theta_1$')
    plt.plot(timespan, theta_array2Element2, label=r'$\theta_2$')
    plt.plot(timespan, theta_array2Element3, label=r'$\theta_3$')
    plt.plot(timespan, theta_array2Element4, label=r'$\theta_4$')
    plt.plot(timespan, theta_array2Element5, label=r'$\theta_5$')
    plt.plot(timespan, theta_array2Element6, label=r'$\theta_6$')
    plt.plot(timespan, theta_array2Element7, label=r'$\theta_7$')
    plt.plot(timespan, theta_array2Element8, label=r'$\theta_8$')
    plt.plot(timespan, theta_array2Element9, label=r'$\theta_9$')
    plt.plot(timespan, theta_array2Element10, label=r'$\theta_{10}$')
    # plt.title(r'Array 2 Element Angles', fontsize=16)
    plt.ylabel('(deg)', fontsize=14)
    plt.xlabel('Time (min)', fontsize=14)
    plt.legend(loc='center left', prop={'size': 12})
    plt.grid(True)

    # Plot array 1 element angle rates
    plt.figure()
    plt.clf()
    plt.plot(timespan, thetaDot_array1Element1, label=r'$\dot{\theta}_1$')
    plt.plot(timespan, thetaDot_array1Element2, label=r'$\dot{\theta}_2$')
    plt.plot(timespan, thetaDot_array1Element3, label=r'$\dot{\theta}_3$')
    plt.plot(timespan, thetaDot_array1Element4, label=r'$\dot{\theta}_4$')
    plt.plot(timespan, thetaDot_array1Element5, label=r'$\dot{\theta}_5$')
    plt.plot(timespan, thetaDot_array1Element6, label=r'$\dot{\theta}_6$')
    plt.plot(timespan, thetaDot_array1Element7, label=r'$\dot{\theta}_7$')
    plt.plot(timespan, thetaDot_array1Element8, label=r'$\dot{\theta}_8$')
    plt.plot(timespan, thetaDot_array1Element9, label=r'$\dot{\theta}_9$')
    plt.plot(timespan, thetaDot_array1Element10, label=r'$\dot{\theta}_{10}$')
    # plt.title(r'Array 1 Element Angle Rates', fontsize=16)
    plt.ylabel('(deg/s)', fontsize=14)
    plt.xlabel('Time (min)', fontsize=14)
    plt.legend(loc='center right', prop={'size': 12})
    plt.grid(True)

    # Plot array 2 element angle rates
    plt.figure()
    plt.clf()
    plt.plot(timespan, thetaDot_array2Element1, label=r'$\dot{\theta}_1$')
    plt.plot(timespan, thetaDot_array2Element2, label=r'$\dot{\theta}_2$')
    plt.plot(timespan, thetaDot_array2Element3, label=r'$\dot{\theta}_3$')
    plt.plot(timespan, thetaDot_array2Element4, label=r'$\dot{\theta}_4$')
    plt.plot(timespan, thetaDot_array2Element5, label=r'$\dot{\theta}_5$')
    plt.plot(timespan, thetaDot_array2Element6, label=r'$\dot{\theta}_6$')
    plt.plot(timespan, thetaDot_array2Element7, label=r'$\dot{\theta}_7$')
    plt.plot(timespan, thetaDot_array2Element8, label=r'$\dot{\theta}_8$')
    plt.plot(timespan, thetaDot_array2Element9, label=r'$\dot{\theta}_9$')
    plt.plot(timespan, thetaDot_array2Element10, label=r'$\dot{\theta}_{10}$')
    # plt.title(r'Array 2 Element Angle Rates', fontsize=16)
    plt.ylabel('(deg/s)', fontsize=14)
    plt.xlabel('Time (min)', fontsize=14)
    plt.legend(loc='center left', prop={'size': 12})
    plt.grid(True)

    # Plot r_BN_N
    plt.figure()
    plt.clf()
    plt.plot(timespan, r_BN_N[:, 0], label=r'$r_{1}$')
    plt.plot(timespan, r_BN_N[:, 1], label=r'$r_{2}$')
    plt.plot(timespan, r_BN_N[:, 2], label=r'$r_{3}$')
    # plt.title(r'${}^\mathcal{N} r_{\mathcal{B}/\mathcal{N}}$ Spacecraft Inertial Trajectory', fontsize=16)
    plt.ylabel('(m)', fontsize=14)
    plt.xlabel('Time (min)', fontsize=14)
    plt.legend(loc='upper left', prop={'size': 12})
    plt.grid(True)

    # Plot sigma_BN
    plt.figure()
    plt.clf()
    plt.plot(timespan, sigma_BN[:, 0], label=r'$\sigma_{1}$')
    plt.plot(timespan, sigma_BN[:, 1], label=r'$\sigma_{2}$')
    plt.plot(timespan, sigma_BN[:, 2], label=r'$\sigma_{3}$')
    # plt.title(r'$\sigma_{\mathcal{B}/\mathcal{N}}$ Spacecraft Inertial MRP Attitude', fontsize=16)
    plt.ylabel('', fontsize=14)
    plt.xlabel('Time (min)', fontsize=14)
    plt.legend(loc='lower left', prop={'size': 12})
    plt.grid(True)

    # Plot omega_BN_B
    plt.figure()
    plt.clf()
    plt.plot(timespan, omega_BN_B[:, 0], label=r'$\omega_{1}$')
    plt.plot(timespan, omega_BN_B[:, 1], label=r'$\omega_{2}$')
    plt.plot(timespan, omega_BN_B[:, 2], label=r'$\omega_{3}$')
    # plt.title(r'Spacecraft Hub Angular Velocity ${}^\mathcal{B} \omega_{\mathcal{B}/\mathcal{N}}$', fontsize=16)
    plt.xlabel('Time (min)', fontsize=14)
    plt.ylabel('(deg/s)', fontsize=14)
    plt.legend(loc='lower right', prop={'size': 12})
    plt.grid(True)

    # Plotting: Conservation quantities
    plt.figure()
    plt.clf()
    plt.plot(timespan, rotAngMom_N[:, 1] - rotAngMom_N[0, 1],
             timespan, rotAngMom_N[:, 2] - rotAngMom_N[0, 2],
             timespan, rotAngMom_N[:, 3] - rotAngMom_N[0, 3])
    plt.title('Rotational Angular Momentum Difference', fontsize=16)
    plt.ylabel('(Nms)', fontsize=14)
    plt.xlabel('Time (min)', fontsize=14)
    plt.grid(True)

    plt.figure()
    plt.clf()
    plt.plot(timespan, rotEnergy[:, 1] - rotEnergy[0, 1])
    plt.title('Total Energy Difference', fontsize=16)
    plt.ylabel('Energy (J)', fontsize=14)
    plt.xlabel('Time (min)', fontsize=14)
    plt.grid(True)

    if show_plots:
        plt.show()
    plt.close("all")

if __name__ == "__main__":
    run(
        True,   # show_plots
    )