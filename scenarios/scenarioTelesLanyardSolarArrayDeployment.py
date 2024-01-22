#
#   Translational Solar Array Deployment Scenario
#   Author:             Leah Kiner
#   Creation Date:      Jan 9, 2024
#

import pytest
import inspect
import os
import numpy as np
import pandas as pd
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import vizSupport
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
    scObject.hub.sigma_BNInit = [0.0, 0.0, 0.0]

    # Add the scObject to the runtime call list
    scSim.AddModelToTask(dynTaskName, scObject)

    # Define the state effector kinematic properties
    r_M1B_B = [0.5, 0.0, 1.0]  # [m]
    r_M2B_B = [-0.5, 0.0, 1.0]  # [m]

    tRamp = 1.0  # [s]
    transAxis_M = np.array([1.0, 0.0, 0.0])
    rotAxis_M = np.array([0.0, 1.0, 0.0])
    lenPanel = 2.0  # [m]

    # Calculate array element inertias (approximate as rectangular prisms)
    massElement = 10  # [kg]
    l = lenPanel  # [m]
    w = 0.75  # [m]
    d = 0.01  # [m]
    I_11 = (1/12) * massElement * (l * l + d * d)  # [kg m^2]
    I_22 = (1/12) * massElement * (l * l + w * w)  # [kg m^2]
    I_33 = (1/12) * massElement * (w * w + d * d)  # [kg m^2]

    # Rotation initial parameters
    array1ThetaInit = 90.0 * macros.D2R  # [rad]
    array2ThetaInit = 90.0 * macros.D2R  # [rad]
    prv_FM1Init1 = array1ThetaInit * rotAxis_M
    prv_FM2Init1 = array2ThetaInit * rotAxis_M
    sigma_FM1Init1 = rbk.PRV2MRP(prv_FM1Init1)
    sigma_FM2Init1 = rbk.PRV2MRP(prv_FM2Init1)
    r_FM1_M1Init = [0.0, 0.0, 0.0]  # [m]
    r_FM2_M2Init = [0.0, 0.0, 0.0]  # [m]

    # Translation initial parameters
    array1TransPosInit = 0.0  # [m]
    array2TransPosInit = 0.0  # [m]

    # Create the solar array elements
    numArrayElements = 5
    array1ElementList = list()
    array2ElementList = list()
    for i in range(numArrayElements):
        array1ElementList.append(prescribedMotionStateEffector.PrescribedMotionStateEffector())
        array2ElementList.append(prescribedMotionStateEffector.PrescribedMotionStateEffector())
        array1ElementList[i].mass = massElement  # [kg]
        array2ElementList[i].mass = massElement  # [kg]
        array1ElementList[i].IPntFc_F = [[I_11, 0.0, 0.0], [0.0, I_22, 0.0], [0.0, 0.0, I_33]]  # [kg m^2]
        array2ElementList[i].IPntFc_F = [[I_11, 0.0, 0.0], [0.0, I_22, 0.0], [0.0, 0.0, I_33]]  # [kg m^2]
        array1ElementList[i].r_MB_B = r_M1B_B  # [m]
        array2ElementList[i].r_MB_B = r_M2B_B  # [m]
        array1ElementList[i].r_FcF_F = [0.5 * lenPanel, 0.0, 0.0]  # [m]
        array2ElementList[i].r_FcF_F = [0.5 * lenPanel, 0.0, 0.0]  # [m]
        array1ElementList[i].r_FM_M = r_FM1_M1Init  # [m]
        array2ElementList[i].r_FM_M = r_FM2_M2Init  # [m]
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

    # Create the array element reference messages
    array1ElementRotMessageList1 = list()
    array2ElementRotMessageList1 = list()
    array1ElementTransMessageList1 = list()
    array2ElementTransMessageList1 = list()
    for i in range(numArrayElements):
        array1ElementMessageData1 = messaging.PrescribedTransMsgPayload()
        array2ElementMessageData1 = messaging.PrescribedTransMsgPayload()
        array1ElementMessageData1.scalarPos = 0.0  # [m]
        array2ElementMessageData1.scalarPos = 0.0  # [m]
        array1ElementMessageData1.scalarVel = 0.0  # [m/s]
        array2ElementMessageData1.scalarVel = 0.0  # [m/s]
        array1ElementTransMessageList1.append(messaging.PrescribedTransMsg().write(array1ElementMessageData1))
        array2ElementTransMessageList1.append(messaging.PrescribedTransMsg().write(array2ElementMessageData1))

        array1ElementMessageData2 = messaging.HingedRigidBodyMsgPayload()
        array2ElementMessageData2 = messaging.HingedRigidBodyMsgPayload()
        array1ElementMessageData2.theta = 0.0 * macros.D2R  # [rad]
        array2ElementMessageData2.theta = array2ThetaInit  # [rad]
        array1ElementMessageData2.thetaDot = 0.0  # [rad/s]
        array2ElementMessageData2.thetaDot = 0.0  # [rad/s]
        array1ElementRotMessageList1.append(messaging.HingedRigidBodyMsg().write(array1ElementMessageData2))
        array2ElementRotMessageList1.append(messaging.HingedRigidBodyMsg().write(array2ElementMessageData2))

    # Initialize the prescribedRot1DOF module configuration data
    tDeploy1 = 5 * 60  # [s]
    tCoast1 = tDeploy1 - (2 * tRamp)

    array1MaxRotAccelList1 = []
    array2MaxRotAccelList1 = []
    for j in range(2):
        for i in range(numArrayElements):
            if j == 0:
                thetaInit = array1ThetaInit  # [rad]
                thetaRef = 0.0 * macros.D2R  # [rad]
            else:
                thetaInit = array2ThetaInit  # [rad]
                thetaRef = array2ThetaInit  # [rad]

            # Determine the angle and angle rate at the end of the ramp segment/start of the coast segment
            if (thetaInit < thetaRef):
                thetaDDotMax = (thetaRef - thetaInit) / ((tCoast1 * tRamp) + (tRamp * tRamp))
            else:
                thetaDDotMax = (thetaRef - thetaInit) / - ((tCoast1 * tRamp) + (tRamp * tRamp))

            if j == 0:
                array1MaxRotAccelList1.append(thetaDDotMax)
            else:
                array2MaxRotAccelList1.append(thetaDDotMax)

    array1PrescribedElementTransList = list()
    array2PrescribedElementTransList = list()
    array1PrescribedElementRotList = list()
    array2PrescribedElementRotList = list()
    for i in range(numArrayElements):
        array1PrescribedElementTransList.append(prescribedTrans.prescribedTrans())
        array2PrescribedElementTransList.append(prescribedTrans.prescribedTrans())
        array1PrescribedElementTransList[i].coastOption = True
        array2PrescribedElementTransList[i].coastOption = True
        array1PrescribedElementTransList[i].tRamp = tRamp  # [s]
        array2PrescribedElementTransList[i].tRamp = tRamp  # [s]
        array1PrescribedElementTransList[i].transAxis_M = transAxis_M
        array2PrescribedElementTransList[i].transAxis_M = transAxis_M
        array1PrescribedElementTransList[i].transAccelMax = 0.0  # [m/s^2]
        array2PrescribedElementTransList[i].transAccelMax = 0.0  # [m/s^2]
        array1PrescribedElementTransList[i].transPosInit = 0.0  # [m]
        array2PrescribedElementTransList[i].transPosInit = 0.0  # [m]
        array1PrescribedElementTransList[i].ModelTag = "prescribedTransArray1Element" + str(i + 1)
        array2PrescribedElementTransList[i].ModelTag = "prescribedTransArray2Element" + str(i + 1)
        
        
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
        array1PrescribedElementRotList[i].thetaInit = array1ThetaInit  # [rad]
        array2PrescribedElementRotList[i].thetaInit = array2ThetaInit  # [rad]
        array1PrescribedElementRotList[i].ModelTag = "prescribedRot1DOFArray1Element" + str(i + 1)
        array2PrescribedElementRotList[i].ModelTag = "prescribedRot1DOFArray2Element" + str(i + 1)

        # Add the prescribedTrans and prescribedRot1DOF modules to runtime call list
        scSim.AddModelToTask(fswTaskName, array1PrescribedElementTransList[i])
        scSim.AddModelToTask(fswTaskName, array2PrescribedElementTransList[i])
        scSim.AddModelToTask(fswTaskName, array1PrescribedElementRotList[i])
        scSim.AddModelToTask(fswTaskName, array2PrescribedElementRotList[i])

        # Connect the angle messages to the profilers
        array1PrescribedElementTransList[i].translatingBodyInMsg.subscribeTo(array1ElementTransMessageList1[i])
        array2PrescribedElementTransList[i].translatingBodyInMsg.subscribeTo(array2ElementTransMessageList1[i])
        array1PrescribedElementRotList[i].spinningBodyInMsg.subscribeTo(array1ElementRotMessageList1[i])
        array2PrescribedElementRotList[i].spinningBodyInMsg.subscribeTo(array2ElementRotMessageList1[i])

        # Connect the profilers to the array element effectors
        array1ElementList[i].prescribedTransMotionInMsg.subscribeTo(array1PrescribedElementTransList[i].prescribedTransMotionOutMsg)
        array2ElementList[i].prescribedTransMotionInMsg.subscribeTo(array2PrescribedElementTransList[i].prescribedTransMotionOutMsg)
        array1ElementList[i].prescribedRotMotionInMsg.subscribeTo(array1PrescribedElementRotList[i].prescribedRotMotionOutMsg)
        array2ElementList[i].prescribedRotMotionInMsg.subscribeTo(array2PrescribedElementRotList[i].prescribedRotMotionOutMsg)

    # Add energy and momentum variables to log
    scObjectLog = scObject.logger(["totRotAngMomPntC_N", "totRotEnergy"], dataRecRate)
    scSim.AddModelToTask(fswTaskName, scObjectLog)

    # Add other states to log
    scStateData = scObject.scStateOutMsg.recorder(dataRecRate)
    array1Element1PrescribedTransDataLog = array1PrescribedElementTransList[0].translatingBodyOutMsg.recorder(dataRecRate)
    array1Element2PrescribedTransDataLog = array1PrescribedElementTransList[1].translatingBodyOutMsg.recorder(dataRecRate)
    array1Element3PrescribedTransDataLog = array1PrescribedElementTransList[2].translatingBodyOutMsg.recorder(dataRecRate)
    array1Element4PrescribedTransDataLog = array1PrescribedElementTransList[3].translatingBodyOutMsg.recorder(dataRecRate)
    array1Element5PrescribedTransDataLog = array1PrescribedElementTransList[4].translatingBodyOutMsg.recorder(dataRecRate)
    array2Element1PrescribedTransDataLog = array2PrescribedElementTransList[0].translatingBodyOutMsg.recorder(dataRecRate)
    array2Element2PrescribedTransDataLog = array2PrescribedElementTransList[1].translatingBodyOutMsg.recorder(dataRecRate)
    array2Element3PrescribedTransDataLog = array2PrescribedElementTransList[2].translatingBodyOutMsg.recorder(dataRecRate)
    array2Element4PrescribedTransDataLog = array2PrescribedElementTransList[3].translatingBodyOutMsg.recorder(dataRecRate)
    array2Element5PrescribedTransDataLog = array2PrescribedElementTransList[4].translatingBodyOutMsg.recorder(dataRecRate)
    array1Element1PrescribedRotDataLog = array1PrescribedElementRotList[0].spinningBodyOutMsg.recorder(dataRecRate)
    array1Element2PrescribedRotDataLog = array1PrescribedElementRotList[1].spinningBodyOutMsg.recorder(dataRecRate)
    array1Element3PrescribedRotDataLog = array1PrescribedElementRotList[2].spinningBodyOutMsg.recorder(dataRecRate)
    array1Element4PrescribedRotDataLog = array1PrescribedElementRotList[3].spinningBodyOutMsg.recorder(dataRecRate)
    array1Element5PrescribedRotDataLog = array1PrescribedElementRotList[4].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element1PrescribedRotDataLog = array2PrescribedElementRotList[0].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element2PrescribedRotDataLog = array2PrescribedElementRotList[1].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element3PrescribedRotDataLog = array2PrescribedElementRotList[2].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element4PrescribedRotDataLog = array2PrescribedElementRotList[3].spinningBodyOutMsg.recorder(dataRecRate)
    array2Element5PrescribedRotDataLog = array2PrescribedElementRotList[4].spinningBodyOutMsg.recorder(dataRecRate)

    scSim.AddModelToTask(fswTaskName, scStateData)
    scSim.AddModelToTask(fswTaskName, array1Element1PrescribedTransDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element2PrescribedTransDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element3PrescribedTransDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element4PrescribedTransDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element5PrescribedTransDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element1PrescribedTransDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element2PrescribedTransDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element3PrescribedTransDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element4PrescribedTransDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element5PrescribedTransDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element1PrescribedRotDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element2PrescribedRotDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element3PrescribedRotDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element4PrescribedRotDataLog)
    scSim.AddModelToTask(fswTaskName, array1Element5PrescribedRotDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element1PrescribedRotDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element2PrescribedRotDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element3PrescribedRotDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element4PrescribedRotDataLog)
    scSim.AddModelToTask(fswTaskName, array2Element5PrescribedRotDataLog)

    # Set up Vizard visualization
    scBodyList = [scObject]
    for i in range(numArrayElements):
        scBodyList.append(["Array1Element" + str(i+1), array1ElementList[i].prescribedMotionConfigLogOutMsg])
        scBodyList.append(["Array2Element" + str(i+1), array2ElementList[i].prescribedMotionConfigLogOutMsg])

    viz = vizSupport.enableUnityVisualization(scSim, dynTaskName, scBodyList, saveFile=filename)

    vizSupport.createCustomModel(viz
                                 , simBodiesToModify=[scObject.ModelTag]
                                 , modelPath="CYLINDER"
                                 , scale=[widthHub, depthHub, lengthHub]
                                 , color=vizSupport.toRGBA255("gray"))

    for i in range(numArrayElements):
        vizSupport.createCustomModel(viz
                                     , simBodiesToModify=["Array1Element" + str(i+1)]
                                     , modelPath="CUBE"
                                     , scale=[l, w, d]
                                     , color=vizSupport.toRGBA255("green"))
        vizSupport.createCustomModel(viz
                                     , simBodiesToModify=["Array2Element" + str(i+1)]
                                     , modelPath="CUBE"
                                     , scale=[l, w, d]
                                     , color=vizSupport.toRGBA255("blue"))

    viz.settings.showSpacecraftAsSprites = -1

    # Initialize the simulation
    scSim.InitializeSimulation()

    # Set the simulation time
    simTime1 = tDeploy1 + 10  # [s]
    scSim.ConfigureStopTime(macros.sec2nano(simTime1))

    # Begin the simulation
    scSim.ExecuteSimulation()

    # Initialize the prescribedTrans module configuration data
    tDeploy2 = 20 * 60  # [s]
    tCoast2 = tDeploy2 - (2 * tRamp)  # [s]

    array1MaxTransAccelList = []
    for i in range(numArrayElements):
        transPosInit = array1TransPosInit  # [m]
        transPosRef = i * lenPanel  # [m]

        # Determine the position and velocity at the end of the ramp segment/start of the coast segment
        if (transPosInit < transPosRef):
            transAccelMax = (transPosRef - transPosInit) / ((tCoast2 * tRamp) + (tRamp * tRamp))  # [m/s^2]
        else:
            transAccelMax = (transPosRef - transPosInit) / - ((tCoast2 * tRamp) + (tRamp * tRamp))  # [m/s^2]

        array1MaxTransAccelList.append(transAccelMax)

    # Create the array element reference messages
    array1ElementTransMessageList2 = list()
    for i in range(numArrayElements):
        array1PrescribedElementTransList[i].transAccelMax = array1MaxTransAccelList[i]  # [m/s^2]

        array1ElementMessageData = messaging.PrescribedTransMsgPayload()
        array1ElementMessageData.scalarPos = i * lenPanel  # [m]
        array1ElementMessageData.scalarVel = 0.0  # [m/s]
        array1ElementTransMessageList2.append(messaging.PrescribedTransMsg().write(array1ElementMessageData))

        # Connect the translational messages to the translational profilers
        array1PrescribedElementTransList[i].translatingBodyInMsg.subscribeTo(array1ElementTransMessageList2[i])

    # Set the simulation time
    simTime2 = tDeploy2 + 10  # [s]
    scSim.ConfigureStopTime(macros.sec2nano(simTime1 + simTime2))

    # Begin the simulation
    scSim.ExecuteSimulation()

    # Initialize the prescribedRot1DOF module configuration data
    tDeploy3 = 5 * 60  # [s]
    tCoast3 = tDeploy3 - (2 * tRamp)

    array2MaxRotAccelList2 = []
    for j in range(2):
        for i in range(numArrayElements):
            thetaInit = array2ThetaInit  # [rad]
            thetaRef = 180.0 * macros.D2R  # [rad]

            # Determine the angle and angle rate at the end of the ramp segment/start of the coast segment
            if (thetaInit < thetaRef):
                thetaDDotMax = (thetaRef - thetaInit) / ((tCoast3 * tRamp) + (tRamp * tRamp))
            else:
                thetaDDotMax = (thetaRef - thetaInit) / - ((tCoast3 * tRamp) + (tRamp * tRamp))

            array2MaxRotAccelList2.append(thetaDDotMax)

    # Create the array element reference angle messages
    array2ElementRotMessageList2 = list()
    for i in range(numArrayElements):
        array2PrescribedElementRotList[i].thetaDDotMax = array2MaxRotAccelList2[i]  # [rad/s^2]

        array2ElementMessageData = messaging.HingedRigidBodyMsgPayload()
        array2ElementMessageData.theta = 180.0 * macros.D2R  # [rad]
        array2ElementMessageData.thetaDot = 0.0  # [rad/s]
        array2ElementRotMessageList2.append(messaging.HingedRigidBodyMsg().write(array2ElementMessageData))

        # Connect the rotation angle messages to the rotational profilers
        array2PrescribedElementRotList[i].spinningBodyInMsg.subscribeTo(array2ElementRotMessageList2[i])

    # Set the simulation time
    simTime3 = tDeploy3 + 10  # [s]
    scSim.ConfigureStopTime(macros.sec2nano(simTime1 + simTime2 + simTime3))

    # Begin the simulation
    scSim.ExecuteSimulation()

    # Initialize the prescribedTrans module configuration data
    tDeploy4 = 20 * 60  # [s]
    tCoast4 = tDeploy4 - (2 * tRamp)

    array2MaxTransAccelList = []
    for i in range(numArrayElements):
        transPosInit = array2TransPosInit  # [m]
        transPosRef = - (i * lenPanel)  # [m]

        # Determine the position and velocity at the end of the ramp segment/start of the coast segment
        if (transPosInit < transPosRef):
            transAccelMax = (transPosRef - transPosInit) / ((tCoast4 * tRamp) + (tRamp * tRamp))  # [m/s^2]
        else:
            transAccelMax = (transPosRef - transPosInit) / - ((tCoast4 * tRamp) + (tRamp * tRamp))  # [m/s^2]

        array2MaxTransAccelList.append(transAccelMax)

    # Create the array element reference messages
    array2ElementTransMessageList2 = list()
    for i in range(numArrayElements):
        array2PrescribedElementTransList[i].transAccelMax = array2MaxTransAccelList[i]  # [m/s^2]

        array2ElementMessageData = messaging.PrescribedTransMsgPayload()
        array2ElementMessageData.scalarPos = - (i * lenPanel)  # [m]
        array2ElementMessageData.scalarVel = 0.0  # [m/s]
        array2ElementTransMessageList2.append(messaging.PrescribedTransMsg().write(array2ElementMessageData))

        # Connect the translational messages to the translational profilers
        array2PrescribedElementTransList[i].translatingBodyInMsg.subscribeTo(array2ElementTransMessageList2[i])

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

    transPos_array1Element1 = array1Element1PrescribedTransDataLog.scalarPos  # [m]
    transPos_array1Element2 = array1Element2PrescribedTransDataLog.scalarPos  # [m]
    transPos_array1Element3 = array1Element3PrescribedTransDataLog.scalarPos  # [m]
    transPos_array1Element4 = array1Element4PrescribedTransDataLog.scalarPos  # [m]
    transPos_array1Element5 = array1Element5PrescribedTransDataLog.scalarPos  # [m]
    transPos_array2Element1 = array2Element1PrescribedTransDataLog.scalarPos  # [m]
    transPos_array2Element2 = array2Element2PrescribedTransDataLog.scalarPos  # [m]
    transPos_array2Element3 = array2Element3PrescribedTransDataLog.scalarPos  # [m]
    transPos_array2Element4 = array2Element4PrescribedTransDataLog.scalarPos  # [m]
    transPos_array2Element5 = array2Element5PrescribedTransDataLog.scalarPos  # [m]
    transVel_array1Element1 = array1Element1PrescribedTransDataLog.scalarVel  # [m/s]
    transVel_array1Element2 = array1Element2PrescribedTransDataLog.scalarVel  # [m/s]
    transVel_array1Element3 = array1Element3PrescribedTransDataLog.scalarVel  # [m/s]
    transVel_array1Element4 = array1Element4PrescribedTransDataLog.scalarVel  # [m/s]
    transVel_array1Element5 = array1Element5PrescribedTransDataLog.scalarVel  # [m/s]
    transVel_array2Element1 = array2Element1PrescribedTransDataLog.scalarVel  # [m/s]
    transVel_array2Element2 = array2Element2PrescribedTransDataLog.scalarVel  # [m/s]
    transVel_array2Element3 = array2Element3PrescribedTransDataLog.scalarVel  # [m/s]
    transVel_array2Element4 = array2Element4PrescribedTransDataLog.scalarVel  # [m/s]
    transVel_array2Element5 = array2Element5PrescribedTransDataLog.scalarVel  # [m/s]

    theta_array1Element1 = array1Element1PrescribedRotDataLog.theta * macros.R2D  # [deg]
    theta_array1Element2 = array1Element2PrescribedRotDataLog.theta * macros.R2D  # [deg]
    theta_array1Element3 = array1Element3PrescribedRotDataLog.theta * macros.R2D  # [deg]
    theta_array1Element4 = array1Element4PrescribedRotDataLog.theta * macros.R2D  # [deg]
    theta_array1Element5 = array1Element5PrescribedRotDataLog.theta * macros.R2D  # [deg]
    theta_array2Element1 = array2Element1PrescribedRotDataLog.theta * macros.R2D  # [deg]
    theta_array2Element2 = array2Element2PrescribedRotDataLog.theta * macros.R2D  # [deg]
    theta_array2Element3 = array2Element3PrescribedRotDataLog.theta * macros.R2D  # [deg]
    theta_array2Element4 = array2Element4PrescribedRotDataLog.theta * macros.R2D  # [deg]
    theta_array2Element5 = array2Element5PrescribedRotDataLog.theta * macros.R2D  # [deg]
    thetaDot_array1Element1 = array1Element1PrescribedRotDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array1Element2 = array1Element2PrescribedRotDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array1Element3 = array1Element3PrescribedRotDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array1Element4 = array1Element4PrescribedRotDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array1Element5 = array1Element5PrescribedRotDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element1 = array2Element1PrescribedRotDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element2 = array2Element2PrescribedRotDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element3 = array2Element3PrescribedRotDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element4 = array2Element4PrescribedRotDataLog.thetaDot * macros.R2D  # [deg/s]
    thetaDot_array2Element5 = array2Element5PrescribedRotDataLog.thetaDot * macros.R2D  # [deg/s]

    # Write hub angular velocity data to a text file for deployment scenario comparison
    omega_BN_B_teles_lanyard_data_file = open(r"/Users/leahkiner/Desktop/GNCBreck2024/DataForPlotting/omega_BN_B_teles_lanyard_data.txt", "w+")
    np.savetxt(omega_BN_B_teles_lanyard_data_file, omega_BN_B)
    omega_BN_B_teles_lanyard_data_file.close()

    plt.close("all")

    # Plot array 1 element positions
    plt.figure()
    plt.clf()
    plt.plot(timespan, transPos_array1Element1, label=r'$l_1$')
    plt.plot(timespan, transPos_array1Element2, label=r'$l_2$')
    plt.plot(timespan, transPos_array1Element3, label=r'$l_3$')
    plt.plot(timespan, transPos_array1Element4, label=r'$l_4$')
    plt.plot(timespan, transPos_array1Element5, label=r'$l_5$')
    # plt.title(r'Array 1 Element Translational Positions', fontsize=16)
    plt.ylabel('(m)', fontsize=14)
    plt.xlabel('Time (min)', fontsize=14)
    plt.legend(loc='upper left', prop={'size': 12})
    plt.grid(True)

    # Plot array 2 element angles
    plt.figure()
    plt.clf()
    plt.plot(timespan, transPos_array2Element1, label=r'$l_1$')
    plt.plot(timespan, transPos_array2Element2, label=r'$l_2$')
    plt.plot(timespan, transPos_array2Element3, label=r'$l_3$')
    plt.plot(timespan, transPos_array2Element4, label=r'$l_4$')
    plt.plot(timespan, transPos_array2Element5, label=r'$l_5$')
    # plt.title(r'Array 2 Element Translational Positions', fontsize=16)
    plt.ylabel('(m)', fontsize=14)
    plt.xlabel('Time (min)', fontsize=14)
    plt.legend(loc='lower left', prop={'size': 12})
    plt.grid(True)

    # # Plot array 1 element velocities
    # plt.figure()
    # plt.clf()
    # plt.plot(timespan, transVel_array1Element1, label=r'$\dot{l}_1$')
    # plt.plot(timespan, transVel_array1Element2, label=r'$\dot{l}_2$')
    # plt.plot(timespan, transVel_array1Element3, label=r'$\dot{l}_3$')
    # plt.plot(timespan, transVel_array1Element4, label=r'$\dot{l}_4$')
    # plt.plot(timespan, transVel_array1Element5, label=r'$\dot{l}_5$')
    # plt.title(r'Array 1 Element Translational Velocities', fontsize=16)
    # plt.ylabel('(m/s)', fontsize=14)
    # plt.xlabel('Time (min)', fontsize=14)
    # plt.legend(loc='center right', prop={'size': 12})
    # plt.grid(True)
    #
    # # Plot array 2 element velocities
    # plt.figure()
    # plt.clf()
    # plt.plot(timespan, transVel_array2Element1, label=r'$\dot{l}_1$')
    # plt.plot(timespan, transVel_array2Element2, label=r'$\dot{l}_2$')
    # plt.plot(timespan, transVel_array2Element3, label=r'$\dot{l}_3$')
    # plt.plot(timespan, transVel_array2Element4, label=r'$\dot{l}_4$')
    # plt.plot(timespan, transVel_array2Element5, label=r'$\dot{l}_5$')
    # plt.title(r'Array 1 Element Translational Velocities', fontsize=16)
    # plt.ylabel('(m/s)', fontsize=14)
    # plt.xlabel('Time (min)', fontsize=14)
    # plt.legend(loc='center right', prop={'size': 12})
    # plt.grid(True)

    # Plot array 1 element angles
    plt.figure()
    plt.clf()
    plt.plot(timespan, theta_array1Element1, label=r'$\theta_1$')
    plt.plot(timespan, theta_array1Element2, label=r'$\theta_2$')
    plt.plot(timespan, theta_array1Element3, label=r'$\theta_3$')
    plt.plot(timespan, theta_array1Element4, label=r'$\theta_4$')
    plt.plot(timespan, theta_array1Element5, label=r'$\theta_5$')
    # plt.title(r'Array 1 Element Angles', fontsize=16)
    plt.ylabel('(deg)', fontsize=14)
    plt.xlabel('Time (min)', fontsize=14)
    plt.legend(loc='center right', prop={'size': 12})
    plt.grid(True)

    # Plot array 2 element angles
    plt.figure()
    plt.clf()
    plt.plot(timespan, theta_array2Element1, label=r'$\theta_1$')
    plt.plot(timespan, theta_array2Element2, label=r'$\theta_2$')
    plt.plot(timespan, theta_array2Element3, label=r'$\theta_3$')
    plt.plot(timespan, theta_array2Element4, label=r'$\theta_4$')
    plt.plot(timespan, theta_array2Element5, label=r'$\theta_5$')
    # plt.title(r'Array 2 Element Angles', fontsize=16)
    plt.ylabel('(deg)', fontsize=14)
    plt.xlabel('Time (min)', fontsize=14)
    plt.legend(loc='center right', prop={'size': 12})
    plt.grid(True)

    # # Plot array 1 element angle rates
    # plt.figure()
    # plt.clf()
    # plt.plot(timespan, thetaDot_array1Element1, label=r'$\dot{\theta}_1$')
    # plt.plot(timespan, thetaDot_array1Element2, label=r'$\dot{\theta}_2$')
    # plt.plot(timespan, thetaDot_array1Element3, label=r'$\dot{\theta}_3$')
    # plt.plot(timespan, thetaDot_array1Element4, label=r'$\dot{\theta}_4$')
    # plt.plot(timespan, thetaDot_array1Element5, label=r'$\dot{\theta}_5$')
    # plt.title(r'Array 1 Element Angle Rates', fontsize=16)
    # plt.ylabel('(deg/s)', fontsize=14)
    # plt.xlabel('Time (min)', fontsize=14)
    # plt.legend(loc='center right', prop={'size': 12})
    # plt.grid(True)
    #
    # # Plot array 2 element angle rates
    # plt.figure()
    # plt.clf()
    # plt.plot(timespan, thetaDot_array2Element1, label=r'$\dot{\theta}_1$')
    # plt.plot(timespan, thetaDot_array2Element2, label=r'$\dot{\theta}_2$')
    # plt.plot(timespan, thetaDot_array2Element3, label=r'$\dot{\theta}_3$')
    # plt.plot(timespan, thetaDot_array2Element4, label=r'$\dot{\theta}_4$')
    # plt.plot(timespan, thetaDot_array2Element5, label=r'$\dot{\theta}_5$')
    # plt.title(r'Array 2 Element Angle Rates', fontsize=16)
    # plt.ylabel('(deg/s)', fontsize=14)
    # plt.xlabel('Time (min)', fontsize=14)
    # plt.legend(loc='center left', prop={'size': 12})
    # plt.grid(True)

    # Plot r_BN_N
    plt.figure()
    plt.clf()
    plt.plot(timespan, r_BN_N[:, 0], label=r'$r_{1}$')
    plt.plot(timespan, r_BN_N[:, 1], label=r'$r_{2}$')
    plt.plot(timespan, r_BN_N[:, 2], label=r'$r_{3}$')
    # plt.title(r'${}^\mathcal{N} r_{\mathcal{B}/\mathcal{N}}$ Spacecraft Inertial Trajectory', fontsize=16)
    plt.ylabel('(m)', fontsize=14)
    plt.xlabel('Time (min)', fontsize=14)
    plt.legend(loc='lower left', prop={'size': 12})
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
    plt.legend(loc='lower left', prop={'size': 12})
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