from scipy.linalg import block_diag
import scipy as sp
import logging
import helics as h
import json
import numpy as np
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional, Union
from scipy.optimize import least_squares
import scipy.sparse
from scipy.io import loadmat, savemat 
from scipy.optimize import minimize
# import random, sys 
import os, pickle
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import (
    AdmittanceMatrix,
    AdmittanceSparse,
    Complex,
    PowersImaginary,
    PowersReal,
    Topology,
    VoltagesAngle,
    VoltagesMagnitude,
)
from datetime import datetime


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def anequ(Num, StateVar, MeaPha):
    byphase = MeaPha.AN  # Assuming MeaPha is a dictionary with 'AN' key in Python
    row_v = len(byphase)
    equ_an = np.zeros(row_v)
    
    for row in range(row_v):
        i = byphase[row][0]  # Adjust for 0-based indexing in Python
        d = byphase[row][1]  # Assumes d is used as-is, 1-based indexing considered in formula
        vx = StateVar[2*Num.Node*(d-1)+2*i-1]
        vy = StateVar[2*Num.Node*(d-1)+2*i]
        equ_an[row] = np.arctan2(vy, vx)
        
    return equ_an

def cequ(Num,StateVar,B,G,Loc):
    byphase = Loc.Zeroinj
    row_p = len(byphase)
    equ_c = np.zeros((row_p))
    equ_d = np.copy(equ_c)

    ## Jacobian for Node Voltage
    for row in range(0,row_p):

        i = Loc.Zeroinj[row][0]
        d = Loc.Zeroinj[row][1]
        x = 2*Num.Node*(d) + 2*i
        y = x + 1

        for k in range(0,Num.Node):
            for t in range(0,3):

                xx = 2*Num.Node*(t)+2*k
                yy = xx + 1

                y_int = int(y/2-0.5)
                # y_int = y_int.astype(int)
                yy_int = int(yy/2-0.5)
                # yy_int = yy_int.astype(int)

                ekt = StateVar[xx][0]
                fkt = StateVar[yy][0]
                Gikdt = G[y_int,yy_int]
                Bikdt = B[y_int,yy_int]

                equ_c[row] = equ_c[row] + Gikdt*ekt - Bikdt*fkt
                equ_d[row] = equ_d[row] + Gikdt*fkt + Bikdt*ekt

    return equ_c, equ_d


def plequ(Num,StateVar,MeaPha,B,G,Loc):

    byphase = MeaPha.PL
    row_p = len(byphase)
    equ_pl = np.zeros(row_p)
    equ_ql  = np.copy(equ_pl)
    TermC  = np.zeros(row_p)
    TermD  = np.zeros(row_p)

    ## Jacobian for Node Voltage
    for row in range(0,row_p):

        j = byphase[row][0]
        d = byphase[row][1]

        i = Loc.Line[j][0]
        k = Loc.Line[j][1]

        for t in range(0,3):

            Gikdt = G[Num.Node*(d-1)+i,Num.Node*(t-1)+k]
            Bikdt = B[Num.Node*(d-1)+i,Num.Node*(t-1)+k]

            eit = StateVar[2*Num.Node*(t-1)+2*i-1].item()
            fit = StateVar[2*Num.Node*(t-1)+2*i].item()
            ekt = StateVar[2*Num.Node*(t-1)+2*k-1].item()
            fkt = StateVar[2*Num.Node*(t-1)+2*k].item()

            TermC[row] = TermC[row] + (Gikdt*eit-Bikdt*fit)-(Gikdt*ekt-Bikdt*fkt)
            TermD[row] = TermD[row] + (Bikdt*eit+Gikdt*fit)-(Bikdt*ekt+Gikdt*fkt)

        ekd = StateVar[2*Num.Node*(d-1)+2*k-1].item()
        fkd = StateVar[2*Num.Node*(d-1)+2*k].item()
        Vkd = ekd + 1j*fkd
        Ijd = TermC[row] + 1j*TermD[row]
        Sflow = Vkd*np.conj(Ijd)
        equ_pl[row] = -Sflow.real
        equ_ql[row] = -Sflow.imag


    return equ_pl, equ_ql


def vnequ(Num, StateVar, MeaPha):
    StateVar = StateVar.flatten()
    equ_vn = StateVar[MeaPha.VN[:,0]]
    return equ_vn


def pnequ(Num, StateVar, MeaPha, B, G):
    npi = len(MeaPha.PN)
    nqi = len(MeaPha.QN)
    nbus = Num.Node

    V = StateVar[:nbus]  # Initialize the bus voltages
    del_ = StateVar[nbus:]  # Initialize the bus angles

    ppi = MeaPha.PN[:, 0]
    qi = MeaPha.QN[:, 0]

    h2 = np.zeros((npi, 1))
    h3 = np.zeros((nqi, 1))

    # Power Injection Measure Equ
    for i in range(npi):
        m = ppi[i]
        for k in range(nbus):
            h2[i] += V[m] * V[k] * (G[m, k] * np.cos(del_[m] - del_[k]) + B[m, k] * np.sin(del_[m] - del_[k]))

    for i in range(nqi):
        m = qi[i]
        for k in range(nbus):
            h3[i] += V[m] * V[k] * (G[m, k] * np.sin(del_[m] - del_[k]) - B[m, k] * np.cos(del_[m] - del_[k]))

    b = -B  # Line conductance matrix
    g = -G  # Line susceptance matrix
    P = np.zeros((npi, 1))  # Initialize node injection power
    Q = np.zeros((npi, 1))
    PP = np.zeros((3, 3))  # Line injection power
    QQ = np.zeros((3, 3))
    equ_pn = h2
    equ_qn = h3

    return equ_pn, equ_qn


def cjoc(Num,B,G,Loc):

    joc_c = np.zeros((Num.Zeroinj, Num.StateVar*Num.Node))
    joc_d = np.copy(joc_c)
    Loc.Zeroinj = np.array(Loc.Zeroinj)

    ## Jacobian for Node Voltage
    for row in range(0,len(Loc.Zeroinj)):

        i = Loc.Zeroinj[row,0]
        d = Loc.Zeroinj[row,1]
        x = 2*Num.Node*(d) + 2*i
        y = x + 1

        # convert it to integer
        x = x.astype(int)
        y = y.astype(int)

        for k in range(0,Num.Node):
            for t in range(0,3):

                xx = 2*Num.Node*(t)+2*k
                yy = xx + 1

                y_int = y/2-0.5
                y_int = y_int.astype(int)
                yy_int = yy/2-0.5
                yy_int = yy_int.astype(int)


                Gikdt = G[y_int,yy_int]
                Bikdt = B[y_int,yy_int]
                joc_c[row,xx] = Gikdt
                joc_c[row,yy] = -Bikdt
                joc_d[row,xx] = Bikdt
                joc_d[row,yy] = Gikdt

    return joc_c, joc_d

def vnjoc(Num, StateVar, MeaPha):
    nvi = len(MeaPha.VN)
    nbus = Num.Node 

    # Jacobian
    H11 = np.zeros((nvi, nbus))

    # H12 - Derivative of V with respect to V..
    H12 = np.zeros((nvi, nbus))
    for k in range(nvi):
        for n in range(nbus):
            if n == k:
                H12[k, n] = 1

    joc_vn = np.hstack((H12, H11))
    return joc_vn

## Jacobian ant measurement equations
def pnjoc(Num,StateVar,MeaPha,B,G):

    npi = len(MeaPha.PN)
    nqi = len(MeaPha.QN)
    nbus = Num.Node 
    StateVar = StateVar.flatten()
    V = StateVar[:nbus]   # Initialize the bus voltages
    V = V.tolist()
    del_ = StateVar[nbus:]  # Initialize the bus angles
    H21 = np.zeros((npi, nbus))
    for i in range(npi):
        m = i
        for k in range(nbus):
            if k == m:
                for n in range(nbus):
                    H21[i, k] += V[m] * V[n] * (-G[m, n] * np.sin(del_[m] - del_[n]) + B[m, n] * np.cos(del_[m] - del_[n]))
                H21[i, k] -= V[m] ** 2 * B[m, m]
            else:
                H21[i, k] = V[m] * V[k] * (G[m, k] * np.sin(del_[m] - del_[k]) - B[m, k] * np.cos(del_[m] - del_[k]))

    # H22 - Derivative of Real Power Injections with V
    H22 = np.zeros((npi, nbus))
    for i in range(npi):
        m = i
        for k in range(nbus):
            if k == m:
                for n in range(nbus):
                    H22[i, k] += V[n] * (G[m, n] * np.cos(del_[m] - del_[n]) + B[m, n] * np.sin(del_[m] - del_[n]))
                H22[i, k] += V[m] * G[m, m]
            else:
                H22[i, k] = V[m] * (G[m, k] * np.cos(del_[m] - del_[k]) + B[m, k] * np.sin(del_[m] - del_[k]))

    # H31 - Derivative of Reactive Power Injections with Angles
    H31 = np.zeros((nqi, nbus))
    for i in range(nqi):
        m = i
        for k in range(nbus):
            if k == m:
                for n in range(nbus):
                    H31[i, k] += V[m] * V[n] * (G[m, n] * np.cos(del_[m] - del_[n]) + B[m, n] * np.sin(del_[m] - del_[n]))
                H31[i, k] -= V[m] ** 2 * G[m, m]
            else:
                H31[i, k] = V[m] * V[k] * (-G[m, k] * np.cos(del_[m] - del_[k]) - B[m, k] * np.sin(del_[m] - del_[k]))

    # H32 - Derivative of Reactive Power Injections with V
    H32 = np.zeros((nqi, nbus))
    for i in range(nqi):
        m = i
        for k in range(nbus):
            if k == m:
                for n in range(nbus):
                    H32[i, k] += V[n] * (G[m, n] * np.sin(del_[m] - del_[n]) - B[m, n] * np.cos(del_[m] - del_[n]))
                H32[i, k] -= V[m] * B[m, m]
            else:
                H32[i, k] = V[m] * (G[m, k] * np.sin(del_[m] - del_[k]) - B[m, k] * np.cos(del_[m] - del_[k]))

    joc_pn = np.hstack((H22, H21))
    joc_qn = np.hstack((H32, H31))
    
    return joc_pn, joc_qn



def computeZero(A):
    N = A.shape[0]
    zero_row = []
    zero_col = []
    for i in range(N):
        if np.sum(np.abs(A[i, :]) < 1e-6) == N:
            zero_row.append(i)
        if np.sum(np.abs(A[:, i]) < 1e-6) == N:
            zero_col.append(i)
    return zero_row, zero_col



def Ini_val(Flag,Num,puTrue,SlackBus):

    Ve1 = np.reshape(puTrue.Ve, (Num.Node, 1))
    Vf1 = np.reshape(puTrue.Vf, (Num.Node, 1))
    VV1 = np.vstack([Ve1, Vf1])
    
    return VV1 




def WeightMatrix(MeaIdx,SensorError):
    WPN = SensorError.PN*np.ones(len(MeaIdx.PN))
    WQN = SensorError.QN*np.ones(len(MeaIdx.QN))
    WVN = SensorError.VN*np.ones(len(MeaIdx.VN))
    WAN = SensorError.AN*np.ones(len(MeaIdx.AN))
    WPL = SensorError.PL*np.ones(len(MeaIdx.PL))
    WQL = SensorError.QL*np.ones(len(MeaIdx.QL))

    WPN = np.diag(WPN**2)
    WQN = np.diag(WQN**2)
    WVN = np.diag(WVN**2)
    WAN = np.diag(WAN**2)
    WPL = np.diag(WPL**2)
    WQL = np.diag(WQL**2)

    R = block_diag(WPN, WQN, WVN, WAN, WPL, WQL)

    return R

def idx2pha(index, n):
    index = np.array(index)  # Convert index to a NumPy array
    a = index / n
    a1 = np.where((a >= 0) & (a < 1))[0]
    a2 = np.where((a >= 1) & (a < 2))[0]
    a3 = np.where((a >= 2))[0]    
    byphase = np.zeros((index.shape[0], 2), dtype=int)
    byphase[a1, 0] = index[a1]
    byphase[a1, 1] = 1
    byphase[a2, 0] = index[a2] - n
    byphase[a2, 1] = 2
    byphase[a3, 0] = index[a3] - 2 * n
    byphase[a3, 1] = 3

    return byphase




def Zero_inj(A):
    # Initialize count of zero injections and list to store their indices
    num_zero_injec = 0
    node_zero_injec = []
    # Iterate over the array's elements
    for i in range(len(A)):  # Rows
        for d in range(0):  # Columns
            if abs(A[i]) < 1e-3:
                num_zero_injec += 1
                node_zero_injec.append((i))

    return num_zero_injec, node_zero_injec


def pha2idx(byphase,n):
    if len(byphase) == 0:
        index = []
    else:
        # print("byphase: ", byphase)
        node = byphase[:,0]
        phase = byphase[:,1]
        row = len(byphase)
        #index = np.zeros((row,1))
        index = np.zeros((row))
        for i in np.arange(0,row):
            if phase[i] == 1:
                index[i] = node[i]
            else:
                if phase[i] == 2:
                    index[i] = node[i] + n
                else:
                    if phase[i] == 3:
                        index[i] = node[i] + 2 * n
    index = index.astype(int)
    return index

def findCharacterLocation(charArray, targetStr):
    # Remove spaces from the target string
    targetCharArray = ''.join(targetStr.split())

    indices = []
    for i, row in enumerate(charArray):
        # Remove spaces from each row and compare
        rowCharArrayNoSpaces = ''.join(row.split())
        if rowCharArrayNoSpaces == targetCharArray:
            indices.append(i)
    return indices


def calcIndex(a):
    
    lenOfList = len(a)
    # a_split = [node.split('.') for node in node_list_strings]
    a_split = [node.split('.') for node in a]
    a_node = [x[0] for x in a_split]
    
    a_phase = np.array([int(x[1]) for x in a_split])
    unique_elements, indices = np.unique(a_node, return_index=True, axis=0)

    # Then, use these indices to sort the unique_elements array according to the order of their first appearance.
    sorted_indices = np.argsort(indices)
    setOfNodes = unique_elements[sorted_indices]
    numNodes = len(setOfNodes)
    a_new = setOfNodes
    index = np.full(lenOfList, np.nan)

    # Loop through each element in the a_node list
    for iii in range(lenOfList):
        idxNode = None
        # Loop through each node in setOfNodes to find a match
        for jjj in range(numNodes):
            if a_node[iii] == setOfNodes[jjj]:
                idxNode = jjj + 1  # +1 to align with MATLAB's 1-based indexing
                break
        
        # Compute idxPhase
        idxPhase = a_phase[iii]
        
        # Assign computed value to the index array
        if idxNode is not None:  # Ensure idxNode was found
            index[iii] = idxNode + numNodes * (idxPhase - 1)

    # Calculate numDim based on the length of a_new and multiplying by 3
    numDim = len(a_new) * 3

    return a_new, index, numDim

def funConvertArrayToThreePhaseMatrix(b, index, numDim):
    # Create a zero array of the specified size with complex data type to store complex numbers
    b1 = np.zeros((numDim, 1), dtype=complex)    
    # Flatten the input array b to ensure it can be used in the assignment without shape mismatch
    b_flat = b.flatten()    
    b1[index.astype(int) - 1, 0] = b_flat
    b_new = b1.reshape((numDim // 3, 3), order='F')

    return b_new

def funConvertYbusToThreePhaseMatrix(c, index, numDim):
    c_new = np.zeros((numDim, numDim), dtype=complex)
    
    index_zero_based = index.astype(int) - 1
    
    for i in range(len(index_zero_based)):
        for j in range(len(index_zero_based)):
            c_new[index_zero_based[i], index_zero_based[j]] = c[i, j]
    

    return c_new


def matrix_to_numpy(admittance: List[List[Complex]]):
    "Convert list of list of our Complex type into a numpy matrix"
    return np.array([[x[0] + 1j * x[1] for x in row] for row in admittance])


class UnitSystem(str, Enum):
    SI = "SI"
    PER_UNIT = "PER_UNIT"


class Power:
    def __init__(self, ids, values):
        self.ids = ids
        self.values = values


class AlgorithmParameters(BaseModel):
    tol: float = 5e-7
    units: UnitSystem = UnitSystem.PER_UNIT
    base_power: Optional[float] = 100.0

    class Config:
        use_enum_values = True


def get_indices(topology, measurement):
    "Get list of indices in the topology for each index of the input measurement"
    inv_map = {v: i for i, v in enumerate(topology.base_voltage_magnitudes.ids)}
    return [inv_map[v] for v in measurement.ids]




def get_y(admittance: Union[AdmittanceMatrix, AdmittanceSparse], ids: List[str]):
    if type(admittance) == AdmittanceMatrix:
        assert ids == admittance.ids
        return matrix_to_numpy(admittance.admittance_matrix)
    elif type(admittance) == AdmittanceSparse:
        node_map = {name: i for (i, name) in enumerate(ids)}
        return scipy.sparse.coo_matrix(
            (
                [v[0] + 1j * v[1] for v in admittance.admittance_list],
                (
                    [node_map[r] for r in admittance.from_equipment],
                    [node_map[c] for c in admittance.to_equipment],
                ),
            )
        ).toarray()


class StateEstimatorFederate:
    "State estimator federate. Wraps state_estimation with pubs and subs"

    def __init__(
        self, 
        federate_name, 
        algorithm_parameters: AlgorithmParameters, 
        input_mapping,
        broker_config: BrokerConfig,
    ):
        "Initializes federate with name and remaps input into subscriptions"
        deltat = 0.1

        self.algorithm_parameters = algorithm_parameters

        # Create Federate Info object that describes the federate properties #
        fedinfo = h.helicsCreateFederateInfo()

        h.helicsFederateInfoSetBroker(fedinfo, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(fedinfo, broker_config.broker_port)

        fedinfo.core_name = federate_name
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_delta, deltat
        )

        self.vfed = h.helicsCreateValueFederate(federate_name, fedinfo)
        logger.info("Value federate created")

        # Register the publication #
        self.sub_voltages_magnitude = self.vfed.register_subscription(
            input_mapping["voltages_magnitude"], "V"
        )
        self.sub_power_P = self.vfed.register_subscription(
            input_mapping["powers_real"], "W"
        )
        self.sub_power_Q = self.vfed.register_subscription(
            input_mapping["powers_imaginary"], "W"
        )
        self.sub_topology = self.vfed.register_subscription(
            input_mapping["topology"], ""
        )
        self.pub_voltage_mag = self.vfed.register_publication(
            "voltage_mag", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_voltage_angle = self.vfed.register_publication(
            "voltage_angle", h.HELICS_DATA_TYPE_STRING, ""
        )

    def run(self):
        "Enter execution and exchange data"
        # Enter execution mode #
        self.vfed.enter_executing_mode()
        logger.info("Entering execution mode")

        granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)

        self.initial_ang = None
        self.initial_V = None
        topology = Topology.parse_obj(self.sub_topology.json)
        ids = topology.base_voltage_magnitudes.ids
        logger.info("Topology has been read")
        slack_index = None
        if not isinstance(topology.admittance, AdmittanceMatrix) and not isinstance(
            topology.admittance, AdmittanceSparse
        ):
            raise "Weighted Least Squares algorithm expects AdmittanceMatrix/Sparse as input"

        for i in range(len(ids)):
            if ids[i] == topology.slack_bus[0]:
                slack_index = i

        while granted_time < h.HELICS_TIME_MAXTIME:

            if not self.sub_voltages_magnitude.is_updated():
                granted_time = h.helicsFederateRequestTime(
                    self.vfed, h.HELICS_TIME_MAXTIME
                )
                continue

            logger.info("start time: " + str(datetime.now()))

            voltages = VoltagesMagnitude.parse_obj(self.sub_voltages_magnitude.json)
            power_P = PowersReal.parse_obj(self.sub_power_P.json)
            power_Q = PowersImaginary.parse_obj(self.sub_power_Q.json)    
  
            pseudo_node_names =  ['65.2', '65.3', '76.2', '76.3']
            pseudo_values_P = [0.72*35/100, 0.74*70/100, 0.73*70/100, 0.76*70/100]
            pseudo_values_Q = [0.72*25/100, 0.74*50/100, 0.73*50/100, 0.76*50/100]

            pseudo_node_P = Power(ids=pseudo_node_names, values=pseudo_values_P)
            pseudo_node_Q = Power(ids=pseudo_node_names, values=pseudo_values_Q)



            remaining_node_names =  ['150R.1', '150R.2', '150R.3', '149.1', '149.2', '149.3', '1.2', '1.3', '3.3', '8.1', '8.2', '8.3', 
                                     '13.1', '13.2', '13.3', '9R.1', '14.1', '18.1', '18.2', '18.3', '15.3', '21.1', '21.2', '21.3', '23.1', 
                                     '23.2', '23.3', '25.1', '25.2', '25.3', '25R.1', '25R.3', '26.1', '26.3', '28.2', '28.3', '27.1', '27.3', 
                                     '30.1', '30.2', '250.1', '250.2', '250.3', '36.1', '36.2', '35.3', '40.1', '40.2', '40.3', '42.2', '42.3', 
                                     '44.1', '44.2', '44.3', '50.1', '50.2', '51.2', '51.3', '151.1', '151.2', '151.3', '52.2', '52.3', '53.2', '53.3', 
                                     '54.1', '54.2', '54.3', '57.1', '57.2', '57.3', '56.1', '56.3', '60.2', '60.3', '61.1', '61.2', '61.3', '62.1', '62.2', 
                                     '64.1', '64.3', '66.1', '66.2', '67.1', '67.2', '67.3', '72.1', '72.2', '72.3', '97.1', 
                                    '97.2', '97.3', '77.1', '77.3', '86.1', '86.3', '78.1', '78.2', '78.3', '79.2', '79.3', '81.1', '81.2', '81.3', '82.2', 
                                    '82.3', '83.1', '83.2', '89.1', '89.2', '89.3', '91.1', '91.2', '91.3', '93.1', '93.2', '93.3', '95.1', '95.3', '98.2', 
                                    '98.3', '99.1', '99.3', '100.1', '100.2', '450.1', '450.2', '450.3', '197.1', '197.2', '197.3', '101.1', '101.2', '101.3', 
                                    '105.1', '105.2', '105.3', '108.1', '108.2', '108.3', '300.1', '300.2', '300.3', '110.1', '135.1', '135.2', '135.3', '152.1', 
                                    '152.2', '152.3', '160R.1', '160R.2', '160R.3', '160.1', '160.2', '160.3', '61S.1', '61S.2', '61S.3', '300_OPEN.1', '300_OPEN.2', 
                                    '300_OPEN.3', '94_OPEN.1', '610.1', '610.2', '610.3']

            
            remaining_node_names = []

            remaining_node_P = Power(ids=remaining_node_names, values=[0.0]*len(remaining_node_names))
            remaining_node_Q = Power(ids=remaining_node_names, values=[0.0]*len(remaining_node_names))
            
            power_P.ids.extend(pseudo_node_P.ids)
            power_P.values.extend(pseudo_node_P.values)

            power_Q.ids.extend(pseudo_node_Q.ids)
            power_Q.values.extend(pseudo_node_Q.values)
            
            
            power_P.ids.extend(remaining_node_P.ids)
            power_P.values.extend(remaining_node_P.values)

            power_Q.ids.extend(remaining_node_Q.ids)
            power_Q.values.extend(remaining_node_Q.values)          


            knownP = get_indices(topology, power_P)
            knownQ = get_indices(topology, power_Q)
            knownV = get_indices(topology, voltages)


            if self.initial_V is None:
                # Flat start or using average measurements
                if len(knownP) + len(knownV) + len(knownQ) > len(ids) * 2:
                    self.initial_V = 1.0
                else:
                    self.initial_V = np.mean(
                        np.array(voltages.values)
                        / np.array(topology.base_voltage_magnitudes.values)[knownV]
                    )
            if self.initial_ang is None:
                self.initial_ang = np.array(topology.base_voltage_angles.values)


            base_voltages = np.array(topology.base_voltage_magnitudes.values)
            ids = topology.base_voltage_magnitudes.ids

            base_power = 100
            
            if self.algorithm_parameters.base_power != None:
                base_power = self.algorithm_parameters.base_power
# ajay


            
            Y = get_y(topology.admittance, ids)
            # Hand-crafted unit conversion (check it, it works)
            Y = (
                base_voltages.reshape(1, -1)
                * Y
                * base_voltages.reshape(-1, 1)
                / (base_power * 1000)
            )       




            vars_to_save = {"power_P": power_P.values, "power_Q": power_Q.values, "voltage": voltages.values, 
            "Y": Y,"ids_PQ": power_P.ids, "base_power": base_power, "base_voltages": base_voltages, 
            "ids_voltage": voltages.ids, "knownP": knownP, "knownQ": knownQ, "knownV": knownV, "NodeIDs":ids}
            

            ids_PQ = power_P.ids
            power_P = np.array(power_P.values)
            power_Q = np.array(power_Q.values)
            voltage = np.array(voltages.values)
            # Y = loaded_vars['Y']
            

            # base_power = loaded_vars['base_power']
            # base_voltages = loaded_vars['base_voltages']
            ids_voltage = voltages.ids

            knownP = np.array(knownP)
            knownQ = np.array(knownQ)
            knownV = np.array(knownV)
            NodeIDs  = ids


            #  add variables here ///////////////////////

            power_P1 = np.zeros(Y.shape[1])
            power_Q1 = np.zeros(Y.shape[1])

            # Assigning values based on locations found
            for i in range(len(ids_PQ)):
                indices = findCharacterLocation(NodeIDs, ids_PQ[i])
                if indices:
                    power_P1[indices] = power_P[i]
                    power_Q1[indices] = power_Q[i]

            # Setting the first three elements as negative sums of the others divided by 3
            power_P1[:3] = -np.sum(power_P1[4:]) / 3
            power_Q1[:3] = -np.sum(power_Q1[4:]) / 3



            SCADA_VN1 = []
            for i in range(len(ids_voltage)):
                indices = findCharacterLocation(NodeIDs, ids_voltage[i])
                SCADA_VN1.extend(indices)

            # Final formatting of data
            SCADA_VN = np.array(SCADA_VN1)
            measure_VN = voltage.T
            vckt = np.array([])

            Pbig = power_P1 / base_power
            Qbig = power_Q1 / base_power
            Ybig = Y
            zeroinj_V = np.array([])
            NodeList = ids_PQ
            Lines = []

            # //////////////////////////////////////


            # # Assuming Ybig is defined elsewhere and is a numpy array
            vckt = np.ones(len(Ybig))  # size(Ybig,1),1

            Vbig = vckt.copy()

            # # node_list_strings = [str(node[0][0]) for node in NodeList]
            # # Matrix multiplication to compute Ibig
            Ibig = np.dot(Ybig, Vbig)
            Ybus = Ybig 
            V_node = Vbig 
            I_node = Ibig 
            P_node = Pbig + 1j * Qbig  # Create a complex array

            V_node = 2.401694479938087e+03 * Vbig

            class Feeder_class:
                def __init__(self, num_nodes, num_lines):
                    self.NumN = num_nodes
                    self.NumL = num_lines

            feeder = Feeder_class(len(NodeIDs), len(Lines))

            # Initialize P_line and I_line
            P_line = np.zeros((feeder.NumL, 3))
            I_line = np.zeros((feeder.NumL, 3))



            Feeder = feeder 
            SlackBus = 0


            ###### Simulation Setup
            class Base:
                def __init__(self, SlackBus, P_node, V_node):
                    # Parameters
                    self.Sbase = np.abs(P_node[SlackBus])
                    self.Vbase = np.abs(V_node[SlackBus])
                    self.Ibase = self.Sbase/self.Vbase
                    self.Ybase = self.Sbase/self.Vbase**2

            Base = Base(SlackBus,P_node,V_node)

            class puTrue:
                def __init__(self, Base, P_node, V_node, I_node, P_line):
                    # Parameters
                    self.VOL = V_node/Base.Vbase
                    self.PN = np.real(P_node)/Base.Sbase
                    self.QN = np.imag(P_node)/Base.Sbase
                    self.IN = np.abs(I_node)/Base.Ibase
                    self.PL = np.real(P_line)/Base.Sbase
                    self.QL = np.imag(P_line)/Base.Sbase
                    self.IL = np.abs(I_line)/Base.Sbase
                    self.Ve = np.real(self.VOL)
                    self.Vf = np.imag(self.VOL)
                    self.VN = np.abs(self.VOL)
                    self.AN = np.angle(self.VOL)

                    # data pre-processing
                    self.PN[np.abs(self.PN) < 1e-8] = 0
                    self.QN[np.abs(self.QN) < 1e-8] = 0
                    self.IN[np.abs(self.IN) < 1e-8] = 0
                    self.PL[np.abs(self.PL) < 1e-8] = 0
                    self.QL[np.abs(self.QL) < 1e-8] = 0
                    self.IL[np.abs(self.IL) < 1e-8] = 0

            puTrue = puTrue(Base, P_node, V_node, I_node, P_line)


            class SensorError:
                def __init__(self):
                    self.PN      = 0.1
                    self.QN      = 0.1
                    self.VN      = 0.001
                    self.AN      = 0.001
                    self.PL      = 0.001
                    self.QL      = 0.001

            SensorError = SensorError()

            ## Numbers
            class Num:
                def __init__(self,Feeder):
                    self.Node      = Feeder.NumN
                    self.Line      = Feeder.NumL
                    self.StateVar  = 3*2
                    self.Zeroinj      = []
                    self.Zerovol      = []

            class Loc:
                def __init__(self):
                    self.Zeroinj      = []
                    self.Zerovol      = []

            Num = Num(Feeder)
            Loc = Loc()


            Num.Zeroinj, Loc.Zeroinj = Zero_inj(puTrue.IN)
            Num.Zerovol, Loc.Zerovol = Zero_inj(puTrue.VN)


            PMU = np.array([])
            POW = np.array([])


            class MeaIdx:
                def __init__(self, Num, PMU, POW, SCADA_VN):
                    self.PN = np.arange(len(NodeList), dtype=int)
                    self.QN = self.PN
                    self.VN = np.array(SCADA_VN, dtype=int)
                    self.AN = np.array([], dtype=int).tolist()
                    self.PL = np.array([], dtype=int).tolist()
                    self.QL = np.array([], dtype=int).tolist()

            MeaIdx = MeaIdx(Num, PMU, POW, SCADA_VN)


            class MeaPha:
                def __init__(self,MeaIdx,Num):
                    self.PN = idx2pha(MeaIdx.PN,Num.Node)
                    self.QN = idx2pha(MeaIdx.QN,Num.Node)
                    self.VN = idx2pha(MeaIdx.VN,Num.Node)
                    self.AN = idx2pha(MeaIdx.AN,Num.Node)
                    self.PL = idx2pha(MeaIdx.PL,Num.Line)
                    self.QL = idx2pha(MeaIdx.QL,Num.Line)

            MeaPha = MeaPha(MeaIdx,Num)



            class Flag: 
                def __init__(self):
                    self.ini = 1
                    self.noise = 2
                    self.zeroinj = 0
                    self.error = 1



            Weight = WeightMatrix(MeaIdx,SensorError)
            Flag = Flag()
            if Flag.noise == 2:  # Use 'flag' instance to check the 'noise' attribute
                Weight = np.eye(Weight.shape[0])

            class Mea:
                def __init__(self, puTrue, Num):
                    self.PN = puTrue.PN 
                    self.QN = puTrue.QN 
                    self.VN = puTrue.VN**2 
                    self.AN = puTrue.AN
                    self.PL = puTrue.PL 
                    self.QL = puTrue.QL 


            Mea = Mea(puTrue, Num)


            measure = np.concatenate((Mea.PN[MeaIdx.PN], Mea.QN[MeaIdx.QN], measure_VN, Mea.AN[MeaIdx.AN], \
                                    Mea.PL[MeaIdx.PL], Mea.QL[MeaIdx.QL]), axis=None)


            StateVar = Ini_val(Flag, Num, puTrue,SlackBus)
            G = np.real(Ybus)/Base.Ybase # Real part of Ybus matrix
            B = np.imag(Ybus)/Base.Ybase # Imag part of Ybus matrix
            iterMax = 100
            tol = 1e-4

            for i in range(0,iterMax):    

                joc_pn, joc_qn = pnjoc(Num, StateVar, MeaPha, B, G)
                joc_vn = vnjoc(Num, StateVar, MeaPha)
                column_count = joc_pn.shape[1] if joc_pn.ndim > 1 else 0  # Example to infer column count

                # if joc_vn.size == 0:
                #     joc_vn = np.empty((0, column_count))
                joc_an = np.empty((0, column_count))
                joc_pl = np.empty((0, column_count))
                joc_ql = np.empty((0, column_count))
                joc_c  = np.empty((0, column_count))
                joc_d  = np.empty((0, column_count))

                # Use np.vstack or np.concatenate with axis=0 for vertical stacking if the dimensions match
                H = np.vstack((joc_pn, joc_qn, joc_vn, joc_an, joc_pl, joc_ql))   

                C = np.vstack((joc_c,joc_d))
                H[:,2*SlackBus+1:-1:2*Num.Node] = 0
                C[:,2*SlackBus+1:-1:2*Num.Node] = 0
                H[:,2*SlackBus:-1:2*Num.Node] = 0
                C[:,2*SlackBus:-1:2*Num.Node] = 0

                equ_pn, equ_qn = pnequ(Num,StateVar,MeaPha,B,G)



                equ_vn = vnequ(Num, StateVar, MeaPha)
                equ_vn = equ_vn.reshape(-1,1)                

                MeaEqu = []
                MeaEqu.append(equ_pn)
                MeaEqu.append(equ_qn)
                MeaEqu.append(equ_vn)
                # MeaEqu.append(equ_an)
                # MeaEqu.append(equ_pl)
                # MeaEqu.append(equ_ql) % all empty

                MeaEqu = np.array(MeaEqu, dtype=object)
                MeaEqu = np.concatenate(MeaEqu, axis=0)
                MeaEqu = MeaEqu.flatten()

                # Objective function
                delta_z = measure - MeaEqu

                G2 = H.T @ np.linalg.inv(Weight) @ delta_z

                G3 = H.T @ np.linalg.inv(Weight) @ H 

                
                sizeG3 = G3.shape[0]

                zeroRow, zeroCol = computeZero(G3)


                # Remove the specified rows and columns from G3
                G3 = np.delete(G3, zeroRow, axis=0)
                G3 = np.delete(G3, zeroCol, axis=1)  

                Ginv = sp.linalg.pinv(G3)


                nonindex = np.setdiff1d(np.arange(sizeG3), zeroRow)



                G3_temp = np.zeros((sizeG3, sizeG3))
                G3_temp[np.ix_(nonindex, nonindex)] = Ginv
                Ginv = G3_temp
                deltaX = Ginv @ G2  # Materix multiplication

                tol_temp = np.max(np.abs(deltaX))
                print("Iteration Count: ", i)
                print("Tol: ", tol_temp)


                if tol_temp < tol:
                    break
                deltaX = deltaX.reshape(-1, 1)
                StateVar = StateVar + deltaX

                
                

            tempVN = StateVar[:Num.Node]
            tempVA = StateVar[Num.Node:]
            seResult = {}

            seResult['Ve'] = tempVN * np.cos(tempVA)
            seResult['Vf'] = tempVN * np.sin(tempVA)
            seResult['VOL'] = seResult['Ve'] + 1j * seResult['Vf']
            seResult['VN'] = tempVN
            seResult['AN'] = tempVA * 180 / np.pi
            seResult['AN'][seResult['VN'] < 1e-4] = 0
            # Create the DATA array

            # FINAL      
            voltage_magnitudes = seResult['VN'] 
            voltage_angles = seResult['AN'] 
            logger.info(f'voltage_magnitudes:{voltage_magnitudes}')
            logger.info(f'voltage_angles:{voltage_magnitudes}')

            self.pub_voltage_mag.publish(
                VoltagesMagnitude(
                    values=list(voltage_magnitudes), ids=ids, time=voltages.time
                ).json()
            )
            self.pub_voltage_angle.publish(
                VoltagesAngle(
                    values=list(voltage_angles), ids=ids, time=voltages.time
                ).json()
            )

            logger.info("end time: " + str(datetime.now()))

        self.destroy()

    def destroy(self):
        "Finalize and destroy the federates"
        h.helicsFederateDisconnect(self.vfed)
        logger.info("Federate disconnected")

        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()

def run_simulator(broker_config: BrokerConfig):
    with open("static_inputs.json") as f:
        config = json.load(f)
        federate_name = config["name"]
        if "algorithm_parameters" in config:
            parameters = AlgorithmParameters.parse_obj(config["algorithm_parameters"])
        else:
            parameters = AlgorithmParameters.parse_obj({})

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    sfed = StateEstimatorFederate(
        federate_name, parameters, input_mapping, broker_config
    )
    sfed.run()

if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))






