import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.HexClasses import HEX, Fluid, Fouling
from utils.utils import get_Tf
import utils.DataframeGenerator as dfg
from scipy.optimize import fsolve
from pathlib import Path

def Simulation(dfs, day, dgen, f_type, hex,
               T1i, T2i, m1, m2, fluid1, fluid2, depo1, depo2
               ):
    t1i = T1i[day]
    t2i = T2i[day]
    m1i = m1[day]
    m2i = m2[day]

    fluid1.get_Inlets(t1i, m1i)
    fluid2.get_Inlets(t2i, m2i)

    fluid1.get_Prams(hex.Ac1, hex.D1, hex.As1)
    fluid2.get_Prams(hex.Ac2, hex.D2, hex.As2)
    UA = 1 / (fluid1.R + hex.Rfi + hex.dRwall + hex.Rfo + fluid2.R)       # W*m^2/n^2*k Overall heat transfer coefficient times surface area (1 / Total Resistance)
    
    
    # pressure drops
    dP1dx = fluid1.get_PressureDrop(fluid1.Cf, hex.D1, fluid1.v)
    dP2dx = fluid2.get_PressureDrop(fluid2.Cf, hex.D2, fluid2.v)
    
    dgen.append_Vars(day + 1, np.mean(UA), 
        t1i, m1i, fluid1.v, hex.D1, fluid1.Re, fluid1.Nu, fluid1.h, fluid1.R, fluid1.Cf, fluid1.tau, dP1dx * hex.dx, depo1.sigma, hex.Rfi,
        t2i, m2i, fluid2.v, hex.D2, fluid2.Re, fluid2.Nu, fluid2.h, fluid2.R, fluid2.Cf, fluid2.tau, dP2dx * hex.dx, depo2.sigma, hex.Rfo)

    '''
    Function for solving outlets
    Q = m1 * cp1 * (t1o - t1i) = m2 * cp2 * (t2i - t2o)
    Q = F * U * A * dTlm
    here we set correlation factor F = 0.993
    '''
    F = 0.993
    def Solve_outlets(sol):
        if f_type == 0:
            dT1 = t2i - t1i
            dT2 = sol[1] - sol[0]
        elif f_type == 1:
            dT1 = t2i - sol[0]
            dT2 = sol[1] - t1i
        dTlm = (dT1 - dT2) / np.log(dT1 / dT2)

        Q = m1i * fluid1.Cp * (sol[0] - t1i)
        return [Q - m2i * fluid2.Cp * (t2i - sol[1]),  Q - F * UA * dTlm]
    
    # sol = fsolve(Solve_outlets, [t1i + 5, t2i - 5])
    # guess starting with the true values, but it DO NOT hanppen in the real world
    sol = fsolve(Solve_outlets, [dfs["F1o"][day], dfs["F2o"][day]]) 
    t1o = sol[0]
    t2o = sol[1]

    # heat duty and film temperature
    Q = m1i * fluid1.Cp * np.abs(t1i - t1o)
    Tf1, Tf2 = get_Tf(Q, np.mean([t1i, t1o]), np.mean([t2i, t2o]), fluid1.R, fluid2.R)
    dgen.append_Outlets(t1o, t2o, np.sum(Q))

    # simulate fouling thickness for the next day
    depo1.FoulingSimu(fluid1.Re, fluid1.Pr, Tf1, fluid1.tau, depo1.k_l0, 24 * 3600)
    # depo2.FoulingSimu(fluid2.Re, fluid2.Pr, Tf2, fluid2.tau, depo2.k_l0, 24 * 3600)

    # update HEX parameters
    hex.update_Prams(depo1.sigma, depo2.sigma, depo1.k_l0, depo2.k_l0)

def main():
    '''
    Load data, suppose we only have monitoring data of inlet temeratures and mass rates
    '''
    # initialise HEX
    hex = HEX(L=6.1, ri=9.93e-3, ro=12.7e-3, R=30e-3, n=1)

    # initialise fluids
    fluid1 = Fluid(m=0.3, Cp=1900, rho=900, Ti=573, k=0.12, mu=4e-6 * 900)
    fluid2 = Fluid(m=1, Cp=4180, rho=1000, Ti=800, k=0.7, mu=8.9e-4)

    # initialise fouling layers
    depo1 = Fouling(pv="CF")
    depo2 = Fouling(pv="CF")
    
    # f_type: 0 - parallel, 1 - counter
    f_type = 1
    d_path = Path("../../py_data/HEXPractice")

    # mode: cinlet/rinlet: constant or random inlet
    mode = "cinlet"
    s_path = Path(f"{d_path}/lumpHEX/{mode}")

    if f_type == 0:
        dfs = pd.read_csv(f"{d_path}/{mode}/parallel.csv", header=0)
    elif f_type == 1:
        dfs = pd.read_csv(f"{d_path}/{mode}/counter.csv", header=0)
    
    T1i = dfs["F1i"]       # Fluid1 (cold) inlet temperature
    m1 = dfs["F1m"]       # Fluid1 (cold) mass flow rate
    T2i = dfs["F2i"]      # Fluid2 (hot) inlet temperature
    m2 = dfs["F2m"]     # Fluid2 (hot) mass flow rate
    
    # dataframe for recording data
    dgen = dfg.GenDataframe()
    
    # start hex simulation
    for day in range(len(dfs)):
        Simulation(dfs, day, dgen, f_type, hex, T1i, T2i, m1, m2, fluid1, fluid2, depo1, depo2)
    
    dgen.export_Vars(f_type, s_path)

if __name__ == '__main__':
    main()