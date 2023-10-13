import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from HexClasses import HEX
from HexClasses import Fluid
from HexClasses import Fouling
from tqdm import tqdm

# time setting
t_final = 50000       # s, simulation time
dt = 1          # s, time step
t = np.arange(0, t_final, dt)

# dataframe for recording data
dfs = pd.DataFrame()
Time = []
UAs = []
T1o = []
T1v = []
T1D = []
T1Re = []
T1Pr = []
T1h = []
T1pd = []
Sigma1 = []
Rf1 =[]
T2o = []
T2v = []
T2D = []
T2Re = []
T2Pr = []
T2h = []
Sigma2 = []
Rf2 =[]

def append_Vars(time, uas, 
                t1o, t1v, t1d, t1re, t1h, t1pd, sigma1, rf1,
                t2o, t2v, t2d, t2re, t2h, sigma2, rf2):
    Time.append(time)
    UAs.append(uas)
    T1o.append(t1o)
    T1v.append(t1v)
    T1D.append(t1d)
    T1Re.append(t1re)
    T1h.append(t1h)
    T1pd.append(t1pd)
    Sigma1.append(sigma1)
    Rf1.append(rf1)
    T2o.append(t2o)
    T2v.append(t2v)
    T2D.append(t2d)
    T2Re.append(t2re)
    T2h.append(t2h)
    Sigma2.append(sigma2)
    Rf2.append(rf2)

if __name__ == '__main__':
    # initialise hex and fluids
    hex = HEX(ri=0.3, ro=0.35, R=0.6, k=397)          # copper
    fluid1 = Fluid(m=5, Cp=470, rho=3100, Ti=303)     # Bromine
    fluid2 = Fluid(m=7, Cp=500, rho=1000, Ti=600)                               # Water
    
    # initialise variables
    L = hex.L    
    n = hex.n
    T0 = hex.T0
    Ac1 = hex.Ac1 * np.ones(n)
    Ac2 = hex.Ac2 * np.ones(n)
    dx = hex.dx
    fluid1.get_Prams(Ac1, hex.D1, hex.As1)
    fluid2.get_Prams(Ac2, hex.D2, hex.As2)
    UA = 1 / (fluid1.R + hex.dRwall + fluid2.R)  * np.ones(n)

    append_Vars(0, np.mean(UA), 
                0, np.mean(fluid1.v), np.mean(hex.D1), np.mean(fluid1.Re), np.mean(fluid1.h), 0, 0, 0,
                0, np.mean(fluid2.v), np.mean(hex.D2), np.mean(fluid2.Re), np.mean(fluid2.h), 0, 0)

    # cold fluid
    m1 = fluid1.m
    Cp1 = fluid1.Cp
    rho1 = fluid1.rho
    T1i = fluid1.Ti
    
    # hot fluid
    m2 = fluid2.m
    Cp2 = fluid2.Cp
    rho2 = fluid2.rho
    T2i = fluid2.Ti

    # fouling layers
    depo1 = Fouling()
    depo2 = Fouling()
    
    # initialise time and temperatures
    T1 = np.ones(n) * T0
    T2 = T1.copy()
    dT1dt = np.ones(n)
    dT2dt = np.ones(n)
    eps = 1e-6        # error

    # plt.figure(figsize=(9, 6))
    # plt.xlabel("Distance (m)")
    # plt.ylabel("Temperature (K)")
    # lplt = 500        # plot lag
    
    f_type = 0
    
    # Parallel flow: 0
    if f_type == 0:
        # T_old = T1
        for j in tqdm(range(1, len(t) + 1), desc="e-10 s"):
            ts = j * dt

            dT1dt[1 : n]  = (m1 * Cp1 * (T1[0 : n - 1] - T1[1 : n]) + UA[1 : n] * (T2[1 : n] - T1[1 : n])) / (rho1 * Cp1 * dx * Ac1[1 : n])
            dT1dt[0]  = (m1 * Cp1 * (T1i - T1[0]) + UA[0] * (T2[0] - T1[0])) / (rho1 * Cp1 * dx * Ac1[0])

            dT2dt[1 : n]  = (m2 * Cp2 * (T2[0 : n - 1] - T2[1 : n]) - UA[1 : n] * (T2[1 : n] - T1[1 : n])) / (rho2 * Cp2 * dx * Ac2[1 : n])
            dT2dt[0]  = (m2 * Cp2 * (T2i - T2[0]) - UA[0] * (T2[0] - T1[0])) / (rho2 * Cp2 * dx * Ac2[0])

            T1 = T1 + dT1dt * dt
            T2 = T2 + dT2dt * dt

            # update parameters
            depo1.FoulingSimu(fluid1.Re, fluid1.Pr, T1, fluid1.tau, 0.2, dt)
            depo2.FoulingSimu(fluid2.Re, fluid1.Pr, T2, fluid2.tau, 0.2, dt)
            hex.update_Prams(depo1.sigma, depo2.sigma, depo1.k_l0, depo2.k_l0)
            Ac1 = hex.Ac1
            Ac2 = hex.Ac2
            dx = hex.dx
            fluid1.get_Prams(Ac1, hex.D1, hex.As1)
            fluid2.get_Prams(Ac2, hex.D2, hex.As2)

            UA = 1 / (fluid1.R + hex.Rfi + hex.dRwall + hex.Rfo + fluid2.R)
            dPdx = fluid1.get_PressureDrop(fluid1.Cf, fluid1.v, hex.rfi)
            # append values

            if np.mod(ts, 3600) == 0.0:
                append_Vars(ts // 3600, np.mean(UA), 
                    T1[-1], np.mean(fluid1.v), np.mean(hex.D1), np.mean(fluid1.Re), np.mean(fluid1.h), np.sum(dPdx * dx), np.mean(depo1.sigma), np.mean(hex.Rfi),
                    T2[-1], np.mean(fluid2.v), np.mean(hex.D2), np.mean(fluid2.Re), np.mean(fluid1.h), np.mean(depo2.sigma), np.mean(hex.Rfo))

    #         if ((j % lplt) == 0):
    #             T_new = T1

    #             plt.plot(x, T2, c = "red", label = f"Fluid2 (hot) outlet = {round(T2[0], 2)} K")
    #             plt.plot(x, T1, c = "blue", label = f"Fluid1 (cold) outlet = {round(T1[-1], 2)} K")
    #             plt.axis([0, L, T0 - 2, T2i + 50])
    #             plt.legend(loc = "upper right")
    #             plt.title(f"$t$ = {j * dt} s")
                
    #             if np.mean((T_new - T_old) ** 2) > eps and j != len(t):
    #                 T_old = T_new
    #                 plt.pause(0.005)
    #                 plt.cla()
    #             else:               # keep the last 
    #                 break
    #     print("Steady state has reached at t =", j * dt, "s.")
    #     plt.show()

    # Counter flow: 1
    else:
        # T_old = T1
        for j in tqdm(range(1, len(t) + 1), desc="e-10 s"):
            ts = j * dt

            dT1dt[1 : n]  = (m1 * Cp1 * (T1[0 : n - 1] - T1[1 : n]) + UA[1 : n] * (T2[1 : n] - T1[1 : n])) / (rho1 * Cp1 * dx * Ac1[1 : n])
            dT1dt[0]  = (m1 * Cp1 * (T1i - T1[0]) + UA[0] * (T2[0] - T1[0])) / (rho1 * Cp1 * dx * Ac1[0])

            dT2dt[0 : n - 1]  = (m2 * Cp2 * (T2[1 : n] - T2[0 : n - 1]) - UA[0 : n - 1] * (T2[0 : n - 1] - T1[0 : n - 1])) / (rho2 * Cp2 * dx * Ac2[0 : n-1])
            dT2dt[n - 1]  = (m2 * Cp2 * (T2i - T2[n - 1]) - UA[n - 1] * (T2[n - 1] - T1[n - 1])) / (rho2 * Cp2 * dx * Ac2[n - 1])

            T1 = T1 + dT1dt * dt
            T2 = T2 + dT2dt * dt
            
            # update parameters
            depo1.FoulingSimu(fluid1.Re, fluid1.Pr, T1, fluid1.tau, 0.2, dt)
            depo2.FoulingSimu(fluid2.Re, fluid1.Pr, T2, fluid2.tau, 0.2, dt)
            hex.update_Prams(depo1.sigma, depo2.sigma, depo1.k_l0, depo2.k_l0)
            Ac1 = hex.Ac1
            Ac2 = hex.Ac2
            dx = hex.dx
            fluid1.get_Prams(Ac1, hex.D1, hex.As1)
            fluid2.get_Prams(Ac2, hex.D2, hex.As2)

            UA = 1 / (fluid1.R + hex.Rfi + hex.dRwall + hex.Rfo + fluid2.R)
            dPdx = fluid1.get_PressureDrop(fluid1.Cf, fluid1.v, hex.rfi)
            # append values

            if np.mod(ts, 3600) == 0.0:
                append_Vars(ts // 3600, np.mean(UA), 
                    T1[-1], np.mean(fluid1.v), np.mean(hex.D1), np.mean(fluid1.Re), np.mean(fluid1.h), np.sum(dPdx * dx), np.mean(depo1.sigma), np.mean(hex.Rfi),
                    T2[-1], np.mean(fluid2.v), np.mean(hex.D2), np.mean(fluid2.Re), np.mean(fluid1.h), np.mean(depo2.sigma), np.mean(hex.Rfo))

    #         if ((j % lplt) == 0):
    #             T_new = T1

    #             plt.plot(x, T2, c = "red", label = f"Fluid2 (hot) outlet = {round(T2[0], 2)} K")
    #             plt.plot(x, T1, c = "blue", label = f"Fluid1 (cold) outlet = {round(T1[-1], 2)} K")
    #             plt.axis([0, L, T0 - 2, T2i + 50])
    #             plt.legend(loc = "upper right")
    #             plt.title(f"$t$ = {j * dt} s")
                
    #             if np.mean((T_new - T_old) ** 2) > eps and j != len(t):
    #                 T_old = T_new
    #                 plt.pause(0.005)
    #                 plt.cla()
    #             else:               # keep the last 
    #                 break
    #     print("Steady state has reached at t =", j * dt, "s.")
    #     plt.show()

# export data
dfs["Time(hr)"] = Time
dfs["UA"] = UAs
dfs["F1o"] = T1o
dfs["F1v"] = T1v
dfs["F1D"] = T1D
dfs["F1Re"] = T1Re
dfs["F1h"] = T1h
dfs["dP"] = T1pd
dfs["Sigma1"] = Sigma1
dfs["Rf1"] = Rf1
dfs["F2o"] = T2o
dfs["F2v"] = T2v
dfs["F2D"] = T2D
dfs["F2Re"] = T2Re
dfs["F2h"] = T2h
dfs["Sigma2"] = Sigma2
dfs["Rf2"] = Rf2

if f_type == 0:
    dfs.to_csv("../HEXdata/parallel.csv", index=False)
else:
    dfs.to_csv("../HEXdata/counter.csv", index=False)
