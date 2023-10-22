import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from HexClasses import HEX, Fluid, Fouling
from utils import get_Tf, get_Q, gen_Inlets

# dataframe for recording data
dfs = pd.DataFrame()
Day = []
UAs = []
T1in = []
T1m = []
T1o = []
T1v = []
T1D = []
T1Re = []
T1Pr = []
T1h = []
T1pd = []
Sigma1 = []
Rf1 =[]
T2in = []
T2m = []
T2o = []
T2v = []
T2D = []
T2Re = []
T2Pr = []
T2h = []
Sigma2 = []
Rf2 =[]
Q = []

'''
append values to lists
'''
def append_Vars(day, uas, 
                t1i, t1m, t1v, t1d, t1re, t1h, t1pd, sigma1, rf1,
                t2i, t2m, t2v, t2d, t2re, t2h, sigma2, rf2):
    Day.append(day)
    UAs.append(uas)
    T1in.append(t1i)
    T1m.append(t1m)
    T1v.append(t1v)
    T1D.append(t1d)
    T1Re.append(t1re)
    T1h.append(t1h)
    T1pd.append(t1pd)
    Sigma1.append(sigma1)
    Rf1.append(rf1)
    T2in.append(t2i)
    T2m.append(t2m)
    T2v.append(t2v)
    T2D.append(t2d)
    T2Re.append(t2re)
    T2h.append(t2h)
    Sigma2.append(sigma2)
    Rf2.append(rf2)

'''
run HEX simulation
'''
def run_HEX(k, hex, f_type, t, dt, eps, fluid1, fluid2, depo1, depo2, lplt):
    n = hex.n
    dx = hex.dx
    T0 = hex.T0
    print("Day", k)
    x = np.linspace(dx / 2, hex.L - dx / 2, n)

    # # reset fluid properties
    T1i, m1, T2i, m2 = gen_Inlets()
    fluid1.get_Inlets(T1i, m1)
    fluid2.get_Inlets(T2i, m2)

    Ac1 = hex.Ac1 * np.ones(n)
    Ac2 = hex.Ac2 * np.ones(n)
    fluid1.get_Prams(Ac1, hex.D1, hex.As1)
    fluid2.get_Prams(Ac2, hex.D2, hex.As2)
    UA = 1 / (fluid1.R + hex.Rfi + hex.dRwall + hex.Rfo + fluid2.R)       # W*m^2/n^2*k Overall heat transfer coefficient times surface area (1 / Total Resistance)
    dPdx = fluid1.get_PressureDrop(fluid1.Cf, fluid1.v, hex.rfi)
    
    # initialise temperatures
    T1 = np.ones(n) * T0
    T2 = T1.copy()
    dT1dt = np.ones(n)
    dT2dt = np.ones(n)

    append_Vars(k, np.mean(UA), 
            T1i, m1, np.mean(fluid1.v), np.mean(hex.D1), np.mean(fluid1.Re), np.mean(fluid1.h), np.sum(dPdx * dx), np.mean(depo1.sigma), np.mean(hex.Rfi),
            T2i, m2, np.mean(fluid2.v), np.mean(hex.D2), np.mean(fluid2.Re), np.mean(fluid2.h), np.mean(depo2.sigma), np.mean(hex.Rfo))

    # Parallel flow: 0
    if f_type == 0:
        for j in range(1, len(t) + 1):
            ts = int(j * dt)
            # heat accumulation = flow in - flow out + heat duty in
            dT1dt[1 : n]  = (m1 * fluid1.Cp * (T1[0 : n - 1] - T1[1 : n]) + UA[1 : n] * (T2[1 : n] - T1[1 : n])) / (fluid1.rho * fluid1.Cp * dx * Ac1[1 : n])
            dT1dt[0]  = (m1 * fluid1.Cp * (T1i - T1[0]) + UA[0] * (T2[0] - T1[0])) / (fluid1.rho * fluid1.Cp * dx * Ac1[0])
            # heat accumulation = flow in - flow out - heat duty out
            dT2dt[1 : n]  = (m2 * fluid2.Cp * (T2[0 : n - 1] - T2[1 : n]) - UA[1 : n] * (T2[1 : n] - T1[1 : n])) / (fluid2.rho * fluid2.Cp * dx * Ac2[1 : n])
            dT2dt[0]  = (m2 * fluid2.Cp * (T2i - T2[0]) - UA[0] * (T2[0] - T1[0])) / (fluid2.rho * fluid2.Cp * dx * Ac2[0])
            
            T1 = T1 + dT1dt * dt
            T2 = T2 + dT2dt * dt

            # monitoring plot
            # plt.figure(1)
            # if ((ts % lplt) == 0):
            #     plt.title(f"Day {k}, {ts} secs")
            #     plt.plot(x, T1, c="blue", label=f"Fluid1 (cold), inlet {np.round(T1i)} K")
            #     plt.plot(x, T2, c="red", label=f"Fluid2 (hot), inlet {np.round(T2i)} k")
            #     plt.xlabel("Distance (m)")
            #     plt.ylabel("Temperature (K)")
            #     plt.legend(loc = "upper right")
            #     plt.plot()
            #     plt.pause(0.005)
            #     plt.cla()

            # break the loop when steady state reaches
            if np.sum(dT1dt) < eps and np.sum(dT2dt) < eps:
                print("Steady state reaches at t =", ts, "secs.")
                plt.close()
                
                # heat duty and film temperature
                dQdt = get_Q(UA, T1, T2)
                Tf1, Tf2 = get_Tf(dQdt, T1, T2, fluid1.R, fluid2.R)

                # simulate fouling thickness for the next day
                depo1.FoulingSimu(fluid1.Re, fluid1.Pr, Tf1, fluid1.tau, depo1.k_l0, 24 * 3600)
                # depo2.FoulingSimu(fluid2.Re, fluid2.Pr, Tf2, fluid2.tau, depo2.k_l0, 24 * 3600)

                # update HEX parameters
                hex.update_Prams(depo1.sigma, depo2.sigma, depo1.k_l0, depo2.k_l0)

                # append the rest variables
                T1o.append(T1[-1])
                T2o.append(T2[-1])
                Q.append(np.sum(dQdt))
                
                break
        
    # Counter flow: 1
    else:
        for j in range(1, len(t) + 1):
            ts = int(j * dt)
            # heat accumulation = flow in - flow out + heat duty in
            dT1dt[1 : n]  = (m1 * fluid1.Cp * (T1[0 : n - 1] - T1[1 : n]) + UA[1 : n] * (T2[1 : n] - T1[1 : n])) / (fluid1.rho * fluid1.Cp * dx * Ac1[1 : n])
            dT1dt[0]  = (m1 * fluid1.Cp * (T1i - T1[0]) + UA[0] * (T2[0] - T1[0])) / (fluid1.rho * fluid1.Cp * hex.dx * Ac1[0])
            # heat accumulation = flow in - flow out - heat duty out
            dT2dt[0 : n - 1]  = (m2 * fluid2.Cp * (T2[1 : n] - T2[0 : -1]) - UA[0 : -1] * (T2[0 : -1] - T1[0 : -1])) / (fluid2.rho * fluid2.Cp * dx * Ac2[0 : -1])
            dT2dt[-1]  = (m2 * fluid2.Cp * (T2i - T2[-1]) - UA[-1] * (T2[-1] - T1[-1])) / (fluid2.rho * fluid2.Cp * dx * Ac2[-1])

            T1 = T1 + dT1dt * dt
            T2 = T2 + dT2dt * dt

            # monitoring plot
            # plt.figure(1)
            # if ((ts % lplt) == 0):
            #     plt.title(f"Day {k}, {ts} secs")
            #     plt.plot(x, T1, c="blue", label=f"Fluid1 (cold), inlet {np.round(T1i)} K")
            #     plt.plot(x, T2, c="red", label=f"Fluid2 (hot), inlet {np.round(T2i)} K")
            #     plt.xlabel("Distance (m)")
            #     plt.ylabel("Temperature (K)")
            #     plt.legend(loc = "upper right")
            #     plt.plot()
            #     plt.pause(0.005)
            #     plt.cla()

            # break the loop when steady state reaches
            if np.sum(dT1dt) < eps and np.sum(dT2dt) < eps:
                print("Steady state reaches at t =", ts, "secs.")

                # heat duty and film temperature
                dQdt = get_Q(UA, T1, T2)
                Tf1, Tf2 = get_Tf(dQdt, T1, T2, fluid1.R, fluid2.R)

                # simulate fouling thickness for the next day
                depo1.FoulingSimu(fluid1.Re, fluid1.Pr, Tf1, fluid1.tau, depo1.k_l0, 24 * 3600)
                # depo2.FoulingSimu(fluid2.Re, fluid2.Pr, Tf2, fluid2.tau, depo2.k_l0, 24 * 3600)

                # update HEX parameters
                hex.update_Prams(depo1.sigma, depo2.sigma, depo1.k_l0, depo2.k_l0)

                # append the rest variables
                T1o.append(T1[-1])
                T2o.append(T2[0])
                Q.append(np.sum(dQdt))
                
                break

if __name__ == '__main__':
    # time setting
    t_final = 10e5       # s, maximum simulation time
    dt = 0.1          # s, time step
    t = np.arange(0, t_final, dt)
    lplt = 60       # monitoring plot lag
    eps = 1e-8      # threshold of psudo steady state
    

    # initialise HEX
    hex = HEX(ri=0.3, ro=0.35, R=0.6, k=397)
    
    # initialise fouling layers
    depo1 = Fouling()
    depo2 = Fouling()
    
    # initialise fluids
    fluid1 = Fluid(Ti=273, Cp=470, rho=3100)
    fluid2 = Fluid(Ti=600, Cp=500, rho=1000)
    
    # start simulation
    f_type = 1      # HEX mode: 0 - parallel, 1 - counter
    days = 200      # running days
    for k in range(1, days + 1):
        run_HEX(k, hex, f_type, t, dt, eps, fluid1, fluid2, depo1, depo2, lplt)

    # export data frame
    dfs["Day"] = Day
    dfs["F1m"] = T1m
    dfs["F1i"] = T1in
    dfs["F1o"] = T1o
    dfs["F1v"] = T1v
    dfs["F1D"] = T1D
    dfs["F1Re"] = T1Re
    dfs["F1h"] = T1h
    dfs["dP"] = T1pd
    dfs["Sigma1"] = Sigma1
    dfs["Rf1"] = Rf1
    dfs["F2m"] = T2m
    dfs["F2i"] = T2in
    dfs["F2o"] = T2o
    dfs["F2v"] = T2v
    dfs["F2D"] = T2D
    dfs["F2Re"] = T2Re
    dfs["F2h"] = T2h
    dfs["Sigma2"] = Sigma2
    dfs["Rf2"] = Rf2
    dfs["UA"] = UAs
    dfs["Q"] = Q

    if f_type == 0:
        dfs.to_csv("../../py_data/HEXPractice/parallel.csv", index=False)
    else:
        dfs.to_csv("../../py_data/HEXPractice/counter.csv", index=False)