import numpy as np
from utils.utils import get_Tf, get_Q, gen_Inlets, export_DayVars
import matplotlib.pyplot as plt

'''
run HEX simulation
'''
def run_HEX(dfs,        # dataframe for storing overall/average daily data
            dpath,      # data path for saving data
            k,          # simulation day number
            ks,         # an array, data of days we want to save
            hex,        # HEX class
            n,          # HEX.n, number of small control volume
            dx,         # HEX.dx, length of each small control volume
            T0,         # HEX.t0, initial temperature
            x,          # label of x axis
            f_type,     # 0 - parallel flow, 1 - counter flow
            t_final,    # s, maximum time to simulate
            eps,        # threshold of
            fluid1,     # fluid 1 class
            fluid2,     # fluid 2 class
            depo1,      # deposit layer 1 class
            depo2,      # deposit layer 2 class
            lplt,       # time lag of monitoring plot
            ran,        # 1 - generate inlet temperatures and mass flows randomly
            T1min,      # if ran == 1, minimum temperature of fluid 1 
            T2min,      # if ran == 1, minimum temperature of fluid 2
            T1diff,     # if ran == 1, maximum temperature difference of fluid 1 
            T2diff,     # if ran == 1, maximum temperature difference of fluid 2
            m1min,      # if ran == 1, minimum mass flow, etc.
            m2min,
            m1diff,
            m2diff
            ):
    
    T1i, m1 = fluid1.Ti, fluid1.m
    T2i, m2 = fluid2.Ti, fluid2.m
    # randomly generate fluid properties
    if ran == 1:
        T1i, m1 = gen_Inlets(Tmin=T1min, Tdiff=T1diff, mmin=m1min, mdiff=m1diff)
        T2i, m2 = gen_Inlets(Tmin=T2min, Tdiff=T2diff, mmin=m2min, mdiff=m2diff)
        fluid1.get_Inlets(T1i, m1)
        fluid2.get_Inlets(T2i, m2)
    
    Ac1 = hex.Ac1 * np.ones(n)
    Ac2 = hex.Ac2 * np.ones(n)
    fluid1.get_Prams(Ac1, hex.D1, hex.As1)
    fluid2.get_Prams(Ac2, hex.D2, hex.As2)
    UA = 1 / (fluid1.R + hex.Rfi + hex.dRwall + hex.Rfo + fluid2.R)       # W*m^2/n^2*k Overall heat transfer coefficient times surface area (1 / Total Resistance)
    
    # pressure drops
    dP1dx = fluid1.get_PressureDrop(fluid1.Cf, hex.D1, fluid1.v)
    dP2dx = fluid2.get_PressureDrop(fluid2.Cf, hex.D2, fluid2.v)
    
    dt = 0.5 / np.max(fluid1.v + fluid2.v) * dx      # max courant numebr = 0.5
    t = np.arange(0, t_final, dt)
    print("Day", k, ", dt = ", np.round(dt, 5))
    
    # initialise temperatures
    T1 = np.ones(n) * T0
    T2 = T1.copy()
    dT1dt = np.zeros(n)
    dT2dt = np.zeros(n)

    dfs.append_Vars(k, np.mean(UA), 
            T1i, m1, np.mean(fluid1.v), np.mean(hex.D1), np.mean(fluid1.Re), np.mean(fluid1.Nu), np.mean(fluid1.h), np.mean(fluid1.R), np.mean(fluid1.Cf), np.mean(fluid1.tau), np.sum(dP1dx * dx), np.mean(depo1.sigma), np.mean(hex.Rfi),
            T2i, m2, np.mean(fluid2.v), np.mean(hex.D2), np.mean(fluid2.Re), np.mean(fluid2.Nu), np.mean(fluid2.h), np.mean(fluid2.R), np.mean(fluid2.Cf), np.mean(fluid2.tau), np.sum(dP2dx * dx), np.mean(depo2.sigma), np.mean(hex.Rfo))

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
            # if (ts != 0) and (ts % lplt == 0):
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
            if np.sum(np.abs(dT1dt)) < eps and np.sum(np.abs(dT2dt)) < eps:
                print("Steady state reaches at t =", ts, "secs.")
                plt.close()
                
                # heat duty and film temperature
                Q = get_Q(UA, T1, T2)
                Tf1, Tf2 = get_Tf(Q, T1, T2, fluid1.R, fluid2.R)

                # simulate fouling thickness for the next day
                depo1.FoulingSimu(fluid1.Re, fluid1.Pr, Tf1, fluid1.tau, depo1.k_l0, 24 * 3600)
                # depo2.FoulingSimu(fluid2.Re, fluid2.Pr, Tf2, fluid2.tau, depo2.k_l0, 24 * 3600)

                # update HEX parameters
                hex.update_Prams(depo1.sigma, depo2.sigma, depo1.k_l0, depo2.k_l0)

                # append the rest variables
                dfs.append_Outlets(T1[-1], T2[-1], np.sum(Q))
                
                if k in ks:
                    export_DayVars(f_type, dpath, k, Q,
                                   T1, fluid1.Re, fluid1.h, fluid1.R, hex.Rfi, depo1.sigma, dP1dx,
                                   T2, fluid2.Re, fluid2.h, fluid2.R, hex.Rfo, depo2.sigma, dP2dx
                                   )
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
            if np.sum(np.abs(dT1dt)) < eps and np.sum(np.abs(dT2dt)) < eps:
                print("Steady state reaches at t =", ts, "secs.")

                # heat duty and film temperature
                Q = get_Q(UA, T1, T2)
                Tf1, Tf2 = get_Tf(Q, T1, T2, fluid1.R, fluid2.R)

                # simulate fouling thickness for the next day
                depo1.FoulingSimu(fluid1.Re, fluid1.Pr, Tf1, fluid1.tau, depo1.k_l0, 24 * 3600)
                # depo2.FoulingSimu(fluid2.Re, fluid2.Pr, Tf2, fluid2.tau, depo2.k_l0, 24 * 3600)

                # update HEX parametersF
                hex.update_Prams(depo1.sigma, depo2.sigma, depo1.k_l0, depo2.k_l0)

                # append the rest variables
                dfs.append_Outlets(T1[-1], T2[-0], np.sum(Q))
                
                if k in ks:
                    export_DayVars(f_type, dpath, k, Q,
                                   T1, fluid1.Re, fluid1.h, fluid1.R, hex.Rfi, depo1.sigma, dP1dx,
                                   T2, fluid2.Re, fluid2.h, fluid2.R, hex.Rfo, depo2.sigma, dP2dx
                                   )
                break