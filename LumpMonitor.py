import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.HexClasses import HEX, Fluid
from scipy.optimize import fsolve
from pathlib import Path

def main():
    # initialise HEX
    hex = HEX(L=6.1, ri=22.9e-3, ro=25.4e-3, R=50e-3, n=1)

    # initialise fluids
    fluid1 = Fluid(m=0.3, Cp=2916, rho=680, Ti=523, k=0.12, mu=4e-6 * 680)
    fluid2 = Fluid(m=0.5, Cp=4180, rho=1000, Ti=603, k=0.6, mu=8.9e-4)
      
    '''
    f_type: 0 - parallel, 1 - counter
     mode: cinlet/rinlet: constant or random inlet
    '''
    f_type = 0
    mode = "rinlet"
    d_path = Path(f"../../py_data/HEXPractice/lumpHEX/{mode}")

    if f_type == 0:
        dfs = pd.read_csv(f"{d_path}/parallel.csv", header=0)
    elif f_type == 1:
        dfs = pd.read_csv(f"{d_path}/counter.csv", header=0)
        
    T1i = dfs["F1i"]       # Fluid1 (cold) inlet temperature
    T1o = dfs["F1o"]       # Fluid1 (cold) outlet temperature
    m1 = dfs["F1m"]        # Fluid1 (cold) mass flow rate
    T2i = dfs["F2i"]       # Fluid2 (hot) inlet temperature
    T2o = dfs["F2o"]       # Fluid2 (hot) outlet temperature
    m2 = dfs["F2m"]        # Fluid2 (hot) mass flow rate
    dp1 = dfs["dP1"]       # Fluid1 (cold) pressure drop

    fluid1.get_Inlets(T1i, m1)
    fluid2.get_Inlets(T2i, m2)
    fluid2.get_Prams(hex.Ac2, hex.D2, hex.As2)
    
    # heat duty Q = m1 * Cp1 * (T1o - T1i)
    Q = m1 * fluid1.Cp * (T1o - T1i)
    
    # get log mean temperature difference
    if f_type == 0:
        dT1 = T2i - T1i
        dT2 = T2o - T1o
    elif f_type == 1:
        dT1 = T2o - T1i
        dT2 = T2i - T1o
    # dTlm = (dT1 - dT2) / ln(dT1 / dT2)
    dTlm = (dT1 - dT2) / np.log(dT1 / dT2)
    
    # UA = Q / (F * dTlm), correction factor F=1
    F = 1
    UA = Q / (F * dTlm)
    
    '''
    Get rid of the Rwall and Rconv2
    R = 1 / UA = Rconv,1 + Rf1 + Rwall + Rf2 + Rconv,2
    here we suppose no fouling in the shell side (Rf2 = 0)
    '''
    Rflu = 1 / UA - hex.dRwall - fluid2.R
    
    # solving fouling thcinkess
    def solve_sigma(sigma):
        rfi = hex.ri - sigma
        Ac1 = np.pi * rfi ** 2
        D1 = 2 * rfi
        Vol1 = fluid1.m / fluid1.rho
        v1 = Vol1 / Ac1
        Re = fluid1.get_Re(v1, D1)
        Cf = fluid1.get_Fricion(Re)
        return Cf  / D1 * fluid1.rho * v1 ** 2 / 2  - dp1 / hex.dx
    
    guess_sigma = (1e-5) * np.ones(len(dfs))
    sigma_sol = fsolve(solve_sigma, guess_sigma)
    
    # predict fouling layer conductivity
    def solve_k():
        rfi = hex.ri - sigma_sol
        Ac1 = np.pi * rfi ** 2
        D1 = 2 * rfi
        As1 = np.pi * D1 * hex.dx
        Vol1 = fluid1.m / fluid1.rho
        v1 = Vol1 / Ac1
        Re = fluid1.get_Re(v1, D1)
        Pr = fluid1.get_Pr()
        Nu = fluid1.get_Nu(Re, Pr)
        h1 = fluid1.get_h(Nu, D1)
        
        num = np.log(hex.ri / rfi) / (2 * np.pi  * hex.dx)
        den = (Rflu - 1 / (As1 * h1))

        return num / den
    
    k_sol = solve_k()
    
    # make plots
    fig, ax = plt.subplots(1, 2)
    fig.set_figheight(6)
    fig.set_figwidth(15)
    
    x = dfs["Day"].to_numpy()
        
    ax[0].plot(x, sigma_sol, c="blue", alpha=0.7, label="Predicted fouling thickness")
    ax[0].plot(x, dfs["Sigma1"].to_numpy(), c="red", alpha=0.7, label="True fouling thickness")
    ax[0].set_ylabel("Thickness (m)")
    ax[0].set_xlabel("Days")
    ax[0].legend()
    
    ax[1].plot(x, k_sol.to_numpy(),  c="blue", alpha=0.7, label="Predicted deposit conductivity")
    ax[1].plot(x, 0.2 * np.ones(len(dfs)), c="red", alpha=0.7, label="True deposit conductivity")
    ax[1].set_ylabel("Conductivity (W/m*k)")
    ax[1].set_xlabel("Days")
    ax[1].legend()
    
    plt.ylim(0.18, 0.22)
    plt.show()

if __name__ == '__main__':
    main()