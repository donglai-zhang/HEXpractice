import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.HexClasses import HEX, Fluid
from scipy.optimize import fsolve
from pathlib import Path

def main(d_path, s_path, f_type):
    # initialise HEX
    hex = HEX(L=6.1, ri=22.9e-3 / 2, ro=25.4e-3 / 2, R=50e-3 / 2, n=1)

    # initialise fluids
    fluid1 = Fluid(m=0.3, Cp=2916, rho=680, Ti=523, k=0.12, mu=4e-6 * 680)
    fluid2 = Fluid(m=0.5, Cp=4180, rho=1000, Ti=603, k=0.6, mu=8.9e-4)
      
    '''
    f_type: 0 - parallel, 1 - counter
     mode: cinlet/rinlet: constant or random inlet
    '''
    
    dfs = pd.read_csv(d_path, header=0)
    df2 = pd.DataFrame([])

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
        return 4 * Cf  / D1 * fluid1.rho * v1 ** 2 / 2  - dp1 / hex.dx
    
    guess_sigma = (1e-7) * np.ones(len(dfs))
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
    
    df2["F1i"] = T1i
    df2["F1o"] = T1o 
    df2["F1m"] = m1
    df2["F2i"] = T2i 
    df2["F2o"] = T2o
    df2["F2m"] = m2
    df2["Sigma1_p"] = sigma_sol
    df2["k_p"] = k_sol
    
    df2.to_csv(s_path, index=False, header=True)
    
    # make plots
    # fig, ax = plt.subplots(1, 2)
    # fig.set_figheight(6)
    # fig.set_figwidth(15)
    
    # x = dfs["Day"].to_numpy()
        
    # ax[0].plot(x, sigma_sol, c="blue", alpha=0.7, label="Predicted fouling thickness")
    # ax[0].plot(x, dfs["Sigma1"].to_numpy(), c="red", alpha=0.7, label="True fouling thickness")
    # ax[0].set_ylabel("Thickness (m)")
    # ax[0].set_xlabel("Days")
    # ax[0].legend()
    
    # ax[1].plot(x, k_sol.to_numpy(),  c="blue", alpha=0.7, label="Predicted deposit conductivity")
    # ax[1].plot(x, 0.2 * np.ones(len(dfs)), c="red", alpha=0.7, label="True deposit conductivity")
    # ax[1].set_ylabel("Conductivity (W/m*k)")
    # ax[1].set_xlabel("Days")
    # ax[1].legend()
    
    # plt.show()

if __name__ == '__main__':
    # d_path = Path("../../py_data/HEXPractice/UQ")
    # s_path = Path("../../py_data/HEXPractice/UQpred")
    # ran = "uniform"
    # f_type = 1
    # fnames = ["mMTiM", "mMTiL", "mMTiH","mLTiM", "mHTiM"]
    
    # for fname in fnames:
    #     Path(f"{s_path}/{ran}/{fname}").mkdir(parents=True, exist_ok=True) 
    #     if f_type == 0:
    #         r_csv = Path(f"{d_path}/{ran}/{fname}/parallel.csv")
    #         s_csv = Path(f"{s_path}/{ran}/{fname}/parallel.csv")
    #     elif f_type == 1:
    #         r_csv = Path(f"{d_path}/{ran}/{fname}/counter.csv")
    #         s_csv = Path(f"{s_path}/{ran}/{fname}/counter.csv")
    #     main(r_csv, s_csv, f_type)
    
    # d_path = Path("../../py_data/HEXPractice/RN/case3/rndata_h.csv")
    # s_path = None
    # main(d_path, s_path, 0)
    
    cases = [1, 2, 3]
    levels = ['l', 'm', 'h']
    
    for case in cases:
        for level in levels:
            d_path = Path(f"../../py_data/HEXPractice/RN/case{case}/rndata_{level}.csv")
            s_path = Path(f"../../py_data/HEXPractice/RN/case{case}/preds_{level}.csv")
            main(d_path, s_path, 0)