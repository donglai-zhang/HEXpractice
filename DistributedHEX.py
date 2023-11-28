import numpy as np
import utils.DataframeGenerator as dfg
from utils.HexClasses import HEX, Fluid, Fouling
from pathlib import Path
from utils.HEXSimulation import run_HEX

def main():
    dfs = dfg.GenDataframe()
    # time setting
    t_final = 1e4   # s, maximum simulation time
    lplt = 60       # s, monitoring plot lag
    eps = 1e-8      # threshold of psudo steady state
    

    # initialise HEX
    hex = HEX(L=6.1, ri=22.9e-3 / 2, ro=25.4e-3 / 2, R=50e-3 / 2)
    n = hex.n
    dx = hex.dx
    T0 = hex.T0
    x = np.linspace(dx / 2, hex.L - dx / 2, n)
    
    '''
    initialise fouling layers
    pv : CF or EDB(in error), parameters version of threshold fouling model
    '''
    depo1 = Fouling(pv="Yeap")
    depo2 = Fouling(pv="Yeap")
    
    # initialise fluids
    fluid1 = Fluid(m=0.3, Cp=2916, rho=680, Ti=523, k=0.12, mu=4e-6 * 680)
    fluid2 = Fluid(m=0.5, Cp=4180, rho=1000, Ti=603, k=0.6, mu=8.9e-4)
    
    # start simulation
    f_type = 0       # flow type: 0 - parallel, 1 - counter
    days = 200       # running days
    d_save = [1, 50, 100, 150, 200]        # days to record daily data of each distributed control volumes
    ran = 1          # 1 - random inlet temperatures and flow rates
    
    # random variables
    T1mean = fluid1.Ti      # if ran == 1, mean temperature of fluid 1 
    T2mean = fluid2.Ti
    m1mean = fluid1.m       # etc.
    m2mean = fluid2.m
    ran_mode = "norm"       # random mode, "norm" or "uniform"
    Ti_diff = 0.1
    m_diff = 0.3
    en_diff = 0.2
    
    # data path
    if ran == 0:
        dpath = Path("../../py_data/HEXPractice/disHEX/cinlet")
        for k in range(1, days + 1):
            run_HEX(dfs, dpath, k, d_save,
                hex, n, dx, T0, x, f_type, t_final, eps, fluid1, fluid2, depo1, depo2, lplt, 
                ran, 0, 0, 0, 0, 0, 0, 0, None)
        dfs.export_Vars(f_type, dpath)
        
    elif ran == 1:
        dpath = Path("../../py_data/HEXPractice/disHEX/rinlet")
        for k in range(1, days + 1):
            run_HEX(dfs, dpath, k, d_save,
                hex, n, dx, T0, x, f_type, t_final, eps, fluid1, fluid2, depo1, depo2, lplt, 
                ran, T1mean, T2mean, Ti_diff, m1mean, m2mean, m_diff, en_diff, ran_mode)
        dfs.export_Vars(f_type, dpath)
    
if __name__ == '__main__':
    main()