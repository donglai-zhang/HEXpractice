import numpy as np
import utils.DataframeGenerator as dfg
from utils.HexClasses import HEX, Fluid, Fouling
from pathlib import Path
from utils.HEXSimulation import run_HEX

def main():
    # time setting
    t_final = 1e4   # s, maximum simulation time
    lplt = 60       # s, monitoring plot lag
    eps = 1e-8      # threshold of psudo steady state
    
    '''
    uncertainty qualification variables
    '''
    T1mean = 523            # if ran == 1, mean temperature of fluid 1 
    T2mean = 603
    m1mean = 0.3            # etc.
    m2mean = 0.5
    ran_mode = "uniform"       # random sampling mode, "norm" or "uniform"
    uq_modes = ["L", "M", "H"]
    m_diffs = [0.1, 0.3, 0.5]       # differences of random data
    Ti_diffs = [0.05, 0.1, 0.15]
    en_diffs = [0.1, 0.2, 0.3]
    
    '''
    start simulation
    '''
    f_type = 0       # flow type: 0 - parallel, 1 - counter
    days = 200       # running days
    d_save = [1, 50, 100, 150, 200]        # days to record daily data of each distributed control volumes

    '''
    constant mass flow mode: "M"
    '''
    m_diff = m_diffs[1]
    for uq_mode, Ti_diff, en_diff in zip(uq_modes, Ti_diffs, en_diffs):
        dfs = dfg.GenDataframe()
        dpath = Path(f"../../py_data/HEXPractice/UQ/{ran_mode}/{'mM' + 'Ti' + uq_mode}")
        
        # initialise HEX
        hex = HEX(L=6.1, ri=22.9e-3 / 2, ro=25.4e-3 / 2, R=50e-3 / 2)
        n = hex.n
        dx = hex.dx
        T0 = hex.T0
        x = np.linspace(dx / 2, hex.L - dx / 2, n)

        #initialise fouling layers
        depo1 = Fouling(pv="Yeap")
        depo2 = Fouling(pv="Yeap")
        
        # initialise fluids
        fluid1 = Fluid(m=0.3, Cp=2916, rho=680, Ti=523, k=0.12, mu=4e-6 * 680)
        fluid2 = Fluid(m=0.5, Cp=4180, rho=1000, Ti=603, k=0.6, mu=8.9e-4)
        
        for k in range(1, days + 1):
            run_HEX(dfs, dpath, k, d_save,
                    hex, n, dx, T0, x, f_type, t_final, eps, fluid1, fluid2, depo1, depo2, lplt, 
                    1, T1mean, T2mean, Ti_diff, m1mean, m2mean, m_diff, en_diff, ran_mode)
        
        dfs.export_Vars(f_type, dpath)
        
    '''
    constant inlet temperature mode: "M"
    '''
    Ti_diff = Ti_diffs[1]
    for uq_mode, m_diff, en_diff in zip(uq_modes, m_diffs, en_diffs):
        dfs = dfg.GenDataframe()
        dpath = Path(f"../../py_data/HEXPractice/UQ/{ran_mode}/{'m' + uq_mode + 'TiM'}")
        
        # initialise HEX
        hex = HEX(L=6.1, ri=22.9e-3 / 2, ro=25.4e-3 / 2, R=50e-3 / 2)
        n = hex.n
        dx = hex.dx
        T0 = hex.T0
        x = np.linspace(dx / 2, hex.L - dx / 2, n)

        #initialise fouling layers
        depo1 = Fouling(pv="Yeap")
        depo2 = Fouling(pv="Yeap")
        
        # initialise fluids
        fluid1 = Fluid(m=0.3, Cp=2916, rho=680, Ti=523, k=0.12, mu=4e-6 * 680)
        fluid2 = Fluid(m=0.5, Cp=4180, rho=1000, Ti=603, k=0.6, mu=8.9e-4)
        
        for k in range(1, days + 1):
            run_HEX(dfs, dpath, k, d_save,
                    hex, n, dx, T0, x, f_type, t_final, eps, fluid1, fluid2, depo1, depo2, lplt, 
                    1, T1mean, T2mean, Ti_diff, m1mean, m2mean, m_diff, en_diff, ran_mode)
        
        dfs.export_Vars(f_type, dpath)
    
if __name__ == '__main__':
    main()