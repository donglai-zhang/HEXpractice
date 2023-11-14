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
    T1mean = 573            # if ran == 1, mean temperature of fluid 1 
    T2mean = 800
    m1mean = 0.3            # etc.
    m2mean = 1
    ran_mode = "norm"       # random sampling mode, "norm" or "uniform"
    uq_modes = ["L", "M", "H"]
    m_diffs = [0.1, 0.3, 0.5]       # differences of random data
    Ti_diffs = [0.1, 0.2, 0.3]
    
    '''
    start simulation
    '''
    f_type = 1       # flow type: 0 - parallel, 1 - counter
    days = 200       # running days
    d_save = [1, 50, 100, 150, 200]        # days to record daily data of each distributed control volumes

    '''
    constant mass flow mode: "M"
    '''
    m_diff = m_diffs[1]
    for uq_mode, Ti_diff in zip(uq_modes, Ti_diffs):
        dfs = dfg.GenDataframe()
        dpath = Path(f"../../py_data/HEXPractice/UQ/{ran_mode}/{'mM' + 'Ti' + uq_mode}")
        
        # initialise HEX
        hex = HEX(L=6.1, ri=9.93e-3, ro=12.7e-3, R=30e-3)
        n = hex.n
        dx = hex.dx
        T0 = hex.T0
        x = np.linspace(dx / 2, hex.L - dx / 2, n)

        #initialise fouling layers
        depo1 = Fouling(pv="CF")
        depo2 = Fouling(pv="CF")
        
        # initialise fluids
        fluid1 = Fluid(m=m1mean, Cp=1900, rho=900, Ti=T1mean, k=0.12, mu=4e-6 * 900)
        fluid2 = Fluid(m=m2mean, Cp=4180, rho=1000, Ti=T2mean, k=0.7, mu=8.9e-4)
        
        for k in range(1, days + 1):
            run_HEX(dfs, dpath, k, d_save,
                    hex, n, dx, T0, x, f_type, t_final, eps, fluid1, fluid2, depo1, depo2, lplt, 
                    1, T1mean, T2mean, Ti_diff, m1mean, m2mean, m_diff, ran_mode)
        
        dfs.export_Vars(f_type, dpath)
        
    '''
    constant inlet temperature mode: "M"
    '''
    Ti_diff = Ti_diffs[1]
    for uq_mode, m_diff in zip(uq_modes, m_diffs):
        dfs = dfg.GenDataframe()
        dpath = Path(f"../../py_data/HEXPractice/UQ/{ran_mode}/{'m' + uq_mode + 'TiM'}")
        
        # initialise HEX
        hex = HEX(L=6.1, ri=9.93e-3, ro=12.7e-3, R=30e-3)
        n = hex.n
        dx = hex.dx
        T0 = hex.T0
        x = np.linspace(dx / 2, hex.L - dx / 2, n)

        #initialise fouling layers
        depo1 = Fouling(pv="CF")
        depo2 = Fouling(pv="CF")
        
        # initialise fluids
        fluid1 = Fluid(m=0.3, Cp=1900, rho=900, Ti=573, k=0.12, mu=4e-6 * 900)
        fluid2 = Fluid(m=1, Cp=4180, rho=1000, Ti=1000, k=0.7, mu=8.9e-4)
        
        for k in range(1, days + 1):
            run_HEX(dfs, dpath, k, d_save,
                    hex, n, dx, T0, x, f_type, t_final, eps, fluid1, fluid2, depo1, depo2, lplt, 
                    1, T1mean, T2mean, Ti_diff, m1mean, m2mean, m_diff, ran_mode)
        
        dfs.export_Vars(f_type, dpath)
    
if __name__ == '__main__':
    main()