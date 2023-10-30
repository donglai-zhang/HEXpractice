import numpy as np
import utils.DataframeGenerator as dfg
from utils.HexClasses import HEX, Fluid, Fouling
from pathlib import Path
from utils.HEXSimulation import run_HEX

def main():
    dfs = dfg.GenDataframe()
    
    # time setting
    t_final = 100       # s, maximum simulation time
    lplt = 60       # s, monitoring plot lag
    eps = 1e-8      # threshold of psudo steady state
    

    # initialise HEX
    hex = HEX(L=6.1, ri=9.93e-3, ro=12.7e-3, R=20e-3)
    n = hex.n
    dx = hex.dx
    T0 = hex.T0
    x = np.linspace(dx / 2, hex.L - dx / 2, n)
    
    # initialise fouling layers
    depo1 = Fouling(mode = "CF")
    depo2 = Fouling(mode = "CF")
    
    # initialise fluids
    fluid1 = Fluid(m=0.3, Cp=1900, rho=900, Ti=573, k=0.12, mu=4e-6)
    fluid2 = Fluid(m=1, Ti=800)
    
    # start simulation
    f_type = 1       # flow type: 0 - parallel, 1 - counter
    days = 200       # running days
    d_save = [1, 50, 100, 150, 200]        # days to record daily data
    ran = 1          # 1 - random inlet temperatures and flow rates
    
    # data path
    if ran == 0:
        dpath = Path("../../py_data/HEXPractice/cinlet")
    elif ran == 1:
        dpath = Path("../../py_data/HEXPractice/rinlet")
    
    for k in range(1, days + 1):
        run_HEX(dfs, dpath, k, d_save,
                hex, n, dx, T0, x, f_type, t_final, 
                eps, fluid1, fluid2, depo1, depo2, lplt, ran)
        
    dfs.export_Vars(f_type, dpath)
    
if __name__ == '__main__':
    main()