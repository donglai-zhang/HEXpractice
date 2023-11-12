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
    hex = HEX(L=6.1, ri=9.93e-3, ro=12.7e-3, R=30e-3)
    n = hex.n
    dx = hex.dx
    T0 = hex.T0
    x = np.linspace(dx / 2, hex.L - dx / 2, n)
    
    '''
    initialise fouling layers
    pv : CF or EDB(in error), parameters version of threshold fouling model
    '''
    depo1 = Fouling(pv="CF")
    depo2 = Fouling(pv="CF")
    
    # initialise fluids
    fluid1 = Fluid(m=0.3, Cp=1900, rho=900, Ti=573, k=0.12, mu=4e-6 * 900)
    fluid2 = Fluid(m=1, Cp=4180, rho=1000, Ti=800, k=0.7, mu=8.9e-4)
    
    # start simulation
    f_type = 0       # flow type: 0 - parallel, 1 - counter
    days = 200       # running days
    d_save = [1, 50, 100, 150, 200]        # days to record daily data of each distributed control volumes
    ran = 0          # 1 - random inlet temperatures and flow rates
    T1min = 563      # if ran == 1, minimum temperature of fluid 1 
    T2min = 790
    T1diff = 20      # if ran == 1, maximum temperature difference of fluid 1 
    T2diff = 20
    m1min = 0.25     # etc.
    m2min = 0.9
    m1diff = 0.1
    m2diff = 0.2
    
    # data path
    if ran == 0:
        dpath = Path("../../py_data/HEXPractice/disHEX/cinlet")
    elif ran == 1:
        dpath = Path("../../py_data/HEXPractice/disHEX/rinlet")
    
    for k in range(1, days + 1):
        run_HEX(dfs, dpath, k, d_save,
                hex, n, dx, T0, x, f_type, t_final, eps, fluid1, fluid2, depo1, depo2, lplt, 
                ran, T1min, T2min, T1diff, T2diff, m1min, m2min, m1diff, m2diff)
        
    dfs.export_Vars(f_type, dpath)
    
if __name__ == '__main__':
    main()