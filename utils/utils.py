import numpy as np
import pandas as pd

'''
get film temperature
inputs 
dQdt: J/t, heat duty
T1/T2: K,cold/hot flow temperatures
R1/R2: # K/W, convective thermal resistance
'''
def get_Tf(Q, T1, T2, R1, R2):
    Ts1 = Q * R1 + T1         # K, surface temperature
    Ts2 = T2 - Q * R2
    Tf1 = T1 + 0.55 * (Ts1 - T1)
    Tf2 = Ts2 + 0.55 * (T2 - Ts2)
    return Tf1, Tf2 

'''
get heat duty
inputs 
UA: W*m^2/n^2*k Overall heat transfer coefficient times surface area (1 / Total Resistance)
T1/T2: K, cold/hold temeratures
Cp: heat 
'''
def get_Q(UA, T1, T2):
    return UA * np.abs(T2 - T1)

'''
generate random inlet parameters
'''
def gen_Inlets(Tmin, Tdiff, mmin, mdiff):
    Ti = Tmin + Tdiff * np.random.rand()
    m = mmin + mdiff * np.random.rand()

    return Ti, m

'''
export daily data vs. HEX distance
'''
def export_DayVars(
    f_type, dpath, k, Q, 
    F1T, F1Re, F1h, F1R, Rf1, Sigma1, dP1dx, 
    F2T, F2Re, F2h, F2R, Rf2, Sigma2, dP2dx
):
    df_day = pd.DataFrame()
    df_day["Q"] = Q
    df_day["F1T"] = F1T
    df_day["F1Re"] = F1Re
    df_day["F1h"] = F1h
    df_day["F1R"] = F1R
    df_day["Rf1"] = Rf1
    df_day["Sigma1"] = Sigma1
    df_day["dP1/dx"] = dP1dx
    df_day["F2T"] = F2T
    df_day["F2Re"] = F2Re
    df_day["F2h"] = F2h
    df_day["F2R"] = F2R
    df_day["Rf2"] = Rf2
    df_day["Sigma2"] = Sigma2
    df_day["dP2/dx"] = dP2dx
    
    if f_type == 0:
        df_day.to_csv(f"{dpath}/parallel_day_{k}.csv", index=False)
    elif f_type == 1:
        df_day.to_csv(f"{dpath}/counter_day_{k}.csv", index=False)  