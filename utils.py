import numpy as np

'''
get film temperature
inputs 
dQdt: J/t, heat duty
T1/T2: K,cold/hot flow temperatures
R1/R2: # K/W, convective thermal resistance
'''
def get_Tf(hd, T1, T2, R1, R2):
    Ts1 = hd * R1 + T1         # K, surface temperature
    Ts2 = T2 - hd * R2
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