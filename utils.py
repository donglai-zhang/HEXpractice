import numpy as np

'''
get film temperature
inputs 
dQdt: J/t, heat duty
T1/T2: K,cold/hot flow temperatures
R1/R2: # K/W, convective thermal resistance
'''
def get_Tf(dQdt, T1, T2, R1, R2):
    Ts1 = dQdt * R1 + T1         # K, surface temperature
    Ts2 = T2 - dQdt * R2
    Tf1 = 0.5 * (T1 + Ts1)
    Tf2 = 0.5 * (Ts2 + T2)
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
def gen_Inlets():
    T1i = 273 + 200 * np.random.rand()
    m1 = 4 + 5 * np.random.rand()
    T2i = 500 + 100 * np.random.rand()
    m2 = 4 + 5 * np.random.rand()

    return T1i, m1, T2i, m2