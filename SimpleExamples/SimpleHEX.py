import numpy as np
import matplotlib.pyplot as plt

class HEX:
    def __init__(self, 
                 L = 6,         # m, pipe length
                 r1 = 0.0093,      # m, pipe radius
                 r2 = 0.0120,     # m, outer pipe radius
                 n = 100,       # of nodes used
                 T0 = 300,      # K, inital temperature of fluid
                 U = 340        # W/n^2*k, oeverall heat transfer coefficient
                ):
        self.L = L
        self.r1 = r1
        self.r2 = r2
        self.n = n
        self.T0 = T0
        self.Ac1 = np.pi * self.r1 ** 2        # cross-sectional area of inner pipe
        self.Ac2 = np.pi * (self.r2 ** 2 - self.r1 ** 2)       # ccross-sectional area of outer annulus
        self.dx = self.L / self.n
        self.U = U
        
        def oht_update(self, U):
            self.U = U

class Fluid:
    def __init__(self, 
                 m = 3,        # kg/s, mass of flow rate
                 Cp = 4180,    # J/kg*K, heat capacity of fluid1 (cold)
                 rho = 10000,   # kg/m^3, density of fluid
                 Ti = 400,     # K, fluid inlet temperature
                ):
        self.m = m
        self.Cp = Cp
        self.rho = rho
        self.Ti = Ti

if __name__ == '__main__':
    # initialise hex and fluids
    hex = HEX()
    fluid1 = Fluid(m=0.3)
    fluid2 = Fluid(m=0.3, Ti = 600)
    
    # initialise variables
    L = hex.L    
    r1 = hex.r1 
    r2 = hex.r2
    n = hex.n
    T0 = hex.T0 
    U = hex.U
    Ac1 = hex.Ac1
    Ac2 = hex.Ac2
    dx = hex.dx
    
    m1 = fluid1.m
    Cp1 = fluid1.Cp
    rho1 = fluid1.rho
    T1i = fluid1.Ti
    
    m2 = fluid2.m
    Cp2 = fluid2.Cp
    rho2 = fluid2.rho
    T2i = fluid2.Ti
    
    # initialise time and temperatures
    t_final = 100    # s, simulation time
    dt = 0.02         # s, time step
    t = np.arange(0, t_final, dt)
    T1 = np.ones(n) * T0
    T2 = T1.copy()
    x = np.linspace(dx / 2, L - dx / 2, n)
    dT1dt = np.zeros(n)
    dT2dt = np.zeros(n)
    
    plt.figure(figsize=(9, 6))
    plt.xlabel("Distance (m)")
    plt.ylabel("Temperature (K)")
    plt.legend(loc = "upper right")
    f_type = 0
    # Parallel flow: 0
    if f_type == 0:
        for j in range(1, len(t)):
            dT1dt[1 : n]  = (m1 * Cp1 * (T1[0 : n - 1] - T1[1 : n]) + U * 2 * np.pi * r1 * dx * (T2[1 : n] - T1[1 : n])) / (rho1 * Cp1 * dx * Ac1)
            dT1dt[0]  = (m1 * Cp1 * (T1i - T1[0]) + U * 2 * np.pi * r1 * dx * (T2[0] - T1[0])) / (rho1 * Cp1 * dx * Ac1)

            dT2dt[1 : n]  = (m2 * Cp2 * (T2[0 : n - 1] - T2[1 : n]) - U * 2 * np.pi * r1 * dx * (T2[1 : n] - T1[1 : n])) / (rho2 * Cp2 * dx * Ac2)
            dT2dt[0]  = (m2 * Cp2 * (T2i - T2[0]) - U * 2 * np.pi * r1 * dx * (T2[0] - T1[0])) / (rho2 * Cp2 * dx * Ac2)

            T1 = T1 + dT1dt * dt
            T2 = T2 + dT2dt * dt
            
            print("courant 1:", np.max(U * 2 * np.pi * r1 * dx) * dt / dx)
            
            if ((j % 10) == 0):
                plt.plot(x, T1, c = "blue", label = "Fluid1 (cold)")
                plt.plot(x, T2, c = "red", label = "Fluid2 (hot)")
                plt.axis([0, L, T0 - 2, 1500])
                plt.legend()
                plt.plot()
                plt.pause(0.005)
                plt.cla()

    # Counter flow: 1
    else:
        for j in range(1, len(t)):
            dT1dt[1 : n]  = (m1 * Cp1 * (T1[0 : n - 1] - T1[1 : n]) + U * 2 * np.pi * r1 * dx * (T2[1 : n] - T1[1 : n])) / (rho1 * Cp1 * dx * Ac1)
            dT1dt[0]  = (m1 * Cp1 * (T1i - T1[0]) + U * 2 * np.pi * r1 * dx * (T2[0] - T1[0])) / (rho1 * Cp1 * dx * Ac1)

            dT2dt[0 : n - 1]  = (m2 * Cp2 * (T2[1 : n] - T2[0 : n - 1]) - U * 2 * np.pi * r1 * dx * (T2[0 : n - 1] - T1[0 : n - 1])) / (rho2 * Cp2 * dx * Ac2)
            dT2dt[n - 1]  = (m2 * Cp2 * (T2i - T2[n - 1]) - U * 2 * np.pi * r1 * dx * (T2[n - 1] - T1[n - 1])) / (rho2 * Cp2 * dx * Ac2)

            T1 = T1 + dT1dt * dt
            T2 = T2 + dT2dt * dt

            if ((j % 10) == 0):
                plt.plot(x, T1, c = "blue", label = "Fluid1 (cold)")
                plt.plot(x, T2, c = "red", label = "Fluid2 (hot)")
                plt.axis([0, L, T0 - 2, 1500])
                plt.plot()
                plt.pause(0.005)
                plt.cla()