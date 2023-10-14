import numpy as np
import matplotlib.pyplot as plt

### Pipe parameters
L = 60      # m, pipe length
r1 = 0.1    # m, pipe radius
r2 = 0.15   # m, outer pipe radius
n = 100     # of nodes used

### Fluid1 parameters (cold)
m1 = 3      # kg/s, mass of flow rate
Cp1 = 4180  # J/kg*K, heat capacity of fluid1 (cold)
rho1 = 1000 # kg/m^3, density of fluid1 (cold)

### Fluid2 parameters (hot)
m2 = 5      # kg/s, mass of flow rate
Cp2 = 4180  # J/kg*K, heat capacity of fluid2 (hot)
rho2 = 1000 # kg/m^3, density of fluid2 (hot)

### Constants
Ac1 = np.pi * r1 ** 2               # cross-sectional area of inner pipe
Ac2 = np.pi * (r2 ** 2 - r1 ** 2)   # ccross-sectional area of outer annulus
dx = L / n                          # node length

### Initialisation
T1i = 400       # K, fluid1 (cold) inlet temperature
T2i = 800       # K, fluid2 (hot) inlet temperature
T0 = 300        # K, inital temperature of fluid

U = 340         # W/n^2*k, oeverall heat transfer coefficient
t_final = 1000  # s, simulation time
dt = 1          # s, time step

T1 = np.ones(n) * T0
T2 = np.ones(n) * T0
x = np.linspace(dx / 2, L - dx / 2, n)

dT1dt = np.ones(n)
dT2dt = np.ones(n)

t = np.arange(0, t_final, dt)

plt.figure(figsize=(9, 6))
### Parallel flow
for j in range(1, len(t)):
    dT1dt[1 : n]  = (m1 * Cp1 * (T1[0 : n - 1] - T1[1 : n]) + U * 2 * np.pi * r1 * dx * (T2[1 : n] - T1[1 : n])) / (rho1 * Cp1 * dx * Ac1)
    dT1dt[0]  = (m1 * Cp1 * (T1i - T1[0]) + U * 2 * np.pi * r1 * dx * (T2[0] - T1[0])) / (rho1 * Cp1 * dx * Ac1)

    dT2dt[1 : n]  = (m2 * Cp2 * (T2[0 : n - 1] - T2[1 : n]) - U * 2 * np.pi * r1 * dx * (T2[1 : n] - T1[1 : n])) / (rho2 * Cp2 * dx * Ac2)
    dT2dt[0]  = (m2 * Cp2 * (T2i - T2[0]) - U * 2 * np.pi * r1 * dx * (T2[0] - T1[0])) / (rho2 * Cp2 * dx * Ac2)

    T1 = T1 + dT1dt * dt
    T2 = T2 + dT2dt * dt
    
    plt.plot(x, T1, c = "blue", label = "Fluid1 (cold)")
    plt.plot(x, T2, c = "red", label = "Fluid2 (hot)")
    plt.axis([0, L, 298, 820])
    plt.xlabel("Distance (m)")
    plt.ylabel("Temperature (K)")
    plt.legend(loc = "upper right")
    plt.plot()
    plt.pause(0.005)
    plt.cla()