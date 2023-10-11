import numpy as np

class HEX:
    def __init__(self, 
                 L = 6,         # m, pipe length
                 ri = 0.1,      # m, tube inner radius
                 ro = 0.12,     # m, tube outer radius
                 R =  0.15,     # m, shell radius
                 k = 397,       # W/(m*K), material thermal conductivity
                 n = 100,       # of nodes used
                 T0 = 273,      # K, inital temperature of fluid
                ):
        self.L = L
        self.ri = ri
        self.ro = ro
        self.R = R
        self.k = k
        self.n = n
        self.T0 = T0
        self.dx = self.L / self.n              # m, length of each node
        self.dRwall = np.log(ro / ri) / (2 * self.k * np.pi * self.dx)      # K/W, wall thermal resistance of each node

        '''Parameters dependent on fouling thickness'''
        self.rfi = np.ones(self.n) * self.ri      # inner radius beyond fouling layer
        self.rfo = np.ones(self.n) * self.ro      # outer radius beyond fouling layer
        self.Ac1, self.D1, self.As1 = self.inner_Cross(self.rfi)
        self.Ac2, self.D2, self.As2 = self.outer_Cross(self.rfo)

    def inner_Cross(self, rfi):
        Ac1 = np.ones(self.n) * np.pi * rfi ** 2        # m^2, cross-sectional area of inner pipe
        D1 = np.ones(self.n) * 2 * rfi      # m, charateristic length of inner pipe
        As1 = np.ones(self.n) * np.pi * D1 * self.dx         # m^2, inner pipe surface area of each node
        return Ac1, D1, As1

    def outer_Cross(self, rfo):
        Ac2 = np.ones(self.n) * np.pi * (self.R ** 2 - rfo ** 2)        # m^2, cross-sectional area of outer annulus
        D2 = np.ones(self.n) * 4 * Ac2 / (2 * np.pi * (self.R + rfo))      # m, charateristic length of shell
        As2 = np.ones(self.n) * 2 * rfo * self.dx         # m^2, outer pipe surface area of each node
        return Ac2, D2, As2

    '''
    update parameters dependent on the fouling thickness
    inputs:
    sigmai, sigmao: innter/outter fouling thickness
    '''
    def update_Prams(self, sigma1, sigma2):
        self.rfi = self.ri - sigma1
        self.rfo = self.ro + sigma2
        self.Ac1, self.D1, self.As1 = self.inner_Cross(self.rfi)
        self.Ac2, self.D2, self.As2 = self.outer_Cross(self.rfo)

class Fluid:
    def __init__(self, 
                 m = 5,        # kg/s, mass of flow rate
                 Cp = 4180,    # J/kg*K, heat capacity of fluid1 (cold)
                 rho = 1000,   # kg/m^3, density of fluid
                 Ti = 373,     # K, fluid inlet temperature
                 k = 0.7,      # W/(m*K), fluid thermal conductivity
                 mu =  0.001   # Pa*s, dynamic viscocity
                ):
        self.m = m
        self.Cp = Cp
        self.rho = rho
        self.Ti = Ti
        self.k = k
        self.mu = mu
        self.V = m / rho       # m^3/s, volume of flow rate
    
    ''' 
    get fluid velocity
    inputs
    Ac: flow crosectional area 
    '''
    def get_Velocity(self, Ac):
        return self.V / Ac     # m/s, flow velocity

    ''' 
    get Reynolds number
    inputs
    D: hydraulic diameter 
    '''
    def get_Re(self, v, D):
        return v * D * self.rho / self.mu

    ''' 
    get Prandtl number 
    '''
    def get_Pr(self):
        return self.mu * self.Cp / self.k
    
    ''' 
    get Nusselt number 
    '''
    def get_Nu(self, Re, Pr):
        return 0.023 * (Re ** 0.8) * (Pr ** 0.4)

    ''' 
    get Convective coefficient 
    '''
    def get_h(self, Nu, D):
        return self.k * Nu / D

    '''
    get friction factor
    '''
    def get_Fricion(self, Re):
        return 0.0035 + 0.0264 * Re ** (-0.42)
    
    '''
    get shear stress
    '''
    def get_Shear(self, Cf, v):
        return  Cf * self.rho * v ** 2 / 2
    
    ''' 
    get All parameters
    '''
    def get_Prams(self, Ac, D, As):
        self.v = self.get_Velocity(Ac)
        self.Re = self.get_Re(self.v, D)
        self.Pr = self.get_Pr()
        self.Nu = self.get_Nu(self.Re, self.Pr)
        self.h = self.get_h(self.Nu, D)     # W/m^2*K, convective coefficient 
        self.R = 1 / (As * self.h)      # K/W, fluid thermal resistance
        self.Cf = self.get_Fricion(self.Re)
        self.tau = self.get_Shear(self.Cf, self.v)

    ''' 
    get pressure drop
    '''
    def get_PressureDrop(self, Cf, v, Rflow):
        return 4 * Cf * self.rho * v ** 2 / (4 * Rflow)

class Fouling:
    def __init__(self,
                 Rf = 0,            # K/W, fouling resistance
                 simga = 0          # m, fouling layer  
                 ):
        self.Rf = Rf
        self.sigma = simga
        # self.dRfs = []              # array recording dRf/dt
        # self.dSigmas = []           # array recording dsigma/dt

        '''Constants'''
        self.alpha = 0.0139         # K*m^2/W*s, constant fouling coefficient
        self.beta = -0.66           # constant fouling coefficient
        self.gamma = 4.03e-11       # m^4*N*k/J, constant fouling coefficient
        self.Ef = 48000             # J/mol, constant activation energy
        self.Rg = 8.3145            # J/K*model, gas constant

    ''' 
    get threshold fouling rate and thickness
    inputs:
    Re: Reynolds number
    Pr: Prandtl number
    Tf: K, film temperature
    tau: Pa, shear stress
    k_L0: W/(m*K), material thermal conductivity of a freshly deposited material (lower limit)
    '''
    def THfouling(self, Re, Pr, Tf, tau, k_L0 = 0.2):
        dRfdt = self.alpha * Re ** self.beta * Pr ** (-0.33) * np.exp(- self.Ef / (self.Rg * Tf)) - self.gamma * tau
        dSigmadt = k_L0 * dRfdt
        return dRfdt, dSigmadt
    
    '''
    start to simulate fouling
    inputs
    inputs of THfouling
    dt: time difference
    '''
    def FoulingSimu(self, Re, Pr, Tf, tau, k_L0, dt):
        dRfdt, dSigmadt = self.THfouling(Re, Pr, Tf, tau, k_L0)
        # self.dRfs.append(dRfdt)
        # self.dSigmas.append(dSigmadt)
        self.Rf += dRfdt * dt
        self.sigma += dSigmadt * dt