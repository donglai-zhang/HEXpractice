import numpy as np
import scipy
from scipy.optimize import fsolve

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
        self.rfi = self.ri      # inner radius beyond fouling layer
        self.rfo = self.ro      # outer radius beyond fouling layer
        self.Ac1, self.D1, self.As1 = self.inner_Cross(self.rfi)
        self.Ac2, self.D2, self.As2 = self.outer_Cross(self.rfo)

    def inner_Cross(self, rfi):
        Ac1 = np.pi * rfi ** 2        # m^2, cross-sectional area of inner pipe
        D1 = 2 * rfi      # m, charateristic length of inner pipe
        As1 = np.pi * D1 * self.dx         # m^2, inner pipe surface area of each node
        return Ac1, D1, As1

    def outer_Cross(self, rfo):
        Ac2 = np.pi * (self.R ** 2 - rfo ** 2)        # m^2, cross-sectional area of outer annulus
        D2 = 2 * (self.R - self.rfo)    # m, charateristic length of shell
        As2 = 2 * np.pi * rfo * self.dx         # m^2, outer pipe surface area of each node
        return Ac2, D2, As2

    '''
    update parameters dependent on the fouling thickness
    inputs:
    sigmai, sigmao: innter/outter fouling thickness
    '''
    def update_Prams(self, sigma1, sigma2, k1, k2):
        self.rfi = self.ri - sigma1
        self.rfo = self.ro + sigma2
        self.Ac1, self.D1, self.As1 = self.inner_Cross(self.rfi)
        self.Ac2, self.D2, self.As2 = self.outer_Cross(self.rfo)

class Fluid:
    def __init__(self, 
                 m = 1,        # kg/s, mass of flow rate
                 Cp = 4180,    # J/kg*K, heat capacity of fluid1 (cold), default water
                 rho = 1000,   # kg/m^3, density of fluid, default water
                 Ti = 373,     # K, fluid inlet temperature, default water
                 k = 0.6,      # W/(m*K), fluid thermal conductivity, default water
                 mu = 8.9e-4   # Pa*s, dynamic viscocity
                ):
        self.m = m
        self.Cp = Cp
        self.rho = rho
        self.Ti = Ti
        self.k = k
        self.mu = mu
        self.V = m / rho       # m^3/s, volume rate
    
    def get_Inlets(self, Ti, m):
        self.Ti = Ti
        self.m = m
        self.V = self.m / self.rho 
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
        return 0.023 * np.power(Re, 0.8) * np.power(Pr, 0.4)

    ''' 
    get Convective coefficient 
    '''
    def get_h(self, Nu, D):
        return self.k * Nu / D

    '''
    get friction factor
    developed laminar flow: Cf = 64 / Re
    developed turbulent flow: Cf = (0.790 ln(e) - 1.64)^-2
    '''
    def get_Fricion(self, Re):
        return 0.0791 * np.power(Re, -0.25)
    
    '''
    get shear stress
    tau = Cf * rho * v^2 / 2
    '''
    def get_Shear(self, Cf, v):
        return  Cf * self.rho * v ** 2 / 2
    
    ''' 
    get pressure drop
    dP/dL = Cf / D * rho * v ** 2 / 2
    '''
    def get_PressureDrop(self, Cf, D, tau):
        return 4 * tau / D
    
    
    ''' 
    get All parameters
    '''
    def get_Prams(self, Ac, D, As):
        self.v = self.get_Velocity(Ac)
        self.Re = self.get_Re(self.v, D)
        self.Pr = self.get_Pr()
        self.Nu = self.get_Nu(self.Re, self.Pr)
        self.h = self.get_h(self.Nu, D)     # W/m^2*K, convective coefficient 
        self.R = 1 / (self.h * As)          # K/W, thermal resistance
        self.Cf = self.get_Fricion(self.Re)
        self.tau = self.get_Shear(self.Cf, self.v)
        self.dPdx = self.get_PressureDrop(self.Cf, D, self.tau)

class Fouling:
    def __init__(self,
                 simga = 0,           # m, fouling layer
                 Rf = 0,              # K/W, fouling resistance
                 k_l0 = 0.2,          # W/(m*K), material thermal conductivity of a freshly deposited material (lower limit)
                 pv = "EP"            # parameter version
                 ):
        # self.dSigmas = []           # array recording dsigma/dt
        self.sigma = simga
        self.Rf = Rf
        self.k_l0 = k_l0   
        self.pv = pv       

        '''Constants'''
        self.Rg = 8.3145            # J/K*mol, gas constant

    ''' 
    get threshold fouling rate and thickness
    inputs:
    Re: Reynolds number
    Pr: Prandtl number
    Tf: K, film temperature
    tau: Pa, shear stressdepo
    k_L0: W/(m*K), material thermal conductivity of a freshly sited material (lower limit)
    '''
    def THfouling(self, Re, Pr, Tf, tau, k_L0 = 0.2):
        if self.pv == "EP":
            # EP version
            self.alpha = 0.0139
            self.gamma = 4.03e-11
            self.Ef = 48000
        if self.pv == "Yeap":
            # EP version
            self.alpha = 0.0011
            self.gamma = 7.3e-12
            self.Ef = 28000
        
        return self.alpha * np.power(Re, -0.66) * np.power(Pr, -0.33) * np.exp(- self.Ef / (self.Rg * Tf)) - self.gamma * tau        # threshold fouling
    
    '''
    start to simulate fouling
    inputs
    inputs of THfouling
    dt: time difference
    period: simulation period
    '''
    def FoulingSimu(self, Re, Pr, Tf, tau, k_L0, ri, period):
        dRfdt = self.THfouling(Re, Pr, Tf, tau, k_L0)
        self.Rf += dRfdt * period
        
        def solve_sigma(sigma):
            return ri / (ri - sigma) - np.exp(k_L0 * self.Rf / (ri - sigma))
        
        if type(dRfdt) == np.ndarray:
            guess = 1e-5 * np.ones(len(dRfdt))
            self.sigma = fsolve(solve_sigma, guess)
        # for lumped
        else:
            guess = 1e-5
            self.sigma = fsolve(solve_sigma, guess)[0]
        
        if np.mean(self.sigma) < 0:
            self.sigma = 0