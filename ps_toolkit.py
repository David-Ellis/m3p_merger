import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erfc, erfinv
from scipy.optimize import root_scalar

# TODO: Take cosmology as an input
print("Loaded PS module.\n")

# Cosmological parameters
Omega_m0 = 0.3
Omega_r0 = 8.486e-5
h = 0.7

G = 6.674e-11 # m^3 kg^(-1) s^(-2)
Mpc = 3.086e22 # m 
Msol = 1.988e30 # kg
Mpcpkm = (1e3/3.0869e22)
H = 100*Mpcpkm*h # /s
Omega_DM = 0.267

# rho_c = 3/(8*np.pi*G)*Mpc**3/Msol*H**2 # Msol / Mpc^(-3)
# rho_bg = 0.267*rho_c

rho_bg = 85342512877.42429 #8.66e+10 # FROM PEAK PATCH

z_eq = Omega_m0/Omega_r0 - 1
a_eq = 1/(1+z_eq)

rho_eq = Omega_m0*(a_eq)**3


def PS_Prob(z, M, z0, sig, f = 0.01, epsilon = 1.686):
    '''Calculate probability of halo projenitors at z having mass greater than f*M
    where M is the mass at the final redshift z0
    
    Parameters
    ----------
    z : redshift (float) 
    M : Mass M_sol (float)
    z0 : final redshift
    siglog: interpolated function for the log of the mass variance 
    f = 0.01 (optional): fraction of total mass for progenitors (float)

    Returns
    -------
    prob : probablity (float)
    '''
    sig1 = sig(f*M)
    sig2 = sig(M)
    
    #test_thresh = 1.69*(z-z0)*(z0+1)
    test_thresh = thresh(z, epsilon)/(GrowthFactor(z)/GrowthFactor(0))-thresh(z0, epsilon)/(GrowthFactor(z0)/GrowthFactor(0))
    prob = erfc(test_thresh/np.sqrt(2*(sig1**2 - sig2**2)))
    #prob = erf((thresh(z)-thresh(0))/np.sqrt(2*(sig1**2 - sig2**2)))
    return prob

def solve_PS_Prob(z, M, z0, siglog, f, epsilon = 1.686, printOutput = False):
    """ Equation to be solved in Find_zcol()
    
    Parameters
    ----------
    z : redshift (float) 
    M : Mass M_sol (float)
    z0 : final redshift
    siglog: interpolated function for the log of the mass variance 
    f (optional): fraction of total mass for progenitors (float)
    """
    return 0.5 - PS_Prob(z, M, z0, siglog, f, epsilon = epsilon)

def Find_zcol(Pdata, kdata, masses, z0, f = 0.01, printOutput = False, krange = None, epsilon = 1.686):
    unconverged = 0
    z_col = np.zeros(len(masses))
    
    HMF_PS, M2, f_PS, sigma0, slinedata = PS_HMF(Pdata, kdata, z=0, krange = krange, epsilon = epsilon)
    
    sig = interp1d(M2, sigma0)
    
    if f*min(masses)<min(M2):
        print("Error: Desired mass and proj fraction too small for HMF data")
        print("min(input mass) = {} Msol".format(min(masses)))
        print("min(HMF mass) = {} Msol".format(min(M2)))
    
    for i, M in enumerate(masses):
        sol = root_scalar(solve_PS_Prob, args = (M, z0, sig, f, epsilon), bracket=(1e7, z0))
        if sol.converged == True:
            z_col[i] = sol.root
        else:
            unconverged += 1
        if printOutput == True:
            print("{} of {} complete".format(i+1, len(masses)), end = "\r")
    if printOutput == True:
        print()
    return z_col

class PowerSpec:
    """ Create padded power spectrum
           - Constant at small k
           - Power law at large k
    """
    def __init__(self, k_data, P_data):
        # Create spline of data
        kdata = k_data[P_data>0]
        P_data = P_data[P_data>0]
        self.Pspline = interp1d(np.log10(k_data), np.log10(P_data), kind='cubic')
        
        # Get spline range
        self.klow = min(k_data)
        self.khigh = max(k_data)
        
        # Get value of P at smallest k (to become const)
        self.Plow = P_data[0]
        
        # Use last 2 point to estimate power law
        p = np.polyfit(np.log10(k_data[-2:]), np.log10(P_data[-2:]), deg=1)
        self.power_slope = p[0]
        self.power_height = p[1]
        
    def val(self, k):
        # TODO: Add ability for this to take single values
        
        output = np.zeros(len(k))
        
        # Calculate values within spline range
        spline_mask = (k >= self.klow)*(k <= self.khigh)
        output[spline_mask] = 10**self.Pspline(np.log10(k[spline_mask]))
        
        # Power spec for k below spline range = const
        small_k_mask = k < self.klow
        output[small_k_mask] = self.Plow
        
        # Power spec for k above spline range = power law
        large_k_mask = k > self.khigh
        output[large_k_mask] = 10**(self.power_slope*np.log10(k[large_k_mask]) + self.power_height)
        
        return output
    

def GrowthFactor(z):
    a = 1/(z+1)
    x = a/a_eq
    return 1+2/3*x

def thresh(z, epsilon = 1.686):
    return epsilon*GrowthFactor(z)/(GrowthFactor(z)-1)

def fit_PS(sigma, z, epsilon = 1.686):
    # Press-Schechter fit
    v = thresh(z, epsilon)/sigma
    return np.sqrt(2/np.pi)*v*np.exp(-v**2/2)

def fit_ST(sigma, z, epsilon = 1.686):
    # Sheth-Tormen fit
    A = 0.3222; a = 0.707; p=0.3
    v = thresh(z, epsilon)/sigma
    return A*np.sqrt(2*a/np.pi)*(1+(1/(a*v**2))**p)*v*np.exp(-v**2*a/2)

def fit_TK(sigma, z):
    
    A0 = 1.858659e-01
    a0 = 1.466904
    b0 = 2.571104
    c0 = 1.193958
    A_exp = 0.14
    a_exp = 0.06
    
    #Tinker and Kravtsov
    A = A0 * (1 + z) ** (-A_exp)
    a = a0 * (1 + z) * (-a_exp)
    
    #Omega_m = Omega_m0*(1+z)**3/(Omega_m0*(1+z**3) + Omega_r0*(1+z)**4 + 0.6911) 
    #x = Omega_m - 1
    del_vir = 200 #18*np.pi**2 + 82*x - 39*x**2
    
    alpha = 10 ** ( - ((0.75/np.log10(del_vir / 75.0))**1.2))
    
    b = b0*(1+z)**(-alpha)
    c = c0
    
    return A*((b / sigma) ** a + 1) * np.exp(- c / sigma**2)


def mass_variance(pspec, k, R, z, filter_mode = 'tophat', printOutput = False):
    '''
    Calculates mass varience from the given power spectrum and returns with the mass 
    scale in units M_sol h^3
    '''
    
    M = rho_bg*4/3*np.pi*R**3 # Msol h^3 
    if printOutput == True:
        print("mass_variance: max(P) = {:.3}".format(max(pspec.val(k))))
    sigma = np.zeros(len(M))
    if filter_mode == 'tophat':
        for i in range(len(R)):
            W_th = 3*(np.sin(k*R[i])-k*R[i]*np.cos(k*R[i]))/(k*R[i])**3
            sigma[i] = np.sqrt(1/(2*np.pi**2)*np.trapz(pspec.val(k)*k**2*W_th**2, x=k))
    else:
        print("ERROR: Unexpected filter mode.")
        
    return sigma, M

#############################################################################
def H(a, cosmology):
    '''
    normalised Hubble constant 
    H(a)/H0
    '''
    omega_l, omega_m, omega_r = cosmology
    return (omega_m*a**(-3) + omega_r*a**(-4) + omega_l)**(1/2)

def D_plus(a):
    '''
    Testing hmf module growth function

    '''
    cosmology = [0.6911, 0.3089, 9.16e-5]
    _, omega_m, _ = cosmology
    Hubble = H(a, cosmology)
    
    
    aprime = np.logspace(-7, np.log10(a), 1000)
    integrand = 1/(aprime*H(aprime, cosmology))**3

    integral = np.trapz(integrand, aprime)
    
    return 5/2*omega_m*Hubble*integral
#############################################################################
    
def PS_HMF(P0, k, z=0, mode = 'PS', printOutput = False, epsilon = 1.686, krange = None):
    '''
    Takes power spectrum linearly evolved to z=0 and returns the HMF as predicted by 
    Press-Schechter
    
    inputs:
        Pf - Power spectrum in units Mpc^3
        k - k modes in units Mpc^(-1)
        z - desired redshift
        mode - desired fitting function:
                > PS: Press-Schechter
                > ST: Sheth-Tormen
                > TK: Tinker-Kravtov
        epsilon - fitting parameter for threshold overdensity
                  default = 1.686
        
    returns:
        HMF - Halo mass functionin dn/dlog10(M) Mpc^-3
        M - Halo masses in Msol
        f - fitting function
        sigma - Mass spectrum
        [pspec, k] - The interpolated/extrapolated power function
    '''
    if printOutput == True:
        print("Running PS calc for z = {}".format(z))
        
    d = GrowthFactor(z)/GrowthFactor(0)
    #d = D_plus(1/(1+z))/D_plus(1)
        
    # create power spectrum object to allow for extrapolation
    pspec = PowerSpec(k, P0)
    
    if krange == None:
        k2 = np.logspace(np.log10(pspec.klow), np.log10(pspec.khigh), int(1e5))
    else:
        k2 = np.logspace(np.log10(min(krange)), np.log10(max(krange)), int(1e5))
                     
    R = np.logspace(np.log10(2*np.pi/max(k2)), np.log10(2*np.pi/min(k2)), 200) # h^(-1) Mpc
    if printOutput == True:
        print("Min R = {}, max R = {}".format(min(R), max(R)))
        
    sigma0, M = mass_variance(pspec, k2, R, z, filter_mode = 'tophat')
    if printOutput == True:
        print("max sig (unev) = {}".format(max(sigma0)))
    
    sigma = sigma0*d
    
    if mode == "PS":
        f = fit_PS(sigma, z, epsilon)
    elif mode == "ST":
        f = fit_ST(sigma, z, epsilon)
    elif mode == "TK":
        f = fit_TK(sigma, z)
    else:
        print("ERROR: Unexpected fitting function.")
        return 0
        
    # derivative of mass varience wrt to M
    dsig_dM = abs(np.gradient(np.log(sigma), np.log(M)))
    
    # Calculate HMF
    HMF = rho_bg * f / M * dsig_dM * np.log(10)#* (thresh(z, epsilon)/sigma)
    
    
    return HMF, M, f, sigma, [pspec.val(k2)*d**2, k2]

def find_conc(c, delta):
    """Equation to be solved to find concentration parameter
    as use in solve_conc().
    """
    frac = c**3/(np.log(1+c)-c/(1-c))
    return frac-delta

def solve_conc(scale_densitys):
    unconverged = 0
    concs = np.zeros(len(scale_densitys))
    for i in range(len(scale_densitys)):
        #print(i, end = ' ')
        sol = root_scalar(find_conc, args = scale_densitys[i], bracket=(1e-4, 1e8))
        if sol.converged == True:
            concs[i] = sol.root
        else:
            unconverged += 1
    return concs
    
def pred_conc(mass, C, z0, zcol = [], pspec = None, mode = "NFW", f = 0.01,krange = None, epsilon = 1.686):
    assert (mode == "NFW") or (mode == "Bullock"), "Error: mode not recognised."
    
    if zcol == []:
        assert pspec != None, \
        "Collapse redshift undefined. Therefore require power spectrum data."
        # Unpack power spectrum data
        Pdata, kdata = pspec
        
        # Estimate zcol from PS
        zcol = Find_zcol(Pdata, kdata, mass, z0, f, epsilon = epsilon, krange = krange)
    xcol = (z_eq+1)/(zcol+1); x0 = (z_eq+1)/(z0+1)
    
    if mode == "NFW":
        Omega_m = Omega_m0*(1+z0)**3/(Omega_m0*(1+z0**3) + Omega_r0*(1+z0)**4 + 0.6911) 
        x = Omega_m - 1
        del_vir =18*np.pi**2 + 82*x - 39*x**2
     
        arg = 3*C/del_vir*(xcol**(-3) + xcol**(-4))/(x0**(-3) + x0**(-4))
        return solve_conc(arg)
        
    elif mode == "Bullock":
        return C * ((xcol**(-3) + xcol**(-4))/(x0**(-3) + x0**(-4)))**(1/3)
    
def massfrac(pspec, redshifts, Mtot, mode = "PS"):
    ''' Use PS to estimate bound mass fraction
    '''
    
    assert mode == "PS" or mode == "ST", \
    '''cal_massfrac() Error: Mode not recognised. Use PS, ST. 
    TK currently not working for mass frac calculations'''
    
    Pdata, kdata = pspec
    
    HMFs = [PS_HMF(Pdata, kdata, z=z, mode = mode)[0] for z in redshifts]
    M = PS_HMF(Pdata, kdata, z=redshifts[0], mode = 'PS')[1]
    
    fb = [np.trapz(HMFs[i]*M, np.log10(M))/Mtot for i in range(len(redshifts))]
    
    return fb

def haloNums(pspec, redshifts, boxVol, cutoffmasses=[0], mode="PS"):
    ''' Use PS to estimate number of halos
    '''
    assert mode == "PS" or mode == "ST", \
    '''cal_massfrac() Error: Mode not recognised. Use PS, ST. 
    TK currently not working for halo num calculations'''
    
    Pdata, kdata = pspec
        
    HMFs = [PS_HMF(Pdata, kdata, z=z, mode = mode)[0] for z in redshifts]    
    M = PS_HMF(Pdata, kdata, 0, mode = 'PS')[1] 
   
    HaloNums = [[np.trapz(HMFs[i][M>mass]*(1e-6)**3*boxVol,
    np.log(M[M>mass])) for i in range(len(redshifts))] for mass in cutoffmasses]
    
    return HaloNums