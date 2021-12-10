import numpy as np
def P_oms(f):
   '''Single link optical metrology noise'''
   return (1.5e-11)**2 * (1 + (2e-3/f)**4)

def P_acc(f):
    '''Single test mass acceleration noise'''
    return (3.5e-15)**2 * (1 + (0.4e-3/f)**2) * (1 + (f/8e-3)**4)

def P_n(f, fstar=19.09e-3, L=2.5e9):
    '''Total noise'''
    return P_oms(f)/L**2 + 2*(1+(np.cos(f/fstar))**2) * P_acc(f)/((2*np.pi*f)**4 * L**2)

def S_c(f):
    '''Confusion Noise'''
    A = 9e-45
    alpha = 0.171
    beta = 292
    kappa = 1020
    gamma = 1680
    f_k = 0.00215
    
    return A * f ** (-7 / 3) * np.exp(-(f**alpha) + beta * f * np.sin(kappa*f)) * (1 + np.tanh(gamma*(f_k-f)))

def S_n(f, fstar=19.09e-3):
    '''Lisa Sensitivity'''
    return S_c(f) + 10/(3*2.5e9**2) * (P_oms(f) + (4*P_acc(f)/(2*np.pi*f)**4)) \
        * (1+0.6*(f / fstar)**2)
        
def noise_f(f):
    '''Detector Noise with amplitude sampled from Gaussian distribution 
       with S_n(f) as variance and a random phase (from uniform dist.)'''
    #1/500 is 1/dT. 
    var = np.ones((1, 200))*S_n(f).T * 1/500
    phase = np.exp(1j*np.random.uniform(0, 2*np.pi, var.shape))
    noise = np.random.normal(0, np.sqrt(var)) * phase
    return noise

#f = np.linspace(1e-4, 1e-1, 200)
#Poms = P_oms(f)
#Pacc = P_acc(f)
#S = S_n(f)
#noise = noise(f)
#print(noise)
