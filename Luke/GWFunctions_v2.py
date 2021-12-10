#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import constants


# In[3]:


def coalescence_time(f, chirp_mass):
    'Coalescence time of BBH merger'
    return 5 * (8 * np.pi * f)**(-8/3) * chirp_mass**(-5/3)

def freq(t, tc, chirp_mass):
    'finding frequency as function of time'
    return (((tc - t)/5)**(-3/8)) / (8*np.pi*(chirp_mass**(5/8)))

def phase(tc, t, ch_mass, phi_c):   #what is this called?????
    return phi_c - 2 * ((tc - t)/(5 * ch_mass))**(5/8)

def phi_t_Lbar(time, phi_0):
    'LISA orbital phase'
    return phi_0 + (2*np.pi*time/constants.year)

def theta_s_t(theta_sbar, phi_t_Lbar, phi_sbar):
    'Source location'
    cos_theta = 0.5 * np.cos(theta_sbar) - (np.sqrt(3)/2)*np.sin(theta_sbar)*np.cos(phi_t_Lbar-phi_sbar)
    return np.arccos(cos_theta)

def alpha_i_t(i, t, alpha_0):
    'LISA Arm orientation'
    T = 31536000
    return 2*np.pi*t/T - np.pi/12 - (i-1)*np.pi/3 + alpha_0

def phi_s_t(theta_sbar, phi_t_bar, phi_sbar, alpha1):
    'Source location in (unbarred) detector frame'
    return alpha1 + np.pi/12 + np.arctan((np.sqrt(3)*np.cos(theta_sbar) + np.sin(theta_sbar)*np.cos(phi_t_bar - phi_sbar))                                   /(2*np.sin(theta_sbar)*np.sin(phi_t_bar - phi_sbar)) )

def psi_s_t(theta_Lbar, phi_Lbar, theta_sbar, phi_sbar, phi_t_Lbar, theta_s_t_):
    'Polarisation angle'
    L_dot_z = 0.5 * np.cos(theta_Lbar) - ( (np.sqrt(3)/2) * np.sin(theta_Lbar) * np.cos(phi_t_Lbar - phi_Lbar) )
    #print(L_dot_z)
    L_dot_n = np.cos(theta_Lbar)*np.cos(theta_sbar) + np.sin(theta_Lbar)*np.sin(theta_sbar)*np.cos(phi_Lbar - phi_sbar)
    global cos_i
    cos_i = L_dot_n
    
    cross = (0.5*np.sin(theta_Lbar)*np.sin(theta_sbar)*np.sin(phi_Lbar - phi_sbar)) -     (np.sqrt(3)/2)*np.cos(phi_t_Lbar)*( (np.cos(theta_Lbar)*np.sin(theta_sbar)*np.sin(phi_sbar) -                                       np.cos(theta_sbar)*np.sin(theta_Lbar)*np.sin(phi_Lbar)) )    - (np.sqrt(3)/2)*np.sin(phi_t_Lbar)*(np.cos(theta_sbar)*np.sin(theta_Lbar)*np.cos(phi_Lbar) -                                       np.cos(theta_Lbar)*np.sin(theta_sbar)*np.cos(phi_sbar))
    
    tan_psi = (L_dot_z - L_dot_n * np.cos(theta_s_t_)) / cross
    
    return np.arctan(tan_psi)

def doppler_phase(f, theta_sbar, phi, phi_sbar):
    'doppler phase due to LISA motion'
    R = constants.astronomical_unit/constants.speed_of_light
    return 2 * np.pi * f * R * np.sin(theta_sbar) * np.cos(phi - phi_sbar)

def F_plus(theta_s, phi_s, psi_s):
    'Detector Beam Pattern Coefficient'
    return (0.5 * (1 + np.cos(theta_s)**2) * np.cos(2*phi_s) * np.cos(2*psi_s)) -                         (np.cos(theta_s) * np.sin(2*phi_s) * np.sin(2*psi_s) )

def F_cross(theta_s, phi_s, psi_s):
    'Detector Beam Pattern Coefficient'
    return (0.5 * (1+np.cos(theta_s)**2) * np.cos(2*phi_s) * np.sin(2*psi_s)) +                         (np.cos(theta_s) * np.sin(2*phi_s) * np.cos(2*psi_s))

def phi_P_I_t(cos_i, F_plus, F_cross): #change to phi
    'Polarisation Phase'
    return np.arctan2( (2*cos_i*F_cross), ((1 + (cos_i**2))*F_plus) )

def A_t(M_c, f, D_L):
    'Waveform Amplitude'
    return 2 * M_c**(5/3) * (np.pi*f)**(2/3) / D_L

def A_p_t(F_plus, F_cross, cos_i):
    'Polarization Amplitude'
    return np.sqrt(3)/2 * (((1+cos_i**2)**2 * F_plus**2) + (4 * cos_i**2 * F_cross**2))**(1/2) 

def h_t(A_t, A_p_t, phase, phi_P_I_t, doppler_phase):
    'Strain signal'
    return A_t * A_p_t * np.cos(phase + phi_P_I_t + doppler_phase)

def phi_f(f, phi_c, chirp_mass):
    'phase as func of freq'
    return phi_c - 2*(8*np.pi*chirp_mass*f)**(-5/3)

def fft(A_p_t, f, chirp_mass, D_L, phi_f, phi_p_t, phi_d_t):
    return (5/96)**0.5 * np.pi**(-2/3) * (1/D_L) * A_p_t * chirp_mass**(5/6) * f**(-7/6) * np.exp(1j * (phi_f - phi_p_t - phi_d_t))

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
    '''Random gaussian noise with random phase between 0 2pi'''
    phase = np.exp(1j*np.random.uniform(0, 2*np.pi, (1,200)))
    noise = np.random.normal(0, 1) * phase
    return noise

def h_func_f(chirps, Nsamples=200):
    M_c = chirps
    theta_Lbar = np.pi/5
    phi_Lbar = np.pi/11
    theta_sbar = 2*np.pi/7
    phi_sbar = 7*np.pi/12
    alpha_0 = 0
    phi_0 = 0
    phi_c = 0
    det_no = 1
    fmin = 1e-4
    D_L = constants.parsec * 1e9 / constants.speed_of_light                   #luminosity distance in seconds (1Gpc).

    #calculating coalescence time, f_isco and t_isco
    tc = coalescence_time(fmin, M_c)                                          
    f_isco = 1 / (np.pi * 6**(3/2) * 2**(6/5) *M_c)
    t_isco = tc - 5 * (8*np.pi*f_isco)**(-8/3) * M_c**(-5/3)

    #time and frequency arrays
    t = np.linspace(0, t_isco, Nsamples)
    f = freq(t, tc, M_c)

    #params
    phi = phase(tc, t, M_c, phi_c)
    phi_f_t = phi_f(f, phi_c, M_c)
    phi_t_Lbar_ = phi_t_Lbar(t, phi_0)
    theta_s_t_ = theta_s_t(theta_sbar, phi_t_Lbar_, phi_sbar)
    alpha_t_ = alpha_i_t(det_no, t, alpha_0)
    phi_s_t_ = phi_s_t(theta_sbar, phi_t_Lbar_, phi_sbar, alpha_t_)
    psi_s_t_ = psi_s_t(theta_Lbar, phi_Lbar, theta_sbar, phi_sbar, phi_t_Lbar_, theta_s_t_)
    doppler_phase_ = doppler_phase(f, theta_sbar, phi_t_Lbar_, phi_sbar)
    F_plus_ = F_plus(theta_s_t_, phi_s_t_, psi_s_t_)
    F_cross_ = F_cross(theta_s_t_, phi_s_t_, psi_s_t_)
    phi_P_I_t_ = phi_P_I_t(cos_i, F_plus_, F_cross_)
    A_t_ = A_t(M_c, f, D_L)
    A_p_t_ = A_p_t(F_plus_, F_cross_, cos_i)

    strain = h_t(A_t_, A_p_t_, phi, phi_P_I_t_, doppler_phase_)
    fourier_signal = fft(A_p_t_, f, M_c, D_L, phi_f_t, phi_P_I_t_, doppler_phase_)
    
    return fourier_signal + noise_f(f)

def h_func_f2(angles, noise_ind, Nsamples=200):
    '''Input params are 2 angles; cos_theta and phi.
    ------------
     Returns whitened strain signal with noise based on LISA sensitivity'''
    M_c = 4.95
    theta_Lbar = np.pi/5
    phi_Lbar = np.pi/11
    theta_sbar = np.arccos(angles[0])
    phi_sbar = angles[1]
    alpha_0 = 0
    phi_0 = 0
    phi_c = 0
    det_no = 1
    fmin = 1e-4
    D_L = constants.parsec * 1e9 / constants.speed_of_light                   #luminosity distance in seconds (1Gpc).

    #calculating coalescence time, f_isco and t_isco
    tc = coalescence_time(fmin, M_c)                                          
    f_isco = 1 / (np.pi * 6**(3/2) * 2**(6/5) *M_c)
    t_isco = tc - 5 * (8*np.pi*f_isco)**(-8/3) * M_c**(-5/3)

    #time and frequency arrays
    t = np.linspace(0, t_isco, Nsamples)
    f = freq(t, tc, M_c)

    #params
    phi = phase(tc, t, M_c, phi_c)
    phi_f_t = phi_f(f, phi_c, M_c)
    phi_t_Lbar_ = phi_t_Lbar(t, phi_0)
    theta_s_t_ = theta_s_t(theta_sbar, phi_t_Lbar_, phi_sbar)
    alpha_t_ = alpha_i_t(det_no, t, alpha_0)
    phi_s_t_ = phi_s_t(theta_sbar, phi_t_Lbar_, phi_sbar, alpha_t_)
    psi_s_t_ = psi_s_t(theta_Lbar, phi_Lbar, theta_sbar, phi_sbar, phi_t_Lbar_, theta_s_t_)
    doppler_phase_ = doppler_phase(f, theta_sbar, phi_t_Lbar_, phi_sbar)
    F_plus_ = F_plus(theta_s_t_, phi_s_t_, psi_s_t_)
    F_cross_ = F_cross(theta_s_t_, phi_s_t_, psi_s_t_)
    phi_P_I_t_ = phi_P_I_t(cos_i, F_plus_, F_cross_)
    A_t_ = A_t(M_c, f, D_L)
    A_p_t_ = A_p_t(F_plus_, F_cross_, cos_i)

    strain = h_t(A_t_, A_p_t_, phi, phi_P_I_t_, doppler_phase_)
    fourier_signal = fft(A_p_t_, f, M_c, D_L, phi_f_t, phi_P_I_t_, doppler_phase_)
 
    return fourier_signal/np.sqrt(S_n(f).T) + noise_ind * noise_f(f)


# In[ ]:




