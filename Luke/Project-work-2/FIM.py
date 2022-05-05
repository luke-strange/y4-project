import numpy as np

M_s = 4.9254909476412675e-06

def central_f(a, M):
    '''
    Central frequency of a perturbed 
    Kerr black hole.
    ---------------
    Inputs:
    a - dimensionless spin parameter
    M     - Black hole mass (seconds)
    
    Returns:
    f     - Central frequency
    '''

    f = 32e3 * (1 - 0.63 * (1-a)**(3/10)) * (M_s/M)
    return f

def quality(a):
    '''
    Quality Factor - Measures approx no. of cycles
    in the waveform.
    '''
    return 2*(1-a)**(-9/20)

def dhdtc(t, tc, mc):
  ''' Derivative wrt tc for sinc model'''
  x = (t-tc)/ (np.pi * mc)
  brac = np.sin(x) + -x * np.cos(x)
  return np.pi * mc**2 * brac / (t-tc)**2

def dhdmc(t, tc, mc):
  '''derivative wrt m_c for sinc model'''
  x = (t-tc)/ (np.pi * mc)
  return (np.sin(x)/x) - (np.cos(x)/mc)

def dhdm(t, a, m):
  '''derivative wrt m for ringdown model'''
  Q = quality(a)
  f = central_f(a, m)
  dhdf = -(2*np.pi)**(0.5) * np.exp(-np.pi*f*t/Q) * ((np.pi*t/Q * np.cos(2*np.pi*f*t)) + (2*np.pi*t*np.sin(2*np.pi*f*t)))
  dfdm = 32e3 * M_s * (0.63*(1-a)**(3/10) -1) / m**2
  return dhdf*dfdm

def dhda(t, a, m): 
  '''Derivative wrt a for ringdown model'''
  Q = quality(a)
  f = central_f(a, m)
  dhdf = -(2*np.pi)**(0.5) * np.exp(-np.pi*f*t/Q) * ((np.pi*t/Q * np.cos(2*np.pi*f*t)) + (2*np.pi*t*np.sin(2*np.pi*f*t)))
  dfda = 3 * 32e3 * 0.63 * M_s / (10 * (1-a)**(7/10) * m)
  dhdq = (2*np.pi)**0.5 * (np.pi*f*t/Q**2) * np.exp(-np.pi*f*t/Q) * np.cos(2*np.pi*f*t)
  dqda = 9 / (10 * (1-a)**(29/20))

  return dhdf*dfda + dhdq*dqda

def inner(v, w, Sn=1):
  '''
  Compute the inner product of two arrays v, w
  '''
  return 2 * np.sum(v*w + v*w)/Sn

def FIM(d1, d2):
  '''Compute the 2d fisher information matrix'''
  L11 = inner(d1, d1)
  L22 = inner(d2, d2)
  L12 = inner(d1, d2)
  #L21 = L12
  return np.array([[L11, L12], [L12, L22]])

def cov(F):
  ''' 
  Find covariance matrix via the inverse
  of the Fisher Information Matrix
  
  Params
  ----------

  F : array

      Fisher Information Matrix

  Returns
  ----------

  cov: array
      
       The covariance matrix (inv. of F)
  '''
  cov = np.linalg.inv(F)
  return cov

def get_sigma(F):
  '''
  Find sigma and correlation values
  '''
  Cov = cov(F)
  Sxx, Syy, Sxy = Cov[0][0], Cov[1][1], Cov[0][1]
  
  sigx = np.sqrt(Sxx)
  sigy = np.sqrt(Syy)
  corr = Sxy / np.sqrt(Sxx*Syy)
  return sigx, sigy, corr