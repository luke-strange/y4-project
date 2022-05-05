#!/usr/bin/env python
# coding: utf-8

# In[57]:


#importing utils.py
from utils import *
from FIM import dhdtc, dhdmc, FIM, get_sigma
region = [[5, 10], [45, 55]]
import os
import matplotlib.pyplot as pp

def h_func_mt(params):
    '''A toy model for sinc function signal of GWs 
    in the time domain.
    t = time
    tc = coalescence time
    Mc = chirp mass'''
    Mc = params[0]
    tc = params[1]
    t = np.linspace(-100, 100, 200)
    x = (t - tc) / Mc
    w = 2*np.pi/Mc
    signal = np.sinc(x/np.pi)
    
    return signal

def syntrain(size,  region, varx=['Mc', 'tc'], 
             varall=True, seed=None, single=True, noise=0):
    """Makes a training set using the ROMAN NN. It returns labels (for `varx`,
        or for all if `varall=True`), indicator vectors, and ROM coefficients
        (with `snr` and `noise`). Note that the coefficients are kept on the GPU.
        Parameters are sampled randomly within `region`."""
    #device = dev
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    with torch.no_grad():
        xs = torch.zeros((size, len(region)), dtype=torch.float, device=device)

        for i, r in enumerate(region):
            xs[:,i] = r[0] + (r[1] - r[0]) * torch.rand((size,), dtype=torch.float, device=device)
        
        xs_1 = xs.detach().cpu().double().numpy()
        
        #generating signals
        signal = np.apply_along_axis(h_func_mt, 1, xs_1)[:,:]
            
        signal_r, signal_i = numpy2cuda(signal.real), 0 #set to 0 as we dont need imaginary part
        
        #setting up real alphas
        alphas = torch.zeros((size, 200), dtype=torch.float if single else torch.double, device=device)

        norm = torch.sqrt(torch.sum(signal_r*signal_r + signal_i*signal_i, dim=1))
        
        #add random signal amplitudes #snr
        A = [8,12]
        const = numpy2cuda(np.random.uniform(*A, size=size))

        #add noise to the signal and normalise
        alphas[:,:] = const[:, np.newaxis]*signal_r/norm[:, np.newaxis] + (noise*torch.randn((size, 200), device=device))


    xr = np.zeros((size, len(region)), 'd')
    xr = xs.detach().cpu().double().numpy()
    del xs, signal_r, signal_i

    for i, r in enumerate(region):
        xr[:,i] = (xr[:,i] - r[0]) / (r[1] - r[0])

    if isinstance(varx, list):
        ix = ['Mc','tc'].index(varx[0])
        jx = ['Mc','tc'].index(varx[1])    

        i = np.digitize(xr[:,ix], xstops, False) - 1
        i[i == -1] = 0; i[i == qdim] = qdim - 1
        px = np.zeros((size, qdim), 'd'); px[range(size), i] = 1

        j = np.digitize(xr[:,jx], xstops, False) - 1
        j[j == -1] = 0; j[j == qdim] = qdim - 1
        py = np.zeros((size, qdim), 'd'); py[range(size), j] = 1

        if varall:
            return xr, np.einsum('ij,ik->ijk', px, py), alphas
        else:
            return xr[:,[ix,jx]], np.einsum('ij,ik->ijk', px, py), alphas    
    else:
        ix = ['Mc','tc'].index(varx)

        i = np.digitize(xr[:,ix], xstops, False) - 1
        i[i == -1] = 0; i[i == qdim] = qdim - 1
        px = np.zeros((size, qdim), 'd'); px[range(size), i] = 1

        if varall:
            return xr, px, alphas
        else:
            return xr[:,ix], px, alphas

#Training the network
      
def syntrainer(net, syntrain, lossfunction=None, iterations=300, 
               batchsize=None, initstep=1e-3, finalv=1e-5, clipgradient=None, validation=None,
               seed=None, single=True):
  """Trains network NN against training sets obtained from `syntrain`,
  iterating at most `iterations`; stops if the derivative of loss
  (averaged over 20 epochs) becomes less than `finalv`."""

  if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)

  indicatorloss = 'l' in lossfunction.__annotations__ and lossfunction.__annotations__['l'] == 'indicator'  
  
  if validation is not None:
    raise NotImplementedError
    
    vlabels = numpy2cuda(validation[1] if indicatorloss else validation[0], single)
    vinputs = numpy2cuda(validation[2], single)
  
  optimizer = optim.Adam(net.parameters(), lr=initstep)

  training_loss, validation_loss = [], []
  
  for epoch in range(iterations):
    t0 = time.time()

    xtrue, indicator, inputs = syntrain()
    labels = numpy2cuda(indicator if indicatorloss else xtrue, single)

    if batchsize is None:
      batchsize = inputs.shape[0]
    batches = inputs.shape[0] // batchsize

    averaged_loss = 0.0    
    
    for i in range(batches):
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs[i*batchsize:(i+1)*batchsize])
      loss = lossfunction(outputs, labels[i*batchsize:(i+1)*batchsize])
      loss.backward()
      
      if clipgradient is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), clipgradient)
      
      optimizer.step()

      # print statistics
      averaged_loss += loss.item()

    training_loss.append(averaged_loss/batches)

    if validation is not None:
      loss = lossfunction(net(vinputs), vlabels)
      validation_loss.append(loss.detach().cpu().item())

    if epoch == 1:
      print("One epoch = {:.1f} seconds.".format(time.time() - t0))

    if epoch % 50 == 0:
      print(epoch,training_loss[-1],validation_loss[-1] if validation is not None else '')

    try:
      if len(training_loss) > iterations/10:
        training_rate = np.polyfit(range(20), training_loss[-20:], deg=1)[0]
        if training_rate < 0 and training_rate > -finalv:
          print(f"Terminating at epoch {epoch} because training loss stopped improving sufficiently: rate = {training_rate}")
          break

      if len(validation_loss) > iterations/10:
        validation_rate = np.polyfit(range(20), validation_loss[-20:], deg=1)[0]        
        if validation_rate > 0:
          print(f"Terminating at epoch {epoch} because validation loss started worsening: rate = {validation_rate}")
          break
    except:
      pass
          
  print("Final",training_loss[-1],validation_loss[-1] if validation is not None else '')
      
  if hasattr(net,'steps'):
    net.steps += iterations
  else:
    net.steps = iterations
    
def synlike(a, syntrain, iterations=1000000):
  with torch.no_grad():
    # note aimag defined with plus here, exp below modified consistently
    # [areal] = [aimag] = nbasis x nsignals 
    areal, aimag = torch.t(a[:,0::2]), torch.t(a[:,1::2])

    # [anorm] = nsignals
    anorm = torch.sum(areal*areal + aimag*aimag, dim=0)
    
    cnt, norm, like = 0, 0, 0
    adapt = None
    while cnt < iterations:
      # [pxt] = nbatch x qdim
      _, pxt, alpha = syntrain()
      cnt = cnt + alpha.shape[0]

      # cpxt = torch.from_numpy(pxt).cuda()
      # handle 2D indicator array (assume square qdim)
      cpxt = numpy2cuda(pxt if pxt.ndim == 2 else pxt.reshape((pxt.shape[0], pxt.shape[1]*pxt.shape[1])),
                        alpha.dtype == torch.float32)
      
      # [alphareal] = [alphaimag] = nbatch x nbasis
      alphareal, alphaimag = alpha[:,0::2], alpha[:,1::2]

      # [norm] = qdim
      norm += torch.sum(cpxt, dim=0)
      
      # [alphanorm] = nbatch
      alphanorm = torch.sum(alphareal*alphareal + alphaimag*alphaimag, dim=1)

      # automatic normalization of exponentials based on the first batch
      loglike = (alphareal @ areal + alphaimag @ aimag - 0.5*alphanorm.unsqueeze(1) - 0.5*anorm) 
      if adapt is None:
        adapt = torch.max(loglike, dim=0)[0]
      loglike -= adapt

      # [like] = qdim x nsignals = (qdim x nbatch) @ [(nbatch x nbasis) @ (nbasis x nsignals) + (nbatch x 1) + nsignals]
      # remember broadcasting tries to match the last dimension [so A @ b = sum(A * b,axis=1)]
      like += torch.t(cpxt) @ torch.exp(loglike)

    # (qdim x nsignals) * (qdim x 1)
    like = like / norm.unsqueeze(1)  
    
    # [ret] = nsignals x qdim = (nsignals x qdim) * qdim
    ret = torch.t(like / torch.sum(like, dim=0))
    
    nret = ret.detach().cpu().numpy()
    
  return nret if pxt.ndim == 2 else nret.reshape((nret.shape[0], pxt.shape[1], pxt.shape[1]))

def plotgauss(xtrue, indicator, inputs, net=None, like=None, varx=None, twodim=False, istart=6, single=True):
    
    if isinstance(like, types.FunctionType):
        like = like(inputs)
    
    netinput = inputs[istart:(istart+1),:]
    with torch.no_grad():
        pars = net(netinput).detach().cpu().numpy().flatten()
    
    if twodim:
        Fxx, Fyy = pars[1::6]**2, pars[3::6]**2

        Fxy = np.arctan(pars[4::6]) / (0.5*math.pi) * pars[1::6] * pars[3::6]

        weight = torch.softmax(torch.from_numpy(pars[5::6]),dim=0).numpy()

        dx  = (pars[2::6] if varx == 'tc' else pars[0::6]) - xmid[:,np.newaxis]

        Cxx = Fxx / (Fxx*Fyy - Fxy*Fxy) if varx == 'tc' else Fyy / (Fxx*Fyy - Fxy*Fxy) 

        pdf = np.sum(weight * np.exp(-0.5*dx**2/Cxx) / np.sqrt(2*math.pi*Cxx) * xwid, axis=1)

      # logmod = scs.logsumexp(np.log(weight) - 0.5*(Fxx*dx*dx + Fyy*dy*dy + 2*Fxy*dx*dy) + 0.5*np.log(Fxx*Fyy - Fxy*Fxy), axis=2)
    else:
        if len(pars) == 2:
            pdf = np.exp(-0.5*(xmid - pars[0])**2/pars[1]**2) / math.sqrt(2*math.pi*pars[1]**2) * xwid
        else:
            wg = torch.softmax(torch.from_numpy(pars[2::3]), dim=0).numpy()
            pdf = np.sum(wg * np.exp(-0.5*(xmid[:,np.newaxis] - pars[0::3])**2/pars[1::3]**2) / np.sqrt(2*math.pi*pars[1::3]**2) * xwid, axis=1)

    fig = pp.figure(1, figsize=fig_size)
    ax = fig.add_subplot(1,1,1)

    if varx=='Mc':
        ax.set_xlabel('Chirp Mass $\mathcal{M}_c$ $[GM_{\odot}/c^3] / s$')
        ax.set_ylabel('$P(M_c|d)$')
        xm = region[0][0] + xmid * (region[0][1] - region[0][0])
        xt = region[0][0] + xtrue * (region[0][1] - region[0][0])
        
    else:
        ax.set_ylabel('$P(t_c|d)$')
        ax.set_xlabel('Coalescence Time $t_c$ / s')
        xm = region[1][0] + xmid * (region[1][1] - region[1][0])
        xt = region[1][0] + xtrue * (region[1][1] - region[1][0])

    ax.plot(xm, pdf, color=dodgerblue, label='Network Posterior')
    ax.plot(xm, like[istart], color=princetonorange, label='Likelihood')

    if xtrue.ndim == 2:
        ix = ['Mc', 'tc'].index(varx)
        ax.axvline(xt[istart, ix], color='C2', ls=':')

    else:
        ax.axvline(xt[istart], color='C2', ls=':')

    ax.legend()                  
    pp.show()
    
def makecontour(xytrue, indicator, inputs, net=None, like=None, istart=0, single=True):
    '''makes the 2d posterior deensity contour plot'''
    if isinstance(like, types.FunctionType):
        like = like(inputs)
  

    netinput = inputs[istart:(istart+1),:]

    with torch.no_grad():
        pars = net(netinput).detach().cpu().numpy().flatten()
          # xm, xe, xc = netmeanGn2(netinput, net=net)

    xs, ys = np.meshgrid(xmid, xmid, indexing='ij')    

    dx  = pars[0::6] - xs[:,:,np.newaxis]
    Fxx = pars[1::6]**2

    dy  = pars[2::6] - ys[:,:,np.newaxis]
    Fyy = pars[3::6]**2

    Fxy = np.arctan(pars[4::6]) / (0.5*math.pi) * pars[1::6] * pars[3::6]


    weight = torch.softmax(torch.from_numpy(pars[5::6]),dim=0).numpy()

    logmod = scs.logsumexp(np.log(weight) - 0.5*(Fxx*dx*dx + Fyy*dy*dy + 2*Fxy*dx*dy) + 0.5*np.log(Fxx*Fyy - Fxy*Fxy), axis=2)

    q = np.exp(logmod)
    q /= np.sum(q)

    pmax = np.max(like[istart])
    qmax = np.max(q)
    vmax = min(pmax, qmax)

    fig = pp.figure(1, figsize=fig_size)
    ax = fig.add_subplot(1,1,1)
    ax.plot([xytrue[istart,0]], [xytrue[istart,1]], 'ro', label='True Value')
    ax.contour(xs, ys, like[istart], colors=princetonorange, alpha=0.8, levels=[0.01*vmax,0.14*vmax,0.61*vmax],                      label='Likelihood')
    ax.contour(xs, ys, q,              colors=dodgerblue, alpha=0.8, levels=[0.01*vmax,0.14*vmax,0.61*vmax],                      label='Network Posterior')
    ax.legend()
    ax.set_ylabel('$P(t_c|d)$')
    ax.set_xlabel('$P(\mathcal{M}_c|d)$')
    pp.show()
    # note these are set-value posterior levels, not true set-mass-containing contours


def net_sigma(xtrue, indicator, inputs, N=5000, net=None, like=None, varx=None, twodim=False, istart=0, single=True):
    '''Takes test set and returns the sigma values from the network evaluated posteriors'''
    sigmas = []
    for i in range(N):
        # make Gaussian mixture

        netinput = inputs[istart+i:(istart+i+1),:]
        with torch.no_grad():
            pars = net(netinput).detach().cpu().numpy().flatten()
    
        if twodim:
            Fxx, Fyy = pars[1::6]**2, pars[3::6]**2

            Fxy = np.arctan(pars[4::6]) / (0.5*math.pi) * pars[1::6] * pars[3::6]

            weight = torch.softmax(torch.from_numpy(pars[5::6]),dim=0).numpy()

            dx  = (pars[2::6] if varx == 'tc' else pars[0::6]) - xmid[:,np.newaxis]

            Cxx = Fxx / (Fxx*Fyy - Fxy*Fxy) if varx == 'tc' else Fyy / (Fxx*Fyy - Fxy*Fxy) 
        sigma = np.sqrt(Cxx)
        sigmas.append(sigma)
    return np.concatenate(sigmas, axis=0)

def plotsigma(*mutest, net):
    Mc = mutest[0][:,0]
    tc = mutest[0][:,1]
    idx = np.argsort(Mc)
    
    ##get sigmas from network
    net_sgms_Mc = net_sigma(*mutest, net=net, varx='Mc', twodim=True, istart=0)
    net_sgms_tc = net_sigma(*mutest, net=net, varx='tc', twodim=True, istart=0)
    
    #rescale back to unnormalised regions
    Mcs, tcs = rescale(Mc, tc, region)
    
    ##get sigmas from FIM
    t = np.linspace(-100, 100, 200)
    sg_mc = []
    sg_tc = []
    for i, j in zip(Mcs, tcs):
        v = dhdtc(t, i, j)
        w = dhdmc(t, i, j)
        I = FIM(v, w)
        got = get_sigma(I)
        sg_mc.append(got[0])
        sg_tc.append(got[1])
    
    diff_Mc = np.subtract(sg_mc, net_sgms_Mc) / net_sgms_Mc
    diff_tc = np.subtract(sg_tc, net_sgms_tc) / net_sgms_tc
    
    fig1 = pp.figure(1, figsize=fig_size)
    ax1 = fig1.add_subplot(111)    # The big subplot
    ax1.scatter(Mcs, diff_Mc, alpha=0.1, color=dodgerblue)
    ax1.set_xlabel('Chirp Mass $\mathcal{M}_c$ $[GM_{\odot}/c^3] / s$')
    ax1.set_ylabel('$\Delta \sigma_{\mathcal{M}_c} / \sigma_{\mathcal{M}_c}$')

    fig2 = pp.figure(2, figsize=fig_size)
    ax2 = fig2.add_subplot(111)
    ax2.scatter(tcs, diff_tc, alpha=0.1, color=princetonorange)
    ax2.set_xlabel('Coalescence Time $t_c$ / s')
    ax2.set_ylabel('$\Delta \sigma_{t_c} / \sigma_{t_c}$')
    pp.show()
    
def perci(xtrue, indicator, inputs, net=None, like=None, varx=None, twodim=False, istart=6, single=True):
    for i in range(1):
    # make Gaussian mixture

        netinput = inputs[istart+i:(istart+i+1),:]
        with torch.no_grad():
            pars = net(netinput).detach().cpu().numpy().flatten()
    
        if twodim:
            Fxx, Fyy = pars[1::6]**2, pars[3::6]**2

            Fxy = np.arctan(pars[4::6]) / (0.5*math.pi) * pars[1::6] * pars[3::6]

            weight = torch.softmax(torch.from_numpy(pars[5::6]),dim=0).numpy()

            dx  = (pars[2::6] if varx == 'tc' else pars[0::6]) - xmid[:,np.newaxis]

            Cxx = Fxx / (Fxx*Fyy - Fxy*Fxy) if varx == 'tc' else Fyy / (Fxx*Fyy - Fxy*Fxy) 

            pdf = np.sum(weight * np.exp(-0.5*dx**2/Cxx) / np.sqrt(2*math.pi*Cxx) * xwid, axis=1)

          # logmod = scs.logsumexp(np.log(weight) - 0.5*(Fxx*dx*dx + Fyy*dy*dy + 2*Fxy*dx*dy) + 0.5*np.log(Fxx*Fyy - Fxy*Fxy), axis=2)
        else:
            if len(pars) == 2:
                pdf = np.exp(-0.5*(xmid - pars[0])**2/pars[1]**2) / math.sqrt(2*math.pi*pars[1]**2) * xwid
            else:
                wg = torch.softmax(torch.from_numpy(pars[2::3]), dim=0).numpy()
                pdf = np.sum(wg * np.exp(-0.5*(xmid[:,np.newaxis] - pars[0::3])**2/pars[1::3]**2) / np.sqrt(2*math.pi*pars[1::3]**2) * xwid, axis=1)
    
        #ax[i].plot(xmid, pdf, color='C0', label='Network Posterior')
        mean, sigma = pars[2::6] if varx == 'tc' else pars[0::6], np.sqrt(Cxx)

        # get xtrue
    
        if xtrue.ndim == 2:
            ix = ['Mc', 'tc'].index(varx)
            Xtrue = xtrue[istart+i, ix]
        else:
            Xtrue = xtrue[istart+i]
  
    return mean, sigma, Xtrue, weight

def err(p, n=1000):
    return np.sqrt(p*(1-p)/n)

def p_p(*mutest, net, N=1000, varx=['Mc', 'tc']):
    '''finds number of test signals in credible regions'''
    CR = np.linspace(0, 1, 66)
    error = err(CR)
    counts1 = np.zeros(len(CR))
    counts2 = np.zeros(len(CR))
    for var in varx:
        norms = [] #normalisation constants for pdfs.
        mus = []
        sigmas = []
        truvals = []
        weights = []
        
        for m in range(int(N)):
        #setting constants and normalising signal
            mu, sigma, truval, weight = perci(*mutest, net=net, varx=var, twodim=True, istart=m)
            func = lambda x: weight * np.exp(-0.5*(x-mu)**2/sigma) / np.sqrt(2*math.pi*sigma) * xwid
            integ = integrate.quad(func, 0, 1)
            A = 1/integ[0]
            norms.append(A)
            mus.append(mu[0])
            sigmas.append(sigma[0])
            truvals.append(truval)
            weights.append(weight[0])

        i = 0
        for j in CR:
            count = 0
            for n, m, s, t, w in zip(norms, mus, sigmas, truvals, weights):
              #print('Mu = ', m)
                n_func = lambda x: n * w * np.exp(-0.5*((x-m)**2)/s) / np.sqrt(2*math.pi*s) * xwid
                dx = abs(m - t)
              #print(m, t, dx)
                upper, lower = m+dx, m-dx
                if upper > 1:
                    upper = 1
                if lower < 0:
                    lower = 0
              #print('limits =', upper, 'and', lower)
                integ = integrate.quad(n_func, lower, upper)[0]
                if integ <= j:
                    count +=1
            if var == 'Mc':
                counts1[i] += count
                    
            else: 
                counts2[i] += count
                    
            i += 1
    return counts1, counts2, error, CR

##Plotting settings 

# Example of matplotlib settings using latex rendering etc
fig_width_pt  = 245.27 #513.17              # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27                   # Convert pt to inches
golden_mean   = (np.sqrt(5)-1.0)/2.0        # Aesthetic ratio
fig_width     = fig_width_pt*inches_per_pt  # width in inches
fig_height    = fig_width*golden_mean       # height in inches
fig_size      = [fig_width,fig_height]

params = {'backend': 'pdf',
        'axes.labelsize': 8,
        'lines.markersize': 4,
        'font.size': 14,
        'xtick.major.size':6,
        'xtick.minor.size':5,  
        'ytick.major.size':6,
        'ytick.minor.size':5, 
        'xtick.major.width':0.5,
        'ytick.major.width':0.5,
        'xtick.minor.width':0.5,
        'ytick.minor.width':0.5,
        'lines.markeredgewidth':1,
        'axes.linewidth':1.2,
        'legend.fontsize': 8,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'savefig.dpi':200,
        'path.simplify':True,
        'font.family': 'serif',
        'font.sans-serif': ['Bitstream Vera Sans'],
        'font.serif': ['Times New Roman'],
        'text.latex.preamble': r'\usepackage{amsmath}',
        'text.usetex':True}

pp.rcParams.update(params)

#%matplotlib inline
pp.rcParams['figure.dpi'] = 150
#%config InlineBackend.figure_format = 'retina'

# Specify some aesthetic colours
dodgerblue      = '#1E90FF'
princetonorange = '#ff8f00'
tanzaniagreen   = '#1eb53a'







