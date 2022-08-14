import numpy as np 
from scipy.fft import irfft2
import scipy


def cme_integrator(p,lm,method,fixed_quad_T=10,quad_order=60,quad_vec_T=np.inf):
    b,bet,gam = p
    u = []
    mx = np.copy(lm)

    #initialize the generating function evaluation points
    mx[-1] = mx[-1]//2 + 1
    
    for i in range(len(mx)):
        l = np.arange(mx[i])
        u_ = np.exp(-2j*np.pi*l/lm[i])-1
        u.append(u_)
    g = np.meshgrid(*[u_ for u_ in u], indexing='ij')
    for i in range(len(mx)):
        g[i] = g[i].flatten()[:,np.newaxis]

    #define function to integrate by quadrature.
    fun = lambda x: INTFUN(x,g,b,bet,gam)
    if method=='quad_vec':
        T = quad_vec_T*(1/bet+1/gam+1)
        gf = scipy.integrate.quad_vec(fun,0,T)[0]
    if method=='fixed_quad':
        T = fixed_quad_T*(1/bet+1/gam+1)
        gf = scipy.integrate.fixed_quad(fun,0,T,n=quad_order)[0]

    #convert back to the probability domain, renormalize to ensure non-negativity.
    gf = np.exp(gf) #gf can be multiplied by k in the argument, but this is not relevant for the 3-parameter input.
    gf = gf.reshape(tuple(mx))
    Pss = irfft2(gf, s=tuple(lm)) 
    EPS=1e-16
    Pss[Pss<EPS]=EPS
    Pss = np.abs(Pss)/np.sum(np.abs(Pss)) #always has to be positive...
    return Pss

def INTFUN(x,g,b,bet,gam):
    """
    Computes the Singh-Bokes integrand at time x. Used for numerical quadrature in cme_integrator.
    """
    if not np.isclose(bet,gam): #compute weights for the ODE solution.
        f = bet/(bet-gam)
        U = b*(np.exp(-bet*x)*(g[0]-g[1]*f)+np.exp(-gam*x)*g[1]*f)
    else:
        g[1] *= (b*gam)
        g[0] *= b
        U = np.exp(-bet*x)*(g[0] + bet * g[1]* x)
    return U/(1-U)

def get_moments(p):
    ''' Returns mean, variance, standard deviation, and mean + 2 sigma for a given rate parameter input.
    '''
    b,beta,gamma=p
    
    r = np.array([1/beta, 1/gamma])
    MU = b*r
    VAR = MU*np.array([1+b,1+b*beta/(beta+gamma)])
    STD = np.sqrt(VAR)
    xmax = np.ceil(MU)
    xmax = np.ceil(xmax + 4*STD)
    xmax = np.clip(xmax,30,np.inf).astype(int)
    return MU, VAR, STD, xmax


def calculate_exact_cme(p,method,xmax_fun):
    
    '''Given parameter vector p, calculate the exact probabilites using CME integrator.'''
 
    p1 = 10**p
    
    MU, VAR, STD, xmax = get_moments(p1)
    
    xmaxc = xmax_fun(xmax)
    
    xmaxc = np.array([int(xmaxc[0]),int(xmaxc[1])])
    
    y = cme_integrator(np.array(p1),xmaxc+1,method=method)
    
    y_save = y[0:xmax[0]+1,0:xmax[1]+1]
    
    return([p,np.array(y_save)])


def generate_param_vectors(N,b_bounds= [1,300], beta_bounds = [0.05,50], gamma_bounds = [0.05,50], ):
    '''Generates N parameter vectors randomly spaced in logspace between bounds.'''
    
    logbnd = np.log10([[1,300],[0.05,50],[0.05,50]])
    dbnd = logbnd[:,1]-logbnd[:,0]
    lbnd = logbnd[:,0]
    param_vectors = np.zeros((N,3))


    i = 0
    a = 0
    while i<N:
        a += 1 
        th = np.random.rand(3)*dbnd+lbnd
        MU, VAR, STD, xmax=get_moments(10**th)

        if xmax[0] > 1e3:
            continue
        if xmax[1] > 1e3:
            continue
        if th[1] == th[2]:
            continue
        else:
            param_vectors[i,:] = np.float32(th)
            i+=1
            
    return(param_vectors)
