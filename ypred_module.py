import numpy as np
from scipy import stats

import torch
torch.set_default_tensor_type(torch.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F

import train_conditional as train
import exact_cme as cme

# if no model is given, will use the final trained model stored in ./models/
model_path = './models/best_model_MODEL'       
npdf = 10

# load in model
model = train.MLP(7,10,256,256)
model.load_state_dict(torch.load(model_path))
model.eval()

# precalculate lngammas
lnfactorial = torch.special.gammaln(torch.arange(1003))


def get_NORM(npdf,quantiles='cheb'):
    '''' Returns quantiles based on the number of kernel functions npdf. 
    Chebyshev or lijear, with chebyshev as default.
    '''
    if quantiles == 'lin':
        q = np.linspace(0,1,npdf+2)[1:-1]
        norm = stats.norm.ppf(q)
        norm = torch.tensor(norm)
        return norm
    if quantiles == 'cheb':
        n = np.arange(npdf)
        q = np.flip((np.cos((2*(n+1)-1)/(2*npdf)*np.pi)+1)/2)

        norm = stats.norm.ppf(q)
        norm = torch.tensor(norm)
        return norm

def get_conditional_moments(MU, VAR, COV, n_range):
    ''' Get moments of conditional distributions (lognormal moment matching) given overall distribution
    mearn, variance, standard deviation, covariance over a range of nascent values
    '''
    logvar = np.log((VAR/MU**2)+1)
    logstd = np.sqrt(logvar)
    logmean = np.log(MU**2/np.sqrt(VAR+MU**2))

    logcov = np.log(COV * np.exp(-(logmean.sum()+logvar.sum()/2)) +1 ) 
    logcorr = logcov/np.sqrt(logvar.prod())

    logmean_cond = logmean[1] + logcorr * logstd[1]/logstd[0] * (np.log(n_range+1) - logmean[0])
    logstd_cond = logstd[1] * np.sqrt(1-logcorr**2)   
   
    # convert to tensors
    logstd_cond = torch.tensor([logstd_cond],dtype=torch.float).repeat(len(n_range),1)
    logmean_cond = torch.tensor(logmean_cond,dtype=torch.float)

    return(logmean_cond,logstd_cond)


def generate_grid(logmean_cond,logstd_cond,NORM):
    ''' Generate grid of kernel means based on the log mean and log standard devation of a conditional distribution.
    Generates the grid of quantile values in NORM, scaled by conditional moments.
    '''
    logmean_cond = torch.reshape(logmean_cond,(-1,1))
    logstd_cond = torch.reshape(logstd_cond,(-1,1))
    translin = torch.exp(torch.add(logmean_cond,logstd_cond*NORM))
    
    return translin

def get_ypred_at_RT(p,w,hyp,n_range,m_range,norm,training=False):
    '''Given a parameter vector (tensor) and weights (tensor), and hyperparameter,
    calculates ypred (Y), or approximate probability. Calculates over array of nascent (n_range) and mature (m_range) values.
    '''
        
    p_vec = 10**p[:,0:3]
    logmean_cond = p[:,3]
    logstd_cond = p[:,4]
    
    
    hyp = hyp*5+1
    
    npdf = w.shape[1]
    
    if training == True:
        n_range = p[:,-1]
        m_range = torch.arange(p[0,-2]+1)
        
    grid = generate_grid(logmean_cond,logstd_cond,norm)
    s = torch.zeros((len(n_range),npdf))
    s[:,:-1] = torch.diff(grid,axis=1)
    s *= hyp
    s[:,-1] = torch.sqrt(grid[:,-1])
  
    
    v = s**2
    r = grid**2/(v-grid)
    p_nb = 1-grid/v
    
    xgrid = m_range.repeat(len(n_range),1)
    Y = torch.zeros((len(n_range),len(m_range)))
    index = torch.tensor(m_range+1,dtype=torch.long)
    GAMMALN_XGRID = torch.index_select(lnfactorial, 0, index).repeat(len(n_range),1)
    
    for i in range(npdf):
        grid_i = grid[:,i].reshape((-1,1))


        r_i = r[:,i].reshape((-1,1))
        w_i = w[:,i].reshape((-1,1))
        p_nb_i = p_nb[:,i]
        
        l = -grid_i + torch.mul(xgrid,torch.log(grid_i )) - GAMMALN_XGRID

        if (p_nb_i > 1e-10).any():

            index = [p_nb_i > 1e-10]
            l[index] += torch.special.gammaln(xgrid[index]+r_i[index]) - torch.special.gammaln(r_i[index]) \
                - xgrid[index]*torch.log(r_i[index] + grid_i[index]) + grid_i[index] \
                + r_i[index]*torch.log(r_i[index]/(r_i[index]+grid_i[index]))

        Y += torch.mul(w_i,torch.exp(l))
    

    EPS = 1e-40
    Y[Y<EPS]=EPS
    return Y



def get_prob(p_in,n_range,m_range,model=model,rand_weights=False):
    ''' Approximates steady state P(n,m) given input rate parameter p over an array of n values and m values.
    Uses input model.
    If no model given, uses default final model. 
    '''
    
    # calculate overall moments
    pv = 10**p_in
    MU, VAR, STD, xmax_m = cme.get_moments(pv)
    COV = pv[0]**2/(pv[1] + pv[2])
    
    # calculate negative binomial P(n) 
    b = pv[0]
    beta = pv[1]
    n = 1/beta
    p = 1/(b+1)

    prob_n = stats.nbinom.pmf(k=n_range, n=n, p=p)
 
    
    # calculate conditional moments
    logmean_cond,logstd_cond = get_conditional_moments(MU, VAR, COV, n_range)
    
    # now convert to tensors
    mat_range = torch.tensor(m_range,dtype=torch.float)
    nas_range = torch.tensor(n_range,dtype=torch.float)
    p_in_array = torch.tensor(p_in,dtype=torch.float).repeat(len(n_range),1)
    xmax_m = torch.tensor(xmax_m[1],dtype=torch.float).repeat(len(n_range),1)

    # and stack for model
    p_array = torch.column_stack((p_in_array,logmean_cond,logstd_cond,xmax_m,nas_range))
    
    # run through model
    w_,hyp_= model(p_array)
    
    if rand_weights == True:
        w_ = torch.rand(w_.shape)
        w_ = w_/torch.sum(w_)
    
    # get conditional probabilites
    npdf = w_.shape[1]
    norm = get_NORM(npdf)
    ypred_cond = get_ypred_at_RT(p_array,w_,hyp_,nas_range,mat_range,norm)
    
    # multiply conditionals P(m|n) by P(n)
    predicted = prob_n.reshape((-1,1))* ypred_cond.detach().numpy()
    
    
    return(predicted)




def approximate_conditional_tensorval(p,n,m):
    
    p = torch.tensor(10**p)
    MU, VAR, STD, xmax = [torch.tensor(x) for x in train.get_moments(p)]
    
    COV = p[0]**2/(p[1]+p[2])
    n = torch.tensor(n)
    m = torch.tensor(m)
    
    logvar = torch.log((VAR/MU**2)+1)
    logstd = torch.sqrt(logvar)
    logmean = torch.log(MU**2/torch.sqrt(VAR+MU**2))

    logcov = torch.log(COV * torch.exp(-(logmean.sum()+logvar.sum()/2)) +1 ) 
    logcorr = logcov/torch.sqrt(logvar.prod())

    logmean_cond = logmean[1] + logcorr * logstd[1]/logstd[0] * (torch.log(n+1) - logmean[0])
    logvar_cond = logvar[1] * (1-logcorr**2)   

    mean_cond = torch.exp(logmean_cond + logvar_cond/2)
    var_cond = torch.exp(2*logmean_cond + logvar_cond) * (torch.exp(logvar_cond) - 1)

    r = 1/p[1]
    r_cond = mean_cond**2/(var_cond-mean_cond)
    p_cond = mean_cond/var_cond
    prefactor = torch.lgamma(n+r) - torch.lgamma(n+1) - torch.lgamma(r) \
                + r * torch.log(r/(r+MU[0])) + n * torch.log(MU[0]/(r+MU[0]))

    y_ = m * torch.log(mean_cond) - mean_cond - torch.lgamma(m+1) 
    filt = torch.logical_and(torch.logical_and(r>0,p_cond>0), p_cond<1)
    y_[filt] += torch.lgamma(m[filt]+r_cond[filt])  - torch.lgamma(r_cond[filt]) \
                + r_cond[filt] * torch.log(r_cond[filt]/(r_cond[filt]+mean_cond[filt])) \
                - m[filt] * torch.log(r_cond[filt]+mean_cond[filt]) + mean_cond[filt]

    P = prefactor +  y_

    return np.exp(P)

    
