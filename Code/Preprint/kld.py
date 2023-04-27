# argument parser
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str)
parser.add_argument('--data_dir', type=str, default = '../../data/simulated_data/')
args = parser.parse_args()

name = args.name
data_dir = args.data_dir


# system
import sys
sys.path.insert(0, '../BIVI/')
sys.path.insert(0, '../analysis_scripts/')

# math
import numpy as np
import torch

# data management
import pandas as pd
import pickle

# Plots
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
plt.rcParams['image.cmap'] = 'Purples'
# colors
import matplotlib.colors as mcolors
sns.color_palette("Purples", as_cmap=True)


# sc
import anndata
import scanpy as sc

# my modules
from distributions import log_prob_NBuncorr, log_prob_poisson, log_prob_NBcorr
from nnNB_module import log_prob_nnNB
# import differential_expression as de

device = 'cuda'

if 'bursty' in name:
    working_setup = 'Bursty-10-NAS_SHAPE'
if 'const' in name:
    working_setup = 'Constitutive-10-NAS_SHAPE'
if 'extrinsic' in name:
    working_setup = 'Extrinsic-10-NAS_SHAPE'
# load in data

# open a file, where you stored the pickled data
results_file = open(f'../../results/{name}_results_dict.pickle', 'rb')
results_dict = pickle.load(results_file)
results_file.close()


data = 'simulated'

if data == 'simulated':
    simulated_params = np.load(f'../../data/simulated_data/{name}_params.npy')
else:
    simulated_params = None



# read in adata
adata = anndata.read_loom(data_dir+f'{name}.loom')
adata.var_names_make_unique()

if 'gene_name' in adata.var.columns:
    adata.var_names = adata.var['gene_name'].to_list()
    

train_index = results_dict.pop('train_index')
test_index = results_dict.pop('test_index')

index_ = test_index

cell_types = results_dict.pop('Cell Type')[index_]   

# anndata objects for spliced and unspliced counts
adata_s = adata[:,adata.var['Spliced']==1].copy()[index_]    
adata_u = adata[:,adata.var['Spliced']==0].copy()[index_]    



unique_ct = np.unique(np.array(adata.obs['Cell Type'].tolist()))


for setup in results_dict.keys():

    results_dict[setup]['params']['mean'] = results_dict[setup]['params']['mean'][index_]
    results_dict[setup]['params']['dispersions'] = results_dict[setup]['params']['dispersions'][index_]
    
    




def reconstruct_ct_distribution(gene,cell_type,setups,max_vals=[10,10]):
    ''' Calculates the reconstructed probabilities for input gene for 
     all cells in cell type for each setup in setups. Averages probabilities over cells.
     Returns prob_dict with averaged probabilities for the specified gene in the specified cell type for 
     the input setups. 
     max_vals delimits the x,y values to calculate probabilities over.
    '''
    
    x,y = max_vals[0],max_vals[1]
    
    ct_number = int(cell_type[2:])

    x_grid, y_grid = torch.meshgrid(torch.range(0,x), torch.range(0,y))
    X = torch.concat((x_grid,y_grid),axis=1)

    prob_dict = {}

    for setup in setups:

        if 'scVI' in setup:
        # vanilla: 
            N = int(len(results_dict[setup]['params']['mean'][0,:])/2)
            mus_u = results_dict[setup]['params']['mean'][cell_types == cell_type,gene]
            thetas_u = results_dict[setup]['params']['dispersions'][cell_types == cell_type,gene]
            mus_s =  results_dict[setup]['params']['mean'][cell_types == cell_type,gene+N]
            thetas_s = results_dict[setup]['params']['dispersions'][cell_types == cell_type,gene+N]

            mu1 = [torch.ones((x+1,y+1))*mus_u[i] for i in range(len(mus_u))]
            mu2 = [ torch.ones((x+1,y+1))*mus_s[i] for i in range(len(mus_s))]
            theta1 = [torch.ones((x+1,y+1))*thetas_u[i] for i in range(len(thetas_u))]
            theta2 = [torch.ones((x+1,y+1))*thetas_s[i] for i in range(len(thetas_s))]
            theta = [torch.concat((theta1[i],theta2[i]),axis=1) for i in range(len(theta1))]
    
        elif '-' in setup:
            mus_u = results_dict[setup]['params']['mean'][cell_types == cell_type,gene]
            mus_s = results_dict[setup]['params']['mean'][cell_types == cell_type,gene+2000]
            thetas = results_dict[setup]['params']['dispersions'][cell_types == cell_type,gene]
        
            mu1 = [torch.ones((x+1,y+1))*mus_u[i] for i in range(len(mus_u))]
            mu2 = [ torch.ones((x+1,y+1))*mus_s[i] for i in range(len(mus_u))]
            
            theta = [torch.ones((x+1,y+1))*thetas[i] for i in range(len(thetas))]
            

        if "scVI" in setup:
            prob = np.array([torch.exp(log_prob_NBuncorr(X, mu1[i], mu2[i], theta[i], eps = 1e-8, THETA_IS = 'NAS_SHAPE')).numpy()
                    for i in range(len(mu1))])


            prob_dict[setup] = np.sum(prob,axis=0)
        
        elif "Constitutive" in setup:
            prob = np.array([torch.exp(log_prob_poisson(X, mu1[i], mu2[i], theta[i], eps = 1e-8, THETA_IS = 'NAS_SHAPE')).numpy()
                    for i in range(len(mu1))])
            prob_dict[setup] = np.sum(prob,axis=0)
        
        elif "Bursty" in setup:
 
            
            prob = np.array([torch.exp(log_prob_nnNB(X.to(torch.device(device)), 
                                   mu1[i].to(torch.device(device)), 
                                   mu2[i].to(torch.device(device)), 
                                   theta[i].to(torch.device(device)), 
                                   eps = 1e-8, THETA_IS = 'NAS_SHAPE')).detach().cpu().numpy()
                                    for i in range(len(mu1))])
            

            prob_dict[setup] = np.sum(prob,axis=0)
            
    
        elif "Extrinsic" in setup:
            prob = np.array([torch.exp(log_prob_NBcorr(X, mu1[i], mu2[i], theta[i], eps = 1e-8, THETA_IS = 'NAS_SHAPE')).numpy()
                    for i in range(len(mu1))])
            prob_dict[setup] = np.sum(prob,axis=0)
        
        
        if ('TRUE' in setup) and ('bursty' in name):
   

            b, beta, gamma  = 10**simulated_params[ct_number,gene,:]
            av_mu_u,av_mu_s = b/beta, b/gamma
            av_theta = 1/beta
            mu1,mu2 = torch.ones((x+1,y+1))*av_mu_u, torch.ones((x+1,y+1))*av_mu_s
            theta = torch.ones((x+1,y+1))*av_theta 
            prob = torch.exp(log_prob_nnNB(X.to(torch.device(device)), 
                                   mu1.to(torch.device(device)), 
                                   mu2.to(torch.device(device)), 
                                   theta.to(torch.device(device)), 
                                   eps = 1e-8, THETA_IS = 'NAS_SHAPE')).detach().cpu().numpy()
                                    

            prob_dict[setup] = prob
            
        if ('TRUE' in setup) and ('const' in name):
            beta, gamma  = 10**simulated_params[ct_number,gene,:]
            av_mu_u,av_mu_s = 1/beta, 1/gamma
            mu1,mu2 = torch.ones((x+1,y+1))*av_mu_u, torch.ones((x+1,y+1))*av_mu_s
            # dummy theta, just to place hold in function, value means nothing
            theta = torch.ones((x+1,y+1))
            prob = torch.exp(log_prob_poisson(X, mu1, mu2, theta, eps = 1e-8, THETA_IS = 'NAS_SHAPE')).detach().cpu().numpy()
    
        
            prob_dict[setup] = prob
            
        if ('TRUE' in setup) and ('extrinsic' in name):
            alpha = simulated_params[ct_number,gene,0]
            beta, gamma  = 10**simulated_params[ct_number,gene,1:]
            av_mu_u,av_mu_s = alpha/beta, alpha/gamma
            av_theta = alpha
            mu1,mu2 = torch.ones((x+1,y+1))*av_mu_u, torch.ones((x+1,y+1))*av_mu_s
            theta = torch.ones((x+1,y+1))*av_theta
            prob = torch.exp(log_prob_NBcorr(X, mu1, mu2, theta, eps = 1e-8, THETA_IS = 'NAS_SHAPE')).detach().cpu().numpy()
            
        
            prob_dict[setup] = prob
            
    return(prob_dict)



def get_KLD(P,Q):
    ''' Calculate KL divergence between P and Q, where Q is ground truth and P is inferred distribution. 
    '''
    
    kld = np.sum( P.flatten()*np.log(P.flatten()/Q.flatten()))
    
    return(kld)
    
    
    
kld_dict = {working_setup : [], 'scVI-10-NAS_SHAPE' : [] }

eps = 10**-40
# cell types
for i,ct in enumerate(unique_ct):
    print('starting cell type:',ct)
    # genes
    for g in range(2000):

        prob_dict = reconstruct_ct_distribution(gene = g, cell_type = ct, 
                                         setups = ['TRUE',working_setup,'scVI-10-NAS_SHAPE'],
                                         max_vals=[50,50])


        P_biVI = prob_dict[working_setup]
        P_biVI = P_biVI/np.sum(P_biVI)
        P_biVI[P_biVI < eps] = eps
        P_scVI = prob_dict['scVI-10-NAS_SHAPE']
        P_scVI = P_scVI/np.sum(P_scVI)
        P_scVI[P_scVI < eps] = eps
        P_true = prob_dict['TRUE']
        P_true = P_true/np.sum(P_true)
        P_true[P_true < eps] = eps
        
        
        kld_dict[working_setup].append(get_KLD(P_biVI,P_true))
        kld_dict['scVI-10-NAS_SHAPE'].append(get_KLD(P_scVI,P_true))
        


# save dictionary to pickle file
with open(f'../../results/{name}_figs/kld_dict.pickle', 'wb') as file:
    pickle.dump(kld_dict, file, protocol=pickle.HIGHEST_PROTOCOL)