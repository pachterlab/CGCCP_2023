
# # Train Models on Various Datasets


# argument parser
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str)
parser.add_argument('--data_dir', type=str, default = '../data/simulated_data/')
args = parser.parse_args()

name = args.name
data_dir = args.data_dir

# install necessary pacakges
# %%capture
# !pip install scanpy -q
# !pip install scvi-tools==0.8.1 -q
# !pip install loompy -q
# !pip install leidenalg -q



# check GPU availability
import torch 
import torch.nn as nn
import torch.nn.functional as F
memory_used = torch.cuda.memory_allocated()
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())




# System
import time, gc

# add module paths to sys path
import sys
sys.path.insert(0,'../custom_distributions/')
sys.path.insert(0, '../BIVAE/')

# Math
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

# to save results
import pickle

# scvi
import anndata
import scvi


# In[ ]:


# import scbivi scripts
import scBIVI
import nnNB_module
import custom_distributions

# set the model to cuda
nnNB_module.model.to(torch.device('cuda'))
NORM = nnNB_module.NORM.to(torch.device('cuda'))

print('TORCH VERSION',torch.__version__)

# # Load in data 
# 
# 
# Change data name to test out different simulated datasets with varying number of celltypes. 

print(name)

# change to hdf5 file if that is what you store data as
adata = anndata.read_loom(data_dir+f'{name}.loom')

if 'gene_name' in adata.var.columns:
    adata.var_names = adata.var['gene_name'].to_list()

# can change as necessary for data. 
adata.obs['Cluster'] = adata.obs['Cell Type']
adata.var_names_make_unique()




#Set up train/test data splits with 5-fold split
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
skf_splits = skf.split(adata, adata.obs['Cluster'])

# Use last of the K-fold splits
for k, (train_index, test_index) in enumerate(skf_splits):
  pass


# In[ ]:


print(f'training on {len(train_index)} cells, testing on {len(test_index)} cells')


# -----
# 
# 
# # Define training function

# In[ ]:


# if anything goes wrong in training, this will catch where it happens
torch.autograd.set_detect_anomaly(True)


# compare setups
def compare_setups(adata, setups, results_dict, hyperparameters, train_index = train_index, test_index = test_index):
  ''' Runs scBIVI on adata for listed setups in setups given hyperparameters, stores outputs in results_dict. 
      Train index and test index are defined globally -- could be nice to pass these in as well? 
  ''' 

  lr = hyperparameters['lr']
  n_epochs = hyperparameters['n_epochs']
  n_hidden = hyperparameters['n_hidden']
  n_layers = hyperparameters['n_layers']

  
  for setup in setups:
    print(setup)
    method,n_latent,constant, = setup.split("-")
    n_latent = int(n_latent)

    # test using only spliced or unspliced in vanilla scVI
    if '.S' in method:
      adata_in = adata[:,adata.var['Spliced']==1].copy()
      print('spliced')
    elif '.U' in method:
      adata_in = adata[:,adata.var['Spliced']==0].copy()
      print('unspliced')
    else:
      adata_in = adata.copy()

    print(adata_in.X.shape)
    scvi.data.setup_anndata(adata_in, layer="counts")

    
    train_adata, test_adata = adata_in[train_index], adata_in[test_index]
    train_adata = train_adata.copy()

    ## Set model parameters
    model_args = {'use_cuda'     : True,
                  'n_latent'     : n_latent,
                  'n_layers'     : n_layers,
                  'dispersion'   : 'gene',
                  'n_hidden'     : n_hidden,
                  'dropout_rate' :  0.1,
                  'gene_likelihood'    :  'nb',
                  'log_variational'    :  True,
                  'latent_distribution':  'normal'
                  }
    #model_args.update(additional_kwargs)

    ## Create model
    if method == 'NBcorr':
        model = scBIVI.scBIVI(train_adata,mode='corr',**model_args)
    elif method == 'NBuncorr':
        model = scBIVI.scBIVI(train_adata,mode='uncorr',**model_args)
    elif method == 'Poisson':
        custom_dist = lambda x,mu1,mu2,theta,eps : custom_distributions.log_prob_poisson(x,mu1,mu2,theta,eps,THETA_IS = constant)
        model = scBIVI.scBIVI(train_adata,mode='custom',custom_dist=custom_dist,**model_args)
    elif method == 'nnNB':
        custom_dist = lambda x,mu1,mu2,theta,eps : nnNB_module.log_prob_nnNB(x,mu1,mu2,theta,eps,THETA_IS = constant,
                                                                             model= nnNB_module.model.to(torch.device('cuda')),
                                                                             norm = NORM)
        model = scBIVI.scBIVI(train_adata,mode='custom',custom_dist=custom_dist,**model_args)
    elif method == 'vanilla.U':
      model = scvi.model.SCVI(train_adata,**model_args)
    elif method == 'vanilla.S':
      model = scvi.model.SCVI(train_adata,**model_args)
    elif method == 'vanilla.full':
      model = scvi.model.SCVI(train_adata,**model_args)
    elif method == 'vanilla.U.P':
      model_args['gene_likelihood'] = 'poisson'
      model = scvi.model.SCVI(train_adata,**model_args)
    elif method == 'vanilla.S.P':
      model_args['gene_likelihood'] = 'poisson'
      model = scvi.model.SCVI(train_adata,**model_args)
    elif method == 'vanilla.full.P':
      model_args['gene_likelihood'] = 'poisson'
      model = scvi.model.SCVI(train_adata,**model_args)
    else:
        raise Exception('Input valid scVI model')

    ## Train model
    start = time.time()
    model.train(n_epochs = n_epochs,
                lr       = lr,
                n_epochs_kl_warmup = n_epochs/2,
                metrics_to_monitor = ['reconstruction_error'],
                frequency = 1,
                train_size = 0.9)

    runtime     = time.time() - start
    memory_used = torch.cuda.memory_allocated()
    results_dict[setup]['runtime'].append(runtime)

    ## Save training history
    df_history = {'reconstruction_error_test_set' : model.history['reconstruction_error_test_set'],
                  'reconstruction_error_train_set': model.history['reconstruction_error_train_set']}
    df_history = pd.DataFrame(df_history)
    df_history = pd.DataFrame(df_history.stack())
    df = df_history
    df.reset_index(inplace=True)
    df.columns = ['Epoch','Loss Type', 'Loss']
    results_dict[setup]['df_history'] = df

    ## Get reconstruction loss on test data
    test_error  = model.get_reconstruction_error(test_adata)
    train_error = model.get_reconstruction_error(train_adata)
    results_dict[setup]['recon_error'].append(np.array([train_error,test_error]))


    results_dict[setup]['params'] = model.get_likelihood_parameters(adata_in)

    ## Extract the embedding space for scVI
    X_out_full = model.get_latent_representation(adata_in)

    adata.obsm[f'X_{method}'] = X_out_full
    results_dict[setup][f'X_{n_latent}'] = X_out_full

    del model
    torch.cuda.empty_cache()
    gc.collect()

  
  return(results_dict,adata)


# # Compare Distributions
# 
# 
# Can change various training hyperparameters.



# Hyper-parameters
hyperparameters = { 'lr'       : 1e-3,
        'n_epochs' : 100, 
        'n_hidden' : 128,
        'n_layers' : 3 }

z  = 10
constant = 'NAS_SHAPE'

setups = [
          f'vanilla.U-{z}-{constant}',
          f'vanilla.S-{z}-{constant}',
          f'vanilla.full-{z}-{constant}',
          f'vanilla.U.P-{z}-{constant}',
          f'vanilla.S.P-{z}-{constant}',
          f'vanilla.full.P-{z}-{constant}',
          f'Poisson-{z}-{constant}',
          f'NBcorr-{z}-{constant}',
          f'nnNB-{z}-{constant}'
          ]

metrics_list = [f'X_{z}','runtime','df_history','params','recon_error']
results_dict = {setup:{metrics: [] for metrics in metrics_list} for setup in setups}


results_dict, adata = compare_setups(adata, setups,results_dict,hyperparameters)
results_dict['Cell Type'] = adata.obs['Cell Type']
results_dict['train_index'] = train_index 
results_dict['test_index'] =  test_index


# # Save results dict
results_file = open(f"../results/{name}_results_dict.pickle", "wb")
pickle.dump(results_dict, results_file)
results_file.close()

print(f'Done with {name}')






