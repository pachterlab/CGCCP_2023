# # Train scBIVI
# 
# This script trains and stores results for different models


# argument parser
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str)
parser.add_argument('--data_dir', type=str, default = '../../data/simulated_data/')
args = parser.parse_args()

name = args.name
data_dir = args.data_dir



# System
import time, gc

# add module paths to sys path
import sys
sys.path.insert(0, '../BIVI/')

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



# import biVI scripts
import biVI


# reproducibility -- set random seeds
scvi._settings.ScviConfig.seed=(8675309)
torch.manual_seed(8675309)
np.random.seed(8675309)
 

# # Load in data 
# 
# 
# Change data name to test out different simulated datasets with varying number of celltypes. 


# ==================================================================================================================

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



print(f'training on {len(train_index)} cells, testing on {len(test_index)} cells')


# ==================================================================================================================

# # Define training function


# if anything goes wrong in training, this will catch where it happens
torch.autograd.set_detect_anomaly(True)


# compare setups
def compare_setups(adata, setups, results_dict, hyperparameters, train_index = train_index, test_index = test_index):
  ''' Runs scBIVI on adata for listed setups in setups given hyperparameters, stores outputs in results_dict. 
      Train index and test index are defined globally -- could be nice to pass these in as well? 
  ''' 

  lr = hyperparameters['lr']
  max_epochs = hyperparameters['max_epochs']
  n_hidden = hyperparameters['n_hidden']
  n_layers = hyperparameters['n_layers']

  
  for setup in setups:
    print(setup, 'with linear decoder')
    method,n_latent,constant, = setup.split("-")
    n_latent = int(n_latent)

    # test using only spliced or unspliced in vanilla scVI
    if '.S' in method:
      adata_in = adata[:,adata.var['Spliced']==1]
      print('spliced')
    elif '.U' in method:
      adata_in = adata[:,adata.var['Spliced']==0]
      print('unspliced')
    else:
      adata_in = adata

    print(adata_in.X.shape)
    #biVI.biVI.setup_anndata(adata_in,layer="counts")
    #categorical_covariate_keys=["cell_source", "donor"],
    #continuous_covariate_keys=["percent_mito", "percent_ribo"])

    
    train_adata, test_adata = adata_in[train_index], adata_in[test_index]
    train_adata = train_adata.copy()
    test_adata = test_adata.copy()
    if 'scVI' in method:
        scvi.model.SCVI.setup_anndata(test_adata,layer="counts")
        scvi.model.SCVI.setup_anndata(train_adata,layer="counts")
    else:
        biVI.biVI.setup_anndata(test_adata,layer="counts")
        biVI.biVI.setup_anndata(train_adata,layer="counts")
    

    ## Set model parameters
    model_args = {
                  'n_latent'     : n_latent,
                  'n_layers'     : n_layers,
                  'dispersion'   : 'gene',
                  'n_hidden'     : n_hidden,
                  'dropout_rate' :  0.1,
                  'gene_likelihood'    :  'nb',
                  'log_variational'    :  True,
                  'latent_distribution':  'normal',
                  }
    #model_args.update(additional_kwargs)

    ## Create model
    if method == 'Extrinsic':
        model = biVI.biVI(train_adata,mode='NBcorr',decoder_type = 'linear',**model_args)
    elif method == 'NBuncorr':
        model = biVI.biVI(train_adata,mode='NBuncorr',**model_args)
    elif method == 'Constitutive':
        model = biVI.biVI(train_adata,mode='Poisson',decoder_type = 'linear',**model_args)
    elif method == 'Bursty':
        model = biVI.biVI(train_adata,mode='Bursty',decoder_type = 'linear',**model_args)
    elif method == 'vanilla.U':
        model_args['gene_likelihood'] = 'nb'
        model = scvi.model.SCVI(train_adata,**model_args)
    elif method == 'vanilla.S':
        model_args['gene_likelihood'] = 'nb'
        model = scvi.model.SCVI(train_adata,**model_args)
    elif method == 'scVI':
        model_args['gene_likelihood'] = 'nb'
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
    plan_kwargs = {'lr' : lr,
                   'n_epochs_kl_warmup' : max_epochs/2,
                   }
    
    start = time.time()
    model.train(max_epochs = max_epochs,
                #early_stopping_monitor = ["reconstruction_loss_validation"],
                train_size = 0.9,
                check_val_every_n_epoch  = 1,
                plan_kwargs = plan_kwargs)

    
    runtime     = time.time() - start
    memory_used = torch.cuda.memory_allocated()
    results_dict[setup]['runtime'].append(runtime)

    ## Save training history
    df_history = {'reconstruction_error_test_set' : [model.history['reconstruction_loss_train']],
                  'reconstruction_error_train_set': [model.history['reconstruction_loss_validation']]}

    
    results_dict[setup]['df_history'] = df_history

    ## Get reconstruction loss on test data
    test_error  = model.get_reconstruction_error(test_adata)
    train_error = model.get_reconstruction_error(train_adata)
    results_dict[setup]['recon_error'].append(np.array([train_error,test_error]))

    # get reconstructed parameters
    results_dict[setup]['params'] = model.get_likelihood_parameters(adata_in)
    results_dict[setup]['norm_params'] = model.get_normalized_expression(adata_in)

    ## Extract the embedding space for scVI
    X_out_full = model.get_latent_representation(adata_in)

    adata.obsm[f'X_{method}'] = X_out_full
    results_dict[setup][f'X_{n_latent}'] = X_out_full
    
    
    # save model for future testing
    
    print('save path',f'../../results/{method}_model_{name}_linear')
    if 'Bursty' in method:
        model.save(f'../../results/{method}_{name}_linear_MODEL',overwrite=True)
        print('model saved')

    del model
    torch.cuda.empty_cache()
    gc.collect()

  
  return(results_dict,adata)


# ==============================================================================================================
# # Compare Distributions

# Can change various training hyperparameters.

print('Training linear models')

# Hyper-parameters
hyperparameters = { 'lr'       : 1e-5,
        'max_epochs' : 400, 
        'n_hidden' : 128,
        'n_layers' : 3 }

z  = 10
constant = 'NAS_SHAPE'

setups = [
#           f'scVI-{z}-{constant}',
          f'Bursty-{z}-{constant}',
          f'Constitutive-{z}-{constant}',
          f'Extrinsic-{z}-{constant}'
          ]

metrics_list = [f'X_{z}','runtime','df_history','params','recon_error','norm_means']
results_dict = {setup:{metrics: [] for metrics in metrics_list} for setup in setups}


results_dict, adata = compare_setups(adata, setups,results_dict,hyperparameters)
results_dict['Cell Type'] = adata.obs['Cell Type']
results_dict['train_index'] = train_index
results_dict['test_index'] = test_index

# # Save results dict

results_file = open(f"../../results/{name}_linear_results_dict.pickle", "wb")
pickle.dump(results_dict, results_file)
results_file.close()

