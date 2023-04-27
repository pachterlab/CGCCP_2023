#!/usr/bin/env python
# coding: utf-8

# # Nearest Neighbor calculation
# 
# Calculate nearest neighbor cluster metric

# In[1]:


import numpy as np
import torch
import pandas as pd

# nearest neighbor classifiers and pearson correlation calculators
from sklearn.neighbors import KNeighborsClassifier 
from scipy import stats #function: stats.pearsonr(x,y)


# In[56]:
def calc_MSE_1D(x,y):
  '''Calculate the MSE between x and y.
  params
  -------
  x : (Z)
  y : (Z)
  
  returns
  -------
  MSE : 1'''

  MSE_ = (x-y)**2
  MSE = np.sum(MSE_)/len(x)
  return(MSE)

def l1(z1,z2):
  ''' Calculates l1 distance between two vectors: absolute value of difference between components. 
  '''
  abs_dist = np.absolute(z1-z2)
  return( np.sum(abs_dist,axis=1) )

def l2(z1,z2):
  ''' Calculates l2 distance between two vectors: square root of difference squared. 
  '''

  dist = np.sqrt( np.sum((z1-z2)**2,axis=1) )

  return( dist )


def squared_dist(z1,z2):
  ''' Calculates l2 distance between two vectors: square root of difference squared. 
  '''

  dist = np.sum((z1-z2)**2,axis=1)

  return( dist )


# def get_nearest_neighbors_percentages(z_array, cluster_memberships, top = 50, distance = 'l2'):
#   ''' Calculate the percent of nearest neighbors in the same cell type or cluster.

#   Parameters
#   ----------
#   z_array : numpy array of z_vectors, shape (num_cells, latent_dim)
#   cluster_memberships : list or array listing cluster or cell type memberships for cells in z_array, length (num_cells)
#   top : how many nearest neighbors to calculate percentage for
#   distance : what distance metric to use to define nearest neighbors, options ['l1','l2','squared_dist']

#   Returns
#   ----------
#   percentages : array of percentages of nearest neighbors in same celltype/cluster for each cell, length (num_cells)
#   '''


#   # set up array to store percentages
#   percentages = np.zeros(len(z_array))

#   if distance == 'l2':
#     dist_func = l2
#   if distance == 'l1':
#     dist_func = l1
#   if distance == 'squared_dist':
#     dist_func = squared_dist
  

#   for i,z_i in enumerate(z_array):
#     z_i = z_array[i,:]

#     z_i_array = np.repeat(z_i.reshape(1,-1), len(z_array), axis = 0)

#     dist_array = dist_func(z_i_array,z_array)

#     # will give indices of top nearest neighbors for z_i -- note, will include z_i itself so add 1
#     idx = np.argpartition(dist_array, (top+1) )[:(top+1)]
  
#     clusters = np.take(cluster_memberships,idx)

#     cluster_z_i = cluster_memberships[i]

#     same_cluster = clusters[clusters == cluster_z_i]

#     percent_same = ( len(same_cluster) - 1 ) / (top)  # make sure to remove the cluster for z_i itself

#     percentages[i] = percent_same

  
#   return(percentages)

def nn_percentages(x,cluster_assignments):
  ''' Calculate the percentage of nearest neighbors in the same cluster.

  params
  ------
  x : (N,Z) N cells, Z latent space
  cluster_assignments : cluster assignments for vectors in x

  returns
  -------
  nn_percent_array = (N) percent of N nearest neighbors in same cluster for each vector of x
  '''
  
  cluster_assignments = np.array(cluster_assignments)
  unique_clusters = np.unique(cluster_assignments)
  x_done = 0
    
  nn_percent_array = np.ones(x.shape[0])
  
  for cluster in unique_clusters:
        cluster_assignments_ = cluster_assignments[cluster_assignments == cluster]
        x_ = x[cluster_assignments == cluster]
        
        # how many neighbors were in this unique cluster
        N_ = len(cluster_assignments_)
        
        # set up nearest neighbor class
        neigh = KNeighborsClassifier(n_neighbors=N_)
        
        # fit model
        neigh.fit(x,cluster_assignments)
        
        # calculate nearest neighbor distance and indices to top N_ neighbors for all vectors in x
        # returns array neigh_ind of shape x by N_
        neigh_ind = neigh.kneighbors(x_,return_distance = False)
        
        nn_percent_cluster_ =  np.array([len(cluster_assignments[n][cluster_assignments[n] 
                                                        == cluster_assignments_[i]])-1 for i,n in enumerate(neigh_ind)])/(N_-1) 
        
        nn_percent_array[x_done : x_done + N_] = nn_percent_cluster_
        x_done = x_done + N_

  return nn_percent_array


def get_metrics(name,results_dict,simulated_params,cluster_assignments,adata):
  ''' Given results_dict from model training, returns MSE between simulated/recon means, Pearson correlation between simulated/recon means,
  and percentage of N nearest neighbors in the same cluster assignment for all cells. 


  params
  ------
  name: name of data
  simulated params: IF you pass simulated params, will calculate MSE and Pearson R between simulated means 
        and reconstructed means \
        rather than observed counts and reconstructed means
  results_dict:  containing keys for each setup:
    ['X_{z}','runtime','df_history','params','recon_error','cell_type']


  outputs
  -------
  metric_dict containing keys:
    ['MSE','MSE',Pearson_R',Pearson_R','nearest_neighbors']
  '''

  # set up dictionary to store things in with the training setups as keys
    
    
  setups = list(results_dict.keys())
  metric_dict = { setup : {} for setup in setups}
  z = list(results_dict[setups[0]].keys())[0][2:]
  print(z)


  # get observed means and dispersions
  obs_means = adata[:,:].layers['counts'].toarray()

  for setup in setups:
    print(setup)

    setup_dict = results_dict[setup]

    setup_metric_dict = {}

    # unpack dictionary
    X_z = setup_dict[f'X_{z}']
    recon_means = setup_dict['params']['mean']
    print(recon_means.shape)
    
    if simulated_params is not None:
        if 'const' in name:
            obs_means_U = 10**simulated_params[:,:,0]   
            obs_means_S = 10**simulated_params[:,:,1]        
            obs_means = np.concatenate((obs_means_U,obs_means_S),axis=1)    
        if 'bursty' in name:
            params = 10**simulated_params
            b,beta,gamma = params[:,:,0],params[:,:,1],params[:,:,2]
            obs_means_U = b/beta       
            obs_means_S = b/gamma        
            obs_means = np.concatenate((obs_means_U,obs_means_S),axis=1)
        if 'BVNB' in name:
            alpha = simulated_params[:,:,0]
            beta = 10**simulated_params[:,:,1]
            gamma = 10**simulated_params[:,:,2]
            obs_means_U =  alpha/beta     
            obs_means_S = alpha/gamma     
            obs_means = np.concatenate((obs_means_U,obs_means_S),axis=1)
    
    if simulated_params is None:
      
      setup_metric_dict['MSE'] = np.array([ calc_MSE_1D(recon_means[i],obs_means[i]) for i in range(len(X_z)) ])
      setup_metric_dict['Pearson_R'] = np.array([ stats.pearsonr(recon_means[i], obs_means[i])[0] for i in range(len(X_z)) ])

    elif simulated_params is not None:
      setup_metric_dict['MSE'] = np.array([ calc_MSE_1D(recon_means[i],obs_means[cluster_assignments[i]]) for i in range(len(X_z)) ])
      setup_metric_dict['Pearson_R'] = np.array([ stats.pearsonr(recon_means[i], obs_means[cluster_assignments[i]])[0] for i in range(len(X_z)) ])

    setup_metric_dict['nearest_neighbors'] = nn_percentages(X_z,cluster_assignments)

    metric_dict[setup] = setup_metric_dict

  return(metric_dict)




def get_metrics_old(name,results_dict,adata,index,N=100):
  ''' Given results_dict from model training, returns MSE between simulated/recon means, Pearson correlation between simulated/recon means,
  and percentage of N nearest neighbors in the same cluster assignment for all cells. 


  params
  ------
  results_dict containing keys:
    ['X_{z}','runtime','df_history','params','recon_error','cell_type']

  outputs
  -------
  metric_dict containing keys:
    ['MSE_S','MSE_U',Pearson_R_S',Pearson_R_U','nearest_neighbors']
  '''

  # set up dictionary to store things in with the training setups as keys
    
  setups = list(results_dict.keys())
  metric_dict = { setup : {} for setup in setups}
  z = list(results_dict[setups[0]].keys())[0][2:]
  print(z)


  # get observed means and dispersions
  obs_means_U = adata[:,adata.var['Spliced']==0].layers['counts'].toarray()
  obs_means_S = adata[:,adata.var['Spliced']==1].layers['counts'].toarray()
  obs_means = adata[:,:].layers['counts'].toarray()


  for setup in setups:
    print(setup)

    setup_dict = results_dict[setup]

    setup_metric_dict = {}

    # unpack dictionary
    X_z = setup_dict[f'X_{z}']
    
    if '.U' in setup:
      recon_means_U = setup_dict['params']['mean'][:,:]
      setup_metric_dict['MSE_U'] = np.array([ calc_MSE_1D(recon_means_U[i],obs_means_U[i]) for i in range(len(X_z)) ])
      setup_metric_dict['Pearson_R_U'] = np.array([ stats.pearsonr(recon_means_U[i], obs_means_U[i])[0] for i in range(len(X_z)) ])

    elif '.S' in setup:
      recon_means_S = setup_dict['params']['mean'][:,:]
      setup_metric_dict['MSE_S'] = np.array([ calc_MSE_1D(recon_means_S[i],obs_means_S[i]) for i in range(len(X_z)) ])
      setup_metric_dict['Pearson_R_S'] = np.array([ stats.pearsonr(recon_means_S[i], obs_means_S[i])[0] for i in range(len(X_z)) ])
    
    else:
      recon_means_U = setup_dict['params']['mean'][:,:int(setup_dict['params']['mean'].shape[1]/2)]
      recon_means_S = setup_dict['params']['mean'][:,int(setup_dict['params']['mean'].shape[1]/2):]
      setup_metric_dict['MSE_U'] = np.array([ calc_MSE_1D(recon_means_U[i], obs_means_U[i]) for i in range(len(X_z)) ])
      setup_metric_dict['Pearson_R_U'] = np.array([ stats.pearsonr(recon_means_U[i], obs_means_U[i])[0] for i in range(len(X_z)) ])
      setup_metric_dict['MSE_S'] = np.array([ calc_MSE_1D(recon_means_S[i], obs_means_S[i]) for i in range(len(X_z)) ])
      setup_metric_dict['Pearson_R_S'] = np.array([ stats.pearsonr(recon_means_S[i], obs_means_S[i])[0] for i in range(len(X_z)) ])


#     if (('.P' not in setup) and ('const' not in name)):      
#       recon_disp = setup_dict['params']['dispersions']
#       setup_metric_dict['alpha correlation'] = stats.pearsonr(sim_disp[0],recon_disp[0,:2000])[0]
    setup_metric_dict['nearest_neighbors'] = nn_percentages(X_z,N,cluster_assignments)

    metric_dict[setup] = setup_metric_dict

  return(metric_dict)

