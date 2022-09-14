#!/usr/bin/env python
# coding: utf-8

# # Nearest Neighbor calculation
# 
# Calculate nearest neighbor cluster metric

# In[1]:


import numpy as np


# In[56]:


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


def get_nearest_neighbors_percentages(z_array, cluster_memberships, top = 50, distance = 'l2'):
  ''' Calculate the percent of nearest neighbors in the same cell type or cluster.

  Parameters
  ----------
  z_array : numpy array of z_vectors, shape (num_cells, latent_dim)
  cluster_memberships : list or array listing cluster or cell type memberships for cells in z_array, length (num_cells)
  top : how many nearest neighbors to calculate percentage for
  distance : what distance metric to use to define nearest neighbors, options ['l1','l2','squared_dist']

  Returns
  ----------
  percentages : array of percentages of nearest neighbors in same celltype/cluster for each cell, length (num_cells)
  '''


  # set up array to store percentages
  percentages = np.zeros(len(z_array))

  if distance == 'l2':
    dist_func = l2
  if distance == 'l1':
    dist_func = l1
  if distance == 'squared_dist':
    dist_func = squared_dist
  

  for i,z_i in enumerate(z_array):
    z_i = z_array[i,:]

    z_i_array = np.repeat(z_i.reshape(1,-1), len(z_array), axis = 0)

    dist_array = dist_func(z_i_array,z_array)

    # will give indices of top nearest neighbors for z_i -- note, will include z_i itself so add 1
    idx = np.argpartition(dist_array, (top+1) )[:(top+1)]
  
    clusters = np.take(cluster_memberships,idx)

    cluster_z_i = cluster_memberships[i]

    same_cluster = clusters[clusters == cluster_z_i]

    percent_same = ( len(same_cluster) - 1 ) / (top)  # make sure to remove the cluster for z_i itself

    percentages[i] = percent_same

  
  return(percentages)

