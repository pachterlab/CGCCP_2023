#!/usr/bin/env python
# coding: utf-8

# # Preprocess Allen Data


# argument parser
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str)
parser.add_argument('--data_dir', type=str, default = '../data/allen/')
args = parser.parse_args()

name = args.name
data_dir = args.data_dir


# system
import os, sys

# numbers
import numpy as np

import pandas as pd

#sc
import anndata
import scanpy as sc

# Plots
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns



# load in raw loom file:
adata = sc.read_loom(data_dir+f'allen_{name}_raw.loom')



# In[348]:


# load in metadata
allen_membership = pd.read_csv(data_dir+'/cluster.membership.csv',skiprows = 1, names=['barcode','cluster_id'])
allen_annot = pd.read_csv(data_dir+'/cluster.annotation.csv')
allen_membership['cell_barcode'] = allen_membership['barcode'].str[:16]
allen_membership['sample'] = allen_membership['barcode'].str[-3:]
allen_membership['cluster_id'] = allen_membership['cluster_id'].astype("category")
allen_annot.set_index('cluster_id',inplace=True)
allen_annot_bc = allen_annot.loc[allen_membership['cluster_id']][['cluster_label','subclass_label','class_label']].set_index(allen_membership.index)
meta = pd.concat((allen_membership,allen_annot_bc),axis=1)

# choose the sample to work on
meta_name = meta[meta['sample'] == name]


# In[349]:


# subset for cells observed in metadata -- remove all others
index = [adata.obs['barcode'][i] in np.array(meta_name['cell_barcode']) for i in range(len(adata))]

adata_A = adata[index,:]


# In[350]:


S = adata_A.layers['spliced'][:]
U = adata_A.layers['unspliced'][:]
S_old = adata.layers['spliced'][:]
U_old = adata.layers['unspliced'][:]


# In[351]:


def knee_plot(S):
    UMI_sorted =  np.sort(np.array(S.sum(1)).flatten())
    x_range =  range(len(UMI_sorted))[::-1]

    plt.scatter(x_range,UMI_sorted,c='k',s=5)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('# UMI')
    plt.ylabel('cell rank')
    plt.hlines(10**4,xmin=0,xmax= len(x_range)+1000,colors='red',linestyles='dashed',label='10^4')
    plt.vlines(10**4,ymin=0,ymax= 10**5,colors='red',linestyles='dashed')
    plt.grid()
    plt.legend()
    plt.title('Cell Rank vs. UMI ')


# In[ ]:


# visualize knee plot, use to filter data
# knee_plot(S_old+U_old)


# In[ ]:


cluster_ids = []
cluster_labels = []
subclass_labels = []
class_labels = []

for i in range(len(adata_A)):
    
    barcode = adata_A.obs['barcode'][i]

    index = np.where(np.array(meta_name['cell_barcode']) == barcode)[0][0]
    cluster_id = meta_name['cluster_id'].to_list()[index]
    cluster_label = meta_name['cluster_label'].to_list()[index]
    subclass_label = meta_name['subclass_label'].to_list()[index]
    class_label = meta_name['class_label'].to_list()[index]
    
    cluster_ids.append(cluster_id)
    cluster_labels.append(cluster_label)
    subclass_labels.append(subclass_label)
    class_labels.append(class_label)


# In[ ]:


adata_A.obs['cluster_id'] = cluster_ids
adata_A.obs['cluster_label'] = cluster_labels
adata_A.obs['subclass_label'] = subclass_labels
adata_A.obs['class_label'] = class_labels
adata_A.obs['Cell Type'] = subclass_labels


# Remove low quality cells
adata_A = adata_A[adata_A.obs['Cell Type'] != 'Low Quality',:]

# Also remove doublets cells
adata_A = adata_A[adata_A.obs['Cell Type'] != 'doublet',:]


# Now, find highly variable genes.
# normalize, log1p, then select highly variable genes :) 

sc.pp.normalize_total(adata_A, target_sum=1e4)
sc.pp.log1p(adata_A)
sc.pp.highly_variable_genes(adata_A, n_top_genes=2000, min_mean=0.0125, max_mean=3, min_disp=0.5)

# Subset to highly variable genes
adata_s = adata_A[:, adata_A.var.highly_variable]


# In[ ]:


adata_old = adata_s
adata_spliced   = anndata.AnnData(adata_A.layers['spliced'])
adata_unspliced = anndata.AnnData(adata_A.layers['unspliced'])

adata_spliced.var = adata_A.var.copy()
adata_unspliced.var = adata_A.var.copy()
adata_spliced.var['Spliced']   = True
adata_unspliced.var['Spliced'] = False
adata_unspliced.var_names = adata_unspliced.var_names + '-u'

adata = anndata.concat([adata_unspliced,adata_spliced],axis=1)
## Change AnnData expression to raw counts for negative binomial distribution
adata.layers["counts"] = adata.X.copy() # preserve counts

# Update obs,var
adata.obs = adata_old.obs.copy()


# In[ ]:


adata.write_loom(f'../data/allen/{name}_processed.loom')


# In[ ]:


adata_hv = adata[:, adata.var.highly_variable]
adata_hv.write_loom(f'../data/allen/{name}_processed_hv.loom')

