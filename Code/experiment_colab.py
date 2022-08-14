import os
import anndata
import numpy as np
import pandas as pd
import scanpy as sc

import scvi
from sklearn.model_selection import StratifiedKFold

import torch

#### Set directories
logdir_data = 'data'

################################################################################
#                                  Load data
################################################################################

adata_dict = {}

loomfile_list = ['brain_5k', 'brain_nuc_5k',
                 'heart_10k_v3', 'heart_1k_v3',
                 'neuron_10k_v3', 'neuron_1k_v3',
                 'pbmc_10k_v3', 'pbmc_1k_v3',
                 'pbmc_10x_10k_fbc', 'pbmc_10x_1k_fbc',
                 'hgForebrainGlut']

# subset for testing

dataset_name = 'hgForebrainGlut'
# dataset_name = 'pbmc_10x_1k_fbc'
# dataset_name = 'pbmc_10k_v3'
loomfile_list = [dataset_name]

for ii in loomfile_list:
    # loomfile = os.path.join(logdir_data,f'loom_10x_kb/{ii}.loom')
    loomfile = os.path.join(logdir_data,f'{ii}.loom')
    adata = anndata.read_loom(loomfile)
    # adata.var_names = adata.var['gene_name']
    adata.var_names = adata.var.index

    # load protein info if available
    if 'fbc' in ii:

      if 'pbmc_10x_1k_fbc' == ii:
        csvfile = os.path.join(logdir_data,f'loom_10x_kb/pbmc725_4_1k.csv')
      elif 'pbmc_10x_1k_fbc' == ii:
        csvfile = os.path.join(logdir_data,f'loom_10x_kb/pbmc_mat_10k.csv')

      df_protein = pd.read_csv(csvfile,index_col=0).transpose()

      cells_intersect = np.intersect1d(adata.obs_names,df_protein.index)

      adata = adata[cells_intersect]
      df_protein = df_protein[np.isin(df_protein.index,cells_intersect)]

      adata.obsm['protein'] = df_protein

    adata_dict[ii] = adata


#### Create logdir
logdir=f'out/{dataset_name}'
logdir_out = os.path.join(logdir)
os.makedirs(logdir_out, exist_ok=True)

####
gene_dict = {'CD3':['CD3D','CD3E','CD3G'],
             'CD4':['CD4'],
             'CD8':['CD8A','CD8B'],
             'CD2':['CD2'],
             'CD45RA':['PTPRC'],
             'CD57':['B3GAT1'],
             'CD16':['FCGR3A','FCGR3B'],
             'CD14':['CD14'],
             'CD11c':['ITGAX'],
             'CD19':['CD19']}

InvertDict = lambda d: dict( (v,k) for k in d for v in d[k] )

cluster_labels = ['CD4+ T','Mono.','B','NK','CD8+ T']

mrna_targets = ['CD3D','CD3E','CD3G','CD4','CD8A','CD8B','CD2',
                'PTPRC','FCGR3A','CD14','ITGAX','CD19']

################################################################################
#                               Pre-processed data
################################################################################

# Select dataset
adata = adata_dict[dataset_name]
adata.var_names_make_unique()

# Basic filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# QC metrics with mito genes
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
plt.savefig(os.path.join(logdir,'violin_plot.svg'))
plt.close()

sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
plt.savefig(os.path.join(logdir,'scatter_total_vs_pct_mt.svg'))
plt.close()


sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
plt.savefig(os.path.join(logdir,'scatter_total_vs_genes.svg'))
plt.close()

# Filter based off the plots
adata = adata[adata.obs.n_genes_by_counts < 4000, :]
adata = adata[adata.obs.pct_counts_mt < 30, :]

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, min_mean=0.0125, max_mean=3, min_disp=0.5)

adata = adata[:, adata.var.highly_variable]

sc.pp.scale(adata, max_value=10)

# Visualize
mrna_targets = ['CD3D','CD3E','CD3G','CD4','CD8A','CD8B','CD2',
                'PTPRC','FCGR3A','CD14','ITGAX','CD19']
colors = np.array(mrna_targets)[np.isin(mrna_targets,adata.var_names)].tolist()

# PCA
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color=colors)
sc.pl.pca(adata)

# UMAP
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)

sc.pl.umap(adata, color=colors)
plt.savefig(os.path.join(logdir,'umap_colors.svg'))
plt.close()

sc.pl.umap(adata)
plt.savefig(os.path.join(logdir,'umap_default.svg'))
plt.close()

# Get raw spliced and unspliced expressions from layers
adata_spliced   = anndata.AnnData(adata.layers['spliced'])
adata_unspliced = anndata.AnnData(adata.layers['unspliced'])

adata_spliced.var = adata.var.copy()
adata_unspliced.var = adata.var.copy()
adata_spliced.var['Spliced']   = True
adata_unspliced.var['Spliced'] = False
adata_unspliced.var_names = adata_unspliced.var_names + '-u'

adata = anndata.concat([adata_spliced,adata_unspliced],axis=1)
## Change AnnData expression to raw counts for negative binomial distribution
adata.layers["counts"] = adata.X.copy() # preserve counts
# adata.obsm['protein'] = obsm_protein

## get leiden based on
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# sc.pp.scale(adata, max_value=10)
colors = np.array(mrna_targets)[np.isin(mrna_targets,adata.var_names)].tolist()

sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=5, n_pcs=40)
sc.tl.leiden(adata, resolution = 0.1)
sc.tl.umap(adata)
# sc.pl.umap(adata, color=['leiden', 'CST3', 'NKG7'])
sc.pl.umap(adata, color=['leiden']+colors)
adata.obs['Cell Type'] = adata.obs['leiden']
plt.savefig(os.path.join(logdir,'umap_leiden.svg'))
plt.close()

## Cluster based on protein expression

if 'protein' in adata.obsm.keys():

    obsm_protein = adata.obsm['protein']

    adata_protein = anndata.AnnData(obsm_protein)
    sc.pp.normalize_total(adata_protein, target_sum=1e4)
    sc.pp.log1p(adata_protein)
    colors = adata_protein.var_names.tolist()

    sc.tl.pca(adata_protein, svd_solver='arpack')
    sc.pp.neighbors(adata_protein,n_neighbors=5,)
    sc.tl.leiden(adata_protein, resolution = 0.1)
    sc.tl.umap(adata_protein)
    sc.pl.umap(adata_protein, color=['leiden']+colors)
    sc.pl.pca(adata_protein, color=['leiden']+colors)

    adata.obs['protein'] = adata_protein.obs['leiden'].values
    adata.obs['Cell Type'] = adata_protein.obs['leiden'].values
    sc.pl.umap(adata, color=['leiden']+['protein'])
    plt.savefig(os.path.join(logdir,'umap_protein.svg'))
    plt.close()

## Downsample the data
p = 1

if p < 1:
  adata.X = np.random.binomial(adata.X.toarray().astype('int32'),p)

# Set up train/test data
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
_=skf.get_n_splits(adata, adata.obs['Cell Type'])

################################################################################
#                                 Run models
################################################################################

print("{}/{}".format(torch.cuda.memory_allocated(),torch.cuda.max_memory_allocated()))

# Hyper-parameters
lr       = 1e-3
n_latent = 2
n_epochs = 10
n_hidden = 500
n_layers = 3

# setups = ['scBIVI-2-combined',
#           'scBIVI-10-combined',
#           "scBIVI-50-combined",
#           "scVI-10-spliced",
#           'scVI-10-unspliced',
#           'scVI-10-combined']

setups = ['scBIVI-10-combined',
          "scVI-10-spliced",
          'scVI-10-unspliced',
          'scVI-10-combined']


metrics_list = ['clf','clf2','recon_error','latent embedding','compute']
results_dict = {setup:{metrics: [] for metrics in metrics_list} for setup in setups}

for k, (train_index, test_index) in enumerate(skf.split(adata, adata.obs['Cell Type'])):

  for setup in setups:

    method,n_latent,datas = setup.split("-")
    n_latent = int(n_latent)

    ## Split the data
    if datas == 'spliced':
      adata_in = adata[:,:int(adata.shape[1]/2)]
    elif datas == 'unspliced':
      adata_in = adata[:,int(adata.shape[1]/2):]
    elif datas == 'combined':
      adata_in = adata
    else:
      raise ValueError("Input valid datas")

    adata_in = adata_in.copy()
    scvi.data.setup_anndata(adata_in, layer="counts")

    train_adata, test_adata = adata_in[train_index], adata_in[test_index]
    train_adata = train_adata.copy()

    ## Crate and train model
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

    if method == 'LDVAE':
        model = scvi.model.LinearSCVI(train_adata,**model_args)
    elif method == 'scVI':
        model = scvi.model.SCVI(train_adata,**model_args)
    elif method == "scBIVI":
        model = scBIVI(train_adata,**model_args)
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
    results_dict[setup]['compute'].append([runtime,memory_used])

    ## Check train history
    df_history = {'reconstruction_error_test_set' : model.history['reconstruction_error_test_set'],
                  'reconstruction_error_train_set': model.history['reconstruction_error_train_set']}
    df_history = pd.DataFrame(df_history)
    df_history = pd.DataFrame(df_history.stack())
    df = df_history
    df.reset_index(inplace=True)
    df.columns = ['Epoch','Loss Type', 'Loss']
    df.to_csv(os.path.join(logdir_out,'history.csv'))
    figname = f"{setup}-{k}"
    sns.lineplot(data=df,x='Epoch', y='Loss', hue = 'Loss Type')
    plt.savefig(os.path.join(logdir,f"{figname}-train-history.pdf"))
    plt.close()

    ## Get reconstruction loss on test data
    test_error  = model.get_reconstruction_error(test_adata)
    train_error = model.get_reconstruction_error(train_adata)
    results_dict[setup]['recon_error'].append(np.array([train_error,test_error]))

    ## Extract the embedding space for scVI
    X_out = model.get_latent_representation(test_adata)

    if k == 0:
      adata_latent = anndata.AnnData(X_out)
      adata_latent.obs = test_adata.obs
      results_dict[setup]['latent embedding'] = adata_latent
      if datas == 'combined':
        test_adata_save = test_adata

    ## Validation steps
    y     = np.array(test_adata.obs['Cell Type'].tolist())

    score_dict = calculate_accuracy(X_out,y)
    results_dict[setup]['clf'].append(score_dict)

    ## Validation with protein
    if 'protein' in adata_in.obs.columns:
      y = np.array(test_adata.obs['leiden'].tolist())
      score_dict = calculate_accuracy(X_out,y)
      results_dict[setup]['clf2'].append(score_dict)

    ## Correlations

    # cg = plot_corr_comparison(X1,X2)
    # figname = f"{setup}-{k}"
    # plt.title(figname)
    # plt.savefig(os.path.join(logdir,f"{figname}-corr.pdf"))
    # plt.close()

    del model
    torch.cuda.empty_cache()
    gc.collect()

################################################################################
#                                Analysis
################################################################################

## NLL
setups = list(results_dict.keys())
df_plot = pd.concat([pd.DataFrame({"Train": -np.array(r['recon_error'])[:,0],
                                   "Test": -np.array(r['recon_error'])[:,1],
                                   'Setup': key}) for key,r in results_dict.items()])

df_plot['KFold'] = df_plot.index
df_plot.reset_index(drop=True)

df_plot.to_csv(os.path.join(logdir,'.svg'))

fig,ax=plt.subplots()
_ = sns.barplot(data=df_plot, x='Setup', y='Test', hue='Setup', dodge=False, ax=ax)
ax.get_legend().remove()
plt.xticks(rotation=45)
plt.savefig(os.path.join(logdir,'nll.svg'))
plt.close()

print(df_plot.groupby("Setup").mean())

## Clustering accuracy
df_plot = pd.concat([pd.DataFrame(r['clf']).assign(Setup=key) for key,r in results_dict.items()])
print(df_plot.groupby("Setup").mean())
df_plot = df_plot.melt(id_vars=['Setup'],var_name='Metric',value_name='Score')

fig,ax=plt.subplots()
_ = sns.barplot(data=df_plot, x='Metric', y='Score', hue='Setup', ax=ax)
# ax.get_legend().remove()
plt.xticks(rotation=45)
plt.savefig(os.path.join(logdir,'clust_acc.svg'))
plt.close()


if 'fbc' in dataset_name:
    ## Clustering accuracy
    df_plot = pd.concat([pd.DataFrame(r['clf2']).assign(Setup=key) for key,r in results_dict.items()])
    print(df_plot.groupby("Setup").mean())
    df_plot = df_plot.melt(id_vars=['Setup'],var_name='Metric',value_name='Score')
    fig,ax=plt.subplots()
    _ = sns.barplot(data=df_plot, x='Metric', y='Score', hue='Setup', ax=ax)
    # ax.get_legend().remove()
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(logdir,'clust_acc_fbc.svg'))
    plt.close()

################################################################################
# Look at local neighborhoods

from sklearn.metrics import jaccard_score
from scipy.spatial.distance import jaccard
from scipy.spatial import distance_matrix
# from scipy.spatial.distance import cdist

def jaccard_index_split(x):
  """ Given a single vector, split into 2 then measure jaccard similarity """
  len_ = int(len(x)/2)
  a = x[:len_]
  b = x[len_:]
  union     = np.union1d(a,b)
  intersect = np.intersect1d(a,b)
  return len(intersect)/len(union)

def knn_overlap(x1,x2,k=20):
  """  """
  knn1 = (-distance_matrix(x1,x1,1)).argsort(axis=1)[:,:k]
  knn2 = (-distance_matrix(x2,x2,1)).argsort(axis=1)[:,:k]

  scores = np.apply_along_axis(jaccard_index_split, 1, np.concatenate([knn1,knn2],1))
  return scores

embeddings_dict = {s: results_dict[s]['latent embedding'].X for s in setups}
embeddings_dict['ambient_raw']  = test_adata_save.X.toarray()
embeddings_dict['ambient_umap'] = test_adata_save.obsm['X_umap'].toarray()

## K-NN agreement between latent embeddings

setups = embeddings_dict.keys()

scores = [[knn_overlap(embeddings_dict[setup1],
                       embeddings_dict[setup2],
                       k=10).mean() for setup1 in setups] for setup2 in setups]

scores = np.array(scores)
scores = pd.DataFrame(scores,columns=setups,index=setups)

g = sns.heatmap(scores,cmap='Blues',annot=True, fmt="0.2f")
plt.savefig(os.path.join(logdir,'heatmap_knn_overlap.svg'))
plt.close()

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances

## Get intra distance

def cluster_with_labels(X_in, method='knn',**kwargs):
  kmeans = KMeans(random_state=0,**kwargs).fit(X_in)
  return kmeans.labels_.astype(str)

def mean_off_diag(X):
  return X[np.triu_indices(len(X),k=1)].mean()

def getIntraDist(X, y):
  dists = [mean_off_diag(pairwise_distances(X[y==l])) for l in np.unique(y)]
  return dists

dists = [getIntraDist(embeddings_dict[setup],
                      test_adata_save.obs['Cell Type']) for setup in setups]

dists = np.array(dists)
scores = np.corrcoef(dists)
scores = pd.DataFrame(scores,columns=setups,index=setups)

g = sns.heatmap(scores,cmap='Blues',annot=True, fmt="0.2f")
plt.savefig(os.path.join(logdir,'heatmap_intradist.svg'))
plt.close()

################################################################################
# Runtime + memory

setups = list(results_dict.keys())
df_plot = pd.concat([pd.DataFrame(results_dict[s]['compute'],
                                  columns=['Runtime','Memory']).assign(Setup=s) for s in setups])

fig,ax=plt.subplots()
_ = sns.barplot(data=df_plot, x='Setup', y='Runtime', hue='Setup', ax=ax, dodge=False)
# ax.get_legend().remove()
plt.xticks(rotation=45)
plt.savefig(os.path.join(logdir,'barplot_runtime.svg'))
plt.close()

fig,ax=plt.subplots()
_ = sns.barplot(data=df_plot, x='Setup', y='Memory', hue='Setup', ax=ax, dodge=False)
# ax.get_legend().remove()
plt.xticks(rotation=45)
plt.savefig(os.path.join(logdir,'barplot_memory.svg'))
plt.close()
