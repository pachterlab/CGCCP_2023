# System
import os, pathlib, time, gc

# Math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

# scvi
import anndata
import scvi
import torch

# scbivi
from scBIVI import scBIVI
from analysis import calculate_accuracy, \
                     plot_corr_comparison, \
                     jaccard_index_split, \
                     knn_overlap

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--logdir', type=str, default='default',
                    help = 'Directory to store the experiment results')
parser.add_argument('--datadir', type=str, default='',
                    help = 'Directory of h5ad/loom file used for training model')
parser.add_argument('--percent_keep', type=float, default=0.2,
                    help = 'Percentage of reads to keep for downsampling')
parser.add_argument('--cluster_method', type=str, default='RNA_leiden',
                    help = 'Cluster method for labels used in StratifiedKFold split')

args = parser.parse_args()

datadir      = args.datadir
logdir       = args.logdir
percent_keep = args.percent_keep
cluster_method_split = args.cluster_method

if logdir == 'default':
    logdir = pathlib.Path(datadir).parent
    if logdir.name == 'data':
        logdir = os.path.join(logdir.parent,'out')


# ==============================================================================
#                        Load and set up data for training
# ==============================================================================

## Load anndata
dataext = pathlib.Path(datadir).suffix
if dataext == '.h5ad':
    adata = anndata.read_h5ad(datadir)
elif dataext == '.loom':
    adata = anndata.read_loom(datadir)

## Downsample the data
if percent_keep < 1:
    X = adata.layers['counts']
    # Convert to numpy array if not already
    try:
        X = X.toarray()
    except:
        pass
    adata.layers['counts'] = np.random.binomial(X.astype('int32'),percent_keep)

# Set up train/test data splits with 5-fold split
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
skf_splits = skf.split(adata, adata.obsm['Cluster'][cluster_method_split])

# ==============================================================================
#                                Run models
# ==============================================================================

print("{}/{}".format(torch.cuda.memory_allocated(),torch.cuda.max_memory_allocated()))

# Hyper-parameters
lr       = 1e-3
n_latent = 2
n_epochs = 20
n_hidden = 1024
n_layers = 3

setups = ['scBIVI-10-combined',
          'scVI-10-combined',
          "scVI-10-spliced",
          'scVI-10-unspliced']

cluster_methods = adata.obsm['Cluster'].columns.to_list()
metrics_list = ['recon_error','latent embedding','compute'] + cluster_methods
results_dict = {setup:{metrics: [] for metrics in metrics_list} for setup in setups}

logdir_train = os.path.join(logdir,'train')
os.makedirs(logdir_train, exist_ok=True)

for k, (train_index, test_index) in enumerate(skf_splits):

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

    ## Create model
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
    figname = f"{setup}-{k}"
    sns.lineplot(data=df,x='Epoch', y='Loss', hue = 'Loss Type')
    plt.savefig(os.path.join(logdir_train,f"{figname}-train-history.pdf"))
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

    ## Validation with cluster accuracy based on labels
    ## Iterate through ground truth labels based on different approach

    for cluster_method, y in test_adata.obsm['Cluster'].iteritems():

        y = np.array(y.tolist())

        score_dict = calculate_accuracy(X_out,y)
        results_dict[setup][cluster_method].append(score_dict)

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

#### Plot NLL
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

#### Plot clustering accuracy

# iterate through different cluster methods
for cluster_method in cluster_methods:
    df_plot = pd.concat([pd.DataFrame(r[cluster_method]).assign(Setup=key) for key,r in results_dict.items()])
    df_plot.to_csv(os.path.join(logdir,f'clust_acc_{cluster_method}.csv'))
    df_plot = df_plot.melt(id_vars=['Setup'],var_name='Metric',value_name='Score')

    fig,ax=plt.subplots()
    _ = sns.barplot(data=df_plot, x='Metric', y='Score', hue='Setup', ax=ax)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(logdir,f'clust_acc_{cluster_method}.svg'))
    plt.close()

#===============================================================================
# Look at local neighborhoods

embeddings_dict = {s: results_dict[s]['latent embedding'].X for s in setups}
embeddings_dict['ambient_raw']  = test_adata_save.X.toarray()
embeddings_dict['ambient_umap'] = test_adata_save.obsm['X_umap'].toarray()

#===============================================================================

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

#===============================================================================

## Get intra distance

def cluster_with_labels(X_in, method='knn',**kwargs):
  kmeans = KMeans(random_state=0,**kwargs).fit(X_in)
  return kmeans.labels_.astype(str)

def mean_off_diag(X):
  return X[np.triu_indices(len(X),k=1)].mean()

def getIntraDist(X, y):
  dists = [mean_off_diag(pairwise_distances(X[y==l])) for l in np.unique(y)]
  return dists

for cluster_method, y in test_adata_save.obsm['Cluster'].iteritems():

    dists = [getIntraDist(embeddings_dict[setup],y) for setup in setups]

    dists = np.array(dists)
    scores = np.corrcoef(dists)
    scores = pd.DataFrame(scores,columns=setups,index=setups)

    g = sns.heatmap(scores,cmap='Blues',annot=True, fmt="0.2f")
    plt.savefig(os.path.join(logdir,f'heatmap_intradist_{cluster_method}.svg'))
    plt.close()

#===============================================================================
# Runtime + memory

setups = list(results_dict.keys())
df_plot = pd.concat([pd.DataFrame(results_dict[s]['compute'],
                                  columns=['Runtime','Memory']).assign(Setup=s) for s in setups])
df_plot.to_csv(os.path.join(logdir,'comp_performance.csv'))

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
