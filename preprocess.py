import os

import pandas as pd

## Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

import anndata
import scanpy as sc

import argparse


# ==============================================================================
#                              Parse input arguments
# ==============================================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--logdir', type=str, default='')
parser.add_argument('--loomfile', type=str, default='')

args = parser.parse_args()

logdir   = args.logdir
loomfile = args.loomfile

os.makedirs(logdir,exist_ok=True)

# logdir='out/pbmc_10k_v3/data'
# loomfile='data/loom_10x_kb/pbmc_10k_v3.loom'

logdir='out/scbivi_gRNA/data'
loomfile='data/scbivi_gRNA.loom'

# ==============================================================================
#                                 Load data
# ==============================================================================

adata = anndata.read_loom(loomfile)

if 'gene_name' in adata.var.columns:
    adata.var_names = adata.var['gene_name'].to_list()

adata.var_names_make_unique()

# ==============================================================================
#                                 Pre-processing
# ==============================================================================

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
plt.savefig(os.path.join(logdir,'violin.svg'))
plt.close()

sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
plt.savefig(os.path.join(logdir,'counts_vs_mt.svg'))
plt.close()

sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
plt.savefig(os.path.join(logdir,'totalcounts_vs_genes_by_counts.svg'))
plt.close()

#### Filter based off the plots
adata = adata[adata.obs.n_genes_by_counts < 4000, :]
adata = adata[adata.obs.pct_counts_mt < 30, :]

# Normalize to get highly variable genes
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, min_mean=0.0125, max_mean=3, min_disp=0.5)

# Subset to highly variable genes
adata = adata[:, adata.var.highly_variable]

# Scale for visualization
sc.pp.scale(adata, max_value=10)

#### Visualize
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata)
plt.savefig(os.path.join(logdir,'pca_RNA.svg'))
plt.close()
sc.pl.pca_variance_ratio(adata, log=True)
plt.savefig(os.path.join(logdir,'pca_variance_ratio.svg'))
plt.close()

sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.pl.umap(adata)
plt.savefig(os.path.join(logdir,'umap_RNA.svg'))
plt.close()

#### Create anndata with combined

adata_old = adata
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

# Update obs,var
adata.obs = adata_old.obs.copy()

# ==============================================================================
#                             Get cluster labels
# ==============================================================================

#### get cluster based on leiden
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pp.scale(adata, max_value=10)

## Plot
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata, resolution=0.1)

sc.tl.umap(adata)
sc.pl.umap(adata, color=['leiden'])
plt.savefig(os.path.join(logdir,'umap_leiden.svg'))
plt.close()

sc.tl.tsne(adata)
sc.pl.tsne(adata, color=['leiden'])
plt.savefig(os.path.join(logdir,'tsne_leiden.svg'))
plt.close()


# Set Cell type as leiden cluster
adata.obsm['Cluster'] = pd.DataFrame({'RNA_leiden': adata.obs['leiden']})

#### Get cluster based on protein expression

# Fix to make sure this works
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

    # Plots of protein expressions with RNA leiden clustering
    sc.pl.umap(adata_protein, color=['leiden']+colors)
    plt.savefig(os.path.join(logdir,'umap_protein_leiden.svg'))
    plt.close()

    sc.pl.pca(adata_protein, color=['leiden']+colors)
    plt.savefig(os.path.join(logdir,'pca_protein_leiden.svg'))
    plt.close()

    # Plots of RNA expressions with protein leiden clustering
    adata.obs['protein'] = adata_protein.obs['leiden'].values
    adata.obs['Cell Type'] = adata_protein.obs['leiden'].values
    sc.pl.umap(adata, color=['leiden']+['protein'])
    plt.savefig(os.path.join(logdir,'umap_protein_clutser.svg'))
    plt.close()

    adata.obsm['Cluster']['Protein_leiden'] = adata.obs['protein']


if 'Guide' in adata.obs.keys():
    adata.obsm['Cluster']['Guide'] = adata.obs['Guide']
    # Plot umap
    sc.pl.umap(adata, color=['Guide'])
    plt.tight_layout()
    plt.savefig(os.path.join(logdir,'umap_Guide.svg'))
    plt.close()
    adata.obsm['Cluster']['Guide'] = adata.obs['Guide']
    # Plot PCA
    sc.pl.pca(adata, color=['Guide'])
    plt.tight_layout()
    plt.savefig(os.path.join(logdir,'pca_Guide.svg'))
    plt.close()
    # Plot tsne
    sc.tl.tsne(adata)
    sc.pl.tsne(adata, color=['Guide'])
    plt.savefig(os.path.join(logdir,'tsne_Guide.svg'))
    plt.close()


# ==============================================================================
#                               Write adata
# ==============================================================================

#### Export adata
adata.write(os.path.join(logdir,'preprocessed.h5ad'))
