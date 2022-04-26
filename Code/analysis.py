import numpy as np
import pandas as pd

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from scipy.cluster.hierarchy import linkage
from matplotlib import pyplot as plt

from sklearn.metrics import jaccard_score
from scipy.spatial.distance import jaccard
from scipy.spatial import distance_matrix
# from scipy.spatial.distance import cdist

def calculate_accuracy(X,y,methods=['ARI','NMI','silhouette'], min_count=5):

    scores_dict = {}

    counts = pd.Series(y).value_counts()

    cat_keep = counts.index[counts > min_count]
    idx_keep = np.isin(y, cat_keep)

    X = X[idx_keep]
    y = y[idx_keep]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,
                                                    random_state=1, stratify=y)

    if "ARI" in methods or "NMI" in methods:
        knn = KNeighborsClassifier()
        knn.fit(X_train,y_train)
        y_pred = knn.predict(X_test)

        if "ARI" in methods:
            scores_dict['ARI'] = adjusted_rand_score(y_test,y_pred)

        if 'NMI' in methods:
            scores_dict['NMI'] = normalized_mutual_info_score(y_test,y_pred)

    if 'silhouette' in methods:
        scores_dict['silhouette'] = silhouette_score(X,y)

    return scores_dict


def plot_corr_comparison(X1,X2):
    """ """

    corr1 = np.corrcoef(X1.transpose())
    corr2 = np.corrcoef(X2.transpose())
    corr1[np.triu_indices(len(corr1))] = corr2[np.triu_indices(len(corr2))]

    Z = linkage(corr1, 'ward')

    cg = sns.clustermap(corr1, row_linkage=Z, col_linkage=Z, cmap='RdBu', center=0)
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)

    return cg


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
