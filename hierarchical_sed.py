import sys
import types

from evaluate import evaluate_clustering

fake_module = types.ModuleType("distutils.msvccompiler")
fake_module.get_build_version = lambda: None
sys.modules["distutils.msvccompiler"] = fake_module

import numpy as np
import random
from matplotlib import pyplot as pl
from matplotlib import rc
from seq_kmeans import SeqKmeans, SoftSeqKmeans
from chainer_edit_distance import edit_distance
from sklearn.manifold import TSNE
import cupy
from itertools import cycle, islice
import time

font = {'family': 'DejaVu Sans', 
        'weight': 'normal'}
rc('font', **font)


def generate_data(motif, alphabet, n, p=0.2):
    out = []
    L = np.random.randint(len(motif) - 3, len(motif) + 3, n)
    for i in range(n):
        new = ''
        k = 0
        while True:
            r = random.random()
            if r < 1 - p:
                if k < len(motif):
                    new += motif[k]
                    k += 1
                else:
                    break
            else:
                if r < (1 - p) + 0.5 * p:
                    if k < len(motif):
                        k += 1
                    else:
                        break
                else:
                    new += alphabet[random.randint(0, len(alphabet) - 1)]
        out.append(new)
    return out


def vis(X, labels, centroids, alphabet, subsample_size=1000):
    alphabet = {alphabet[i]: i + 1 for i in range(len(alphabet))}
    if subsample_size is not None and len(X) > subsample_size:
        ind = np.random.choice(len(X), subsample_size, replace=False)
        X = X[ind]
        labels = labels[ind]
    max_length = np.max([len(seq) for seq in X] + [len(seq) for seq in centroids])
    encoded = np.zeros((len(X) + len(centroids), max_length), dtype=np.uint8)
    for i, x in enumerate(X):
        for j, c in enumerate(x):
            encoded[i, j] = alphabet[c]

    for i, x in enumerate(centroids):
        for j, c in enumerate(x):
            encoded[i + len(X), j] = alphabet[c]
    encoded = cupy.array(encoded)
    I = np.broadcast_to(np.arange(len(encoded)), (len(encoded), len(encoded)))
    J = np.ravel(I.T)
    I = np.ravel(I)

    dist = edit_distance(encoded[I], encoded[J])
    dist = cupy.asnumpy(dist.reshape((len(encoded), len(encoded))))

    tsne = TSNE(metric='precomputed', max_iter=10000, perplexity=100, init='random')
    points = tsne.fit_transform(dist)
    labels = np.concatenate((labels, np.full(len(centroid), len(centroid), np.int32)), axis=0)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(len(centroid) + 1))))
    colors = colors[labels]
    # print(labels)
    indexes = np.random.choice(len(X), 200, replace=False)
    pl.figure()
    pl.scatter(points[indexes, 0], points[indexes, 1], c=colors[indexes], s=5, alpha=0.8)
    for i in indexes:
        pl.text(points[i, 0], points[i, 1], X[i], color=colors[i], horizontalalignment='center',
                verticalalignment='bottom', fontsize=7, alpha=0.8)
    pl.scatter(points[len(X):, 0], points[len(X):, 1], c='black', s=20)
    for i in range(len(centroids)):
        pl.text(points[len(X) + i, 0], points[len(X) + i, 1], centroids[i], color='black',
                horizontalalignment='center', verticalalignment='bottom', fontsize=11)
    pl.savefig('images/simulated.png', dpi=600)
    pl.show()


class SimpleCluster:
    def __init__(self, data_indices=None, centroid=""):
        self.data_indices = data_indices if data_indices is not None else []
        self.centroid = centroid


class HierarchicalSoftSeqKmeans:
    def __init__(self, hierarchy, centroid_length, alphabet, n_iter=100):
        """
        hierarchy: list of k values for each level, e.g., [100, 20, 10, 5]
        centroid_length: length of centroids
        alphabet: alphabet for sequences
        n_iter: number of iterations for each k-means run
        """
        if not hierarchy:
            raise ValueError("Hierarchy cannot be empty")
        
        self.hierarchy = hierarchy
        self.centroid_length = centroid_length
        self.alphabet = alphabet
        self.n_iter = n_iter
        self.data = None
        self.final_clusters = []
        
    def fit(self, data):
        """Fit hierarchical clustering on data"""
        self.data = np.unique(data)  # Remove duplicates like in original
        
        print(f"Starting hierarchical clustering with hierarchy: {self.hierarchy}")
        print(f"Data size: {len(self.data)}")
        
        # Initialize with all data as one cluster
        current_clusters = [SimpleCluster(data_indices=list(range(len(self.data))))]
        
        # Process each level in the hierarchy
        for level, k in enumerate(self.hierarchy):
            print(f"Processing level {level} with k={k} on {len(current_clusters)} clusters")
            
            next_clusters = []
            
            # Process each cluster from current level
            for cluster in current_clusters:
                if not cluster.data_indices:
                    continue
                
                # Extract data for this cluster
                cluster_data = self.data[cluster.data_indices]
                
                # Skip if not enough data points
                if len(cluster_data) <= 1:
                    # Keep as single cluster with the data point as centroid
                    next_clusters.append(SimpleCluster(
                        data_indices=cluster.data_indices,
                        centroid=cluster_data[0] if len(cluster_data) > 0 else ""
                    ))
                    continue
                
                # Run SoftSeqKmeans on this cluster
                effective_k = min(k, len(cluster_data))
                seq_kmeans = SoftSeqKmeans(effective_k, self.centroid_length, self.alphabet)
                seq_kmeans.fit(cluster_data, n_iter=self.n_iter)
                
                # Get labels and centroids
                labels = seq_kmeans.transform(cluster_data)
                centroid_raw = seq_kmeans.get_centroid()
                centroids = self.alphabet[np.argmax(centroid_raw, axis=1)]
                centroids = [''.join(seq) for seq in centroids]
                
                # Create subclusters
                subclusters = [SimpleCluster(centroid=cent) for cent in centroids]
                
                # Assign data points to subclusters
                for i, label in enumerate(labels):
                    original_idx = cluster.data_indices[i]
                    subclusters[label].data_indices.append(original_idx)
                
                # Add non-empty subclusters to next level
                for subcluster in subclusters:
                    if subcluster.data_indices:
                        next_clusters.append(subcluster)
            
            # Update current clusters for next iteration
            current_clusters = next_clusters
        
        # Store final result
        self.final_clusters = current_clusters
        
        print(f"Hierarchical clustering completed with {len(self.final_clusters)} final clusters!")
        
    def get_labels_and_centroids(self):
        """Get final labels and centroids"""
        if not self.final_clusters:
            return np.array([]), []
        
        # Create labels array
        labels = np.full(len(self.data), -1, dtype=int)
        centroids = []
        
        for cluster_id, cluster in enumerate(self.final_clusters):
            # Assign cluster label to all data points in this cluster
            for data_idx in cluster.data_indices:
                labels[data_idx] = cluster_id
            centroids.append(cluster.centroid)
        
        return labels, centroids
    
    def save_clusters_to_file(self, filename):
        """Save clusters to file"""
        with open(filename, 'w') as f:
            print(f"Saving {len(self.final_clusters)} clusters to file.")
            
            for cluster in self.final_clusters:
                # Write centroid
                f.write(f"{cluster.centroid}\n")
                f.write("*************\n")
                
                # Write all data points in this cluster
                for data_idx in cluster.data_indices:
                    f.write(f"{self.data[data_idx]}\n")
                f.write("\n")


def test_hierarchical_softseqkmeans():
    alphabet = np.array(['T', 'A', 'G', 'C'])
    test_type = 'real_big'  # 'real_small' or 'real_big' 'simulated'

    if test_type == 'simulated':
        motifs = ['TAGCGA', 'ATGCAT', 'CCTTGA']
        seq_per_motif = 3000
        data = np.concatenate([generate_data(m, alphabet, seq_per_motif) for m in motifs], axis=0)
        centroid_length = 6
        hierarchy = [3]  

    elif test_type == 'real_small':
        # load data from indices_55000.txt file. each line is a sequence
        with open('indices_55000.txt', 'r') as f:
            data = [line.strip() for line in f.readlines()]
        data = np.array(data)
        centroid_length = 12
        hierarchy = [100, 10]  # 2-level: 100 -> 10
        
    elif test_type == 'real_big':
        with open('indices_1481653.txt', 'r') as f:
            data = [line.strip() for line in f.readlines()]
        data = np.array(data)
        centroid_length = 14
        hierarchy = [1000] 

    print(f"Original data shape: {data.shape}")
    print("Sample data:", data[np.random.choice(len(data), 5)])
    
    st = time.time()
    
    # Create and fit hierarchical clustering
    hkmeans = HierarchicalSoftSeqKmeans(hierarchy, centroid_length, alphabet, n_iter=100)
    hkmeans.fit(data)
    
    et = time.time()
    print(f"Hierarchical clustering time: {et - st:.2f} seconds")
    
    # Get results
    labels, centroids = hkmeans.get_labels_and_centroids()
    
    print(f"Final labels shape: {labels.shape}")
    print(f"Number of centroids: {len(centroids)}")
    print("Sample centroids:", centroids[:5])
    
    # Save results
    hkmeans.save_clusters_to_file("1_4mil_sed_1000.txt")
    
    # Evaluate clustering
    evaluate_clustering(hkmeans.data, labels, centroids)


if __name__ == '__main__':
    test_hierarchical_softseqkmeans()