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


if __name__ == '__main__':
    alphabet = np.array(['T', 'A', 'G', 'C'])
    test_type = 'real_small' # 'real_small' or 'real_big' 'simulated'

    if test_type == 'simulated':
        motifs = ['TAGCGA', 'ATGCAT', 'CCTTGA']
        seq_per_motif = 3000
        data = np.concatenate([generate_data(m, alphabet, seq_per_motif) for m in motifs], axis=0)

        num_clusters = 3
        centroid_length = 6

    if test_type == 'real_small':
        # load data from indices_55000.txt file. each line is a sequence
        with open('indices_55000.txt', 'r') as f:
            data = [line.strip() for line in f.readlines()]
        # change data to np.array
        data = np.array(data)

        num_clusters = 100
        centroid_length = 12

    if test_type == 'real_big':
        with open('indices_1481653.txt', 'r') as f:
            data = [line.strip() for line in f.readlines()]
        # change data to np.array
        data = np.array(data)

        num_clusters = 1000
        centroid_length = 14

    # print("data shape:", data.shape)
    # print(data[np.random.choice(len(data), 10)])

    st = time.time()

    # clusters = SeqKmeans(num_clusters, centroid_length, alphabet)
    clusters = SoftSeqKmeans(num_clusters, centroid_length, alphabet)
    clusters.fit(data, n_iter=100)

    data = np.unique(data)
    labels = clusters.transform(data)

    centroid = clusters.get_centroid()
    centroid = alphabet[np.argmax(centroid, axis=1)]
    centroid = [''.join(seq) for seq in centroid]

    et = time.time()
    print(f"Clustering time: {et - st:.2f} seconds")

    # print(f'{centroid=}')
    # vis(data, labels, centroid, alphabet)

    metrics = evaluate_clustering(data, labels, centroid)
