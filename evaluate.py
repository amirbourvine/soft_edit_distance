from itertools import combinations
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def edit_distance(s1, s2):
    """Calculate edit distance (Levenshtein distance) between two strings."""
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]

def evaluate_clustering(data, labels, centroids):
    """Comprehensive clustering evaluation"""
    print("=== CLUSTERING EVALUATION METRICS ===\n")
    
    # Basic statistics
    n_samples = len(data)
    n_clusters = len(set(labels))
    cluster_sizes = np.bincount(labels)
    
    print(f"Dataset size: {n_samples}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Average cluster size: {np.mean(cluster_sizes):.2f}")
    print(f"Cluster size std: {np.std(cluster_sizes):.2f}")
    print(f"Min/Max cluster size: {np.min(cluster_sizes)}/{np.max(cluster_sizes)}\n")
    
    
    print("=== CORE METRICS ===")
    
    # calculate and print max,min,avg of distance to centroids
    distances = []
    for i, seq in enumerate(data):
        centroid = centroids[labels[i]]
        dist = edit_distance(seq, centroid)
        distances.append(dist)

    print(f"Min distance to centroids: {np.min(distances)}\n")
    print(f"Max distance to centroids: {np.max(distances)}\n")
    print(f"Average distance to centroids: {np.mean(distances):.4f}\n")


    # calcaulate an print min,max,avg distance between centroids
    centroid_distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist = edit_distance(centroids[i], centroids[j])
            centroid_distances.append(dist)

    print(f"Min distance between centroids: {np.min(centroid_distances)}\n")
    print(f"Max distance between centroids: {np.max(centroid_distances)}\n")
    print(f"Average distance between centroids: {np.mean(centroid_distances):.4f}\n")

    # take top x couples of closest centroids, and for each couple iterate through the clusters to find the closest strings, report the minimal distance amongst all top x couples
    top_x = 1
    closest_centroids = sorted(zip(centroid_distances, range(len(centroids))), key=lambda x: x[0])[:top_x]
    
    min_distance = float('inf')

    for dist, (i, j) in zip(closest_centroids, combinations(range(len(centroids)), 2)):
        # iterate in for loop through strings in cluster i, for each string calculate distances to all strings in cluster j, and report the minimal
        for seq_i in data[labels == i]:
            for seq_j in data[labels == j]:
                d = edit_distance(seq_i, seq_j)
                if d < min_distance:
                    min_distance = d
    
    print(f"Min distance between closest samples from closest centroids: {min_distance}\n")
        
