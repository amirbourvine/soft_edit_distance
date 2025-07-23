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

def calculate_wcss(data, labels, centroids):
    """Within-Cluster Sum of Squares"""
    total_distance = 0.0
    for i in range(len(data)):
        cluster_id = labels[i]
        distance = edit_distance(data[i], centroids[cluster_id])
        total_distance += distance
    return total_distance

def calculate_bcss(data, labels, centroids):
    """Between-Cluster Sum of Squares"""
    # Calculate overall centroid (most frequent character at each position)
    max_len = max(len(seq) for seq in data)
    overall_centroid = ""
    
    for pos in range(max_len):
        char_counts = defaultdict(int)
        for seq in data:
            if pos < len(seq):
                char_counts[seq[pos]] += 1
        if char_counts:
            overall_centroid += max(char_counts, key=char_counts.get)
    
    # Calculate BCSS
    cluster_sizes = np.bincount(labels)
    bcss = 0.0
    
    for cluster_id, centroid in enumerate(centroids):
        if cluster_sizes[cluster_id] > 0:
            distance = edit_distance(centroid, overall_centroid)
            bcss += cluster_sizes[cluster_id] * distance
    
    return bcss

def calculate_silhouette_score(data, labels):
    """Silhouette Score adapted for edit distance"""
    n = len(data)
    silhouette_scores = []
    
    # Group data by clusters
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(i)
    
    for i in range(n):
        cluster_id = labels[i]
        
        # Calculate a(i) - average distance to points in same cluster
        same_cluster = [j for j in clusters[cluster_id] if j != i]
        if len(same_cluster) == 0:
            a_i = 0
        else:
            a_i = np.mean([edit_distance(data[i], data[j]) for j in same_cluster])
        
        # Calculate b(i) - min average distance to points in other clusters
        b_i = float('inf')
        for other_cluster_id, other_indices in clusters.items():
            if other_cluster_id != cluster_id and len(other_indices) > 0:
                avg_dist = np.mean([edit_distance(data[i], data[j]) for j in other_indices])
                b_i = min(b_i, avg_dist)
        
        if b_i == float('inf'):
            b_i = 0
        
        # Silhouette score for point i
        if max(a_i, b_i) == 0:
            s_i = 0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)
        
        silhouette_scores.append(s_i)
    
    return np.mean(silhouette_scores)

def calculate_intra_cluster_distances(data, labels):
    """Calculate statistics of intra-cluster distances"""
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(i)
    
    intra_distances = []
    cluster_stats = {}
    
    for cluster_id, indices in clusters.items():
        if len(indices) <= 1:
            cluster_stats[cluster_id] = {'mean': 0, 'std': 0, 'max': 0, 'min': 0}
            continue
            
        distances = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                dist = edit_distance(data[indices[i]], data[indices[j]])
                distances.append(dist)
                intra_distances.append(dist)
        
        if distances:
            cluster_stats[cluster_id] = {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'max': np.max(distances),
                'min': np.min(distances)
            }
    
    return {
        'overall_mean': np.mean(intra_distances) if intra_distances else 0,
        'overall_std': np.std(intra_distances) if intra_distances else 0,
        'overall_max': np.max(intra_distances) if intra_distances else 0,
        'cluster_stats': cluster_stats
    }

def calculate_inter_cluster_distances(centroids):
    """Calculate distances between cluster centroids"""
    n_clusters = len(centroids)
    distances = []
    
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist = edit_distance(centroids[i], centroids[j])
            distances.append(dist)
    
    return {
        'mean': np.mean(distances) if distances else 0,
        'std': np.std(distances) if distances else 0,
        'min': np.min(distances) if distances else 0,
        'max': np.max(distances) if distances else 0
    }

def calculate_calinski_harabasz_score(data, labels, centroids):
    """Calinski-Harabasz Index adapted for edit distance"""
    n_samples = len(data)
    n_clusters = len(set(labels))
    
    if n_clusters == 1:
        return 0.0
    
    wcss = calculate_wcss(data, labels, centroids)
    bcss = calculate_bcss(data, labels, centroids)
    
    if wcss == 0:
        return float('inf')
    
    ch_score = (bcss / (n_clusters - 1)) / (wcss / (n_samples - n_clusters))
    return ch_score

def calculate_davies_bouldin_score(data, labels, centroids):
    """Davies-Bouldin Index adapted for edit distance"""
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(i)
    
    cluster_distances = {}
    
    # Calculate average intra-cluster distance for each cluster
    for cluster_id, indices in clusters.items():
        if len(indices) <= 1:
            cluster_distances[cluster_id] = 0
            continue
            
        total_dist = 0
        count = 0
        for i in indices:
            dist = edit_distance(data[i], centroids[cluster_id])
            total_dist += dist
            count += 1
        
        cluster_distances[cluster_id] = total_dist / count if count > 0 else 0
    
    # Calculate Davies-Bouldin score
    db_scores = []
    unique_clusters = list(clusters.keys())
    
    for i, cluster_i in enumerate(unique_clusters):
        max_ratio = 0
        for j, cluster_j in enumerate(unique_clusters):
            if i != j:
                centroid_dist = edit_distance(centroids[cluster_i], centroids[cluster_j])
                if centroid_dist > 0:
                    ratio = (cluster_distances[cluster_i] + cluster_distances[cluster_j]) / centroid_dist
                    max_ratio = max(max_ratio, ratio)
        db_scores.append(max_ratio)
    
    return np.mean(db_scores) if db_scores else 0

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
    
    # Core metrics
    wcss = calculate_wcss(data, labels, centroids)
    bcss = calculate_bcss(data, labels, centroids)
    # silhouette = calculate_silhouette_score(data, labels)
    
    print("=== CORE METRICS ===")
    print(f"WCSS (Within-Cluster Sum of Squares): {wcss:.2f}")
    print(f"BCSS (Between-Cluster Sum of Squares): {bcss:.2f}")
    print(f"WCSS/n_samples (avg distance to centroid): {wcss/n_samples:.4f}")
    # print(f"Silhouette Score: {silhouette:.4f}")
    
    # Cluster quality indices
    ch_score = calculate_calinski_harabasz_score(data, labels, centroids)
    # db_score = calculate_davies_bouldin_score(data, labels, centroids)
    
    print(f"Calinski-Harabasz Score: {ch_score:.2f}")
    # print(f"Davies-Bouldin Score: {db_score:.4f}\n")
    
    # Distance analysis
    # intra_stats = calculate_intra_cluster_distances(data, labels)
    # inter_stats = calculate_inter_cluster_distances(centroids)
    
    # print("=== DISTANCE ANALYSIS ===")
    # print("Intra-cluster distances (within clusters):")
    # print(f"  Mean: {intra_stats['overall_mean']:.4f}")
    # print(f"  Std:  {intra_stats['overall_std']:.4f}")
    # print(f"  Max:  {intra_stats['overall_max']}")
    
    # print("Inter-cluster distances (between centroids):")
    # print(f"  Mean: {inter_stats['mean']:.4f}")
    # print(f"  Std:  {inter_stats['std']:.4f}")
    # print(f"  Min:  {inter_stats['min']}")
    # print(f"  Max:  {inter_stats['max']}\n")
    
    # # Separation ratio
    # if intra_stats['overall_mean'] > 0:
    #     separation_ratio = inter_stats['mean'] / intra_stats['overall_mean']
    #     print(f"Separation Ratio (inter/intra): {separation_ratio:.4f}")
    #     print("  > 1.0 indicates well-separated clusters\n")
    
    return {
        'wcss': wcss,
        'bcss': bcss,
        # 'silhouette': silhouette,
        'calinski_harabasz': ch_score,
        # 'davies_bouldin': db_score,
        # 'intra_cluster_stats': intra_stats,
        # 'inter_cluster_stats': inter_stats,
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes
    }

# Usage with your existing variables:
# metrics = evaluate_clustering(data, labels, centroid)