import sys
import types

fake_module = types.ModuleType("distutils.msvccompiler")
fake_module.get_build_version = lambda: None
sys.modules["distutils.msvccompiler"] = fake_module

import numpy as np
import time
from typing import List, Tuple, Optional
import gc
from seq_kmeans import SoftSeqKmeans  # Using SoftSeqKmeans as shown in the original code


class Cluster:
    """Container for a single cluster with its centroid and data points."""
    def __init__(self, centroid: str, data_indices: List[int]):
        self.centroid = centroid
        self.data_indices = data_indices


class HierarchicalClusterer:
    """
    Large-scale hierarchical clustering that breaks down the problem into manageable chunks.
    
    Handles clustering of up to 1B items by iteratively clustering mega-clusters and
    reorganizing based on centroid clustering.
    """
    
    def __init__(self, 
                 alphabet: np.ndarray,
                 mega_cluster_size: int = 1_000_000,
                 centroids_per_mega: int = 1000,
                 final_clusters: int = 1000,
                 centroid_length: int = 20,
                 n_iter_per_clustering: int = 100):
        """
        Initialize the hierarchical clusterer.
        
        Args:
            alphabet: Array of possible characters/symbols
            mega_cluster_size: Maximum size of each mega-cluster (default: 1M)
            centroids_per_mega: Number of centroids per mega-cluster (default: 1000)
            final_clusters: Final number of clusters (default: 1000)
            centroid_length: Length of centroids for the base clusterer
            n_iter_per_clustering: Number of iterations for each clustering operation
        """
        self.alphabet = alphabet
        self.mega_cluster_size = mega_cluster_size
        self.centroids_per_mega = centroids_per_mega
        self.final_clusters_num = final_clusters
        self.centroid_length = centroid_length
        self.n_iter_per_clustering = n_iter_per_clustering
        
        # State variables
        self.mega_clusters = []
        self.current_centroids = None
        self.final_clusters = []
        self.data = None  # Store original data for saving clusters
        
    def _create_initial_mega_clusters(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Split the data into initial mega-clusters.
        
        Args:
            data: Input data array
            
        Returns:
            List of mega-clusters (numpy arrays)
        """
        print(f"Creating initial mega-clusters from {len(data)} items...")
        
        # Shuffle data to ensure random distribution
        shuffled_indices = np.random.permutation(len(data))
        shuffled_data = data[shuffled_indices]
        
        mega_clusters = []
        n_mega_clusters = (len(data) + self.mega_cluster_size - 1) // self.mega_cluster_size
        
        for i in range(n_mega_clusters):
            start_idx = i * self.mega_cluster_size
            end_idx = min((i + 1) * self.mega_cluster_size, len(data))
            mega_cluster = shuffled_data[start_idx:end_idx]
            mega_clusters.append(mega_cluster)
            
        print(f"Created {len(mega_clusters)} mega-clusters")
        return mega_clusters
    
    def _cluster_mega_cluster(self, mega_cluster_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster a single mega-cluster and return centroids and labels.
        
        Args:
            mega_cluster_data: Data for one mega-cluster
            
        Returns:
            Tuple of (centroids as strings, labels)
        """
        # Create unique data to avoid duplicates during clustering
        unique_data = np.unique(mega_cluster_data)

        # Create clusterer instance
        clusterer = SoftSeqKmeans(
            self.centroids_per_mega, 
            self.centroid_length, 
            self.alphabet
        )

        if unique_data.shape[0] < self.centroids_per_mega:
            print(f"Not enough unique data points for clustering (found {unique_data.shape[0]}, needed {self.centroids_per_mega})")
            exit(0)

        # Fit the clusterer
        clusterer.fit(unique_data, n_iter=self.n_iter_per_clustering)
        
        # Get labels for all original data
        labels = clusterer.transform(mega_cluster_data)
        
        # Get centroids
        centroid_probs = clusterer.get_centroid()
        centroid_indices = np.argmax(centroid_probs, axis=1)
        centroids = [''.join(self.alphabet[seq]) for seq in centroid_indices]
        
        return np.array(centroids), labels
    
    def _cluster_all_mega_clusters(self, mega_clusters: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Cluster all mega-clusters and collect their centroids.
        
        Args:
            mega_clusters: List of mega-cluster data arrays
            
        Returns:
            Tuple of (list of centroid arrays, list of label arrays)
        """
        print(f"Clustering {len(mega_clusters)} mega-clusters...")
        
        all_centroids = []
        all_labels = []
        
        for i, mega_cluster in enumerate(mega_clusters):
            if i % 10 == 0:
                print(f"  Processing mega-cluster {i+1}/{len(mega_clusters)}")
                
            centroids, labels = self._cluster_mega_cluster(mega_cluster)
            all_centroids.append(centroids)
            all_labels.append(labels)
            
            # Memory cleanup
            gc.collect()
            
        return all_centroids, all_labels
    
    def _cluster_centroids(self, all_centroids: List[np.ndarray]) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """
        Cluster all centroids from mega-clusters to reorganize them.
        
        Args:
            all_centroids: List of centroid arrays from each mega-cluster
            
        Returns:
            Tuple of (final centroids, assignment of original centroids to new clusters as (mega_idx, local_centroid_idx) tuples)
        """
        print("Clustering all centroids...")
        
        # Flatten all centroids
        flat_centroids = np.concatenate(all_centroids, axis=0)
        print(f"  Total centroids to cluster: {len(flat_centroids)}")
        
        # Remove duplicates
        unique_centroids = np.unique(flat_centroids)
        
        # Cluster the centroids
        centroid_clusterer = SoftSeqKmeans(
            self.final_clusters_num,
            self.centroid_length,
            self.alphabet
        )
        
        centroid_clusterer.fit(unique_centroids, n_iter=self.n_iter_per_clustering)
        
        # Get labels for all original centroids
        centroid_labels = centroid_clusterer.transform(flat_centroids)
        
        # Get final centroids
        final_centroid_probs = centroid_clusterer.get_centroid()
        final_centroid_indices = np.argmax(final_centroid_probs, axis=1)
        final_centroids = np.array([''.join(self.alphabet[seq]) for seq in final_centroid_indices])
        
        # Organize assignments: which original centroids belong to which new clusters
        assignments = [[] for _ in range(self.final_clusters_num)]
        centroid_idx = 0
        
        for mega_idx, centroids in enumerate(all_centroids):
            for local_centroid_idx in range(len(centroids)):
                cluster_id = centroid_labels[centroid_idx]
                assignments[cluster_id].append((mega_idx, local_centroid_idx))
                centroid_idx += 1
                
        return final_centroids, assignments
    
    def _reorganize_data(self, 
                        mega_clusters: List[np.ndarray], 
                        all_labels: List[np.ndarray], 
                        assignments: List[List[Tuple[int, int]]]) -> List[np.ndarray]:
        """
        Reorganize original data based on centroid clustering results.
        
        Args:
            mega_clusters: Original mega-clusters
            all_labels: Labels from clustering each mega-cluster
            assignments: Assignment of centroids to new clusters
            
        Returns:
            New mega-clusters organized by centroid clustering
        """
        print("Reorganizing data into new mega-clusters...")
        
        new_mega_clusters = [[] for _ in range(self.final_clusters_num)]
        
        # For each new cluster, collect all data points that were assigned to its centroids
        for new_cluster_id, centroid_assignments in enumerate(assignments):
            for mega_idx, local_centroid_idx in centroid_assignments:
                # Find all data points in this mega-cluster assigned to this centroid
                mega_cluster_data = mega_clusters[mega_idx]
                mega_cluster_labels = all_labels[mega_idx]
                
                # Get data points assigned to this centroid
                mask = mega_cluster_labels == local_centroid_idx
                assigned_data = mega_cluster_data[mask]
                
                new_mega_clusters[new_cluster_id].extend(assigned_data)
        
        # Convert to numpy arrays and filter empty clusters
        result_clusters = []
        for cluster_data in new_mega_clusters:
            if len(cluster_data) > 0:
                result_clusters.append(np.array(cluster_data))
                
        print(f"Created {len(result_clusters)} new mega-clusters")
        return result_clusters
    
    def _create_final_clusters(self, 
                             mega_clusters: List[np.ndarray], 
                             all_labels: List[np.ndarray], 
                             final_centroids: np.ndarray,
                             assignments: List[List[Tuple[int, int]]]) -> List[Cluster]:
        """
        Create final cluster objects with centroids and data indices.
        
        Args:
            mega_clusters: Original mega-clusters
            all_labels: Labels from clustering each mega-cluster
            final_centroids: Final centroids from centroid clustering
            assignments: Assignment of original centroids to new clusters
            
        Returns:
            List of Cluster objects
        """
        print("Creating final cluster objects...")
        
        clusters = []
        
        for cluster_id in range(len(final_centroids)):
            centroid = final_centroids[cluster_id]
            data_indices = []
            
            # Get assignments for this cluster
            centroid_assignments = assignments[cluster_id] if cluster_id < len(assignments) else []
            
            # Collect all data indices for this cluster
            current_offset = 0
            for mega_idx, mega_cluster in enumerate(mega_clusters):
                mega_cluster_labels = all_labels[mega_idx]
                
                # Find which local centroids in this mega-cluster belong to current cluster
                local_centroids_in_cluster = [local_idx for (m_idx, local_idx) in centroid_assignments if m_idx == mega_idx]
                
                for local_centroid_idx in local_centroids_in_cluster:
                    # Find data points assigned to this local centroid
                    mask = mega_cluster_labels == local_centroid_idx
                    local_indices = np.where(mask)[0]
                    
                    # Convert to global indices
                    global_indices = [current_offset + idx for idx in local_indices]
                    data_indices.extend(global_indices)
                
                current_offset += len(mega_cluster)
            
            if len(data_indices) > 0:  # Only create cluster if it has data points
                clusters.append(Cluster(centroid, data_indices))
        
        print(f"Created {len(clusters)} final clusters")
        return clusters
    
    def fit(self, data: np.ndarray, n_iterations: int = 5, verbose: bool = True) -> List[Cluster]:
        """
        Fit the hierarchical clusterer and return final clusters.
        
        Args:
            data: Input data to cluster
            n_iterations: Number of hierarchical iterations
            verbose: Whether to print progress information
            
        Returns:
            List of Cluster objects containing centroids and data indices
        """
        print(f"Starting hierarchical clustering with {len(data)} items...")
        print(f"Configuration: {n_iterations} iterations, {self.mega_cluster_size} items per mega-cluster")
        
        # Store original data for saving clusters later
        self.data = data.copy()
        
        # Initialize with random mega-clusters
        current_mega_clusters = self._create_initial_mega_clusters(data)
        
        for iteration in range(n_iterations):
            if verbose:
                print(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")
                
            start_time = time.time()
            
            # Step 1: Cluster each mega-cluster
            all_centroids, all_labels = self._cluster_all_mega_clusters(current_mega_clusters)
            
            # Step 2: Cluster all centroids to reorganize
            final_centroids, assignments = self._cluster_centroids(all_centroids)
            
            # Step 3: Handle final iteration differently
            if iteration < n_iterations - 1:
                # Reorganize data for next iteration
                current_mega_clusters = self._reorganize_data(current_mega_clusters, all_labels, assignments)
            else:
                # On final iteration, create final cluster objects
                self.current_centroids = final_centroids
                self.final_clusters = self._create_final_clusters(
                    current_mega_clusters, all_labels, final_centroids, assignments
                )
            
            iteration_time = time.time() - start_time
            if verbose:
                print(f"Iteration {iteration + 1} completed in {iteration_time:.2f} seconds")
                
            # Memory cleanup
            gc.collect()
        
        return
    
    def get_centroids(self) -> np.ndarray:
        """Get the final centroids."""
        if self.current_centroids is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self.current_centroids
    
    def get_clusters(self) -> List[Cluster]:
        """Get the final clusters."""
        if not self.final_clusters:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self.final_clusters
    
    def save_clusters_to_file(self, filename: str):
        """Save clusters to file"""
        if not self.final_clusters:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        if self.data is None:
            raise RuntimeError("Original data not available for saving.")
            
        with open(filename, 'w') as f:
            print(f"Saving {len(self.final_clusters)} clusters to file: {filename}")
            
            for i, cluster in enumerate(self.final_clusters):
                # Write centroid
                f.write(f"{cluster.centroid}\n")
                f.write("*************\n")
                
                # Write all data points in this cluster
                for data_idx in cluster.data_indices:
                    f.write(f"{self.data[data_idx]}\n")
                f.write("\n")
        
        print(f"Clusters saved successfully to {filename}")
    
    def get_memory_usage_mb(self) -> float:
        """Estimate current memory usage in MB."""
        total_size = 0
        
        if self.mega_clusters:
            for cluster in self.mega_clusters:
                total_size += cluster.nbytes
                
        if self.current_centroids is not None:
            total_size += self.current_centroids.nbytes
            
        return total_size / (1024 * 1024)


def example_usage():
    """
    Example of how to use the HierarchicalClusterer.
    """
    # Setup (similar to original code)
    alphabet = np.array(['T', 'A', 'G', 'C'])
    
    # Generate or load your large dataset here
    # For example, you could load from a file:
    print("before loading data")

    with open('indices_1m.txt', 'r') as f:
        data = np.array([line.strip() for line in f.readlines()])
    
    print("after loading data")

    # Initialize the hierarchical clusterer
    clusterer = HierarchicalClusterer(
        alphabet=alphabet,
        mega_cluster_size=250_000,  # Adjust based on your memory constraints
        centroids_per_mega=500,   # Adjust based on your needs
        final_clusters=500,       # Final number of clusters you want
        centroid_length=20,       # Length of sequence centroids
        n_iter_per_clustering=100  # Iterations for each clustering step
    )

    print("after defining clusterer")

    # For actual use, load your 1B item dataset and run clustering:
    clusterer.fit(data, n_iterations=3)
    clusterer.save_clusters_to_file("huge_clusters_125m_indices.txt")
    


if __name__ == "__main__":
    example_usage()