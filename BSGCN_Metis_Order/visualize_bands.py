# -*- coding: utf-8 -*-
# Force re-encoding
import os
import numpy as np
import scipy.sparse as sp
import pymetis
import matplotlib.pyplot as plt

np.random.seed(0)

def load_graph_data(dataset_dir, graph_name):
    """
    Load graph data from a specified directory and return a sparse adjacency matrix (CSR format).
    This function reads directed edges, reports their count, and then returns a symmetric,
    undirected version of the graph suitable for algorithms like Metis.
    """
    edge_file = os.path.join(dataset_dir, graph_name + '.txt')
    
    if not os.path.exists(edge_file):
        print(f"Edge file not found at {edge_file}")
        return None

    row, col = [], []
    with open(edge_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            try:
                node1, node2 = map(int, line.strip().split())
                row.append(node1)
                col.append(node2)
            except ValueError:
                # Skip lines that do not contain two integers
                continue
    
    num_directed_edges_from_file = len(row)
    print(f"Read {num_directed_edges_from_file} directed edges (lines) from file.")

    if not row:
        print("No edges found in the file.")
        return None

    # Create a directed graph matrix first from the raw edges
    data = np.ones(len(row))
    num_nodes = max(max(row), max(col)) + 1
    # Use coo_matrix as it's efficient for initial creation
    matrix_directed = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    # Remove duplicates that might exist in the file and self-loops
    matrix_directed.sum_duplicates()
    matrix_directed.setdiag(0)
    matrix_directed.eliminate_zeros()

    # Report the number of unique directed edges
    num_unique_directed_edges = matrix_directed.nnz
    
    # Now, create a symmetric, undirected version for Metis
    # A_sym = A + A.T ensures that if an edge (u, v) exists, (v, u) also exists.
    matrix_undirected = (matrix_directed + matrix_directed.T).tocsr()
    matrix_undirected.sum_duplicates() # Consolidate elements created by the sum
    matrix_undirected.setdiag(0) # Ensure no self-loops in the final graph
    matrix_undirected.eliminate_zeros()

    num_undirected_edges = matrix_undirected.nnz // 2

    print(f"\nGraph loaded from {edge_file}")
    print(f"Number of vertices: {num_nodes}")
    print(f"Number of unique directed edges: {num_unique_directed_edges} (official number is ~30.6M)")
    print(f"Number of undirected edges (used for Metis): {num_undirected_edges}")
    print(f"Average degree (based on undirected graph): {matrix_undirected.sum(axis=1).mean():.2f}\n")

    return matrix_undirected

def matrix_to_adjacency_list(matrix):
    """Converts a sparse matrix to an adjacency list."""
    adjacency_list = [[] for _ in range(matrix.shape[0])]
    rows, cols = matrix.nonzero()
    for row, col in zip(rows, cols):
        # We assume the matrix is symmetric, so we only need to add one direction
        # if we iterate through all non-zero elements.
        adjacency_list[row].append(col)
    return adjacency_list

def partition_graph(matrix, num_parts):
    """
    Partitions the graph using Metis.
    Assumes the input matrix is already symmetric.
    """
    # The input matrix from load_graph_data is already symmetric.
    adjacency_list = matrix_to_adjacency_list(matrix)
    print(f"Partitioning graph with {num_parts} parts using Metis...")
    _, partitions = pymetis.part_graph(num_parts, adjacency=adjacency_list)
    return np.array(partitions)

def reorder_matrix_by_partition(matrix, partitions, num_parts):
    """Reorders the matrix rows and columns based on Metis partitions."""
    print("Reordering matrix based on partitions...")
    partitioned_indices = []
    for i in range(num_parts):
        partitioned_indices.extend(np.where(partitions == i)[0])
    
    reordered_matrix = matrix[partitioned_indices, :][:, partitioned_indices]
    return reordered_matrix

def partition_into_bands(matrix, t_abs=37, t_rel=8):
    """
    Partition the matrix into bands based on row length variations.
    """
    if not sp.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()

    num_rows = matrix.shape[0]
    row_lengths = np.diff(matrix.indptr)
    
    band_boundaries = [0]
    
    for i in range(1, num_rows):
        len_i = row_lengths[i]
        len_prev = row_lengths[i-1]
        
        # Check for new band creation
        # 1. Absolute difference
        if abs(len_i - len_prev) > t_abs:
            band_boundaries.append(i)
            continue
            
        # 2. Relative difference
        if len_prev > 0 and (len_i / len_prev > t_rel or len_prev / len_i > t_rel):
            band_boundaries.append(i)
        elif len_prev == 0 and len_i > t_abs: # Handle case where previous row was empty
             band_boundaries.append(i)

    band_boundaries.append(num_rows)
    
    # Remove duplicates and sort
    unique_boundaries = sorted(list(set(band_boundaries)))
    
    print(f"Partitioned into {len(unique_boundaries) - 1} bands.")
    return unique_boundaries

def analyze_and_print_band_stats(band_boundaries, t_band=128):
    """
    Analyzes bands, prints the count of small bands, and details of large bands.
    """
    if len(band_boundaries) < 2:
        print("Not enough boundaries to form any bands.")
        return

    small_bands_count = 0
    large_bands_details = []

    # Iterate through the boundaries to define each band
    for i in range(len(band_boundaries) - 1):
        start_row = band_boundaries[i]
        end_row = band_boundaries[i+1]
        length = end_row - start_row

        if length < t_band:
            small_bands_count += 1
        else:
            # For a large band, store its details. The node range is [start, end-1].
            large_bands_details.append({
                'start': start_row, 
                'end': end_row - 1, 
                'length': length
            })
            
    total_bands = small_bands_count + len(large_bands_details)

    print("\n--- Band Statistics ---")
    print(f"Total number of bands: {total_bands}")
    print(f"Threshold for large band (T_band): {t_band} rows")
    print("-" * 25)
    print(f"Number of small bands: {small_bands_count}")
    print(f"Number of large bands: {len(large_bands_details)}")
    
    if large_bands_details:
        print("\nDetails of large bands:")
        for band in large_bands_details:
            print(f"  - Node ID Range: [{band['start']}, {band['end']}], Length: {band['length']} rows")
            
    print("--- End of Statistics ---\n")


def plot_row_lengths_with_bands(matrix, band_boundaries, title, graph_name, stage):
    """
    Plot row length distribution and mark band boundaries with vertical lines.
    """
    save_dir = os.path.join('./figure', graph_name)
    os.makedirs(save_dir, exist_ok=True)
    
    row_lengths = np.diff(matrix.indptr)
    
    plt.figure(figsize=(12, 6))
    plt.plot(row_lengths, label='Row Length', color='lightblue', linewidth=1)
    
    # Plot a smoothed version for better visualization
    window_size = 50
    if len(row_lengths) > window_size:
        smoothed_lengths = np.convolve(row_lengths, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size//2, len(row_lengths) - window_size//2 + 1), smoothed_lengths, label=f'Smoothed (window={window_size})', color='blue')

    for boundary in band_boundaries:
        plt.axvline(x=boundary, color='r', linestyle='--', linewidth=0.8)
        
    plt.xlabel('Row Index', fontsize=14)
    plt.ylabel('Row Length (Number of Non-zeros)', fontsize=14)
    plt.title(f'{title} - {graph_name}', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    file_path = os.path.join(save_dir, f"Band_Partition_{stage}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Band visualization saved to {file_path}")

# --- Main Program ---
if __name__ == "__main__":
    # Define the number of partitions for Metis
    num_parts = 64

    # Corrected dataset path based on user's latest changes
    # dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/Enron'
    # graph_names = ['Enron']
    dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/txt_file/pokec'
    graph_names = ['pokec']
    
    for graph_name in graph_names:
        print(f"--- Processing {graph_name} ---")
        adj_matrix = load_graph_data(dataset_dir, graph_name)
        
        if adj_matrix is None:
            continue

        # 1. Partition the graph using Metis
        partitions = partition_graph(adj_matrix, num_parts)

        # 2. Reorder the matrix based on the partitions
        reordered_matrix = reorder_matrix_by_partition(adj_matrix, partitions, num_parts)

        # 3. Partition the reordered matrix into Bands
        band_boundaries = partition_into_bands(reordered_matrix)
        
        # 4. Analyze and print statistics about the bands
        analyze_and_print_band_stats(band_boundaries)

        # 5. Visualize the bands on the reordered matrix
        plot_row_lengths_with_bands(
            reordered_matrix, 
            band_boundaries, 
            f"Band Partition after Metis (parts={num_parts})", 
            graph_name, 
            f"After_Metis_{num_parts}_Parts"
        )

        print(f"Band partitioning and visualization for {graph_name} after Metis is complete.")
