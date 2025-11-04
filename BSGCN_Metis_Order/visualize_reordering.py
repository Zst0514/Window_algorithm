import argparse
import numpy as np
import pymetis as metis
import matplotlib.pyplot as plt
from data_loader import load_graph_data
import scipy.sparse as sp
import os

np.random.seed(0) 

def matrix_to_undirected_adj_list(matrix):
    """Converts a CSR matrix to a GCNAX-style undirected adjacency list."""
    adj_list = [[] for _ in range(matrix.shape[0])]
    matrix_coo = matrix.tocoo()
    for i, j in zip(matrix_coo.row, matrix_coo.col):
        adj_list[i].append(j)
        adj_list[j].append(i)
    return adj_list

def save_sparse_matrix_plot(matrix, title, graph_name, stage, point_size):
    """Generates and saves a sparsity plot of the matrix."""
    save_dir = os.path.join('./figure', graph_name)
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 10))
    row, col = matrix.nonzero()

    # Downsample for very large graphs to keep visualization responsive
    num_edges = len(row)
    if num_edges > 1e6:
        print(f"--> Downsampling {num_edges:,} edges to 1,000,000 for plotting...")
        indices = np.random.choice(num_edges, size=int(1e6), replace=False)
        row, col = row[indices], col[indices]

    plt.scatter(col, row, s=point_size, color='blue')
    plt.title(title, fontsize=24)
    plt.xlabel('Columns', fontsize=22)
    plt.ylabel('Rows', fontsize=22)
    plt.gca().invert_yaxis()

    file_path = os.path.join(save_dir, f"Adj_{stage}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved sparsity plot to '{file_path}'")

def main():
    parser = argparse.ArgumentParser(description="Visualize METIS Graph Reordering")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--nparts', type=int, default=64, help="Number of partitions for METIS.")
    parser.add_argument('--marker-size', type=float, default=0.005, help="Marker size for the sparsity plot.")
    args = parser.parse_args()

    adj, _ = load_graph_data(args.dataset)
    if adj is None:
        return

    adj = adj.tocsr()
    print("Generating plot for original matrix...")
    save_sparse_matrix_plot(
        adj,
        f"Original Sparsity Pattern for '{args.dataset}'",
        args.dataset,
        "Before",
        args.marker_size
    )

    print("\nPerforming METIS reordering...")
    adj_list = matrix_to_undirected_adj_list(adj)
    _ , partitions = metis.part_graph(args.nparts, adjacency=adj_list)
    partitions = np.array(partitions)
    
    # Use the exact, stable reordering logic from GCNAX
    partitioned_indices = []
    for i in range(args.nparts):
        partitioned_indices.extend(np.where(partitions == i)[0])

    adj_reordered = adj[partitioned_indices, :][:, partitioned_indices]
    print("METIS reordering complete.")

    print("\nGenerating plot for reordered matrix...")
    save_sparse_matrix_plot(
        adj_reordered,
        f"Sparsity Pattern for '{args.dataset}' (After METIS, {args.nparts} parts)",
        args.dataset,
        "After",
        args.marker_size
    )

    print("\nVisualization script finished.")

if __name__ == "__main__":
    main()
