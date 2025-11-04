"""
Main entry point for running the BSGCN simulation.
"""
from pathlib import Path
import argparse

import config
from data_loader import load_graph_data
from simulator import Simulator

def run_simulation_for_partition(adj_matrix, graph_name, dataset_dir, dataset_stats, num_partitions, output_root):
    """Run simulation for a specific partition number."""
    print(f"\n{'='*60}")
    print(f"Running simulation with {num_partitions} partitions")
    print(f"{'='*60}\n")
    
    # Temporarily modify config
    original_partitions = config.HDN_IDENTIFICATION['NUM_PARTITIONS']
    config.HDN_IDENTIFICATION['NUM_PARTITIONS'] = num_partitions
    
    try:
        # Initialize and Run Simulator
        bsgcn_simulator = Simulator(
            config,
            dataset_name=graph_name,
            dataset_stats=dataset_stats,
            output_root=output_root,
        )
        final_stats = bsgcn_simulator.run(adj_matrix)

        # Report Results
        if final_stats:
            final_stats.report()
            log_path = bsgcn_simulator.write_log(final_stats)
            if log_path:
                print(f"Detailed log saved to {log_path}")
        else:
            print("Simulation did not produce statistics.")
            
    finally:
        # Restore original config
        config.HDN_IDENTIFICATION['NUM_PARTITIONS'] = original_partitions

def main():
    """
    Configures and runs the BSGCN simulation for a specified dataset.
    """
    parser = argparse.ArgumentParser(description="BSGCN Metis Order Simulator")
    parser.add_argument('--dataset', type=str, default='proteins', help="Dataset name")
    parser.add_argument('--partition', type=int, nargs='+', default=[4, 8, 16, 32, 64, 128],
                       help="Number of partitions (can specify multiple)")
    args = parser.parse_args()
    
    # --- Simulation Setup ---
    # For now, we hardcode the dataset. In the future, this could come from args.
    if args.dataset == 'dblp':
        dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/txt_file/dblp'
        graph_name = 'dblp'
    elif args.dataset == 'proteins':
        dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/ogbn/'
        graph_name = 'proteins'
    elif args.dataset == 'imdbbin':
        dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/imdbbin'
        graph_name = 'imdbbin'
    elif args.dataset == 'Physics':
        dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/Physics'
        graph_name = 'Physics'
    elif args.dataset == 'pubmed':
        dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/pubmed'
        graph_name = 'pubmed'
    elif args.dataset == 'pokec':
        dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/txt_file/pokec'
        graph_name = 'pokec'
    elif args.dataset == 'collab':
        dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/collab'
        graph_name = 'collab'
    elif args.dataset == 'yelp':
        dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/yelp'
        graph_name = 'yelp'
    # Add more datasets as needed
    elif args.dataset == 'cora':
        dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/'
        graph_name = 'cora'
    else:
        # Default or add more dataset mappings
        dataset_dir = f'/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/ogbn'
        graph_name = args.dataset
    
    print(f"Dataset: {graph_name}")
    print(f"Partitions to run: {args.partition}")
    print(f"Total: {len(args.partition)} configurations\n")
    
    # --- Load Data ---
    print(f"Loading dataset: {graph_name}...")
    adj_matrix = load_graph_data(dataset_dir, graph_name)
    if adj_matrix is None:
        print(f"Failed to load data for {graph_name}. Exiting.")
        return

    num_nodes = adj_matrix.shape[0]
    directed_edges = getattr(adj_matrix, 'directed_edge_count', adj_matrix.nnz)
    undirected_edges = adj_matrix.nnz // 2
    avg_degree = (2.0 * undirected_edges / num_nodes) if num_nodes else 0.0

    dataset_stats = {
        'graph_name': graph_name,
        'dataset_dir': dataset_dir,
        'num_nodes': num_nodes,
        'directed_edges': directed_edges,
        'undirected_edges': undirected_edges,
        'avg_degree': avg_degree,
    }

    output_root = Path(__file__).resolve().parent / 'output'

    # --- Run simulations for each partition number ---
    for partition_num in args.partition:
        run_simulation_for_partition(
            adj_matrix, graph_name, dataset_dir, dataset_stats, 
            partition_num, output_root
        )
    
    print(f"\n{'='*60}")
    print(f"All {len(args.partition)} simulations completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()