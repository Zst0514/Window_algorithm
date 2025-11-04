"""
Main entry point for running the BSGCN simulation.
"""
import config
from data_loader import load_graph_data
from simulator import Simulator
from stats_logger import StatsLogger

def main():
    """
    Configures and runs the BSGCN simulation for a specified dataset.
    """
    # --- Simulation Setup ---
    # For now, we hardcode the dataset. In the future, this could come from args.
    # dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/Enron/'
    # graph_name = 'Enron'
    # dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/txt_file/pokec'
    # graph_name = 'pokec'
    dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/nell'
    graph_name = 'nell'
    # dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/flickr'
    # graph_name = 'flickr'
    # dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/imdbbin'
    # graph_name = 'imdbbin'
    # dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/txt_file/collab'
    # graph_name = 'collab'
    # dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/txt_file/dblp'
    # graph_name = 'dblp'
    # dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/reddit'
    # graph_name = 'reddit_graph'

    # 2. graphsaint_related
    # dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/yelp'
    # graph_name = 'yelp_graph'
    # dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/ogbn/'
    # graph_name = 'proteins'
    # dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/ogbn/'
    # graph_name = 'arxiv'
    # dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/'
    # graph_name = 'pubmed'
    
    # 3.HyMM added
    # dataset_dir = '/home/zhangshangtong/Window_algorithm/GCNAX/Nowtest/datasets/added'
    # graph_name = 'Amazon-Computers'
    # graph_name = 'Amazon-Photo'
    # graph_name = 'Computer-Science'
    # graph_name = 'Physics'
    
    
    # --- Load Data ---
    adj_matrix = load_graph_data(dataset_dir, graph_name)
    if adj_matrix is None:
        print(f"Failed to load data for {graph_name}. Exiting.")
        return

    # --- Initialize and Run Simulator ---
    bsgcn_simulator = Simulator(config)
    final_stats = bsgcn_simulator.run(adj_matrix)

    # --- Report Results ---
    if final_stats:
        final_stats.report()
    else:
        print("Simulation did not produce statistics.")

if __name__ == "__main__":
    main()
