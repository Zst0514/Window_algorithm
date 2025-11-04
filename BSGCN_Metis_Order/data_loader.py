# -*- coding: utf-8 -*-
import csv
import gzip
import os
import pickle

import numpy as np
import scipy.sparse as sp

def load_graph_data(dataset_dir, graph_name):
    """Load a graph adjacency matrix in CSR format.

    The loader first attempts to read a ``.txt`` edge list from ``dataset_dir``.  If the
    text file is unavailable, it falls back to a compressed sparse matrix saved as
    ``graph_name.npz`` either in the same folder or in ``../graph_name/graph_name.npz``
    relative to ``dataset_dir``.  It additionally recognises Planetoid pickles
    (``ind.<name>.graph``) and Open Graph Benchmark style folders that provide
    ``edge.csv`` or ``edge.csv.gz`` within a ``raw`` directory.  Regardless of the
    source, the returned matrix is symmetrised, stripped of self-loops and ready for
    the simulator pipeline.
    """

    edge_file = os.path.join(dataset_dir, graph_name + '.txt')
    npz_candidates = [
        os.path.join(dataset_dir, graph_name + '.npz'),
        os.path.join(os.path.dirname(os.path.dirname(dataset_dir)), graph_name, graph_name + '.npz'),
    ]

    planetoid_candidates = [
        os.path.join(dataset_dir, f'ind.{graph_name}.graph'),
        os.path.join(dataset_dir, graph_name, f'ind.{graph_name}.graph'),
    ]

    ogb_candidates = [
        os.path.join(dataset_dir, 'raw', 'edge.csv'),
        os.path.join(dataset_dir, 'raw', 'edge.csv.gz'),
        os.path.join(dataset_dir, graph_name, 'raw', 'edge.csv'),
        os.path.join(dataset_dir, graph_name, 'raw', 'edge.csv.gz'),
        os.path.join(dataset_dir, 'edge.csv'),
        os.path.join(dataset_dir, 'edge.csv.gz'),
    ]

    if os.path.exists(edge_file):
        return _load_from_txt(edge_file)

    for npz_path in npz_candidates:
        if os.path.exists(npz_path):
            return _load_from_npz(npz_path)

    for graph_path in planetoid_candidates:
        if os.path.exists(graph_path):
            return _load_planetoid_graph(graph_path)

    for csv_path in ogb_candidates:
        if os.path.exists(csv_path):
            return _load_from_edge_csv(csv_path)

    print("No compatible graph file found. Tried:")
    print(f"  - {edge_file}")
    for npz_path in npz_candidates:
        print(f"  - {npz_path}")
    for graph_path in planetoid_candidates:
        print(f"  - {graph_path}")
    for csv_path in ogb_candidates:
        print(f"  - {csv_path}")
    return None


def _attach_directed_edge_metadata(matrix, directed_edges):
    """Record the directed edge count on the returned CSR matrix."""

    matrix = matrix.tocsr()
    try:
        matrix.directed_edge_count = int(directed_edges)
    except Exception:
        pass
    return matrix


def _load_from_txt(edge_file):
    row, col = [], []
    invalid_lines = 0
    with open(edge_file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # Replace common separators (commas, tabs) with whitespace so that
            # inputs like "u,v" or "u,v,weight" can be parsed without raising
            # ValueError.  Only the first two entries are interpreted as the
            # edge endpoints; any additional tokens (weights, metadata, etc.)
            # are ignored.
            tokens = stripped.replace(',', ' ').split()

            if len(tokens) < 2:
                invalid_lines += 1
                continue

            try:
                node1 = int(tokens[0])
                node2 = int(tokens[1])
            except ValueError:
                invalid_lines += 1
                continue

            row.append(node1)
            col.append(node2)

    num_directed_edges_from_file = len(row)
    print(f"Read {num_directed_edges_from_file} directed edges (lines) from file.")

    if not row:
        print("No edges found in the file.")
        if invalid_lines:
            print(f"Encountered {invalid_lines} lines that could not be parsed as edges.")
        return None

    if invalid_lines:
        print(f"Skipped {invalid_lines} non-edge lines during parsing.")

    data = np.ones(len(row))
    num_nodes = max(max(row), max(col)) + 1
    matrix_directed = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    matrix_directed.sum_duplicates()
    matrix_directed.setdiag(0)
    matrix_directed.eliminate_zeros()

    matrix_undirected = _symmetrise(matrix_directed)
    matrix_undirected = _attach_directed_edge_metadata(
        matrix_undirected, matrix_directed.nnz
    )
    _report_graph(matrix_undirected, edge_file, directed_edges=matrix_directed.nnz)
    return matrix_undirected


def _load_from_npz(npz_path):
    try:
        matrix = sp.load_npz(npz_path).tocsr()
    except Exception as exc:  # pragma: no cover - defensive, depends on external files
        print(f"Failed to load sparse matrix from {npz_path}: {exc}")

        try:
            with np.load(npz_path, allow_pickle=True) as payload:
                matrix = _reconstruct_sparse_from_npz(payload)
        except Exception as secondary_exc:  # pragma: no cover - depends on external files
            print(f"Secondary attempt to interpret {npz_path} as generic NPZ failed: {secondary_exc}")
            return None
    else:
        matrix = matrix.tocsr()

    if matrix is None:
        print(f"File {npz_path} did not contain recognised sparse matrix arrays.")
        return None

    directed_edges = matrix.nnz
    matrix = _symmetrise(matrix)
    matrix = _attach_directed_edge_metadata(matrix, directed_edges)
    _report_graph(matrix, npz_path)
    return matrix


def _reconstruct_sparse_from_npz(payload):
    """Attempt to rebuild a sparse matrix from a generic ``np.load`` payload."""

    keys = set(payload.files)

    # torch-geometric style: adj_data/adj_indices/adj_indptr/adj_shape
    if {'adj_data', 'adj_indices', 'adj_indptr', 'adj_shape'} <= keys:
        data = payload['adj_data']
        indices = payload['adj_indices']
        indptr = payload['adj_indptr']
        shape = tuple(payload['adj_shape'])
        return sp.csr_matrix((data, indices, indptr), shape=shape)

    # scipy.sparse save with custom key names data/indices/indptr/shape
    if {'data', 'indices', 'indptr', 'shape'} <= keys:
        data = payload['data']
        indices = payload['indices']
        indptr = payload['indptr']
        shape = tuple(payload['shape'])
        return sp.csr_matrix((data, indices, indptr), shape=shape)

    # COO representation using row/col (optionally data)
    if {'row', 'col'} <= keys:
        row = payload['row']
        col = payload['col']
        data = payload['data'] if 'data' in payload else np.ones_like(row)
        shape = tuple(payload['shape']) if 'shape' in payload else (row.max() + 1, col.max() + 1)
        coo = sp.coo_matrix((data, (row, col)), shape=shape)
        coo.sum_duplicates()
        return coo.tocsr()

    return None


def _load_planetoid_graph(graph_path):
    """Load Planetoid-style graph pickles used by Cora/Citeseer/Pubmed."""

    try:
        with open(graph_path, 'rb') as f:
            graph_dict = pickle.load(f, encoding='latin1')
    except Exception as exc:  # pragma: no cover - depends on external data
        print(f"Failed to load Planetoid graph from {graph_path}: {exc}")
        return None

    if not isinstance(graph_dict, dict) or not graph_dict:
        print(f"Planetoid graph file {graph_path} did not contain adjacency data.")
        return None

    nodes = set(graph_dict.keys())
    for neighbors in graph_dict.values():
        nodes.update(neighbors)

    index_map = {node_id: idx for idx, node_id in enumerate(sorted(nodes))}

    row = []
    col = []
    for src, neighbors in graph_dict.items():
        src_idx = index_map[src]
        for dst in neighbors:
            dst_idx = index_map[dst]
            row.append(src_idx)
            col.append(dst_idx)

    if not row:
        print(f"Planetoid graph {graph_path} contains no edges.")
        return None

    data = np.ones(len(row))
    size = len(index_map)
    matrix_directed = sp.coo_matrix((data, (row, col)), shape=(size, size))
    matrix_directed.sum_duplicates()
    matrix_directed.setdiag(0)
    matrix_directed.eliminate_zeros()

    matrix_undirected = _symmetrise(matrix_directed)
    matrix_undirected = _attach_directed_edge_metadata(
        matrix_undirected, matrix_directed.nnz
    )
    _report_graph(matrix_undirected, graph_path, directed_edges=matrix_directed.nnz)
    return matrix_undirected


def _load_from_edge_csv(csv_path):
    """Load edge lists stored in CSV format (e.g., OGB datasets)."""

    open_fn = gzip.open if csv_path.endswith('.gz') else open
    row, col = [], []
    invalid_lines = 0

    try:
        with open_fn(csv_path, 'rt', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            for tokens in reader:
                if not tokens:
                    continue

                try:
                    node1 = int(tokens[0])
                    node2 = int(tokens[1])
                except (ValueError, IndexError):
                    invalid_lines += 1
                    continue

                row.append(node1)
                col.append(node2)
    except OSError as exc:  # pragma: no cover - depends on external files
        print(f"Failed to read CSV edges from {csv_path}: {exc}")
        return None

    print(f"Read {len(row)} directed edges from CSV file {csv_path}.")

    if not row:
        if invalid_lines:
            print(f"Encountered {invalid_lines} non-edge rows while parsing {csv_path}.")
        return None

    if invalid_lines:
        print(f"Skipped {invalid_lines} rows that did not contain valid edge endpoints.")

    data = np.ones(len(row))
    num_nodes = max(max(row), max(col)) + 1
    matrix_directed = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    matrix_directed.sum_duplicates()
    matrix_directed.setdiag(0)
    matrix_directed.eliminate_zeros()

    matrix_undirected = _symmetrise(matrix_directed)
    matrix_undirected = _attach_directed_edge_metadata(
        matrix_undirected, matrix_directed.nnz
    )
    _report_graph(matrix_undirected, csv_path, directed_edges=matrix_directed.nnz)
    return matrix_undirected


def _symmetrise(matrix):
    matrix = matrix.tocsr()
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    symmetric = (matrix + matrix.T).tocsr()
    symmetric.sum_duplicates()
    symmetric.setdiag(0)
    symmetric.eliminate_zeros()
    return symmetric


def _report_graph(matrix, source_path, directed_edges=None):
    num_nodes = matrix.shape[0]
    if directed_edges is None:
        directed_edges = getattr(matrix, "directed_edge_count", matrix.nnz)
    num_undirected_edges = matrix.nnz // 2

    print(f"\nGraph loaded from {source_path}")
    print(f"Number of vertices: {num_nodes}")
    print(f"Number of directed edges (reported): {directed_edges}")
    print(f"Number of undirected edges (used for simulation): {num_undirected_edges}")
    if num_nodes > 0:
        avg_degree_directed = directed_edges / num_nodes
        avg_degree_undirected = (2.0 * num_undirected_edges) / num_nodes
        print(f"Average degree (directed graph): {avg_degree_directed:.2f}")
        print(f"Average degree (undirected graph): {avg_degree_undirected:.2f}")
    print()