"""
Configuration file for the BSGCN Simulator.

This file centralizes all architectural and simulation parameters.
"""

# --- Band Partitioning Parameters ---
# These parameters are used to classify bands based on row length variations.
# Values are based on user's latest tuning for power-law distribution graphs.
BAND_PARTITION = {
    # 'T_ABS': 65.97,  # collab
    # 'T_ABS': 18.75,  # pokec
    # 'T_ABS': 37,  # pokec
    'T_ABS': 5,  # dblp
    # 'T_ABS': 10,  # imdbbin
    # 'T_ABS': 14,  # arxiv
    # 'T_ABS': 38,  # amazon-photo
    # 'T_ABS': 42,  # amazon-computers
    # 'T_ABS': 18,  # computer-science
    # 'T_ABS': 14.38,  # physics
    # 'T_ABS': 298.5,  # proteins
    # 'T_ABS': 38,  # yelp
    # 'T_ABS': 15,  # pubmed
    # 'T_ABS': 10.08,  # flickr
    # 'T_ABS': 8,  # nell
    'SMOOTHING_WINDOW': 3,   # Size of the sliding window used to smooth row lengths
    'SMOOTHING_MODE': 'mean',  # 'mean' or 'median' smoothing
}

# --- Band Classification Parameters ---
# Threshold to classify a band as 'large' or 'small'.
BAND_CLASSIFICATION = {
    'T_BAND': 128,  # A band with 128 or more rows is considered large.
}

# --- HDN/LDN Identification Parameters ---
# Parameters for identifying High-Degree Nodes (HDNs) locally.
HDN_IDENTIFICATION = {
    'TOP_K': 4096,          # Number of top-degree nodes to select as HDNs per partition
    'NUM_PARTITIONS': 8,   # Number of contiguous partitions used for local HDN selection
}

# --- Architectural Parameters (from experiment.tex) ---
# Defines the hardware configuration of the BSGCN accelerator.
ARCHITECTURE = {
    'MAC_ARRAY_SHAPE': (1, 16), # 1x16 MAC array
    'CLOCK_FREQ_GHZ': 1.0,
    'FEATURE_LENGTH': 16, # Hidden layer dimension
    'FEATURE_DATA_TYPE_SIZE': 8, # bytes, e.g., for float32
    
    # On-chip memory sizes in Kilobytes (KB)
    'HDN_SPM_KB': 512,      # High-Degree Node Scratchpad Memory
    'GLOBAL_CACHE_KB': 16,  # Global Cache (primarily for LDNs in large bands)
    'SMB_KB': 12,           # Sparse Matrix Buffer
    'INDEX_CAM_KB': 12,     # Index Content Addressable Memory for HDN IDs
    
    # Off-chip memory
    'DRAM_BANDWIDTH_GB_S': 128,
}

# --- Large Band Processing Parameters ---
# Defines the window shapes to be profiled for the adaptive window-wise strategy.
LARGE_BAND_PROCESSING = {
    'WINDOW_SHAPES': [(8, 1), (4, 2), (2, 4), (1, 8)],
    # 'WINDOW_SHAPES': [(1, 8), (2, 4), (4, 2), (8, 1)],
    'PROFILING_ROWS': 32, # Number of rows to use for profiling in large bands
}

# --- Simulation Logging Controls ---
SIMULATION_LOGGING = {
    'PRINT_SMALL_BAND_DETAILS': False,
}