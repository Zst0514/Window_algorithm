"""
Core simulator logic for the BSGCN accelerator.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp

try:
    import pymetis  # type: ignore
except ImportError:  # pragma: no cover - optional dependency at runtime
    pymetis = None

from stats_logger import StatsLogger
from components import Cache, IndexCAM

class Simulator:
    def __init__(self, config, dataset_name=None, dataset_stats=None, output_root=None):
        self.config = config
        self.dataset_name = dataset_name
        self.dataset_stats = dataset_stats or {}
        self.output_root = Path(output_root) if output_root is not None else None
        self.stats = StatsLogger()

        # Initialize hardware components based on config
        # We'll need feature length for cache capacity calculation
        feature_len = self.config.ARCHITECTURE.get('FEATURE_LENGTH', 16)
        feature_dtype_size = self.config.ARCHITECTURE.get('FEATURE_DATA_TYPE_SIZE', 4)

        self.hdn_spm = Cache('HDN-SPM', config.ARCHITECTURE['HDN_SPM_KB'], feature_len, feature_dtype_size)
        self.global_cache = Cache('Global Cache', config.ARCHITECTURE['GLOBAL_CACHE_KB'], feature_len, feature_dtype_size)
        self.index_cam = IndexCAM('Index CAM', config.ARCHITECTURE['INDEX_CAM_KB'])
        self.hdn_ids = set()
        self._hdn_mask = None
        self._permutation = None
        self._log_messages = []
        self._band_summary = {}
        self._partition_summary = {}

    def _log(self, message, record=True):
        """Print a message and optionally record it for the run log."""
        print(message)
        if record:
            self._log_messages.append(message)

    def _compute_partition_boundaries(self, matrix):
        """Compute contiguous partition boundaries without reordering the matrix."""
        if not sp.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()

        num_nodes = matrix.shape[0]
        num_parts = max(self.config.HDN_IDENTIFICATION.get('NUM_PARTITIONS', 64), 1)

        base_size, remainder = divmod(num_nodes, num_parts)
        boundaries = [0]
        current_start = 0

        for part_idx in range(num_parts):
            part_size = base_size + (1 if part_idx < remainder else 0)
            current_end = min(current_start + part_size, num_nodes)

            if current_end > boundaries[-1]:
                boundaries.append(current_end)

            current_start = current_end

            if current_start >= num_nodes:
                break

        if boundaries[-1] != num_nodes:
            boundaries.append(num_nodes)

        return matrix, boundaries

    def _apply_metis_reordering(self, matrix):
        """Reorder the matrix by Metis partitions and return new boundaries."""
        if not sp.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()

        num_nodes = matrix.shape[0]
        num_parts = max(self.config.HDN_IDENTIFICATION.get('NUM_PARTITIONS', 64), 1)

        if num_parts <= 1 or pymetis is None:
            matrix, boundaries = self._compute_partition_boundaries(matrix)
            counts = np.diff(boundaries)
            strategy = 'sequential'
            note = 'pymetis not available' if pymetis is None else 'single partition requested'
            if pymetis is None:
                self._log("pymetis not available; skipping Metis reordering and using sequential partitions.")
            else:
                self._log("Metis reordering skipped because only one partition was requested.")
            self._partition_summary = {
                'strategy': strategy,
                'note': note,
                'requested_parts': num_parts,
                'actual_parts': len(boundaries) - 1,
                'partition_sizes': counts.tolist(),
            }
            self._log(f"Using sequential partitioning with {len(boundaries) - 1} segments.")
            return matrix, boundaries, np.arange(num_nodes)

        self._log("Performing Metis reordering...")

        indptr = matrix.indptr
        indices = matrix.indices
        adjacency_list = [indices[indptr[i]:indptr[i + 1]].tolist() for i in range(num_nodes)]

        try:
            _, partitions = pymetis.part_graph(num_parts, adjacency=adjacency_list)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self._log(f"Metis partitioning failed ({exc}). Falling back to sequential order.")
            matrix, boundaries = self._compute_partition_boundaries(matrix)
            counts = np.diff(boundaries)
            self._partition_summary = {
                'strategy': 'sequential',
                'note': f'fallback due to error: {exc}',
                'requested_parts': num_parts,
                'actual_parts': len(boundaries) - 1,
                'partition_sizes': counts.tolist(),
            }
            self._log(f"Using sequential partitioning with {len(boundaries) - 1} segments.")
            return matrix, boundaries, np.arange(num_nodes)

        partitions = np.asarray(partitions)
        permutation = np.argsort(partitions, kind='mergesort')

        reordered = matrix[permutation, :][:, permutation]

        counts = np.bincount(partitions, minlength=num_parts)
        boundaries = [0]
        for count in counts:
            if count <= 0:
                continue
            boundaries.append(boundaries[-1] + int(count))

        if boundaries[-1] != num_nodes:
            boundaries.append(num_nodes)

        partition_sizes = [int(size) for size in counts.tolist() if size > 0]
        self._partition_summary = {
            'strategy': 'metis',
            'requested_parts': num_parts,
            'actual_parts': len(boundaries) - 1,
            'partition_sizes': partition_sizes,
        }
        self._log(f"Metis reordering with {num_parts} parts complete.")
        self._log(f"Partition sizes (after Metis reorder): {partition_sizes}")
        return reordered, boundaries, permutation

    def _process_small_band(self, matrix, band):
        """
        Simulates the processing of a small band using the HDN-centric,
        row-wise traversal strategy.
        """
        start = band['start_row']
        end = band['end_row']
        
        feature_len = self.config.ARCHITECTURE.get('FEATURE_LENGTH', 16)
        feature_dtype_size = self.config.ARCHITECTURE.get('FEATURE_DATA_TYPE_SIZE', 4)
        
        logging_cfg = getattr(self.config, 'SIMULATION_LOGGING', {})
        if logging_cfg.get('PRINT_SMALL_BAND_DETAILS', False):
            self._log(f"  Processing small band [{start}, {end-1}] ({end-start} rows)")

        indptr = matrix.indptr
        indices = matrix.indices
        hdn_mask = self._hdn_mask
        hdn_spm = self.hdn_spm
        stats = self.stats

        dram_read_bytes = feature_len * feature_dtype_size
        mac_cycles = feature_len / self.config.ARCHITECTURE['MAC_ARRAY_SHAPE'][1]

        log_spm_access = stats.log_hdn_spm_access
        log_cycles = stats.log_cycles
        log_dram = stats.log_dram_read
        spm_access = hdn_spm.access

        for row_idx in range(start, end):
            row_start = indptr[row_idx]
            row_end = indptr[row_idx + 1]

            for nnz_ptr in range(row_start, row_end):
                neighbor_idx = indices[nnz_ptr]

                if hdn_mask is not None and hdn_mask[neighbor_idx]:
                    is_hit = spm_access(neighbor_idx, row_idx)
                    log_spm_access(is_hit)
                    log_cycles(1, 'hdn_spm_access')
                else:
                    log_spm_access(False)
                    log_dram(dram_read_bytes)
                    log_cycles(100, 'dram_access')

                log_cycles(mac_cycles, 'computation')

    def _simulate_window_pass(
        self,
        matrix,
        band,
        window_shape,
        logger,
        hdn_spm=None,
        global_cache=None,
        *,
        record_ldn_access=False,
    ):
        """
        Simulates processing a part of a band with a given window shape, including
        a more realistic cycle model for computation.
        Returns the incremental cycles recorded in ``logger`` for convenience.
        """
        alpha, beta = window_shape
        start_row = band['start_row']
        end_row = band['end_row']

        feature_len = self.config.ARCHITECTURE.get('FEATURE_LENGTH', 16)
        feature_dtype_size = self.config.ARCHITECTURE.get('FEATURE_DATA_TYPE_SIZE', 4)
        dram_read_bytes = feature_len * feature_dtype_size
        mac_array_width = self.config.ARCHITECTURE['MAC_ARRAY_SHAPE'][1]

        hdn_spm = hdn_spm or self.hdn_spm
        global_cache = global_cache or self.global_cache

        starting_cycles = logger.total_cycles

        # Iterate through the band in windows of size alpha
        for i in range(start_row, end_row, alpha):
            current_window_start_row = i
            current_window_end_row = min(i + alpha, end_row)

            # In this more realistic model, we process a window's worth of non-zeros
            # before calculating computation cycles based on parallelism.
            non_zeros_in_window = 0
            
            for row_idx in range(current_window_start_row, current_window_end_row):
                start_ptr = matrix.indptr[row_idx]
                end_ptr = matrix.indptr[row_idx + 1]
                non_zeros_in_window += (end_ptr - start_ptr)
                column_indices = matrix.indices[start_ptr:end_ptr]

                for neighbor_idx in column_indices:
                    # Cache access logic remains the same
                    if self._hdn_mask is not None and self._hdn_mask[neighbor_idx]:
                        is_hit = hdn_spm.access(neighbor_idx, row_idx)
                        logger.log_hdn_spm_access(is_hit)
                        logger.log_cycles(1, 'hdn_spm_access')
                    else:
                        # Record the failed HDN-SPM lookup before consulting the GLC
                        logger.log_hdn_spm_access(False)
                        is_hit = global_cache.access(neighbor_idx, row_idx)
                        logger.log_global_cache_access(is_hit, count_as_ldn=record_ldn_access)
                        logger.log_cycles(1, 'glc_access')

                        if not is_hit:
                            logger.log_dram_read(dram_read_bytes)
                            logger.log_cycles(100, 'dram_access')

            total_mac_ops = non_zeros_in_window * feature_len
            
            effective_parallelism = min(mac_array_width, beta)

            compute_cycles = np.ceil(total_mac_ops / effective_parallelism)
            logger.log_cycles(compute_cycles, 'computation')

        return logger.total_cycles - starting_cycles


    def _process_large_band(self, matrix, band):
        """
        Simulates processing of a large band using the adaptive,
        profiling-guided window-wise strategy.
        """
        profiling_rows = self.config.LARGE_BAND_PROCESSING.get('PROFILING_ROWS', 32)
        window_shapes = self.config.LARGE_BAND_PROCESSING['WINDOW_SHAPES']

        start_node = band['start_row']
        end_node = band['end_row']

        best_shape = window_shapes[0]  # Default shape

        # --- Profiling Phase ---
        if band['length'] >= profiling_rows and len(window_shapes) > 1:
            profiling_length = min(band['length'], profiling_rows)
            profiling_band = {
                'start_row': start_node,
                'end_row': start_node + profiling_length,
                'length': profiling_length,
            }

            baseline_hdn_cache = self.hdn_spm.clone()
            baseline_glc = self.global_cache.clone()

            min_cycles = None
            for shape in window_shapes:
                hdn_clone = baseline_hdn_cache.clone()
                glc_clone = baseline_glc.clone()
                profiling_logger = StatsLogger()
                cycles = self._simulate_window_pass(
                    matrix,
                    profiling_band,
                    shape,
                    profiling_logger,
                    hdn_spm=hdn_clone,
                    global_cache=glc_clone,
                    record_ldn_access=False,
                )
                if min_cycles is None or cycles < min_cycles:
                    min_cycles = cycles
                    best_shape = shape

        # --- Execution Phase ---
        self._simulate_window_pass(
            matrix,
            band,
            best_shape,
            self.stats,
            hdn_spm=self.hdn_spm,
            global_cache=self.global_cache,
            record_ldn_access=True,
        )
            
    def _identify_hdns(self, matrix, partition_boundaries):
        """
        Identifies High-Degree Nodes (HDNs) based on the local, partition-wise strategy.
        """
        self._log("Identifying HDNs using local partition-wise strategy...")
        top_k = self.config.HDN_IDENTIFICATION['TOP_K']
        num_nodes = matrix.shape[0]

        degrees = np.array(matrix.sum(axis=1)).flatten()
        hdn_ids = set()

        for start_node, end_node in zip(partition_boundaries[:-1], partition_boundaries[1:]):
            if end_node <= start_node:
                continue

            partition_indices = np.arange(start_node, end_node)
            partition_degrees = degrees[partition_indices]

            # Determine the number of HDNs to select in this partition
            num_hdns_in_partition = min(top_k, len(partition_degrees))

            if num_hdns_in_partition > 0:
                local_hdn_offsets = np.argsort(-partition_degrees)[:num_hdns_in_partition]
                global_hdn_ids = partition_indices[local_hdn_offsets]
                hdn_ids.update(global_hdn_ids)

        self.hdn_ids = hdn_ids
        mask = np.zeros(num_nodes, dtype=bool)
        if hdn_ids:
            mask[list(hdn_ids)] = True
        self._hdn_mask = mask
        self._log(f"Identified {len(self.hdn_ids)} unique HDNs across all partitions.")

    def _partition_into_bands(self, matrix):
        """
        Partitions the matrix into bands based on row length variations and classifies them.
        Returns a list of band objects.
        """
        self._log("Partitioning matrix into bands...")
        if not sp.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()

        num_rows = matrix.shape[0]
        row_lengths = np.diff(matrix.indptr).astype(np.int64)

        partition_cfg = self.config.BAND_PARTITION
        t_abs = partition_cfg['T_ABS']
        smoothing_window = max(int(partition_cfg.get('SMOOTHING_WINDOW', 1)), 1)
        smoothing_mode = partition_cfg.get('SMOOTHING_MODE', 'mean').lower()
        t_band = self.config.BAND_CLASSIFICATION['T_BAND']

        if smoothing_window > 1:
            left_pad = smoothing_window // 2
            right_pad = smoothing_window - 1 - left_pad
            padded = np.pad(row_lengths, (left_pad, right_pad), mode='edge')

            if smoothing_mode == 'median':
                windows = np.lib.stride_tricks.sliding_window_view(padded, smoothing_window)
                smoothed_lengths = np.median(windows, axis=-1)
            else:
                kernel = np.ones(smoothing_window, dtype=float) / smoothing_window
                smoothed_lengths = np.convolve(padded, kernel, mode='valid')
        else:
            smoothed_lengths = row_lengths.astype(float)

        band_boundaries = [0]
        for i in range(1, num_rows):
            len_i = smoothed_lengths[i]
            len_prev = smoothed_lengths[i-1]

            triggered = False
            if abs(len_i - len_prev) > t_abs:
                triggered = True
            elif len_prev == 0 and len_i > t_abs:
                triggered = True

            if triggered:
                split_idx = i
                if split_idx != band_boundaries[-1]:
                    band_boundaries.append(split_idx)

        band_boundaries.append(num_rows)
        unique_boundaries = sorted(list(set(band_boundaries)))
        
        bands = []
        small_bands_count = 0
        large_bands_details = []
        hdn_mask = self._hdn_mask if self._hdn_mask is not None else None
        ldn_mask = np.logical_not(hdn_mask) if hdn_mask is not None else None
        hdn_total = int(hdn_mask.sum()) if hdn_mask is not None else 0
        ldn_total = int(ldn_mask.sum()) if ldn_mask is not None else 0
        ldn_in_small = 0
        ldn_in_large = 0
        hdn_in_small = 0
        hdn_in_large = 0

        for i in range(len(unique_boundaries) - 1):
            start_row = unique_boundaries[i]
            end_row = unique_boundaries[i+1]
            length = end_row - start_row

            band_info = {'start_row': start_row, 'end_row': end_row, 'length': length}

            if length < t_band:
                small_bands_count += 1
                band_info['type'] = 'small'
                if ldn_mask is not None:
                    ldn_in_small += int(ldn_mask[start_row:end_row].sum())
                if hdn_mask is not None:
                    hdn_in_small += int(hdn_mask[start_row:end_row].sum())
            else:
                band_info['type'] = 'large'
                large_bands_details.append({
                    'start': start_row, 'end': end_row - 1, 'length': length
                })
                if ldn_mask is not None:
                    ldn_in_large += int(ldn_mask[start_row:end_row].sum())
                if hdn_mask is not None:
                    hdn_in_large += int(hdn_mask[start_row:end_row].sum())
            bands.append(band_info)

        # --- Print Statistics ---
        self._band_summary = {
            'total_bands': len(bands),
            'small_bands': small_bands_count,
            'large_bands': len(large_bands_details),
            'hdn_total': hdn_total,
            'hdn_small': hdn_in_small,
            'hdn_large': hdn_in_large,
            'ldn_total': ldn_total,
            'ldn_small': ldn_in_small,
            'ldn_large': ldn_in_large,
            'large_band_details': large_bands_details,
        }

        self._log("\n--- Band Statistics ---")
        self._log(f"Total number of bands: {len(bands)}")
        self._log(f"Number of small bands: {small_bands_count}")
        self._log(f"Number of large bands: {len(large_bands_details)}")
        if ldn_mask is not None and ldn_total > 0:
            small_share = (ldn_in_small / ldn_total) * 100 if ldn_total else 0.0
            large_share = (ldn_in_large / ldn_total) * 100 if ldn_total else 0.0
            self._log("\nLDN distribution across bands:")
            self._log(f"  - Small bands: {ldn_in_small} LDNs ({small_share:.2f}% of all LDNs)")
            self._log(f"  - Large bands: {ldn_in_large} LDNs ({large_share:.2f}% of all LDNs)")
        elif ldn_mask is not None:
            self._log("\nLDN distribution across bands:")
            self._log("  - No LDNs detected; all nodes are classified as HDNs.")

        if hdn_mask is not None and hdn_total > 0:
            h_small_share = (hdn_in_small / hdn_total) * 100 if hdn_total else 0.0
            h_large_share = (hdn_in_large / hdn_total) * 100 if hdn_total else 0.0
            self._log("\nHDN distribution across bands:")
            self._log(f"  - Small bands: {hdn_in_small} HDNs ({h_small_share:.2f}% of all HDNs)")
            self._log(f"  - Large bands: {hdn_in_large} HDNs ({h_large_share:.2f}% of all HDNs)")
        # if large_bands_details:
        #     self._log("\nDetails of large bands:")
        #     for band in large_bands_details:
        #         self._log(f"  - Node ID Range: [{band['start']}, {band['end']}], Length: {band['length']} rows")
        # self._log("--- End of Statistics ---\n")

        return bands

    def run(self, adj_matrix):
        """Main entry point to run the simulation."""
        self._log("Starting BSGCN simulation...")

        dataset_dir = self.dataset_stats.get('dataset_dir') if isinstance(self.dataset_stats, dict) else None
        if dataset_dir:
            self._log(f"Dataset directory: {dataset_dir}")

        if isinstance(self.dataset_stats, dict) and self.dataset_stats:
            nodes = self.dataset_stats.get('num_nodes')
            directed = self.dataset_stats.get('directed_edges')
            undirected = self.dataset_stats.get('undirected_edges')
            avg_degree = self.dataset_stats.get('avg_degree')
            summary = (
                f"Dataset summary -> nodes: {nodes}, directed edges: {directed}, "
                f"undirected edges: {undirected}"
            )
            if avg_degree is not None:
                try:
                    summary += f", avg degree: {float(avg_degree):.2f}"
                except (TypeError, ValueError):
                    summary += f", avg degree: {avg_degree}"
            self._log(summary)

        partition_cfg = self.config.BAND_PARTITION
        classification_cfg = self.config.BAND_CLASSIFICATION
        hdn_cfg = self.config.HDN_IDENTIFICATION
        arch_cfg = self.config.ARCHITECTURE
        smoothing_window = partition_cfg.get('SMOOTHING_WINDOW', 1)
        smoothing_mode = partition_cfg.get('SMOOTHING_MODE', 'mean')

        self._log("Key configuration parameters:")
        self._log(
            f"  - T_ABS: {partition_cfg['T_ABS']}, smoothing window: {smoothing_window} "
            f"(mode: {smoothing_mode})"
        )
        self._log(f"  - T_BAND: {classification_cfg['T_BAND']}")
        self._log(
            f"  - HDN Top-K per partition: {hdn_cfg['TOP_K']}, requested partitions: {hdn_cfg['NUM_PARTITIONS']}"
        )
        self._log(
            f"  - HDN-SPM size: {arch_cfg['HDN_SPM_KB']} KB, Global Cache size: {arch_cfg['GLOBAL_CACHE_KB']} KB"
        )
        self._log(
            f"  - SMB size: {arch_cfg['SMB_KB']} KB, Index CAM size: {arch_cfg['INDEX_CAM_KB']} KB"
        )
        self._log(
            f"  - Feature length: {arch_cfg['FEATURE_LENGTH']} (dtype bytes: {arch_cfg['FEATURE_DATA_TYPE_SIZE']}), "
            f"MAC array shape: {arch_cfg['MAC_ARRAY_SHAPE']}"
        )
        self._log(
            "  - Large band profiling rows: "
            f"{self.config.LARGE_BAND_PROCESSING.get('PROFILING_ROWS', 0)}, window shapes: "
            f"{self.config.LARGE_BAND_PROCESSING.get('WINDOW_SHAPES', [])}"
        )

        # Step 1: Optionally reorder the matrix using Metis partitions
        matrix, partition_boundaries, permutation = self._apply_metis_reordering(adj_matrix)
        self._permutation = permutation

        # Step 2: Identify HDNs on the (potentially reordered) matrix
        self._identify_hdns(matrix, partition_boundaries)

        # Step 3: Partition the matrix into bands
        bands = self._partition_into_bands(matrix)
        
        # Step 4: Iterate through bands and call the appropriate processor
        total_bands = len(bands)
        small_count = 0
        large_count = 0
        
        for i, band in enumerate(bands):
            if i % 10000 == 0:  # Progress update every 10k bands
                self._log(
                    f"Processing band {i}/{total_bands} ({i/total_bands*100:.1f}%)",
                    record=False,
                )
                
            if band['type'] == 'small':
                self._process_small_band(matrix, band)
                small_count += 1
            else:
                self._process_large_band(matrix, band)
                large_count += 1
        
        self._log(f"Processed {small_count} small bands and {large_count} large bands")

        self._log("\nSimulation complete. Final statistics will be reported.")

        return self.stats

    def write_log(self, stats_logger):
        """Persist a detailed log of the most recent run to disk."""
        if self.output_root is None or self.dataset_name is None:
            return None

        requested_parts = self.config.HDN_IDENTIFICATION.get('NUM_PARTITIONS', 0)
        actual_parts = self._partition_summary.get('actual_parts') if self._partition_summary else None
        if not actual_parts:
            actual_parts = requested_parts

        dataset_dir = self.output_root / str(self.dataset_name)
        partition_dir = dataset_dir / f"Metis_{actual_parts}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        log_path = partition_dir / "simulation.log"

        lines = []
        lines.append(f"Timestamp: {datetime.now().isoformat()}")
        lines.append(f"Dataset: {self.dataset_name}")
        dataset_dir_path = None
        if isinstance(self.dataset_stats, dict):
            dataset_dir_path = self.dataset_stats.get('dataset_dir')
        if dataset_dir_path:
            lines.append(f"Dataset directory: {dataset_dir_path}")

        if isinstance(self.dataset_stats, dict) and self.dataset_stats:
            lines.append("")
            lines.append("Dataset Overview:")
            nodes = self.dataset_stats.get('num_nodes')
            directed = self.dataset_stats.get('directed_edges')
            undirected = self.dataset_stats.get('undirected_edges')
            avg_degree = self.dataset_stats.get('avg_degree')
            if nodes is not None:
                lines.append(f"  Nodes: {nodes}")
            if directed is not None:
                lines.append(f"  Directed edges: {directed}")
            if undirected is not None:
                lines.append(f"  Undirected edges: {undirected}")
            if avg_degree is not None:
                try:
                    lines.append(f"  Average degree (undirected): {float(avg_degree):.2f}")
                except (TypeError, ValueError):
                    lines.append(f"  Average degree (undirected): {avg_degree}")

        lines.append("")
        lines.append("Configuration Parameters:")
        partition_cfg = self.config.BAND_PARTITION
        classification_cfg = self.config.BAND_CLASSIFICATION
        hdn_cfg = self.config.HDN_IDENTIFICATION
        arch_cfg = self.config.ARCHITECTURE
        smoothing_window = partition_cfg.get('SMOOTHING_WINDOW', 1)
        smoothing_mode = partition_cfg.get('SMOOTHING_MODE', 'mean')
        lines.append(f"  T_ABS: {partition_cfg['T_ABS']}")
        lines.append(f"  Smoothing window: {smoothing_window} (mode: {smoothing_mode})")
        lines.append(f"  T_BAND: {classification_cfg['T_BAND']}")
        lines.append(f"  HDN Top-K per partition: {hdn_cfg['TOP_K']}")
        lines.append(f"  Requested partitions: {hdn_cfg['NUM_PARTITIONS']}")
        lines.append(f"  HDN-SPM size: {arch_cfg['HDN_SPM_KB']} KB")
        lines.append(f"  Global Cache size: {arch_cfg['GLOBAL_CACHE_KB']} KB")
        lines.append(f"  SMB size: {arch_cfg['SMB_KB']} KB")
        lines.append(f"  Index CAM size: {arch_cfg['INDEX_CAM_KB']} KB")
        lines.append(f"  Feature length: {arch_cfg['FEATURE_LENGTH']}")
        lines.append(f"  Feature dtype bytes: {arch_cfg['FEATURE_DATA_TYPE_SIZE']}")
        lines.append(f"  MAC array shape: {arch_cfg['MAC_ARRAY_SHAPE']}")
        lines.append(
            "  Large band profiling rows: "
            f"{self.config.LARGE_BAND_PROCESSING.get('PROFILING_ROWS', 0)}"
        )
        lines.append(
            f"  Window shapes: {self.config.LARGE_BAND_PROCESSING.get('WINDOW_SHAPES', [])}"
        )

        lines.append("")
        lines.append("Partition Summary:")
        if self._partition_summary:
            strategy = self._partition_summary.get('strategy', 'unknown')
            lines.append(f"  Strategy: {strategy}")
            lines.append(f"  Requested partitions: {self._partition_summary.get('requested_parts')}")
            lines.append(f"  Actual partitions: {actual_parts}")
            partition_sizes = self._partition_summary.get('partition_sizes')
            if partition_sizes:
                lines.append(f"  Partition sizes: {partition_sizes}")
            note = self._partition_summary.get('note')
            if note:
                lines.append(f"  Note: {note}")
        else:
            lines.append("  (no partition summary recorded)")

        lines.append("")
        lines.append("Band Summary:")
        if self._band_summary:
            lines.append(f"  Total bands: {self._band_summary.get('total_bands')}")
            lines.append(f"  Small bands: {self._band_summary.get('small_bands')}")
            lines.append(f"  Large bands: {self._band_summary.get('large_bands')}")
            hdn_total = self._band_summary.get('hdn_total', 0)
            hdn_small = self._band_summary.get('hdn_small', 0)
            hdn_large = self._band_summary.get('hdn_large', 0)
            ldn_total = self._band_summary.get('ldn_total', 0)
            ldn_small = self._band_summary.get('ldn_small', 0)
            ldn_large = self._band_summary.get('ldn_large', 0)
            if hdn_total:
                h_small_share = (hdn_small / hdn_total) * 100 if hdn_total else 0.0
                h_large_share = (hdn_large / hdn_total) * 100 if hdn_total else 0.0
                lines.append(
                    f"  HDNs in small bands: {hdn_small} ({h_small_share:.2f}% of all HDNs)"
                )
                lines.append(
                    f"  HDNs in large bands: {hdn_large} ({h_large_share:.2f}% of all HDNs)"
                )
            if ldn_total:
                small_share = (ldn_small / ldn_total) * 100 if ldn_total else 0.0
                large_share = (ldn_large / ldn_total) * 100 if ldn_total else 0.0
                lines.append(
                    f"  LDNs in small bands: {ldn_small} ({small_share:.2f}% of all LDNs)"
                )
                lines.append(
                    f"  LDNs in large bands: {ldn_large} ({large_share:.2f}% of all LDNs)"
                )
            else:
                lines.append("  LDNs: none detected (all nodes classified as HDNs)")
            large_details = self._band_summary.get('large_band_details') or []
            if large_details:
                lines.append("  Large band details:")
                for band in large_details:
                    lines.append(
                        f"    - Rows [{band['start']}, {band['end']}], length: {band['length']}"
                    )
        else:
            lines.append("  (no band summary recorded)")

        if self._log_messages:
            lines.append("")
            lines.append("Runtime log:")
            lines.extend(self._log_messages)

        if stats_logger is not None:
            lines.append("")
            lines.extend(stats_logger.generate_report_lines())

        with log_path.open('w', encoding='utf-8') as handle:
            handle.write("\n".join(str(line) for line in lines) + "\n")

        return log_path