"""
Models the hardware components of the BSGCN accelerator.
"""
from collections import OrderedDict

class Cache:
    """
    A more detailed cache model supporting capacity in bytes and a RidxLRU replacement policy.
    """
    def __init__(self, name, size_kb, feature_len, feature_dtype_size):
        self.name = name
        self.size_kb = size_kb
        self.feature_len = feature_len
        self.feature_dtype_size = feature_dtype_size
        self.capacity_bytes = size_kb * 1024
        self.feature_vector_size = feature_len * feature_dtype_size
        
        if self.feature_vector_size == 0:
            self.num_entries_capacity = 0
        else:
            self.num_entries_capacity = self.capacity_bytes // self.feature_vector_size
        
        # For RidxLRU, this will store: {node_id: last_used_row_idx}
        self.cache_data = OrderedDict()

        self.hits = 0
        self.misses = 0

    def access(self, node_id, current_row_idx):
        """
        Accesses the cache. Returns True for a hit, False for a miss.
        Implements the RidxLRU policy on update/insertion.
        """
        if node_id in self.cache_data:
            # Hit
            self.hits += 1
            # Update the last used row index for RidxLRU
            self.cache_data[node_id] = current_row_idx
            return True
        else:
            # Miss
            self.misses += 1
            if len(self.cache_data) >= self.num_entries_capacity:
                self._evict()
            # Add new item
            self.cache_data[node_id] = current_row_idx
            return False

    def clone(self):
        """Create a deep copy of the cache, preserving its configuration and state."""
        cloned = Cache(self.name, self.size_kb, self.feature_len, self.feature_dtype_size)
        cloned.cache_data = OrderedDict(self.cache_data)
        cloned.hits = self.hits
        cloned.misses = self.misses
        return cloned

    def _evict(self):
        """
        Evicts an item based on RidxLRU policy:
        Evict the entry with the smallest `last_used_row_idx`.
        """
        if not self.cache_data:
            return
        
        # Find the item with the minimum associated row index
        try:
            evict_candidate_id = min(self.cache_data, key=self.cache_data.get)
            del self.cache_data[evict_candidate_id]
        except ValueError:
            # Cache is empty, nothing to evict
            pass

    def get_stats(self):
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0
        return {
            'name': self.name,
            'hits': self.hits,
            'misses': self.misses,
            'total_accesses': total_accesses,
            'hit_rate': f"{hit_rate:.2%}"
        }

class IndexCAM:
    """A model for the Index Content Addressable Memory."""
    def __init__(self, name, size_kb):
        self.name = name
        self.size_kb = size_kb
        self.hits = 0
        self.misses = 0

    def lookup(self, node_id):
        # Placeholder for CAM lookup logic.
        pass

    def get_stats(self):
        return {
            'name': self.name,
            'hits': self.hits,
            'misses': self.misses,
            'total_accesses': self.hits + self.misses
        }