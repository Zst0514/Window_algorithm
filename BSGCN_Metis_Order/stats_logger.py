"""
Collects and logs statistics during the simulation run.
"""

class StatsLogger:
    def __init__(self):
        self.total_cycles = 0
        self.dram_bytes_read = 0
        self.breakdown = {}

        # Cache statistics
        self.hdn_spm_hits = 0
        self.hdn_spm_misses = 0
        self.global_cache_hits = 0
        self.global_cache_misses = 0
        # Detailed accounting for large-band LDN activity
        self.glc_ldn_accesses = 0
        self.glc_ldn_hits = 0

    def log_cycles(self, cycles, category):
        """Logs cycles and adds them to a specific category."""
        self.total_cycles += cycles
        if category not in self.breakdown:
            self.breakdown[category] = 0
        self.breakdown[category] += cycles

    def log_dram_read(self, num_bytes):
        """Logs the number of bytes read from DRAM."""
        self.dram_bytes_read += num_bytes

    def log_hdn_spm_access(self, is_hit):
        """Logs HDN-SPM access (hit or miss)."""
        if is_hit:
            self.hdn_spm_hits += 1
        else:
            self.hdn_spm_misses += 1

    def log_global_cache_access(self, is_hit, *, count_as_ldn=False):
        """Logs Global Cache access (hit or miss).

        Parameters
        ----------
        is_hit:
            Whether the access resulted in a cache hit.
        count_as_ldn:
            When ``True``, the access is attributed to an LDN lookup in a large
            band, which is the quantity of interest for the simulator's cache
            statistics.
        """
        if is_hit:
            self.global_cache_hits += 1
        else:
            self.global_cache_misses += 1
        if count_as_ldn:
            self.glc_ldn_accesses += 1
            if is_hit:
                self.glc_ldn_hits += 1

    def generate_report_lines(self):
        """Return the formatted report as a list of strings."""
        lines = []
        lines.append("--- Simulation Statistics Report ---")
        lines.append(f"Total Execution Cycles: {self.total_cycles}")
        lines.append(f"Total DRAM Data Read: {self.dram_bytes_read / (1024**2):.2f} MB")
        lines.append("")

        if self.total_cycles > 0:
            lines.append("Cycle Breakdown:")
            for category, cycles in self.breakdown.items():
                percentage = (cycles / self.total_cycles) * 100 if self.total_cycles else 0.0
                lines.append(f"  - {category}: {cycles} cycles ({percentage:.2f}%)")
            lines.append("")

        lines.append("Cache Performance:")

        total_hdn_spm_accesses = self.hdn_spm_hits + self.hdn_spm_misses
        if total_hdn_spm_accesses > 0:
            hdn_spm_hit_rate = (self.hdn_spm_hits / total_hdn_spm_accesses) * 100
            lines.append(
                "  - HDN-SPM: "
                f"{self.hdn_spm_hits}/{total_hdn_spm_accesses} hits "
                f"({hdn_spm_hit_rate:.2f}% hit rate)"
            )
        else:
            lines.append("  - HDN-SPM: No accesses recorded")

        total_glc_accesses = self.glc_ldn_accesses or (self.global_cache_hits + self.global_cache_misses)
        if total_glc_accesses > 0:
            hits = self.glc_ldn_hits if self.glc_ldn_accesses else self.global_cache_hits
            glc_hit_rate = (hits / total_glc_accesses) * 100
            lines.append(
                "  - Global Cache: "
                f"{hits}/{total_glc_accesses} hits "
                f"({glc_hit_rate:.2f}% hit rate)"
            )
            if self.glc_ldn_accesses:
                lines.append(
                    f"    (LDN accesses in large bands: {self.glc_ldn_accesses})"
                )
        else:
            lines.append("  - Global Cache: No LDN accesses recorded")

        lines.append("--- End of Report ---")
        return lines

    def report(self):
        """Prints a final report of all collected statistics."""
        lines = self.generate_report_lines()
        print()
        for line in lines:
            print(line)
        print()