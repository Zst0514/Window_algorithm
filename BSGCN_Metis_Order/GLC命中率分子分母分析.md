# BSGCN_Metis_Order GLC 命中率分子和分母分析

## 核心结论

**GLC 命中率的分子和分母**：
- **分子**：`self.glc_ldn_hits` （大带执行阶段，LDN 访问 GLC 的命中次数）
- **分母**：`self.glc_ldn_accesses` （大带执行阶段，LDN 访问 GLC 的总访问次数）

## 统计逻辑详解

### 1. 统计范围

#### ❌ 小带（Small Bands）不统计 GLC
```python:152-197:BSGCN_Metis_Order/simulator.py
def _process_small_band(self, matrix, band):
    ...
    for row_idx in range(start, end):
        for nnz_ptr in range(row_start, row_end):
            neighbor_idx = indices[nnz_ptr]
            if hdn_mask is not None and hdn_mask[neighbor_idx]:
                # HDN: 访问 HDN-SPM
                is_hit = spm_access(neighbor_idx, row_idx)
                logger.log_hdn_spm_access(is_hit)
            else:
                # LDN: 直接访问 DRAM，跳过了 GLC！
                log_spm_access(False)
                log_dram(dram_read_bytes)
                log_cycles(100, 'dram_access')
```

**关键问题**：小带中的 LDN **直接访问 DRAM**，**完全跳过 GLC**！

#### ✅ 大带（Large Bands）才统计 GLC

大带处理分为两个阶段：

##### 阶段1：Profiling（不计入统计）
```python:308:322:BSGCN_Metis_Order/simulator.py
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
        record_ldn_access=False,  # ← 关键：不记录
    )
```

##### 阶段2：实际执行（计入统计）
```python:315:323:BSGCN_Metis_Order/simulator.py
self._simulate_window_pass(
    matrix,
    band,
    best_shape,
    self.stats,
    hdn_spm=self.hdn_spm,
    global_cache=self.global_cache,
    record_ldn_access=True,  # ← 关键：记录 LDN 访问
)
```

### 2. 统计计数器

```python:14-18:BSGCN_Metis_Order/stats_logger.py
self.global_cache_hits = 0
self.global_cache_misses = 0
# Detailed accounting for large-band LDN activity
self.glc_ldn_accesses = 0  # ← 大带中 LDN 访问 GLC 的总次数
self.glc_ldn_hits = 0      # ← 大带中 LDN 访问 GLC 的命中次数
```

### 3. 记录函数

```python:38:57:BSGCN_Metis_Order/stats_logger.py
def log_global_cache_access(self, is_hit, *, count_as_ldn=False):
    """Logs Global Cache access (hit or miss).
    
    Parameters
    ----------
    is_hit: Whether the access resulted in a cache hit.
    count_as_ldn: When ``True``, the access is attributed to an LDN lookup
    """
    if is_hit:
        self.global_cache_hits += 1
    else:
        self.global_cache_misses += 1
    if count_as_ldn:  # ← 只有大带执行阶段才会调用
        self.glc_ldn_accesses += 1
        if is_hit:
            self.glc_ldn_hits += 1
```

### 4. 报告生成

```python:87:99:BSGCN_Metis_Order/stats_logger.py
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
        lines.append(f"    (LDN accesses in large bands: {self.glc_ldn_accesses})")
```

## 为什么 Nell 数据集中只有 76 次访问？

### 用户观察到的情况

```
--- Band Statistics ---
Total number of bands: 4354
Number of small bands: 4289
Number of large bands: 65

LDN distribution across bands:
  - Small bands: 40762 LDNs (66.11% of all LDNs)
  - Large bands: 20897 LDNs (33.89% of all LDNs)
  
Global Cache: 13/76 hits (17.11% hit rate)
```

### 原因分析

**GLC 只统计大带执行阶段的 LDN neighbor 访问**：

1. **分母是"访问次数"，不是"节点数"**
   - 76 次是 `neighbor_idx` 的访问次数
   - 不是 20897（大带中 LDN 的行数）

2. **为什么只有 76 次？**
   - 虽然大带中有 20897 行 LDN
   - 但这些 LDN 的邻居节点（neighbor）大部分是 **HDN**
   - 只有 **少数邻居是 LDN** 且需要访问 GLC
   - 具体逻辑：
     ```python
     for neighbor_idx in column_indices:
         if hdn_mask[neighbor_idx]:
             # 访问 HDN-SPM（不计入 GLC）
         else:
             # 访问 GLC（计入 glc_ldn_accesses）
             logger.log_global_cache_access(is_hit, count_as_ldn=True)
     ```

3. **Nell 数据集的特性**
   - 平均度数：3.83（非常稀疏）
   - 大多数边的起点（row）或终点（neighbor）都是 HDN
   - 导致大带中 LDN 访问其他 LDN 的情况很少

### 具体例子

假设大带中有 1000 个 LDN 节点：
- 每个节点的平均度数为 4
- 有 4000 条边
- 但其中：
  - 3500 条边指向 HDN → 访问 HDN-SPM
  - 400 条边指向其他分区的 LDN → 直接访问 DRAM
  - **只有 100 条边指向同一大带的 LDN → 访问 GLC**

这就是为什么 20897 个 LDN 节点，但只有 76 次 GLC 访问的原因！

## 总结

### 分子和分母的定义

| 项目 | 值 | 说明 |
|------|-----|------|
| **分母** | `glc_ldn_accesses` | 大带执行阶段，LDN 访问其他 LDN 且需要查 GLC 的次数 |
| **分子** | `glc_ldn_hits` | 上述访问中，在 GLC 中命中的次数 |
| **排除** | 小带的所有访问 | 小带中 LDN 直接访问 DRAM，不经过 GLC |
| **排除** | Profiling 阶段的访问 | Profiling 阶段不记录统计 |

### 为什么数值很小？

1. 小带（66% LDN）的访问完全不经过 GLC
2. 大带中 LDN 的邻居大多是 HDN（走 HDN-SPM）
3. 大带中 LDN 访问其他 LDN 的情况本身就很少
4. 稀疏图的连通性导致同质邻居少

**这符合设计**：GLC 只服务于大带中 LDN→LDN 的访问模式。

