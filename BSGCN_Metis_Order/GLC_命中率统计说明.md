# BSGCN_Metis_Order GLC 命中率统计说明

## 概述
Global Cache (GLC) 是 BSGCN 架构中专门为 Low-Degree Nodes (LDN) 设计的缓存，采用 RidxLRU 替换策略。

## 关键代码位置

### 1. GLC 访问逻辑 (`simulator.py` 第 193-202 行)
```python
else:
    # Record the failed HDN-SPM lookup before consulting the GLC
    logger.log_hdn_spm_access(False)
    is_hit = self.global_cache.access(neighbor_idx, row_idx)
    logger.log_global_cache_access(is_hit)
    logger.log_cycles(1, 'glc_access')
    
    if not is_hit:
        logger.log_dram_read(dram_read_bytes)
        logger.log_cycles(100, 'dram_access')
```

### 2. Cache 类实现 (`components.py` 第 6-71 行)
- **容量计算**：基于配置的 KB 大小和特征向量大小
- **数据结构**：`OrderedDict` 存储 `{node_id: last_used_row_idx}`
- **替换策略**：RidxLRU (Row-indexed LRU)
  - 命中时更新 `last_used_row_idx`
  - Miss 时替换 `last_used_row_idx` 最小的条目

### 3. 统计和报告 (`stats_logger.py` 第 69-79 行)
```python
# Global Cache hit rate (hits / LDN accesses only)
total_glc_accesses = self.global_cache_hits + self.global_cache_misses
if total_glc_accesses > 0:
    glc_hit_rate = (self.global_cache_hits / total_glc_accesses) * 100
    print(f"  - Global Cache: {self.global_cache_hits}/{total_glc_accesses} hits ({glc_hit_rate:.2f}% hit rate)")
```

## 配置参数

### GLC 大小
- **位置**：`config.py` 第 51 行
- **默认值**：`'GLOBAL_CACHE_KB': 64` (64 KB)
- **计算**：`num_entries = (64 * 1024) / (16 * 8) = 512` 个节点

### RidxLRU 策略
- **原理**：根据节点最后使用的行索引进行替换
- **优势**：考虑空间局部性（相邻行更可能在短时间内访问相同的邻居节点）

## 统计逻辑总结

1. **访问分类**：
   - **HDN** → HDN-SPM (不在 GLC 统计中)
   - **LDN** → 先查 GLC → Miss 则从 DRAM 读取

2. **命中率计算**：
   - 只统计 LDN 对 GLC 的访问
   - `GLC Hit Rate = global_cache_hits / (global_cache_hits + global_cache_misses)`

3. **内存层次**：
   - **HDN-SPM** (512 KB) → HDNs
   - **GLC** (64 KB) → LDNs
   - **DRAM** → Cache Miss 的数据源

## 主要特点

- ✅ 只统计 LDN 访问（不包括 HDN）
- ✅ 使用 RidxLRU 替换策略
- ✅ 命中/未命中准确统计
- ✅ 与 HDN-SPM 命中率分开统计和报告

