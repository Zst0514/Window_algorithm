# GLC 访问次数 vs LDN 行数的关系

## 用户的困惑

查看 collab 数据集的结果：
```
LDN distribution across bands:
  - Small bands: 3760 LDNs (1.22% of all LDNs)
  - Large bands: 303179 LDNs (98.78% of all LDNs)
  
Global Cache: 9028187/9808557 hits (92.04% hit rate)
```

**问题**：为什么 GLC 访问了 980万次，但大带中只有 30万 LDN？

## 错误的类比

很多用户会错误地认为：
- ❌ **错误理解**：GLC 访问次数 ≈ LDN 行数
- ✅ **正确理解**：GLC 访问次数 = LDN neighbor 的访问次数（每条边都计数）

## 正确的关系

### 303179 的含义

**303179 = 大带中作为 row index 的 LDN 节点数**

这是行数（nodes），不是访问次数！

### 9808557 的含义

**9808557 = 大带执行阶段，访问所有 LDN neighbor 的总次数**

这是访问次数（edge traversals），等于所有边对 LDN neighbor 的遍历次数。

## 为什么差距这么大？

### Collab 数据集的特性

| 指标 | 值 |
|------|-----|
| 总节点数 | 372,475 |
| 总边数（undirected） | 12,286,079 |
| 平均度数 | 65.97 |
| 大带中 LDN 行数 | 303,179 |
| GLC 访问次数 | 9,808,557 |

### 计算说明

在大带中，LDN 行处理时的访问模式：

```
对于每个 LDN row i:
    遍历其所有 neighbor j:
        if j 是 HDN:
            → 访问 HDN-SPM（不计入 GLC）
        if j 是 LDN:
            → 访问 GLC（计入 glc_ldn_accesses）
```

### 数值估计

假设 303179 个 LDN 行，平均度数为 65.97：

- **理论最大访问次数**：303179 × 65.97 ≈ **20,000,000 次**
  - 但实际只有 **9,808,557 次**
  - 差异来自：这些 LDN 的 neighbor 中有部分是 HDN（走 HDN-SPM 路径）

- **GLC 访问占比**：9,808,557 / 20,000,000 ≈ **49%**
  - 说明大带中 LDN 的邻居约有一半是 LDN（走 GLC）
  - 另一半是 HDN（走 HDN-SPM）

## 详细对比

### 正确的理解

| 维度 | 303,179 | 9,808,557 |
|------|---------|-----------|
| **单位** | 行数 (rows) | 访问次数 (accesses) |
| **含义** | 大带中 LDN 节点的数量 | 大带执行时访问 LDN neighbor 的总次数 |
| **层级** | Node level | Edge level |
| **关系** | 一个 LDN 节点 | 该 LDN 的多个邻居（边） |

### 类比说明

想象一个城市：
- **303179** = 大带中的 LDN "居民"人数
- **9,808,557** = 这些居民访问其他 LDN 邻居的**总次数**

如果每个 LDN 平均有 65 个邻居，那么：
- 总邻居访问次数：303179 × 65 ≈ 1,970万次
- 其中约 980万次是访问 LDN 邻居（走 GLC）
- 其余是访问 HDN 邻居（走 HDN-SPM）

## 与其他数据集的对比

### Nell 数据集

```
LDN distribution across bands:
  - Large bands: 20897 LDNs
Global Cache: 13/76 hits (17.11% hit rate)
```

- GLC 访问次数：76
- 平均每个 LDN：76 / 20897 ≈ 0.0036 次

**为什么这么少？**
- Nell 非常稀疏（平均度数 3.83）
- LDN 的邻居大部分是 HDN
- 导致 LDN→LDN 的边很少

### Collab 数据集

```
LDN distribution across bands:
  - Large bands: 303179 LDNs
Global Cache: 9028187/9808557 hits (92.04% hit rate)
```

- GLC 访问次数：9,808,557
- 平均每个 LDN：9,808,557 / 303,179 ≈ 32.4 次

**为什么这么多？**
- Collab 更稠密（平均度数 65.97）
- LDN 的邻居中约有一半是其他 LDN
- 导致大量 LDN→LDN 的边需要访问 GLC

## 总结

### 关键点

1. **GLC 访问次数 ≠ LDN 行数**
   - 前者统计的是**边**（neighbor 访问）
   - 后者统计的是**节点**（row count）

2. **合理的数量级**
   - GLC 访问次数 = LDN 行数 × 平均度数 × LDN neighbor 比例
   - Collab：303179 × 65.97 × 0.5 ≈ 9,800,000 ✅

3. **稀疏图 vs 稠密图**
   - Nell（稀疏）：GLC 访问很少
   - Collab（稠密）：GLC 访问多

### 公式

```
GLC 访问次数 = Σ(大带中所有边的 LDN neighbor 访问)
             = Σ_{LDN row i} (邻居中 LDN 的个数)
             = 大带中 LDN→LDN 边的总数

大带中 LDN 行数 = 作为 row index 的 LDN 节点数
                 = 大带中 LDN 节点的个数
                 
显然：边数 >> 节点数
```

**因此，980万 > 30万是完全正常和合理的！**

