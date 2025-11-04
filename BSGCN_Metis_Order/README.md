# BSGCN_Metis_Order

基于Metis图分割的BSGCN（Band-based Sparse Graph Convolutional Network）模拟器实现。

## 项目概述

本项目实现了一个高效的图神经网络加速器模拟器，使用Metis图分割算法进行图重排序，并通过自适应窗口策略优化稀疏矩阵计算。

## 主要特性

- **Metis图分割**: 使用Metis算法对图进行分区和重排序
- **自适应窗口策略**: 根据带状结构动态调整计算窗口
- **硬件模拟**: 模拟MAC阵列、内存层次结构等硬件组件
- **可视化工具**: 提供带状结构和重排序结果的可视化

## 文件结构

- `config.py` - 配置参数（硬件架构、算法阈值等）
- `simulator.py` - 主模拟器实现
- `components.py` - 硬件组件（MAC阵列、内存等）
- `data_loader.py` - 图数据加载器
- `run_simulation.py` - 运行脚本
- `stats_logger.py` - 性能统计和日志记录
- `visualize_bands.py` - 带状结构可视化
- `visualize_reordering.py` - 重排序结果可视化

## 使用方法

1. **配置参数**:
   ```python
   # 编辑 config.py 中的参数
   BAND_PARTITION = {'T_ABS': 37, 'T_REL': 8}
   ARCHITECTURE = {'MAC_ARRAY_SHAPE': (1, 16), ...}
   ```

2. **运行模拟**:
   ```bash
   python run_simulation.py
   ```

3. **可视化结果**:
   ```bash
   python visualize_bands.py
   python visualize_reordering.py
   ```

## 配置说明

### 带状分割参数
- `T_ABS`: 绝对差异阈值
- `T_REL`: 相对差异阈值
- `T_BAND`: 带状分类阈值

### 硬件架构参数
- `MAC_ARRAY_SHAPE`: MAC阵列形状
- `CLOCK_FREQ_GHZ`: 时钟频率
- `FEATURE_LENGTH`: 特征维度
- 各种内存大小配置

## 研究背景

本项目研究图神经网络在硬件加速器上的高效实现，特别关注：
- 稀疏矩阵的带状结构优化
- 内存访问模式优化
- 并行计算策略
- 硬件资源利用率提升

## 依赖要求

- Python 3.7+
- NumPy
- SciPy
- NetworkX
- Matplotlib

## 许可证

仅供学术研究使用。
