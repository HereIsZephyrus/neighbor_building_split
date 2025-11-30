# 建筑模式分割与LCZ分类系统

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | 中文文档

基于**图注意力网络（GAT）**和**谱聚类**的两阶段城市建筑分类系统，用于本地气候区（LCZ）分类。本项目通过维诺图生成建筑空间邻接关系，然后应用深度学习实现准确且空间一致的建筑分类。

## 概述

城市建筑分类对气候研究、城市规划和能源建模至关重要。本项目解决了将建筑分类为本地气候区（LCZ1-9）的挑战，同时保持空间一致性——相邻的相似建筑应属于同一类别。

### 两阶段方法

```
第一阶段：GAT分类                    第二阶段：谱聚类
┌─────────────────────────┐         ┌─────────────────────────────────┐
│ 判别性特征               │         │ 形态学特征                       │
│ (高度、反射率等)          │ ──────► │ (面积、形状、朝向)                │
│                         │         │                                 │
│ "这是什么类型的建筑？"    │         │ "哪些建筑应该归为一组？"           │
└─────────────────────────┘         └─────────────────────────────────┘
           │                                       │
           ▼                                       ▼
       初步预测                            空间一致的最终标签
```

**核心创新**：任务解耦——GAT专注于分类准确性，而谱聚类通过置信度加权多数投票确保空间连续性。

## 功能特性

- **基于维诺图的邻接关系**：通过形态学膨胀生成建筑邻接关系
- **距离感知GAT**：边特征编码空间距离的图注意力网络
- **两阶段分类**：结合判别性分类和空间平滑
- **置信度加权投票**：高置信度的GAT预测对聚类标签有更大影响
- **MPI并行处理**：支持大规模数据集的分布式处理
- **LCZ相似度损失**：感知LCZ类别语义关系的自定义损失函数
- **连通分量处理**：自动分离和处理断开的建筑组

## 安装

### 环境要求

- Python 3.10+
- CUDA 12.0+（GPU加速）
- MPI（可选，用于并行处理）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/your-repo/neighbor_building_split.git
cd neighbor_building_split

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或: venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装PyTorch Geometric（根据CUDA版本调整）
pip install torch-geometric torch-scatter torch-sparse
```

### 环境配置

在项目根目录创建 `.env` 文件：

```env
DISTRICT=/path/to/districts.shp
BUILDINGS=/path/to/buildings.shp
OUTPUT_DIR=/path/to/output
```

## 快速开始

### 1. 生成维诺图和邻接矩阵

```bash
# 单线程处理
python -m src.extractor \
    --generate-voronoi-diagram \
    --district-path /path/to/districts.shp

# MPI并行处理（8进程）
mpirun -n 8 python -m src.extractor \
    --generate-voronoi-diagram \
    --district-path /path/to/districts.shp \
    --use-mpi
```

### 2. 训练GAT模型

```bash
# 交叉验证训练
python -m src.gat --train \
    --adjacency-dir output/voronoi \
    --sample-buildings /path/to/buildings.shp \
    --sample-districts /path/to/districts.shp \
    --output-root-dir output/gat \
    --config src/gat/training_config.yaml \
    --mode cv

# 最终模型训练（使用全部数据）
python -m src.gat --train \
    --adjacency-dir output/voronoi \
    --sample-buildings /path/to/buildings.shp \
    --sample-districts /path/to/districts.shp \
    --output-root-dir output/gat \
    --mode final
```

### 3. 运行推理

```bash
python -m src.gat --inference \
    --model-path output/gat/models/final_model.pth \
    --adjacency-dir output/voronoi \
    --building-path /path/to/buildings.shp \
    --output-root-dir output/predictions
```

## 模块概述

### Extractor模块 (`src/extractor/`)

从建筑足迹生成维诺图并计算空间邻接矩阵。

**工作流程：**
1. 加载区县和建筑shapefile
2. 在每个区县内栅格化建筑足迹
3. 通过形态学膨胀生成维诺图分区
4. 矢量化维诺图多边形
5. 计算邻接矩阵（相邻建筑间距离）

**主要组件：**
- `VoronoiGenerator`：基于形态学膨胀的维诺图生成
- `Rasterizer`：建筑足迹栅格化
- `ShapefileReader`：空间数据加载和筛选

**输出文件：**
- `district_{id}_voronoi.shp`：维诺图多边形shapefile
- `district_{id}_adjacency.pkl`：建筑邻接矩阵（距离）

### GAT模块 (`src/gat/`)

用于建筑分类的图注意力网络，带谱聚类后处理。

**训练流程：**
1. 加载建筑特征和邻接矩阵
2. 构建图：节点（建筑）和边（邻接关系）
3. 使用相似度感知损失训练GAT模型
4. K折交叉验证进行超参数调优

**推理流程：**
1. 加载训练好的模型和建筑数据
2. GAT前向传播 → 嵌入向量 + 初步预测
3. 提取聚类特征（形态学）
4. 对每个连通分量执行谱聚类
5. 聚类内置信度加权多数投票
6. 输出最终空间一致的预测结果

**主要组件：**
- `EdgeConvLayer`：距离感知图注意力层
- `GAT`：多层图注意力网络
- `spectral_clustering.py`：两阶段聚类管道
- `SimilarityAwareCrossEntropyLoss`：LCZ感知损失函数

## 配置说明

### 训练配置 (`src/gat/training_config.yaml`)

```yaml
model:
  hidden_dim: 32        # 隐藏层维度
  num_layers: 3         # GAT层数
  num_heads: 8          # 注意力头数
  dropout: 0.6          # Dropout比例

training:
  epochs: 2000          # 训练轮数
  lr: 0.005            # 学习率
  patience: 120        # 早停耐心值
  k_fold: 5            # 交叉验证折数
  lambda_smooth: 0.3   # 平滑损失权重

spectral_clustering:
  embedding_weight: 0.6              # 嵌入权重
  feature_weight: 0.2                # 特征权重
  distance_weight: 0.2               # 距离权重
  min_cluster_size: 5                # 最小聚类大小
  use_confidence_weighted_voting: true  # 置信度加权投票

similarity_loss:
  enabled: true        # 启用相似度损失
  temperature: 0.05    # 温度参数
```

### 特征配置 (`src/gat/features_config.yaml`)

```yaml
# GAT分类特征（判别性）
gat_features:
  - height      # 建筑高度
  - albedo      # 反射率
  - hwratio     # 高宽比
  - area        # 面积

# 谱聚类特征（形态学）
clustering_features:
  - height      # 高度
  - area        # 面积
  - perimeter   # 周长
  - orientatio  # 朝向
  - elongation  # 伸长率
  - concavity   # 凹度
  - circularit  # 圆度
```

## 项目结构

```
neighbor_building_split/
├── src/
│   ├── extractor/              # 维诺图和邻接模块
│   │   ├── __main__.py         # CLI入口
│   │   ├── processor.py        # 区县处理协调器
│   │   ├── converter/
│   │   │   ├── voronoi_generator.py
│   │   │   └── rasterizer.py
│   │   ├── reader/
│   │   │   └── shapefile_reader.py
│   │   └── utils/
│   │       ├── adjacency.py
│   │       └── config.py
│   │
│   └── gat/                    # GAT分类模块
│       ├── __main__.py         # CLI入口
│       ├── train.py            # 训练协调器
│       ├── inference.py        # 推理管道
│       ├── models/
│       │   ├── gat.py          # GAT模型
│       │   └── gat_layer.py    # EdgeConv注意力层
│       ├── training/
│       │   ├── trainer.py
│       │   ├── similarity_loss.py
│       │   └── config.py
│       ├── utils/
│       │   ├── spectral_clustering.py
│       │   └── feature_extractor.py
│       ├── training_config.yaml
│       └── features_config.yaml
│
├── docs/                       # 文档（中文）
├── scripts/                    # 实用脚本
├── requirements.txt
└── README.md
```

## 数据要求

### 建筑Shapefile

必需属性：
| 字段 | 描述 | 使用模块 |
|------|------|---------|
| `id` | 建筑唯一标识 | 两者 |
| `height` | 建筑高度（米） | GAT、聚类 |
| `albedo` | 表面反射率 | GAT |
| `hwratio` | 高宽比 | GAT |
| `area` | 占地面积（m²） | GAT、聚类 |
| `perimeter` | 周长（米） | 聚类 |
| `orientatio` | 朝向（度） | 聚类 |
| `lcz` | 真实标签（仅训练） | 训练 |

### 区县Shapefile

必需属性：
| 字段 | 描述 |
|------|------|
| `FID` 或 `fid` | 区县标识 |
| `geometry` | 区县多边形 |

## 高级用法

### MPI并行训练

```bash
mpirun -n 8 python -m src.gat --train \
    --adjacency-dir output/voronoi \
    --sample-buildings buildings.shp \
    --sample-districts districts.shp \
    --output-root-dir output/gat \
    --mode cv
```

### 恢复训练

```bash
python -m src.gat --train \
    --resume output/gat/checkpoints/checkpoint_epoch_100.pth \
    ...
```

### 自定义聚类缩放器

```bash
python -m src.gat --inference \
    --model-path model.pth \
    --clustering-scaler-path custom_scaler.pkl \
    ...
```

## 输出文件

### 训练输出

```
output/gat/
├── models/
│   ├── best_model.pth          # 最佳验证模型
│   └── final_model.pth         # 最终训练模型
├── logs/
│   └── training.log
└── runs/                       # TensorBoard日志
```

### 推理输出

```
output/predictions/
├── district_{id}_embeddings.pkl        # GAT嵌入
├── district_{id}_building_predictions.gpkg   # 建筑预测
├── district_{id}_voronoi_predictions.gpkg    # 维诺图预测
└── embeddings_summary.pkl              # 汇总统计
```

## 常见问题

### 问题排查

1. **CUDA内存不足**
   - 减小训练配置中的 `batch_size`
   - 设置 `node_threshold` 对大图使用小批量采样

2. **找不到建筑**
   - 检查建筑和区县shapefile的坐标系一致性
   - 验证数据集之间的空间交叉

3. **分类准确率低**
   - 调整 `similarity_loss.temperature`（更低 = 更少平滑）
   - 调优谱聚类权重（`embedding_weight`、`feature_weight`）
   - 增加训练 `epochs` 或调整 `patience`

## 引用

如果您在研究中使用本项目，请引用：

```bibtex
@software{building_lcz_classification,
  title = {Building Pattern Segmentation and LCZ Classification},
  year = {2025},
  url = {https://github.com/your-repo/neighbor_building_split}
}
```

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 详细文档

中文详细文档：
- [算法思想](docs/算法思想.md) - 算法概念和理论基础
- [设计细节](docs/设计细节.md) - 架构和设计决策
- [实现方法](docs/实现方法.md) - 实现指南

