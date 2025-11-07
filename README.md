# BGC-DETR: 基于DETR架构的生物基因簇检测模型

## 项目简介

BGC-DETR是一个基于DETR（DEtection TRansformer）架构的生物基因簇（BGC, Biosynthetic Gene Cluster）检测模型。该项目利用深度学习技术对生物序列进行端到端的基因簇检测，能够识别和定位不同类型的生物合成基因簇。

### 主要特性

- **端到端检测**: 基于DETR架构，无需复杂的后处理步骤
- **多类别支持**: 支持NRPS、PKS、terpene、saccharide、ribosomal、other等多种BGC类型
- **序列级处理**: 直接处理蛋白质序列，无需预先的特征工程
- **分布式训练**: 支持多GPU分布式训练
- **5折交叉验证**: 提供完整的交叉验证流程

## 项目架构

### 核心组件

```
BGC-DETR/
├── models/                 # 模型定义
│   ├── detr.py            # DETR主模型
│   ├── transformer.py     # Transformer架构
│   ├── backbone.py        # 骨干网络
│   ├── matcher.py         # 匈牙利匹配器
│   └── position_encoding.py # 位置编码
├── data.py                # 数据集处理
├── engine.py              # 训练和评估引擎
├── main_by_cluster.py     # 主训练脚本
├── fasta_to_embeddings_simple.py # 序列嵌入提取
├── validate_final.py      # 模型验证脚本
├── scripts/               # 工具脚本
│   ├── example_usage.py   # 使用示例
│   ├── parse_gbk.py       # GenBank文件解析
│   └── process_genomes.py # 基因组处理
└── requirements.txt       # 依赖包
```

### 模型架构

1. **Backbone网络**: 处理预计算的蛋白质嵌入
2. **Transformer**: 编码器-解码器架构学习序列表示
3. **DETR模型**: 整合backbone和transformer，进行目标检测
4. **损失函数**: 包含分类、边界框、GIoU等多种损失
5. **后处理器**: 将模型输出转换为最终预测结果

## 环境要求

### 系统要求
- Python 3.8+
- CUDA 11.0+ (GPU训练)
- 至少12GB GPU内存

### 依赖包

```bash
pip install -r requirements.txt
```

主要依赖：
- torch>=2.0.0
- torchvision>=0.15.0
- fair-esm>=2.0.0 (ESM2蛋白质语言模型)
- biopython>=1.79
- numpy>=1.21
- scikit-learn>=1.0
- tensorboard>=2.11

## 数据集说明

### MIBIG数据集

本项目使用**MIBIG (Minimum Information about a Biosynthetic Gene cluster)**数据库作为主要训练数据集。MIBIG是全球最大的生物合成基因簇标准化数据库，包含超过3000个经过实验验证的BGC条目。

#### MIBIG数据特点
- **高质量标注**: 所有BGC都经过人工审核和实验验证
- **标准化格式**: 使用统一的标注标准和文件格式
- **类型丰富**: 涵盖NRPS、PKS、terpene、saccharide、ribosomal等多种BGC类型
- **持续更新**: 定期发布新版本，数据不断增长

#### 数据获取方式
```bash
# 下载MIBIG数据集（GenBank格式）
wget https://dl.secondarymetabolites.org/mibig/mibig_gbk_3.1.tar.gz
tar -xzf mibig_gbk_3.1.tar.gz
```

**注意**: 由于MIBIG数据集文件较大（约1.5GB解压后），本项目不在GitHub仓库中包含原始数据。请用户自行下载并放置在`data/genomes/`目录下。

### 数据预处理流程

#### 第一步：BGC数据解析

从MIBIG的GenBank文件中提取BGC信息和蛋白质序列：

```bash
# 解析BGC文件，提取正样本数据
python scripts/parse_gbk.py \
    --gbk_dir data/genomes/ \
    --out_json data/bgc_mapping.json \
    --out_protein_json data/bgc_proteins.json \
    --non_bgc_dir data/output \
    --target_length 128 \
    --augment_times 5
```

#### 第二步：基因组序列处理

处理完整基因组，提取非BGC区域作为负样本：

```bash
# 使用antiSMASH处理基因组，识别BGC区域
python scripts/process_genomes.py \
    --genome_dir data/nine_genomes_gbff/ \
    --output_dir data/output/
```

#### 第三步：CDS序列提取

从GenBank文件中提取所有CDS的氨基酸序列：

```bash
# 批量提取CDS序列
python scripts/cds_mapping_sequences.py \
    --gbk_dir data/genomes/ \
    --out_json data/cds_sequences.json
```

#### 第四步：生成负样本

从非BGC区域生成负样本数据：

```bash
# 添加负样本到训练数据
python scripts/parse_gbk.py \
    --add_negatives \
    --mapping_json data/bgc_mapping.json \
    --proteins_json data/bgc_proteins.json \
    --output_dir data/output \
    --target_length 128 \
    --min_negatives 20000
```

#### 第五步：序列嵌入提取

使用ESM2蛋白质语言模型提取序列嵌入：

```bash
# 使用GPU多进程加速嵌入提取
python scripts/example_usage.py \
    --bgc_proteins data/bgc_proteins.json \
    --cds_sequences data/cds_sequences.json \
    --output_dir embeddings/ \
    --model esm2_t33_650M_UR50D \
    --layer 33 \
    --device cuda \
    --gpu_ids 0,1,2,3
```

### 正负样本制作详解

#### 正样本制作策略

1. **数据来源**: 从MIBIG数据库的标准化GenBank文件提取
2. **序列长度标准化**: 
   - 目标长度：128个CDS
   - 不足128个的BGC通过添加非BGC区域的CDS进行填充
   - 超过128个的BGC进行截断或分割
3. **数据增强**:
   - 每个原始BGC生成5个增强样本
   - 随机选择非BGC CDS进行前后填充
   - 保持BGC核心区域不变

#### 负样本制作策略

1. **数据来源**: 使用antiSMASH处理完整基因组，提取非BGC区域
2. **滑窗采样**:
   - 窗口大小：128个连续CDS
   - 滑动步长：30个CDS（避免过度重叠）
   - 确保所有CDS都有protein_id标识
3. **质量控制**:
   - 过滤掉包含未知蛋白质的序列段
   - 确保序列完整性和一致性

#### 数据平衡

- **正样本**: 约15,000个（3,000个原始BGC × 5倍增强）
- **负样本**: 约20,000个
- **类别分布**: 保持各BGC类型的相对平衡

### 文件格式说明

#### 映射文件格式 (bgc_mapping.json)
```json
{
    "BGC0000001_aug_1": [
        {
            "start": 0,
            "end": 127,
            "type": "NRPS"
        }
    ],
    "NEG_00001": [
        {
            "start": 0,
            "end": 127,
            "type": "no-object"
        }
    ]
}
```

#### 蛋白质文件格式 (bgc_proteins.json)
```json
{
    "BGC0000001_aug_1": [
        "AEK75490.1",
        "AEK75491.1",
        "AEK75492.1"
    ]
}
```

#### CDS序列文件格式 (cds_sequences.json)
```json
{
    "AEK75490.1": "MKLIVGAGPGDYGLYKATQHAHDLLTRVTYIPLFQN...",
    "AEK75491.1": "MRSDKFVWLLTGRRRSDTLLEFPYLDRTFVEPCTLD..."
}
```

## 使用流程

### 模型训练

#### 训练命令
```bash
# 基础训练（单GPU）
python main_by_cluster.py \
    --train_mapping_json data/bgc_mapping.json \
    --train_emb_dir embeddings/ \
    --val_mapping_json data/val/bgc_mapping.json \
    --val_emb_dir data/val/embeddings/ \
    --num_classes 7 \
    --num_queries 8 \
    --epochs 500 \
    --lr 1e-4 \
    --output_dir outputs/

# 分布式训练（多GPU）
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    main_by_cluster.py \
    --train_mapping_json data/bgc_mapping.json \
    --train_emb_dir embeddings/ \
    --val_mapping_json data/val/bgc_mapping.json \
    --val_emb_dir data/val/embeddings/ \
    --num_classes 7 \
    --num_queries 8 \
    --epochs 500 \
    --lr 1e-4 \
    --output_dir outputs/
```

#### 主要训练参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--train_mapping_json` | 训练集映射文件 | data/bgc_mapping.json |
| `--train_emb_dir` | 训练集嵌入目录 | embeddings/ |
| `--val_mapping_json` | 验证集映射文件 | data/val/bgc_mapping.json |
| `--val_emb_dir` | 验证集嵌入目录 | data/val/embeddings/ |
| `--num_classes` | BGC类别数量 | 7 |
| `--hidden_dim` | 隐藏层维度 | 256 |
| `--num_queries` | 查询对象数量 | 8 |
| `--epochs` | 训练轮数 | 500 |
| `--lr` | 学习率 | 1e-4 |
| `--batch_size` | 批次大小 | 4 |
| `--weight_decay` | 权重衰减 | 1e-4 |

#### 训练过程特点

1. **5折交叉验证**: 自动进行5折交叉验证，确保模型泛化能力
2. **自动保存**: 自动保存最佳模型检查点到`outputs/checkpoint_best.pth`
3. **损失函数**: 结合分类损失、边界框损失和GIoU损失
4. **训练监控**: 支持TensorBoard可视化训练过程

### 模型验证和预测

#### 验证训练好的模型

```bash
python validate_final.py \
    --checkpoint_path outputs/checkpoint_best.pth \
    --val_mapping_json data/val/bgc_mapping.json \
    --val_emb_dir data/val/embeddings/ \
    --output_dir validation_results/ \
    --num_classes 7 \
    --num_queries 8
```

#### 预测新数据

1. **准备新数据**:
```bash
# 1. 转换数据格式
python scripts/convert_train_val_to_bgc_format.py \
    --train_json new_train.json \
    --val_json new_val.json \
    --out_dir data/new_converted

# 2. 提取序列嵌入
python scripts/example_usage.py \
    --bgc_proteins data/new_converted/val/bgc_proteins.json \
    --cds_sequences data/new_converted/val/cds_sequences.json \
    --output_dir data/new_converted/val/embeddings \
    --device cuda
```

2. **运行预测**:
```bash
python validate_final.py \
    --checkpoint_path outputs/checkpoint_best.pth \
    --val_mapping_json data/new_converted/val/bgc_mapping.json \
    --val_emb_dir data/new_converted/val/embeddings/ \
    --output_dir predictions/ \
    --num_classes 7 \
    --num_queries 8
```

#### 验证结果输出

验证脚本会生成以下结果：
- **性能指标**: 准确率、精确率、召回率、F1分数、mAP
- **类别统计**: 每个BGC类型的详细性能
- **预测文件**: JSON格式的详细预测结果
- **混淆矩阵**: 类别预测的混淆矩阵分析

## 工具文件详解

### 核心脚本功能

#### 1. `scripts/parse_gbk.py` - BGC数据解析器
**功能**: 从GenBank文件解析BGC信息，生成训练数据

**输入**:
- GenBank文件目录 (`--gbk_dir`)
- 非BGC CDS数据目录 (`--non_bgc_dir`, 可选)

**输出**:
- BGC映射文件 (`bgc_mapping.json`)
- 蛋白质序列文件 (`bgc_proteins.json`)

**运行方式**:
```bash
# 解析正样本
python scripts/parse_gbk.py \
    --gbk_dir data/genomes/ \
    --out_json data/bgc_mapping.json \
    --out_protein_json data/bgc_proteins.json \
    --target_length 128 \
    --augment_times 5

# 添加负样本
python scripts/parse_gbk.py \
    --add_negatives \
    --mapping_json data/bgc_mapping.json \
    --proteins_json data/bgc_proteins.json \
    --output_dir data/output \
    --min_negatives 20000
```

#### 2. `scripts/process_genomes.py` - 基因组处理器
**功能**: 使用antiSMASH处理基因组，识别BGC区域并提取非BGC CDS

**输入**:
- 基因组文件目录 (`--genome_dir`)

**输出**:
- 每个基因组对应的非BGC CDS文件 (`*_non_bgc_cds.json`)

**运行方式**:
```bash
python scripts/process_genomes.py \
    --genome_dir data/nine_genomes_gbff/ \
    --output_dir data/output/
```

#### 3. `scripts/cds_mapping_sequences.py` - CDS序列提取器
**功能**: 从GenBank文件提取CDS的氨基酸序列

**输入**:
- GenBank文件或目录 (`--single_gbk` 或 `--gbk_dir`)

**输出**:
- CDS序列文件 (`cds_sequences.json`)

**运行方式**:
```bash
# 批量处理
python scripts/cds_mapping_sequences.py \
    --gbk_dir data/genomes/ \
    --out_json data/cds_sequences.json

# 单文件处理
python scripts/cds_mapping_sequences.py \
    --single_gbk data/genomes/BGC0001317.gbk \
    --out_json single_cds_sequences.json
```

#### 4. `scripts/example_usage.py` - 序列嵌入提取器
**功能**: 使用ESM2模型批量提取蛋白质序列嵌入

**输入**:
- BGC蛋白质文件 (`--bgc_proteins`)
- CDS序列文件 (`--cds_sequences`)

**输出**:
- 嵌入文件目录，每个BGC一个`.npy`文件

**运行方式**:
```bash
# GPU多进程处理
python scripts/example_usage.py \
    --bgc_proteins data/bgc_proteins.json \
    --cds_sequences data/cds_sequences.json \
    --output_dir embeddings/ \
    --model esm2_t33_650M_UR50D \
    --device cuda \
    --gpu_ids 0,1,2,3

# CPU处理（较慢）
python scripts/example_usage.py \
    --bgc_proteins data/bgc_proteins.json \
    --cds_sequences data/cds_sequences.json \
    --output_dir embeddings/ \
    --device cpu
```

#### 5. `scripts/convert_train_val_to_bgc_format.py` - 数据格式转换器
**功能**: 将标准JSON格式转换为BGC-DETR训练格式

**输入**:
- 训练集JSON (`--train_json`)
- 验证集JSON (`--val_json`)

**输出**:
- 转换后的训练和验证数据

**运行方式**:
```bash
python scripts/convert_train_val_to_bgc_format.py \
    --train_json trainnew.json \
    --val_json validationnew.json \
    --out_dir data/converted
```

### 完整数据处理流程图

```
MIBIG数据集 (.gbk文件)
        ↓
1. parse_gbk.py → bgc_mapping.json + bgc_proteins.json
        ↓
2. cds_mapping_sequences.py → cds_sequences.json
        ↓
3. process_genomes.py → 非BGC CDS数据
        ↓
4. parse_gbk.py --add_negatives → 添加负样本
        ↓
5. example_usage.py → 序列嵌入 (.npy文件)
        ↓
6. main_by_cluster.py → 模型训练
        ↓
7. validate_final.py → 模型验证和预测
```

### 支持的BGC类型

BGC-DETR支持7种主要的生物合成基因簇类型：

| 类型 | 英文名称 | 中文描述 | 典型产物 |
|------|----------|----------|----------|
| **NRPS** | Nonribosomal Peptide Synthetase | 非核糖体肽合成酶 | 抗生素、表面活性剂 |
| **PKS** | Polyketide Synthase | 聚酮合酶 | 抗菌素、抗癌药物 |
| **RiPP** | Ribosomally Synthesized and Post-translationally Modified Peptides | 核糖体合成修饰肽 | 细菌素、环肽 |
| **Terpene** | Terpene | 萜类化合物 | 香料、药物前体 |
| **Saccharide** | Saccharide | 糖类化合物 | 抗生素糖基 |
| **Alkaloid** | Alkaloid | 生物碱 | 植物毒素、药物 |
| **Other** | Other Types | 其他类型 | 各种次生代谢产物 |

**特殊标签**:
- **no-object**: 背景类，表示非BGC区域

## 模型性能

### 评估指标

- **分类指标**: 准确率、精确率、召回率、F1分数
- **检测指标**: mAP (mean Average Precision)
- **IoU阈值**: 0.5 (标准目标检测评估)

### 典型性能

在标准数据集上的典型性能：
- 整体准确率: ~85%
- mAP@0.5: ~0.75
- F1分数: ~0.80

## 高级功能

### 1. 类别平衡策略

支持多种类别平衡策略：
- `upsample`: 上采样少数类
- `downsample`: 下采样多数类
- `weighted`: 加权采样

### 2. 5折交叉验证

自动进行5折交叉验证，确保模型泛化能力。

### 3. 分布式训练

支持多GPU分布式训练，提高训练效率。

### 4. 模型检查点

自动保存最佳模型检查点，支持断点续训。

## 重要注意事项

### 数据文件说明

**GitHub仓库限制**: 
- 由于MIBIG数据集文件过大（约1.5GB），本仓库不包含原始数据文件
- 用户需要自行下载MIBIG数据集并按照上述流程处理
- 所有中间处理文件（嵌入文件、映射文件等）也需要用户自行生成

### 数据获取建议

1. **MIBIG数据集下载**:
```bash
# 创建数据目录
mkdir -p data/genomes
cd data/genomes

# 下载最新版MIBIG数据集
wget https://dl.secondarymetabolites.org/mibig/mibig_gbk_3.1.tar.gz
tar -xzf mibig_gbk_3.1.tar.gz
mv mibig_gbk_3.1/* .
```

2. **完整基因组数据**（用于负样本生成）:
```bash
# 下载示例基因组数据
mkdir -p data/nine_genomes_gbff
# 从NCBI下载完整基因组的GenBank文件
# 或使用项目提供的示例基因组数据
```

### 系统要求

- **存储空间**: 至少50GB可用空间（包括原始数据、嵌入文件等）
- **内存需求**: 32GB RAM（处理大型数据集时）
- **GPU**: 8GB+ GPU内存（推荐24GB以上用于大规模训练）
- **时间估算**: 完整数据处理流程约需要8-12小时（取决于硬件配置）

## 故障排除

### 常见问题

1. **内存不足**
   - 减小batch_size (推荐从4减至2或1)
   - 减少num_workers
   - 使用梯度累积
   - 分批处理数据

2. **CUDA错误**
   - 检查CUDA版本兼容性 (推荐CUDA 11.6+)
   - 确保GPU内存充足
   - 使用 `--device cpu` 作为备选方案
   - 减少GPU并行数量

3. **数据加载错误**
   - 检查文件路径是否正确
   - 验证JSON文件格式
   - 确保所有嵌入文件存在
   - 检查文件权限

4. **序列嵌入提取慢**
   - 使用多GPU并行: `--gpu_ids 0,1,2,3`
   - 启用断点续传功能（默认开启）
   - 考虑使用更小的ESM2模型

### 调试技巧

1. **使用小数据集测试**:
```bash
# 仅处理前100个BGC进行测试
head -n 100 data/bgc_mapping.json > data/test_mapping.json
```

2. **启用详细日志**:
```bash
python main_by_cluster.py --verbose
```

3. **检查数据预处理步骤**:
```bash
# 验证数据格式
python -c "import json; print(len(json.load(open('data/bgc_mapping.json'))))"
```

4. **监控GPU使用**:
```bash
watch -n 1 nvidia-smi
```

## 快速开始指南

如果您想快速体验BGC-DETR，请按照以下步骤操作：

### 1. 环境安装
```bash
# 克隆仓库
git clone <repository-url>
cd BGC-DETR

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载数据
```bash
# 下载MIBIG数据集
mkdir -p data/genomes
cd data/genomes
wget https://dl.secondarymetabolites.org/mibig/mibig_gbk_3.1.tar.gz
tar -xzf mibig_gbk_3.1.tar.gz
mv mibig_gbk_3.1/* .
cd ../..
```

### 3. 数据预处理（完整流程）
```bash
# 步骤1: 解析BGC数据
python scripts/parse_gbk.py \
    --gbk_dir data/genomes/ \
    --out_json data/bgc_mapping.json \
    --out_protein_json data/bgc_proteins.json \
    --target_length 128 \
    --augment_times 1

# 步骤2: 提取CDS序列
python scripts/cds_mapping_sequences.py \
    --gbk_dir data/genomes/ \
    --out_json data/cds_sequences.json

# 步骤3: 生成序列嵌入（这一步最耗时）
python scripts/example_usage.py \
    --bgc_proteins data/bgc_proteins.json \
    --cds_sequences data/cds_sequences.json \
    --output_dir embeddings/ \
    --device cuda

# 步骤4: 添加负样本（可选，需要额外的基因组数据）
# python scripts/parse_gbk.py --add_negatives ...
```

### 4. 模型训练
```bash
python main_by_cluster.py \
    --train_mapping_json data/bgc_mapping.json \
    --train_emb_dir embeddings/ \
    --num_classes 7 \
    --num_queries 8 \
    --epochs 100 \
    --output_dir outputs/
```

### 5. 模型验证
```bash
python validate_final.py \
    --checkpoint_path outputs/checkpoint_best.pth \
    --val_mapping_json data/bgc_mapping.json \
    --val_emb_dir embeddings/ \
    --output_dir validation_results/
```

## 致谢

- MIBIG数据库团队提供高质量的BGC数据
- Facebook Research的ESM团队提供蛋白质语言模型
- DETR原作者提供优秀的目标检测框架

## 许可证

本项目采用MIT许可证，详情请见LICENSE文件。

## 引用

如果您在研究中使用了BGC-DETR，请引用我们的工作：
```bibtex
@article{bgc-detr2024,
  title={BGC-DETR: End-to-End Biosynthetic Gene Cluster Detection using DETR Architecture},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024}
}
```

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

### 开发环境设置

1. Fork本仓库
2. 创建功能分支
3. 安装开发依赖
4. 运行测试
5. 提交Pull Request

**注意**: 本项目仍在积极开发中，API可能会有变化。请关注更新日志了解最新变化。 