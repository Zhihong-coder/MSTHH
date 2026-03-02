# MSTHH: A Unified Framework for Asynchronous and Heterogeneous Multimodal Traffic Prediction
Zhihong Wanga, Wei Lia,∗, Zhuoxuan Lianga, Junhui Jianga, Xiaohua Jiab and Moustafa Youssefc

![MSTHH Architecture](https://github.com/Zhihong-coder/MSTHH/blob/master/picture/MSTHH.png)

Performance on Traffic Forecasting Benchmarks

![performance](https://github.com/Zhihong-coder/MSTHH/blob/master/picture/performance.png)
## Required Packages
```bash
pytorch>=1.11
numpy
pandas
matplotlib
pyyaml
pickle
torchinfo
```
## dataset
```bash
METRLA
PEMSBAY
PEMS03
PEMS04
PEMS07
PEMS08
```

## 一、运行方式

### 快速开始（与原始数据格式兼容，单模态模式）

原始数据 `data.npz` 只有 flow + tod + dow 三个特征，设 `num_modalities=1`：

```bash
cd model/
python train_msthh.py -d pems08 -g 0
```

### 完整多模态模式（推荐，复现论文结果）

需要 `data.npz` 中包含 flow, speed, OCC 三个交通特征（前3个通道），设 `num_modalities=3`：

```bash
cd model/
python train_msthh.py -d pems08 -g 0 --multimodal
```

---

## 二、核心修改位置

### 修改1：`model/MSTHH.py`（新增，~550行）

实现了论文全部4个阶段：

| 类名 | 对应论文公式 | 说明 |
|---|---|---|
| `FrequencyAdaptivePositionalEncoding` | Eq.5 | FAP编码，per-modality可学习频率ω_m, f_m |
| `TemporalScaleAwareEmbedding` | Eq.6-7 | TSA嵌入 = W_emb·X + PE + TE |
| `HierarchicalMultiScaleGCN` | Eq.8-9 | K尺度自适应邻接矩阵 + L层图卷积 |
| `CrossModalTemporalAligner` | Eq.10 | 软注意力跨模态时间对齐（无信息损失） |
| `DualBranchDecomposition` | Eq.11-13 | 共享/特定表征分解 + 正交约束 + 重建损失 |
| `UnifiedSemanticSpaceProjection` | Eq.14-15 | USSP，L2归一化到单位超球面 |
| `IntraModalCL` | Eq.16-17 | 模态内对比学习（噪声增强正样本对） |
| `InterModalCL` | Eq.18 | 模态间对比学习（时空锚点处对齐） |
| `InstanceLevelCL` | Eq.19 | 实例级对比学习（注意力加权全局表征） |
| `MambaTemporalBlock` | Eq.22 | Mamba SSM，O(T)复杂度时序建模 |
| `GCNMambaLayer` | § 4.3 | GCN↔Mamba交替架构，一个完整层 |
| `MissingAwareFusion` | Eq.23 | MAF，数据缺失时自适应切换到全局先验 |
| `MultiStepDecoder` | Eq.24 | Transformer解码器多步预测 |
| `MSTHH.compute_total_loss` | Eq.25-27 | 自适应多任务损失（可学习σ1,σ2,σ3） |

#### 关键代码段：模型前向传播

```python
def forward(self, x, modal_mask=None):
    # x: (B, T, N, C)  C = M + [tod] + [dow]
    
    # Stage I: 编码 + 对齐
    H_list    = [self.tsa[m](x[..., m:m+1]) for m in range(M)]
    H_list    = [self.hgcn[m](H) for m, H in enumerate(H_list)]
    H_aligned = [cross_modal_align(H_list[i], H_list) for i in range(M)]
    
    # Stage II: 语义对齐
    shared, specific, L_ortho, L_recon = self.dual_branch(H_aligned)
    z_unified = [self.ussp(h) for h in shared]
    L_cl = intra_cl + inter_cl + instance_cl  # 三级对比损失
    
    # Stage III: 融合
    z_concat = concat([z_u + z_sp + modal_emb for each modality])
    z = GCN-Mamba(z_concat)   # 交替GCN-Mamba层
    z_fused = MAF(z, pooled)  # 缺失感知融合
    
    # Stage IV: 预测
    pred = Transformer_Decoder(z_fused)
    return pred  # eval; or (pred, L_cl, L_ortho, L_recon) in train
```

### 修改2：`model/MSTHH.yaml`（新增）

```yaml
# 与 STAEformer.yaml 结构完全一致，新增以下关键参数：
model_args:
  num_modalities: 3        # M=3 (flow,speed,OCC) for PEMS03/04/07/08
                           # M=1 for METR-LA/PEMS-BAY (single modal)
  d_model: 64              # per-modality hidden dimension
  num_gcn_scales: 3        # K=3 spatial scales (local, medium, global)
  num_gcn_layers: 3        # L_gcn=3 graph conv layers
  num_gcn_mamba_layers: 3  # L_gcn_mamba=3 alternating GCN-Mamba layers
  sigma1_init: 0.8         # task loss uncertainty (learnable)
  sigma2_init: 2.0         # CL loss uncertainty (learnable)
  sigma3_init: 0.5         # reg loss uncertainty (learnable)
```

### 修改3：`model/train_msthh.py`（新增）

相对于原始 `train.py` 的关键改动：

#### (A) 导入变更
```python
# 原始
from model.STAEformer import STAEformer

# 修改后
from model.MSTHH import MSTHH
from lib.data_prepare_multimodal import get_multimodal_dataloaders
```

#### (B) 训练循环：处理多返回值 + 联合损失

```python
# 原始 train_one_epoch
out_batch = model(x_batch)
loss = criterion(out_batch, y_batch)

# 修改后 train_one_epoch（核心改动）
out = model(x)
if isinstance(out, tuple):
    pred, L_cl, L_ortho, L_recon = out   # ← 训练时返回4个值
else:
    pred = out
    L_cl = L_ortho = L_recon = 0

# 自适应多任务损失（Eq.25-27）
loss, _ = model.compute_total_loss(
    pred, y, L_cl, L_ortho, L_recon,
    criterion, epoch, max_epochs
)
```

#### (C) 验证/测试：无需修改
```python
# eval_model 和 predict 保持不变——模型在 eval() 时只返回 pred
model.eval()
pred = model(x)  # 只返回预测值，无损失
```

#### (D) 数据加载：根据 --multimodal 标志选择
```python
if args.multimodal and num_modalities > 1:
    trainset_loader, ... = get_multimodal_dataloaders(data_path, num_modalities=3, ...)
else:
    trainset_loader, ... = get_dataloaders_from_index_data(data_path, ...)
```

### 修改4：`lib/data_prepare_multimodal.py`（新增）

为多模态数据加载，关键逻辑：

```python
# 从 data.npz 加载前 num_modalities 个通道作为独立模态
traffic = raw[..., :num_modalities]   # (T, N, M)

# 每个模态独立归一化
scaler = MultiModalScaler().fit([traffic[..., m] for m in range(M)])

# x shape: (B, T, N, M+[tod]+[dow])   y shape: (B, T, N, 1)  -- flow only
```

---

## 三、数据准备（多模态模式）

若 PEMS 原始数据只有单通道，需要重新预处理使 `data.npz` 包含全部3个交通特征。

### PEMS04 为例（`.h5` 或 `.npz` 原始多通道）：

```python
import numpy as np

# 读取原始 PEMS 数据（含 flow, speed, OCC）
data = np.load("raw_pems04.npz")["data"]  # (T, N, 3)

# 计算 time-of-day 和 day-of-week
steps_per_day = 288
T = data.shape[0]
tod = (np.arange(T) % steps_per_day / steps_per_day)   # (T,)
dow = ((np.arange(T) // steps_per_day) % 7).astype(float)

# 拼接为 (T, N, 5)
tod_arr = np.tile(tod[:, None, None], (1, data.shape[1], 1))
dow_arr = np.tile(dow[:, None, None], (1, data.shape[1], 1))
full = np.concatenate([data, tod_arr, dow_arr], axis=-1).astype(np.float32)

# 保存
np.savez_compressed("data/PEMS04/data.npz", data=full)
# 然后正常生成 index.npz（与原始方法相同）
```

---

## 四、模型参数对比

| 参数 | STAEformer | MSTHH (M=1) | MSTHH (M=3) |
|---|---|---|---|
| PEMS08 Params | ~125K | ~87K | ~180K |
| MACs (PEMS08) | ~167M | ~100M | ~300M |
| 训练时/epoch | ~38s | ~28s | ~45s |
| 推理时/batch | ~4.2s | ~3.8s | ~5.5s |

> 注：以上数据与论文 Table 4/5 一致，基于 RTX 4090。

---

## 五、常见问题

### Q1: `ModuleNotFoundError: No module named 'mamba_main'`
Mamba 未安装时自动降级为 GRU（性能稍差但可运行）。安装方法：
```bash
pip install mamba-ssm causal-conv1d
```

### Q2: 多模态模式下 `num_modalities=3` 但数据只有1个通道
代码会自动复制 channel 0 填充缺失模态（性能与单模态模式等同）。正确方式是使用 `--multimodal` 标志 + 多通道数据。

### Q3: 显存不足
调小 `d_model`（如 32）或减少 `num_gcn_mamba_layers`（如 2）。

### Q4: 损失出现 NaN
增大 `sigma2_init`（如 3.0）或降低学习率（如 0.0005）。检查数据归一化是否正常。

---

## 六、从 train.py 直接调用（最小改动方式）

如果不想使用新的训练脚本，只需在原始 `train.py` 中做以下 **5处修改**：

```python
# 1. 替换导入
# from model.STAEformer import STAEformer
from model.MSTHH import MSTHH

# 2. 替换模型创建
# model = STAEformer(**cfg["model_args"])
model = MSTHH(**cfg["model_args"])

# 3. 替换 yaml 文件名
# with open(f"{model_name}.yaml") as f:
with open("MSTHH.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = cfg[dataset]

# 4. 修改 train_one_epoch 的前向和损失计算（见上文）

# 5. 将 visualization 目录名改为 visualization_msthh（可选）
```
