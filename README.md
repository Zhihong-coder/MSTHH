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
## Dataset
| Dataset | # Sensors | # Time Steps | Interval | Features | Download |
|---------|-----------|--------------|----------|----------|----------|
| **PEMS03** | 358 | 26,208 | 5 min | Flow, Speed, OCC | [Download PEMS03](https://github.com/guoshnBJTU/ASTGNN/tree/main/data) |
| **PEMS04** | 307 | 16,992 | 5 min | Flow, Speed, OCC | [Download PEMS04](https://github.com/guoshnBJTU/ASTGNN/tree/main/data) |
| **PEMS07** | 883 | 28,224 | 5 min | Flow, Speed, OCC | [Download PEMS07](https://github.com/guoshnBJTU/ASTGNN/tree/main/data) |
| **PEMS08** | 170 | 17,856 | 5 min | Flow, Speed, OCC | [Download PEMS08](https://github.com/guoshnBJTU/ASTGNN/tree/main/data) |
| **METR-LA** | 207 | 34,272 | 5 min | Flow, Speed | [Download METR-LA](https://github.com/liyaguang/DCRNN) |
| **PEMS-BAY** | 325 | 52,116 | 5 min | Flow, Speed | [Download PEMS-BAY](https://github.com/liyaguang/DCRNN) |

The real-world PEMS datasets used in our experiments are available at the official portal: [https://pems.dot.ca.gov/](https://pems.dot.ca.gov/)

## Usage

### Quick Start (single-modal mode, compatible with original data format)

When `data.npz` contains only flow + time-of-day + day-of-week features, set `num_modalities=1`:

```bash
cd model/
python train_msthh.py -d pems08 -g 0
```

### Full Multimodal Mode (recommended for reproducing paper results)

Requires `data.npz` to include flow, speed, and OCC as the first 3 channels; set `num_modalities=3`:

```bash
cd model/
python train_msthh.py -d pems08 -g 0 --multimodal
```

## Data Preparation (Multimodal Mode)

If the raw PEMS data contains only a single channel, you need to preprocess it so that `data.npz` includes all three traffic features.

### Example: PEMS04 (from raw `.h5` or `.npz` with multiple channels)

```python
import numpy as np

# Load raw PEMS data (contains flow, speed, OCC)
data = np.load("raw_pems04.npz")["data"]  # (T, N, 3)

# Compute time-of-day and day-of-week
steps_per_day = 288
T = data.shape[0]
tod = (np.arange(T) % steps_per_day / steps_per_day)        # (T,)
dow = ((np.arange(T) // steps_per_day) % 7).astype(float)   # (T,)

# Expand and concatenate to shape (T, N, 5)
tod_arr = np.tile(tod[:, None, None], (1, data.shape[1], 1))
dow_arr = np.tile(dow[:, None, None], (1, data.shape[1], 1))
full = np.concatenate([data, tod_arr, dow_arr], axis=-1).astype(np.float32)

# Save
np.savez_compressed("data/PEMS04/data.npz", data=full)
# Then generate index.npz as usual (same procedure as the original pipeline)
```
