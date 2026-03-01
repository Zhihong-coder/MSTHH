"""
lib/data_prepare_multimodal.py
──────────────────────────────
Multimodal data loader for MSTHH.

For PEMS03/04/07/08 the raw .npz contains shape (T, N, 3):
    channel 0: flow         (vehicles / hour)
    channel 1: speed  OR    occupancy  – depends on dataset source
    channel 2: occupancy OR speed

For METR-LA / PEMS-BAY the .npz typically has shape (T, N, 2) or (T, N, 3):
    channel 0: speed
    channel 1: flow   (if present)

This loader returns x of shape (B, T, N, C) where:
    C = num_modalities + [1 if tod] + [1 if dow]

It normalises ONLY the traffic channels (not tod/dow).
The existing single-modal get_dataloaders_from_index_data is kept intact.
"""

import os
import numpy as np
import torch
from .utils import print_log, StandardScaler, vrange


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

class MultiModalScaler:
    """Independent StandardScaler per modality channel."""

    def __init__(self):
        self.scalers = []

    def fit(self, data_list):
        """data_list: list of (T,N) arrays, one per modality."""
        self.scalers = []
        for d in data_list:
            sc = StandardScaler(mean=float(d.mean()), std=float(d.std()))
            self.scalers.append(sc)
        return self

    def transform(self, data):
        """data: (..., M) – last dim is modality index. Returns same shape."""
        out = data.copy()
        for m, sc in enumerate(self.scalers):
            out[..., m] = sc.transform(data[..., m])
        return out

    def inverse_transform(self, data):
        """Inverse-transform the FIRST modality only (prediction target = flow)."""
        # data: (B, T, N, 1)  or  (B, T, N) – always invert channel 0
        sc = self.scalers[0]
        if isinstance(data, torch.Tensor):
            return data * sc.std + sc.mean
        return data * sc.std + sc.mean


# ─────────────────────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────────────────────

def get_multimodal_dataloaders(
    data_dir,
    num_modalities   = 3,     # how many traffic channels to load as separate modalities
    tod              = True,
    dow              = True,
    batch_size       = 16,
    log              = None,
):
    """
    Returns:
        trainset_loader, valset_loader, testset_loader, scaler

    x shape: (B, T, N, C) where C = num_modalities + [1 if tod] + [1 if dow]
    y shape: (B, T, N, 1)   – flow only (channel 0)

    The scaler.inverse_transform inverts channel 0 (flow).
    """
    # ── Load raw data ────────────────────────────────────────────────────────
    raw = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)
    # raw shape: (T_total, N, C_raw) where C_raw >= num_modalities
    T_total, N, C_raw = raw.shape

    if C_raw < num_modalities:
        print_log(
            f"[WARNING] data has only {C_raw} channels but num_modalities={num_modalities}. "
            f"Reducing num_modalities to {C_raw}.",
            log=log
        )
        num_modalities = C_raw

    # ── Load sample index ────────────────────────────────────────────────────
    index      = np.load(os.path.join(data_dir, "index.npz"))
    train_idx  = index["train"]
    val_idx    = index["val"]
    test_idx   = index["test"]

    def make_xy(split_index):
        xi = vrange(split_index[:, 0], split_index[:, 1])  # (S, T_in)
        yi = vrange(split_index[:, 1], split_index[:, 2])  # (S, T_out)
        return xi, yi

    x_tr_idx, y_tr_idx = make_xy(train_idx)
    x_va_idx, y_va_idx = make_xy(val_idx)
    x_te_idx, y_te_idx = make_xy(test_idx)

    # ── Build feature array ──────────────────────────────────────────────────
    # tod and dow: computed from the *index* position in the dataset
    # (same convention as the original data_prepare.py)
    # We compute fractional time-of-day from index 1 and day-of-week from index 2
    # IF the raw data already contains these – otherwise recompute.

    # Traffic channels
    traffic = raw[..., :num_modalities]   # (T_total, N, M)

    # tod / dow: try to read from raw data; if not present, derive from time index
    extra_channels = []
    if tod:
        if C_raw > num_modalities:
            # Assume the next channel after the M traffic ones is tod
            tod_ch = raw[..., num_modalities]   # (T_total, N)
        else:
            # Fall back: compute from raw position (assume 5-min intervals)
            steps_per_day = 288
            t_idx = np.arange(T_total) % steps_per_day / steps_per_day
            tod_ch = np.tile(t_idx[:, None], (1, N)).astype(np.float32)
        extra_channels.append(tod_ch[..., None])

    if dow:
        if C_raw > num_modalities + (1 if tod else 0):
            dow_offset = num_modalities + (1 if tod else 0)
            dow_ch = raw[..., dow_offset]
        else:
            # Fall back: compute from raw position
            steps_per_day = 288
            day_idx = (np.arange(T_total) // steps_per_day) % 7
            dow_ch = np.tile(day_idx[:, None], (1, N)).astype(np.float32)
        extra_channels.append(dow_ch[..., None])

    # Concatenate: (T_total, N, M + n_time)
    if extra_channels:
        data = np.concatenate([traffic] + extra_channels, axis=-1)
    else:
        data = traffic

    # ── Fit scaler on training traffic channels only ─────────────────────────
    x_train_traffic = traffic[x_tr_idx]   # (S_train, T_in, N, M)
    scaler = MultiModalScaler().fit(
        [x_train_traffic[..., m].ravel() for m in range(num_modalities)]
    )

    # ── Slice & normalise ────────────────────────────────────────────────────
    def build_split(x_idx, y_idx):
        x = data[x_idx]                              # (S, T_in, N, C)
        y = traffic[y_idx][..., :1]                  # (S, T_out, N, 1)  flow only

        # Normalise traffic channels in x (in-place copy)
        x = x.copy()
        for m in range(num_modalities):
            x[..., m] = scaler.scalers[m].transform(x[..., m])

        return x, y

    x_train, y_train = build_split(x_tr_idx, y_tr_idx)
    x_val,   y_val   = build_split(x_va_idx, y_va_idx)
    x_test,  y_test  = build_split(x_te_idx, y_te_idx)

    print_log(f"Multimodal Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Multimodal Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}",   log=log)
    print_log(f"Multimodal Testset: \tx-{x_test.shape} \ty-{y_test.shape}",  log=log)
    print_log(f"Num modalities: {num_modalities}, Extra channels: "
              f"tod={tod}, dow={dow}", log=log)

    # ── Build DataLoaders ────────────────────────────────────────────────────
    def make_loader(x, y, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(x), torch.FloatTensor(y)
        )
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (
        make_loader(x_train, y_train, shuffle=True),
        make_loader(x_val,   y_val,   shuffle=False),
        make_loader(x_test,  y_test,  shuffle=False),
        scaler,
    )
