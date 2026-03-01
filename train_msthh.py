"""
model/train_msthh.py
────────────────────
Training & evaluation script for MSTHH.

Key differences from train.py (STAEformer):
  1. Imports MSTHH model and multimodal data loader.
  2. Training loop unpacks (pred, L_cl, L_ortho, L_recon) from model.
  3. Uses model.compute_total_loss() (adaptive multi-task loss, Eq.25-27).
  4. Validation / test remain unchanged (model returns pred only in eval mode).
  5. Supports both single-modal (existing data) and multimodal data.

Usage:
    # Single-modal (existing data, num_modalities=1):
    python train_msthh.py -d pems08

    # Multi-modal (flow+speed+OCC in data.npz channels 0,1,2):
    python train_msthh.py -d pems08 --multimodal
"""

import argparse
import copy
import datetime
import json
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torchinfo import summary

sys.path.append("..")
from lib.utils import (
    MaskedMAELoss, print_log, seed_everything, set_cpu_num, CustomJSONEncoder
)
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from lib.data_prepare_multimodal import get_multimodal_dataloaders
from model.MSTHH import MSTHH

CUDA_LAUNCH_BLOCKING = 1


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_model(model, loader, criterion):
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        pred = SCALER.inverse_transform(pred)
        losses.append(criterion(pred, y).item())
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y_all, pred_all = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        pred = SCALER.inverse_transform(pred)
        pred_all.append(pred.cpu().numpy())
        y_all.append(y.cpu().numpy())
    return (np.vstack(y_all).squeeze(),
            np.vstack(pred_all).squeeze())


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    clip_grad, epoch, max_epochs):
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # MSTHH returns (pred, L_cl, L_ortho, L_recon) during training
        out = model(x)
        if isinstance(out, tuple):
            pred, L_cl, L_ortho, L_recon = out
        else:
            pred = out
            L_cl = L_ortho = L_recon = torch.tensor(0.0, device=DEVICE)

        pred = SCALER.inverse_transform(pred)

        # Adaptive multi-task loss (Eq.25-27)
        loss, _ = model.compute_total_loss(
            pred, y, L_cl, L_ortho, L_recon,
            criterion, epoch, max_epochs
        )

        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    scheduler.step()
    return float(np.mean(losses))


def train(model, train_loader, val_loader, optimizer, scheduler, criterion,
          clip_grad=0, max_epochs=200, early_stop=30, verbose=1,
          log=None, save=None):

    model = model.to(DEVICE)
    wait, min_val_loss = 0, np.inf
    train_losses, val_losses = [], []
    best_epoch, best_state = 0, None

    for epoch in range(max_epochs):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, clip_grad, epoch, max_epochs
        )
        val_loss = eval_model(model, val_loader, criterion)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                f"Epoch {epoch+1:4d}",
                f"  Train Loss = {tr_loss:.5f}",
                f"  Val Loss = {val_loss:.5f}",
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    tr_rmse, tr_mae, tr_mape = RMSE_MAE_MAPE(*predict(model, train_loader))
    va_rmse, va_mae, va_mape = RMSE_MAE_MAPE(*predict(model, val_loader))

    msg  = f"Early stopping at epoch: {epoch+1}\n"
    msg += f"Best at epoch {best_epoch+1}:\n"
    msg += f"Train Loss = {train_losses[best_epoch]:.5f}\n"
    msg += f"Train  RMSE={tr_rmse:.5f}  MAE={tr_mae:.5f}  MAPE={tr_mape:.5f}\n"
    msg += f"Val    Loss = {val_losses[best_epoch]:.5f}\n"
    msg += f"Val    RMSE={va_rmse:.5f}  MAE={va_mae:.5f}  MAPE={va_mape:.5f}"
    print_log(msg, log=log)

    if save:
        torch.save(best_state, save)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

def visual(true, preds=None, name="./pic/test.pdf"):
    plt.figure(figsize=(8, 4))
    plt.plot(true[:50], label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds[:50], label="Prediction", linewidth=2)
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(name, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def test_model(model, loader, log=None):
    model.eval()
    print_log("─" * 40 + " Test " + "─" * 40, log=log)

    os.makedirs("visualization_msthh", exist_ok=True)
    t0 = time.time()
    y_true, y_pred = predict(model, loader)
    t1 = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out = f"All Steps  RMSE={rmse_all:.5f}  MAE={mae_all:.5f}  MAPE={mape_all:.5f}\n"

    for i in range(y_pred.shape[1]):
        r, m, p = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out += f"Step {i+1:2d}    RMSE={r:.5f}  MAE={m:.5f}  MAPE={p:.5f}\n"
        visual(y_true[:, i, :].flatten(), y_pred[:, i, :].flatten(),
               os.path.join("visualization_msthh", f"test_step_{i+1}.pdf"))

    print_log(out, log=log, end="")
    print_log(f"Inference time: {t1-t0:.2f} s", log=log)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",    type=str,  default="pems08")
    parser.add_argument("-g", "--gpu_num",    type=int,  default=0)
    parser.add_argument("--multimodal",       action="store_true",
                        help="Use multimodal data loader (loads all traffic channels)")
    args = parser.parse_args()

    seed = torch.randint(1000, (1,))
    seed_everything(seed)
    set_cpu_num(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset    = args.dataset.upper()
    data_path  = f"../data/{dataset}"
    model_name = MSTHH.__name__

    with open("MSTHH.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # ── Build model ──────────────────────────────────────────────────────────
    model = MSTHH(**cfg["model_args"])

    # ── Logging ──────────────────────────────────────────────────────────────
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = "../logs/"
    os.makedirs(log_path, exist_ok=True)
    log = open(os.path.join(log_path, f"{model_name}-{dataset}-{now}.log"), "a")
    log.seek(0)
    log.truncate()

    # ── Data ─────────────────────────────────────────────────────────────────
    num_modalities = cfg["model_args"]["num_modalities"]
    print_log(dataset, log=log)

    if args.multimodal and num_modalities > 1:
        # Full multimodal loader
        (trainset_loader, valset_loader, testset_loader, SCALER
         ) = get_multimodal_dataloaders(
            data_path,
            num_modalities = num_modalities,
            tod            = cfg.get("time_of_day", True),
            dow            = cfg.get("day_of_week", True),
            batch_size     = cfg.get("batch_size", 16),
            log            = log,
        )
    else:
        # Single-modal loader (existing format: flow + tod + dow)
        if num_modalities > 1:
            print_log(
                "[INFO] num_modalities>1 but --multimodal not set; "
                "falling back to single-modal data. "
                "Model will replicate channel 0 for all modalities.",
                log=log
            )
        (trainset_loader, valset_loader, testset_loader, SCALER
         ) = get_dataloaders_from_index_data(
            data_path,
            tod        = cfg.get("time_of_day"),
            dow        = cfg.get("day_of_week"),
            batch_size = cfg.get("batch_size", 64),
            log        = log,
        )

    print_log(log=log)

    # ── Save path ────────────────────────────────────────────────────────────
    save_dir = "../saved_models/"
    os.makedirs(save_dir, exist_ok=True)
    save = os.path.join(save_dir, f"{model_name}-{dataset}-{now}.pt")

    # ── Loss & optimizer ─────────────────────────────────────────────────────
    if dataset in ("METRLA", "PEMSBAY"):
        criterion = MaskedMAELoss()
    elif dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg.get("weight_decay", 0),
        eps          = cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones = cfg["milestones"],
        gamma      = cfg.get("lr_decay_rate", 0.1),
        verbose    = False,
    )

    # ── Print summary ────────────────────────────────────────────────────────
    print_log(f"─── {model_name} ───", log=log)
    print_log(json.dumps(cfg, ensure_ascii=False, indent=2, cls=CustomJSONEncoder), log=log)

    sample_x = next(iter(trainset_loader))[0]
    print_log(summary(model, list(sample_x.shape), verbose=0), log=log)
    print_log(log=log)
    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(f"Multimodal mode: {args.multimodal}", log=log)
    print_log(log=log)

    # ── Train ────────────────────────────────────────────────────────────────
    model = train(
        model, trainset_loader, valset_loader,
        optimizer, scheduler, criterion,
        clip_grad  = cfg.get("clip_grad"),
        max_epochs = cfg.get("max_epochs", 200),
        early_stop = cfg.get("early_stop", 30),
        verbose    = 1,
        log        = log,
        save       = save,
    )

    print_log(f"Saved: {save}", log=log)

    # ── Test ─────────────────────────────────────────────────────────────────
    test_model(model, testset_loader, log=log)
    log.close()
