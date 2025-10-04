#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_mlp.py — Train/validate/test a simple MLP on fingerprint dataset.npz
Now also saves a CSV with test predictions vs labels.

Usage:
  python train_mlp.py --data-dir out_b1_reg --hidden 256,128 --epochs 80
  python train_mlp.py --data-dir out_b1_cls --task auto --class-weight 1
"""

import argparse, json, os, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import joblib

# ----------------------- utils -----------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dataset_npz(data_dir: Path):
    npz = np.load(data_dir / "dataset.npz", allow_pickle=True)
    X_train = npz["X_train"].astype(np.float32)
    y_train = npz["y_train"]
    X_val   = npz["X_val"].astype(np.float32)
    y_val   = npz["y_val"]
    X_test  = npz["X_test"].astype(np.float32)
    y_test  = npz["y_test"]
    feat_names = list(npz["feature_names"])
    # label stats for de-normalization (regression)
    stats_path = data_dir / "label_stats.json"
    label_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    # optional label encoder (classification)
    le_path = data_dir / "label_encoder.pkl"
    label_encoder = joblib.load(le_path) if le_path.exists() else None
    return (X_train, y_train, X_val, y_val, X_test, y_test, feat_names, label_stats, label_encoder)

def infer_task(y_train, task_arg: str):
    if task_arg != "auto":
        return task_arg
    return "cls" if (y_train.dtype.kind in "iu" and y_train.ndim == 1) else "reg"

def invert_labels(y_pred_norm: np.ndarray, label_stats: dict) -> np.ndarray:
    mode = label_stats.get("mode", "none")
    y = y_pred_norm.astype(np.float32)
    if mode == "minmax":
        y_min = np.array(label_stats.get("y_min"), dtype=np.float32)
        y_max = np.array(label_stats.get("y_max"), dtype=np.float32)
        if y_min.size == y.shape[1] and y_max.size == y.shape[1]:
            y = y * (y_max - y_min) + y_min
    elif mode == "zscore":
        ym = np.array(label_stats.get("y_mean"), dtype=np.float32)
        ys = np.array(label_stats.get("y_std"),  dtype=np.float32)
        if ym.size == y.shape[1] and ys.size == y.shape[1]:
            y = y * ys + ym
    return y

# ----------------------- model -----------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,128), dropout=0.1, use_bn=True):
        super().__init__()
        layers = []; prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h)]
            if use_bn: layers += [nn.BatchNorm1d(h)]
            layers += [nn.ReLU(inplace=True)]
            if dropout and dropout > 0: layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ----------------------- training -----------------------
def run_epoch(model, loader, task, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0; n = 0
    all_preds, all_targets = [], []
    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            if task == "cls":
                loss = criterion(out, yb.long())
                preds = out.argmax(dim=1)
            else:
                loss = criterion(out, yb.float())
                preds = out.detach()
            if is_train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * xb.size(0); n += xb.size(0)
            all_targets.append(yb.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
    avg_loss = total_loss / max(1, n)
    y_true = np.concatenate(all_targets, axis=0); y_pred = np.concatenate(all_preds, axis=0)
    metrics = {}
    if task == "cls":
        metrics["acc"] = float(accuracy_score(y_true, y_pred))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    else:
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        metrics["mae"] = mae; metrics["rmse"] = rmse
    return avg_loss, metrics

def save_test_csv(task, model, loader, device, out_csv: Path,
                  label_stats: dict, label_encoder=None):
    model.eval()
    rows = []
    with torch.no_grad():
        for idx, (xb, yb) in enumerate(loader):
            out = model(xb.to(device))
            if task == "cls":
                probs = torch.softmax(out, dim=1).cpu().numpy()
                y_pred = probs.argmax(axis=1)
                y_true = yb.numpy().astype(int)
                p_top1 = probs[np.arange(len(y_pred)), y_pred]
                for i in range(len(y_pred)):
                    row = {
                        "idx": idx * loader.batch_size + i,
                        "y_true": int(y_true[i]),
                        "y_pred": int(y_pred[i]),
                        "correct": int(y_true[i] == y_pred[i]),
                        "prob_top1": float(p_top1[i]),
                    }
                    if label_encoder is not None:
                        # map to string labels
                        try:
                            row["y_true_label"] = str(label_encoder.inverse_transform([y_true[i]])[0])
                            row["y_pred_label"] = str(label_encoder.inverse_transform([y_pred[i]])[0])
                        except Exception:
                            pass
                    rows.append(row)
            else:
                y_pred = out.cpu().numpy()
                y_true = yb.numpy().astype(np.float32)
                # per-sample errors in current label domain
                err = y_pred - y_true
                err_norm = np.sqrt(np.sum(err**2, axis=1))
                # de-normalize if possible
                y_pred_dn = invert_labels(y_pred, label_stats)
                y_true_dn = invert_labels(y_true, label_stats)
                err_dn = y_pred_dn - y_true_dn
                err_dn_norm = np.sqrt(np.sum(err_dn**2, axis=1))
                for i in range(len(y_pred)):
                    row = {
                        "idx": idx * loader.batch_size + i,
                        "y_true_0": float(y_true[i,0]),
                        "y_true_1": float(y_true[i,1]) if y_true.shape[1] > 1 else 0.0,
                        "y_pred_0": float(y_pred[i,0]),
                        "y_pred_1": float(y_pred[i,1]) if y_pred.shape[1] > 1 else 0.0,
                        "dx": float(err[i,0]),
                        "dy": float(err[i,1]) if err.shape[1] > 1 else 0.0,
                        "err_norm": float(err_norm[i]),
                        "y_true_0_denorm": float(y_true_dn[i,0]),
                        "y_true_1_denorm": float(y_true_dn[i,1]) if y_true_dn.shape[1] > 1 else 0.0,
                        "y_pred_0_denorm": float(y_pred_dn[i,0]),
                        "y_pred_1_denorm": float(y_pred_dn[i,1]) if y_pred_dn.shape[1] > 1 else 0.0,
                        "dx_denorm": float(err_dn[i,0]),
                        "dy_denorm": float(err_dn[i,1]) if err_dn.shape[1] > 1 else 0.0,
                        "err_denorm": float(err_dn_norm[i]),
                    }
                    rows.append(row)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved test predictions CSV → {out_csv.resolve()} (rows={len(df)})")

# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="dir containing dataset.npz etc.")
    ap.add_argument("--task", type=str, default="auto", choices=["auto","reg","cls"])
    ap.add_argument("--hidden", type=str, default="256,128")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bn", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--class-weight", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pred-csv", type=Path, default=None, help="where to save test predictions CSV")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("→ Device:", device)

    X_train, y_train, X_val, y_val, X_test, y_test, feat_names, label_stats, label_encoder = load_dataset_npz(args.data_dir)

    task = infer_task(y_train, args.task)
    print(f"→ Task: {task} | Features: {X_train.shape[1]} | Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # tensors & loaders
    X_train_t = torch.from_numpy(X_train); X_val_t = torch.from_numpy(X_val); X_test_t = torch.from_numpy(X_test)
    if task == "cls":
        y_train_t = torch.from_numpy(y_train.astype(np.int64))
        y_val_t   = torch.from_numpy(y_val.astype(np.int64))
        y_test_t  = torch.from_numpy(y_test.astype(np.int64))
        out_dim = int(np.max([y_train.max(), y_val.max(), y_test.max()])) + 1
    else:
        if y_train.ndim == 1:
            y_train = y_train[:, None]; y_val = y_val[:, None]; y_test = y_test[:, None]
        y_train_t = torch.from_numpy(y_train.astype(np.float32))
        y_val_t   = torch.from_numpy(y_val.astype(np.float32))
        y_test_t  = torch.from_numpy(y_test.astype(np.float32))
        out_dim   = y_train.shape[1]

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=args.batch_size, shuffle=False)

    # model
    hidden = tuple(int(x.strip()) for x in args.hidden.split(",") if x.strip())
    model = MLP(in_dim=X_train.shape[1], out_dim=out_dim, hidden=hidden,
                dropout=args.dropout, use_bn=bool(args.bn)).to(device)

    # loss
    if task == "cls":
        if bool(args.class_weight):
            classes, counts = np.unique(y_train, return_counts=True)
            weights = np.ones(out_dim, dtype=np.float32)
            weights[classes] = (1.0 / np.maximum(1, counts)).astype(np.float32)
            weights = weights / weights.mean()
            criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
            print("→ Using class weights")
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=("min" if task=="reg" else "max"), factor=0.5, patience=3, verbose=True
    )

    # train
    best_metric = None; wait = 0; best_path = args.data_dir / "best_mlp.pt"
    print("→ Start training")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, task, device, criterion, optimizer)
        val_loss,   val_metrics   = run_epoch(model, val_loader,   task, device, criterion, optimizer=None)

        if task == "cls":
            monitor = val_metrics["acc"]; scheduler.step(monitor)
            is_better = (best_metric is None) or (monitor > best_metric)
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} | acc={val_metrics['acc']:.4f} f1={val_metrics['f1_macro']:.4f}")
        else:
            monitor = -val_metrics["rmse"]; scheduler.step(val_metrics["rmse"])
            is_better = (best_metric is None) or (monitor > best_metric)
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} | rmse={val_metrics['rmse']:.4f} mae={val_metrics['mae']:.4f}")

        if is_better:
            best_metric = monitor; wait = 0
            torch.save({"model": model.state_dict(),
                        "in_dim": X_train.shape[1], "out_dim": out_dim,
                        "hidden": hidden, "dropout": args.dropout, "bn": bool(args.bn),
                        "task": task}, best_path)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"→ Early stopping at epoch {epoch}")
                break

    # eval on test
    print("→ Load best checkpoint & evaluate on test")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_metrics = run_epoch(model, test_loader, task, device, criterion, optimizer=None)
    print("Test metrics:", test_metrics)

    # extra: de-normalized metrics for regression
    if task == "reg":
        preds, gts = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb.to(device)).cpu().numpy()
                preds.append(out); gts.append(yb.numpy())
        preds = np.vstack(preds); gts = np.vstack(gts)
        preds_dn = invert_labels(preds, label_stats)
        gts_dn   = invert_labels(gts,   label_stats)
        rmse = float(np.sqrt(np.mean((preds_dn - gts_dn) ** 2)))
        mae  = float(np.mean(np.abs(preds_dn - gts_dn)))
        print(f"Test (de-normalized) → RMSE={rmse:.3f}  MAE={mae:.3f}")

    # save CSV
    out_csv = args.pred_csv if args.pred_csv is not None else (args.data_dir / "test_predictions.csv")
    save_test_csv(task, model, test_loader, device, out_csv, label_stats, label_encoder)

    print(f"✅ Best model saved at: {best_path.resolve()}")

if __name__ == "__main__":
    main()


# ## 回归（像素坐标）
# python train_mlp.py --data-dir out_b1_reg --hidden 256,128 --epochs 80 --pred-csv test_predictions.csv

# # 分类（网格 cell）
# python train_mlp.py --data-dir out_b1_cls --task auto --epochs 100 --class-weight 1 --pred-csv test_preds_cls.csv
