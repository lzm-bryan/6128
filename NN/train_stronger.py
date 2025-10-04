#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_stronger.py — Stronger models for fingerprinting on dataset.npz
Models: LightGBM (lgbm), XGBoost (xgb), FT-Transformer (ftt)

Outputs:
- best model (in data_dir/)
- test predictions CSV (like train_mlp.py)
"""

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# common metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import MultiOutputRegressor

# ----- helpers -----
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)

def load_dataset_npz(data_dir: Path):
    npz = np.load(data_dir / "dataset.npz", allow_pickle=True)
    X_train = npz["X_train"].astype(np.float32)
    y_train = npz["y_train"]
    X_val   = npz["X_val"].astype(np.float32)
    y_val   = npz["y_val"]
    X_test  = npz["X_test"].astype(np.float32)
    y_test  = npz["y_test"]
    feat_names = list(npz["feature_names"])
    stats_path = data_dir / "label_stats.json"
    label_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    le_path = data_dir / "label_encoder.pkl"
    label_encoder = joblib.load(le_path) if le_path.exists() else None
    return X_train, y_train, X_val, y_val, X_test, y_test, feat_names, label_stats, label_encoder

def infer_task(y_train, task_arg):
    if task_arg != "auto": return task_arg
    return "cls" if (y_train.dtype.kind in "iu" and y_train.ndim==1) else "reg"

def invert_labels(y_pred, label_stats):
    mode = label_stats.get("mode","none")
    y = y_pred.astype(np.float32)
    if mode == "minmax":
        y_min = np.array(label_stats.get("y_min"), dtype=np.float32)
        y_max = np.array(label_stats.get("y_max"), dtype=np.float32)
        if y_min.size == y.shape[1] and y_max.size == y.shape[1]:
            y = y*(y_max-y_min)+y_min
    elif mode == "zscore":
        ym = np.array(label_stats.get("y_mean"), dtype=np.float32)
        ys = np.array(label_stats.get("y_std"),  dtype=np.float32)
        if ym.size == y.shape[1] and ys.size == y.shape[1]:
            y = y*ys+ym
    return y

def save_test_csv_reg(y_true, y_pred, label_stats, out_csv: Path):
    err = y_pred - y_true
    err_norm = np.sqrt(np.sum(err**2, axis=1))
    y_true_dn = invert_labels(y_true, label_stats)
    y_pred_dn = invert_labels(y_pred, label_stats)
    err_dn = y_pred_dn - y_true_dn
    err_dn_norm = np.sqrt(np.sum(err_dn**2, axis=1))
    rows = []
    for i in range(len(y_true)):
        rows.append({
            "idx": i,
            "y_true_0": float(y_true[i,0]),
            "y_true_1": float(y_true[i,1]) if y_true.shape[1]>1 else 0.0,
            "y_pred_0": float(y_pred[i,0]),
            "y_pred_1": float(y_pred[i,1]) if y_pred.shape[1]>1 else 0.0,
            "dx": float(err[i,0]),
            "dy": float(err[i,1]) if err.shape[1]>1 else 0.0,
            "err_norm": float(err_norm[i]),
            "y_true_0_denorm": float(y_true_dn[i,0]),
            "y_true_1_denorm": float(y_true_dn[i,1]) if y_true_dn.shape[1]>1 else 0.0,
            "y_pred_0_denorm": float(y_pred_dn[i,0]),
            "y_pred_1_denorm": float(y_pred_dn[i,1]) if y_pred_dn.shape[1]>1 else 0.0,
            "dx_denorm": float(err_dn[i,0]),
            "dy_denorm": float(err_dn[i,1]) if err_dn.shape[1]>1 else 0.0,
            "err_denorm": float(err_dn_norm[i]),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {out_csv} (rows={len(rows)})")

def save_test_csv_cls(y_true, y_pred, prob=None, label_encoder=None, out_csv: Path=None):
    rows = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        row = {"idx": i, "y_true": int(yt), "y_pred": int(yp), "correct": int(yt==yp)}
        if prob is not None:
            row["prob_top1"] = float(prob[i, yp])
        if label_encoder is not None:
            row["y_true_label"] = str(label_encoder.inverse_transform([yt])[0])
            row["y_pred_label"] = str(label_encoder.inverse_transform([yp])[0])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {out_csv} (rows={len(rows)})")

# ==================== LightGBM / XGBoost ====================
def train_lgbm(args, X_train, y_train, X_val, y_val, X_test, y_test, task, label_stats, label_encoder):
    from lightgbm import LGBMRegressor, LGBMClassifier
    pred_csv = args.pred_csv or (args.data_dir / "test_predictions_lgbm.csv")

    if task == "reg":
        # 多输出回归
        base = LGBMRegressor(
            n_estimators=4000, learning_rate=0.03, num_leaves=64,
            subsample=0.8, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=args.seed, n_jobs=-1
        )
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train, **{"verbose": False})
        # 验证与测试
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
        mae  = float(np.mean(np.abs(y_pred - y_test)))
        print(f"[LGBM] Test RMSE={rmse:.4f}  MAE={mae:.4f}")
        # 反归一化指标
        y_pred_dn = invert_labels(y_pred, label_stats)
        y_test_dn = invert_labels(y_test, label_stats)
        rmse_dn = float(np.sqrt(np.mean((y_pred_dn - y_test_dn)**2)))
        mae_dn  = float(np.mean(np.abs(y_pred_dn - y_test_dn)))
        print(f"[LGBM] Test (denorm) RMSE={rmse_dn:.2f}  MAE={mae_dn:.2f}")
        # CSV
        save_test_csv_reg(y_test, y_pred, label_stats, pred_csv)
        joblib.dump(model, args.data_dir / "best_lgbm.pkl")
        print("✅ Model saved:", (args.data_dir / "best_lgbm.pkl").resolve())
    else:
        n_classes = int(np.max([y_train.max(), y_val.max(), y_test.max()])) + 1
        objective = "binary" if n_classes == 2 else "multiclass"
        clf = LGBMClassifier(
            objective=objective, n_estimators=4000, learning_rate=0.05,
            num_leaves=64, subsample=0.8, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=args.seed, n_jobs=-1
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="multi_logloss" if n_classes>2 else "binary_logloss",
                verbose=False, early_stopping_rounds=100)
        prob = clf.predict_proba(X_test)
        if n_classes == 2 and prob.ndim == 1:
            prob = np.vstack([1-prob, prob]).T
        y_pred = np.argmax(prob, axis=1)
        acc = float(accuracy_score(y_test, y_pred))
        f1  = float(f1_score(y_test, y_pred, average="macro"))
        print(f"[LGBM] Test acc={acc:.4f}  f1_macro={f1:.4f}")
        save_test_csv_cls(y_test, y_pred, prob, label_encoder, pred_csv)
        joblib.dump(clf, args.data_dir / "best_lgbm.pkl")
        print("✅ Model saved:", (args.data_dir / "best_lgbm.pkl").resolve())

def train_xgb(args, X_train, y_train, X_val, y_val, X_test, y_test, task, label_stats, label_encoder):
    import xgboost as xgb
    pred_csv = args.pred_csv or (args.data_dir / "test_predictions_xgb.csv")

    if task == "reg":
        base = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.03, max_depth=8,
            subsample=0.8, colsample_bytree=0.9, reg_lambda=1.0,
            objective="reg:squarederror", random_state=args.seed, n_jobs=-1
        )
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
        mae  = float(np.mean(np.abs(y_pred - y_test)))
        print(f"[XGB] Test RMSE={rmse:.4f}  MAE={mae:.4f}")
        y_pred_dn = invert_labels(y_pred, label_stats)
        y_test_dn = invert_labels(y_test, label_stats)
        rmse_dn = float(np.sqrt(np.mean((y_pred_dn - y_test_dn)**2)))
        mae_dn  = float(np.mean(np.abs(y_pred_dn - y_test_dn)))
        print(f"[XGB] Test (denorm) RMSE={rmse_dn:.2f}  MAE={mae_dn:.2f}")
        save_test_csv_reg(y_test, y_pred, label_stats, pred_csv)
        joblib.dump(model, args.data_dir / "best_xgb.pkl")
        print("✅ Model saved:", (args.data_dir / "best_xgb.pkl").resolve())
    else:
        n_classes = int(np.max([y_train.max(), y_val.max(), y_test.max()])) + 1
        params = dict(
            n_estimators=3000, learning_rate=0.05, max_depth=9,
            subsample=0.8, colsample_bytree=0.9, reg_lambda=1.0,
            objective=("multi:softprob" if n_classes>2 else "binary:logistic"),
            random_state=args.seed, n_jobs=-1
        )
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=100)
        prob = clf.predict_proba(X_test)
        if n_classes == 2 and prob.ndim == 1:
            prob = np.vstack([1-prob, prob]).T
        y_pred = np.argmax(prob, axis=1)
        acc = float(accuracy_score(y_test, y_pred))
        f1  = float(f1_score(y_test, y_pred, average="macro"))
        print(f"[XGB] Test acc={acc:.4f}  f1_macro={f1:.4f}")
        save_test_csv_cls(y_test, y_pred, prob, label_encoder, pred_csv)
        joblib.dump(clf, args.data_dir / "best_xgb.pkl")
        print("✅ Model saved:", (args.data_dir / "best_xgb.pkl").resolve())

# ==================== FT-Transformer (PyTorch) ====================
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=192, depth=4, heads=8, dropout=0.1, out_dim=2, task="reg"):
        super().__init__()
        # 每个特征一个线性映射 -> token（TabTransformer 思路）
        self.feature_embeds = nn.ModuleList([nn.Linear(1, d_token) for _ in range(n_features)])
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=heads, dim_feedforward=d_token*4,
                                                   dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_token, out_dim)
        )
        self.task = task

    def forward(self, x):  # x: [B, F]
        # 转成 [B, F, 1] 再逐特征线性映射到 token
        B, F = x.shape
        tokens = []
        x1 = x.unsqueeze(-1)  # [B,F,1]
        for i in range(F):
            tokens.append(self.feature_embeds[i](x1[:, i]))  # [B,d_token]
        X = torch.stack(tokens, dim=1)  # [B,F,d_token]
        X = self.encoder(X)             # [B,F,d_token]
        X = self.norm(X.mean(dim=1))    # mean-pool
        out = self.head(X)              # [B,out_dim]
        return out

def train_ftt(args, X_train, y_train, X_val, y_val, X_test, y_test, task, label_stats, label_encoder):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    pred_csv = args.pred_csv or (args.data_dir / "test_predictions_ftt.csv")

    # tensors & loaders
    X_train_t = torch.from_numpy(X_train); X_val_t = torch.from_numpy(X_val); X_test_t = torch.from_numpy(X_test)
    if task == "cls":
        y_train_t = torch.from_numpy(y_train.astype(np.int64))
        y_val_t   = torch.from_numpy(y_val.astype(np.int64))
        y_test_t  = torch.from_numpy(y_test.astype(np.int64))
        out_dim = int(np.max([y_train.max(), y_val.max(), y_test.max()])) + 1
    else:
        if y_train.ndim == 1:
            y_train = y_train[:,None]; y_val = y_val[:,None]; y_test = y_test[:,None]
        y_train_t = torch.from_numpy(y_train.astype(np.float32))
        y_val_t   = torch.from_numpy(y_val.astype(np.float32))
        y_test_t  = torch.from_numpy(y_test.astype(np.float32))
        out_dim = y_train.shape[1]

    tr = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    va = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=args.batch_size)
    te = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=args.batch_size)

    model = FTTransformer(n_features=X_train.shape[1], d_token=args.hidden, depth=args.depth,
                          heads=args.heads, dropout=args.dropout, out_dim=out_dim, task=task).to(device)
    if task == "cls":
        crit = nn.CrossEntropyLoss()
    else:
        crit = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode=("min" if task=="reg" else "max"),
                                                           factor=0.5, patience=3, verbose=True)

    best_metric, wait = None, 0
    best_path = args.data_dir / "best_ftt.pt"

    for epoch in range(1, args.epochs+1):
        # train
        model.train(); tot, n = 0.0, 0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb if task=="cls" else yb.float())
            optim.zero_grad(); loss.backward(); optim.step()
            tot += loss.item()*xb.size(0); n += xb.size(0)
        tr_loss = tot/max(1,n)

        # val
        model.eval(); tot, n = 0.0, 0
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = crit(out, yb if task=="cls" else yb.float())
                tot += loss.item()*xb.size(0); n += xb.size(0)
                preds.append(out.detach().cpu().numpy()); gts.append(yb.detach().cpu().numpy())
        va_loss = tot/max(1,n)
        preds = np.vstack(preds); gts = np.vstack(gts)

        if task == "cls":
            val_metric = float(accuracy_score(gts, preds.argmax(axis=1)))
            scheduler.step(val_metric)
            better = (best_metric is None) or (val_metric > best_metric)
            print(f"[Epoch {epoch:03d}] tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} | acc={val_metric:.4f}")
        else:
            rmse = float(np.sqrt(np.mean((preds - gts)**2)))
            scheduler.step(rmse)
            better = (best_metric is None) or (-rmse > best_metric)
            print(f"[Epoch {epoch:03d}] tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} | rmse={rmse:.4f}")

        if better:
            best_metric = (val_metric if task=="cls" else -rmse); wait = 0
            torch.save({"model": model.state_dict(),
                        "cfg": dict(n_features=X_train.shape[1], d_token=args.hidden,
                                    depth=args.depth, heads=args.heads, dropout=args.dropout,
                                    out_dim=out_dim, task=task)}, best_path)
        else:
            wait += 1
            if wait >= args.patience:
                print("→ Early stopping"); break

    # test & CSV
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in te:
            out = model(xb.to(device)).cpu().numpy()
            preds.append(out); gts.append(yb.numpy())
    y_pred = np.vstack(preds); y_true = np.vstack(gts)

    if task == "cls":
        y_pred_cls = y_pred.argmax(axis=1)
        acc = float(accuracy_score(y_true, y_pred_cls))
        f1  = float(f1_score(y_true, y_pred_cls, average="macro"))
        print(f"[FTT] Test acc={acc:.4f}  f1={f1:.4f}")
        prob = np.exp(y_pred - y_pred.max(axis=1, keepdims=True))
        prob = prob / prob.sum(axis=1, keepdims=True)
        save_test_csv_cls(y_true.astype(int), y_pred_cls.astype(int), prob, label_encoder,
                          args.pred_csv or (args.data_dir / "test_predictions_ftt.csv"))
    else:
        rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
        mae  = float(np.mean(np.abs(y_pred - y_true)))
        print(f"[FTT] Test RMSE={rmse:.4f}  MAE={mae:.4f}")
        save_test_csv_reg(y_true, y_pred, label_stats,
                          args.pred_csv or (args.data_dir / "test_predictions_ftt.csv"))
    print("✅ Best FTT saved:", (args.data_dir / "best_ftt.pt").resolve())

# ==================== main ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--task", type=str, default="auto", choices=["auto","reg","cls"])
    ap.add_argument("--model", type=str, required=True, choices=["lgbm","xgb","ftt"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pred-csv", type=Path, default=None)

    # FT-Transformer hyperparams
    ap.add_argument("--hidden", type=int, default=192, help="FTT token dim")
    ap.add_argument("--depth", type=int, default=4, help="FTT layers")
    ap.add_argument("--heads", type=int, default=8, help="FTT heads")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    set_seed(args.seed)
    X_tr, y_tr, X_va, y_va, X_te, y_te, feats, label_stats, label_encoder = load_dataset_npz(args.data_dir)
    task = infer_task(y_tr, args.task)
    print(f"→ Model: {args.model} | Task: {task} | dim={X_tr.shape[1]} | Train/Val/Test = {len(X_tr)}/{len(X_va)}/{len(X_te)}")

    if args.model == "lgbm":
        train_lgbm(args, X_tr, y_tr, X_va, y_va, X_te, y_te, task, label_stats, label_encoder)
    elif args.model == "xgb":
        train_xgb(args, X_tr, y_tr, X_va, y_va, X_te, y_te, task, label_stats, label_encoder)
    else:
        train_ftt(args, X_tr, y_tr, X_va, y_va, X_te, y_te, task, label_stats, label_encoder)

if __name__ == "__main__":
    main()


# # 1) LightGBM（推荐先试）
# pip install lightgbm xgboost  # 若未安装
# python train_stronger.py --model lgbm --data-dir out_b1_reg --pred-csv test_pred_lgbm.csv
#
# # 2) XGBoost
# python train_stronger.py --model xgb --data-dir out_b1_reg --pred-csv test_pred_xgb.csv
#
# # 3) FT-Transformer（深度模型）
# python train_stronger.py --model ftt --data-dir out_b1_reg --hidden 256 --depth 4 --heads 8 --epochs 120
