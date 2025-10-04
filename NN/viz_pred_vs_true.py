#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
viz_pred_vs_true.py — visualize predictions vs ground-truth for indoor fingerprinting

Inputs: CSV from train_mlp.py (columns may include *_denorm, or normalized ones)
Plots:
  1) Floor overlay (if --floor-dir given): scatter GT vs Pred with connecting lines, colored by error
  2) Error histogram (and key percentiles)
  3) Parity plots: x_true vs x_pred, y_true vs y_pred

Usage:
  python viz_pred_vs_true.py --csv out_b1_reg/test_predictions.csv --floor-dir .\site1\B1 --out b1_predviz.png

  # 没有 floor 图，直接用归一化坐标：
  python viz_pred_vs_true.py --csv out_b1_reg/test_predictions.csv --use-norm 1 --out b1_predviz_norm.png

  # 只画前 1000 个点（避免太密）
  python viz_pred_vs_true.py --csv out_b1_reg/test_predictions.csv --floor-dir .\site1\B1 --max-n 1000
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_floor_size(floor_dir: Path):
    fi = floor_dir / "floor_info.json"
    if not fi.exists():
        return None, None
    obj = json.loads(fi.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        return float(obj["map_info"]["width"]), float(obj["map_info"]["height"])
    if isinstance(obj, list) and obj:
        return float(obj[0]["map_info"]["width"]), float(obj[0]["map_info"]["height"])
    return None, None

def load_floor_image(floor_dir: Path):
    imgp = floor_dir / "floor_image.png"
    if imgp.exists():
        import matplotlib.image as mpimg
        return mpimg.imread(imgp)
    return None

def pick_cols(df: pd.DataFrame, use_norm: bool):
    """
    Return (x_true, y_true, x_pred, y_pred, err, label) arrays.
    Prefer *_denorm columns unless use_norm=1.
    """
    if not use_norm and {"y_true_0_denorm","y_true_1_denorm","y_pred_0_denorm","y_pred_1_denorm"}.issubset(df.columns):
        x_t = df["y_true_0_denorm"].astype(float).to_numpy()
        y_t = df["y_true_1_denorm"].astype(float).to_numpy()
        x_p = df["y_pred_0_denorm"].astype(float).to_numpy()
        y_p = df["y_pred_1_denorm"].astype(float).to_numpy()
        err = df["err_denorm"].astype(float).to_numpy() if "err_denorm" in df.columns else np.sqrt((x_p-x_t)**2 + (y_p-y_t)**2)
        label = "(pixel/metric domain)"
    else:
        # fallback to normalized domain
        x_t = df["y_true_0"].astype(float).to_numpy()
        y_t = df["y_true_1"].astype(float).to_numpy()
        x_p = df["y_pred_0"].astype(float).to_numpy()
        y_p = df["y_pred_1"].astype(float).to_numpy()
        err = df["err_norm"].astype(float).to_numpy() if "err_norm" in df.columns else np.sqrt((x_p-x_t)**2 + (y_p-y_t)**2)
        label = "(normalized domain)"
    return x_t, y_t, x_p, y_p, err, label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="test_predictions.csv")
    ap.add_argument("--floor-dir", type=Path, default=None, help="包含 floor_info.json / floor_image.png 的目录")
    ap.add_argument("--use-norm", type=int, default=0, help="1=使用归一化坐标列(y_*), 0=优先使用 *_denorm 列")
    ap.add_argument("--max-n", type=int, default=0, help="最多绘制多少个样本（0=不限制）")
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--ms", type=float, default=10, help="散点大小")
    ap.add_argument("--out", type=Path, default=Path("pred_vs_true.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.max_n and args.max_n > 0 and len(df) > args.max_n:
        df = df.iloc[:args.max_n].copy()

    x_t, y_t, x_p, y_p, err, label_domain = pick_cols(df, bool(args.use_norm))

    # 统计 & 打印
    def pct(x, q): return float(np.percentile(x, q))
    mae = float(np.mean(np.sqrt((x_p-x_t)**2 + (y_p-y_t)**2)))
    print("→ Samples:", len(df))
    print("→ Error stats ({}):".format(label_domain))
    print("   mean={:.3f}  median(P50)={:.3f}  P75={:.3f}  P90={:.3f}  max={:.3f}".format(
        float(np.mean(err)), pct(err,50), pct(err,75), pct(err,90), float(np.max(err))
    ))

    # 画布：3 列
    fig = plt.figure(figsize=(16, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1], wspace=0.25)

    # ── (1) overlay
    ax0 = fig.add_subplot(gs[0,0])
    if args.floor_dir:
        W, H = load_floor_size(args.floor_dir)
        img = load_floor_image(args.floor_dir)
        if img is not None and W is not None and H is not None:
            # 注意：楼层像素坐标通常原点左上，所以用 origin='upper'
            ax0.imshow(img, extent=[0, W, 0, H], origin='upper', alpha=0.7)
            ax0.set_xlim(0, W); ax0.set_ylim(H, 0)  # y 轴朝下以贴合像素坐标
        else:
            ax0.set_aspect('equal')
    # 用误差上色
    sc1 = ax0.scatter(x_t, y_t, s=args.ms, c='lime', alpha=args.alpha, label='GT')
    sc2 = ax0.scatter(x_p, y_p, s=args.ms, c=err, cmap='viridis', alpha=args.alpha, label='Pred')
    # 连线
    for xt, yt, xp, yp in zip(x_t, y_t, x_p, y_p):
        ax0.plot([xt, xp], [yt, yp], linewidth=0.6, color='black', alpha=0.25)
    cb = fig.colorbar(sc2, ax=ax0, fraction=0.035, pad=0.02)
    cb.set_label(f'Error {label_domain}')
    ax0.set_title(f"Pred vs GT on floor {label_domain}")
    ax0.legend(loc='upper right', frameon=True)

    # ── (2) error histogram
    ax1 = fig.add_subplot(gs[0,1])
    ax1.hist(err, bins=40, alpha=0.9)
    ax1.set_title("Error histogram")
    ax1.set_xlabel(f"Error {label_domain}")
    ax1.set_ylabel("Count")
    for q, c in [(50, 'r'), (75, 'orange'), (90, 'g')]:
        v = np.percentile(err, q)
        ax1.axvline(v, color=c, linestyle='--', linewidth=1)
        ax1.text(v, ax1.get_ylim()[1]*0.95, f"P{q}:{v:.1f}", color=c, rotation=90, va='top')

    # ── (3) parity plots
    ax2 = fig.add_subplot(gs[0,2])
    # x 轴
    ax2.scatter(x_t, x_p, s=8, alpha=0.6, label='x')
    # y 轴
    ax2.scatter(y_t, y_p, s=8, alpha=0.6, label='y')
    lo = float(min(np.min(x_t), np.min(y_t), np.min(x_p), np.min(y_p)))
    hi = float(max(np.max(x_t), np.max(y_t), np.max(x_p), np.max(y_p)))
    ax2.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title("Parity: true vs pred")
    ax2.set_xlabel(f"True {label_domain}")
    ax2.set_ylabel(f"Pred {label_domain}")
    ax2.legend()

    fig.suptitle(f"Samples={len(df)} | mean err={np.mean(err):.2f} | P50={np.percentile(err,50):.2f} | P90={np.percentile(err,90):.2f}", y=1.02)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"✅ Saved figure: {args.out.resolve()}")

if __name__ == "__main__":
    main()


# python viz_pred_vs_true.py --csv out_b1_reg/test_predictions.csv --floor-dir .\site1\B1 --out b1_predviz.png

# 看一部分
# python viz_pred_vs_true.py --csv out_b1_reg/test_predictions.csv --floor-dir .\site1\B1 --max-n 1500
