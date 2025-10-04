#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_fingerprint_dataset.py
将 fingerprint CSV 预处理为 MLP 可用的 .npz（train/val/test）+ 预处理器文件。

示例：
  # 回归：预测像素坐标，坐标归一化到 0~1（读取 floor_info.json 获取宽高）
  python preprocess_fingerprint_dataset.py --csv b1_dataset.csv --task reg \
      --label-space px --normalize-labels minmax --floor-dir .\site1\B1 --out-dir out_b1_reg

  # 分类：预测网格 cell（需要生成 CSV 时使用过 --grid-px）
  python preprocess_fingerprint_dataset.py --csv b1_grid_50px.csv --task cls \
      --out-dir out_b1_cls --stratify 1
"""

import argparse, json, os, re, math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

# tqdm 可选
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---------- utils ----------
def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def read_floor_size(floor_dir: Optional[Path]) -> Tuple[Optional[float], Optional[float]]:
    """从 floor_info.json 读取楼层宽高（像素）。"""
    if not floor_dir:
        return None, None
    fi = floor_dir / "floor_info.json"
    if not fi.exists():
        return None, None
    obj = load_json(fi)
    if isinstance(obj, dict):
        return float(obj["map_info"]["width"]), float(obj["map_info"]["height"])
    if isinstance(obj, list) and obj:
        return float(obj[0]["map_info"]["width"]), float(obj[0]["map_info"]["height"])
    return None, None

def pick_feature_columns(df: pd.DataFrame,
                         use_mag: bool = True,
                         use_acc: bool = True,
                         use_gyr: bool = True) -> List[str]:
    """根据前缀挑选特征列。"""
    cols = []
    def add_by_prefix(prefixes):
        for p in prefixes:
            cols.extend([c for c in df.columns if c.startswith(p)])
    if use_mag:
        add_by_prefix(["bx_", "by_", "bz_", "bmag_"])
    if use_acc:
        add_by_prefix(["anorm_"])
    if use_gyr:
        add_by_prefix(["gnorm_"])
    # 去重并保持稳定顺序
    seen, ordered = set(), []
    for c in cols:
        if c not in seen:
            ordered.append(c); seen.add(c)
    return ordered

def maybe_progress(iterable, desc="", total=None, unit="it"):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit, ncols=100)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="由 prep_fingerprint_csv.py 生成的 CSV")
    ap.add_argument("--task", type=str, default="reg", choices=["reg","cls"], help="回归或分类")
    ap.add_argument("--label-space", type=str, default="px", choices=["px","m"], help="回归标签坐标系")
    ap.add_argument("--normalize-labels", type=str, default="none", choices=["none","minmax","zscore"],
                    help="对回归标签进行归一化（minmax 依赖楼层宽高或数据范围）")
    ap.add_argument("--floor-dir", type=Path, default=None, help="提供 floor_info.json 以便 label minmax 使用 W,H")
    ap.add_argument("--drop-out-of-range", type=int, default=1, help="回归：剔除超出 [0,W/H] 的样本")
    ap.add_argument("--use-mag", type=int, default=1)
    ap.add_argument("--use-acc", type=int, default=1)
    ap.add_argument("--use-gyr", type=int, default=1)
    ap.add_argument("--min-mag", type=int, default=3, help="最少磁力样本的过滤阈值（列 n_mag）")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--stratify", type=int, default=1, help="分类任务是否分层抽样")
    ap.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取 CSV
    print(f"→ 读取：{args.csv}")
    df = pd.read_csv(args.csv)

    # 2) 过滤：至少有一定数量的磁力数据
    if "n_mag" in df.columns and args.min_mag > 0:
        before = len(df)
        df = df[df["n_mag"] >= args.min_mag].copy()
        print(f"→ 过滤 n_mag >= {args.min_mag}: {before} -> {len(df)}")

    # 3) 选择特征列
    feat_cols = pick_feature_columns(df, bool(args.use_mag), bool(args.use_acc), bool(args.use_gyr))
    if not feat_cols:
        raise RuntimeError("没有可用的特征列（检查 CSV 是否包含 bx_*/by_*/bz_* 等列）")
    print(f"→ 特征数：{len(feat_cols)}")
    (args.out_dir / "feature_names.txt").write_text("\n".join(feat_cols), encoding="utf-8")

    X = df[feat_cols].to_numpy(dtype=np.float32)

    # 4) 构建标签
    if args.task == "reg":
        if args.label_space == "px":
            y = df[["label_x_px","label_y_px"]].to_numpy(dtype=np.float32)
            W, H = read_floor_size(args.floor_dir)
            # 剔除越界
            if bool(args.drop_out_of_range) and W is not None and H is not None:
                mask = (y[:,0] >= 0) & (y[:,0] <= W) & (y[:,1] >= 0) & (y[:,1] <= H)
                b0 = len(y); y = y[mask]; X = X[mask]; df = df.loc[mask]
                print(f"→ 回归标签越界过滤：{b0} -> {len(y)}")
        else:
            y = df[["label_x_m","label_y_m"]].to_numpy(dtype=np.float32)
            W = H = None

        # 标签归一化
        label_stats = {}
        if args.normalize_labels == "minmax":
            if args.label_space == "px" and (W is not None and H is not None):
                y_min = np.array([0.0, 0.0], dtype=np.float32)
                y_max = np.array([float(W), float(H)], dtype=np.float32)
            else:
                y_min = np.nanmin(y, axis=0); y_max = np.nanmax(y, axis=0)
            y_scale = (y_max - y_min); y_scale[y_scale == 0] = 1.0
            y = (y - y_min) / y_scale
            label_stats = {"mode":"minmax","y_min":y_min.tolist(),"y_max":y_max.tolist()}
        elif args.normalize_labels == "zscore":
            ym = np.nanmean(y, axis=0); ys = np.nanstd(y, axis=0); ys[ys==0]=1.0
            y = (y - ym) / ys
            label_stats = {"mode":"zscore","y_mean":ym.tolist(),"y_std":ys.tolist()}
        else:
            label_stats = {"mode":"none"}

    else:  # classification
        if "label_cell_id" not in df.columns:
            raise RuntimeError("分类任务需要 CSV 中存在 label_cell_id（生成 CSV 时应设置 --grid-px）")
        labels = df["label_cell_id"].astype(str).to_numpy()
        le = LabelEncoder()
        y = le.fit_transform(labels).astype(np.int64)
        joblib.dump(le, args.out_dir / "label_encoder.pkl")
        print(f"→ 类别数：{len(le.classes_)}（已保存 label_encoder.pkl）")
        label_stats = {"mode":"classes","classes":le.classes_.tolist()}

    # 5) 缺失值 & 标准化（基于训练集拟合）
    # 先划分临时 train/tmp，再从 tmp 划分 val/test，避免数据泄露
    test_ratio = args.test_ratio
    val_ratio = args.val_ratio
    assert 0 < test_ratio < 1 and 0 < val_ratio < 1 and (test_ratio + val_ratio) < 1

    stratify_arr = None
    if args.task == "cls" and bool(args.stratify):
        stratify_arr = y
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(test_ratio + val_ratio), random_state=args.random_state, stratify=stratify_arr
    )
    # 计算 val 占 X_tmp 的比例
    val_ratio_rel = val_ratio / (test_ratio + val_ratio)
    stratify_tmp = y_tmp if (args.task=="cls" and bool(args.stratify)) else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1 - val_ratio_rel), random_state=args.random_state, stratify=stratify_tmp
    )

    print(f"→ 划分完成：train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)

    X_val = scaler.transform(imputer.transform(X_val))
    X_test = scaler.transform(imputer.transform(X_test))

    # 6) 保存
    np.savez_compressed(
        args.out_dir / "dataset.npz",
        X_train=X_train.astype(np.float32),
        y_train=y_train,
        X_val=X_val.astype(np.float32),
        y_val=y_val,
        X_test=X_test.astype(np.float32),
        y_test=y_test,
        feature_names=np.array(feat_cols, dtype=object)
    )
    joblib.dump(imputer, args.out_dir / "imputer.pkl")
    joblib.dump(scaler, args.out_dir / "scaler.pkl")
    (args.out_dir / "label_stats.json").write_text(json.dumps(label_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7) 一点可读性日志
    def describe(arr, name):
        if arr.ndim == 2 and arr.shape[1] > 1:
            print(f"  {name}: shape={arr.shape}")
        else:
            cls_or_reg = "classes" if args.task=="cls" else "targets"
            print(f"  {name}: shape={arr.shape} ({cls_or_reg})")
    print("✅ 已保存：", (args.out_dir / "dataset.npz").resolve())
    describe(X_train, "X_train"); describe(X_val, "X_val"); describe(X_test, "X_test")

if __name__ == "__main__":
    main()

# python preprocess_fingerprint_dataset.py --csv b1_dataset.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site1\B1 --out-dir out_b1_reg
# b1_dataset_dense

# 怎么跑
# 回归（预测像素坐标，0~1 归一化）
# python preprocess_fingerprint_dataset.py \
#   --csv b1_dataset.csv \
#   --task reg --label-space px --normalize-labels minmax \
#   --floor-dir .\site1\B1 \
#   --out-dir out_b1_reg
#
# 分类（预测网格 cell）
#
# （确保你的 CSV 是用 --grid-px 生成的，label_cell_id 不为空）
#
# python preprocess_fingerprint_dataset.py \
#   --csv b1_grid_50px.csv \
#   --task cls --stratify 1 \
#   --out-dir out_b1_cls
#
# 产物说明（在 out_dir）
#
# dataset.npz：包含 X_train, y_train, X_val, y_val, X_test, y_test, feature_names
#
# imputer.pkl、scaler.pkl：训练用的缺失值填充器和标准化器（推理时复用）
#
# label_encoder.pkl（仅分类）：类别编码器
#
# label_stats.json：坐标归一化参数或类别列表
#
# feature_names.txt：使用的特征列名（方便你在 MLP 中核对输入维度）