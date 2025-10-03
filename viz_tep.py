#!/usr/bin/env python3
import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------
# Helpers
# -------------------------------

def detect_feature_columns(df: pd.DataFrame) -> List[str]:
    feats = [c for c in df.columns if c.startswith("xmeas_") or c.startswith("xmv_")]
    # stable order: family then index
    feats.sort(key=lambda s: (s.split("_")[0], int(s.split("_")[1])))
    return feats

def find_injection_index(g: pd.DataFrame) -> Optional[int]:
    """
    Returns the row index (relative to g) of the first time the fault switches
    from 0 to >0 within a single (faultNumber, simulationRun) group.
    If it never switches, returns None.
    """
    if "faultNumber" not in g.columns:
        return None
    f = g["faultNumber"].to_numpy()
    if len(f) == 0:
        return None
    # In many TEP files, for a faulted run the first segment is normal (0),
    # then switches to the run's fault id. Find first nonzero after leading zeros.
    nz = np.where(f != 0)[0]
    if nz.size == 0:
        return None
    inj = int(nz[0])
    # sanity: require that before inj we were 0
    if inj > 0 and int(f[inj - 1]) == 0:
        return inj
    return None

def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd == 0] = 1.0
    return (x - mu) / sd

def robust_unit(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd == 0] = 1.0
    z = (x - mu) / (3.0 * sd)
    return np.clip(z, -1.0, 1.0)

def downsample(t: np.ndarray, X: np.ndarray, every: int) -> Tuple[np.ndarray, np.ndarray]:
    if every <= 1:
        return t, X
    return t[::every], X[::every, :]


# -------------------------------
# Core plotting
# -------------------------------

def plot_run_timeseries(
    df: pd.DataFrame,
    fault: int,
    run: int,
    features: Optional[List[str]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    normalize: Optional[str] = None,  # None | 'z' | 'unit'
    rolling: Optional[int] = None,    # e.g., 5 for simple smoothing
    downsample_every: int = 1,
    sample_col: str = "sample",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    savepath: Optional[str] = None,
):
    """
    Plot selected features for one (faultNumber, simulationRun) trace.
    - normalize: 'z' => z-score; 'unit' => robust +/-3σ -> [-1, 1]
    - rolling: window size (in samples) for simple moving average per channel
    - downsample_every: plot every k-th point for speed
    """

    required = {"faultNumber", "simulationRun", sample_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Select this run
    g = df[(df["faultNumber"] == fault) & (df["simulationRun"] == run)].copy()
    if g.empty:
        raise ValueError(f"No data for faultNumber={fault}, simulationRun={run}")

    # Time index
    if sample_col in g.columns:
        t = g[sample_col].to_numpy()
    else:
        t = np.arange(len(g), dtype=int)

    # Feature columns
    all_feats = detect_feature_columns(df)
    if features:
        # keep order as given by user but validate they exist
        bad = [c for c in features if c not in all_feats]
        if bad:
            raise ValueError(f"Requested features not found: {bad}")
        feat_cols = features
    else:
        feat_cols = all_feats

    # Cut interval
    if start is None:
        start = 0
    if end is None or end > len(g):
        end = len(g)
    if start < 0 or end <= start:
        raise ValueError("Invalid start/end range")
    g = g.iloc[start:end].reset_index(drop=True)
    t = t[start:end]

    # Assemble X
    X = g[feat_cols].to_numpy(dtype=float)

    # Optional rolling mean per channel
    if rolling and rolling > 1:
        X = pd.DataFrame(X).rolling(rolling, min_periods=1, center=False).mean().to_numpy()

    # Normalization
    if normalize == "z":
        X = zscore(X)
    elif normalize == "unit":
        X = robust_unit(X)

    # Downsample for speed
    t, X = downsample(t, X, downsample_every)

    # Injection index (relative to ORIGINAL group), then map to sliced/downsampled index
    inj_abs = find_injection_index(df[(df["faultNumber"] == fault) & (df["simulationRun"] == run)])
    inj_rel = None
    if inj_abs is not None:
        if start <= inj_abs < end:
            inj_rel = inj_abs - start
            if downsample_every > 1:
                inj_rel = inj_rel // downsample_every

    # Plot
    plt.figure(figsize=figsize)
    for j, col in enumerate(feat_cols):
        plt.plot(t, X[:, j], label=col, linewidth=1.0)

    if inj_rel is not None:
        plt.axvline(x=t[inj_rel], linestyle="--", linewidth=1.5)
        plt.text(t[inj_rel], plt.gca().get_ylim()[1], " injection", va="top", ha="left")

    ttl = title or f"TEP | fault={fault} run={run} | {len(feat_cols)} signals"
    if normalize:
        ttl += f" | norm={normalize}"
    if rolling and rolling > 1:
        ttl += f" | roll={rolling}"
    if downsample_every > 1:
        ttl += f" | ↓x{downsample_every}"
    plt.title(ttl)
    plt.xlabel(sample_col)
    plt.ylabel("value")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=150)
        print(f"Saved: {savepath}")
    else:
        plt.show()
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    fault: int,
    run: int,
    features: Optional[List[str]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    method: str = "pearson",
    figsize: Tuple[int, int] = (8, 7),
    savepath: Optional[str] = None,
):
    """
    Quick correlation view for a run/interval.
    """
    g = df[(df["faultNumber"] == fault) & (df["simulationRun"] == run)].copy()
    if g.empty:
        raise ValueError(f"No data for faultNumber={fault}, simulationRun={run}")

    if start is None:
        start = 0
    if end is None or end > len(g):
        end = len(g)
    g = g.iloc[start:end]

    feat_cols = features or detect_feature_columns(df)
    X = g[feat_cols]

    C = X.corr(method=method)

    plt.figure(figsize=figsize)
    plt.imshow(C.values, aspect="auto", interpolation="nearest")
    plt.xticks(ticks=np.arange(len(feat_cols)), labels=feat_cols, rotation=90, fontsize=7)
    plt.yticks(ticks=np.arange(len(feat_cols)), labels=feat_cols, fontsize=7)
    plt.colorbar(label=f"{method} correlation")
    plt.title(f"Correlation | fault={fault} run={run} | {start}:{end}")
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=150)
        print(f"Saved: {savepath}")
    else:
        plt.show()
    plt.close()


# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="TEP plotting utility")
    ap.add_argument("--csv", required=True, help="Path to a TEP CSV (train/val/test)")
    ap.add_argument("--fault", type=int, required=True, help="faultNumber to visualize (0=Normal)")
    ap.add_argument("--run", type=int, required=True, help="simulationRun to visualize")
    ap.add_argument("--features", nargs="*", default=None, help="feature names (e.g., xmeas_1 xmeas_5 xmv_3). Default: all")
    ap.add_argument("--start", type=int, default=None, help="start sample index")
    ap.add_argument("--end", type=int, default=None, help="end sample index (exclusive)")
    ap.add_argument("--normalize", choices=[None, "z", "unit"], default=None, help="channel-wise normalization")
    ap.add_argument("--rolling", type=int, default=None, help="rolling-mean window size")
    ap.add_argument("--downsample", type=int, default=1, help="plot every k-th sample")
    ap.add_argument("--corr", action="store_true", help="plot correlation heatmap instead of timeseries")
    ap.add_argument("--out", default=None, help="optional path to save the figure")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.corr:
        plot_correlation_heatmap(
            df=df,
            fault=args.fault,
            run=args.run,
            features=args.features,
            start=args.start,
            end=args.end,
            savepath=args.out,
        )
    else:
        plot_run_timeseries(
            df=df,
            fault=args.fault,
            run=args.run,
            features=args.features,
            start=args.start,
            end=args.end,
            normalize=args.normalize,
            rolling=args.rolling,
            downsample_every=args.downsample,
            savepath=args.out,
        )

if __name__ == "__main__":
    main()
