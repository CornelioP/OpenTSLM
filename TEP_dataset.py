import os
import kagglehub
import pyreadr
import pandas as pd
from typing import Dict, Any, List

# ---------- I/O helpers ----------

def read_R_to_Pandas(file_path: str) -> pd.DataFrame:
    rdata: Dict[str, Any] = pyreadr.read_r(file_path)
    if not rdata:
        raise ValueError(f"No objects found in {file_path}")
    # If multiple objects exist, pick the first dataframe-like
    for k, v in rdata.items():
        if isinstance(v, pd.DataFrame):
            df = v
            break
    else:
        # Fallback: take first value
        df = next(iter(rdata.values()))
    # enforce dtypes
    for c in ["faultNumber", "simulationRun", "sample"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    # sort for determinism
    sort_cols = [c for c in ["faultNumber", "simulationRun", "sample"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    print(f"[{os.path.basename(file_path)}] shape={df.shape}")
    return df

def clean_data(df: pd.DataFrame, drop_faults: List[int] = [3, 9, 15]) -> pd.DataFrame:
    if "faultNumber" in df.columns:
        return df[~df["faultNumber"].isin(drop_faults)].reset_index(drop=True)
    return df

# ---------- Split by simulationRun (prevents leakage) ----------

def split_by_run(df_faultfree: pd.DataFrame,
                 df_faulty: pd.DataFrame,
                 train_ratio: float = 0.8,
                 random_state: int = 42):
    # Work only on TRAIN files for train/val split
    # Get unique run ids per fault number to keep class balance
    rng = pd.Series([0], dtype=int).sample(n=1, random_state=random_state)  # seed

    def split_one(df: pd.DataFrame):
        # If no runs column, fallback to simple split
        if "simulationRun" not in df.columns:
            n = len(df)
            cut = int(train_ratio * n)
            return df.iloc[:cut].copy(), df.iloc[cut:].copy()
        parts = []
        parts_val = []
        if "faultNumber" in df.columns:
            groups = df.groupby("faultNumber", dropna=False)
        else:
            groups = [(None, df)]
        for fault, g in groups:
            runs = g["simulationRun"].dropna().unique()
            runs = sorted([int(r) for r in runs])
            if len(runs) == 0:
                # all normal without runs? just row split
                cut = int(train_ratio * len(g))
                parts.append(g.iloc[:cut].copy())
                parts_val.append(g.iloc[cut:].copy())
                continue
            k = max(1, int(round(train_ratio * len(runs))))
            # deterministic split
            train_runs = set(runs[:k])
            g_tr = g[g["simulationRun"].isin(train_runs)]
            g_va = g[~g["simulationRun"].isin(train_runs)]
            parts.append(g_tr)
            parts_val.append(g_va)
        df_tr = pd.concat(parts, ignore_index=True)
        df_va = pd.concat(parts_val, ignore_index=True)
        return df_tr, df_va

    tr_ff, va_ff = split_one(df_faultfree)
    tr_fy, va_fy = split_one(df_faulty)

    train_df = pd.concat([tr_ff, tr_fy], ignore_index=True)
    val_df   = pd.concat([va_ff, va_fy], ignore_index=True)

    # final sort
    sort_cols = [c for c in ["faultNumber", "simulationRun", "sample"] if c in train_df.columns]
    if sort_cols:
        train_df = train_df.sort_values(sort_cols).reset_index(drop=True)
        val_df   = val_df.sort_values(sort_cols).reset_index(drop=True)
    return train_df, val_df

# ---------- Main ----------

if __name__ == "__main__":
    path = kagglehub.dataset_download("averkij/tennessee-eastman-process-simulation-dataset")

    def P(name): return os.path.join(path, name)

    # Load
    df_train_faultfree = read_R_to_Pandas(P("TEP_FaultFree_Training.RData"))
    df_test_faultfree  = read_R_to_Pandas(P("TEP_FaultFree_Testing.RData"))
    df_train_faulty    = read_R_to_Pandas(P("TEP_Faulty_Training.RData"))
    df_test_faulty     = read_R_to_Pandas(P("TEP_Faulty_Testing.RData"))

    # Clean (configurable)
    df_train_faultfree = clean_data(df_train_faultfree)
    df_test_faultfree  = clean_data(df_test_faultfree)
    df_train_faulty    = clean_data(df_train_faulty)
    df_test_faulty     = clean_data(df_test_faulty)

    # Sanity: consistent feature columns
    feat_cols = sorted([c for c in df_train_faultfree.columns if c.startswith("xmeas_") or c.startswith("xmv_")])
    for dname, df in {
        "train_ff": df_train_faultfree,
        "test_ff":  df_test_faultfree,
        "train_fy": df_train_faulty,
        "test_fy":  df_test_faulty,
    }.items():
        f = sorted([c for c in df.columns if c.startswith("xmeas_") or c.startswith("xmv_")])
        if f != feat_cols:
            missing = set(feat_cols) - set(f)
            extra   = set(f) - set(feat_cols)
            raise ValueError(f"{dname} has different feature columns.\nMissing: {missing}\nExtra: {extra}")

    # Split TRAIN files into train/val by simulationRun
    trainingData, validationData = split_by_run(df_train_faultfree, df_train_faulty, train_ratio=0.8)

    # TEST = concatenation of the two test files
    testingData = pd.concat([df_test_faultfree, df_test_faulty], ignore_index=True)
    testingData = testingData.sort_values(["faultNumber","simulationRun","sample"]).reset_index(drop=True)

    # Info
    print("Training shape:", trainingData.shape)
    print("Validation shape:", validationData.shape)
    print("Testing shape:", testingData.shape)

    # # Optional: save to disk
    # out_dir = "data/tep_csv"
    # os.makedirs(out_dir, exist_ok=True)
    # trainingData.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    # validationData.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    # testingData.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    # # or faster/lighter:
    # # trainingData.to_parquet(os.path.join(out_dir, "train.parquet"))
