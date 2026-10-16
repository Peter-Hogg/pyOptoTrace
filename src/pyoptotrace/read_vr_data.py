import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def read_virmen_log(
    path,
    ncols=12,
    dtype=np.float64,
    candidate_ncols=(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
    column_names=None,
):
    """
    Parameters
    ----------
    path : str | Path
        Path to *_virmenLog.data
    ncols : int or None
        Number of columns per iteration. If None, will try candidate_ncols.
    dtype : numpy dtype
        Usually float64 (MATLAB fwrite default for a double array).
    candidate_ncols : iterable[int]
        Tried in order if ncols is None; first one that divides cleanly wins.
    column_names : list[str] or None
        If given, must match ncols. If None, columns are 'c0', 'c1', ...

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)
    raw = np.fromfile(path, dtype=dtype)
    n = raw.size
    if n == 0:
        raise ValueError(f"{path} is empty")

    if ncols is None:
        for c in candidate_ncols:
            if n % c == 0:
                ncols = c
                break
        if ncols is None:
            raise ValueError(
                f"Could not infer ncols. Total doubles = {n}. "
                f"None of {candidate_ncols} divide evenly. "
                f"Open your runtimeCodeFun and find the fwrite call to count columns."
            )

    if n % ncols != 0:
        raise ValueError(
            f"{n} doubles is not divisible by ncols={ncols}. "
            f"Either ncols is wrong or the file was truncated mid-write."
        )

    data = raw.reshape(-1, ncols)

    if column_names is None:
        column_names = [f"c{i}" for i in range(ncols)]
    elif len(column_names) != ncols:
        raise ValueError(f"column_names has {len(column_names)} entries but ncols={ncols}")

    return pd.DataFrame(data, columns=column_names)


def clean_virmen_log(df, smooth_velocity_ms=50):
    """
    Parameters
    ----------
    df : DataFrame from read_virmen_log (columns c0..cN)
    smooth_velocity_ms : float
        Causal moving-average window for velocity, in ms. Set to 0 to disable.

    Returns
    -------
    DataFrame with renamed/derived columns:
        t_datenum    : original MATLAB datenum (kept for traceability)
        t            : seconds since first sample
        timestamp    : pandas datetime (UTC-naive, MATLAB local clock)
        dt           : per-sample interval (s); first row = NaN
        position     : track position (raw c4)
        lap          : lap number (int)
        imaging      : imaging gate (bool)
        velocity     : d(position)/dt (units/s), optionally smoothed
        imaging_on   : True at frames where imaging just turned on
        imaging_off  : True at frames where imaging just turned off
        lap_start    : True at frames where lap incremented
    """
    MATLAB_EPOCH_OFFSET = 719529
    out = df.copy()

    # --- time axis ---
    t_datenum = out["c0"].to_numpy()
    out["t_datenum"] = t_datenum
    out["t"] = (t_datenum - t_datenum[0]) * 86400.0
    out["timestamp"] = pd.to_datetime(t_datenum - MATLAB_EPOCH_OFFSET, unit="D")
    dt = np.diff(out["t"].to_numpy(), prepend=np.nan)
    out["dt"] = dt

    # --- position / velocity ---
    out = out.rename(columns={"c4": "position", "c7": "lap", "c9": "imaging"})
    out["lap"] = out["lap"].astype(np.int64)
    out["imaging"] = out["imaging"].astype(bool)

    pos = out["position"].to_numpy()
    raw_vel = np.empty_like(pos)
    raw_vel[0] = np.nan
    raw_vel[1:] = np.diff(pos) / dt[1:]

    if smooth_velocity_ms and smooth_velocity_ms > 0:
        # window length in samples, using median dt to be robust to dropped frames
        median_dt = np.nanmedian(dt[1:])
        win = max(1, int(round((smooth_velocity_ms / 1000.0) / median_dt)))
        # simple causal moving average that tolerates NaNs
        s = pd.Series(raw_vel).rolling(win, min_periods=1).mean()
        out["velocity"] = s.to_numpy()
    else:
        out["velocity"] = raw_vel

    # --- event edges ---
    img = out["imaging"].to_numpy()
    out["imaging_on"] = np.r_[False, (img[1:] & ~img[:-1])]
    out["imaging_off"] = np.r_[False, (~img[1:] & img[:-1])]

    lap = out["lap"].to_numpy()
    out["lap_start"] = np.r_[False, np.diff(lap) > 0]
    out = out.assign(
        hhmmss=lambda x: (
            x["timestamp"].dt.hour * 10000
            + x["timestamp"].dt.minute * 100
            + x["timestamp"].dt.second
        )
    )
    return out


def imaging_epochs(df):
    """Return list of (start_idx, end_idx_exclusive, t_start, t_end) for each
    contiguous imaging==True run."""
    img = df["imaging"].to_numpy()
    if not img.any():
        return []
    edges = np.diff(img.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if img[0]:
        starts = np.r_[0, starts]
    if img[-1]:
        ends = np.r_[ends, len(img)]
    t = df["t"].to_numpy()
    return [(s, e, t[s], t[e - 1]) for s, e in zip(starts, ends)]


def lap_summary(df):
    """One row per lap: start/end time, duration, mean velocity, distance,
    fraction of time imaged."""
    g = df.groupby("lap", sort=True)
    summary = pd.DataFrame(
        {
            "t_start": g["t"].first(),
            "t_end": g["t"].last(),
            "duration_s": g["t"].last() - g["t"].first(),
            "n_samples": g.size(),
            "pos_start": g["position"].first(),
            "pos_end": g["position"].last(),
            "pos_min": g["position"].min(),
            "pos_max": g["position"].max(),
            "mean_velocity": g["velocity"].mean(),
            "frac_imaged": g["imaging"].mean(),
        }
    )
    return summary.reset_index()


def add_epoch_id(df, col="epoch_id", off_value=pd.NA):
    """
    Tag each imaging epoch with a unique integer ID.
    Non-imaging rows get `off_value` (default pd.NA; use -1 if you want an int dtype).
    """
    img = df["imaging"].to_numpy()
    # cumulative count of imaging_on edges → 0 before first epoch, 1 during first, etc.
    edges = np.r_[False, (img[1:] & ~img[:-1])]  # imaging_on
    epoch_counter = np.cumsum(edges)  # 0,0,...,1,1,1,...,2,2,2,...

    if off_value is pd.NA:
        # use nullable integer dtype so it stays an int with NA where imaging is off
        ids = pd.array(np.where(img, epoch_counter, 0), dtype="Int64")
        ids[~img] = pd.NA
    else:
        ids = np.where(img, epoch_counter, off_value).astype(np.int64)

    df = df.copy()
    df[col] = ids
    return df


def match_acquisitions(virmen_path, df):
    """
    Match Luminos acquisition folders (HHMMSS-named) to ViRMEn imaging-on events.

    Returns
    -------
    dict mapping folder_hhmmss (int) -> dict with:
        't_start'   : ViRMEn elapsed time at imaging-on (seconds)
        'hhmmss'    : ViRMEn HHMMSS at imaging-on
        'offset_s'  : virmen - folder, in seconds-of-day
        'df_idx'    : row index in df where imaging turned on
        'epoch_id'  : epoch ID assigned by add_epoch_id (if present)
    """
    if "epoch_id" not in list(df.columns):
        df = add_epoch_id(df, col="epoch_id", off_value=pd.NA)

    imaging_folders = glob.glob(os.path.dirname(virmen_path) + "/*/")
    folder_hhmmss = []
    for p in imaging_folders:
        name = os.path.basename(os.path.normpath(p))
        if name[:6] == "Snaps" or not name[:6].isdigit():
            continue
        folder_hhmmss.append(int(name[:6]))

    def _sod(h):
        return (h // 10000) * 3600 + ((h // 100) % 100) * 60 + (h % 100)

    on_rows = df[df["imaging_on"]]
    has_epoch_id = "epoch_id" in df.columns

    out = {}
    used_virmen = set()
    for f in sorted(folder_hhmmss):
        f_sod = _sod(f)
        best = None
        for i, row in on_rows.iterrows():
            if i in used_virmen:
                continue
            off = _sod(int(row["hhmmss"])) - f_sod
            if abs(off) <= 5 and (best is None or abs(off) < abs(best[2])):
                best = (i, row, off)
        if best is None:
            continue
        i, row, off = best
        used_virmen.add(i)
        out[f] = {
            "t_start": float(row["t"]),
            "hhmmss": int(row["hhmmss"]),
            "offset_s": off,
            "df_idx": int(i),
            "epoch_id": int(row["epoch_id"]) if has_epoch_id else None,
        }
    return out
