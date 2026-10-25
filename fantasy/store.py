"""Parquet storage: dated snapshots plus a cumulative, de-duplicated game log."""
import json
from pathlib import Path

import pandas as pd

from .config import DATA_DIR

SNAPSHOT_TABLES = ["teams", "rosters", "players", "matchups", "pro_schedule"]
LOG_KEY = ["player_id", "season", "scoring_period"]


def snapshot_dir(day: str, root: Path = DATA_DIR) -> Path:
    return root / "snapshots" / day


def save_snapshot(day: str, tables: dict[str, pd.DataFrame], meta: dict, root: Path = DATA_DIR):
    out = snapshot_dir(day, root)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def merge_game_logs(new: pd.DataFrame, root: Path = DATA_DIR) -> pd.DataFrame:
    """Append new per-game rows; a re-fetched game (stat corrections) replaces the old row."""
    path = root / "game_logs.parquet"
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    merged = new.drop_duplicates(LOG_KEY, keep="last").sort_values(LOG_KEY, ignore_index=True)
    root.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return merged


def list_snapshots(root: Path = DATA_DIR) -> list[str]:
    d = root / "snapshots"
    return sorted(p.name for p in d.iterdir() if (p / "meta.json").exists()) if d.exists() else []


def load_snapshot(day: str | None = None, root: Path = DATA_DIR) -> dict:
    """Load one day's snapshot (latest by default) plus the full game log."""
    days = list_snapshots(root)
    if not days:
        raise FileNotFoundError(f"No snapshots in {root / 'snapshots'}; run `python -m fantasy.collect`.")
    day = day or days[-1]
    d = snapshot_dir(day, root)
    snap = {name: pd.read_parquet(d / f"{name}.parquet")
            for name in SNAPSHOT_TABLES if (d / f"{name}.parquet").exists()}
    meta = json.loads((d / "meta.json").read_text())
    # JSON object keys are strings; restore the int keys used by the simulator.
    meta["lineup_slot_counts"] = {int(k): v for k, v in meta.get("lineup_slot_counts", {}).items()}
    meta["matchup_periods"] = {int(k): v for k, v in meta.get("matchup_periods", {}).items()}
    snap["meta"] = meta
    snap["day"] = day
    logs = root / "game_logs.parquet"
    snap["game_logs"] = pd.read_parquet(logs) if logs.exists() else pd.DataFrame(columns=LOG_KEY)
    return snap
