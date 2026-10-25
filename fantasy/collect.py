"""Daily collector: `python -m fantasy.collect` (add --public to test without a league)."""
import argparse
from datetime import datetime

import pandas as pd

from .config import DATA_DIR, DEFAULT_CATEGORY_IDS, load_settings
from .espn_client import ET, EspnClient, parse_league, parse_players, parse_pro_schedule
from .nba_schedule import DEFAULT_SLOT_COUNTS, period_for, resolve_matchup_periods
from . import store


def collect(public: bool = False, root=DATA_DIR, pool_size: int = 450) -> dict:
    settings = load_settings()
    client = EspnClient(settings)
    today = datetime.now(ET).date()

    pro = parse_pro_schedule(client.pro_schedule())
    if public:
        league = {
            "teams": pd.DataFrame(columns=["team_id", "abbrev", "name", "owner", "wins", "losses", "ties"]),
            "rosters": pd.DataFrame(columns=["team_id", "player_id", "lineup_slot"]),
            "matchups": pd.DataFrame(),
            "meta": {"league_name": "Public player pool (no league)",
                     "category_ids": DEFAULT_CATEGORY_IDS,
                     "lineup_slot_counts": DEFAULT_SLOT_COUNTS,
                     "matchup_weeks": None,
                     "current_scoring_period": period_for(today, pro)},
        }
    else:
        league = parse_league(client.league())

    meta = league["meta"]
    meta["matchup_periods"] = resolve_matchup_periods(pro, meta.pop("matchup_weeks"))
    live = meta.get("current_matchup_days")
    if live and not set(live) <= set(meta["matchup_periods"].get(meta["current_matchup_period"], [])):
        print(f"WARNING: ESPN's live matchup covers days {live}, which doesn't match the computed "
              f"week {meta['current_matchup_period']}; week boundaries may be off.")
    meta.update(season=settings.season, team_id=settings.team_id, collected_at=datetime.now(ET).isoformat(),
                public=public)
    period = meta["current_scoring_period"] or 1
    pool = client.player_pool(period, limit=pool_size, public=public)
    # deep-bench rostered players can fall outside the top-N by ownership
    missing = set(league["rosters"]["player_id"]) - {e["player"]["id"] for e in pool}
    if missing:
        pool += client.player_pool(period, limit=len(missing), ids=sorted(missing))
    players, logs = parse_players(pool, settings.season)

    tables = {"teams": league["teams"], "rosters": league["rosters"], "players": players,
              "matchups": league["matchups"], "pro_schedule": pro}
    day = today.isoformat()
    store.save_snapshot(day, tables, meta, root)
    all_logs = store.merge_game_logs(logs, root)
    return {"day": day, "teams": len(league["teams"]), "rostered": len(league["rosters"]),
            "players": len(players), "free_agents": int((players["on_team_id"] == 0).sum()),
            "game_logs_today": len(logs), "game_logs_total": len(all_logs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--public", action="store_true", help="skip the league; fetch the public player pool only")
    ap.add_argument("--pool-size", type=int, default=450)
    args = ap.parse_args()
    summary = collect(public=args.public, pool_size=args.pool_size)
    for k, v in summary.items():
        print(f"{k:>16}: {v}")


if __name__ == "__main__":
    main()
