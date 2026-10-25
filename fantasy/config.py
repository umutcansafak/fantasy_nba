"""Settings loaded from the environment (.env locally, Actions secrets in CI)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
load_dotenv(ROOT / ".env")

# ESPN stat ids used by the simulator.
STAT_IDS = {
    "PTS": "0", "BLK": "1", "STL": "2", "AST": "3", "REB": "6", "TO": "11",
    "FGM": "13", "FGA": "14", "FTM": "15", "FTA": "16", "3PM": "17", "3PA": "18",
    "MIN": "40", "GP": "42",
}
STAT_NAMES = {v: k for k, v in STAT_IDS.items()}
# Raw per-game stats we keep in game logs (everything categories are built from).
RAW_STATS = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA", "MIN"]

# Category id (ESPN scoring statId) -> definition. Percentages are ratios of raw stats.
CATEGORIES = {
    "0": {"name": "PTS"}, "6": {"name": "REB"}, "3": {"name": "AST"},
    "2": {"name": "STL"}, "1": {"name": "BLK"}, "17": {"name": "3PM"},
    "19": {"name": "FG%", "ratio": ("FGM", "FGA")},
    "20": {"name": "FT%", "ratio": ("FTM", "FTA")},
    "11": {"name": "TO", "lower_is_better": True},
}
DEFAULT_CATEGORY_IDS = ["19", "20", "17", "6", "3", "2", "1", "11", "0"]

# Lineup slot ids (ESPN). BE/IR never score.
SLOT_NAMES = {
    0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C", 5: "G", 6: "F", 7: "SG/SF",
    8: "G/F", 9: "PF/C", 10: "F/C", 11: "UT", 12: "BE", 13: "IR", 15: "Rookie",
}
BENCH_SLOTS = {12, 13}

PRO_TEAMS = {
    0: "FA", 1: "ATL", 2: "BOS", 3: "NOP", 4: "CHI", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GSW", 10: "HOU", 11: "IND", 12: "LAC", 13: "LAL", 14: "MIA",
    15: "MIL", 16: "MIN", 17: "BKN", 18: "NYK", 19: "ORL", 20: "PHI", 21: "PHX",
    22: "POR", 23: "SAC", 24: "SAS", 25: "OKC", 26: "UTA", 27: "WAS", 28: "TOR",
    29: "MEM", 30: "CHA",
}


@dataclass(frozen=True)
class Settings:
    league_id: str
    team_id: int | None
    season: int
    espn_s2: str | None
    swid: str | None


def load_settings() -> Settings:
    team_id = os.getenv("TEAM_ID") or None
    return Settings(
        league_id=os.getenv("LEAGUE_ID", "").strip(),
        team_id=int(team_id) if team_id else None,
        season=int(os.getenv("SEASON", "2027")),
        espn_s2=os.getenv("ESPN_S2") or None,
        swid=os.getenv("SWID") or None,
    )
