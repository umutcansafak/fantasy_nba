"""Thin client for ESPN's (undocumented) fantasy basketball JSON API + normalizers.

The fantasy site is a JS app that reads these same endpoints. Private leagues
need the `espn_s2` and `SWID` cookies from a logged-in browser.
"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .config import CATEGORIES, DEFAULT_CATEGORY_IDS, PRO_TEAMS, RAW_STATS, STAT_IDS, Settings

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}"
ET = ZoneInfo("America/New_York")
# Public pseudo-league ESPN uses for its own player pages; works without cookies.
PUBLIC_LEAGUE = "leaguedefaults/3"


class EspnError(RuntimeError):
    pass


class EspnClient:
    def __init__(self, settings: Settings, timeout: int = 60):
        self.settings = settings
        self.base = BASE.format(season=settings.season)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "fantasy-nba-sim/0.1"
        if settings.espn_s2 and settings.swid:
            self.session.cookies.set("espn_s2", settings.espn_s2)
            self.session.cookies.set("SWID", settings.swid)

    def _get(self, url, params=None, fantasy_filter=None):
        headers = {"x-fantasy-filter": json.dumps(fantasy_filter)} if fantasy_filter else None
        r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
        if r.status_code in (401, 403):
            raise EspnError(
                f"ESPN returned {r.status_code}: the league is private and the "
                "ESPN_S2/SWID cookies are missing or expired."
            )
        if r.status_code == 404:
            raise EspnError(f"ESPN returned 404 for {r.url}: check LEAGUE_ID and SEASON.")
        r.raise_for_status()
        data = r.json()
        return data[0] if isinstance(data, list) else data

    @property
    def league_url(self):
        if not self.settings.league_id:
            raise EspnError("LEAGUE_ID is not set (see .env.example).")
        return f"{self.base}/segments/0/leagues/{self.settings.league_id}"

    def league(self):
        views = ["mTeam", "mRoster", "mMatchup", "mSettings", "mStandings", "mStatus"]
        return self._get(self.league_url, params={"view": views})

    def pro_schedule(self):
        return self._get(self.base, params={"view": "proTeamSchedules_wl"})

    def player_pool(self, scoring_period: int, limit: int = 450, top_periods: int = 40,
                    public: bool = False, page_size: int = 150, ids: list[int] | None = None):
        """Players sorted by % owned, with season splits and recent per-game logs.

        `top_periods` per-game splits come back per player and reach into last
        season early in the year, which is what lets ranges be estimated in week 1.
        """
        url = f"{self.base}/segments/0/{PUBLIC_LEAGUE}" if public else self.league_url
        season = self.settings.season
        players = []
        for offset in range(0, limit, page_size):
            flt = {"players": {
                "limit": min(page_size, limit - offset),
                "offset": offset,
                "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
                "filterStatsForTopScoringPeriodIds": {
                    "value": top_periods,
                    "additionalValue": [f"00{season}", f"10{season}", f"02{season}", f"00{season - 1}"],
                },
            }}
            if ids:
                flt["players"]["filterIds"] = {"value": list(ids)}
            elif not public:
                flt["players"]["filterStatus"] = {"value": ["FREEAGENT", "WAIVERS", "ONTEAM"]}
            page = self._get(url, params={"view": "kona_player_info", "scoringPeriodId": scoring_period},
                             fantasy_filter=flt)["players"]
            players.extend(page)
            if len(page) < page_size:
                break
        return players


# ---------------------------------------------------------------- normalizers

def _stat_row(stats: dict) -> dict:
    return {name: float(stats.get(STAT_IDS[name], 0.0)) for name in RAW_STATS}


def parse_pro_schedule(data) -> pd.DataFrame:
    """One row per (NBA team, scoring period) game, with the game's ET calendar date."""
    rows = []
    for team in data["settings"]["proTeams"]:
        for period, games in team.get("proGamesByScoringPeriod", {}).items():
            for g in games:
                rows.append({
                    "pro_team_id": team["id"],
                    "pro_team": PRO_TEAMS.get(team["id"], team.get("abbrev")),
                    "scoring_period": int(period),
                    "game_id": g["id"],
                    "date": datetime.fromtimestamp(g["date"] / 1000, ET).date(),
                    "opponent": PRO_TEAMS.get(
                        g["awayProTeamId"] if g["homeProTeamId"] == team["id"] else g["homeProTeamId"]),
                })
    return pd.DataFrame(rows)


def parse_players(pool, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (players, game_logs) frames from a kona_player_info response."""
    players, logs = [], []
    for entry in pool:
        p = entry["player"]
        expected = p.get("expectedReturnDate")
        row = {
            "player_id": p["id"],
            "name": p["fullName"],
            "pro_team_id": p.get("proTeamId", 0),
            "pro_team": PRO_TEAMS.get(p.get("proTeamId", 0), "FA"),
            "eligible_slots": [s for s in p.get("eligibleSlots", [])],
            "injury_status": p.get("injuryStatus") or "ACTIVE",
            "expected_return": pd.Timestamp(*expected[:3]).date() if expected else None,
            "on_team_id": entry.get("onTeamId", 0),
            "status": entry.get("status"),
            "pct_owned": p.get("ownership", {}).get("percentOwned", 0.0),
        }
        for split in p.get("stats", []):
            stats = split.get("stats") or {}
            sid, split_type = split["id"], split.get("statSplitTypeId")
            if split_type == 5 and stats:  # one game; empty stats = did not play
                logs.append({"player_id": p["id"], "season": split["seasonId"],
                             "scoring_period": split["scoringPeriodId"], **_stat_row(stats)})
            elif sid in (f"00{season}", f"10{season}", f"02{season}", f"00{season - 1}") and stats:
                key = {f"00{season}": "season", f"10{season}": "proj",
                       f"02{season}": "last15", f"00{season - 1}": "prev"}[sid]
                gp = float(stats.get(STAT_IDS["GP"], 0.0))
                row[f"{key}_gp"] = gp
                for name, value in _stat_row(stats).items():
                    row[f"{key}_{name}"] = value / gp if gp else 0.0
        players.append(row)
    pl = pd.DataFrame(players)
    gl = pd.DataFrame(logs, columns=["player_id", "season", "scoring_period", *RAW_STATS])
    return pl, gl.drop_duplicates(["player_id", "season", "scoring_period"])


def parse_league(data) -> dict:
    """Teams, rosters, matchup schedule/results and the settings the sim needs."""
    settings = data.get("settings", {})
    teams, rosters = [], []
    members = {m["id"]: f"{m.get('firstName', '')} {m.get('lastName', '')}".strip()
               for m in data.get("members", [])}
    for t in data.get("teams", []):
        name = t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip()
        rec = t.get("record", {}).get("overall", {})
        teams.append({
            "team_id": t["id"], "abbrev": t.get("abbrev"), "name": name,
            "owner": ", ".join(members.get(o, "") for o in t.get("owners", [])),
            "wins": rec.get("wins", 0), "losses": rec.get("losses", 0), "ties": rec.get("ties", 0),
        })
        for e in t.get("roster", {}).get("entries", []):
            rosters.append({"team_id": t["id"], "player_id": e["playerId"],
                            "lineup_slot": e.get("lineupSlotId")})

    matchups = []
    for m in data.get("schedule", []):
        sides = [s for s in ("home", "away") if s in m]
        for s in sides:
            other = m.get("away" if s == "home" else "home", {})
            cum = m[s].get("cumulativeScore", {})
            base = {"matchup_period": m["matchupPeriodId"], "team_id": m[s]["teamId"],
                    "opponent_id": other.get("teamId"), "winner": m.get("winner"),
                    "cat_wins": cum.get("wins"), "cat_losses": cum.get("losses"), "cat_ties": cum.get("ties")}
            by_stat = cum.get("scoreByStat") or {}
            if not by_stat:
                matchups.append({**base, "stat_id": None, "score": None})
            for stat_id, v in by_stat.items():
                matchups.append({**base, "stat_id": stat_id, "score": v.get("score")})

    scoring_items = settings.get("scoringSettings", {}).get("scoringItems", [])
    cat_ids = [str(i["statId"]) for i in scoring_items if str(i["statId"]) in CATEGORIES]
    status = data.get("status", {})
    # days ESPN has already attached to the live matchup; used to sanity-check week boundaries
    live_days = sorted({int(d) for m in data.get("schedule", [])
                        if m.get("matchupPeriodId") == status.get("currentMatchupPeriod")
                        for side in ("home", "away") for d in m.get(side, {}).get("pointsByScoringPeriod", {})})
    return {
        "teams": pd.DataFrame(teams),
        "rosters": pd.DataFrame(rosters, columns=["team_id", "player_id", "lineup_slot"]),
        "matchups": pd.DataFrame(matchups),
        "meta": {
            "league_name": settings.get("name"),
            "category_ids": cat_ids or DEFAULT_CATEGORY_IDS,
            "lineup_slot_counts": {int(k): v for k, v in
                                   settings.get("rosterSettings", {}).get("lineupSlotCounts", {}).items()},
            # matchup period -> calendar weeks; collect resolves these to days
            "matchup_weeks": {int(k): v for k, v in
                                settings.get("scheduleSettings", {}).get("matchupPeriods", {}).items()},
            "current_matchup_period": status.get("currentMatchupPeriod"),
            "current_matchup_days": live_days,
            "current_scoring_period": data.get("scoringPeriodId") or status.get("latestScoringPeriod"),
            "first_scoring_period": status.get("firstScoringPeriod", 1),
            "final_scoring_period": status.get("finalScoringPeriod"),
        },
    }
