# Fantasy NBA matchup simulator

Pulls your ESPN H2H 9-cat league every day and, for any week, simulates both
lineups 5,000 times from players' real box scores, giving a projected range and
a P(win) for each category, plus which free agents would swing it.

## How the data is fetched
ESPN has no official API, but the fantasy site is a JS app reading JSON from
`lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/2027/...`.
`fantasy/espn_client.py` calls those endpoints directly:

| Data | Endpoint / view | Auth |
|---|---|---|
| Teams, rosters, lineup slots, matchup schedule & weekly category scores | `leagues/{id}?view=mTeam,mRoster,mMatchup,mSettings` | league cookies |
| Player pool (rostered + free agents), injuries, season / last-15 / projected splits, last 40 per-game box scores | `leagues/{id}?view=kona_player_info` | league cookies |
| NBA schedule (games per team per day) | `seasons/2027?view=proTeamSchedules_wl` | public |

## Setup
1. `python3 -m venv .venv && .venv/bin/pip install --only-binary=:all: -r requirements.txt`
2. `cp .env.example .env` and fill in:
   - `LEAGUE_ID`: from `fantasy.espn.com/basketball/league?leagueId=...`
   - `TEAM_ID`: from your team page URL (`teamId=...`)
   - `ESPN_S2`, `SWID`: Chrome DevTools → Application → Cookies → `https://fantasy.espn.com`
     (needed for private leagues; they last for months; refresh them if you get 401s)
3. `.venv/bin/python -m fantasy.collect`, which writes `data/snapshots/<today>/` and `data/game_logs.parquet`
4. `.venv/bin/streamlit run app.py`

Try it without a league: `python -m fantasy.collect --public` stores the public
player pool, and the app then shows a demo league of two drafted teams.

## Daily collection (GitHub Actions)
Push this folder to GitHub. The league (286070) is public, so no secrets are
needed; if it ever goes private, add `ESPN_S2` and `SWID` as repo secrets. `.github/workflows/collect.yml` runs every day at
10:00 UTC and commits the new snapshot; `git pull` before opening the dashboard.
Run it once by hand from the Actions tab (“Run workflow”).

## Shared report (GitHub Pages)
After collecting, the same workflow runs `python -m fantasy.report` and publishes
`site/index.html` to https://umutcansafak.github.io/fantasy_nba/: where the live week stands (score now,
projected final, Locked/Swing/Lost per category), my matchup for
the live week (ESPN totals so far + simulated remaining days) and next week's
preview, with free-agent pickups. Build it locally with `python -m fantasy.report`.

## This week (`fantasy/live.py`)
The dashboard opens on the live matchup against your scheduled opponent: score now
(categories led / trailed / tied from ESPN's recorded totals), projected final and
chance to win, starts left, and each category marked Locked (≥90%), Leaning win,
Swing (35–65%), Leaning loss or Lost (≤10%) for the rest of the week.

## How the projection works (`fantasy/projection.py`)
- Week = the league's matchup period (scoring period = one day). On Sunday the default is next week.
- Each day, players whose NBA team plays are ranked by a 9-cat z-score value
  and placed into active slots (`fantasy/lineup.py`); extras sit.
- OUT/suspended players are skipped until their expected return date; day-to-day players play with probability 0.7.
- Each start is sampled from the player's recent games (this season weighted 3× over last season's last 20).
  Those last-season games are the season's final stretch, so for players with 50+ games last season each
  stat is rescaled to their full-season per-game average (variation kept, level corrected).
  Players with fewer than 5 games get Poisson games around their projection.
- FG%/FT% are total makes / total attempts; TO is lower-is-better; ties count as half.

`python -m tests.smoke_test [data_root]` runs lineup unit checks and an end-to-end simulation.

## Trade ideas (`pages/1_Trades.py`, `fantasy/trades.py`)
Streamlit-only (not on the shared Pages site). For the rest of the season, each team's
strength = expected categories won per week vs the average opponent, from players'
per-game means/variances × remaining games (playoff weeks ×2). Every 1-for-1 and 2-for-2
swap with every team is scored; the best per partner are re-checked with real daily
lineups. Win-win = both teams' strength rises. Fairness (0-100) = ½ gain balance +
½ "looks even" by standard 9-cat player value.
