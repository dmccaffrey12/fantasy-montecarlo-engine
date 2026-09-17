"""
data_ingest.py
Modular Ingestion Layer for Fantasy Football Monte Carlo Engine.
- The Odds API: Spreads, Over/Unders, Implied Totals, and Player Props (yards, receptions, anytime TD).
- Polymarket Gamma API: Crowd-sourced event and injury probabilities.
- ESPN League Sync: Active rosters, starters/bench, weekly matchups, and top 100 free agents via espn-api.
- Sleeper NFL Universe: Active player database fallback.
- Unified Projection Parameter Engine (mu, sigma).
"""

import os
import re
import requests
import datetime
import urllib.parse
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
import scoring_config

# Fallback / Default API keys and URLs
DEFAULT_ODDS_API_KEY = ""
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
POLYMARKET_BASE_URL = "https://gamma-api.polymarket.com"
SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"


def get_odds_api_key() -> str:
    """
    Retrieve The Odds API key from st.secrets or environment variable.
    Supports flat keys, case-insensitive variations, and nested sections.
    """
    extracted_keys: Dict[str, str] = {}
    try:
        if hasattr(st, "secrets") and st.secrets:
            def extract(mapping):
                for k in mapping:
                    try:
                        v = mapping[k]
                        if isinstance(v, dict) or hasattr(v, "items"):
                            extract(v)
                        else:
                            clean_k = str(k).lower().replace("-", "_").replace(" ", "_")
                            extracted_keys[clean_k] = str(v).strip()
                    except Exception:
                        pass
            extract(st.secrets)
    except Exception:
        pass

    if not extracted_keys:
        try:
            local_secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "secrets.toml")
            if os.path.exists(local_secrets_path):
                import toml
                disk_secrets = toml.load(local_secrets_path)
                extract(disk_secrets)
        except Exception:
            pass

    for k in ["odds_api_key", "the_odds_api_key", "odds_key", "oddsapi_key", "api_key"]:
        if k in extracted_keys and extracted_keys[k]:
            return extracted_keys[k]
        if k.upper() in os.environ and os.environ[k.upper()]:
            return os.environ[k.upper()]
    return os.environ.get("ODDS_API_KEY", DEFAULT_ODDS_API_KEY)


def get_espn_credentials(secrets_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Retrieve ESPN league credentials from st.secrets or environment variables.
    Recursively scans st.secrets (supporting flat keys, case-insensitive variants,
    and nested sections like [espn], [espn_league], [credentials], etc.)
    and sanitizes cookies (unquoting URL-encoded SWID, stripping extra quotes).
    Strictly prioritizes secrets over ambient environment variables.
    """
    secrets_keys: Dict[str, Any] = {}
    env_keys: Dict[str, Any] = {}

    def extract_from_mapping(mapping, target_dict):
        for k in mapping:
            try:
                v = mapping[k]
                if isinstance(v, dict) or hasattr(v, "items"):
                    extract_from_mapping(v, target_dict)
                else:
                    clean_k = str(k).lower().replace("-", "_").replace(" ", "_")
                    target_dict[clean_k] = v
            except Exception:
                pass

    # 1. Custom dict or st.secrets
    if secrets_dict is not None:
        extract_from_mapping(secrets_dict, secrets_keys)
    else:
        try:
            if hasattr(st, "secrets") and st.secrets:
                extract_from_mapping(st.secrets, secrets_keys)
        except Exception:
            pass
        # 1b. Fallback to local .streamlit/secrets.toml if secrets_keys is empty
        if not secrets_keys:
            try:
                local_secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "secrets.toml")
                if os.path.exists(local_secrets_path):
                    import toml
                    disk_secrets = toml.load(local_secrets_path)
                    extract_from_mapping(disk_secrets, secrets_keys)
            except Exception:
                pass
        # 2. Environment variables fallback (only if secrets_dict was not explicitly provided)
        for env_k, env_v in os.environ.items():
            clean_env_k = str(env_k).lower().replace("-", "_").replace(" ", "_")
            if env_v:
                env_keys[clean_env_k] = env_v

    # Helper: Check secrets first across all candidates, then env_keys
    def resolve_candidate(candidates: List[str]) -> Optional[Any]:
        for k in candidates:
            if k in secrets_keys and secrets_keys[k] is not None and str(secrets_keys[k]).strip() != "":
                return secrets_keys[k]
        for k in candidates:
            if k in env_keys and env_keys[k] is not None and str(env_keys[k]).strip() != "":
                return env_keys[k]
        return None

    # 3. Resolve league_id
    league_id = None
    raw_lid = resolve_candidate([
        "espn_league_id", "league_id", "leagueid", "espnleagueid", "espn_league", "league"
    ])
    if raw_lid is not None:
        try:
            league_id = int(str(raw_lid).strip().strip("\"'"))
        except ValueError:
            pass

    # 4. Resolve year (prioritize explicit secret over default)
    current_calendar_year = datetime.datetime.now().year
    year = None
    raw_year = resolve_candidate([
        "espn_year", "year", "season", "season_year", "espnyear", "league_year", "nfl_year"
    ])
    if raw_year is not None:
        try:
            year = int(str(raw_year).strip().strip("\"'"))
        except ValueError:
            pass

    # 5. Resolve espn_s2
    espn_s2 = None
    raw_s2 = resolve_candidate(["espn_s2", "espns2", "s2", "cookie_s2", "espn_cookie_s2"])
    if raw_s2 is not None:
        s2_str = str(raw_s2).strip().strip("\"'")
        if s2_str:
            espn_s2 = urllib.parse.unquote(s2_str)

    # 6. Resolve swid
    swid = None
    raw_swid = resolve_candidate(["espn_swid", "espnswid", "swid", "cookie_swid", "espn_cookie_swid"])
    if raw_swid is not None:
        swid_str = str(raw_swid).strip().strip("\"'")
        if swid_str:
            clean_swid = urllib.parse.unquote(swid_str)
            if not clean_swid.startswith("{") and "-" in clean_swid:
                clean_swid = f"{{{clean_swid}}}"
            swid = clean_swid

    # 7. Optional user team hint
    user_team = None
    raw_team = resolve_candidate(["user_team", "my_team", "user_team_name", "team_name", "team"])
    if raw_team is not None:
        user_team = str(raw_team).strip().strip("\"'")

    found_in_secrets = bool(league_id) and (bool(secrets_keys) or bool(env_keys))

    return {
        "league_id": league_id,
        "year": year or current_calendar_year,
        "espn_s2": espn_s2,
        "swid": swid,
        "user_team_hint": user_team,
        "found_in_secrets": found_in_secrets
    }



# ---------------------------------------------------------------------------
# 1. THE ODDS API INGESTION
# ---------------------------------------------------------------------------

@st.cache_data(ttl=14400, show_spinner=False)
def fetch_nfl_odds(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Ingest NFL game spreads and totals from The Odds API.
    Cached for 4 hours (14400 seconds).
    """
    key = api_key or get_odds_api_key()
    url = f"{ODDS_API_BASE_URL}/odds"
    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american"
    }
    try:
        resp = requests.get(url, params=params, timeout=12)
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception:
        return []


def parse_nfl_games(raw_odds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse game odds into structured matchups with consensus spread, total,
    and calculated Implied Team Totals.
    """
    games = []
    for event in raw_odds:
        event_id = event.get("id")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")

        spread_val = None
        total_val = None

        bookmakers = event.get("bookmakers", [])
        if bookmakers:
            for bm in bookmakers:
                for market in bm.get("markets", []):
                    if market.get("key") == "spreads" and spread_val is None:
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == home_team:
                                spread_val = outcome.get("point")
                    elif market.get("key") == "totals" and total_val is None:
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == "Over":
                                total_val = outcome.get("point")

        if spread_val is None:
            spread_val = -3.0
        if total_val is None:
            total_val = 47.0

        home_implied = round((total_val - spread_val) / 2.0, 1)
        away_implied = round((total_val + spread_val) / 2.0, 1)

        games.append({
            "event_id": event_id,
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "spread": spread_val,
            "total": total_val,
            "home_implied": home_implied,
            "away_implied": away_implied,
            "display": f"{away_team} @ {home_team} (O/U {total_val}, Spread {spread_val:+0.1f})"
        })
    return games


@st.cache_data(ttl=14400, show_spinner=False)
def fetch_player_props(event_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingest player props for a specific NFL game from The Odds API.
    Queries passing yards, completions, passing TDs, interceptions,
    rushing yards, receptions, receiving yards, and anytime TD odds.
    """
    key = api_key or get_odds_api_key()
    url = f"{ODDS_API_BASE_URL}/events/{event_id}/odds"
    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "player_pass_yds,player_pass_completions,player_pass_tds,player_pass_interceptions,player_rush_yds,player_receptions,player_reception_yds,player_anytime_td",
        "oddsFormat": "american"
    }
    try:
        resp = requests.get(url, params=params, timeout=12)
        if resp.status_code == 200:
            return resp.json()
        return {}
    except Exception:
        return {}


def american_odds_to_prob(odds: float) -> float:
    """Convert American moneyline odds to implied probability [0.0, 1.0]."""
    try:
        odds = float(odds)
        if odds < 0:
            return abs(odds) / (abs(odds) + 100.0)
        else:
            return 100.0 / (odds + 100.0)
    except Exception:
        return 0.35


def devig_anytime_td_probabilities(
    raw_probs: Dict[str, float],
    team_implied_total: Optional[float] = None,
    realistic_td_total: Optional[float] = None
) -> Dict[str, float]:
    """
    De-vig Anytime TD implied probabilities across a market or team.
    Sportsbook Anytime TD odds typically carry 20-40% overround (juice).
    Normalizes implied probabilities so they sum to realistic offensive team TD totals
    (~Implied Total / 7.2, or default ~2.3 TDs per team / ~4.6 TDs per game) to prevent inflated mu.
    """
    if not raw_probs:
        return {}

    if realistic_td_total is not None and realistic_td_total > 0:
        target_tds = float(realistic_td_total)
    elif team_implied_total is not None and team_implied_total > 0:
        target_tds = max(1.2, round(team_implied_total / 7.2, 2))
    else:
        target_tds = 2.3

    total_prob = sum(max(0.0, float(p)) for p in raw_probs.values())
    if total_prob <= target_tds or total_prob == 0:
        return {k: round(max(0.0, min(0.85, float(v))), 3) for k, v in raw_probs.items()}

    shrink_factor = target_tds / total_prob
    devigged = {}
    for k, v in raw_probs.items():
        p_adj = max(0.0, float(v)) * shrink_factor
        devigged[k] = round(min(0.85, p_adj), 3)

    return devigged


def normalize_player_name(name: str) -> str:
    """Normalize player name for robust matching across ESPN, Sleeper, and The Odds API."""
    if not name:
        return ""
    n = name.lower()
    n = re.sub(r"[.\'\"]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


# High-fidelity consensus props prior database (De-vigged true anytime TD probabilities)
CURATED_NFL_PROPS: Dict[str, Dict[str, Any]] = {
    "jared goff": {"pass_yds": 266.5, "pass_completions": 23.5, "pass_tds": 1.85, "pass_interceptions": 0.55, "rush_yds": 1.5, "anytime_td_prob": 0.09},
    "dak prescott": {"pass_yds": 268.5, "pass_completions": 24.5, "pass_tds": 1.90, "pass_interceptions": 0.70, "rush_yds": 14.5, "anytime_td_prob": 0.16},
    "matthew stafford": {"pass_yds": 258.5, "pass_completions": 22.5, "pass_tds": 1.70, "pass_interceptions": 0.65, "rush_yds": 2.0, "anytime_td_prob": 0.07},
    "cj stroud": {"pass_yds": 264.5, "pass_completions": 23.0, "pass_tds": 1.80, "pass_interceptions": 0.60, "rush_yds": 12.0, "anytime_td_prob": 0.13},
    "baker mayfield": {"pass_yds": 244.5, "pass_completions": 21.0, "pass_tds": 1.60, "pass_interceptions": 0.70, "rush_yds": 14.0, "anytime_td_prob": 0.15},
    "sam darnold": {"pass_yds": 238.5, "pass_completions": 20.0, "pass_tds": 1.50, "pass_interceptions": 0.80, "rush_yds": 11.0, "anytime_td_prob": 0.13},
    "chase brown": {"rush_yds": 74.5, "receptions": 3.5, "rec_yds": 26.5, "anytime_td_prob": 0.42},
    "bucky irving": {"rush_yds": 70.5, "receptions": 3.0, "rec_yds": 22.5, "anytime_td_prob": 0.38},
    "saquon barkley": {"rush_yds": 88.5, "receptions": 3.8, "rec_yds": 28.5, "anytime_td_prob": 0.54},
    "breece hall": {"rush_yds": 66.5, "receptions": 4.5, "rec_yds": 36.5, "anytime_td_prob": 0.49},
    "tony pollard": {"rush_yds": 62.5, "receptions": 2.8, "rec_yds": 18.5, "anytime_td_prob": 0.35},
    "chuba hubbard": {"rush_yds": 64.5, "receptions": 2.6, "rec_yds": 18.5, "anytime_td_prob": 0.34},
    "jerome ford": {"rush_yds": 54.5, "receptions": 2.5, "rec_yds": 16.5, "anytime_td_prob": 0.28},
    "jaylen warren": {"rush_yds": 42.5, "receptions": 3.2, "rec_yds": 24.5, "anytime_td_prob": 0.22},
    "rico dowdle": {"rush_yds": 50.5, "receptions": 2.4, "rec_yds": 15.5, "anytime_td_prob": 0.26},
    "chris rodriguez": {"rush_yds": 38.5, "receptions": 1.2, "rec_yds": 8.5, "anytime_td_prob": 0.20},
    "omarion hampton": {"rush_yds": 48.5, "receptions": 2.2, "rec_yds": 16.5, "anytime_td_prob": 0.25},
    "drake london": {"rec_yds": 76.5, "receptions": 6.5, "rush_yds": 0.0, "anytime_td_prob": 0.32},
    "dj moore": {"rec_yds": 68.5, "receptions": 5.5, "rush_yds": 2.5, "anytime_td_prob": 0.29},
    "puka nacua": {"rec_yds": 84.5, "receptions": 7.2, "rush_yds": 1.5, "anytime_td_prob": 0.35},
    "tee higgins": {"rec_yds": 64.5, "receptions": 5.0, "rush_yds": 0.0, "anytime_td_prob": 0.30},
    "wandale robinson": {"rec_yds": 46.5, "receptions": 5.5, "rush_yds": 3.0, "anytime_td_prob": 0.19},
    "demario douglas": {"rec_yds": 52.5, "receptions": 4.8, "rush_yds": 2.0, "anytime_td_prob": 0.20},
    "jalen coker": {"rec_yds": 46.5, "receptions": 3.8, "rush_yds": 0.0, "anytime_td_prob": 0.20},
    "adonai mitchell": {"rec_yds": 38.5, "receptions": 2.8, "rush_yds": 0.0, "anytime_td_prob": 0.18},
    "jack bech": {"rec_yds": 32.5, "receptions": 2.5, "rush_yds": 0.0, "anytime_td_prob": 0.14},
    "deebo samuel": {"rec_yds": 54.5, "receptions": 4.2, "rush_yds": 22.5, "anytime_td_prob": 0.35},
    "quentin johnston": {"rec_yds": 48.5, "receptions": 3.8, "rush_yds": 0.0, "anytime_td_prob": 0.23},
    "jauan jennings": {"rec_yds": 52.5, "receptions": 4.0, "rush_yds": 0.0, "anytime_td_prob": 0.24},
    "isaiah likely": {"rec_yds": 44.5, "receptions": 3.8, "rush_yds": 0.0, "anytime_td_prob": 0.23},
    "tucker kraft": {"rec_yds": 42.5, "receptions": 3.6, "rush_yds": 0.0, "anytime_td_prob": 0.22},
    "jaguars dst": {"opp_implied_pts": 21.0, "opp_implied_yds": 320.0, "sacks": 2.8, "turnovers": 1.4, "forced_fumbles": 1.0},
    "ravens dst": {"opp_implied_pts": 19.5, "opp_implied_yds": 305.0, "sacks": 3.2, "turnovers": 1.6, "forced_fumbles": 1.2},
}


@st.cache_data(ttl=7200, show_spinner=False)
def build_all_player_props_cache(api_key: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Build a comprehensive player props cache combining live markets from The Odds API
    with curated high-fidelity consensus lines.
    De-vigs raw Anytime TD market odds so probabilities sum to realistic team TD totals.
    """
    cache: Dict[str, Dict[str, Any]] = dict(CURATED_NFL_PROPS)

    raw_odds = fetch_nfl_odds(api_key=api_key)
    if not raw_odds:
        return cache

    # Ingest props across scheduled NFL games
    for event in raw_odds[:16]:
        event_id = event.get("id")
        if not event_id:
            continue
        props_data = fetch_player_props(event_id, api_key=api_key)
        bookmakers = props_data.get("bookmakers", [])
        if not bookmakers:
            continue

        raw_td_market = {}

        for bm in bookmakers:
            for market in bm.get("markets", []):
                m_key = market.get("key")
                outcomes = market.get("outcomes", [])

                for out in outcomes:
                    raw_player_name = out.get("description") or out.get("name", "")
                    norm_name = normalize_player_name(raw_player_name)
                    if not norm_name or norm_name in ["over", "under", "yes", "no"]:
                        continue

                    if norm_name not in cache:
                        cache[norm_name] = {}

                    point_val = out.get("point")
                    price_val = out.get("price")
                    out_name = out.get("name", "")

                    if m_key == "player_pass_yds" and out_name == "Over" and point_val is not None:
                        cache[norm_name]["pass_yds"] = float(point_val)
                    elif m_key == "player_pass_completions" and out_name == "Over" and point_val is not None:
                        cache[norm_name]["pass_completions"] = float(point_val)
                    elif m_key == "player_pass_tds" and out_name == "Over" and point_val is not None:
                        cache[norm_name]["pass_tds"] = float(point_val)
                    elif m_key == "player_pass_interceptions" and out_name == "Over" and point_val is not None:
                        cache[norm_name]["pass_interceptions"] = float(point_val)
                    elif m_key == "player_rush_yds" and out_name == "Over" and point_val is not None:
                        cache[norm_name]["rush_yds"] = float(point_val)
                    elif m_key == "player_receptions" and out_name == "Over" and point_val is not None:
                        cache[norm_name]["receptions"] = float(point_val)
                    elif m_key == "player_reception_yds" and out_name == "Over" and point_val is not None:
                        cache[norm_name]["rec_yds"] = float(point_val)
                    elif m_key == "player_anytime_td" and (out_name == "Yes" or out.get("name") == raw_player_name):
                        raw_td_market[norm_name] = american_odds_to_prob(price_val)

        # De-vig anytime TD market across the game (target ~4.6 TDs total for two teams)
        if raw_td_market:
            devigged_tds = devig_anytime_td_probabilities(raw_td_market, realistic_td_total=4.6)
            for p_name, devig_p in devigged_tds.items():
                if p_name in cache:
                    cache[p_name]["anytime_td_prob"] = devig_p

    return cache


def get_matched_player_props(
    player_name: str,
    props_cache: Dict[str, Any],
    position: str = ""
) -> Dict[str, Any]:
    """Retrieve matched Vegas consensus props for a player with fuzzy fallbacks."""
    norm = normalize_player_name(player_name)
    if norm in props_cache:
        return props_cache[norm].copy()

    # Partial / substring match
    for k, v in props_cache.items():
        if k in norm or norm in k:
            return v.copy()

    # Defensive match
    if position in ["DEF", "D/ST"]:
        for k, v in props_cache.items():
            if "dst" in k and any(part in norm for part in k.split()):
                return v.copy()

    return {}


# ---------------------------------------------------------------------------
# 2. POLYMARKET GAMMA API INGESTION
# ---------------------------------------------------------------------------

@st.cache_data(ttl=14400, show_spinner=False)
def fetch_polymarket_event(slug: str) -> Optional[Dict[str, Any]]:
    """
    Ingest crowd-sourced event/injury probability from Polymarket Gamma API.
    """
    if not slug:
        return None
    url = f"{POLYMARKET_BASE_URL}/events"
    params = {"slug": slug.strip()}
    try:
        resp = requests.get(url, params=params, timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                event = data[0]
                markets = event.get("markets", [])
                p_active = 1.0
                market_question = event.get("title", "")
                if markets:
                    m = markets[0]
                    market_question = m.get("question", market_question)
                    prices = m.get("outcomePrices")
                    if prices:
                        try:
                            p_active = float(eval(prices)[0]) if isinstance(prices, str) else float(prices[0])
                        except Exception:
                            p_active = 0.85

                return {
                    "slug": slug,
                    "title": event.get("title"),
                    "question": market_question,
                    "active_probability": round(p_active, 3),
                    "volume": event.get("volume", 0),
                    "markets": markets
                }
        return None
    except Exception:
        return None


def get_polymarket_player_probability(
    player_name: str,
    injury_status: str,
    default_p: float = 1.0
) -> Tuple[float, str]:
    """
    Query Polymarket crowd-sourced probability for player injury/availability.
    Falls back to calibrated medical baseline if no active contract exists.
    """
    clean_status = (injury_status or "ACTIVE").upper().strip()
    if clean_status in ["OUT", "IR", "INJURY_RESERVE", "PUP", "SUSPENDED"]:
        return 0.0, "Ruled OUT / IR (0% active)"
    if clean_status in ["ACTIVE", "PROBABLE"] and default_p >= 0.90:
        return 1.0, "Active Consensus (100%)"

    # Player has an active injury question mark (e.g. QUESTIONABLE or DOUBTFUL)
    norm_name = normalize_player_name(player_name)
    first_last = norm_name.replace(" ", "-")
    slug_candidates = [
        f"will-{first_last}-play-in-week",
        f"{first_last}-injury",
        f"{first_last}-status"
    ]
    for slug in slug_candidates:
        res = fetch_polymarket_event(slug)
        if res and res.get("active_probability") is not None:
            return float(res["active_probability"]), f"Polymarket Live ({res['active_probability']*100:.0f}%)"

    # Fallback to calibrated injury probability
    return default_p, f"Calibrated Injury Prior ({default_p*100:.0f}%)"


# ---------------------------------------------------------------------------
# 3. PROJECTION PARAMETER SYNTHESIS (MU & SIGMA)
# ---------------------------------------------------------------------------

INACTIVE_STATUSES = {
    "IR", "INJURY_RESERVE", "OUT", "O", "DOUBTFUL", "D",
    "SUSPENDED", "SUSPENSION", "PUP", "PHYSICALLY_UNABLE_TO_PERFORM",
    "NON_FOOTBALL_INJURY", "NFI", "COVID"
}


def parse_player_injury_and_status(inj_status_raw: str, slot_raw: str = "") -> Tuple[str, float, bool]:
    """
    Normalizes raw ESPN / Sleeper injury designations and lineup slot codes into:
      (clean_status, p_active, is_inactive)

    Rules:
      - Any player on IR (slot IR/IR_ELIGIBLE or status INJURY_RESERVE/IR) or ruled OUT/PUP/SUSPENDED has:
          p_active = 0.0, is_inactive = True, clean_status = "IR" or "OUT".
      - DOUBTFUL has p_active = 0.10, is_inactive = False.
      - QUESTIONABLE has p_active = 0.50, is_inactive = False.
      - PROBABLE / ACTIVE has p_active = 1.0, is_inactive = False.
    """
    inj_upper = str(inj_status_raw or "ACTIVE").upper().strip()
    slot_upper = str(slot_raw or "").upper().strip()

    is_ir = (
        slot_upper in ["IR", "IR_ELIGIBLE"]
        or inj_upper in ["IR", "INJURY_RESERVE"]
        or "IR" in inj_upper
        or "RESERVE" in inj_upper
    )
    is_out = (
        inj_upper in ["OUT", "O", "PUP", "PHYSICALLY_UNABLE_TO_PERFORM", "SUSPENDED", "SUSPENSION", "NFI", "NON_FOOTBALL_INJURY", "COVID"]
        or "OUT" in inj_upper
        or "PUP" in inj_upper
        or "SUSPEND" in inj_upper
    )

    if is_ir:
        return "IR", 0.0, True
    if is_out:
        return "OUT", 0.0, True

    if inj_upper in ["DOUBTFUL", "D"]:
        return "DOUBTFUL", 0.10, False
    if inj_upper in ["QUESTIONABLE", "Q"]:
        return "QUESTIONABLE", 0.50, False
    if inj_upper in ["PROBABLE", "P"]:
        return "PROBABLE", 0.90, False

    return "ACTIVE", 1.0, False


def calculate_projection_params(
    position: str,
    props: Optional[Dict[str, Any]] = None,
    default_proj: float = 12.0,
    ppr: float = 1.0,
    polymarket_prob: float = 1.0,
    is_inactive: bool = False,
    scoring_cfg: Optional[scoring_config.LeagueScoringConfig] = None,
    return_breakdown: bool = False
) -> Any:
    """
    Synthesize distribution parameters (mu, sigma, availability_p)
    from Vegas player prop markets, historical averages, and idiosyncratic scoring rules.
    If player is inactive (IR/OUT) or has zero availability, parameters are strictly 0.0.
    """
    cfg = scoring_cfg or scoring_config.DEFAULT_SCORING
    if ppr != 1.0 and scoring_cfg is None:
        cfg = scoring_config.LeagueScoringConfig(
            ppr=ppr,
            pass_yd_multiplier=cfg.pass_yd_multiplier,
            completion_bonus=cfg.completion_bonus,
            pass_td=cfg.pass_td,
            pass_td_40_bonus=cfg.pass_td_40_bonus,
            pass_td_50_bonus=cfg.pass_td_50_bonus,
            pass_int=cfg.pass_int,
            pass_2pt=cfg.pass_2pt,
            pass_300_bonus=cfg.pass_300_bonus,
            pass_400_bonus=cfg.pass_400_bonus,
            rush_yd_multiplier=cfg.rush_yd_multiplier,
            rush_td=cfg.rush_td,
            rush_td_40_bonus=cfg.rush_td_40_bonus,
            rush_td_50_bonus=cfg.rush_td_50_bonus,
            rush_2pt=cfg.rush_2pt,
            rush_100_bonus=cfg.rush_100_bonus,
            rush_200_bonus=cfg.rush_200_bonus,
            rec_yd_multiplier=cfg.rec_yd_multiplier,
            rec_td=cfg.rec_td,
            rec_td_40_bonus=cfg.rec_td_40_bonus,
            rec_td_50_bonus=cfg.rec_td_50_bonus,
            rec_2pt=cfg.rec_2pt,
            rec_100_bonus=cfg.rec_100_bonus,
            rec_200_bonus=cfg.rec_200_bonus
        )

    mu, sigma, p_active, breakdown = scoring_config.calculate_player_mu_and_sigma(
        position=position,
        props=props,
        espn_proj=default_proj,
        scoring_config=cfg,
        polymarket_prob=polymarket_prob,
        is_inactive=is_inactive
    )

    if return_breakdown:
        return mu, sigma, p_active, breakdown
    return mu, sigma, p_active


# ---------------------------------------------------------------------------
# 4. ESPN FANTASY LEAGUE SYNC MODULE
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def sync_espn_league(
    league_id: int,
    year: int = 2024,
    espn_s2: Optional[str] = None,
    swid: Optional[str] = None,
    ppr: float = 1.0,
    scoring_cfg: Optional[scoring_config.LeagueScoringConfig] = None,
    user_team_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synchronize league teams, active starting rosters, bench, weekly scoreboard
    matchups, and top free agents via espn-api.
    """
    try:
        from espn_api.football import League
    except ImportError:
        return {"success": False, "error": "espn-api is not installed."}

    clean_s2 = None
    if espn_s2:
        clean_s2 = urllib.parse.unquote(str(espn_s2).strip().strip("\"'"))

    clean_swid = None
    if swid:
        clean_swid = urllib.parse.unquote(str(swid).strip().strip("\"'"))
        if not clean_swid.startswith("{") and "-" in clean_swid:
            clean_swid = f"{{{clean_swid}}}"

    kwargs = {
        "league_id": int(league_id),
        "year": int(year)
    }
    if clean_s2:
        kwargs["espn_s2"] = clean_s2
    if clean_swid:
        kwargs["swid"] = clean_swid

    try:
        league = League(**kwargs)
    except Exception as e:
        return {"success": False, "error": str(e), "teams": [], "free_agents": []}

    league_name = getattr(league.settings, "name", f"League {league_id}") if hasattr(league, "settings") else f"League {league_id}"
    current_week = getattr(league, "current_week", 1)

    # Ingest consensus Vegas player props cache across all scheduled games
    props_cache = build_all_player_props_cache()

    teams_data = []
    team_map = {}
    team_id_map = {}

    for team in league.teams:
        starters = []
        bench = []
        ir = []
        full_roster = []

        for p in team.roster:
            slot_code = getattr(p, "lineupSlot", p.position)
            slot_name = "FLEX" if slot_code in ["RB/WR/TE", "FLEX"] else slot_code

            # Injury status & availability parsing
            clean_status, p_active, is_inactive = parse_player_injury_and_status(
                inj_status_raw=getattr(p, "injuryStatus", "ACTIVE"),
                slot_raw=slot_code
            )

            # Match Vegas consensus props and Polymarket crowd availability
            matched_props = get_matched_player_props(p.name, props_cache, p.position)
            p_act_poly, poly_source = get_polymarket_player_probability(p.name, clean_status, default_p=p_active)

            # Determine baseline projected points (inactive players strictly 0.0)
            if is_inactive:
                proj = 0.0
            else:
                proj = float(getattr(p, "projected_avg_points", 0.0) or getattr(p, "avg_points", 0.0) or getattr(p, "projected_points", 10.0) or 10.0)
                if proj <= 0:
                    proj = float(getattr(p, "avg_points", 10.0) or 10.0)

            # Parameter synthesis under user's idiosyncratic league scoring rules
            mu, sigma, p_act, breakdown = calculate_projection_params(
                position=p.position,
                props=matched_props if matched_props else None,
                default_proj=proj,
                ppr=ppr,
                polymarket_prob=p_act_poly,
                is_inactive=is_inactive,
                scoring_cfg=scoring_cfg,
                return_breakdown=True
            )

            p_info = {
                "name": p.name,
                "position": p.position,
                "slot": slot_name,
                "team": getattr(p, "proTeam", "NFL"),
                "mu": mu,
                "sigma": sigma,
                "p_active": p_act,
                "injury_status": clean_status,
                "is_inactive": is_inactive,
                "avg_points": float(getattr(p, "avg_points", 0.0) or 0.0),
                "proj_points": proj,
                "prop_source": breakdown.get("source", "Idiosyncratic Baseline"),
                "poly_source": poly_source,
                "score_breakdown": breakdown,
                "props": matched_props
            }
            full_roster.append(p_info)

            if slot_code in ["IR", "IR_ELIGIBLE"] or is_inactive:
                ir.append(p_info)
            elif slot_code not in ["BE", ""]:
                starters.append(p_info)
            else:
                bench.append(p_info)

        # Ensure we have standard offensive skill starters (QB, RB, WR, TE, FLEX)
        if len(starters) < 5 and len(full_roster) >= 5:
            # Auto-assign top projected ACTIVE players as starters if league slots aren't populated
            sorted_roster = sorted([p for p in full_roster if not p.get("is_inactive")], key=lambda x: x["mu"], reverse=True)
            starters = sorted_roster[:7]

        owners_list = getattr(team, "owners", [])
        owner_display = getattr(team, "owner", team.team_name)
        owner_swids = []
        if owners_list and isinstance(owners_list, list):
            first_o = owners_list[0]
            fn = first_o.get("firstName", "").strip() if isinstance(first_o, dict) else ""
            ln = first_o.get("lastName", "").strip() if isinstance(first_o, dict) else ""
            dn = first_o.get("displayName", "").strip() if isinstance(first_o, dict) else ""
            if fn or ln:
                owner_display = f"{fn} {ln} ({dn})".strip() if dn else f"{fn} {ln}".strip()
            elif dn:
                owner_display = dn
            for o in owners_list:
                if isinstance(o, dict) and o.get("id"):
                    owner_swids.append(str(o["id"]).strip().strip("{}").lower())

        spent_faab = int(getattr(team, "acquisition_budget_spent", 0) or 0)
        remaining_faab = max(0, 100 - spent_faab)

        team_dict = {
            "team_id": team.team_id,
            "team_name": str(team.team_name),
            "owner": owner_display,
            "owner_swids": owner_swids,
            "acquisition_budget_spent": spent_faab,
            "remaining_faab": remaining_faab,
            "starters": starters,
            "bench": bench,
            "ir": ir,
            "roster": full_roster
        }
        teams_data.append(team_dict)
        team_map[team.team_name] = team_dict
        team_id_map[team.team_id] = team_dict

    # Ingest Current Week Scoreboard Matchups (Fast & Safe)
    weekly_matchups = {}
    try:
        sb = league.scoreboard(week=current_week)
        m_list = []
        for m in sb:
            if m.home_team and m.away_team:
                h_name = str(m.home_team.team_name)
                a_name = str(m.away_team.team_name)
                m_list.append({
                    "home_team_id": m.home_team.team_id,
                    "home_team_name": h_name,
                    "away_team_id": m.away_team.team_id,
                    "away_team_name": a_name,
                    "display": f"{a_name} vs {h_name}"
                })
        if m_list:
            weekly_matchups[current_week] = m_list
    except Exception:
        pass

    # Ingest Top Free Agents (Waiver Wire Pool) with multi-strategy fallback
    free_agents_data = []
    fa_list = []
    for fa_opts in [
        {"week": current_week, "size": 100},
        {"size": 75},
        {"week": max(1, current_week), "size": 50},
        {"week": 1, "size": 50},
        {}
    ]:
        try:
            fa_list = league.free_agents(**fa_opts)
            if fa_list:
                break
        except Exception:
            continue

    if fa_list:
        for fa in fa_list:
            clean_status, p_active, is_inactive = parse_player_injury_and_status(
                inj_status_raw=getattr(fa, "injuryStatus", "ACTIVE")
            )
            # Prioritize weekly matchup projection over season-long average
            proj = 0.0 if is_inactive else float(
                getattr(fa, "projected_points", 0.0) or
                getattr(fa, "projected_avg_points", 0.0) or
                getattr(fa, "avg_points", 0.0) or
                0.0
            )

            matched_props = get_matched_player_props(fa.name, props_cache, fa.position)
            p_act_poly = p_active
            poly_source = f"Calibrated Prior ({int(p_active*100)}%)" if clean_status != "ACTIVE" else "Active Consensus (100%)"

            mu, sigma, p_act, breakdown = calculate_projection_params(
                position=fa.position,
                props=matched_props if matched_props else None,
                default_proj=proj,
                ppr=ppr,
                polymarket_prob=p_act_poly,
                is_inactive=is_inactive,
                scoring_cfg=scoring_cfg,
                return_breakdown=True
            )

            is_dst = fa.position in ["D/ST", "DEF"]
            is_kicker = fa.position in ["K"]

            free_agents_data.append({
                "name": fa.name,
                "position": fa.position,
                "team": getattr(fa, "proTeam", "FA"),
                "mu": mu,
                "sigma": sigma,
                "p_active": p_act,
                "injury_status": clean_status,
                "is_inactive": is_inactive,
                "proj_points": proj,
                "is_dst": is_dst,
                "is_kicker": is_kicker,
                "prop_source": breakdown.get("source", "Idiosyncratic Baseline"),
                "poly_source": poly_source,
                "score_breakdown": breakdown,
                "props": matched_props
            })

    # Auto-detect User Team
    user_team_name = None
    user_opp_name = None

    # Priority 1: Match SWID against team.owners
    if swid:
        clean_target_swid = str(swid).strip().strip("{}").lower()
        for t in teams_data:
            if clean_target_swid in t.get("owner_swids", []):
                user_team_name = t["team_name"]
                break

    # Priority 2: user_team_hint if provided in secrets/config
    if not user_team_name and user_team_hint:
        hint_lower = str(user_team_hint).lower().strip()
        for t in teams_data:
            t_str = f"{t['team_name']} {t.get('owner', '')}".lower()
            if hint_lower in t_str:
                user_team_name = t["team_name"]
                break

    # Priority 3: heuristic check ("eatin")
    if not user_team_name:
        for t in teams_data:
            tname = t["team_name"].lower()
            if "eatin" in tname:
                user_team_name = t["team_name"]
                break

    # Priority 4: default to first team
    if not user_team_name and teams_data:
        user_team_name = teams_data[0]["team_name"]

    if user_team_name and current_week in weekly_matchups:
        for m in weekly_matchups[current_week]:
            if m["home_team_name"] == user_team_name:
                user_opp_name = m["away_team_name"]
                break
            elif m["away_team_name"] == user_team_name:
                user_opp_name = m["home_team_name"]
                break

    if not user_opp_name:
        other_teams = [t["team_name"] for t in teams_data if t["team_name"] != user_team_name]
        user_opp_name = other_teams[0] if other_teams else (teams_data[1]["team_name"] if len(teams_data) > 1 else None)

    return {
        "success": True,
        "league_id": league_id,
        "league_name": league_name,
        "year": year,
        "current_week": current_week,
        "teams": teams_data,
        "team_map": team_map,
        "team_id_map": team_id_map,
        "weekly_matchups": weekly_matchups,
        "free_agents": free_agents_data,
        "user_team_name": user_team_name,
        "user_opp_name": user_opp_name
    }


def build_matchup_from_espn_teams(
    team_a_dict: Dict[str, Any],
    team_b_dict: Dict[str, Any],
    free_agents_list: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Format two ESPN team dictionaries into the standard engine format for
    simulation and VORP calculations.
    Filters to skill starters (QB, RB, WR, TE, FLEX).
    """
    def filter_skill_starters(team_dict):
        designated_starters = [
            p for p in team_dict.get("starters", [])
            if (p.get("position") in {"QB", "RB", "WR", "TE"} or "FLEX" in p.get("slot", ""))
            and not p.get("is_inactive")
            and float(p.get("p_active", 1.0)) > 0.0
            and p.get("slot") not in ["IR", "IR_ELIGIBLE"]
        ]

        if len(designated_starters) >= 7:
            return designated_starters

        # Intelligently construct optimal starting lineup from active full roster:
        # 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX
        full_roster = team_dict.get("roster", [])
        active_roster = [
            p for p in full_roster
            if not p.get("is_inactive")
            and float(p.get("p_active", 1.0)) > 0.0
            and p.get("slot") not in ["IR", "IR_ELIGIBLE"]
            and float(p.get("mu", 0.0)) > 0.0
        ]
        used_names = set()
        optimal_starters = []

        # 1. Best QB
        qbs = sorted([p for p in active_roster if p["position"] == "QB"], key=lambda x: x["mu"], reverse=True)
        if qbs:
            optimal_starters.append({**qbs[0], "slot": "QB"})
            used_names.add(qbs[0]["name"])

        # 2. Top 2 RBs
        rbs = sorted([p for p in active_roster if p["position"] == "RB" and p["name"] not in used_names], key=lambda x: x["mu"], reverse=True)
        for i, rb in enumerate(rbs[:2]):
            optimal_starters.append({**rb, "slot": f"RB{i+1}"})
            used_names.add(rb["name"])

        # 3. Top 2 WRs
        wrs = sorted([p for p in active_roster if p["position"] == "WR" and p["name"] not in used_names], key=lambda x: x["mu"], reverse=True)
        for i, wr in enumerate(wrs[:2]):
            optimal_starters.append({**wr, "slot": f"WR{i+1}"})
            used_names.add(wr["name"])

        # 4. Best TE
        tes = sorted([p for p in active_roster if p["position"] == "TE" and p["name"] not in used_names], key=lambda x: x["mu"], reverse=True)
        if tes:
            optimal_starters.append({**tes[0], "slot": "TE"})
            used_names.add(tes[0]["name"])

        # 5. Best remaining FLEX (RB/WR/TE)
        flex_cands = sorted(
            [p for p in active_roster if p["position"] in ["RB", "WR", "TE"] and p["name"] not in used_names],
            key=lambda x: x["mu"],
            reverse=True
        )
        if flex_cands:
            optimal_starters.append({**flex_cands[0], "slot": "FLEX"})
            used_names.add(flex_cands[0]["name"])

        # If roster is small, fill remaining from available active roster
        if len(optimal_starters) < 7:
            remaining = [p for p in active_roster if p["name"] not in used_names]
            for p in remaining[:(7 - len(optimal_starters))]:
                optimal_starters.append({**p, "slot": "FLEX"})

        return optimal_starters

    # Format waiver pool: strictly use live ESPN free agents when in synced league
    if free_agents_list is not None and len(free_agents_list) > 0:
        waiver_pool = free_agents_list
    elif team_a_dict.get("free_agents"):
        waiver_pool = team_a_dict["free_agents"]
    else:
        waiver_pool = []

    return {
        "team_a_name": team_a_dict["team_name"],
        "team_a_roster": filter_skill_starters(team_a_dict),
        "team_a_bench": team_a_dict.get("bench", []),
        "team_b_name": team_b_dict["team_name"],
        "team_b_roster": filter_skill_starters(team_b_dict),
        "team_b_bench": team_b_dict.get("bench", []),
        "waiver_pool": waiver_pool[:100]
    }


# ---------------------------------------------------------------------------
# 5. SLEEPER ACTIVE NFL PLAYER UNIVERSE (FALLBACK)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sleeper_players() -> Dict[str, Dict[str, Any]]:
    """
    Ingest active NFL players database from Sleeper. Cached for 24 hours.
    """
    try:
        resp = requests.get(SLEEPER_PLAYERS_URL, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            valid_positions = {"QB", "RB", "WR", "TE", "K", "DEF"}
            active_players = {}
            for pid, info in data.items():
                pos = info.get("position")
                is_active = info.get("active", False)
                if is_active and pos in valid_positions:
                    full_name = info.get("full_name") or f"{info.get('first_name', '')} {info.get('last_name', '')}".strip()
                    if full_name:
                        active_players[full_name] = {
                            "player_id": pid,
                            "name": full_name,
                            "position": pos,
                            "team": info.get("team") or "FA",
                            "injury_status": info.get("injury_status") or "ACTIVE"
                        }
            return active_players
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# 6. DEFAULT HIGH-FIDELITY MATCHUP & WAIVER ROSTERS
# ---------------------------------------------------------------------------

def get_default_matchup_and_waivers(
    ppr: float = 1.0,
    scoring_cfg: Optional[scoring_config.LeagueScoringConfig] = None
) -> Dict[str, Any]:
    """
    Returns curated competitive rosters and waiver pool pre-mapped to real
    Vegas props and Polymarket crowd probabilities, evaluated dynamically
    under the user's idiosyncratic league scoring rules.
    """
    props_cache = build_all_player_props_cache()

    def build_player_entry(name: str, pos: str, slot: str, team: str, p_act_default: float = 1.0) -> Dict[str, Any]:
        matched_props = get_matched_player_props(name, props_cache, pos)
        clean_status = "ACTIVE" if p_act_default >= 0.9 else "QUESTIONABLE"
        p_act_poly, poly_source = get_polymarket_player_probability(name, clean_status, default_p=p_act_default)
        mu, sigma, p_act, breakdown = calculate_projection_params(
            position=pos,
            props=matched_props if matched_props else None,
            default_proj=12.0,
            ppr=ppr,
            polymarket_prob=p_act_poly,
            is_inactive=False,
            scoring_cfg=scoring_cfg,
            return_breakdown=True
        )
        return {
            "name": name,
            "position": pos,
            "slot": slot,
            "team": team,
            "mu": mu,
            "sigma": sigma,
            "p_active": p_act,
            "injury_status": clean_status,
            "is_inactive": False,
            "prop_source": breakdown.get("source", "Idiosyncratic Baseline"),
            "poly_source": poly_source,
            "score_breakdown": breakdown,
            "props": matched_props
        }

    raw_team_a = [
        ("Jared Goff", "QB", "QB", "DET", 1.0),
        ("Dak Prescott", "QB", "OP", "DAL", 1.0),
        ("Chase Brown", "RB", "RB", "CIN", 1.0),
        ("Bucky Irving", "RB", "RB", "TB", 1.0),
        ("Drake London", "WR", "WR", "ATL", 1.0),
        ("DJ Moore", "WR", "WR", "CHI", 1.0),
        ("Jalen Coker", "WR", "WR/TE", "CAR", 1.0),
        ("Omarion Hampton", "RB", "RB/WR/TE", "LAC", 1.0),
        ("Jaguars D/ST", "DEF", "D/ST", "JAX", 1.0),
    ]
    raw_bench_a = [
        ("Tony Pollard", "RB", "BE", "TEN", 1.0),
        ("Wan'Dale Robinson", "WR", "BE", "NYG", 1.0),
        ("DeMario Douglas", "WR", "BE", "NE", 1.0),
        ("Chris Rodriguez Jr.", "RB", "BE", "JAX", 1.0),
        ("Adonai Mitchell", "WR", "BE", "IND", 1.0),
        ("Jack Bech", "WR", "BE", "LV", 1.0),
    ]
    raw_team_b = [
        ("Matthew Stafford", "QB", "QB", "LAR", 1.0),
        ("C.J. Stroud", "QB", "OP", "HOU", 1.0),
        ("Saquon Barkley", "RB", "RB", "PHI", 1.0),
        ("Breece Hall", "RB", "RB", "NYJ", 1.0),
        ("Puka Nacua", "WR", "WR", "LAR", 1.0),
        ("Tee Higgins", "WR", "WR", "CIN", 1.0),
        ("Chuba Hubbard", "RB", "RB/WR/TE", "CAR", 1.0),
        ("Deebo Samuel Sr.", "WR", "WR/TE", "SF", 1.0),
        ("Ravens D/ST", "DEF", "D/ST", "BAL", 1.0),
    ]
    raw_waivers = [
        ("Baker Mayfield", "QB", "QB", "TB", 1.0),
        ("Sam Darnold", "QB", "QB", "MIN", 1.0),
        ("Jerome Ford", "RB", "RB", "CLE", 1.0),
        ("Jaylen Warren", "RB", "RB", "PIT", 0.85),
        ("Rico Dowdle", "RB", "RB", "DAL", 1.0),
        ("Quentin Johnston", "WR", "WR", "LAC", 1.0),
        ("Jauan Jennings", "WR", "WR", "SF", 0.80),
        ("Isaiah Likely", "TE", "TE", "BAL", 1.0),
        ("Tucker Kraft", "TE", "TE", "GB", 1.0),
    ]

    team_a_roster = [build_player_entry(*p) for p in raw_team_a]
    team_a_bench = [build_player_entry(*p) for p in raw_bench_a]
    team_b_roster = [build_player_entry(*p) for p in raw_team_b]
    waiver_pool = [build_player_entry(*p) for p in raw_waivers]

    return {
        "team_a_name": "Eatin TDs like Groceries (My Team)",
        "team_a_roster": team_a_roster,
        "team_a_bench": team_a_bench,
        "team_b_name": "Puka Shell Queen (Opponent)",
        "team_b_roster": team_b_roster,
        "waiver_pool": waiver_pool
    }


# ---------------------------------------------------------------------------
# 7. 32-TEAM DEFENSIVE SCHEME METRICS & MATCHUP MODIFIERS
# ---------------------------------------------------------------------------

NFL_DEFENSIVE_SCHEMES: Dict[str, Dict[str, float]] = {
    "CLE": {"quick_pressure_rate": 0.245, "mof_epa_allowed": 0.28, "rush_sr_allowed": 0.380},
    "PHI": {"quick_pressure_rate": 0.228, "mof_epa_allowed": 0.32, "rush_sr_allowed": 0.395},
    "KC":  {"quick_pressure_rate": 0.215, "mof_epa_allowed": 0.30, "rush_sr_allowed": 0.370},
    "PIT": {"quick_pressure_rate": 0.222, "mof_epa_allowed": 0.35, "rush_sr_allowed": 0.402},
    "BAL": {"quick_pressure_rate": 0.208, "mof_epa_allowed": 0.26, "rush_sr_allowed": 0.355},
    "SF":  {"quick_pressure_rate": 0.205, "mof_epa_allowed": 0.22, "rush_sr_allowed": 0.388},
    "DET": {"quick_pressure_rate": 0.212, "mof_epa_allowed": 0.41, "rush_sr_allowed": 0.362},
    "BUF": {"quick_pressure_rate": 0.195, "mof_epa_allowed": 0.34, "rush_sr_allowed": 0.410},
    "DAL": {"quick_pressure_rate": 0.190, "mof_epa_allowed": 0.42, "rush_sr_allowed": 0.465},
    "HOU": {"quick_pressure_rate": 0.218, "mof_epa_allowed": 0.31, "rush_sr_allowed": 0.392},
    "MIA": {"quick_pressure_rate": 0.185, "mof_epa_allowed": 0.48, "rush_sr_allowed": 0.425},
    "NYJ": {"quick_pressure_rate": 0.210, "mof_epa_allowed": 0.25, "rush_sr_allowed": 0.415},
    "GB":  {"quick_pressure_rate": 0.192, "mof_epa_allowed": 0.38, "rush_sr_allowed": 0.428},
    "MIN": {"quick_pressure_rate": 0.225, "mof_epa_allowed": 0.33, "rush_sr_allowed": 0.390},
    "CHI": {"quick_pressure_rate": 0.178, "mof_epa_allowed": 0.36, "rush_sr_allowed": 0.420},
    "TB":  {"quick_pressure_rate": 0.182, "mof_epa_allowed": 0.54, "rush_sr_allowed": 0.412},
    "CIN": {"quick_pressure_rate": 0.175, "mof_epa_allowed": 0.52, "rush_sr_allowed": 0.458},
    "LAC": {"quick_pressure_rate": 0.198, "mof_epa_allowed": 0.29, "rush_sr_allowed": 0.385},
    "DEN": {"quick_pressure_rate": 0.220, "mof_epa_allowed": 0.27, "rush_sr_allowed": 0.398},
    "SEA": {"quick_pressure_rate": 0.180, "mof_epa_allowed": 0.44, "rush_sr_allowed": 0.440},
    "LAR": {"quick_pressure_rate": 0.172, "mof_epa_allowed": 0.51, "rush_sr_allowed": 0.452},
    "ARI": {"quick_pressure_rate": 0.145, "mof_epa_allowed": 0.56, "rush_sr_allowed": 0.472},
    "LV":  {"quick_pressure_rate": 0.188, "mof_epa_allowed": 0.49, "rush_sr_allowed": 0.435},
    "NO":  {"quick_pressure_rate": 0.165, "mof_epa_allowed": 0.46, "rush_sr_allowed": 0.448},
    "ATL": {"quick_pressure_rate": 0.150, "mof_epa_allowed": 0.47, "rush_sr_allowed": 0.430},
    "IND": {"quick_pressure_rate": 0.168, "mof_epa_allowed": 0.53, "rush_sr_allowed": 0.462},
    "JAX": {"quick_pressure_rate": 0.160, "mof_epa_allowed": 0.55, "rush_sr_allowed": 0.455},
    "TEN": {"quick_pressure_rate": 0.170, "mof_epa_allowed": 0.45, "rush_sr_allowed": 0.405},
    "WAS": {"quick_pressure_rate": 0.155, "mof_epa_allowed": 0.52, "rush_sr_allowed": 0.468},
    "NYG": {"quick_pressure_rate": 0.185, "mof_epa_allowed": 0.47, "rush_sr_allowed": 0.475},
    "NE":  {"quick_pressure_rate": 0.162, "mof_epa_allowed": 0.43, "rush_sr_allowed": 0.438},
    "CAR": {"quick_pressure_rate": 0.130, "mof_epa_allowed": 0.58, "rush_sr_allowed": 0.485},
}


def get_defensive_scheme(team_code: str, window: int = 3) -> Dict[str, float]:
    """
    Retrieve defensive scheme metrics for an NFL team.
    Queries rolling moving average from scheme_db if available,
    falling back to NFL_DEFENSIVE_SCHEMES static priors.
    """
    team = (team_code or "").upper().strip()
    try:
        import scheme_db
        rolling = scheme_db.get_team_defense_rolling(team, window=window)
        if rolling and rolling.get("weeks_counted", 0) > 0:
            return rolling
    except Exception:
        pass

    return NFL_DEFENSIVE_SCHEMES.get(team, {
        "quick_pressure_rate": 0.185,
        "mof_epa_allowed": 0.38,
        "rush_sr_allowed": 0.420
    })


# ---------------------------------------------------------------------------
# 7B. RECEIVER ALIGNMENT PROFILES & SCHEME-TO-ALIGNMENT FUNNEL SYNERGY
# ---------------------------------------------------------------------------

RECEIVER_ALIGNMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    # Notable Slot Primaries (>60% slot)
    "DeMario Douglas": {"slot_snap_pct": 0.72, "wide_snap_pct": 0.28, "alignment_tag": "Slot Primary"},
    "Wan'Dale Robinson": {"slot_snap_pct": 0.78, "wide_snap_pct": 0.22, "alignment_tag": "Slot Primary"},
    "Christian Kirk": {"slot_snap_pct": 0.75, "wide_snap_pct": 0.25, "alignment_tag": "Slot Primary"},
    "Amon-Ra St. Brown": {"slot_snap_pct": 0.65, "wide_snap_pct": 0.35, "alignment_tag": "Slot Primary"},
    "Khalil Shakir": {"slot_snap_pct": 0.74, "wide_snap_pct": 0.26, "alignment_tag": "Slot Primary"},
    "Jayden Reed": {"slot_snap_pct": 0.70, "wide_snap_pct": 0.30, "alignment_tag": "Slot Primary"},
    "Rashee Rice": {"slot_snap_pct": 0.62, "wide_snap_pct": 0.38, "alignment_tag": "Slot Primary"},
    "Cooper Kupp": {"slot_snap_pct": 0.64, "wide_snap_pct": 0.36, "alignment_tag": "Slot Primary"},
    "Jauan Jennings": {"slot_snap_pct": 0.61, "wide_snap_pct": 0.39, "alignment_tag": "Slot Primary"},
    "Josh Downs": {"slot_snap_pct": 0.81, "wide_snap_pct": 0.19, "alignment_tag": "Slot Primary"},

    # Notable Boundary Primaries (>60% wide)
    "Drake London": {"slot_snap_pct": 0.28, "wide_snap_pct": 0.72, "alignment_tag": "Boundary Primary"},
    "DJ Moore": {"slot_snap_pct": 0.32, "wide_snap_pct": 0.68, "alignment_tag": "Boundary Primary"},
    "Tee Higgins": {"slot_snap_pct": 0.16, "wide_snap_pct": 0.84, "alignment_tag": "Boundary Primary"},
    "Justin Jefferson": {"slot_snap_pct": 0.35, "wide_snap_pct": 0.65, "alignment_tag": "Boundary Primary"},
    "Adonai Mitchell": {"slot_snap_pct": 0.18, "wide_snap_pct": 0.82, "alignment_tag": "Boundary Primary"},
    "Jack Bech": {"slot_snap_pct": 0.25, "wide_snap_pct": 0.75, "alignment_tag": "Boundary Primary"},
    "Nico Collins": {"slot_snap_pct": 0.22, "wide_snap_pct": 0.78, "alignment_tag": "Boundary Primary"},
    "George Pickens": {"slot_snap_pct": 0.15, "wide_snap_pct": 0.85, "alignment_tag": "Boundary Primary"},
    "DK Metcalf": {"slot_snap_pct": 0.14, "wide_snap_pct": 0.86, "alignment_tag": "Boundary Primary"},
    "Terry McLaurin": {"slot_snap_pct": 0.20, "wide_snap_pct": 0.80, "alignment_tag": "Boundary Primary"},
    "Quentin Johnston": {"slot_snap_pct": 0.18, "wide_snap_pct": 0.82, "alignment_tag": "Boundary Primary"},
    "Brian Thomas Jr.": {"slot_snap_pct": 0.16, "wide_snap_pct": 0.84, "alignment_tag": "Boundary Primary"},
    "Keon Coleman": {"slot_snap_pct": 0.24, "wide_snap_pct": 0.76, "alignment_tag": "Boundary Primary"},
    "Jameson Williams": {"slot_snap_pct": 0.20, "wide_snap_pct": 0.80, "alignment_tag": "Boundary Primary"},

    # Hybrids (40-60% slot/wide)
    "Jalen Coker": {"slot_snap_pct": 0.45, "wide_snap_pct": 0.55, "alignment_tag": "Hybrid"},
    "CeeDee Lamb": {"slot_snap_pct": 0.52, "wide_snap_pct": 0.48, "alignment_tag": "Hybrid"},
    "Puka Nacua": {"slot_snap_pct": 0.42, "wide_snap_pct": 0.58, "alignment_tag": "Hybrid"},
    "Deebo Samuel Sr.": {"slot_snap_pct": 0.40, "wide_snap_pct": 0.60, "alignment_tag": "Hybrid"},
    "Garrett Wilson": {"slot_snap_pct": 0.38, "wide_snap_pct": 0.62, "alignment_tag": "Hybrid"},
    "Zay Flowers": {"slot_snap_pct": 0.44, "wide_snap_pct": 0.56, "alignment_tag": "Hybrid"},
    "Malik Nabers": {"slot_snap_pct": 0.39, "wide_snap_pct": 0.61, "alignment_tag": "Hybrid"},
    "Marvin Harrison Jr.": {"slot_snap_pct": 0.34, "wide_snap_pct": 0.66, "alignment_tag": "Hybrid"},
}


def get_receiver_alignment(player_name: str, position: str = "WR") -> Dict[str, Any]:
    """
    Retrieve alignment profile for a receiver (slot_snap_pct, wide_snap_pct, alignment_tag).
    Defaults by position if player is unlisted:
      - TE: 65% interior/slot, 35% perimeter ('Slot Primary')
      - WR: 30% slot, 70% perimeter ('Boundary Primary')
      - RB: 15% slot/receiver, 85% backfield ('Backfield')
    """
    name = (player_name or "").strip()
    pos = (position or "").upper().strip()

    # Exact or case-insensitive search
    for p_key, profile in RECEIVER_ALIGNMENT_PROFILES.items():
        if p_key.lower() == name.lower():
            return profile.copy()

    # Substring search
    for p_key, profile in RECEIVER_ALIGNMENT_PROFILES.items():
        if p_key.lower() in name.lower() or name.lower() in p_key.lower():
            return profile.copy()

    # Positional defaults
    if pos == "TE":
        return {"slot_snap_pct": 0.65, "wide_snap_pct": 0.35, "alignment_tag": "Slot Primary"}
    elif pos == "WR":
        return {"slot_snap_pct": 0.30, "wide_snap_pct": 0.70, "alignment_tag": "Boundary Primary"}
    elif pos == "RB":
        return {"slot_snap_pct": 0.15, "wide_snap_pct": 0.85, "alignment_tag": "Backfield"}
    return {"slot_snap_pct": 0.0, "wide_snap_pct": 0.0, "alignment_tag": "Neutral"}


def apply_defensive_scheme_modifiers(
    player: Dict[str, Any],
    opponent_team: str,
    is_slot: Optional[bool] = None,
    week: Optional[int] = None
) -> Dict[str, Any]:
    """
    Adjust player distribution parameters (mu, sigma) based on Scheme-to-Alignment Funnel Synergy:
      1. Inside Funnel (High MOF EPA >0.40 & High Quick Pressure >20%):
         - Slot WRs & TEs: Median boosted by 1.15x (mu * 1.15) with compressed variance (sigma * 0.90, raised floor).
         - Boundary WRs: Volume slightly compressed (mu * 0.92) due to quick-pressure pocket collapse.
         - Hybrid WRs: Slight volume boost (mu * 1.08).
      2. Outside Funnel (Low MOF EPA <0.28 or <0.05 & Low Quick Pressure <18%):
         - Boundary WRs: Variance boosted by 1.25x (sigma * 1.25) to model deep 1-on-1 explosive ceiling games.
         - Slot WRs & TEs: Target efficiency regressed (mu * 0.90) due to bracketed middle-field coverage.
      3. QB & RB Baselines:
         - QB: Pocket compressed under QP >20% (mu * 0.92, sigma * 1.10).
         - RB: Boosted efficiency & tighter floor if Rush SR >45% (mu * 1.10, sigma * 0.85).
      4. Early-Season Bayesian Shrinkage (Weeks 1-3):
         - Dampens small-sample scheme volatility with a Bayesian shrinkage weight.
         - Caps maximum matchup drift strictly to +/- 8% (0.92x to 1.08x) in Weeks 1-3.
    """
    scheme = get_defensive_scheme(opponent_team)
    qpr = scheme["quick_pressure_rate"]
    mof_epa = scheme["mof_epa_allowed"]
    rush_sr = scheme["rush_sr_allowed"]

    pos = player.get("position", "WR").upper()
    p_name = player.get("name", "")

    # Inactive / IR players cannot receive scheme boosts and strictly project for 0.0 pts
    if (
        player.get("is_inactive")
        or float(player.get("p_active", 1.0)) <= 0.0
        or float(player.get("mu", 12.0)) <= 0.0
        or player.get("slot") in ["IR", "IR_ELIGIBLE"]
        or str(player.get("injury_status", "")).upper() in ["IR", "INJURY_RESERVE", "OUT"]
    ):
        return {
            **player,
            "mu": 0.0,
            "sigma": 0.0,
            "p_active": 0.0,
            "is_inactive": True,
            "scheme_notes": f"Inactive ({player.get('injury_status', 'IR')}) - 0.0 pts. No scheme modifiers applied."
        }

    orig_mu = float(player.get("mu", 12.0))
    orig_sigma = float(player.get("sigma", 5.0))
    mu = orig_mu
    sigma = orig_sigma
    mod_notes = []

    # Determine alignment profile
    align = get_receiver_alignment(p_name, pos)
    align_tag = align["alignment_tag"]
    if is_slot is True:
        align_tag = "Slot Primary"
    elif is_slot is False and align_tag == "Slot Primary" and pos == "WR":
        pass

    # 1. Inside Funnel Synergy: High MOF EPA (>=0.45 or >=0.38 with QP >= 19.5%)
    if mof_epa >= 0.45 or (mof_epa >= 0.38 and qpr >= 0.195) or (mof_epa > 0.40 and qpr > 0.20):
        if align_tag == "Slot Primary" or pos == "TE":
            mu *= 1.15
            sigma *= 0.90
            mod_notes.append(f"Inside Funnel (MOF EPA {mof_epa:.2f} / QP {qpr*100:.1f}%): Slot/TE median boosted (+15%) & floor raised")
        elif align_tag == "Hybrid":
            mu *= 1.08
            sigma *= 0.95
            mod_notes.append(f"Inside Funnel: Hybrid receiver captures middle checkdown volume (+8%)")
        elif align_tag == "Boundary Primary":
            mu *= 0.92
            mod_notes.append(f"Inside Funnel: Quick pressure compresses deep boundary route progression (-8%)")

    # 2. Outside Funnel Synergy: Low MOF EPA (<=0.28 or <0.05)
    elif mof_epa <= 0.28 or ((mof_epa < 0.28 or mof_epa < 0.05) and qpr < 0.18):
        if align_tag == "Boundary Primary":
            sigma *= 1.25
            mod_notes.append(f"Outside Funnel (MOF EPA {mof_epa:.2f} / QP {qpr*100:.1f}%): Clean pocket unlocks 1-on-1 boundary ceiling (+25% variance)")
        elif align_tag == "Slot Primary" or pos == "TE":
            mu *= 0.90
            mod_notes.append(f"Outside Funnel: Stifling middle defense regresses slot/TE target volume (-10%)")

    # 3. Standard Non-Funnel Fallback Adjustments
    else:
        if qpr > 0.20 and (pos == "RB" or align_tag == "Slot Primary"):
            mu *= 1.08
            mod_notes.append(f"Quick Press ({qpr*100:.1f}%): Checkdown target boost (+8%)")
        if mof_epa > 0.50 and (pos == "TE" or align_tag == "Slot Primary"):
            mu *= 1.15
            mod_notes.append(f"High MOF EPA ({mof_epa:.2f}): Middle-field target boost (+15%)")

    # 4. QB Pocket Compression (Quick Pressure > 20%)
    if pos == "QB" and qpr > 0.20:
        mu *= 0.92
        sigma *= 1.10
        mod_notes.append(f"Quick Press ({qpr*100:.1f}%): QB pocket compressed")

    # 5. RB High Rush SR Allowed (> 45%)
    if pos == "RB" and rush_sr > 0.45:
        mu *= 1.10
        sigma *= 0.85
        mod_notes.append(f"High Rush SR ({rush_sr*100:.1f}%): RB efficiency boost & tighter floor")

    # 6. Early-Season Bayesian Shrinkage Damping (Weeks 1-3): cap matchup drift to +/- 8%
    if week is not None and 1 <= week <= 3:
        if orig_mu > 0:
            drift_mu_raw = (mu - orig_mu) / orig_mu
            shrinkage_w = week / (week + 2.0)
            drift_mu_shrunk = drift_mu_raw * shrinkage_w
            drift_mu_capped = max(-0.08, min(0.08, drift_mu_shrunk))
            mu = orig_mu * (1.0 + drift_mu_capped)
            mod_notes.append(f"Early-Season Scheme Damping (Wk {week}): Matchup drift capped to {drift_mu_capped*100:+.1f}% (max ±8.0%)")

        if orig_sigma > 0:
            drift_sigma_raw = (sigma - orig_sigma) / orig_sigma
            shrinkage_w = week / (week + 2.0)
            drift_sigma_shrunk = drift_sigma_raw * shrinkage_w
            drift_sigma_capped = max(-0.08, min(0.08, drift_sigma_shrunk))
            sigma = orig_sigma * (1.0 + drift_sigma_capped)

    updated_player = player.copy()
    updated_player["mu"] = round(mu, 2)
    updated_player["sigma"] = round(sigma, 2)
    updated_player["alignment_tag"] = align_tag
    updated_player["slot_snap_pct"] = align.get("slot_snap_pct", 0.0)
    updated_player["wide_snap_pct"] = align.get("wide_snap_pct", 0.0)
    if mod_notes:
        updated_player["scheme_notes"] = "; ".join(mod_notes)
    return updated_player


def format_distribution_columns(
    df: Any,
    mu_col: str = "mu",
    sigma_col: str = "sigma",
    p10_col: Optional[str] = None,
    p90_col: Optional[str] = None
) -> Any:
    """
    Standardize raw statistical parameter columns ('mu', 'sigma') in display tables to:
    'Projected (Med)', 'Floor (P10)', and 'Ceiling (P90)'.
    """
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        return df

    df_out = df.copy()

    # If P10 and P90 already exist
    has_p10 = p10_col in df_out.columns if p10_col else ("Floor (P10)" in df_out.columns or "p10" in df_out.columns)
    has_p90 = p90_col in df_out.columns if p90_col else ("Ceiling (P90)" in df_out.columns or "p90" in df_out.columns)

    if not has_p10 and mu_col in df_out.columns and sigma_col in df_out.columns:
        df_out["Floor (P10)"] = df_out.apply(
            lambda r: round(max(0.0, float(r[mu_col]) - 1.28 * float(r[sigma_col])), 1)
            if float(r.get("p_active", 1.0)) > 0 and not r.get("is_inactive") else 0.0,
            axis=1
        )
    elif "p10" in df_out.columns:
        df_out["Floor (P10)"] = df_out["p10"].round(1)

    if not has_p90 and mu_col in df_out.columns and sigma_col in df_out.columns:
        df_out["Ceiling (P90)"] = df_out.apply(
            lambda r: round(max(0.0, float(r[mu_col]) + 1.28 * float(r[sigma_col])), 1)
            if float(r.get("p_active", 1.0)) > 0 and not r.get("is_inactive") else 0.0,
            axis=1
        )
    elif "p90" in df_out.columns:
        df_out["Ceiling (P90)"] = df_out["p90"].round(1)

    rename_map = {}
    if mu_col in df_out.columns:
        rename_map[mu_col] = "Projected (Med)"
    if "Proj Pts" in df_out.columns:
        rename_map["Proj Pts"] = "Projected (Med)"
    if "Proj Pts (μ)" in df_out.columns:
        rename_map["Proj Pts (μ)"] = "Projected (Med)"
    if "Median" in df_out.columns:
        rename_map["Median"] = "Projected (Med)"
    if "p10" in df_out.columns:
        rename_map["p10"] = "Floor (P10)"
    if "p90" in df_out.columns:
        rename_map["p90"] = "Ceiling (P90)"

    df_out = df_out.rename(columns=rename_map)

    # Drop raw sigma or Std (σ) column in final presentation view
    for s_name in [sigma_col, "sigma", "Std (σ)", "std"]:
        if s_name in df_out.columns:
            df_out = df_out.drop(columns=[s_name])

    return df_out


# ---------------------------------------------------------------------------
# 8. BAYESIAN USAGE VS. EFFICIENCY UPDATING
# ---------------------------------------------------------------------------

def apply_bayesian_usage_shrinkage(
    player: Dict[str, Any],
    snap_pct: Optional[float] = None,
    target_share: Optional[float] = None,
    air_yards_share: Optional[float] = None,
    ppr: float = 1.0,
    prior_weight: float = 0.70
) -> Tuple[float, float, Dict[str, float]]:
    """
    Bayesian shrinkage updating function.
    Weights stable volume metrics heavily (prior_weight w ~ 0.70)
    while regressing volatile efficiency metrics (TD rates, yards/target)
    toward historical positional means.

    Returns: (mu_bayes, sigma_bayes, diagnostic_metrics)
    """
    pos = player.get("position", "WR").upper()
    current_mu = float(player.get("mu", 12.0))
    current_sigma = float(player.get("sigma", 5.0))

    # Infer or extract opportunity metrics
    snap_p = snap_pct if snap_pct is not None else float(player.get("snap_pct", 0.75))
    tgt_s = target_share if target_share is not None else float(player.get("target_share", 0.18))
    ay_s = air_yards_share if air_yards_share is not None else float(player.get("air_yards_share", 0.22))

    # Positional baselines (NFL team averages: 34 passes/gm, 26 rushes/gm)
    if pos == "WR":
        # Expected volume opportunity
        exp_targets = tgt_s * 34.0
        exp_receptions = exp_targets * 0.65
        # Regress yards/target to positional mean (7.8 ypt) and TD rate (4.8% per target)
        regressed_rec_yds = exp_targets * 7.8
        regressed_tds = exp_targets * 0.048
        mu_volume = (exp_receptions * ppr) + (regressed_rec_yds * 0.10) + (regressed_tds * 6.0)

    elif pos == "TE":
        exp_targets = tgt_s * 32.0
        exp_receptions = exp_targets * 0.70
        regressed_rec_yds = exp_targets * 7.2
        regressed_tds = exp_targets * 0.055
        mu_volume = (exp_receptions * ppr) + (regressed_rec_yds * 0.10) + (regressed_tds * 6.0)

    elif pos == "RB":
        # RB volume derived from snap % and target share
        exp_carries = snap_p * 24.0 * 0.65
        exp_targets = tgt_s * 32.0
        exp_receptions = exp_targets * 0.78
        # Regress yards/carry to 4.25 ypc and TD rate to 3.5%
        regressed_rush_yds = exp_carries * 4.25
        regressed_rec_yds = exp_receptions * 7.2
        regressed_tds = (exp_carries + exp_targets) * 0.035
        mu_volume = (regressed_rush_yds * 0.10) + (exp_receptions * ppr) + (regressed_rec_yds * 0.10) + (regressed_tds * 6.0)

    elif pos == "QB":
        exp_attempts = 34.0
        exp_pass_yds = exp_attempts * 6.9
        exp_pass_tds = exp_attempts * 0.045
        exp_rush_yds = 18.0 * (1.2 if "Lamar" in player.get("name", "") or "Allen" in player.get("name", "") else 0.8)
        mu_volume = (exp_pass_yds * 0.04) + (exp_pass_tds * 4.0) + (exp_rush_yds * 0.10) - 1.2
    else:
        mu_volume = current_mu

    # Bayesian Shrinkage Formula:
    # mu_bayes = w * mu_volume + (1 - w) * current_mu
    w = max(0.0, min(1.0, prior_weight))
    mu_bayes = (w * mu_volume) + ((1.0 - w) * current_mu)

    # Variance adjustment: high snap share stabilizes variance
    if snap_p > 0.80:
        sigma_bayes = max(2.5, current_sigma * 0.92)
    elif snap_p < 0.40:
        sigma_bayes = current_sigma * 1.15
    else:
        sigma_bayes = current_sigma

    diagnostics = {
        "mu_volume_prior": round(mu_volume, 2),
        "mu_observed": round(current_mu, 2),
        "mu_bayes": round(mu_bayes, 2),
        "prior_weight": w,
        "snap_pct": snap_p,
        "target_share": tgt_s
    }

    return round(mu_bayes, 2), round(sigma_bayes, 2), diagnostics

