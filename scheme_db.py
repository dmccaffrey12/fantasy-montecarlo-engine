"""
scheme_db.py
Persistent Scheme Storage and Historical Tracking Engine for NFL Analytics.
Stores historical weekly scheme metrics and player usage metrics in local Parquet files.

Features:
- Parquet storage in data/scheme_history.parquet and data/player_usage_history.parquet
- Week-level upsert / overwrite logic (overwrites existing week data on re-upload)
- Rolling 3-week moving average aggregation (get_team_defense_rolling)
- Scheme trend detection (get_scheme_trend): delta between latest week and season average
- Player weekly usage trajectory (get_player_usage_history)
- Automatic initial seed generation from 32-team baseline
"""

import os
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Path configurations
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SCHEME_PARQUET = os.path.join(DATA_DIR, "scheme_history.parquet")
SCHEME_CSV = os.path.join(DATA_DIR, "scheme_history.csv")
PLAYER_PARQUET = os.path.join(DATA_DIR, "player_usage_history.parquet")
PLAYER_CSV = os.path.join(DATA_DIR, "player_usage_history.csv")

# Baseline NFL Defensive Scheme Priors (fallback when history is shallow)
DEFAULT_DEFENSIVE_SCHEMES: Dict[str, Dict[str, float]] = {
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


def _ensure_data_dir() -> None:
    """Ensure data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)


def load_scheme_history() -> pd.DataFrame:
    """Load historical team scheme metrics from parquet (or CSV fallback)."""
    _ensure_data_dir()
    if os.path.exists(SCHEME_PARQUET):
        try:
            return pd.read_parquet(SCHEME_PARQUET)
        except Exception:
            pass
    if os.path.exists(SCHEME_CSV):
        try:
            return pd.read_csv(SCHEME_CSV)
        except Exception:
            pass
    # If no data exists yet, initialize defaults
    init_default_scheme_history()
    if os.path.exists(SCHEME_PARQUET):
        return pd.read_parquet(SCHEME_PARQUET)
    return pd.DataFrame(columns=[
        "week", "team", "opponent", "pressure_rate", "quick_pressure_rate",
        "mof_pct", "mof_epa_allowed", "pass_oe", "epa_db", "epa_rush", "rush_sr_allowed"
    ])


def load_player_history() -> pd.DataFrame:
    """Load historical player usage metrics from parquet (or CSV fallback)."""
    _ensure_data_dir()
    if os.path.exists(PLAYER_PARQUET):
        try:
            return pd.read_parquet(PLAYER_PARQUET)
        except Exception:
            pass
    if os.path.exists(PLAYER_CSV):
        try:
            return pd.read_csv(PLAYER_CSV)
        except Exception:
            pass
    # If no data exists, initialize defaults
    init_default_scheme_history()
    if os.path.exists(PLAYER_PARQUET):
        return pd.read_parquet(PLAYER_PARQUET)
    return pd.DataFrame(columns=[
        "week", "player_name", "team", "position", "target_share", "yprr", "adot", "rush_epa", "rush_sr"
    ])


def save_newsletter_metrics(
    team_df: pd.DataFrame,
    player_df: pd.DataFrame,
    week: int
) -> Dict[str, Any]:
    """
    Save or update weekly SumerSports metrics in persistent storage.
    If data for the given week already exists, it is overwritten for that week.
    Other weeks remain preserved.
    """
    _ensure_data_dir()
    target_week = int(week)
    results = {"success": True, "week": target_week, "team_rows": 0, "player_rows": 0}

    # 1. Update Team Scheme Metrics
    if team_df is not None and not team_df.empty:
        df_new_teams = team_df.copy()
        df_new_teams["week"] = target_week
        df_new_teams["team"] = df_new_teams["team"].astype(str).str.upper().str.strip()
        if "opponent" in df_new_teams.columns:
            df_new_teams["opponent"] = df_new_teams["opponent"].astype(str).str.upper().str.strip()

        num_cols_team = ["pressure_rate", "quick_pressure_rate", "mof_pct", "mof_epa_allowed", "pass_oe", "epa_db", "epa_rush", "rush_sr_allowed"]
        for col in num_cols_team:
            if col in df_new_teams.columns:
                df_new_teams[col] = pd.to_numeric(df_new_teams[col], errors="coerce").fillna(0.0)

        existing_teams = load_scheme_history()
        if not existing_teams.empty:
            retained_teams = existing_teams[existing_teams["week"] != target_week]
            combined_teams = pd.concat([retained_teams, df_new_teams], ignore_index=True)
        else:
            combined_teams = df_new_teams

        combined_teams = combined_teams.sort_values(by=["week", "team"]).reset_index(drop=True)
        try:
            combined_teams.to_parquet(SCHEME_PARQUET, index=False)
        except Exception:
            combined_teams.to_csv(SCHEME_CSV, index=False)
        results["team_rows"] = len(df_new_teams)

    # 2. Update Player Usage Metrics
    if player_df is not None and not player_df.empty:
        df_new_players = player_df.copy()
        df_new_players["week"] = target_week
        df_new_players["player_name"] = df_new_players["player_name"].astype(str).str.strip()
        if "team" in df_new_players.columns:
            df_new_players["team"] = df_new_players["team"].astype(str).str.upper().str.strip()
        if "position" in df_new_players.columns:
            df_new_players["position"] = df_new_players["position"].astype(str).str.upper().str.strip()

        num_cols_player = ["target_share", "yprr", "adot", "rush_epa", "rush_sr"]
        for col in num_cols_player:
            if col in df_new_players.columns:
                df_new_players[col] = pd.to_numeric(df_new_players[col], errors="coerce").fillna(0.0)

        existing_players = load_player_history()
        if not existing_players.empty:
            retained_players = existing_players[existing_players["week"] != target_week]
            combined_players = pd.concat([retained_players, df_new_players], ignore_index=True)
        else:
            combined_players = df_new_players

        combined_players = combined_players.sort_values(by=["week", "player_name"]).reset_index(drop=True)
        try:
            combined_players.to_parquet(PLAYER_PARQUET, index=False)
        except Exception:
            combined_players.to_csv(PLAYER_CSV, index=False)
        results["player_rows"] = len(df_new_players)

    return results


def get_team_defense_rolling(team: str, window: int = 3) -> Dict[str, float]:
    """
    Retrieve rolling moving average defense metrics for a team over the last `window` weeks.
    """
    t_clean = (team or "").upper().strip()
    prior = DEFAULT_DEFENSIVE_SCHEMES.get(t_clean, {
        "quick_pressure_rate": 0.185,
        "mof_epa_allowed": 0.38,
        "rush_sr_allowed": 0.420
    })

    df = load_scheme_history()
    if df.empty or "team" not in df.columns:
        return {
            "quick_pressure_rate": prior["quick_pressure_rate"],
            "mof_epa_allowed": prior["mof_epa_allowed"],
            "rush_sr_allowed": prior["rush_sr_allowed"],
            "pressure_rate": prior.get("quick_pressure_rate", 0.185) * 1.6,
            "mof_pct": 0.40,
            "weeks_counted": 0,
            "is_rolling": False
        }

    team_data = df[df["team"] == t_clean].sort_values(by="week", ascending=True)
    if team_data.empty:
        return {
            "quick_pressure_rate": prior["quick_pressure_rate"],
            "mof_epa_allowed": prior["mof_epa_allowed"],
            "rush_sr_allowed": prior["rush_sr_allowed"],
            "pressure_rate": prior.get("quick_pressure_rate", 0.185) * 1.6,
            "mof_pct": 0.40,
            "weeks_counted": 0,
            "is_rolling": False
        }

    recent_data = team_data.tail(window)
    n_weeks = len(recent_data)

    obs_qpr = float(recent_data["quick_pressure_rate"].mean())
    obs_mof = float(recent_data["mof_epa_allowed"].mean())
    obs_rsr = float(recent_data["rush_sr_allowed"].mean())
    obs_pr = float(recent_data["pressure_rate"].mean()) if "pressure_rate" in recent_data.columns else obs_qpr * 1.6
    obs_mofp = float(recent_data["mof_pct"].mean()) if "mof_pct" in recent_data.columns else 0.40

    if n_weeks < window:
        obs_weight = n_weeks / float(window)
        prior_weight = 1.0 - obs_weight
        final_qpr = (obs_qpr * obs_weight) + (prior["quick_pressure_rate"] * prior_weight)
        final_mof = (obs_mof * obs_weight) + (prior["mof_epa_allowed"] * prior_weight)
        final_rsr = (obs_rsr * obs_weight) + (prior["rush_sr_allowed"] * prior_weight)
    else:
        final_qpr = obs_qpr
        final_mof = obs_mof
        final_rsr = obs_rsr

    return {
        "quick_pressure_rate": round(final_qpr, 3),
        "mof_epa_allowed": round(final_mof, 3),
        "rush_sr_allowed": round(final_rsr, 3),
        "pressure_rate": round(obs_pr, 3),
        "mof_pct": round(obs_mofp, 3),
        "weeks_counted": n_weeks,
        "is_rolling": True
    }


def get_scheme_trend(team: str) -> Dict[str, Any]:
    """
    Compute delta between the latest week's metrics and the team's season average.
    """
    t_clean = (team or "").upper().strip()
    df = load_scheme_history()
    default_resp = {
        "team": t_clean,
        "latest_week": 0,
        "delta_quick_pressure": 0.0,
        "delta_mof_epa": 0.0,
        "delta_rush_sr": 0.0,
        "trend_summary": "Neutral / Insufficient Data"
    }

    if df.empty or "team" not in df.columns:
        return default_resp

    team_data = df[df["team"] == t_clean].sort_values(by="week", ascending=True)
    if len(team_data) < 2:
        if len(team_data) == 1:
            default_resp["latest_week"] = int(team_data.iloc[-1]["week"])
            default_resp["trend_summary"] = "Single week sample baseline"
        return default_resp

    latest_row = team_data.iloc[-1]
    latest_week = int(latest_row["week"])
    season_avg = team_data.mean(numeric_only=True)

    delta_qp = round(float(latest_row["quick_pressure_rate"] - season_avg["quick_pressure_rate"]), 3)
    delta_mof = round(float(latest_row["mof_epa_allowed"] - season_avg["mof_epa_allowed"]), 3)
    delta_rsr = round(float(latest_row["rush_sr_allowed"] - season_avg["rush_sr_allowed"]), 3)

    notes = []
    if delta_qp > 0.025:
        notes.append("Pass Rush Surging")
    elif delta_qp < -0.025:
        notes.append("Pass Rush Cooling")

    if delta_mof > 0.08:
        notes.append("MOF Vulnerability Growing")
    elif delta_mof < -0.08:
        notes.append("MOF Tightening")

    if delta_rsr > 0.04:
        notes.append("Run Defense Bleeding")
    elif delta_rsr < -0.04:
        notes.append("Run Defense Stifling")

    summary = ", ".join(notes) if notes else "Stable Scheme Baseline"

    return {
        "team": t_clean,
        "latest_week": latest_week,
        "delta_quick_pressure": delta_qp,
        "delta_mof_epa": delta_mof,
        "delta_rush_sr": delta_rsr,
        "trend_summary": summary
    }


def get_player_usage_history(player_name: str) -> pd.DataFrame:
    """
    Retrieve historical weekly usage metrics for a specific player.
    """
    p_clean = (player_name or "").strip().lower()
    df = load_player_history()
    if df.empty or "player_name" not in df.columns:
        return pd.DataFrame(columns=["week", "player_name", "team", "position", "target_share", "yprr", "adot", "rush_epa", "rush_sr"])

    mask = df["player_name"].str.lower().str.contains(p_clean, regex=False)
    matched = df[mask].sort_values(by="week", ascending=True).reset_index(drop=True)
    return matched


def get_all_teams_in_db() -> List[str]:
    """Return sorted list of all unique team codes in the scheme history."""
    df = load_scheme_history()
    if df.empty or "team" not in df.columns:
        return sorted(list(DEFAULT_DEFENSIVE_SCHEMES.keys()))
    teams = sorted(list(df["team"].unique()))
    return teams if teams else sorted(list(DEFAULT_DEFENSIVE_SCHEMES.keys()))


def get_all_players_in_db() -> List[str]:
    """Return sorted list of all unique player names in player usage history."""
    df = load_player_history()
    if df.empty or "player_name" not in df.columns:
        return []
    return sorted(list(df["player_name"].unique()))


def init_default_scheme_history(force: bool = False) -> None:
    """
    Seed initial 3 weeks of scheme data and key player usage metrics
    if database is empty or force=True.
    """
    _ensure_data_dir()
    if not force and os.path.exists(SCHEME_PARQUET):
        return

    team_rows = []
    for week in [1, 2, 3]:
        for team, priors in DEFAULT_DEFENSIVE_SCHEMES.items():
            noise_qp = ((hash(f"{team}_{week}_qp") % 21) - 10) / 1000.0
            noise_mof = ((hash(f"{team}_{week}_mof") % 21) - 10) / 200.0
            noise_rsr = ((hash(f"{team}_{week}_rsr") % 21) - 10) / 1000.0

            qp = max(0.10, min(0.35, priors["quick_pressure_rate"] + noise_qp))
            mof = max(0.15, min(0.70, priors["mof_epa_allowed"] + noise_mof))
            rsr = max(0.30, min(0.55, priors["rush_sr_allowed"] + noise_rsr))

            team_rows.append({
                "week": week,
                "team": team,
                "opponent": "OPP",
                "pressure_rate": round(qp * 1.62, 3),
                "quick_pressure_rate": round(qp, 3),
                "mof_pct": round(0.38 + (week * 0.01), 3),
                "mof_epa_allowed": round(mof, 2),
                "pass_oe": round(1.2 if qp > 0.20 else -1.5, 1),
                "epa_db": round(0.04, 2),
                "epa_rush": round(-0.06, 2),
                "rush_sr_allowed": round(rsr, 3)
            })

    df_teams = pd.DataFrame(team_rows)
    try:
        df_teams.to_parquet(SCHEME_PARQUET, index=False)
    except Exception:
        df_teams.to_csv(SCHEME_CSV, index=False)

    sample_players = [
        ("CeeDee Lamb", "DAL", "WR", [0.28, 0.31, 0.29], [2.65, 2.82, 2.45], [10.2, 11.5, 9.8], 0.0, 0.0),
        ("Amon-Ra St. Brown", "DET", "WR", [0.26, 0.29, 0.28], [2.40, 2.55, 2.48], [7.8, 8.1, 8.0], 0.0, 0.0),
        ("Justin Jefferson", "MIN", "WR", [0.30, 0.28, 0.32], [2.75, 2.60, 2.90], [12.4, 11.8, 13.1], 0.0, 0.0),
        ("Demario Douglas", "NE", "WR", [0.22, 0.25, 0.26], [1.95, 2.10, 2.05], [6.5, 7.1, 6.8], 0.0, 0.0),
        ("Jalen Coker", "CAR", "WR", [0.12, 0.16, 0.18], [1.45, 1.68, 1.75], [9.8, 10.4, 10.1], 0.0, 0.0),
        ("Bijan Robinson", "ATL", "RB", [0.18, 0.19, 0.20], [1.60, 1.75, 1.82], [1.2, 1.5, 1.8], 0.12, 0.48),
        ("Breece Hall", "NYJ", "RB", [0.16, 0.18, 0.17], [1.50, 1.62, 1.55], [0.8, 1.1, 0.9], 0.08, 0.46),
        ("Trey McBride", "ARI", "TE", [0.24, 0.26, 0.25], [2.10, 2.25, 2.18], [7.2, 7.5, 7.0], 0.0, 0.0),
        ("Brock Bowers", "LV", "TE", [0.25, 0.24, 0.27], [2.20, 2.15, 2.35], [8.0, 7.9, 8.4], 0.0, 0.0),
    ]

    player_rows = []
    for p_name, p_team, p_pos, tgts, yprrs, adots, r_epa, r_sr in sample_players:
        for w_idx, week in enumerate([1, 2, 3]):
            player_rows.append({
                "week": week,
                "player_name": p_name,
                "team": p_team,
                "position": p_pos,
                "target_share": tgts[w_idx],
                "yprr": yprrs[w_idx],
                "adot": adots[w_idx],
                "rush_epa": r_epa,
                "rush_sr": r_sr
            })

    df_players = pd.DataFrame(player_rows)
    try:
        df_players.to_parquet(PLAYER_PARQUET, index=False)
    except Exception:
        df_players.to_csv(PLAYER_CSV, index=False)


def get_defensive_funnel_profile(team: str) -> Dict[str, Any]:
    """
    Analyzes an NFL defense to classify its target funnel tendency:
      - 'Inside Funnel (Slot / TE)': High MOF EPA allowed (>0.40) & High Quick Pressure (>20%)
      - 'Outside Funnel (Boundary)': Low MOF EPA (<0.28 or <0.05) & Low Quick Pressure (<18%)
      - 'Balanced Coverage Shell': Intermediate/balanced tendencies
    """
    rolling = get_team_defense_rolling(team, window=3)
    qpr = rolling.get("quick_pressure_rate", 0.185)
    mof_epa = rolling.get("mof_epa_allowed", 0.38)
    rush_sr = rolling.get("rush_sr_allowed", 0.420)
    pr = rolling.get("pressure_rate", qpr * 1.6)

    if mof_epa >= 0.45 or (mof_epa >= 0.38 and qpr >= 0.195):
        funnel_type = "Inside Funnel (Slot / TE)"
        badge_color = "#10b981"
        description = (
            f"Porous middle-of-field coverage (MOF EPA {mof_epa:.2f}) and defensive pressure ({qpr*100:.1f}%). "
            "Passing volume funnels heavily inside to Slot Receivers & Tight Ends."
        )
        tactical_edge = "Slot Primary & TE (+15% median boost, compressed floor)"
        recommended_alignment = "Slot Primary"
    elif mof_epa <= 0.28:
        funnel_type = "Outside Funnel (Boundary)"
        badge_color = "#38bdf8"
        description = (
            f"Stifling middle-of-field coverage (MOF EPA {mof_epa:.2f}). "
            "Defenses bracket the middle, forcing targets to perimeter boundary receivers."
        )
        tactical_edge = "Boundary Primary (+25% variance ceiling games)"
        recommended_alignment = "Boundary Primary"
    else:
        funnel_type = "Balanced Coverage Shell"
        badge_color = "#94a3b8"
        description = f"Neutral coverage balance (MOF EPA {mof_epa:.2f}, Quick Press {qpr*100:.1f}%). Standard volume distribution applies."
        tactical_edge = "Neutral coverage alignment"
        recommended_alignment = "Hybrid / Best Volume"

    is_inside = funnel_type == "Inside Funnel (Slot / TE)"
    is_outside = funnel_type == "Outside Funnel (Boundary)"

    return {
        "team": (team or "").upper(),
        "funnel_type": funnel_type,
        "funnel_classification": funnel_type,
        "badge_color": badge_color,
        "quick_pressure_rate": qpr,
        "mof_epa_allowed": mof_epa,
        "rush_sr_allowed": rush_sr,
        "pressure_rate": pr,
        "description": description,
        "summary": description,
        "tactical_edge": tactical_edge,
        "recommended_alignment": recommended_alignment,
        "is_inside_funnel": is_inside,
        "is_outside_funnel": is_outside
    }


def detect_floor_surges_and_buy_lows(min_target_share: float = 0.12) -> pd.DataFrame:
    """
    Scans historical weekly player trajectories to identify:
      - 📈 FLOOR SURGING (Target in Trades):
          Target share and weekly floor are trending upward consecutively, signaling an expanding baseline role.
      - 💎 CONTRARIAN BUY-LOW (TD Regression Due):
          Elite underlying volume (target share >= 20% or YPRR >= 2.0) with deflated fantasy scoring.
      - ⚠️ SELL HIGH (Fragile Volume):
          High recent scoring despite low target share (<15%) and low efficiency.
    """
    df_players = load_player_history()
    if df_players.empty or "player_name" not in df_players.columns:
        return pd.DataFrame()

    results = []
    for p_name, group in df_players.groupby("player_name"):
        sorted_g = group.sort_values(by="week", ascending=True)
        if len(sorted_g) < 2:
            continue

        tgt_shares = sorted_g["target_share"].tolist()
        yprrs = sorted_g["yprr"].tolist()
        adots = sorted_g["adot"].tolist()
        latest = sorted_g.iloc[-1]
        earliest = sorted_g.iloc[0]

        latest_tgt = latest["target_share"]
        latest_yprr = latest["yprr"]
        latest_adot = latest["adot"]
        delta_tgt = round(latest_tgt - earliest["target_share"], 3)
        delta_yprr = round(latest_yprr - earliest["yprr"], 2)

        # Classification logic
        if delta_tgt >= 0.03 and latest_tgt >= 0.18 and latest_yprr >= 1.60:
            classification = "📈 FLOOR SURGING"
            trade_action = "BUY / TARGET"
            rationale = (
                f"Target share surged {delta_tgt*100:+.1f}% across 3 weeks (now {latest_tgt*100:.1f}%) "
                f"with strong {latest_yprr:.2f} YPRR. Expanding weekly baseline makes them an elite trade target."
            )
        elif latest_tgt >= 0.22 and latest_yprr >= 1.90:
            classification = "💎 CONTRARIAN BUY-LOW"
            trade_action = "STRONG BUY"
            rationale = (
                f"Alpha-level utilization ({latest_tgt*100:.1f}% target share, {latest_yprr:.2f} YPRR). "
                "Touchdown positive regression due; exploit owner impatience."
            )
        elif latest_tgt < 0.15 and delta_tgt <= 0:
            classification = "⚠️ SELL HIGH / FADE"
            trade_action = "SELL / BENCH"
            rationale = (
                f"Sub-15% target share ({latest_tgt*100:.1f}%) and declining route volume ({delta_tgt*100:+.1f}%). "
                "Fragile baseline role vulnerable to dud weeks."
            )
        else:
            classification = "⚖️ STABLE ROLE"
            trade_action = "HOLD"
            rationale = f"Stable role ({latest_tgt*100:.1f}% target share, {latest_yprr:.2f} YPRR). Fairly priced."

        results.append({
            "Player": p_name,
            "Team": latest["team"],
            "Pos": latest["position"],
            "Latest Tgt%": f"{latest_tgt*100:.1f}%",
            "Tgt% Delta": f"{delta_tgt*100:+.1f}%",
            "Latest YPRR": round(latest_yprr, 2),
            "YPRR Delta": f"{delta_yprr:+.2f}",
            "aDOT": round(latest_adot, 1),
            "Trade Signal": classification,
            "Action": trade_action,
            "Trade & Strategy Rationale": rationale,
            "Raw Tgt": latest_tgt,
            "Raw Delta": delta_tgt
        })

    if not results:
        return pd.DataFrame()

    df_out = pd.DataFrame(results)
    order_map = {"BUY / TARGET": 0, "STRONG BUY": 1, "SELL / BENCH": 2, "HOLD": 3}
    df_out["sort_key"] = df_out["Action"].map(order_map).fillna(4)
    df_out = df_out.sort_values(by=["sort_key", "Raw Delta"], ascending=[True, False]).drop(columns=["sort_key", "Raw Tgt", "Raw Delta"]).reset_index(drop=True)
    return df_out


if not os.path.exists(SCHEME_PARQUET):
    init_default_scheme_history()
