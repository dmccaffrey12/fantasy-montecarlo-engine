"""
scheme_db.py
Persistent Dual-Document Scheme Repository, Longitudinal Narrative Ledger, and Expectation-vs-Reality Engine.

Repository Architecture (data/):
1. `data/scheme_reviews.parquet` (& `data/scheme_history.parquet` mirror):
   - Realized in-game stats: Team, Opponent, Week, Pressure Rate %, Quick Pressure %,
     MOF %, MOF EPA, Pass OE %, EPA/DB, EPA/Rush, Rush SR, Matchup Takeaway, Week Takeaway.
   - Player receiving/rushing tables (`data/player_usage_history.parquet`):
     Player, Team, Position, Target %, YPRR, aDOT, Rush EPA, Rush SR.
2. `data/scheme_previews.parquet`:
   - Matchup previews: Team, Opponent, Week, Cover 1/3 Man %, Cover 2/4/6 Zone %, Blitz %,
     Expected Quick Pressure %, Expected MOF EPA, Explosive Play Rate Allowed %,
     Funnel Matchup Focus (Inside/Slot vs Boundary advantage), Tactical Takeaways.
3. `data/team_narratives.json`:
   - Multi-week longitudinal ledger mapping each team's trajectory from Week 1 to present:
     * Preseason expectation
     * Weekly preview hypotheses vs. weekly review execution
     * Structural regime tags ('PASS_PROTECTION_COLLAPSE', 'RUN_GAME_BROKEN', 'SLOT_FUNNEL_HIGH', 'GARBAGE_TIME_ALPHA')
     * Rolling 3-week averages & Bayesian prior decay / floor-ceiling modifiers.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
NEWSLETTERS_DIR = os.path.join(BASE_DIR, "newsletters")

# Primary Dual-Document Parquet & JSON Ledger Paths
REVIEWS_PARQUET = os.path.join(DATA_DIR, "scheme_reviews.parquet")
REVIEWS_CSV = os.path.join(DATA_DIR, "scheme_reviews.csv")
PREVIEWS_PARQUET = os.path.join(DATA_DIR, "scheme_previews.parquet")
PREVIEWS_CSV = os.path.join(DATA_DIR, "scheme_previews.csv")
PLAYER_ANGLES_PARQUET = os.path.join(DATA_DIR, "player_preview_angles.parquet")
TEAM_NARRATIVES_JSON = os.path.join(DATA_DIR, "team_narratives.json")

# Backward-compatible aliases
SCHEME_PARQUET = os.path.join(DATA_DIR, "scheme_history.parquet")
SCHEME_CSV = os.path.join(DATA_DIR, "scheme_history.csv")
PLAYER_PARQUET = os.path.join(DATA_DIR, "player_usage_history.parquet")
PLAYER_CSV = os.path.join(DATA_DIR, "player_usage_history.csv")

# Baseline NFL Defensive Scheme Priors
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
    "WSH": {"quick_pressure_rate": 0.155, "mof_epa_allowed": 0.52, "rush_sr_allowed": 0.468},
    "NYG": {"quick_pressure_rate": 0.185, "mof_epa_allowed": 0.47, "rush_sr_allowed": 0.475},
    "NE":  {"quick_pressure_rate": 0.162, "mof_epa_allowed": 0.43, "rush_sr_allowed": 0.438},
    "CAR": {"quick_pressure_rate": 0.130, "mof_epa_allowed": 0.58, "rush_sr_allowed": 0.485},
    "JAC": {"quick_pressure_rate": 0.160, "mof_epa_allowed": 0.55, "rush_sr_allowed": 0.455},
}

PRESEASON_TEAM_EXPECTATIONS: Dict[str, str] = {
    "CLE": "Aggressive Man-Heavy Pass Rush (Top-5 Quick Pressure Expected) & Play-Action Vertical Offense",
    "PHI": "Fangio Two-High Shell Transition & High-Efficiency RPO / Tush-Push Ground Attack",
    "CAR": "Rebuild Zone Shell Vulnerable in Middle-of-Field (Projected Inside Slot/TE Funnel)",
    "TB":  "Todd Bowles Exotic Blitz Package & High-Volume Interior Slot/Play-Action Passing",
    "BUF": "Nickel Two-High Disguise Defense & Josh Allen High-EPA Spread Progression",
    "DET": "Press-Man Heavy Run Wall (Top-3 Rush Defense) & Elite Play-Action Slot/Seam Attack",
    "NYG": "Dexter Lawrence Interior Pass Rush & High-Target Alpha Perimeter Passing",
    "MIN": "Brian Flores Zero-Blitz / Simulated Pressure Lab & Vertical Play-Action",
    "BAL": "Split-Safety disguise & Historic QB/RB Dual-Threat Rushing Efficiency",
    "SF":  "Wide-9 4-Man Rush & Shanahan Motion/YAC Middle-Field Dominance",
    "KC":  "Spagnuolo Press-Man Blitz & High-Percentage Underneath Option Offense",
}


def _ensure_data_dir() -> None:
    """Ensure data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)


def _save_df_parquet(df: pd.DataFrame, parquet_path: str, csv_fallback: Optional[str] = None) -> None:
    _ensure_data_dir()
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        if csv_fallback:
            df.to_csv(csv_fallback, index=False)


def _load_df_parquet(parquet_path: str, csv_fallback: Optional[str] = None) -> pd.DataFrame:
    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_fallback and os.path.exists(csv_fallback):
        try:
            return pd.read_csv(csv_fallback)
        except Exception:
            pass
    return pd.DataFrame()


def load_scheme_reviews() -> pd.DataFrame:
    """Load realized weekly game reviews from `data/scheme_reviews.parquet` (or `scheme_history.parquet`)."""
    _ensure_data_dir()
    df = _load_df_parquet(REVIEWS_PARQUET, REVIEWS_CSV)
    if not df.empty:
        return df
    df_hist = _load_df_parquet(SCHEME_PARQUET, SCHEME_CSV)
    if not df_hist.empty:
        _save_df_parquet(df_hist, REVIEWS_PARQUET, REVIEWS_CSV)
        return df_hist
    init_default_scheme_history()
    return _load_df_parquet(REVIEWS_PARQUET, REVIEWS_CSV)


def load_scheme_history() -> pd.DataFrame:
    """Backward-compatible loader for realized team scheme reviews."""
    return load_scheme_reviews()


def load_scheme_previews() -> pd.DataFrame:
    """Load pre-game weekly matchup previews from `data/scheme_previews.parquet`."""
    _ensure_data_dir()
    df = _load_df_parquet(PREVIEWS_PARQUET, PREVIEWS_CSV)
    if not df.empty:
        return df
    init_default_scheme_history()
    return _load_df_parquet(PREVIEWS_PARQUET, PREVIEWS_CSV)


def load_player_history() -> pd.DataFrame:
    """Load historical player usage metrics from `data/player_usage_history.parquet`."""
    _ensure_data_dir()
    df = _load_df_parquet(PLAYER_PARQUET, PLAYER_CSV)
    if not df.empty:
        return df
    init_default_scheme_history()
    return _load_df_parquet(PLAYER_PARQUET, PLAYER_CSV)


def load_player_preview_angles() -> pd.DataFrame:
    """Load player-level matchup preview angles."""
    _ensure_data_dir()
    return _load_df_parquet(PLAYER_ANGLES_PARQUET)


def save_preview_metrics(
    preview_df: pd.DataFrame,
    player_angles_df: Optional[pd.DataFrame] = None,
    week: int = 1
) -> Dict[str, Any]:
    """
    Idempotently upsert weekly matchup preview metrics into `data/scheme_previews.parquet`
    and rebuild `data/team_narratives.json`.
    """
    _ensure_data_dir()
    target_week = int(week)
    results = {"success": True, "doc_type": "Weekly Preview", "week": target_week, "preview_rows": 0, "angle_rows": 0}

    if preview_df is not None and not preview_df.empty:
        df_new = preview_df.copy()
        df_new["week"] = target_week
        df_new["team"] = df_new["team"].astype(str).str.upper().str.strip()
        if "opponent" in df_new.columns:
            df_new["opponent"] = df_new["opponent"].astype(str).str.upper().str.strip()

        num_cols = [
            "man_coverage_rate", "zone_coverage_rate", "blitz_rate",
            "expected_quick_pressure_rate", "expected_mof_epa", "explosive_play_rate_allowed"
        ]
        for col in num_cols:
            if col in df_new.columns:
                df_new[col] = pd.to_numeric(df_new[col], errors="coerce").fillna(0.0)

        existing = _load_df_parquet(PREVIEWS_PARQUET, PREVIEWS_CSV)
        if not existing.empty and "week" in existing.columns:
            retained = existing[existing["week"] != target_week]
            combined = pd.concat([retained, df_new], ignore_index=True)
        else:
            combined = df_new

        combined = combined.sort_values(by=["week", "team"]).reset_index(drop=True)
        _save_df_parquet(combined, PREVIEWS_PARQUET, PREVIEWS_CSV)
        results["preview_rows"] = len(df_new)

    if player_angles_df is not None and not player_angles_df.empty:
        df_ang = player_angles_df.copy()
        df_ang["week"] = target_week
        existing_ang = _load_df_parquet(PLAYER_ANGLES_PARQUET)
        if not existing_ang.empty and "week" in existing_ang.columns:
            retained_ang = existing_ang[existing_ang["week"] != target_week]
            combined_ang = pd.concat([retained_ang, df_ang], ignore_index=True)
        else:
            combined_ang = df_ang
        combined_ang = combined_ang.sort_values(by=["week", "player_name"]).reset_index(drop=True)
        _save_df_parquet(combined_ang, PLAYER_ANGLES_PARQUET)
        results["angle_rows"] = len(df_ang)

    rebuild_team_narratives()
    return results


def save_newsletter_metrics(
    team_df: pd.DataFrame,
    player_df: pd.DataFrame,
    week: int
) -> Dict[str, Any]:
    """
    Idempotently upsert realized weekly review metrics into `data/scheme_reviews.parquet`
    (and `data/scheme_history.parquet` + `data/player_usage_history.parquet`) and rebuild
    `data/team_narratives.json`.
    """
    _ensure_data_dir()
    target_week = int(week)
    results = {"success": True, "doc_type": "Weekly Review", "week": target_week, "team_rows": 0, "player_rows": 0}

    # 1. Update Team Realized Scheme Reviews
    if team_df is not None and not team_df.empty:
        df_new_teams = team_df.copy()
        df_new_teams["week"] = target_week
        df_new_teams["team"] = df_new_teams["team"].astype(str).str.upper().str.strip()
        if "opponent" in df_new_teams.columns:
            df_new_teams["opponent"] = df_new_teams["opponent"].astype(str).str.upper().str.strip()

        num_cols_team = [
            "pressure_rate", "quick_pressure_rate", "mof_pct",
            "mof_epa_allowed", "pass_oe", "epa_db", "epa_rush", "rush_sr_allowed"
        ]
        for col in num_cols_team:
            if col in df_new_teams.columns:
                df_new_teams[col] = pd.to_numeric(df_new_teams[col], errors="coerce").fillna(0.0)

        if "matchup_takeaway" not in df_new_teams.columns:
            df_new_teams["matchup_takeaway"] = "Realized weekly scheme review"
        if "week_takeaway" not in df_new_teams.columns:
            df_new_teams["week_takeaway"] = f"Week {target_week} Review"

        existing_teams = load_scheme_reviews()
        if not existing_teams.empty:
            retained_teams = existing_teams[existing_teams["week"] != target_week]
            combined_teams = pd.concat([retained_teams, df_new_teams], ignore_index=True)
        else:
            combined_teams = df_new_teams

        combined_teams = combined_teams.sort_values(by=["week", "team"]).reset_index(drop=True)
        _save_df_parquet(combined_teams, REVIEWS_PARQUET, REVIEWS_CSV)
        _save_df_parquet(combined_teams, SCHEME_PARQUET, SCHEME_CSV)
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
        _save_df_parquet(combined_players, PLAYER_PARQUET, PLAYER_CSV)
        results["player_rows"] = len(df_new_players)

    rebuild_team_narratives()
    return results


def get_team_expectation_delta(team: str, week: Optional[int] = None) -> Dict[str, Any]:
    """
    Compare pre-game preview expectations (`scheme_previews.parquet`) against post-game
    review realities (`scheme_reviews.parquet`) for an NFL team.

    Rules & Simulation Regime Modifiers:
      1. If a team repeatedly fails to execute previewed pass concepts due to quick pressure
         (`realized_qpr - expected_qpr >= 0.025` or `realized_qpr >= 0.22`), accelerate
         Bayesian prior decay (`prior_decay_multiplier = 0.65`) and flag `PASS_PROTECTION_COLLAPSE`.
      2. If a defense repeatedly bleeds higher MOF EPA than previewed
         (`realized_mof_epa - expected_mof_epa >= 0.05` or `realized_mof_epa >= 0.44`),
         elevate opposing Slot WR / TE floor (`slot_te_floor_boost = 1.08`) and ceiling
         modifiers (`slot_te_mu_boost = 1.18`, `slot_te_ceiling_boost = 1.12`) and flag `SLOT_FUNNEL_HIGH`.
      3. If rushing efficiency collapses (`epa_rush <= -0.10` or `rush_sr_allowed <= 0.365`),
         flag `RUN_GAME_BROKEN`.
      4. If pass rate over expected spikes (`pass_oe >= 4.0`), flag `GARBAGE_TIME_ALPHA`.
    """
    t_clean = (team or "").upper().strip()
    df_rev = load_scheme_reviews()
    df_prev = load_scheme_previews()

    team_rev = df_rev[df_rev["team"] == t_clean].sort_values(by="week") if not df_rev.empty and "team" in df_rev.columns else pd.DataFrame()
    team_prev = df_prev[df_prev["team"] == t_clean].sort_values(by="week") if not df_prev.empty and "team" in df_prev.columns else pd.DataFrame()

    if week is not None:
        team_rev_wk = team_rev[team_rev["week"] <= int(week)] if not team_rev.empty else pd.DataFrame()
        team_prev_wk = team_prev[team_prev["week"] <= int(week)] if not team_prev.empty else pd.DataFrame()
        if team_rev_wk.empty:
            team_rev_wk = team_rev
        if team_prev_wk.empty:
            team_prev_wk = team_prev
    else:
        team_rev_wk = team_rev
        team_prev_wk = team_prev

    if team_rev_wk.empty:
        prior = DEFAULT_DEFENSIVE_SCHEMES.get(t_clean, {"quick_pressure_rate": 0.185, "mof_epa_allowed": 0.38, "rush_sr_allowed": 0.42})
        tags = ["PASS_PROTECTION_COLLAPSE"] if float(prior.get("quick_pressure_rate", 0.0)) >= 0.23 else []
        if float(prior.get("mof_epa_allowed", 0.0)) >= 0.50:
            tags.append("SLOT_FUNNEL_HIGH")
        return {
            "team": t_clean,
            "week": week or 1,
            "expected_qpr": prior["quick_pressure_rate"],
            "realized_qpr": prior["quick_pressure_rate"],
            "qpr_delta": 0.0,
            "expected_mof_epa": prior["mof_epa_allowed"],
            "realized_mof_epa": prior["mof_epa_allowed"],
            "mof_epa_delta": 0.0,
            "realized_pass_oe": 0.0,
            "realized_epa_rush": -0.05,
            "regime_tags": tags,
            "prior_decay_multiplier": 0.65 if "PASS_PROTECTION_COLLAPSE" in tags else 1.0,
            "slot_te_mu_boost": 1.18 if "SLOT_FUNNEL_HIGH" in tags else 1.0,
            "slot_te_floor_boost": 1.08 if "SLOT_FUNNEL_HIGH" in tags else 1.0,
            "slot_te_ceiling_boost": 1.12 if "SLOT_FUNNEL_HIGH" in tags else 1.0,
            "verdict": "Baseline Prior"
        }

    # Evaluate across recent weeks (up to 3-week window) and target week
    recent_rev = team_rev_wk.tail(3) if len(team_rev_wk) >= 2 else team_rev.tail(3)
    realized_qpr = float(recent_rev["quick_pressure_rate"].mean())
    realized_pr = float(recent_rev["pressure_rate"].mean()) if "pressure_rate" in recent_rev.columns else realized_qpr * 1.6
    realized_mof = float(recent_rev["mof_epa_allowed"].mean())
    realized_poe = float(recent_rev["pass_oe"].mean()) if "pass_oe" in recent_rev.columns else 0.0
    realized_epa_rush = float(recent_rev["epa_rush"].mean()) if "epa_rush" in recent_rev.columns else -0.05
    realized_rsr = float(recent_rev["rush_sr_allowed"].mean()) if "rush_sr_allowed" in recent_rev.columns else 0.42
    prior_scheme = DEFAULT_DEFENSIVE_SCHEMES.get(t_clean, {"quick_pressure_rate": 0.185, "mof_epa_allowed": 0.36})

    if not team_prev_wk.empty:
        recent_prev = team_prev_wk.tail(3) if len(team_prev_wk) >= 2 else team_prev.tail(3)
        expected_qpr = float(recent_prev["expected_quick_pressure_rate"].mean()) if "expected_quick_pressure_rate" in recent_prev.columns else realized_qpr * 0.90
        expected_mof = float(recent_prev["expected_mof_epa"].mean()) if "expected_mof_epa" in recent_prev.columns else realized_mof * 0.90
    else:
        expected_qpr = float(prior_scheme["quick_pressure_rate"]) * 0.88
        expected_mof = float(prior_scheme["mof_epa_allowed"]) * 0.88

    qpr_delta = round(realized_qpr - expected_qpr, 3)
    mof_delta = round(realized_mof - expected_mof, 3)

    regime_tags: List[str] = []
    prior_decay_multiplier = 1.0
    slot_te_mu_boost = 1.0
    slot_te_floor_boost = 1.0
    slot_te_ceiling_boost = 1.0
    verdict_parts: List[str] = []

    # 1. PASS_PROTECTION_COLLAPSE check
    if (qpr_delta >= 0.025 or realized_qpr >= 0.18 or realized_pr >= 0.30) and float(prior_scheme.get("quick_pressure_rate", 0.0)) >= 0.22:
        regime_tags.append("PASS_PROTECTION_COLLAPSE")
        prior_decay_multiplier = 0.65  # Accelerate Bayesian prior decay by 35%
        verdict_parts.append(
            f"Quick/overall pressure spiked (QP {realized_qpr*100:.1f}%, PR {realized_pr*100:.1f}% realized vs {expected_qpr*100:.1f}% previewed) -> Accelerated Bayesian prior decay (0.65x)"
        )

    # 2. SLOT_FUNNEL_HIGH check
    if mof_delta >= 0.05 or realized_mof >= 0.44 or float(prior_scheme.get("mof_epa_allowed", 0.0)) >= 0.50:
        regime_tags.append("SLOT_FUNNEL_HIGH")
        slot_te_mu_boost = 1.18
        slot_te_floor_boost = 1.08
        slot_te_ceiling_boost = 1.12
        verdict_parts.append(
            f"Defense bled elevated MOF EPA ({realized_mof:+.2f} realized vs {expected_mof:+.2f} previewed) -> Opposing Slot/TE floor & ceiling elevated"
        )

    # 3. RUN_GAME_BROKEN check
    if realized_epa_rush <= -0.10 or realized_rsr <= 0.365:
        regime_tags.append("RUN_GAME_BROKEN")
        verdict_parts.append(f"Inefficient ground game (Rush EPA {realized_epa_rush:+.2f}, Rush SR {realized_rsr*100:.1f}%)")

    # 4. GARBAGE_TIME_ALPHA check
    if realized_poe >= 4.0:
        regime_tags.append("GARBAGE_TIME_ALPHA")
        verdict_parts.append(f"Elevated Pass Over Expected ({realized_poe:+.1f}% POE) driving late-game passing volume")

    verdict = " | ".join(verdict_parts) if verdict_parts else "Execution aligned with pre-game preview expectations"

    return {
        "team": t_clean,
        "week": int(recent_rev.iloc[-1]["week"]),
        "expected_qpr": round(expected_qpr, 3),
        "realized_qpr": round(realized_qpr, 3),
        "qpr_delta": qpr_delta,
        "expected_mof_epa": round(expected_mof, 2),
        "realized_mof_epa": round(realized_mof, 2),
        "mof_epa_delta": mof_delta,
        "realized_pass_oe": round(realized_poe, 1),
        "realized_epa_rush": round(realized_epa_rush, 2),
        "regime_tags": regime_tags,
        "prior_decay_multiplier": prior_decay_multiplier,
        "slot_te_mu_boost": slot_te_mu_boost,
        "slot_te_floor_boost": slot_te_floor_boost,
        "slot_te_ceiling_boost": slot_te_ceiling_boost,
        "verdict": verdict,
    }


def rebuild_team_narratives() -> Dict[str, Any]:
    """
    Build `data/team_narratives.json`: multi-week longitudinal ledger mapping each team's
    trajectory from Week 1 to present:
      1. Preseason expectation
      2. Weekly preview hypotheses vs. weekly review execution
      3. Structural regime tags ('PASS_PROTECTION_COLLAPSE', 'RUN_GAME_BROKEN', 'SLOT_FUNNEL_HIGH', 'GARBAGE_TIME_ALPHA')
      4. Rolling 3-week averages & player usage trajectories.
    """
    _ensure_data_dir()
    df_rev = _load_df_parquet(REVIEWS_PARQUET, REVIEWS_CSV)
    if df_rev.empty:
        df_rev = _load_df_parquet(SCHEME_PARQUET, SCHEME_CSV)
    df_prev = _load_df_parquet(PREVIEWS_PARQUET, PREVIEWS_CSV)

    all_teams = sorted(list(set(
        list(DEFAULT_DEFENSIVE_SCHEMES.keys())
        + (df_rev["team"].unique().tolist() if not df_rev.empty and "team" in df_rev.columns else [])
        + (df_prev["team"].unique().tolist() if not df_prev.empty and "team" in df_prev.columns else [])
    )))

    narratives: Dict[str, Any] = {}
    for team in all_teams:
        t_rev = df_rev[df_rev["team"] == team].sort_values(by="week") if not df_rev.empty and "team" in df_rev.columns else pd.DataFrame()
        t_prev = df_prev[df_prev["team"] == team].sort_values(by="week") if not df_prev.empty and "team" in df_prev.columns else pd.DataFrame()

        weeks = sorted(list(set(
            (t_rev["week"].astype(int).tolist() if not t_rev.empty else [])
            + (t_prev["week"].astype(int).tolist() if not t_prev.empty else [])
        )))

        weekly_ledger: List[Dict[str, Any]] = []
        for w in weeks:
            r_row = t_rev[t_rev["week"] == w].iloc[-1] if not t_rev.empty and (t_rev["week"] == w).any() else None
            p_row = t_prev[t_prev["week"] == w].iloc[-1] if not t_prev.empty and (t_prev["week"] == w).any() else None

            opp = str(r_row["opponent"] if r_row is not None else (p_row["opponent"] if p_row is not None else "OPP"))
            exp_qpr = float(p_row["expected_quick_pressure_rate"]) if p_row is not None and "expected_quick_pressure_rate" in p_row else (float(r_row["quick_pressure_rate"]) * 0.88 if r_row is not None else 0.185)
            real_qpr = float(r_row["quick_pressure_rate"]) if r_row is not None else exp_qpr
            exp_mof = float(p_row["expected_mof_epa"]) if p_row is not None and "expected_mof_epa" in p_row else (float(r_row["mof_epa_allowed"]) * 0.88 if r_row is not None else 0.36)
            real_mof = float(r_row["mof_epa_allowed"]) if r_row is not None else exp_mof
            pass_oe = float(r_row["pass_oe"]) if r_row is not None and "pass_oe" in r_row else 0.0
            pr_rate = float(r_row["pressure_rate"]) if r_row is not None and "pressure_rate" in r_row else real_qpr * 1.6

            prev_hyp = (
                f"{p_row.get('funnel_focus', 'Balanced')} Funnel ({p_row.get('man_coverage_rate', 0.28)*100:.0f}% Man / {p_row.get('zone_coverage_rate', 0.72)*100:.0f}% Zone) — {p_row.get('tactical_takeaways', '')}"
                if p_row is not None else f"Standard baseline shell vs {opp}"
            )
            rev_exec = (
                str(r_row.get("matchup_takeaway", "")) or f"Realized QP {real_qpr*100:.1f}%, MOF EPA {real_mof:+.2f}, Pass OE {pass_oe:+.1f}%"
                if r_row is not None else "Pending game review"
            )

            qpr_d = round(real_qpr - exp_qpr, 3)
            mof_d = round(real_mof - exp_mof, 3)
            weekly_ledger.append({
                "week": int(w),
                "opponent": opp,
                "preview_hypothesis": prev_hyp,
                "review_execution": rev_exec,
                "expected_qpr": round(exp_qpr, 3),
                "realized_qpr": round(real_qpr, 3),
                "qpr_delta": qpr_d,
                "pressure_rate": round(pr_rate, 3),
                "expected_mof_epa": round(exp_mof, 2),
                "realized_mof_epa": round(real_mof, 2),
                "mof_epa_delta": mof_d,
                "pass_oe": round(pass_oe, 1),
            })

        delta_info = get_team_expectation_delta(team)
        rolling_info = get_team_defense_rolling(team, window=3)

        narratives[team] = {
            "team": team,
            "preseason_expectation": PRESEASON_TEAM_EXPECTATIONS.get(
                team, f"Balanced NFL Scheme Baseline (Prior QP {DEFAULT_DEFENSIVE_SCHEMES.get(team, {}).get('quick_pressure_rate', 0.185)*100:.1f}%)"
            ),
            "regime_tags": delta_info["regime_tags"],
            "expectation_delta": delta_info,
            "rolling_3wk": rolling_info,
            "weekly_ledger": weekly_ledger,
        }

    try:
        with open(TEAM_NARRATIVES_JSON, "w", encoding="utf-8") as f:
            json.dump(narratives, f, indent=2)
    except Exception:
        pass

    return narratives


def load_team_narratives() -> Dict[str, Any]:
    """Load `data/team_narratives.json` (or rebuild if missing)."""
    _ensure_data_dir()
    if os.path.exists(TEAM_NARRATIVES_JSON):
        try:
            with open(TEAM_NARRATIVES_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data:
                    return data
        except Exception:
            pass
    return rebuild_team_narratives()


def get_team_longitudinal_deep_dive(team: str) -> Dict[str, Any]:
    """
    Return the complete longitudinal profile for `team` (Week 1 to present) for the
    Streamlit Scheme Lab Deep Dive view:
      - Preseason expectation & Active Regime Tags
      - Weekly Preview vs Review timeline DataFrame
      - Team weekly trend DataFrame (Pressure Rate, Quick Pressure, MOF EPA, Pass OE)
      - Primary WR/TE/RB Target Share trajectory DataFrame
    """
    t_clean = (team or "CLE").upper().strip()
    narratives = load_team_narratives()
    team_nar = narratives.get(t_clean) or rebuild_team_narratives().get(t_clean, {})

    df_rev = load_scheme_reviews()
    df_players = load_player_history()

    t_rev = df_rev[df_rev["team"] == t_clean].sort_values(by="week") if not df_rev.empty and "team" in df_rev.columns else pd.DataFrame()
    t_players = df_players[df_players["team"] == t_clean].sort_values(by=["week", "target_share"], ascending=[True, False]) if not df_players.empty and "team" in df_players.columns else pd.DataFrame()

    ledger_rows = team_nar.get("weekly_ledger", [])
    df_timeline = pd.DataFrame(ledger_rows) if ledger_rows else pd.DataFrame()

    return {
        "team": t_clean,
        "preseason_expectation": team_nar.get("preseason_expectation", ""),
        "regime_tags": team_nar.get("regime_tags", []),
        "expectation_delta": team_nar.get("expectation_delta", get_team_expectation_delta(t_clean)),
        "rolling_3wk": team_nar.get("rolling_3wk", get_team_defense_rolling(t_clean, window=3)),
        "timeline_df": df_timeline,
        "team_metrics_df": t_rev,
        "player_usage_df": t_players,
    }


def batch_ingest_newsletters(
    folder_path: str = "newsletters",
    auto_seed_if_empty: bool = True
) -> Dict[str, Any]:
    """
    Scan `newsletters/` for PDF files matching `week*_preview.pdf`, `week*_review.pdf`, or any `.pdf`,
    sort them in chronological order (Week 1 Preview -> Week 1 Review -> Week 2 Preview -> ...),
    parse each file, idempotently upsert into `scheme_previews.parquet` and `scheme_reviews.parquet`,
    and rebuild `team_narratives.json`.
    """
    import pdf_parser

    target_dir = folder_path if os.path.isabs(folder_path) else os.path.join(BASE_DIR, folder_path)
    os.makedirs(target_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(target_dir) if f.lower().endswith(".pdf")]
    if not pdf_files and auto_seed_if_empty:
        pdf_parser.seed_sample_newsletters_folder(target_dir, weeks=[1, 2, 3])
        pdf_files = [f for f in os.listdir(target_dir) if f.lower().endswith(".pdf")]

    def _sort_key(filename: str):
        fn_lower = filename.lower()
        wk_m = re.search(r"week[_\-\s]*(\d+)", fn_lower)
        wk_num = int(wk_m.group(1)) if wk_m else 99
        # Process Preview (0) before Review (1) within the same week
        type_order = 0 if "preview" in fn_lower else 1
        return (wk_num, type_order, fn_lower)

    sorted_files = sorted(pdf_files, key=_sort_key)
    processed_records: List[Dict[str, Any]] = []
    previews_count = 0
    reviews_count = 0

    for fname in sorted_files:
        full_path = os.path.join(target_dir, fname)
        det = pdf_parser.detect_newsletter_type(full_path)
        wk = det["week"]
        dtype = det["doc_type"]

        if det["short_type"] == "PREVIEW":
            parsed = pdf_parser.parse_preview_pdf(full_path, week_num=wk)
            if parsed.get("success"):
                prev_df = parsed.get("preview_metrics", pd.DataFrame())
                ang_df = parsed.get("player_angles", pd.DataFrame())
                save_res = save_preview_metrics(prev_df, ang_df, week=wk)
                previews_count += 1
                processed_records.append({
                    "Week": wk,
                    "Document Type": "Weekly Preview",
                    "Filename": fname,
                    "Teams Extracted": save_res["preview_rows"],
                    "Players / Angles": save_res["angle_rows"],
                    "Status": "✅ Upserted to scheme_previews.parquet"
                })
            else:
                processed_records.append({
                    "Week": wk,
                    "Document Type": dtype,
                    "Filename": fname,
                    "Teams Extracted": 0,
                    "Players / Angles": 0,
                    "Status": f"❌ Error: {parsed.get('error', 'Unknown')}"
                })
        else:
            parsed = pdf_parser.parse_review_pdf(full_path, week_num=wk)
            if parsed.get("success"):
                t_df = parsed.get("team_metrics", pd.DataFrame())
                p_df = parsed.get("player_metrics", pd.DataFrame())
                save_res = save_newsletter_metrics(t_df, p_df, week=wk)
                reviews_count += 1
                processed_records.append({
                    "Week": wk,
                    "Document Type": "Weekly Review",
                    "Filename": fname,
                    "Teams Extracted": save_res["team_rows"],
                    "Players / Angles": save_res["player_rows"],
                    "Status": "✅ Upserted to scheme_reviews.parquet"
                })
            else:
                processed_records.append({
                    "Week": wk,
                    "Document Type": dtype,
                    "Filename": fname,
                    "Teams Extracted": 0,
                    "Players / Angles": 0,
                    "Status": f"❌ Error: {parsed.get('error', 'Unknown')}"
                })

    narratives = rebuild_team_narratives()
    summary_df = pd.DataFrame(processed_records)

    return {
        "success": True,
        "folder_path": target_dir,
        "files_processed": len(processed_records),
        "previews_upserted": previews_count,
        "reviews_upserted": reviews_count,
        "narratives_count": len(narratives),
        "summary_df": summary_df,
        "records": processed_records,
    }


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

    df = load_scheme_reviews()
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

    if t_clean in DEFAULT_DEFENSIVE_SCHEMES:
        final_qpr = max(final_qpr, prior["quick_pressure_rate"] * 0.90)
        final_mof = max(final_mof, prior["mof_epa_allowed"] * 0.86)
        final_rsr = max(final_rsr, prior["rush_sr_allowed"] * 0.97)

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
    df = load_scheme_reviews()
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
    """Retrieve historical weekly usage metrics for a specific player."""
    p_clean = (player_name or "").strip().lower()
    df = load_player_history()
    if df.empty or "player_name" not in df.columns:
        return pd.DataFrame(columns=["week", "player_name", "team", "position", "target_share", "yprr", "adot", "rush_epa", "rush_sr"])

    mask = df["player_name"].str.lower().str.contains(p_clean, regex=False)
    matched = df[mask].sort_values(by="week", ascending=True).reset_index(drop=True)
    return matched


def get_all_teams_in_db() -> List[str]:
    """Return sorted list of all unique team codes in the scheme history."""
    df = load_scheme_reviews()
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
    Seed initial 3 weeks of scheme reviews, scheme previews, player usage metrics,
    and longitudinal team narratives if database is empty or force=True.
    """
    _ensure_data_dir()
    if not force and os.path.exists(REVIEWS_PARQUET) and os.path.exists(PREVIEWS_PARQUET) and os.path.exists(SCHEME_PARQUET):
        return

    team_rows = []
    preview_rows = []
    for week in [1, 2, 3]:
        for team, priors in DEFAULT_DEFENSIVE_SCHEMES.items():
            noise_qp = ((hash(f"{team}_{week}_qp") % 21) - 10) / 1000.0
            noise_mof = ((hash(f"{team}_{week}_mof") % 21) - 10) / 200.0
            noise_rsr = ((hash(f"{team}_{week}_rsr") % 21) - 10) / 1000.0

            qp = max(0.10, min(0.35, priors["quick_pressure_rate"] + noise_qp))
            mof = max(0.15, min(0.70, priors["mof_epa_allowed"] + noise_mof))
            rsr = max(0.30, min(0.55, priors["rush_sr_allowed"] + noise_rsr))

            exp_qp = max(0.10, round(qp - 0.028 if team in ["CLE", "PHI", "DET"] else qp * 0.94, 3))
            exp_mof = max(0.15, round(mof - 0.06 if team in ["CAR", "TB", "ARI", "JAX"] else mof * 0.94, 2))
            man_rt = 0.36 if qp > 0.20 else 0.25
            zone_rt = round(1.0 - man_rt, 3)
            funnel_focus = "Inside/Slot" if mof >= 0.42 else ("Boundary" if mof <= 0.29 else "Balanced")

            preview_rows.append({
                "week": week,
                "team": team,
                "opponent": "OPP",
                "man_coverage_rate": round(man_rt, 3),
                "zone_coverage_rate": zone_rt,
                "blitz_rate": round(0.24 + (qp * 0.25), 3),
                "expected_quick_pressure_rate": exp_qp,
                "expected_mof_epa": exp_mof,
                "explosive_play_rate_allowed": round(0.10 + (mof * 0.05), 3),
                "funnel_focus": funnel_focus,
                "tactical_takeaways": f"Week {week} Preview: {funnel_focus} funnel hypothesis ({man_rt*100:.0f}% Man / {zone_rt*100:.0f}% Zone)"
            })

            team_rows.append({
                "week": week,
                "team": team,
                "opponent": "OPP",
                "pressure_rate": round(qp * 1.62, 3),
                "quick_pressure_rate": round(qp, 3),
                "mof_pct": round(0.38 + (week * 0.01), 3),
                "mof_epa_allowed": round(mof, 2),
                "pass_oe": round(4.2 if team in ["CLE", "CAR", "BUF"] else (1.2 if qp > 0.20 else -1.5), 1),
                "epa_db": round(0.04, 2),
                "epa_rush": round(-0.12 if team in ["CLE", "CAR", "LV"] else -0.06, 2),
                "rush_sr_allowed": round(rsr, 3),
                "matchup_takeaway": f"Week {week} Review: Realized QP {qp*100:.1f}% vs {exp_qp*100:.1f}% exp; MOF EPA {mof:+.2f}",
                "week_takeaway": f"Week {week} Realized Scheme Review"
            })

    df_teams = pd.DataFrame(team_rows)
    _save_df_parquet(df_teams, REVIEWS_PARQUET, REVIEWS_CSV)
    _save_df_parquet(df_teams, SCHEME_PARQUET, SCHEME_CSV)

    df_previews = pd.DataFrame(preview_rows)
    _save_df_parquet(df_previews, PREVIEWS_PARQUET, PREVIEWS_CSV)

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
    _save_df_parquet(df_players, PLAYER_PARQUET, PLAYER_CSV)
    rebuild_team_narratives()


def get_defensive_funnel_profile(team: str) -> Dict[str, Any]:
    """
    Analyze an NFL defense to classify its target funnel tendency.
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
    Scan historical weekly player trajectories to identify Floor Surging and Buy-Low trade targets.
    """
    df_players = load_player_history()
    if df_players.empty or "player_name" not in df_players.columns:
        return pd.DataFrame()

    results = []
    for p_name, group in df_players.groupby("player_name"):
        sorted_g = group.sort_values(by="week", ascending=True)
        if len(sorted_g) < 2:
            continue

        latest = sorted_g.iloc[-1]
        earliest = sorted_g.iloc[0]

        latest_tgt = latest["target_share"]
        latest_yprr = latest["yprr"]
        latest_adot = latest["adot"]
        delta_tgt = round(latest_tgt - earliest["target_share"], 3)
        delta_yprr = round(latest_yprr - earliest["yprr"], 2)

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


if not os.path.exists(REVIEWS_PARQUET) or not os.path.exists(PREVIEWS_PARQUET):
    init_default_scheme_history()
