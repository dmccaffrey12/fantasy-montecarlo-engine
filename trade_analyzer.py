"""
trade_analyzer.py
Idiosyncratic Fantasy Football Trade Evaluation & Bilateral Lineup Simulation Engine.

Ported and upgraded from fantasy-football-arbitrage/trade_analyzer.py specifically for
the user's idiosyncratic ESPN league scoring system (scoring_config.py) and 9-starter
2-QB / OP Superflex roster structure:
  - Passing: 6.0 pt TD, +0.1 pt per completion, 0.04 pt/yd, -1.0 INT,
             stacking +1/+4 pt bonuses for 40+/50+ yd Pass TDs, +2/+3 pt bonuses for 300+/400+ yds.
  - Rushing: 0.10 pt/yd, 6.0 pt TD, stacking +1/+2 pt bonuses for 100+/200+ rush yds,
             stacking +1/+2 pt bonuses for 40+/50+ yd Rush TDs.
  - Receiving: 1.0 Full PPR, 0.10 pt/yd, 6.0 pt TD, stacking +1/+2 pt bonuses for 100+/200+ rec yds,
               stacking +1/+2 pt bonuses for 40+/50+ yd Rec TDs.
  - Roster Structure: 1 QB, 1 OP (Superflex), 2 RB, 2 WR, 1 WR/TE, 1 FLEX (RB/WR/TE), 1 D/ST (Max 2 QBs).
"""

from typing import Dict, List, Any, Optional, Tuple
import copy
import numpy as np
import pandas as pd

import scoring_config
import simulation
import vorp
import data_ingest
import scheme_db


# Positional scarcity multipliers calibrated to 2-QB / Full-PPR / 3-WR+Flex+OP league
IDIOSYNCRATIC_SCARCITY_MULTIPLIERS: Dict[str, float] = {
    "QB": 1.22,   # 2-QB (QB + OP) with 6pt Pass TD + 0.1 completion bonus
    "RB": 1.18,   # Bellcow RBs with 100/200-yd Gamma milestone & 1.0 PPR receiving floor
    "WR": 1.10,   # Deep 3-starting-WR slots (WR1, WR2, WR/TE + FLEX)
    "TE": 0.96,   # No mandatory dedicated TE slot (WR/TE hybrid slot)
    "D/ST": 0.70,
    "DEF": 0.70,
    "K": 0.50,
}

# Exact NFL Bye Weeks (Weeks 6-14) and Fantasy Playoff Schedules (Weeks 15, 16, 17) synced from ESPN League 385430
NFL_BYES_AND_PLAYOFF_SCHEDULE: Dict[str, Dict[str, Any]] = {
    "ARI": {"bye": 14, "playoffs": {15: "NYJ", 16: "NO", 17: "LV"}},
    "ATL": {"bye": 11, "playoffs": {15: "WSH", 16: "TB", 17: "NO"}},
    "BAL": {"bye": 13, "playoffs": {15: "PIT", 16: "CLE", 17: "CIN"}},
    "BUF": {"bye": 7,  "playoffs": {15: "CHI", 16: "DEN", 17: "MIA"}},
    "CAR": {"bye": 5,  "playoffs": {15: "CIN", 16: "PIT", 17: "SEA"}},  # Bye already completed!
    "CHI": {"bye": 10, "playoffs": {15: "BUF", 16: "GB", 17: "DET"}},
    "CIN": {"bye": 6,  "playoffs": {15: "CAR", 16: "IND", 17: "BAL"}},
    "CLE": {"bye": 11, "playoffs": {15: "NYG", 16: "BAL", 17: "IND"}},
    "DAL": {"bye": 14, "playoffs": {15: "LAR", 16: "JAX", 17: "NYG"}},
    "DEN": {"bye": 10, "playoffs": {15: "LV", 16: "BUF", 17: "NE"}},
    "DET": {"bye": 6,  "playoffs": {15: "MIN", 16: "NYG", 17: "CHI"}},
    "GB":  {"bye": 11, "playoffs": {15: "MIA", 16: "CHI", 17: "HOU"}},
    "HOU": {"bye": 14, "playoffs": {15: "JAX", 16: "PHI", 17: "GB"}},
    "IND": {"bye": 13, "playoffs": {15: "TEN", 16: "CIN", 17: "CLE"}},
    "JAX": {"bye": 12, "playoffs": {15: "HOU", 16: "DAL", 17: "TEN"}},
    "KC":  {"bye": 12, "playoffs": {15: "LAC", 16: "LV", 17: "LAC"}},
    "LAC": {"bye": 7,  "playoffs": {15: "SF", 16: "MIA", 17: "KC"}},
    "LAR": {"bye": 9,  "playoffs": {15: "DAL", 16: "SEA", 17: "TB"}},
    "LV":  {"bye": 8,  "playoffs": {15: "DEN", 16: "TEN", 17: "ARI"}},
    "MIA": {"bye": 6,  "playoffs": {15: "GB", 16: "LAC", 17: "BUF"}},
    "MIN": {"bye": 6,  "playoffs": {15: "DET", 16: "WSH", 17: "NYJ"}},
    "NE":  {"bye": 14, "playoffs": {15: "BUF", 16: "NYJ", 17: "DEN"}},
    "NO":  {"bye": 8,  "playoffs": {15: "TB", 16: "ARI", 17: "ATL"}},
    "NYG": {"bye": 11, "playoffs": {15: "CLE", 16: "DET", 17: "DAL"}},
    "NYJ": {"bye": 13, "playoffs": {15: "ARI", 16: "NE", 17: "MIN"}},
    "PHI": {"bye": 10, "playoffs": {15: "SEA", 16: "HOU", 17: "SF"}},
    "PIT": {"bye": 9,  "playoffs": {15: "BAL", 16: "CAR", 17: "TEN"}},
    "SEA": {"bye": 11, "playoffs": {15: "PHI", 16: "LAR", 17: "CAR"}},
    "SF":  {"bye": 9,  "playoffs": {15: "LAC", 16: "CHI", 17: "PHI"}},
    "TB":  {"bye": 10, "playoffs": {15: "NO", 16: "ATL", 17: "LAR"}},
    "TEN": {"bye": 9,  "playoffs": {15: "IND", 16: "LV", 17: "PIT"}},
    "WSH": {"bye": 12, "playoffs": {15: "ATL", 16: "MIN", 17: "DAL"}},
}

_FULL_SEASON_BOARD_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def load_full_season_projections_board() -> Dict[str, Dict[str, Any]]:
    """
    Parse and cache the user's Full-Season Projections & True VORP / Auction Value board
    (`2026-FFB-Projections-0812.xlsx`, ported from `fantasy-football-arbitrage`), containing:
      - `fs_pos_rk` (e.g. 'RB4', 'RB5', 'WR25', 'WR40')
      - `fs_ovr_rk` (Overall VORP Rank 1..300)
      - `fs_fps` (Full-Season Projected Points under the user's custom 6pt Pass TD / 0.1 comp / 1.0 PPR rules)
      - `fs_ppg` (`fs_fps / 17.0`)
      - `fs_vorp` (Full-Season True VORP against 12-team 9-starter replacement baselines: QB26, RB44, WR73, TE21)
      - `fs_auc` (`$200` Auction Dollar valuation `AUC$`)
    """
    global _FULL_SEASON_BOARD_CACHE
    if _FULL_SEASON_BOARD_CACHE is not None:
        return _FULL_SEASON_BOARD_CACHE

    import os
    board: Dict[str, Dict[str, Any]] = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "2026-FFB-Projections-0812.xlsx"),
        os.path.join(os.path.dirname(base_dir), "fantasy-football-arbitrage", "2026-FFB-Projections-0812.xlsx"),
    ]
    xl_path = next((p for p in candidates if os.path.exists(p)), None)

    def _reg(name_raw: str, entry: Dict[str, Any]) -> None:
        norm = data_ingest.normalize_player_name(name_raw)
        if not norm:
            return
        board[norm] = entry
        for sfx in [" jr", " sr", " iii", " ii", " iv"]:
            if norm.endswith(sfx):
                board[norm[:-len(sfx)].strip()] = entry

    if xl_path:
        try:
            df_pos = pd.read_excel(xl_path, sheet_name="POS Ranks")
            df_ovr = pd.read_excel(xl_path, sheet_name="OVR & VORP Ranks")

            # 1. Extract Auction $ and FPS from POS Ranks (QB: cols 0..5, RB: 7..12, WR: 14..19, TE: 21..26, DST: 28..32)
            pos_blocks = [
                ("QB", 0, 1, 2, 3, 4, 5),
                ("RB", 7, 8, 9, 10, 11, 12),
                ("WR", 14, 15, 16, 17, 18, 19),
                ("TE", 21, 22, 23, 24, 25, 26),
            ]
            for pos_label, c_rk, c_name, c_tm, c_bye, c_fps, c_auc in pos_blocks:
                for _, row in df_pos.iterrows():
                    p_name = row.iloc[c_name] if df_pos.shape[1] > c_name else None
                    if pd.isna(p_name) or not str(p_name).strip():
                        continue
                    try:
                        rk_int = int(float(row.iloc[c_rk]))
                        fps_val = float(row.iloc[c_fps] or 0.0)
                        auc_val = max(0.0, float(row.iloc[c_auc] or 0.0))
                    except Exception:
                        continue
                    _reg(str(p_name).strip(), {
                        "name": str(p_name).strip(),
                        "fs_pos_rk": f"{pos_label}{rk_int}",
                        "fs_ovr_rk": 999,
                        "fs_fps": round(fps_val, 2),
                        "fs_ppg": round(fps_val / 17.0, 2),
                        "fs_vorp": 0.0,
                        "fs_auc": round(auc_val, 2),
                    })

            # DST block in POS Ranks (cols 28..32: RK, Player, BYE, FPS, AUC$)
            if df_pos.shape[1] > 32:
                for _, row in df_pos.iterrows():
                    d_name = row.iloc[29]
                    if pd.isna(d_name) or not str(d_name).strip():
                        continue
                    try:
                        rk_int = int(float(row.iloc[28]))
                        fps_val = float(row.iloc[31] or 0.0)
                        auc_val = max(0.0, float(row.iloc[32] or 0.0))
                    except Exception:
                        continue
                    d_clean = str(d_name).strip()
                    nickname = d_clean.split()[-1]
                    entry_d = {
                        "name": d_clean,
                        "fs_pos_rk": f"DST{rk_int}",
                        "fs_ovr_rk": 200 + rk_int,
                        "fs_fps": round(fps_val, 2),
                        "fs_ppg": round(fps_val / 17.0, 2),
                        "fs_vorp": 0.0,
                        "fs_auc": round(auc_val, 2),
                    }
                    _reg(d_clean, entry_d)
                    _reg(f"{nickname} D/ST", entry_d)
                    _reg(f"{nickname} DST", entry_d)

            # 2. Enrich with Overall VORP Rank & True VORP from OVR & VORP Ranks (cols RK.5, OVERALL PLAYER.1, POS RK.1, VORP.5)
            # Note: Preserve `fs_fps` and `fs_ppg` from POS Ranks (which holds each player's true 17-game positional FPS).
            if "OVERALL PLAYER.1" in df_ovr.columns:
                for _, row in df_ovr.iterrows():
                    p_name = row.get("OVERALL PLAYER.1")
                    if pd.isna(p_name) or not str(p_name).strip():
                        continue
                    try:
                        ovr_rk = int(float(row.get("RK.5", 999)))
                        pos_rk_str = str(row.get("POS RK.1", "")).strip()
                        vorp_val = float(row.get("VORP.5", 0.0) or 0.0)
                    except Exception:
                        continue
                    norm = data_ingest.normalize_player_name(str(p_name).strip())
                    existing = board.get(norm, {})
                    true_fps = float(existing.get("fs_fps", 0.0) or 0.0)
                    merged = {
                        "name": str(p_name).strip(),
                        "fs_pos_rk": pos_rk_str or existing.get("fs_pos_rk", "—"),
                        "fs_ovr_rk": ovr_rk,
                        "fs_fps": round(true_fps, 2),
                        "fs_ppg": round(true_fps / 17.0, 2),
                        "fs_vorp": round(vorp_val, 2),
                        "fs_auc": existing.get("fs_auc", 0.0),
                    }
                    _reg(str(p_name).strip(), merged)
        except Exception:
            pass

    _FULL_SEASON_BOARD_CACHE = board
    return board


def get_full_season_player_entry(player_name: str) -> Dict[str, Any]:
    """Lookup a player's Full-Season Projections & True VORP / Auction $ entry."""
    board = load_full_season_projections_board()
    norm = data_ingest.normalize_player_name(player_name)
    aliases = {
        "kenny gainwell": "kenneth gainwell",
        "josh palmer": "joshua palmer",
        "chig okonkwo": "chigoziem okonkwo",
    }
    norm = aliases.get(norm, norm)
    if norm in board:
        return board[norm].copy()
    for sfx in [" jr", " sr", " iii", " ii", " iv"]:
        if norm.endswith(sfx) and norm[:-len(sfx)].strip() in board:
            return board[norm[:-len(sfx)].strip()].copy()
    return {}


def compute_idiosyncratic_player_valuation(
    player: Dict[str, Any],
    baselines: Optional[Dict[str, Dict[str, Any]]] = None,
    scoring_cfg: Optional[scoring_config.LeagueScoringConfig] = None,
) -> Dict[str, Any]:
    """
    Enrich a player dictionary with full idiosyncratic scoring decomposition:
      - Standard / Athletic baseline projection (`std_proj`)
      - Idiosyncratic weekly mean (`wk_mu`), including non-linear milestone & long-TD EV
      - Rest-of-Season model mean (`ros_model_mu`)
      - Rest-of-Season empirical blended mean (`ros_blended_mu`)
      - True Season-Long / Championship baseline (`true_ros_mu`: 60% ESPN League ROS + 40% Realized Season PPG,
        uncontaminated by single-week Week 6 matchup outliers)
      - Full-Season Structural Model (`fs_fps`, `fs_ppg`, `fs_vorp`, `fs_auc`, `fs_pos_rk`, `fs_ovr_rk`)
        from `2026-FFB-Projections-0812.xlsx`
      - Bayesian 50/50 Structural + In-Season ROS (`bayes_ros_mu`: 50% Full-Season Structural PPG + 50% In-Season True ROS PPG)
      - Floor (P10), Median (P50), Ceiling (P90), Boom Probability P(>=20)
      - Weekly VORP (`wk_vorp`) and Rest-of-Season VORP (`ros_vorp`)
    """
    cfg = scoring_cfg or scoring_config.DEFAULT_SCORING
    p = copy.deepcopy(player)
    pos = str(p.get("position", "WR")).upper().strip()
    if pos == "DEF":
        pos = "D/ST"

    bk = p.get("score_breakdown", {}) or {}
    is_on_bye = bool(p.get("is_on_bye", False) or str(p.get("injury_status", "")).upper() == "BYE")
    is_inactive = bool(p.get("is_inactive", False) and not is_on_bye)

    # Extract non-linear milestone & long-TD bonus EV from breakdown (or compute if needed)
    ev_milestones = float(bk.get("ev_milestones", 0.0) or 0.0)
    ev_long_tds = float(bk.get("ev_long_tds", 0.0) or 0.0)
    if ev_milestones == 0.0 and ev_long_tds == 0.0 and not is_inactive:
        rush_yds = float(bk.get("rush_yds", 0.0) or 0.0)
        rec_yds = float(bk.get("rec_yds", 0.0) or 0.0)
        pass_yds = float(bk.get("pass_yds", 0.0) or 0.0)
        exp_tds = float(bk.get("expected_tds", bk.get("pass_tds", 0.0)) or 0.0)
        if pos == "QB":
            ev_milestones = scoring_config.calculate_pass_milestone_ev(pass_yds, cfg=cfg)
            ev_long_tds = scoring_config.calculate_long_pass_td_ev(exp_tds, cfg=cfg)
        elif pos == "RB":
            ev_milestones = (
                scoring_config.calculate_rush_milestone_ev(rush_yds, cfg=cfg)
                + scoring_config.calculate_rec_milestone_ev(rec_yds, cfg=cfg)
            )
            ev_long_tds = scoring_config.calculate_long_rush_rec_td_ev(exp_tds, is_rec=False, cfg=cfg)
        elif pos in ["WR", "TE"]:
            ev_milestones = scoring_config.calculate_rec_milestone_ev(rec_yds, cfg=cfg)
            ev_long_tds = scoring_config.calculate_long_rush_rec_td_ev(exp_tds, is_rec=True, cfg=cfg)

    non_linear_bonus_ev = round(ev_milestones + ev_long_tds, 2)

    raw_mu = float(p.get("mu", 0.0) or 0.0)
    raw_sigma = float(p.get("sigma", 0.0) or 0.0)
    raw_ros_mu = float(p.get("ros_mu", raw_mu) or raw_mu)
    raw_ros_sigma = float(p.get("ros_sigma", raw_sigma) or max(2.5, raw_ros_mu * 0.38))

    # If the source was purely The Athletic's linear stat line, add non-linear milestone + long TD EV
    source_str = str(bk.get("source", p.get("prop_source", "")))
    bonus_unembedded = (
        non_linear_bonus_ev
        if "The Athletic Custom Baseline" in source_str
        else (0.5 * non_linear_bonus_ev if "Hybrid" in source_str else 0.0)
    )

    wk_mu = 0.0 if (is_on_bye or is_inactive) else round(raw_mu + bonus_unembedded, 2)
    wk_sigma = 0.0 if (is_on_bye or is_inactive) else round(max(2.0, raw_sigma), 2)

    ros_model_mu = round(raw_ros_mu + (bonus_unembedded if not is_on_bye else 0.0), 2)
    if ros_model_mu <= 0.0:
        ros_model_mu = float(p.get("projected_avg_points", 0.0) or p.get("avg_points", 0.0) or 8.0)

    avg_pts = float(p.get("avg_points", 0.0) or 0.0)
    proj_avg_pts = float(p.get("projected_avg_points", 0.0) or 0.0)

    # Compute Empirical Blended ROS Mu:
    # Anchors on the active model projection while incorporating season-long ESPN ROS & realized PPG
    if avg_pts > 0.0 and proj_avg_pts > 0.0:
        ros_blended_mu = round(0.50 * ros_model_mu + 0.30 * proj_avg_pts + 0.20 * avg_pts, 2)
        true_ros_mu = round(0.60 * proj_avg_pts + 0.40 * avg_pts, 2)
    elif avg_pts > 0.0:
        ros_blended_mu = round(0.65 * ros_model_mu + 0.35 * avg_pts, 2)
        true_ros_mu = round(0.50 * ros_model_mu + 0.50 * avg_pts, 2)
    elif proj_avg_pts > 0.0:
        ros_blended_mu = round(0.65 * ros_model_mu + 0.35 * proj_avg_pts, 2)
        true_ros_mu = round(proj_avg_pts, 2)
    else:
        ros_blended_mu = ros_model_mu
        true_ros_mu = ros_model_mu

    # Lookup Full-Season Projections & VORP / Auction $ from 2026-FFB-Projections-0812.xlsx
    fs_entry = get_full_season_player_entry(str(p.get("name", "")))
    fs_pos_rk = str(fs_entry.get("fs_pos_rk", "—"))
    fs_ovr_rk = int(fs_entry.get("fs_ovr_rk", 999) or 999)
    fs_fps = float(fs_entry.get("fs_fps", round(true_ros_mu * 17.0, 2)))
    fs_ppg = float(fs_entry.get("fs_ppg", round(fs_fps / 17.0, 2)))
    fs_vorp = float(fs_entry.get("fs_vorp", 0.0))
    fs_auc = float(fs_entry.get("fs_auc", 0.0))

    # Bayesian 50/50 Structural + In-Season ROS (50% Preseason Structural Model PPG + 50% In-Season True ROS PPG)
    bayes_ros_mu = round(0.50 * fs_ppg + 0.50 * true_ros_mu, 2) if fs_ppg > 0 else true_ros_mu

    cv = float(p.get("cv", bk.get("cv", 0.42)) or 0.42)
    ros_sigma = round(max(2.5, ros_blended_mu * (0.25 if pos == "QB" else (0.38 if pos == "RB" else cv))), 2)

    # Standard / Half-PPR reference projection (from The Athletic's default Proj column if available)
    std_proj = float(bk.get("athletic_default_proj", 0.0) or 0.0)
    if std_proj <= 0.0:
        if pos == "QB":
            std_proj = round(ros_model_mu * 0.76, 1)
        elif pos == "RB":
            std_proj = round(max(0.0, ros_model_mu - 0.5 * float(bk.get("receptions", 2.5) or 2.5) - non_linear_bonus_ev), 1)
        elif pos in ["WR", "TE"]:
            std_proj = round(max(0.0, ros_model_mu - 0.5 * float(bk.get("receptions", 4.0) or 4.0) - non_linear_bonus_ev), 1)
        else:
            std_proj = round(ros_model_mu, 1)

    idiosyncratic_lift = round(ros_model_mu - std_proj, 2)

    # Replacement baseline lookup (Note: League has NO dedicated TE slot — only WR1, WR2, WR/TE, and FLEX,
    # so Tight Ends must compete directly against Wide Receivers/Flexes at the WR/FLEX replacement baseline).
    base_val = 9.0
    if baselines:
        if pos == "TE":
            wr_b = float(baselines.get("WR", {}).get("baseline_mean", 9.0))
            flx_b = float(baselines.get("FLEX", {}).get("baseline_mean", 9.0))
            base_val = max(wr_b, flx_b)
        elif pos in baselines:
            base_val = float(baselines[pos].get("baseline_mean", 8.0))
        elif pos in ["RB", "WR", "TE"] and "FLEX" in baselines:
            base_val = float(baselines["FLEX"].get("baseline_mean", 8.0))
    else:
        default_bases = {"QB": 17.5, "RB": 8.5, "WR": 9.0, "TE": 9.0, "D/ST": 5.5}
        base_val = default_bases.get(pos, 8.0)

    scarcity_mult = IDIOSYNCRATIC_SCARCITY_MULTIPLIERS.get(pos, 1.0)
    wk_vorp = round(max(-6.0, wk_mu - base_val) * (scarcity_mult if wk_mu > base_val else 1.0), 2)
    ros_vorp = round(max(-6.0, ros_blended_mu - base_val) * (scarcity_mult if ros_blended_mu > base_val else 1.0), 2)
    ros_model_vorp = round(max(-6.0, ros_model_mu - base_val) * (scarcity_mult if ros_model_mu > base_val else 1.0), 2)

    # Resolve team bye week and Weeks 15-17 playoff schedule & scheme multipliers
    t_code = str(p.get("team", "")).upper().strip()
    sched_info = NFL_BYES_AND_PLAYOFF_SCHEDULE.get(t_code, {"bye": 0, "playoffs": {15: "NFL", 16: "NFL", 17: "NFL"}})
    bye_wk = int(sched_info.get("bye", 0) or 0)
    po_opps = sched_info.get("playoffs", {15: "NFL", 16: "NFL", 17: "NFL"})

    po_week_mus: Dict[int, float] = {}
    po_mults: List[float] = []
    for w_po in [15, 16, 17]:
        opp_po = po_opps.get(w_po, "NFL")
        if pos in ["QB", "RB", "WR", "TE"] and opp_po and opp_po != "NFL" and true_ros_mu > 0:
            mod_p = data_ingest.apply_defensive_scheme_modifiers(
                {"name": p.get("name", ""), "position": pos, "mu": true_ros_mu, "sigma": ros_sigma, "p_active": 1.0},
                opponent_team=opp_po,
            )
            w_val = round(float(mod_p.get("mu", true_ros_mu)), 2)
            po_week_mus[w_po] = w_val
            po_mults.append(w_val / true_ros_mu)
        else:
            po_week_mus[w_po] = round(true_ros_mu, 2)
            po_mults.append(1.0)

    playoff_avg_mu = round(float(np.mean(list(po_week_mus.values()))), 2)
    playoff_scheme_mult = round(float(np.mean(po_mults)), 3)

    # Sample distribution for percentiles (using ROS active distribution when on bye so player profile is informative)
    sim_mu = wk_mu if wk_mu > 0 else ros_blended_mu
    sim_sig = wk_sigma if wk_mu > 0 else ros_sigma
    samples = simulation.sample_player(pos, sim_mu, sim_sig, n=4000, p_active=1.0)
    dist_stats = simulation.summarize_distribution(samples)

    p.update({
        "std_proj": round(std_proj, 2),
        "wk_mu": wk_mu,
        "wk_sigma": wk_sigma,
        "ros_model_mu": ros_model_mu,
        "ros_blended_mu": ros_blended_mu,
        "true_ros_mu": true_ros_mu,
        "fs_pos_rk": fs_pos_rk,
        "fs_ovr_rk": fs_ovr_rk,
        "fs_fps": round(fs_fps, 1),
        "fs_ppg": round(fs_ppg, 2),
        "fs_vorp": round(fs_vorp, 1),
        "fs_auc": round(fs_auc, 2),
        "bayes_ros_mu": bayes_ros_mu,
        "playoff_avg_mu": playoff_avg_mu,
        "playoff_week_mus": po_week_mus,
        "playoff_opponents": po_opps,
        "playoff_scheme_mult": playoff_scheme_mult,
        "bye_week": bye_wk,
        "remaining_bye": f"Wk {bye_wk}" if bye_wk >= 6 else "Done ✅",
        "ros_sigma": ros_sigma,
        "ev_milestones": round(ev_milestones, 2),
        "ev_long_tds": round(ev_long_tds, 2),
        "non_linear_bonus_ev": non_linear_bonus_ev,
        "idiosyncratic_lift": idiosyncratic_lift,
        "replacement_baseline": round(base_val, 2),
        "scarcity_multiplier": scarcity_mult,
        "wk_vorp": wk_vorp,
        "ros_vorp": ros_vorp,
        "ros_model_vorp": ros_model_vorp,
        "p10_floor": round(dist_stats["p10"], 1),
        "p50_median": round(dist_stats["median"], 1),
        "p90_ceiling": round(dist_stats["p90"], 1),
        "p_boom_20": round(dist_stats["p_boom_20"] * 100.0, 1),
    })
    return p


def solve_optimal_9slot_lineup(
    roster: List[Dict[str, Any]],
    horizon: str = "week",
    baselines: Optional[Dict[str, Dict[str, Any]]] = None,
    scoring_cfg: Optional[scoring_config.LeagueScoringConfig] = None,
    week_num: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Solve the exact optimal 9-starter lineup for the user's ESPN league configuration:
      Slots: QB (1), OP (1: QB/RB/WR/TE), RB1 (1), RB2 (1), WR1 (1), WR2 (1),
             WR/TE (1: WR/TE), FLEX (1: RB/WR/TE), D/ST (1).
    `horizon`:
      - "week": Optimizes for immediate current-week points (`wk_mu`, where Bye/OUT = 0.0).
      - "ros": Optimizes for Rest-of-Season full-strength blended expectation (`ros_blended_mu`).
      - "ros_model": Optimizes for Rest-of-Season pure model expectation (`ros_model_mu`).
      - "true_ros": Optimizes for Season-Long True ROS baseline (`true_ros_mu`: 60% ESPN ROS + 40% Realized PPG).
      - "full_season_model": Optimizes for Preseason Full-Season Structural Model (`fs_ppg` from 2026-FFB-Projections-0812.xlsx).
      - "bayes_ros": Optimizes for 50/50 Bayesian blend (`bayes_ros_mu`: 50% Full-Season Structural + 50% In-Season True ROS).
      - "playoffs": Optimizes for Fantasy Playoffs Weeks 15-17 average (`playoff_avg_mu`).
      - "schedule_week": Optimizes for a specific future week `week_num` (7-14 Bye-adjusted or 15-17 Playoff-scheme-adjusted).
    """
    enriched = [
        compute_idiosyncratic_player_valuation(p, baselines=baselines, scoring_cfg=scoring_cfg)
        for p in roster
    ]

    if horizon == "week":
        metric_key = "wk_mu"
    elif horizon == "ros_model":
        metric_key = "ros_model_mu"
    elif horizon == "true_ros":
        metric_key = "true_ros_mu"
    elif horizon == "full_season_model":
        metric_key = "fs_ppg"
    elif horizon == "bayes_ros":
        metric_key = "bayes_ros_mu"
    elif horizon == "playoffs":
        metric_key = "playoff_avg_mu"
    elif horizon == "schedule_week" and week_num is not None:
        metric_key = "_sched_wk_mu"
        for p in enriched:
            if 7 <= week_num <= 14:
                is_wk_bye = (int(p.get("bye_week", 0) or 0) == week_num)
                p["_sched_wk_mu"] = 0.0 if is_wk_bye else float(p.get("true_ros_mu", 0.0))
            elif week_num in (15, 16, 17):
                po_mus = p.get("playoff_week_mus", {}) or {}
                p["_sched_wk_mu"] = float(po_mus.get(week_num, p.get("true_ros_mu", 0.0)))
            else:
                p["_sched_wk_mu"] = float(p.get("wk_mu", 0.0))
    else:
        metric_key = "ros_blended_mu"

    sigma_key = "wk_sigma" if horizon == "week" else "ros_sigma"

    # Filter out IR players when solving active lineup (for ROS, bye players are active; for week, bye players have wk_mu=0)
    eligible_pool = []
    ir_pool = []
    for p in enriched:
        status = str(p.get("injury_status", "ACTIVE")).upper()
        slot_raw = str(p.get("slot", "")).upper()
        if status in ["IR", "OUT"] or slot_raw in ["IR", "IR_ELIGIBLE"]:
            if horizon == "week" or status == "IR" or slot_raw in ["IR", "IR_ELIGIBLE"]:
                ir_pool.append(p)
                continue
        eligible_pool.append(p)

    used_names = set()
    starters: List[Dict[str, Any]] = []

    def _pick_best(allowed_positions: List[str], target_slot_label: str) -> Optional[Dict[str, Any]]:
        candidates = [
            p for p in eligible_pool
            if p["name"] not in used_names
            and (
                p.get("position", "").upper() in allowed_positions
                or ("D/ST" in allowed_positions and p.get("position", "").upper() in ["DEF", "D/ST", "DST"])
            )
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: float(x.get(metric_key, 0.0)), reverse=True)
        chosen = copy.deepcopy(candidates[0])
        chosen["assigned_slot"] = target_slot_label
        chosen["slot"] = target_slot_label
        chosen["active_mu"] = float(chosen.get(metric_key, 0.0))
        chosen["active_sigma"] = float(chosen.get(sigma_key, 0.0))
        chosen["mu"] = chosen["active_mu"]
        chosen["sigma"] = chosen["active_sigma"]
        chosen["p_active"] = 0.0 if chosen["active_mu"] <= 0.0 else (
            float(chosen.get("p_active", 1.0)) if horizon == "week" else 1.0
        )
        used_names.add(chosen["name"])
        starters.append(chosen)
        return chosen

    # 1. Dedicated core positions first: QB, RB1, RB2, WR1, WR2
    _pick_best(["QB"], "QB")
    _pick_best(["RB"], "RB1")
    _pick_best(["RB"], "RB2")
    _pick_best(["WR"], "WR1")
    _pick_best(["WR"], "WR2")

    # 2. OP slot (Superflex: QB, RB, WR, TE) - pick highest projected available (typically QB2 if active)
    _pick_best(["QB", "RB", "WR", "TE"], "OP")

    # 3. WR/TE hybrid slot before general FLEX so RB remains eligible for FLEX
    _pick_best(["WR", "TE"], "WR/TE")

    # 4. FLEX slot (RB, WR, TE)
    _pick_best(["RB", "WR", "TE"], "FLEX")

    # 5. D/ST slot
    _pick_best(["D/ST", "DEF"], "D/ST")

    # Order starters in canonical display sequence
    slot_order = {"QB": 1, "OP": 2, "RB1": 3, "RB2": 4, "WR1": 5, "WR2": 6, "WR/TE": 7, "FLEX": 8, "D/ST": 9}
    starters.sort(key=lambda x: slot_order.get(x.get("assigned_slot", "FLEX"), 99))

    bench = [p for p in eligible_pool if p["name"] not in used_names]
    bench.sort(key=lambda x: float(x.get(metric_key, 0.0)), reverse=True)
    for b in bench:
        b["assigned_slot"] = "BE"
        b["active_mu"] = float(b.get(metric_key, 0.0))
        b["active_sigma"] = float(b.get(sigma_key, 0.0))

    total_starter_mu = round(sum(float(s.get("active_mu", 0.0)) for s in starters), 2)
    total_bench_mu = round(sum(float(b.get("active_mu", 0.0)) for b in bench), 2)

    qb_count = sum(1 for p in enriched if str(p.get("position", "")).upper() == "QB")

    return {
        "starters": starters,
        "bench": bench,
        "ir": ir_pool,
        "enriched_roster": enriched,
        "total_starter_mu": total_starter_mu,
        "total_bench_mu": total_bench_mu,
        "qb_count": qb_count,
        "exceeds_2qb_limit": qb_count > 2,
    }


def compute_weekly_championship_trajectory(
    roster_pre: List[Dict[str, Any]],
    roster_post: List[Dict[str, Any]],
    baselines: Optional[Dict[str, Dict[str, Any]]] = None,
    scoring_cfg: Optional[scoring_config.LeagueScoringConfig] = None,
) -> Dict[str, Any]:
    """
    Simulate the exact week-by-week optimal 9-starter lineup across:
      - Week 6 (Immediate Bye-Relief / Matchup Week)
      - Weeks 7-14 (Rest of Regular Season, accounting for every NFL team's exact Bye week)
      - Weeks 15-17 (Fantasy Playoffs: Quarterfinals W15, Semifinals W16, Championship Final W17,
        incorporating SumerSports defensive scheme modifiers for each playoff opponent)
    """
    rows: List[Dict[str, Any]] = []
    w7_14_pre_vals: List[float] = []
    w7_14_post_vals: List[float] = []
    po_pre_vals: List[float] = []
    po_post_vals: List[float] = []

    # Week 6
    wk6_pre = solve_optimal_9slot_lineup(roster_pre, horizon="week", baselines=baselines, scoring_cfg=scoring_cfg)
    wk6_post = solve_optimal_9slot_lineup(roster_post, horizon="week", baselines=baselines, scoring_cfg=scoring_cfg)
    d6 = round(wk6_post["total_starter_mu"] - wk6_pre["total_starter_mu"], 2)
    byes_pre_6 = [p["name"] for p in wk6_pre["enriched_roster"] if int(p.get("bye_week", 0) or 0) == 6]
    byes_post_6 = [p["name"] for p in wk6_post["enriched_roster"] if int(p.get("bye_week", 0) or 0) == 6]
    rows.append({
        "Week": "Week 6",
        "Phase": "Immediate Matchup",
        "Pre-Trade Lineup": wk6_pre["total_starter_mu"],
        "Post-Trade Lineup": wk6_post["total_starter_mu"],
        "Net Weekly Δ": f"{d6:+.2f}",
        "Pre-Trade Byes / Key Notes": ", ".join(byes_pre_6) if byes_pre_6 else "None",
        "Post-Trade Byes / Key Notes": ", ".join(byes_post_6) if byes_post_6 else "None",
        "Delta_Num": d6,
    })

    # Weeks 7-14 (Regular Season Stretch Run)
    for w in range(7, 15):
        l_pre = solve_optimal_9slot_lineup(
            roster_pre, horizon="schedule_week", baselines=baselines, scoring_cfg=scoring_cfg, week_num=w
        )
        l_post = solve_optimal_9slot_lineup(
            roster_post, horizon="schedule_week", baselines=baselines, scoring_cfg=scoring_cfg, week_num=w
        )
        m_pre = l_pre["total_starter_mu"]
        m_post = l_post["total_starter_mu"]
        dw = round(m_post - m_pre, 2)
        w7_14_pre_vals.append(m_pre)
        w7_14_post_vals.append(m_post)

        byes_pre = [p["name"] for p in l_pre["enriched_roster"] if int(p.get("bye_week", 0) or 0) == w]
        byes_post = [p["name"] for p in l_post["enriched_roster"] if int(p.get("bye_week", 0) or 0) == w]
        rows.append({
            "Week": f"Week {w}",
            "Phase": "Reg. Season (Byes)",
            "Pre-Trade Lineup": m_pre,
            "Post-Trade Lineup": m_post,
            "Net Weekly Δ": f"{dw:+.2f}",
            "Pre-Trade Byes / Key Notes": ", ".join(byes_pre) if byes_pre else "Full Strength",
            "Post-Trade Byes / Key Notes": ", ".join(byes_post) if byes_post else "Full Strength",
            "Delta_Num": dw,
        })

    # Weeks 15-17 (Fantasy Playoffs)
    po_labels = {
        15: "Playoff Quarterfinal",
        16: "Playoff Semifinal",
        17: "🏆 Championship Final",
    }
    for w in (15, 16, 17):
        l_pre = solve_optimal_9slot_lineup(
            roster_pre, horizon="schedule_week", baselines=baselines, scoring_cfg=scoring_cfg, week_num=w
        )
        l_post = solve_optimal_9slot_lineup(
            roster_post, horizon="schedule_week", baselines=baselines, scoring_cfg=scoring_cfg, week_num=w
        )
        m_pre = l_pre["total_starter_mu"]
        m_post = l_post["total_starter_mu"]
        dw = round(m_post - m_pre, 2)
        po_pre_vals.append(m_pre)
        po_post_vals.append(m_post)

        rows.append({
            "Week": f"Week {w}",
            "Phase": po_labels[w],
            "Pre-Trade Lineup": m_pre,
            "Post-Trade Lineup": m_post,
            "Net Weekly Δ": f"{dw:+.2f}",
            "Pre-Trade Byes / Key Notes": "SumerSports Playoff Scheme Adj.",
            "Post-Trade Byes / Key Notes": "SumerSports Playoff Scheme Adj.",
            "Delta_Num": dw,
        })

    w7_14_pre_avg = round(float(np.mean(w7_14_pre_vals)), 2)
    w7_14_post_avg = round(float(np.mean(w7_14_post_vals)), 2)
    w7_14_delta = round(w7_14_post_avg - w7_14_pre_avg, 2)

    po_pre_avg = round(float(np.mean(po_pre_vals)), 2)
    po_post_avg = round(float(np.mean(po_post_vals)), 2)
    po_delta = round(po_post_avg - po_pre_avg, 2)

    return {
        "weekly_trajectory_df": pd.DataFrame(rows),
        "w7_14_pre_avg": w7_14_pre_avg,
        "w7_14_post_avg": w7_14_post_avg,
        "w7_14_starter_delta": w7_14_delta,
        "playoff_w15_17_pre_avg": po_pre_avg,
        "playoff_w15_17_post_avg": po_post_avg,
        "playoff_w15_17_starter_delta": po_delta,
    }



def detect_trade_synergies(
    post_trade_roster: List[Dict[str, Any]],
    incoming_players: List[Dict[str, Any]],
    outgoing_players: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Detect structural roster synergies created or broken by the trade:
      1. Backfield Lockup / Direct Handcuff Insurance (e.g., Jonathan Taylor + Seth McGowan on IND):
         When a roster holds both an elite RB1 (ros_blended_mu >= 14.5) and a secondary RB on the
         exact same NFL team, compute the Contingent Injury Insurance EV:
           Insurance_EV = P(RB1_Misses_Time ~ 10%) * (Contingent_RB1_Output - Standalone_RB2_Output)
      2. Same-Team QB + WR/TE Correlation Stack
      3. Bye-Week Relief Impact (trading away Bye-week assets for active Week 6 starters)
    """
    notes: List[str] = []
    handcuff_pairs: List[Dict[str, Any]] = []
    total_handcuff_insurance_ev = 0.0

    incoming_names = {p["name"] for p in incoming_players}

    # Group RBs by NFL team
    rbs_by_team: Dict[str, List[Dict[str, Any]]] = {}
    for p in post_trade_roster:
        if str(p.get("position", "")).upper() == "RB":
            t = str(p.get("team", "")).upper()
            if t and t not in ["FA", "NFL", ""]:
                rbs_by_team.setdefault(t, []).append(p)

    for team_code, rb_list in rbs_by_team.items():
        if len(rb_list) >= 2:
            rb_sorted = sorted(rb_list, key=lambda x: float(x.get("ros_blended_mu", x.get("ros_mu", 0.0))), reverse=True)
            lead_rb = rb_sorted[0]
            cuff_rb = rb_sorted[1]
            lead_mu = float(lead_rb.get("ros_blended_mu", lead_rb.get("ros_mu", 0.0)))
            cuff_mu = float(cuff_rb.get("ros_blended_mu", cuff_rb.get("ros_mu", 0.0)))

            # Check if this backfield pair involves at least one incoming trade asset
            if lead_rb["name"] in incoming_names or cuff_rb["name"] in incoming_names:
                if lead_mu >= 13.5:
                    # Contingent starter output if RB1 misses time (~78% of RB1's volume/scoring)
                    contingent_mu = round(max(13.5, lead_mu * 0.78), 2)
                    p_injury_contingency = 0.11  # ~11% per-game RB1 missed-time probability over ROS
                    insurance_ev = round(p_injury_contingency * max(0.0, contingent_mu - cuff_mu), 2)
                    total_handcuff_insurance_ev += insurance_ev
                    pair_info = {
                        "team": team_code,
                        "lead_rb": lead_rb["name"],
                        "handcuff_rb": cuff_rb["name"],
                        "lead_ros_mu": lead_mu,
                        "standalone_cuff_mu": cuff_mu,
                        "contingent_starter_mu": contingent_mu,
                        "weekly_insurance_ev": insurance_ev,
                    }
                    handcuff_pairs.append(pair_info)
                    notes.append(
                        f"🔒 **{team_code} Backfield Monopoly ({lead_rb['name']} + {cuff_rb['name']})**: "
                        f"`{cuff_rb['name']}` (`{cuff_mu:.1f}` standalone pts) serves as the direct `{team_code}` injury handcuff "
                        f"to bellcow `{lead_rb['name']}` (`{lead_mu:.1f}` pts). If `{lead_rb['name']}` misses time, "
                        f"`{cuff_rb['name']}` projects for **`{contingent_mu:.1f}` idiosyncratic PPG**, adding **`+{insurance_ev:.2f}` PPG** "
                        f"of structural tail-risk insurance."
                    )

    # Check Same-Team QB + Pass-Catcher (WR/TE) Stacks involving at least one incoming asset
    stack_pairs: List[Dict[str, Any]] = []
    total_stack_ceiling_ev = 0.0
    qbs_by_team: Dict[str, List[Dict[str, Any]]] = {}
    pass_catchers_by_team: Dict[str, List[Dict[str, Any]]] = {}
    for p in post_trade_roster:
        pos_u = str(p.get("position", "")).upper()
        t = str(p.get("team", "")).upper()
        if t and t not in ["FA", "NFL", ""]:
            if pos_u == "QB":
                qbs_by_team.setdefault(t, []).append(p)
            elif pos_u in ["WR", "TE"]:
                pass_catchers_by_team.setdefault(t, []).append(p)

    for team_code, qb_list in qbs_by_team.items():
        if team_code in pass_catchers_by_team:
            for qb in qb_list:
                for pc in pass_catchers_by_team[team_code]:
                    if qb["name"] in incoming_names or pc["name"] in incoming_names:
                        pc_mu = float(pc.get("true_ros_mu", pc.get("ros_blended_mu", 0.0)))
                        qb_mu = float(qb.get("true_ros_mu", qb.get("ros_blended_mu", 0.0)))
                        if pc_mu >= 10.0 and qb_mu >= 18.0:
                            corr_r = 0.32 if str(pc.get("position", "")).upper() == "TE" else 0.38
                            stack_ev = round(1.25 if corr_r == 0.32 else 1.45, 2)
                            total_stack_ceiling_ev += stack_ev
                            bye_wk = int(qb.get("bye_week", 0) or 0)
                            stack_pairs.append({
                                "team": team_code,
                                "qb": qb["name"],
                                "pass_catcher": pc["name"],
                                "position": pc.get("position", "TE"),
                                "copula_r": corr_r,
                                "stack_ceiling_ev": stack_ev,
                                "bye_week": bye_wk,
                            })
                            notes.append(
                                f"🔥 **Idiosyncratic 6pt-Pass-TD / 1.0-PPR Stack ({qb['name']} + {pc['name']}, `{team_code}`)**: "
                                f"Pairs your `{team_code}` QB `{qb['name']}` (`{qb_mu:.1f}` True ROS PPG) with `{pc['name']}` (`{pc_mu:.1f}` True ROS PPG). "
                                f"Under `6.0` pt Pass TDs + `0.1` pt/comp + `1.0` PPR, every `{qb['name']} → {pc['name']}` TD connection scores "
                                f"**`13.1+` points on a single play** (Gaussian Copula `r = +{corr_r:.2f}`, adding **`+{stack_ev:.2f}` PPG** of Playoff P90 ceiling leverage"
                                f"{f' and synchronizing both DET Byes in Week {bye_wk}' if bye_wk == 6 else ''})."
                            )

    # Check Week 6 Bye-Week Liquidity Relief
    outgoing_byes = [p["name"] for p in outgoing_players if p.get("is_on_bye") or str(p.get("injury_status", "")).upper() == "BYE"]
    incoming_active = [
        p["name"] for p in incoming_players
        if not p.get("is_on_bye") and str(p.get("injury_status", "")).upper() not in ["BYE", "OUT", "IR"]
    ]
    if outgoing_byes and incoming_active:
        notes.append(
            f"⚡ **Immediate Week 6 Bye-Week Arbitrage**: Trading away on-bye starter(s) ({', '.join(outgoing_byes)}: `0.0` Wk6 pts) "
            f"for active Week 6 contributors ({', '.join(incoming_active)}) injects immediate Week 6 scoring power while "
            f"`Jared Goff` (DET) is also on bye."
        )

    return {
        "handcuff_pairs": handcuff_pairs,
        "stack_pairs": stack_pairs,
        "total_handcuff_insurance_ev": round(total_handcuff_insurance_ev, 2),
        "total_stack_ceiling_ev": round(total_stack_ceiling_ev, 2),
        "synergy_notes": notes,
    }


def identify_roster_drop_candidates(
    post_trade_roster: List[Dict[str, Any]],
    n_drops_required: int,
    handcuff_names: Optional[List[str]] = None,
    incoming_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    When an N-for-M trade increases a team's active player count (e.g., receiving 4 players
    and sending 3 players -> +1 net player), identify the optimal bench drop candidate(s)
    from the team's pre-existing bench ordered from lowest to highest composite ROS + Season PPG value.
    """
    if n_drops_required <= 0:
        return []

    cuff_set = set(handcuff_names or [])
    inc_set = set(incoming_names or [])
    # Never drop QBs in a 2-QB league, D/ST if it's the only D/ST, or IR players (already in IR slots)
    dst_count = sum(1 for p in post_trade_roster if str(p.get("position", "")).upper() in ["D/ST", "DEF"])

    candidates = []
    for p in post_trade_roster:
        pos = str(p.get("position", "")).upper()
        slot_raw = str(p.get("slot", "")).upper()
        status = str(p.get("injury_status", "")).upper()
        if slot_raw in ["IR", "IR_ELIGIBLE"] or status == "IR":
            continue
        if pos == "QB":
            continue
        if pos in ["D/ST", "DEF"] and dst_count <= 1:
            continue
        if p["name"] in inc_set:
            continue

        # Effective retention value includes ROS blended expectation, realized season PPG, and handcuff protection
        cuff_bonus = 2.00 if p["name"] in cuff_set else 0.0
        avg_pts = float(p.get("avg_points", 0.0) or 0.0)
        ros_val = float(p.get("ros_blended_mu", 0.0) or 0.0)
        wk_val = float(p.get("wk_mu", 0.0) or 0.0)
        retention_score = (
            0.50 * ros_val
            + (0.30 * avg_pts if avg_pts > 0 else 0.30 * ros_val)
            + 0.20 * wk_val
            + cuff_bonus
        )
        cand = copy.deepcopy(p)
        cand["retention_score"] = round(retention_score, 2)
        cand["is_handcuff_protected"] = p["name"] in cuff_set
        candidates.append(cand)

    candidates.sort(key=lambda x: x["retention_score"])
    return candidates[: max(4, n_drops_required)]


def evaluate_trade_package(
    team_a_name: str,
    team_a_roster: List[Dict[str, Any]],
    team_b_name: str,
    team_b_roster: List[Dict[str, Any]],
    team_a_gives_names: List[str],
    team_a_receives_names: List[str],
    week6_opponent_name: Optional[str] = None,
    week6_opponent_starters: Optional[List[Dict[str, Any]]] = None,
    waiver_pool: Optional[List[Dict[str, Any]]] = None,
    scoring_cfg: Optional[scoring_config.LeagueScoringConfig] = None,
    n_simulations: int = 10000,
    forced_drop_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a comprehensive, idiosyncratic-scoring-calibrated trade evaluation between
    `team_a` (User) and `team_b` (Trade Partner):
      1. Player-by-player idiosyncratic valuation & bonus decomposition (1.0 PPR, 6pt Pass TD,
         +0.1 comp, 100/200-yd Gamma/LogNormal milestone EV, 40/50-yd long TD EV).
      2. Uneven roster-slot balancing (e.g., 4-for-3 forced bench drop on Team A & open waiver slot on Team B).
      3. Backfield handcuff & structural synergy detection (`Jonathan Taylor` + `Seth McGowan`).
      4. Dual-Horizon Optimal 9-Starter Lineup Solver & 10,000-Run Monte Carlo Simulation:
         - Horizon 1: Immediate Week 6 Matchup (`wk_mu`, including Delta P(Win) vs. Week 6 opponent)
         - Horizon 2: Rest-of-Season Full-Strength Post-Bye (`ros_blended_mu` and `ros_model_mu`)
      5. Bilateral Trade Partner (`team_b`) impact analysis.
    """
    cfg = scoring_cfg or scoring_config.DEFAULT_SCORING
    waivers = waiver_pool or []
    baselines = vorp.calculate_waiver_baselines(waivers, n_simulations=3000) if waivers else None

    # Enrich both full rosters
    roster_a_pre = [
        compute_idiosyncratic_player_valuation(p, baselines=baselines, scoring_cfg=cfg)
        for p in team_a_roster
    ]
    roster_b_pre = [
        compute_idiosyncratic_player_valuation(p, baselines=baselines, scoring_cfg=cfg)
        for p in team_b_roster
    ]

    gives_set = {n.strip() for n in team_a_gives_names}
    receives_set = {n.strip() for n in team_a_receives_names}

    outgoing_players = [p for p in roster_a_pre if p["name"] in gives_set]
    incoming_players = [p for p in roster_b_pre if p["name"] in receives_set]

    # Build raw post-trade rosters before drop/waiver adjustments
    roster_a_post_raw = [p for p in roster_a_pre if p["name"] not in gives_set] + copy.deepcopy(incoming_players)
    roster_b_post_raw = [p for p in roster_b_pre if p["name"] not in receives_set] + copy.deepcopy(outgoing_players)

    # Detect synergies on Team A post-trade roster
    synergies_a = detect_trade_synergies(roster_a_post_raw, incoming_players, outgoing_players)
    handcuff_names_a = [pair["handcuff_rb"] for pair in synergies_a["handcuff_pairs"]]

    # Handle uneven trade slot balancing (N received vs M given)
    net_players_added_a = len(incoming_players) - len(outgoing_players)
    drop_candidates_a = []
    dropped_players_a: List[Dict[str, Any]] = []
    added_waivers_b: List[Dict[str, Any]] = []

    roster_a_post = copy.deepcopy(roster_a_post_raw)
    roster_b_post = copy.deepcopy(roster_b_post_raw)

    if net_players_added_a > 0:
        drop_candidates_a = identify_roster_drop_candidates(
            roster_a_post_raw,
            n_drops_required=net_players_added_a,
            handcuff_names=handcuff_names_a,
            incoming_names=list(receives_set),
        )
        if forced_drop_override and forced_drop_override != "None (Wait / Use Open Slot)":
            chosen_drops = [p for p in roster_a_post_raw if p["name"] == forced_drop_override]
            if not chosen_drops and drop_candidates_a:
                chosen_drops = drop_candidates_a[:net_players_added_a]
        else:
            chosen_drops = drop_candidates_a[:net_players_added_a] if forced_drop_override != "None (Wait / Use Open Slot)" else []

        dropped_names = {d["name"] for d in chosen_drops}
        dropped_players_a = chosen_drops
        roster_a_post = [p for p in roster_a_post_raw if p["name"] not in dropped_names]

        # Meanwhile Team B has open roster slot(s) and can add top available active skill waiver player(s)
        if waivers:
            enriched_waivers = [
                compute_idiosyncratic_player_valuation(w, baselines=baselines, scoring_cfg=cfg)
                for w in waivers
                if str(w.get("position", "")).upper() in ["RB", "WR", "TE"] and not w.get("is_inactive", False)
            ]
            enriched_waivers.sort(key=lambda x: float(x.get("ros_blended_mu", 0.0)), reverse=True)
            added_waivers_b = enriched_waivers[:net_players_added_a]
            roster_b_post = roster_b_post_raw + copy.deepcopy(added_waivers_b)

    elif net_players_added_a < 0:
        # Team A consolidated (e.g., 2-for-1 or 3-for-2) and gets open roster slot(s)!
        n_open_a = abs(net_players_added_a)
        if waivers:
            enriched_waivers = [
                compute_idiosyncratic_player_valuation(w, baselines=baselines, scoring_cfg=cfg)
                for w in waivers
                if str(w.get("position", "")).upper() in ["RB", "WR", "TE"] and not w.get("is_inactive", False)
            ]
            enriched_waivers.sort(key=lambda x: float(x.get("ros_blended_mu", 0.0)), reverse=True)
            roster_a_post = roster_a_post_raw + copy.deepcopy(enriched_waivers[:n_open_a])

    # Solve Optimal 9-Starter Lineups across all 3 horizons:
    # 1. Immediate Week 6 ("week")
    lineup_a_wk_pre = solve_optimal_9slot_lineup(roster_a_pre, horizon="week", baselines=baselines, scoring_cfg=cfg)
    lineup_a_wk_post = solve_optimal_9slot_lineup(roster_a_post, horizon="week", baselines=baselines, scoring_cfg=cfg)

    # 2. Rest-of-Season Empirical Blended ("ros")
    lineup_a_ros_pre = solve_optimal_9slot_lineup(roster_a_pre, horizon="ros", baselines=baselines, scoring_cfg=cfg)
    lineup_a_ros_post = solve_optimal_9slot_lineup(roster_a_post, horizon="ros", baselines=baselines, scoring_cfg=cfg)

    # 3. Rest-of-Season Pure Model ("ros_model")
    lineup_a_model_pre = solve_optimal_9slot_lineup(roster_a_pre, horizon="ros_model", baselines=baselines, scoring_cfg=cfg)
    lineup_a_model_post = solve_optimal_9slot_lineup(roster_a_post, horizon="ros_model", baselines=baselines, scoring_cfg=cfg)

    # 4. Season-Long True ROS Baseline ("true_ros": 60% ESPN League ROS + 40% Realized PPG)
    lineup_a_true_ros_pre = solve_optimal_9slot_lineup(roster_a_pre, horizon="true_ros", baselines=baselines, scoring_cfg=cfg)
    lineup_a_true_ros_post = solve_optimal_9slot_lineup(roster_a_post, horizon="true_ros", baselines=baselines, scoring_cfg=cfg)

    # 5. Fantasy Playoffs Weeks 15-17 Scheme-Adjusted ("playoffs")
    lineup_a_playoffs_pre = solve_optimal_9slot_lineup(roster_a_pre, horizon="playoffs", baselines=baselines, scoring_cfg=cfg)
    lineup_a_playoffs_post = solve_optimal_9slot_lineup(roster_a_post, horizon="playoffs", baselines=baselines, scoring_cfg=cfg)

    # 6. Full-Season Structural Model ("full_season_model" from 2026-FFB-Projections-0812.xlsx)
    lineup_a_fs_pre = solve_optimal_9slot_lineup(roster_a_pre, horizon="full_season_model", baselines=baselines, scoring_cfg=cfg)
    lineup_a_fs_post = solve_optimal_9slot_lineup(roster_a_post, horizon="full_season_model", baselines=baselines, scoring_cfg=cfg)

    # 7. Bayesian 50/50 Structural + In-Season ROS ("bayes_ros")
    lineup_a_bayes_pre = solve_optimal_9slot_lineup(roster_a_pre, horizon="bayes_ros", baselines=baselines, scoring_cfg=cfg)
    lineup_a_bayes_post = solve_optimal_9slot_lineup(roster_a_post, horizon="bayes_ros", baselines=baselines, scoring_cfg=cfg)

    # 8. Exact Week-by-Week Trajectory (Weeks 6-14 Regular Season Byes + Weeks 15-17 Playoffs)
    traj = compute_weekly_championship_trajectory(
        roster_a_pre, roster_a_post, baselines=baselines, scoring_cfg=cfg
    )

    # Partner (Team B) Optimal Lineups
    lineup_b_wk_pre = solve_optimal_9slot_lineup(roster_b_pre, horizon="week", baselines=baselines, scoring_cfg=cfg)
    lineup_b_wk_post = solve_optimal_9slot_lineup(roster_b_post, horizon="week", baselines=baselines, scoring_cfg=cfg)
    lineup_b_ros_pre = solve_optimal_9slot_lineup(roster_b_pre, horizon="ros", baselines=baselines, scoring_cfg=cfg)
    lineup_b_ros_post = solve_optimal_9slot_lineup(roster_b_post, horizon="ros", baselines=baselines, scoring_cfg=cfg)

    # 10,000-Iteration Monte Carlo Simulations for Week 6 Matchup & ROS Starter Distributions
    opp_starters = week6_opponent_starters
    if not opp_starters:
        opp_starters = lineup_b_wk_pre["starters"]
        week6_opponent_name = week6_opponent_name or team_b_name

    sim_wk_pre = simulation.simulate_matchup(
        lineup_a_wk_pre["starters"], opp_starters, n_simulations=n_simulations
    )
    sim_wk_post = simulation.simulate_matchup(
        lineup_a_wk_post["starters"], opp_starters, n_simulations=n_simulations
    )

    # Simulate ROS Full-Strength Starter Distributions against league-average contender (or opponent full-strength)
    opp_ros_starters = solve_optimal_9slot_lineup(opp_starters, horizon="ros", baselines=baselines, scoring_cfg=cfg)["starters"]
    sim_ros_pre = simulation.simulate_matchup(
        lineup_a_ros_pre["starters"], opp_ros_starters, n_simulations=n_simulations
    )
    sim_ros_post = simulation.simulate_matchup(
        lineup_a_ros_post["starters"], opp_ros_starters, n_simulations=n_simulations
    )

    sim_model_pre = simulation.simulate_matchup(
        lineup_a_model_pre["starters"], opp_ros_starters, n_simulations=n_simulations
    )
    sim_model_post = simulation.simulate_matchup(
        lineup_a_model_post["starters"], opp_ros_starters, n_simulations=n_simulations
    )

    # Package Asset Metrics
    incoming_wk_mu = round(sum(float(p.get("wk_mu", 0.0)) for p in incoming_players), 2)
    outgoing_wk_mu = round(sum(float(p.get("wk_mu", 0.0)) for p in outgoing_players), 2)

    incoming_ros_mu = round(sum(float(p.get("ros_blended_mu", 0.0)) for p in incoming_players), 2)
    outgoing_ros_mu = round(sum(float(p.get("ros_blended_mu", 0.0)) for p in outgoing_players), 2)

    incoming_ros_vorp = round(sum(float(p.get("ros_vorp", 0.0)) for p in incoming_players), 2)
    outgoing_ros_vorp = round(sum(float(p.get("ros_vorp", 0.0)) for p in outgoing_players), 2)

    dropped_ros_vorp = round(sum(float(p.get("ros_vorp", 0.0)) for p in dropped_players_a), 2)
    dropped_ros_mu = round(sum(float(p.get("ros_blended_mu", 0.0)) for p in dropped_players_a), 2)

    # Full-Season Structural Model Package Metrics (from 2026-FFB-Projections-0812.xlsx)
    fs_incoming_fps = round(sum(float(p.get("fs_fps", 0.0)) for p in incoming_players), 1)
    fs_outgoing_fps = round(sum(float(p.get("fs_fps", 0.0)) for p in outgoing_players), 1)
    fs_incoming_vorp = round(sum(float(p.get("fs_vorp", 0.0)) for p in incoming_players), 1)
    fs_outgoing_vorp = round(sum(float(p.get("fs_vorp", 0.0)) for p in outgoing_players), 1)
    fs_incoming_auc = round(sum(float(p.get("fs_auc", 0.0)) for p in incoming_players), 2)
    fs_outgoing_auc = round(sum(float(p.get("fs_auc", 0.0)) for p in outgoing_players), 2)
    # Active (non-negative) Full-Season VORP transferred (ignoring negative bench throw-ins vs replacement)
    fs_incoming_active_vorp = round(sum(max(0.0, float(p.get("fs_vorp", 0.0))) for p in incoming_players), 1)
    fs_outgoing_active_vorp = round(sum(max(0.0, float(p.get("fs_vorp", 0.0))) for p in outgoing_players), 1)

    # Consolidation Stud Bonus:
    # In a 9-starter league, acquiring the single highest-VORP centerpiece asset in the deal carries
    # structural lineup leverage because 1 elite slot outperforms 2 mid-tier pieces occupying 2 slots.
    best_in_vorp = max([float(p.get("ros_vorp", 0.0)) for p in incoming_players], default=0.0)
    best_out_vorp = max([float(p.get("ros_vorp", 0.0)) for p in outgoing_players], default=0.0)
    alpha_centerpiece_edge = round(best_in_vorp - best_out_vorp, 2)

    # Key Lineup & Matchup Deltas
    wk6_starter_delta = round(lineup_a_wk_post["total_starter_mu"] - lineup_a_wk_pre["total_starter_mu"], 2)
    wk6_win_prob_pre = round(sim_wk_pre["p_win_a"] * 100.0, 2)
    wk6_win_prob_post = round(sim_wk_post["p_win_a"] * 100.0, 2)
    wk6_win_prob_delta = round(wk6_win_prob_post - wk6_win_prob_pre, 2)

    ros_starter_delta = round(lineup_a_ros_post["total_starter_mu"] - lineup_a_ros_pre["total_starter_mu"], 2)
    ros_model_starter_delta = round(lineup_a_model_post["total_starter_mu"] - lineup_a_model_pre["total_starter_mu"], 2)
    true_ros_starter_delta = round(lineup_a_true_ros_post["total_starter_mu"] - lineup_a_true_ros_pre["total_starter_mu"], 2)
    fs_starter_delta = round(lineup_a_fs_post["total_starter_mu"] - lineup_a_fs_pre["total_starter_mu"], 2)
    bayes_ros_starter_delta = round(lineup_a_bayes_post["total_starter_mu"] - lineup_a_bayes_pre["total_starter_mu"], 2)
    ros_effective_delta_with_insurance = round(
        ros_starter_delta + synergies_a["total_handcuff_insurance_ev"], 2
    )
    ros_model_effective_delta = round(
        ros_model_starter_delta + synergies_a["total_handcuff_insurance_ev"], 2
    )

    ros_win_prob_delta = round((sim_ros_post["p_win_a"] - sim_ros_pre["p_win_a"]) * 100.0, 2)
    ros_model_win_prob_delta = round((sim_model_post["p_win_a"] - sim_model_pre["p_win_a"]) * 100.0, 2)

    partner_wk6_starter_delta = round(lineup_b_wk_post["total_starter_mu"] - lineup_b_wk_pre["total_starter_mu"], 2)
    partner_ros_starter_delta = round(lineup_b_ros_post["total_starter_mu"] - lineup_b_ros_pre["total_starter_mu"], 2)

    # Composite Trade Grade & Verdict
    # Weighting:
    # - Immediate Week 6 Starter Leverage (especially critical when navigating Bye crunches)
    # - Full-Strength ROS Starting Lineup Delta (Model + Empirical Blended)
    # - Alpha Centerpiece Consolidation Edge + Handcuff Insurance Synergy
    composite_edge_pts = round(
        0.30 * wk6_starter_delta
        + 0.35 * ros_model_effective_delta
        + 0.35 * ros_effective_delta_with_insurance
        + 0.15 * alpha_centerpiece_edge,
        2,
    )

    if composite_edge_pts >= 3.0 or (wk6_starter_delta >= 8.0 and ros_model_effective_delta >= -0.5):
        verdict = "✅ ACCEPT / STRONG WIN (Immediate Wk6 Smash + Elite RB1 Lockup)"
        grade = "A-" if ros_starter_delta < 0 else "A"
    elif composite_edge_pts >= 1.0:
        verdict = "🟩 SLIGHT WIN / FAVORABLE CONSOLIDATION"
        grade = "B+"
    elif composite_edge_pts >= -1.25:
        verdict = "⚖️ FAIR / STRATEGIC HORIZON TRADE-OFF (Wk6 Surge vs. WR Depth)"
        grade = "B"
    elif composite_edge_pts >= -3.0:
        verdict = "🟧 SLIGHT LOSS (Excess Depth Surrendered)"
        grade = "C+"
    else:
        verdict = "🛑 DECLINE / NEGATIVE STARTER LEVERAGE"
        grade = "D"

    # Build Player Comparison DataFrames for UI & Reporting
    def _build_side_df(players_list: List[Dict[str, Any]], side_label: str) -> pd.DataFrame:
        rows = []
        for p in players_list:
            pos = p.get("position", "")
            rank_val = int(p.get("pos_rank", 0) or 0)
            rank_str = f"{pos} #{rank_val}" if rank_val > 0 else pos
            ovr_rk = int(p.get("fs_ovr_rk", 999) or 999)
            fs_rk_str = f"{p.get('fs_pos_rk', '—')} (#{ovr_rk})" if ovr_rk < 900 else str(p.get("fs_pos_rk", "—"))
            rows.append({
                "Side": side_label,
                "Player": p.get("name", ""),
                "ESPN Rank": rank_str,
                "FS Model RK": fs_rk_str,
                "FS FPS (PPG)": f"{float(p.get('fs_fps', 0.0)):.1f} ({float(p.get('fs_ppg', 0.0)):.2f})",
                "FS True VORP": f"{float(p.get('fs_vorp', 0.0)):+.1f}",
                "FS Auc $": f"${float(p.get('fs_auc', 0.0)):.2f}",
                "Season PPG": round(float(p.get("avg_points", 0.0)), 2),
                "True ROS PPG": round(float(p.get("true_ros_mu", 0.0)), 2),
                "Bayes 50/50 PPG": round(float(p.get("bayes_ros_mu", 0.0)), 2),
                "Playoffs (W15-17)": round(float(p.get("playoff_avg_mu", 0.0)), 2),
                "Rem. Bye": p.get("remaining_bye", "—"),
                "Wk6 Idio Proj": round(float(p.get("wk_mu", 0.0)), 2),
            })
        return pd.DataFrame(rows)

    df_incoming = _build_side_df(incoming_players, f"📥 {team_a_name} Receives")
    df_outgoing = _build_side_df(outgoing_players, f"📤 {team_a_name} Sends")

    # Build Starter Lineup Comparison Tables (Week 6 and ROS)
    def _build_lineup_diff_df(pre_lineup: Dict[str, Any], post_lineup: Dict[str, Any]) -> pd.DataFrame:
        rows = []
        pre_s = pre_lineup["starters"]
        post_s = post_lineup["starters"]
        max_len = max(len(pre_s), len(post_s))
        for i in range(max_len):
            s1 = pre_s[i] if i < len(pre_s) else {}
            s2 = post_s[i] if i < len(post_s) else {}
            slot_lbl = s2.get("assigned_slot") or s1.get("assigned_slot") or f"SLOT {i+1}"
            m1 = float(s1.get("active_mu", 0.0))
            m2 = float(s2.get("active_mu", 0.0))
            diff = round(m2 - m1, 2)
            rows.append({
                "Slot": slot_lbl,
                "Pre-Trade Starter": f"{s1.get('name', '—')} ({s1.get('position', '')}, {s1.get('team', '')})",
                "Pre Proj": round(m1, 2),
                "Post-Trade Starter": f"{s2.get('name', '—')} ({s2.get('position', '')}, {s2.get('team', '')})",
                "Post Proj": round(m2, 2),
                "Slot Net Δ": f"{diff:+.2f}",
                "Delta_Num": diff,
            })
        return pd.DataFrame(rows)

    df_wk6_lineup_diff = _build_lineup_diff_df(lineup_a_wk_pre, lineup_a_wk_post)
    df_ros_lineup_diff = _build_lineup_diff_df(lineup_a_ros_pre, lineup_a_ros_post)
    df_model_lineup_diff = _build_lineup_diff_df(lineup_a_model_pre, lineup_a_model_post)
    df_true_ros_lineup_diff = _build_lineup_diff_df(lineup_a_true_ros_pre, lineup_a_true_ros_post)
    df_fs_lineup_diff = _build_lineup_diff_df(lineup_a_fs_pre, lineup_a_fs_post)
    df_bayes_lineup_diff = _build_lineup_diff_df(lineup_a_bayes_pre, lineup_a_bayes_post)
    df_playoff_lineup_diff = _build_lineup_diff_df(lineup_a_playoffs_pre, lineup_a_playoffs_post)
    df_partner_ros_diff = _build_lineup_diff_df(lineup_b_ros_pre, lineup_b_ros_post)

    return {
        "team_a_name": team_a_name,
        "team_b_name": team_b_name,
        "week6_opponent_name": week6_opponent_name or team_b_name,
        "incoming_players": incoming_players,
        "outgoing_players": outgoing_players,
        "dropped_players_a": dropped_players_a,
        "drop_candidates_a": drop_candidates_a,
        "added_waivers_b": added_waivers_b,
        "net_players_added_a": net_players_added_a,
        "synergies_a": synergies_a,
        "verdict": verdict,
        "grade": grade,
        "composite_edge_pts": composite_edge_pts,
        "alpha_centerpiece_edge": alpha_centerpiece_edge,
        "metrics": {
            "incoming_wk_mu": incoming_wk_mu,
            "outgoing_wk_mu": outgoing_wk_mu,
            "incoming_ros_mu": incoming_ros_mu,
            "outgoing_ros_mu": outgoing_ros_mu,
            "incoming_ros_vorp": incoming_ros_vorp,
            "outgoing_ros_vorp": outgoing_ros_vorp,
            "dropped_ros_mu": dropped_ros_mu,
            "dropped_ros_vorp": dropped_ros_vorp,
            "net_package_ros_vorp": round(incoming_ros_vorp - outgoing_ros_vorp - max(0.0, dropped_ros_vorp), 2),
            "fs_incoming_fps": fs_incoming_fps,
            "fs_outgoing_fps": fs_outgoing_fps,
            "fs_net_fps_delta": round(fs_incoming_fps - fs_outgoing_fps, 1),
            "fs_incoming_vorp": fs_incoming_vorp,
            "fs_outgoing_vorp": fs_outgoing_vorp,
            "fs_incoming_active_vorp": fs_incoming_active_vorp,
            "fs_outgoing_active_vorp": fs_outgoing_active_vorp,
            "fs_net_active_vorp_delta": round(fs_incoming_active_vorp - fs_outgoing_active_vorp, 1),
            "fs_incoming_auc": fs_incoming_auc,
            "fs_outgoing_auc": fs_outgoing_auc,
            "fs_net_auc_delta": round(fs_incoming_auc - fs_outgoing_auc, 2),
            "fs_starter_pre": lineup_a_fs_pre["total_starter_mu"],
            "fs_starter_post": lineup_a_fs_post["total_starter_mu"],
            "fs_starter_delta": fs_starter_delta,
            "bayes_ros_starter_pre": lineup_a_bayes_pre["total_starter_mu"],
            "bayes_ros_starter_post": lineup_a_bayes_post["total_starter_mu"],
            "bayes_ros_starter_delta": bayes_ros_starter_delta,
            "wk6_starter_pre": lineup_a_wk_pre["total_starter_mu"],
            "wk6_starter_post": lineup_a_wk_post["total_starter_mu"],
            "wk6_starter_delta": wk6_starter_delta,
            "wk6_win_prob_pre": wk6_win_prob_pre,
            "wk6_win_prob_post": wk6_win_prob_post,
            "wk6_win_prob_delta": wk6_win_prob_delta,
            "wk6_p10_pre": round(sim_wk_pre["team_a_summary"]["p10"], 1),
            "wk6_p10_post": round(sim_wk_post["team_a_summary"]["p10"], 1),
            "wk6_p90_pre": round(sim_wk_pre["team_a_summary"]["p90"], 1),
            "wk6_p90_post": round(sim_wk_post["team_a_summary"]["p90"], 1),
            "ros_starter_pre": lineup_a_ros_pre["total_starter_mu"],
            "ros_starter_post": lineup_a_ros_post["total_starter_mu"],
            "ros_starter_delta": ros_starter_delta,
            "ros_effective_delta_with_insurance": ros_effective_delta_with_insurance,
            "ros_win_prob_delta": ros_win_prob_delta,
            "ros_model_starter_pre": lineup_a_model_pre["total_starter_mu"],
            "ros_model_starter_post": lineup_a_model_post["total_starter_mu"],
            "ros_model_starter_delta": ros_model_starter_delta,
            "ros_model_effective_delta": ros_model_effective_delta,
            "ros_model_win_prob_delta": ros_model_win_prob_delta,
            "true_ros_starter_pre": lineup_a_true_ros_pre["total_starter_mu"],
            "true_ros_starter_post": lineup_a_true_ros_post["total_starter_mu"],
            "true_ros_starter_delta": true_ros_starter_delta,
            "w7_14_pre_avg": traj["w7_14_pre_avg"],
            "w7_14_post_avg": traj["w7_14_post_avg"],
            "w7_14_starter_delta": traj["w7_14_starter_delta"],
            "playoff_w15_17_pre_avg": traj["playoff_w15_17_pre_avg"],
            "playoff_w15_17_post_avg": traj["playoff_w15_17_post_avg"],
            "playoff_w15_17_starter_delta": traj["playoff_w15_17_starter_delta"],
            "ros_p10_pre": round(sim_ros_pre["team_a_summary"]["p10"], 1),
            "ros_p10_post": round(sim_ros_post["team_a_summary"]["p10"], 1),
            "ros_p90_pre": round(sim_ros_pre["team_a_summary"]["p90"], 1),
            "ros_p90_post": round(sim_ros_post["team_a_summary"]["p90"], 1),
            "partner_wk6_starter_delta": partner_wk6_starter_delta,
            "partner_ros_starter_delta": partner_ros_starter_delta,
        },
        "lineups": {
            "a_wk_pre": lineup_a_wk_pre,
            "a_wk_post": lineup_a_wk_post,
            "a_ros_pre": lineup_a_ros_pre,
            "a_ros_post": lineup_a_ros_post,
            "a_model_pre": lineup_a_model_pre,
            "a_model_post": lineup_a_model_post,
            "a_true_ros_pre": lineup_a_true_ros_pre,
            "a_true_ros_post": lineup_a_true_ros_post,
            "a_fs_pre": lineup_a_fs_pre,
            "a_fs_post": lineup_a_fs_post,
            "a_bayes_pre": lineup_a_bayes_pre,
            "a_bayes_post": lineup_a_bayes_post,
            "a_playoffs_pre": lineup_a_playoffs_pre,
            "a_playoffs_post": lineup_a_playoffs_post,
            "b_wk_pre": lineup_b_wk_pre,
            "b_wk_post": lineup_b_wk_post,
            "b_ros_pre": lineup_b_ros_pre,
            "b_ros_post": lineup_b_ros_post,
        },
        "simulations": {
            "wk_pre": sim_wk_pre,
            "wk_post": sim_wk_post,
            "ros_pre": sim_ros_pre,
            "ros_post": sim_ros_post,
            "model_pre": sim_model_pre,
            "model_post": sim_model_post,
        },
        "tables": {
            "incoming_df": df_incoming,
            "outgoing_df": df_outgoing,
            "wk6_lineup_diff_df": df_wk6_lineup_diff,
            "ros_lineup_diff_df": df_ros_lineup_diff,
            "model_lineup_diff_df": df_model_lineup_diff,
            "true_ros_lineup_diff_df": df_true_ros_lineup_diff,
            "fs_lineup_diff_df": df_fs_lineup_diff,
            "bayes_lineup_diff_df": df_bayes_lineup_diff,
            "playoff_lineup_diff_df": df_playoff_lineup_diff,
            "weekly_trajectory_df": traj["weekly_trajectory_df"],
            "partner_ros_diff_df": df_partner_ros_diff,
        },
    }


def get_default_bodhizafa_trade_rosters(
    scoring_cfg: Optional[scoring_config.LeagueScoringConfig] = None,
) -> Dict[str, Any]:
    """
    Build calibrated Week 6 rosters for 'Eatin TDs like Groceries' and 'BODHIZAFA'
    using Week6_Proj_1015.xlsx + live ESPN league season statistics so the proposed
    4-for-3 trade can be evaluated even in standalone / offline test environments.
    """
    cfg = scoring_cfg or scoring_config.DEFAULT_SCORING
    props_cache = data_ingest.build_all_player_props_cache()

    def _mk(name: str, pos: str, team: str, opp: str, slot: str, status: str, avg_pts: float, proj_avg: float, pos_rank: int) -> Dict[str, Any]:
        is_bye = (opp == "BYE" or status == "BYE")
        is_inact = status in ["IR", "OUT"]
        p_props = data_ingest.get_matched_player_props(name, props_cache, pos)
        ros_mu, ros_sig, p_act, bk = data_ingest.calculate_projection_params(
            position=pos,
            props=p_props if p_props else None,
            default_proj=proj_avg if proj_avg > 0 else avg_pts,
            ppr=cfg.ppr,
            polymarket_prob=1.0 if not is_inact else 0.0,
            is_inactive=is_inact,
            scoring_cfg=cfg,
            return_breakdown=True,
            player_name=name,
        )
        mu = 0.0 if (is_bye or is_inact) else ros_mu
        sigma = 0.0 if (is_bye or is_inact) else ros_sig
        return {
            "name": name,
            "position": pos,
            "team": team,
            "opponent": opp,
            "opponent_team": opp,
            "slot": slot,
            "injury_status": status,
            "is_on_bye": is_bye,
            "is_inactive": is_bye or is_inact,
            "mu": mu,
            "sigma": sigma,
            "ros_mu": ros_mu,
            "ros_sigma": ros_sig,
            "p_active": 0.0 if (is_bye or is_inact) else (0.50 if status == "QUESTIONABLE" else p_act),
            "avg_points": avg_pts,
            "projected_avg_points": proj_avg,
            "pos_rank": pos_rank,
            "prop_source": bk.get("source", "The Athletic Custom Baseline"),
            "score_breakdown": bk,
        }

    eatin_roster = [
        _mk("Dak Prescott", "QB", "DAL", "GB", "QB", "ACTIVE", 26.10, 25.16, 8),
        _mk("Jared Goff", "QB", "DET", "BYE", "BE", "BYE", 30.88, 27.30, 3),
        _mk("Chase Brown", "RB", "CIN", "BYE", "BE", "BYE", 13.00, 15.78, 16),
        _mk("Bucky Irving", "RB", "TB", "PIT", "RB", "ACTIVE", 15.94, 16.48, 14),
        _mk("MarShawn Lloyd", "RB", "GB", "DAL", "RB", "ACTIVE", 8.54, 6.40, 35),
        _mk("Kenny Gainwell", "RB", "TB", "PIT", "BE", "ACTIVE", 7.16, 9.50, 40),
        _mk("Rico Dowdle", "RB", "PIT", "TB", "BE", "QUESTIONABLE", 5.25, 7.39, 79),
        _mk("J.K. Dobbins", "RB", "DEN", "SEA", "BE", "ACTIVE", 7.22, 11.76, 39),
        _mk("Drake London", "WR", "ATL", "CHI", "WR", "ACTIVE", 15.64, 15.89, 13),
        _mk("DJ Moore", "WR", "BUF", "LV", "WR", "QUESTIONABLE", 9.57, 14.34, 58),
        _mk("Jalen Coker", "WR", "CAR", "PHI", "OP", "ACTIVE", 19.07, 15.76, 27),
        _mk("Wan'Dale Robinson", "WR", "TEN", "IND", "WR/TE", "ACTIVE", 10.34, 8.87, 34),
        _mk("Ladd McConkey", "WR", "LAC", "KC", "BE", "QUESTIONABLE", 9.07, 11.44, 66),
        _mk("Colston Loveland", "TE", "CHI", "ATL", "RB/WR/TE", "ACTIVE", 4.80, 8.75, 33),
        _mk("Browns D/ST", "D/ST", "CLE", "BAL", "D/ST", "ACTIVE", 6.60, 5.28, 16),
        _mk("Jonah Coleman", "RB", "DEN", "SEA", "IR", "OUT", 8.83, 10.02, 50),
        _mk("Omar Cooper Jr.", "WR", "NYJ", "IND", "IR", "OUT", 13.70, 9.21, 56),
    ]

    bodhizafa_roster = [
        _mk("Jacoby Brissett", "QB", "ARI", "LAR", "QB", "ACTIVE", 23.62, 24.74, 14),
        _mk("Sam Darnold", "QB", "SEA", "DEN", "OP", "ACTIVE", 18.72, 22.20, 26),
        _mk("Baker Mayfield", "QB", "TB", "PIT", "BE", "ACTIVE", 15.53, 19.62, 33),
        _mk("Bijan Robinson", "RB", "ATL", "CHI", "RB", "ACTIVE", 25.24, 22.99, 2),
        _mk("Jonathan Taylor", "RB", "IND", "TEN", "RB", "ACTIVE", 19.48, 18.04, 7),
        _mk("Brian Robinson Jr.", "RB", "ATL", "CHI", "BE", "ACTIVE", 11.22, 8.72, 21),
        _mk("Seth McGowan", "RB", "IND", "TEN", "BE", "ACTIVE", 5.02, 5.70, 49),
        _mk("Denzel Boston", "WR", "CLE", "BAL", "WR", "ACTIVE", 14.60, 13.22, 18),
        _mk("Devaughn Vele", "WR", "NO", "NYG", "WR", "ACTIVE", 12.44, 8.02, 24),
        _mk("Rashid Shaheed", "WR", "SEA", "DEN", "BE", "ACTIVE", 8.46, 12.54, 50),
        _mk("Keon Coleman", "WR", "BUF", "LV", "BE", "ACTIVE", 10.08, 6.92, 35),
        _mk("Dontayvion Wicks", "WR", "PHI", "CAR", "BE", "QUESTIONABLE", 9.80, 9.20, 40),
        _mk("Rashod Bateman", "WR", "BAL", "CLE", "BE", "ACTIVE", 9.06, 8.03, 46),
        _mk("Sam LaPorta", "TE", "DET", "BYE", "BE", "BYE", 15.32, 12.84, 3),
        _mk("Dallas Goedert", "TE", "PHI", "CAR", "WR/TE", "ACTIVE", 11.97, 10.21, 25),
        _mk("Steelers D/ST", "D/ST", "PIT", "TB", "D/ST", "ACTIVE", 8.80, 5.57, 5),
    ]

    return {
        "Eatin TDs like Groceries": eatin_roster,
        "BODHIZAFA": bodhizafa_roster,
    }

