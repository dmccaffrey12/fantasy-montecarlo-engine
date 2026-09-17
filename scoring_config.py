"""
scoring_config.py
Centralized Scoring Configuration & Valuation Engine for Idiosyncratic League Rules.

Corroborated Rules from User League Settings:
1. Passing:
   - 0.04 pts / yd (1 pt per 25 yds)
   - +0.1 pt per completed pass
   - 6 pts per passing TD (PTD)
   - +1 pt bonus for 40+ yard passing TD (PTD40)
   - +4 pt bonus for 50+ yard passing TD (PTD50) [Stacks: 50+ yd TD = +5 total bonus]
   - -1 pt per interception thrown (INT)
   - +2 pts per 2-pt passing conversion (2PC)
   - +2 pts for 300-399 yard passing game (P300)
   - +3 pts for 400+ yard passing game (P400) [Stacks: 400+ yds = +5 total bonus]

2. Rushing:
   - 0.10 pts / yd (1 pt per 10 yds)
   - 6 pts per rushing TD (RTD)
   - +1 pt bonus for 40+ yard rushing TD (RTD40)
   - +2 pt bonus for 50+ yard rushing TD (RTD50) [Stacks: 50+ yd TD = +3 total bonus]
   - +2 pts per 2-pt rushing conversion (2PR)
   - +1 pt for 100-199 yard rushing game (RY100)
   - +2 pts for 200+ yard rushing game (RY200) [Stacks: 200+ yds = +3 total bonus]

3. Receiving:
   - 1.0 Full PPR (1 pt per reception)
   - 0.10 pts / yd (1 pt per 10 yds)
   - 6 pts per receiving TD (RETD)
   - +1 pt bonus for 40+ yard receiving TD (RETD40)
   - +2 pt bonus for 50+ yard receiving TD (RETD50) [Stacks: 50+ yd TD = +3 total bonus]
   - +2 pts per 2-pt receiving conversion (2PRE)
   - +1 pt for 100-199 yard receiving game (REY100)
   - +2 pts for 200+ yard receiving game (REY200) [Stacks: 200+ yds = +3 total bonus]

4. Individual Return Yards & Miscellaneous:
   - 1 pt per 25 kickoff return yards (0.04/yd)
   - 1 pt per 25 punt return yards (0.04/yd)
   - 6 pts per return TD (KRTD / PRTD)
   - -2 pts per fumble lost (FUML)

5. Team Defense / Special Teams (D/ST):
   - Sacks: +1 pt
   - Interceptions: +2 pts
   - Fumbles Recovered: +2 pts
   - Forced Fumbles: +1 pt
   - Safeties: +2 pts
   - Blocked Kicks (Punt, PAT, FG): +2 pts
   - Defensive / ST Return TD: +6 pts
   - 2-pt Return: +2 pts, 1-pt Safety: +1 pt
   - Points Allowed Brackets:
     0 pts: +5 | 1-6 pts: +4 | 7-13 pts: +3 | 14-17 pts: +1 | 18-27 pts: 0
     28-34 pts: -1 | 35-45 pts: -3 | 46+ pts: -5
   - Yards Allowed Brackets:
     <100 yds: +5 | 100-199 yds: +3 | 200-299 yds: +2 | 300-349 yds: 0
     350-399 yds: -1 | 400-449 yds: -3 | 450-499 yds: -5 | 500-549 yds: -6 | 550+ yds: -7
   - Individual return yards do NOT apply to D/ST.

6. Kickers:
   - Excluded (not used in league).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import math
from scipy.stats import norm


@dataclass
class LeagueScoringConfig:
    """Dataclass encapsulating the user's idiosyncratic league scoring settings."""
    # Passing
    pass_yd_multiplier: float = 0.04        # 1 pt / 25 yds
    completion_bonus: float = 0.1           # 0.1 per completion
    pass_td: float = 6.0                    # 6 pts per TD
    pass_td_40_bonus: float = 1.0           # 40+ yd TD bonus
    pass_td_50_bonus: float = 4.0           # 50+ yd TD bonus (stacks with 40+ -> +5)
    pass_int: float = -1.0                  # -1 pt per INT
    pass_2pt: float = 2.0                   # 2-pt conversion
    pass_300_bonus: float = 2.0             # 300-399 yd game
    pass_400_bonus: float = 3.0             # 400+ yd game (stacks with 300+ -> +5)

    # Rushing
    rush_yd_multiplier: float = 0.10        # 1 pt / 10 yds
    rush_td: float = 6.0                    # 6 pts per TD
    rush_td_40_bonus: float = 1.0           # 40+ yd TD bonus
    rush_td_50_bonus: float = 2.0           # 50+ yd TD bonus (stacks with 40+ -> +3)
    rush_2pt: float = 2.0                   # 2-pt conversion
    rush_100_bonus: float = 1.0             # 100-199 yd game
    rush_200_bonus: float = 2.0             # 200+ yd game (stacks with 100+ -> +3)

    # Receiving
    ppr: float = 1.0                        # 1.0 Full PPR
    rec_yd_multiplier: float = 0.10         # 1 pt / 10 yds
    rec_td: float = 6.0                     # 6 pts per TD
    rec_td_40_bonus: float = 1.0            # 40+ yd TD bonus
    rec_td_50_bonus: float = 2.0            # 50+ yd TD bonus (stacks with 40+ -> +3)
    rec_2pt: float = 2.0                    # 2-pt conversion
    rec_100_bonus: float = 1.0              # 100-199 yd game
    rec_200_bonus: float = 2.0              # 200+ yd game (stacks with 100+ -> +3)

    # Return Yards & Turnovers (Skill Players)
    kr_yd_multiplier: float = 0.04          # 1 pt / 25 kickoff return yards
    pr_yd_multiplier: float = 0.04          # 1 pt / 25 punt return yards
    return_td: float = 6.0                  # 6 pts per return TD
    fumble_lost: float = -2.0               # -2 pts per fumble lost

    # D/ST Base Scoring
    dst_sack: float = 1.0
    dst_int: float = 2.0
    dst_fumble_rec: float = 2.0
    dst_fumble_forced: float = 1.0
    dst_safety: float = 2.0
    dst_blocked_kick: float = 2.0
    dst_td: float = 6.0
    dst_2pt_ret: float = 2.0
    dst_1pt_safety: float = 1.0


DEFAULT_SCORING = LeagueScoringConfig()

STANDARD_FULL_PPR_SCORING = LeagueScoringConfig(
    ppr=1.0,
    pass_td=4.0,
    completion_bonus=0.0,
    pass_td_40_bonus=0.0,
    pass_td_50_bonus=0.0,
    pass_300_bonus=0.0,
    pass_400_bonus=0.0,
    rush_td_40_bonus=0.0,
    rush_td_50_bonus=0.0,
    rush_100_bonus=0.0,
    rush_200_bonus=0.0,
    rec_td_40_bonus=0.0,
    rec_td_50_bonus=0.0,
    rec_100_bonus=0.0,
    rec_200_bonus=0.0
)

STANDARD_HALF_PPR_SCORING = LeagueScoringConfig(
    ppr=0.5,
    pass_td=4.0,
    completion_bonus=0.0,
    pass_td_40_bonus=0.0,
    pass_td_50_bonus=0.0,
    pass_300_bonus=0.0,
    pass_400_bonus=0.0,
    rush_td_40_bonus=0.0,
    rush_td_50_bonus=0.0,
    rush_100_bonus=0.0,
    rush_200_bonus=0.0,
    rec_td_40_bonus=0.0,
    rec_td_50_bonus=0.0,
    rec_100_bonus=0.0,
    rec_200_bonus=0.0
)


def get_scoring_config_by_name(name: str) -> LeagueScoringConfig:
    """Retrieve the appropriate LeagueScoringConfig preset by format name."""
    if not name:
        return DEFAULT_SCORING
    name_clean = str(name).lower()
    if "half" in name_clean:
        return STANDARD_HALF_PPR_SCORING
    elif "standard full" in name_clean or "4pt" in name_clean:
        return STANDARD_FULL_PPR_SCORING
    return DEFAULT_SCORING


def get_points_allowed_score(pa: float) -> float:
    """Return D/ST fantasy points based on Points Allowed (PA) brackets."""
    if pa <= 0:
        return 5.0
    elif pa <= 6:
        return 4.0
    elif pa <= 13:
        return 3.0
    elif pa <= 17:
        return 1.0
    elif pa <= 27:
        return 0.0
    elif pa <= 34:
        return -1.0
    elif pa <= 45:
        return -3.0
    else:
        return -5.0


def get_yards_allowed_score(ya: float) -> float:
    """Return D/ST fantasy points based on Total Yards Allowed (YA) brackets."""
    if ya < 100:
        return 5.0
    elif ya < 200:
        return 3.0
    elif ya < 300:
        return 2.0
    elif ya < 350:
        return 0.0
    elif ya < 400:
        return -1.0
    elif ya < 450:
        return -3.0
    elif ya < 500:
        return -5.0
    elif ya < 550:
        return -6.0
    else:
        return -7.0


def calculate_pass_milestone_ev(pass_yds_mean: float, sigma: float, cfg: Optional[LeagueScoringConfig] = None) -> float:
    """
    Calculate expected value of stacking 300+ and 400+ yard passing bonuses:
      E[Bonus] = 2.0 * P(Y >= 300) + 3.0 * P(Y >= 400)
    """
    cfg = cfg or DEFAULT_SCORING
    if pass_yds_mean <= 0 or sigma <= 0:
        return 0.0
    # Stacking probabilities from Normal distribution
    p_300 = 1.0 - norm.cdf(300.0, loc=pass_yds_mean, scale=sigma)
    p_400 = 1.0 - norm.cdf(400.0, loc=pass_yds_mean, scale=sigma)
    return float(cfg.pass_300_bonus * p_300 + cfg.pass_400_bonus * p_400)


def calculate_rush_milestone_ev(rush_yds_mean: float, sigma: float, cfg: Optional[LeagueScoringConfig] = None) -> float:
    """
    Calculate expected value of stacking 100+ and 200+ yard rushing bonuses:
      E[Bonus] = 1.0 * P(Y >= 100) + 2.0 * P(Y >= 200)
    """
    cfg = cfg or DEFAULT_SCORING
    if rush_yds_mean <= 0 or sigma <= 0:
        return 0.0
    p_100 = 1.0 - norm.cdf(100.0, loc=rush_yds_mean, scale=sigma)
    p_200 = 1.0 - norm.cdf(200.0, loc=rush_yds_mean, scale=sigma)
    return float(cfg.rush_100_bonus * p_100 + cfg.rush_200_bonus * p_200)


def calculate_rec_milestone_ev(rec_yds_mean: float, sigma: float, cfg: Optional[LeagueScoringConfig] = None) -> float:
    """
    Calculate expected value of stacking 100+ and 200+ yard receiving bonuses:
      E[Bonus] = 1.0 * P(Y >= 100) + 2.0 * P(Y >= 200)
    """
    cfg = cfg or DEFAULT_SCORING
    if rec_yds_mean <= 0 or sigma <= 0:
        return 0.0
    p_100 = 1.0 - norm.cdf(100.0, loc=rec_yds_mean, scale=sigma)
    p_200 = 1.0 - norm.cdf(200.0, loc=rec_yds_mean, scale=sigma)
    return float(cfg.rec_100_bonus * p_100 + cfg.rec_200_bonus * p_200)


def calculate_long_pass_td_ev(pass_tds_mean: float, cfg: Optional[LeagueScoringConfig] = None) -> float:
    """
    Expected value of 40+ and 50+ yard passing TD bonuses.
    In the NFL, ~16% of pass TDs are 40+ yards and ~8% are 50+ yards.
    Stacking bonus: 40+ yd gives +1.0, 50+ yd gives +4.0 additional (+5.0 total).
    """
    cfg = cfg or DEFAULT_SCORING
    if pass_tds_mean <= 0:
        return 0.0
    p_40 = 0.16
    p_50 = 0.08
    return float(pass_tds_mean * (p_40 * cfg.pass_td_40_bonus + p_50 * cfg.pass_td_50_bonus))


def calculate_long_rush_rec_td_ev(tds_mean: float, is_rec: bool, cfg: Optional[LeagueScoringConfig] = None) -> float:
    """
    Expected value of 40+ and 50+ yard rush/rec TD bonuses.
    For rushing, ~6% are 40+ yds, ~3% are 50+ yds.
    For receiving, ~14% are 40+ yds, ~7% are 50+ yds.
    Stacking bonus: +1.0 for 40+, +2.0 additional for 50+ (+3.0 total).
    """
    cfg = cfg or DEFAULT_SCORING
    if tds_mean <= 0:
        return 0.0
    p_40 = 0.14 if is_rec else 0.06
    p_50 = 0.07 if is_rec else 0.03
    b40 = cfg.rec_td_40_bonus if is_rec else cfg.rush_td_40_bonus
    b50 = cfg.rec_td_50_bonus if is_rec else cfg.rush_td_50_bonus
    return float(tds_mean * (p_40 * b40 + p_50 * b50))


def calculate_player_mu_and_sigma(
    position: str,
    props: Optional[Dict[str, Any]] = None,
    espn_proj: float = 12.0,
    scoring_config: Optional[LeagueScoringConfig] = None,
    polymarket_prob: float = 1.0,
    is_inactive: bool = False
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Synthesize distribution parameters (mu, sigma, p_active) and a detailed breakdown
    using live Vegas player props or calibrated baseline decomposition under the user's
    idiosyncratic league scoring system.
    """
    cfg = scoring_config or DEFAULT_SCORING

    # Inactive players immediately return 0.0
    if is_inactive or polymarket_prob <= 0.0:
        return 0.0, 0.0, 0.0, {"source": "Inactive", "notes": "Ruled OUT / IR (0.0 pts)"}

    pos = (position or "WR").upper().strip()
    props = props or {}

    # Extract props if present
    pass_yds = float(props.get("pass_yds", 0.0) or 0.0)
    pass_completions = float(props.get("pass_completions", 0.0) or 0.0)
    pass_tds = float(props.get("pass_tds", 0.0) or 0.0)
    pass_ints = float(props.get("pass_interceptions", 0.0) or 0.0)
    rush_yds = float(props.get("rush_yds", 0.0) or 0.0)
    rush_td_prob = float(props.get("anytime_td_prob", 0.0) or 0.0)
    receptions = float(props.get("receptions", 0.0) or 0.0)
    rec_yds = float(props.get("rec_yds", 0.0) or 0.0)

    has_vegas_props = (pass_yds > 0 or rush_yds > 0 or receptions > 0 or rec_yds > 0 or rush_td_prob > 0)
    breakdown: Dict[str, Any] = {"source": "Vegas Consensus Props" if has_vegas_props else "Idiosyncratic Baseline"}

    if pos == "QB":
        if has_vegas_props and pass_yds > 0:
            comps = pass_completions if pass_completions > 0 else round(pass_yds / 11.2, 1)
            tds = pass_tds if pass_tds > 0 else max(0.8, round(pass_yds / 160.0, 2))
            ints = pass_ints if pass_ints > 0 else 0.70
            r_yds = rush_yds if rush_yds > 0 else 12.0
            r_tds = rush_td_prob * 0.4 if rush_td_prob > 0 else 0.15
        else:
            # Calibrate from ESPN standard projection (standard QB: 250 yds, 1.6 TDs, 22 comps, 15 rush yds, 0.8 INTs)
            base = max(8.0, espn_proj)
            scale = base / 17.5  # standard QB avg ~ 17.5 in standard 4pt format
            pass_yds = round(240.0 * scale, 1)
            comps = round(21.5 * scale, 1)
            tds = round(1.55 * scale, 2)
            ints = round(0.75 * scale, 2)
            r_yds = round(15.0 * scale, 1)
            r_tds = round(0.15 * scale, 2)

        # Idiosyncratic Passing Calculation:
        pts_pass_yds = pass_yds * cfg.pass_yd_multiplier
        pts_comps = comps * cfg.completion_bonus
        pts_pass_tds = tds * cfg.pass_td
        pts_rush_yds = r_yds * cfg.rush_yd_multiplier
        pts_rush_tds = r_tds * cfg.rush_td
        pts_ints = ints * cfg.pass_int  # cfg.pass_int is -1.0

        # Milestone & Long TD stacking bonuses
        sigma_yds = max(30.0, pass_yds * 0.18)
        ev_milestones = calculate_pass_milestone_ev(pass_yds, sigma_yds, cfg)
        ev_long_tds = calculate_long_pass_td_ev(tds, cfg)

        mu = pts_pass_yds + pts_comps + pts_pass_tds + pts_rush_yds + pts_rush_tds + pts_ints + ev_milestones + ev_long_tds
        sigma = max(3.5, mu * 0.25 + (ev_long_tds * 0.5))

        breakdown.update({
            "pass_yds": round(pass_yds, 1),
            "completions": round(comps, 1),
            "pass_tds": round(tds, 2),
            "rush_yds": round(r_yds, 1),
            "ints": round(ints, 2),
            "pts_pass_yds": round(pts_pass_yds, 2),
            "pts_completions": round(pts_comps, 2),
            "pts_pass_tds": round(pts_pass_tds, 2),
            "pts_rush_yds": round(pts_rush_yds, 2),
            "pts_rush_tds": round(pts_rush_tds, 2),
            "pts_ints": round(pts_ints, 2),
            "ev_milestones": round(ev_milestones, 2),
            "ev_long_tds": round(ev_long_tds, 2)
        })

    elif pos == "RB":
        if has_vegas_props and (rush_yds > 0 or receptions > 0 or rush_td_prob > 0):
            r_yds = rush_yds if rush_yds > 0 else 45.0
            recs = receptions if receptions > 0 else 2.2
            v_yds = rec_yds if rec_yds > 0 else (recs * 7.5)
            td_prob = rush_td_prob if rush_td_prob > 0 else 0.40
        else:
            base = max(4.0, espn_proj)
            scale = base / 13.0
            r_yds = round(58.0 * scale, 1)
            recs = round(2.8 * scale, 1)
            v_yds = round(21.0 * scale, 1)
            td_prob = min(0.95, round(0.48 * scale, 2))

        pts_rush_yds = r_yds * cfg.rush_yd_multiplier
        pts_recs = recs * cfg.ppr
        pts_rec_yds = v_yds * cfg.rec_yd_multiplier
        pts_tds = td_prob * cfg.rush_td

        # Milestone & Long TD stacking bonuses
        sigma_rush = max(18.0, r_yds * 0.35)
        ev_milestones = calculate_rush_milestone_ev(r_yds, sigma_rush, cfg)
        ev_long_tds = calculate_long_rush_rec_td_ev(td_prob, is_rec=False, cfg=cfg)

        mu = pts_rush_yds + pts_recs + pts_rec_yds + pts_tds + ev_milestones + ev_long_tds
        sigma = max(2.5, mu * 0.38 + (ev_long_tds * 0.4))

        breakdown.update({
            "rush_yds": round(r_yds, 1),
            "receptions": round(recs, 1),
            "rec_yds": round(v_yds, 1),
            "td_prob": round(td_prob, 2),
            "pts_rush_yds": round(pts_rush_yds, 2),
            "pts_recs": round(pts_recs, 2),
            "pts_rec_yds": round(pts_rec_yds, 2),
            "pts_tds": round(pts_tds, 2),
            "ev_milestones": round(ev_milestones, 2),
            "ev_long_tds": round(ev_long_tds, 2)
        })

    elif pos in ["WR", "TE"]:
        is_te = (pos == "TE")
        if has_vegas_props and (receptions > 0 or rec_yds > 0 or rush_td_prob > 0):
            recs = receptions if receptions > 0 else 4.2
            v_yds = rec_yds if rec_yds > 0 else (recs * (9.5 if is_te else 12.0))
            r_yds = rush_yds if rush_yds > 0 else 0.0
            td_prob = rush_td_prob if rush_td_prob > 0 else (0.30 if is_te else 0.38)
        else:
            base = max(3.0, espn_proj)
            bench = 10.5 if is_te else 13.0
            scale = base / bench
            recs = round((3.8 if is_te else 4.5) * scale, 1)
            v_yds = round((42.0 if is_te else 58.0) * scale, 1)
            r_yds = round(2.0 * scale if not is_te else 0.0, 1)
            td_prob = min(0.95, round((0.32 if is_te else 0.42) * scale, 2))

        pts_recs = recs * cfg.ppr
        pts_rec_yds = v_yds * cfg.rec_yd_multiplier
        pts_rush_yds = r_yds * cfg.rush_yd_multiplier
        pts_tds = td_prob * cfg.rec_td

        # Milestone & Long TD stacking bonuses
        sigma_rec = max(18.0, v_yds * 0.40)
        ev_milestones = calculate_rec_milestone_ev(v_yds, sigma_rec, cfg)
        ev_long_tds = calculate_long_rush_rec_td_ev(td_prob, is_rec=True, cfg=cfg)

        mu = pts_recs + pts_rec_yds + pts_rush_yds + pts_tds + ev_milestones + ev_long_tds
        sigma = max(2.5, mu * 0.45 + (ev_long_tds * 0.4))

        breakdown.update({
            "receptions": round(recs, 1),
            "rec_yds": round(v_yds, 1),
            "rush_yds": round(r_yds, 1),
            "td_prob": round(td_prob, 2),
            "pts_recs": round(pts_recs, 2),
            "pts_rec_yds": round(pts_rec_yds, 2),
            "pts_rush_yds": round(pts_rush_yds, 2),
            "pts_tds": round(pts_tds, 2),
            "ev_milestones": round(ev_milestones, 2),
            "ev_long_tds": round(ev_long_tds, 2)
        })

    elif pos in ["DEF", "D/ST"]:
        # D/ST scoring from opponent implied points & yards
        opp_implied_pts = float(props.get("opp_implied_pts", 22.0) or 22.0)
        opp_implied_yds = float(props.get("opp_implied_yds", 330.0) or 330.0)

        pts_pa = get_points_allowed_score(opp_implied_pts)
        pts_ya = get_yards_allowed_score(opp_implied_yds)
        exp_sacks = float(props.get("sacks", 2.6))
        exp_turnovers = float(props.get("turnovers", 1.3))
        exp_forced_fumbles = float(props.get("forced_fumbles", 0.9))

        pts_sacks = exp_sacks * cfg.dst_sack
        pts_turnovers = exp_turnovers * cfg.dst_int
        pts_ff = exp_forced_fumbles * cfg.dst_fumble_forced

        mu = max(1.0, pts_pa + pts_ya + pts_sacks + pts_turnovers + pts_ff)
        sigma = max(2.5, mu * 0.42)

        breakdown.update({
            "opp_implied_pts": opp_implied_pts,
            "opp_implied_yds": opp_implied_yds,
            "pts_pa": pts_pa,
            "pts_ya": pts_ya,
            "pts_sacks": round(pts_sacks, 1),
            "pts_turnovers": round(pts_turnovers, 1),
            "pts_ff": round(pts_ff, 1)
        })

    else:
        # Default fallback
        mu = max(3.0, espn_proj)
        sigma = max(2.0, mu * 0.35)

    p_active = max(0.0, min(1.0, float(polymarket_prob)))
    return round(mu, 2), round(sigma, 2), round(p_active, 3), breakdown
