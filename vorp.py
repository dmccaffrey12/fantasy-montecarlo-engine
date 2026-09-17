"""
vorp.py
Dynamic Weekly Value Over Replacement Player (VORP) Engine.
- Establishes dynamic weekly baselines from the top unowned waiver wire free agents
  for QB, RB, WR, TE, and FLEX.
- Calculates traditional Expected Points VORP: E[Player] - E[Baseline].
- Calculates Distributional Win-Rate VORP (Delta P(Win)):
  The marginal win probability gained by starting this player over the waiver wire replacement.
- Ranks waiver wire targets dynamically by win-rate leverage and ceiling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from simulation import sample_player, simulate_matchup, summarize_distribution


def calculate_waiver_baselines(
    waiver_pool: List[Dict[str, Any]],
    n_simulations: int = 10000
) -> Dict[str, Any]:
    """
    Determine the top unowned waiver wire baseline player for each position:
    QB, RB, WR, TE, and FLEX.

    The flex baseline is the highest projected remaining player among RB/WR/TE
    after the primary positional baselines are accounted for.
    """
    # 1. Simulate all waiver players
    simulated_waivers = []
    for player in waiver_pool:
        pos = player.get("position", "WR").upper()
        mu = float(player.get("mu", 8.0))
        sigma = float(player.get("sigma", 4.0))
        p_active = float(player.get("p_active", 1.0))

        sim = sample_player(pos, mu, sigma, n=n_simulations, p_active=p_active)
        summary = summarize_distribution(sim)

        simulated_waivers.append({
            **player,
            "sim": sim,
            "mean": summary["mean"],
            "median": summary["median"],
            "p10": summary["p10"],
            "p90": summary["p90"]
        })

    # 2. Find top player for each primary position
    baselines = {}
    used_player_names = set()

    for pos in ["QB", "RB", "WR", "TE"]:
        pos_players = [p for p in simulated_waivers if p["position"] == pos]
        if pos_players:
            top_p = max(pos_players, key=lambda x: x["mean"])
            baselines[pos] = {
                "player_name": top_p["name"],
                "position": pos,
                "team": top_p.get("team", "FA"),
                "baseline_mean": round(top_p["mean"], 2),
                "baseline_p10": round(top_p["p10"], 2),
                "baseline_p90": round(top_p["p90"], 2),
                "sim": top_p["sim"]
            }
            used_player_names.add(top_p["name"])
        else:
            # Fallback default baseline
            fallback_mu = 14.0 if pos == "QB" else (9.0 if pos in ["RB", "WR"] else 7.0)
            fallback_sim = sample_player(pos, fallback_mu, fallback_mu * 0.4, n=n_simulations)
            baselines[pos] = {
                "player_name": f"Generic Waiver {pos}",
                "position": pos,
                "team": "FA",
                "baseline_mean": fallback_mu,
                "baseline_p10": float(np.percentile(fallback_sim, 10)),
                "baseline_p90": float(np.percentile(fallback_sim, 90)),
                "sim": fallback_sim
            }

    # 3. Find FLEX baseline (top remaining RB/WR/TE not used as primary baseline)
    flex_candidates = [
        p for p in simulated_waivers
        if p["position"] in ["RB", "WR", "TE"] and p["name"] not in used_player_names
    ]

    if flex_candidates:
        top_flex = max(flex_candidates, key=lambda x: x["mean"])
        baselines["FLEX"] = {
            "player_name": top_flex["name"],
            "position": top_flex["position"],
            "team": top_flex.get("team", "FA"),
            "baseline_mean": round(top_flex["mean"], 2),
            "baseline_p10": round(top_flex["p10"], 2),
            "baseline_p90": round(top_flex["p90"], 2),
            "sim": top_flex["sim"]
        }
    else:
        # Fallback to secondary flex baseline
        fallback_flex = baselines.get("WR", baselines["RB"])
        baselines["FLEX"] = {
            "player_name": f"Secondary {fallback_flex['player_name']}",
            "position": "FLEX",
            "team": "FA",
            "baseline_mean": round(fallback_flex["baseline_mean"] * 0.85, 2),
            "baseline_p10": round(fallback_flex["baseline_p10"] * 0.85, 2),
            "baseline_p90": round(fallback_flex["baseline_p90"] * 0.85, 2),
            "sim": fallback_flex["sim"] * 0.85
        }

    return baselines


def calculate_roster_vorp(
    active_roster: List[Dict[str, Any]],
    baselines: Dict[str, Any],
    opponent_roster: Optional[List[Dict[str, Any]]] = None,
    n_simulations: int = 10000
) -> pd.DataFrame:
    """
    Evaluate active roster players against their positional and flex baselines.
    Returns a DataFrame with:
      - Player Name, Position, Slot
      - Projected Score E[Player], Floor (P10), Ceiling (P90)
      - Replacement Player, Baseline Score
      - Points VORP
      - Win-Rate VORP (Delta P(Win)) if opponent_roster is provided
    """
    # Baseline matchup win rate if opponent provided
    base_matchup = None
    if opponent_roster:
        base_matchup = simulate_matchup(active_roster, opponent_roster, n_simulations=n_simulations)
        team_win_rate = base_matchup["p_win_a"]
    else:
        team_win_rate = 0.5

    records = []
    for idx, player in enumerate(active_roster):
        name = player.get("name", "Unknown")
        pos = player.get("position", "WR").upper()
        slot = player.get("slot", pos).upper()
        mu = float(player.get("mu", 12.0))
        sigma = float(player.get("sigma", 5.0))
        p_active = float(player.get("p_active", 1.0))

        # Sample player
        p_sim = sample_player(pos, mu, sigma, n=n_simulations, p_active=p_active)
        summary = summarize_distribution(p_sim)

        # Determine relevant replacement baseline
        if "FLEX" in slot:
            baseline_info = baselines.get("FLEX", baselines.get(pos, {}))
        else:
            baseline_info = baselines.get(pos, baselines.get("FLEX", {}))

        baseline_mean = baseline_info.get("baseline_mean", 8.0)
        baseline_name = baseline_info.get("player_name", "Waiver Baseline")

        # 1. Expected Points VORP
        pts_vorp = round(summary["mean"] - baseline_mean, 2)

        # 2. Win-Rate VORP (Delta P(Win))
        delta_p_win = 0.0
        if opponent_roster and base_matchup:
            # Create counterfactual roster replacing this player with waiver baseline
            replacement_player = {
                "name": baseline_name,
                "position": baseline_info.get("position", pos),
                "slot": slot,
                "mu": baseline_mean,
                "sigma": baseline_mean * 0.4,
                "p_active": 1.0
            }
            counterfactual_roster = [p for i, p in enumerate(active_roster) if i != idx] + [replacement_player]
            cf_matchup = simulate_matchup(counterfactual_roster, opponent_roster, n_simulations=n_simulations)
            cf_win_rate = cf_matchup["p_win_a"]
            delta_p_win = round((team_win_rate - cf_win_rate) * 100.0, 2)

        records.append({
            "Player": name,
            "Pos": pos,
            "Slot": slot,
            "Projected (Med)": round(summary["median"], 1),
            "Proj Pts": round(summary["mean"], 1),
            "Floor (P10)": round(summary["p10"], 1),
            "Ceiling (P90)": round(summary["p90"], 1),
            "Waiver Baseline": f"{baseline_name} ({baseline_mean:.1f})",
            "Points VORP": pts_vorp,
            "Win% VORP (Delta P(Win))": f"{delta_p_win:+0.1f}%" if opponent_roster else "N/A"
        })

    df = pd.DataFrame(records)
    return df.sort_values(by="Points VORP", ascending=False).reset_index(drop=True)


def calculate_outperformance_score(
    sim_samples: np.ndarray,
    threshold: float = 12.0
) -> float:
    """
    Calculate Outperformance Score (OS) for streaming targets:
    OS = P(Score >= 12.0 points)
    """
    return float(np.mean(sim_samples >= threshold) * 100.0)


def calculate_contingency_leverage(
    players_pool: List[Dict[str, Any]],
    n_simulations: int = 10000
) -> pd.DataFrame:
    """
    Calculate Contingency Leverage Index for backup running backs (handcuffs):
      Contingency Index = P(Starter Injury) * Conditional RB1 Simulated Output.
    Ranks bench assets by their 90th percentile contingency upside rather than standalone median.
    """
    records = []
    # Known lead backs vs backup handcuffs mapping / hazard rates
    handcuff_hazard_rates = {
        "Jerome Ford": 0.20,
        "Jaylen Warren": 0.22,
        "Rico Dowdle": 0.25,
        "Isaac Guerendo": 0.25,
        "Braelon Allen": 0.18,
        "Tyler Allgeier": 0.18,
        "Blake Corum": 0.18,
        "Ray Davis": 0.15,
        "Zach Charbonnet": 0.20,
        "Chuba Hubbard": 0.22
    }

    for player in players_pool:
        pos = player.get("position", "WR").upper()
        if pos != "RB":
            continue

        name = player.get("name", "Unknown")
        team = player.get("team", "FA")
        standalone_mu = float(player.get("mu", 8.0))
        standalone_sigma = float(player.get("sigma", 4.0))

        # Check if player is a backup / rotational back (standalone mu <= 13.0)
        is_handcuff = standalone_mu <= 13.5 or name in handcuff_hazard_rates

        if is_handcuff:
            # Baseline injury hazard rate of starter
            p_starter_injury = handcuff_hazard_rates.get(name, 0.18)

            # Conditional RB1 workload: inherits bellcow role (~15.8 projected points, 6.2 sigma)
            conditional_mu = max(14.5, standalone_mu * 1.55)
            conditional_sigma = max(5.5, conditional_mu * 0.40)

            cond_sim = sample_player("RB", conditional_mu, conditional_sigma, n=n_simulations)
            cond_summary = summarize_distribution(cond_sim)

            # Contingency Leverage Index = P(Injury) * Conditional E[Score]
            contingency_index = round(p_starter_injury * cond_summary["mean"], 2)

            # Standalone sim
            standalone_sim = sample_player("RB", standalone_mu, standalone_sigma, n=n_simulations)
            stand_summary = summarize_distribution(standalone_sim)

            records.append({
                "Player": name,
                "Team": team,
                "Standalone Med": round(stand_summary["median"], 1),
                "Starter Inj. Hazard": f"{p_starter_injury * 100:.0f}%",
                "Conditional RB1 Pts": round(cond_summary["mean"], 1),
                "Contingency P90 Ceiling": round(cond_summary["p90"], 1),
                "Contingency Index": contingency_index
            })

    df = pd.DataFrame(records)
    if not df.empty:
        return df.sort_values(by="Contingency Index", ascending=False).reset_index(drop=True)
    return pd.DataFrame(columns=["Player", "Team", "Standalone Med", "Starter Inj. Hazard", "Conditional RB1 Pts", "Contingency P90 Ceiling", "Contingency Index"])


def calculate_suggested_faab(
    net_pts_gain: float,
    win_leverage: float = 0.0,
    remaining_faab: int = 100,
    priority_rank: int = 1,
    is_stash: bool = False,
    contingency_ceiling: float = 0.0
) -> Tuple[int, int, str]:
    """
    Dynamically size FAAB bid based on marginal win% leverage, net projected points gain,
    or 90th-percentile contingent ceiling, proportioned to remaining team FAAB.
    Returns: (faab_dollar, faab_pct, faab_display_str)
    e.g. (14, 14, "$14 (14%)")
    """
    rem = max(1, int(remaining_faab))

    if is_stash:
        # For contingent stashes, size based on P90 ceiling upside
        if contingency_ceiling >= 24.0:
            bid_pct = 0.16
        elif contingency_ceiling >= 20.0:
            bid_pct = 0.11
        elif contingency_ceiling >= 16.0:
            bid_pct = 0.07
        else:
            bid_pct = 0.04
    else:
        # Standard streamer / starter upgrade based on win leverage and net pts
        if win_leverage >= 0.05 or net_pts_gain >= 4.0:
            bid_pct = 0.22 - min(0.06, (priority_rank - 1) * 0.03)
        elif win_leverage >= 0.03 or net_pts_gain >= 2.5:
            bid_pct = 0.14 - min(0.04, (priority_rank - 1) * 0.02)
        elif win_leverage >= 0.015 or net_pts_gain >= 1.2:
            bid_pct = 0.08 - min(0.03, (priority_rank - 1) * 0.01)
        elif net_pts_gain > 0.3:
            bid_pct = 0.04
        else:
            bid_pct = 0.02

    bid_pct = max(0.01, min(0.40, bid_pct))
    faab_bid = max(1, int(round(rem * bid_pct)))
    faab_pct_int = int(round((faab_bid / rem) * 100))
    display_str = f"${faab_bid} ({faab_pct_int}%)"
    return faab_bid, faab_pct_int, display_str


def rank_waiver_wire_pool(
    waiver_pool: List[Dict[str, Any]],
    baselines: Dict[str, Any],
    n_simulations: int = 10000,
    roster_qbs: Optional[List[Dict[str, Any]]] = None,
    mode: str = "streamer"
) -> pd.DataFrame:
    """
    Rank dynamic waiver wire pickups by their expected fantasy points,
    upside ceiling relative to the positional replacement baseline,
    and High-Probability Streamer Outperformance Score (OS = P(Score >= 12.0)).
    Also annotates 2-QB league roster context against rostered quarterbacks.
    Supports mode='streamer' vs mode='stash'.
    """
    records = []
    min_qb = min(roster_qbs, key=lambda q: float(q.get("mu", 0.0))) if roster_qbs else None

    # Known contingent hazard rates for stashes
    handcuff_rates = {
        "Jerome Ford": 0.20,
        "Jaylen Warren": 0.22,
        "Rico Dowdle": 0.25,
        "Isaac Guerendo": 0.25,
        "Braelon Allen": 0.18,
        "Tyler Allgeier": 0.18,
        "Blake Corum": 0.18,
        "Ray Davis": 0.15,
        "Zach Charbonnet": 0.20,
        "Chuba Hubbard": 0.22
    }

    for player in waiver_pool:
        name = player.get("name", "Unknown")
        pos = player.get("position", "WR").upper()
        team = player.get("team", "FA")
        mu = float(player.get("mu", 8.0))
        sigma = float(player.get("sigma", 4.0))
        p_active = float(player.get("p_active", 1.0))

        if mode == "stash" and (pos == "RB" or name in handcuff_rates):
            # Evaluate under contingent injury scenario
            p_haz = handcuff_rates.get(name, 0.18)
            cond_mu = max(14.5, mu * 1.55)
            cond_sig = max(5.5, cond_mu * 0.40)
            sim = sample_player(pos, cond_mu, cond_sig, n=n_simulations, p_active=1.0)
            summary = summarize_distribution(sim)
            streamer_os = calculate_outperformance_score(sim, threshold=14.0)
        else:
            sim = sample_player(pos, mu, sigma, n=n_simulations, p_active=p_active)
            summary = summarize_distribution(sim)
            streamer_os = calculate_outperformance_score(sim, threshold=12.0)

        # Compare against baseline
        baseline_info = baselines.get(pos, baselines.get("FLEX", {}))
        baseline_mean = baseline_info.get("baseline_mean", 8.0)
        net_val = round(summary["mean"] - baseline_mean, 2)

        # 2-QB League Context
        if pos == "QB" and min_qb:
            diff_qb = summary["mean"] - float(min_qb.get("mu", 0.0))
            qb_context = f"{diff_qb:+.1f} vs {min_qb.get('name')}"
        elif pos == "QB":
            qb_context = "2-QB Limit Evaluated"
        else:
            qb_context = "Eligible"

        records.append({
            "Player": name,
            "Pos": pos,
            "Team": team,
            "Projected (Med)": round(summary["median"], 1),
            "Proj Pts": round(summary["mean"], 1),
            "Floor (P10)": round(summary["p10"], 1),
            "Ceiling (P90)": round(summary["p90"], 1),
            "Streamer OS (P≥12)": f"{streamer_os:.1f}%",
            "Pos Baseline": baseline_mean,
            "Net Over Baseline": net_val,
            "2-QB Context": qb_context,
            "P(Active)": f"{p_active * 100:.0f}%"
        })

    df = pd.DataFrame(records)
    if mode == "stash":
        return df.sort_values(by="Ceiling (P90)", ascending=False).reset_index(drop=True)
    return df.sort_values(by="Proj Pts", ascending=False).reset_index(drop=True)


def calculate_waiver_roster_upgrades(
    user_bench: List[Dict[str, Any]],
    waiver_pool: List[Dict[str, Any]],
    n_simulations: int = 5000,
    full_roster: Optional[List[Dict[str, Any]]] = None,
    max_roster_qbs: int = 2,
    remaining_faab: int = 100,
    mode: str = "streamer"
) -> pd.DataFrame:
    """
    Evaluates top available waiver wire free agents against the user's current bench and roster.
    Identifies clear net-positive Add/Drop transactions formatted as Atomic Transaction Pairs:
      "[Priority #] Add [Target] ➔ Drop [Worst Bench Asset] | Net Gain | Suggested FAAB: $X (Y%)"

    Supports:
      - mode='streamer': 1-Week Spot Streamers (prioritizes immediate median projected points).
      - mode='stash': High-Upside Contingent Stashes (ranks by 90th-percentile injury contingency).
      - Sizing suggested FAAB bid dynamically based on marginal leverage and remaining team FAAB.

    Specialized 2-QB League Roster Constraint:
      - The league permits a maximum of 2 quarterbacks rostered total.
      - Any waiver recommendation for QB is compared EXCLUSIVELY against the 2 quarterbacks
        on the roster, and NO ONE ELSE! (Never recommend dropping a WR, RB, or TE for a QB).
      - If adding a waiver QB, the Cut Candidate must be one of the rostered QBs.
      - For non-QB waiver adds (RB/WR/TE), cut candidates are selected strictly from non-QB bench players.
    """
    if not waiver_pool or (not user_bench and not full_roster):
        return pd.DataFrame()

    # 1. Identify and evaluate all rostered QBs
    if full_roster is None:
        try:
            import data_ingest
            default_data = data_ingest.get_default_matchup_and_waivers()
            full_roster = default_data.get("team_a_roster", [])
        except Exception:
            full_roster = []

    combined_roster = list(full_roster or []) + list(user_bench or [])
    seen_names = set()
    dedup_roster = []
    for p in combined_roster:
        p_name = p.get("name")
        if p_name and p_name not in seen_names:
            seen_names.add(p_name)
            dedup_roster.append(p)

    roster_qbs = [p for p in dedup_roster if p.get("position", "").upper() == "QB"]

    evaluated_qbs = []
    for qb in roster_qbs:
        q_pos = "QB"
        q_mu = float(qb.get("mu", 16.0))
        q_sig = float(qb.get("sigma", 4.8))
        q_act = float(qb.get("p_active", 1.0))
        sim_q = sample_player(q_pos, q_mu, q_sig, n=n_simulations, p_active=q_act)
        summ_q = summarize_distribution(sim_q)
        evaluated_qbs.append({
            **qb,
            "mean": summ_q["mean"],
            "median": summ_q["median"],
            "p10": summ_q["p10"],
            "p90": summ_q["p90"]
        })
    # Sort rostered QBs ascending by projected mean (lowest projected rostered QB first)
    evaluated_qbs.sort(key=lambda x: x["mean"])

    # 2. Calculate distributions for user bench
    bench_evaluated = []
    for b in user_bench:
        pos = b.get("position", "WR").upper()
        mu = float(b.get("mu", 8.0))
        sigma = float(b.get("sigma", 4.0))
        p_act = float(b.get("p_active", 1.0))
        sim = sample_player(pos, mu, sigma, n=n_simulations, p_active=p_act)
        summ = summarize_distribution(sim)
        bench_evaluated.append({
            **b,
            "mean": summ["mean"],
            "median": summ["median"],
            "p10": summ["p10"],
            "p90": summ["p90"]
        })

    # Sort bench by mean ascending (lowest projected players are primary drop candidates)
    bench_evaluated.sort(key=lambda x: x["mean"])
    non_qb_bench = [b for b in bench_evaluated if b.get("position", "").upper() != "QB"]

    # Known contingent hazard rates for stashes
    handcuff_rates = {
        "Jerome Ford": 0.20,
        "Jaylen Warren": 0.22,
        "Rico Dowdle": 0.25,
        "Isaac Guerendo": 0.25,
        "Braelon Allen": 0.18,
        "Tyler Allgeier": 0.18,
        "Blake Corum": 0.18,
        "Ray Davis": 0.15,
        "Zach Charbonnet": 0.20,
        "Chuba Hubbard": 0.22
    }

    # 3. Evaluate waiver targets
    upgrades = []
    is_stash_mode = (mode == "stash")

    for w in waiver_pool:
        w_pos = w.get("position", "WR").upper()
        w_mu = float(w.get("mu", 8.0))
        w_sig = float(w.get("sigma", 4.0))
        w_act = float(w.get("p_active", 1.0))
        w_name = w.get("name", "Unknown")

        if is_stash_mode and (w_pos == "RB" or w_name in handcuff_rates):
            # Model high-upside injury contingency: bellcow upside if starter is sidelined
            cond_mu = max(14.8, w_mu * 1.55)
            cond_sig = max(5.5, cond_mu * 0.40)
            sim_w = sample_player(w_pos, cond_mu, cond_sig, n=n_simulations, p_active=1.0)
            summ_w = summarize_distribution(sim_w)
            os_score = calculate_outperformance_score(sim_w, threshold=14.0)
        else:
            sim_w = sample_player(w_pos, w_mu, w_sig, n=n_simulations, p_active=w_act)
            summ_w = summarize_distribution(sim_w)
            os_score = calculate_outperformance_score(sim_w, threshold=12.0)

        if w_pos == "QB":
            # 2-QB League Rule Enforcement:
            # Waiver QBs MUST be compared against the 2 rostered QBs, and NO ONE ELSE.
            if len(evaluated_qbs) >= max_roster_qbs:
                drop_cand = evaluated_qbs[0]  # Lowest projected rostered QB
                delta_mu = summ_w["mean"] - drop_cand["mean"]
                delta_p90 = summ_w["p90"] - drop_cand["p90"]

                if delta_mu > 0.5 or delta_p90 > 2.0:
                    priority = "HIGH" if delta_mu >= 3.0 else ("MEDIUM" if delta_mu >= 1.5 else "SPECULATIVE")
                    rationale = (
                        f"Direct 2-QB League Upgrade over {drop_cand['name']} (QB) "
                        f"(+{delta_mu:.1f} pts, +{delta_p90:.1f} ceiling). Enforces 2-QB roster maximum."
                    )
                    upgrades.append({
                        "Target Waiver Add": w_name,
                        "Pos": "QB",
                        "Team": w.get("team", "FA"),
                        "Add Proj": round(summ_w["mean"], 1),
                        "Add P10": round(summ_w["p10"], 1),
                        "Add P90": round(summ_w["p90"], 1),
                        "Projected (Med)": round(summ_w["median"], 1),
                        "Floor (P10)": round(summ_w["p10"], 1),
                        "Ceiling (P90)": round(summ_w["p90"], 1),
                        "Streamer OS%": f"{os_score:.1f}%",
                        "Cut Candidate": drop_cand["name"],
                        "Cut Pos": "QB",
                        "Cut Proj": round(drop_cand["mean"], 1),
                        "Net Proj Gain": round(delta_mu, 1),
                        "Net P90 Gain": round(delta_p90, 1),
                        "Priority": priority,
                        "Upgrade Rationale": rationale,
                        "is_stash": False
                    })
                continue

            elif len(evaluated_qbs) < max_roster_qbs:
                if not non_qb_bench:
                    continue
                drop_cand = non_qb_bench[0]
                delta_mu = summ_w["mean"] - drop_cand["mean"]
                delta_p90 = summ_w["p90"] - drop_cand["p90"]
                if delta_mu > 1.0 or delta_p90 > 3.0:
                    priority = "HIGH" if delta_mu >= 3.5 else ("MEDIUM" if delta_mu >= 1.8 else "SPECULATIVE")
                    rationale = f"Fill 2nd QB roster spot over {drop_cand['name']} ({drop_cand['position']}) (+{delta_mu:.1f} pts)"
                    upgrades.append({
                        "Target Waiver Add": w_name,
                        "Pos": "QB",
                        "Team": w.get("team", "FA"),
                        "Add Proj": round(summ_w["mean"], 1),
                        "Add P10": round(summ_w["p10"], 1),
                        "Add P90": round(summ_w["p90"], 1),
                        "Projected (Med)": round(summ_w["median"], 1),
                        "Floor (P10)": round(summ_w["p10"], 1),
                        "Ceiling (P90)": round(summ_w["p90"], 1),
                        "Streamer OS%": f"{os_score:.1f}%",
                        "Cut Candidate": drop_cand["name"],
                        "Cut Pos": drop_cand["position"],
                        "Cut Proj": round(drop_cand["mean"], 1),
                        "Net Proj Gain": round(delta_mu, 1),
                        "Net P90 Gain": round(delta_p90, 1),
                        "Priority": priority,
                        "Upgrade Rationale": rationale,
                        "is_stash": False
                    })
                continue

        else:
            # Non-QB waiver targets (RB, WR, TE):
            if not non_qb_bench:
                continue

            if is_stash_mode:
                # In Stash mode, sort drop candidates by P90 ceiling ascending (drop lowest ceiling bench asset)
                bench_by_ceiling = sorted(non_qb_bench, key=lambda x: x["p90"])
                drop_cand = bench_by_ceiling[0]
                delta_mu = summ_w["mean"] - drop_cand["mean"]
                delta_p90 = summ_w["p90"] - drop_cand["p90"]

                if delta_p90 > 2.0 or summ_w["p90"] >= 18.0:
                    priority = "HIGH" if delta_p90 >= 6.0 else ("MEDIUM" if delta_p90 >= 3.0 else "SPECULATIVE")
                    rationale = f"High-Upside Contingent Stash over {drop_cand['name']} (P90 Ceiling: {summ_w['p90']:.1f} vs {drop_cand['p90']:.1f} pts)"
                    upgrades.append({
                        "Target Waiver Add": w_name,
                        "Pos": w_pos,
                        "Team": w.get("team", "FA"),
                        "Add Proj": round(summ_w["mean"], 1),
                        "Add P10": round(summ_w["p10"], 1),
                        "Add P90": round(summ_w["p90"], 1),
                        "Projected (Med)": round(summ_w["median"], 1),
                        "Floor (P10)": round(summ_w["p10"], 1),
                        "Ceiling (P90)": round(summ_w["p90"], 1),
                        "Streamer OS%": f"{os_score:.1f}%",
                        "Cut Candidate": drop_cand["name"],
                        "Cut Pos": drop_cand["position"],
                        "Cut Proj": round(drop_cand["mean"], 1),
                        "Net Proj Gain": round(delta_mu, 1),
                        "Net P90 Gain": round(delta_p90, 1),
                        "Priority": priority,
                        "Upgrade Rationale": rationale,
                        "is_stash": True
                    })
            else:
                # 1-Week Spot Streamer Mode
                same_pos_bench = [b for b in non_qb_bench if b["position"] == w_pos]
                drop_cand = same_pos_bench[0] if same_pos_bench else non_qb_bench[0]

                delta_mu = summ_w["mean"] - drop_cand["mean"]
                delta_p90 = summ_w["p90"] - drop_cand["p90"]

                if delta_mu > 1.0 or delta_p90 > 3.0:
                    priority = "HIGH" if delta_mu >= 3.5 else ("MEDIUM" if delta_mu >= 1.8 else "SPECULATIVE")
                    rationale = (
                        f"Direct position upgrade over {drop_cand['name']} (+{delta_mu:.1f} pts, +{delta_p90:.1f} ceiling)"
                        if drop_cand["position"] == w_pos else
                        f"Roster depth upgrade over {drop_cand['name']} ({drop_cand['position']}) (+{delta_mu:.1f} pts)"
                    )

                    upgrades.append({
                        "Target Waiver Add": w_name,
                        "Pos": w_pos,
                        "Team": w.get("team", "FA"),
                        "Add Proj": round(summ_w["mean"], 1),
                        "Add P10": round(summ_w["p10"], 1),
                        "Add P90": round(summ_w["p90"], 1),
                        "Projected (Med)": round(summ_w["median"], 1),
                        "Floor (P10)": round(summ_w["p10"], 1),
                        "Ceiling (P90)": round(summ_w["p90"], 1),
                        "Streamer OS%": f"{os_score:.1f}%",
                        "Cut Candidate": drop_cand["name"],
                        "Cut Pos": drop_cand["position"],
                        "Cut Proj": round(drop_cand["mean"], 1),
                        "Net Proj Gain": round(delta_mu, 1),
                        "Net P90 Gain": round(delta_p90, 1),
                        "Priority": priority,
                        "Upgrade Rationale": rationale,
                        "is_stash": False
                    })

    df = pd.DataFrame(upgrades)
    if df.empty:
        return df

    # Sort based on strategy mode
    if is_stash_mode:
        df = df.sort_values(by=["Net P90 Gain", "Add P90"], ascending=[False, False]).reset_index(drop=True)
    else:
        df = df.sort_values(by=["Net Proj Gain", "Add Proj"], ascending=[False, False]).reset_index(drop=True)

    # Calculate dynamic FAAB bid & format Atomic Transaction Pairs
    atomic_pairs = []
    faab_displays = []
    for idx, row in df.iterrows():
        p_rank = idx + 1
        faab_val, faab_pct, faab_str = calculate_suggested_faab(
            net_pts_gain=float(row["Net Proj Gain"]),
            win_leverage=float(row["Net Proj Gain"]) * 0.015,
            remaining_faab=remaining_faab,
            priority_rank=p_rank,
            is_stash=row.get("is_stash", False),
            contingency_ceiling=float(row.get("Add P90", 0.0))
        )
        faab_displays.append(faab_str)
        if row.get("is_stash", False):
            pair_str = f"[Priority #{p_rank}] Add {row['Target Waiver Add']} ➔ Drop {row['Cut Candidate']} | Net Ceiling +{row['Net P90 Gain']:.1f} pts | Suggested FAAB: {faab_str}"
        else:
            pair_str = f"[Priority #{p_rank}] Add {row['Target Waiver Add']} ➔ Drop {row['Cut Candidate']} | Net Gain {row['Net Proj Gain']:+.1f} pts | Suggested FAAB: {faab_str}"
        atomic_pairs.append(pair_str)

    df["Suggested FAAB"] = faab_displays
    df["Atomic Transaction Pair"] = atomic_pairs
    df["Priority"] = [f"#{i+1}" for i in range(len(df))]

    return df
