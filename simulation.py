"""
simulation.py
Monte Carlo Simulation Engine for Weekly Fantasy Football.
- 10,000 iterations per matchup.
- Positional distributions:
    * QB: Normal distribution N(mu, sigma^2)
    * RB: Gamma distribution Gamma(k, theta)
    * WR/TE: Log-Normal distribution LogNormal(mu_log, sigma_log)
- Availability modeling with Bernoulli trials via Polymarket crowd probability.
- Head-to-head win probability P(Win), margins of victory, and outcome percentiles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

import data_ingest
import scheme_db


def sample_qb(mu: float, sigma: float, n: int = 10000, p_active: float = 1.0) -> np.ndarray:
    """
    Quarterback distribution: Normal distribution N(mu, sigma^2).
    Symmetric around consensus expectation with variance modeling passing yards, TDs, and INT risks.
    Truncated at 0. Availability factored via Bernoulli trial.
    """
    mu = max(1.0, float(mu))
    sigma = max(0.5, float(sigma))
    samples = np.random.normal(loc=mu, scale=sigma, size=n)
    samples = np.maximum(0.0, samples)

    if p_active < 1.0:
        availability = np.random.binomial(n=1, p=max(0.0, min(1.0, p_active)), size=n)
        samples = samples * availability
    return samples


def sample_rb(mu: float, sigma: float, n: int = 10000, p_active: float = 1.0) -> np.ndarray:
    """
    Running Back distribution: Gamma distribution Gamma(k, theta).
    Right-skewed non-negative model capturing high touch volume with explosive breakaway potential.
      k = (mu / sigma)^2       (shape parameter)
      theta = sigma^2 / mu     (scale parameter)
      E[X] = k * theta = mu
      Var(X) = k * theta^2 = sigma^2
    """
    mu = max(1.0, float(mu))
    sigma = max(0.5, float(sigma))

    # Parameterize Gamma shape and scale
    k = (mu / sigma) ** 2
    theta = (sigma ** 2) / mu

    samples = np.random.gamma(shape=k, scale=theta, size=n)

    if p_active < 1.0:
        availability = np.random.binomial(n=1, p=max(0.0, min(1.0, p_active)), size=n)
        samples = samples * availability
    return samples


def sample_wr_te(mu: float, sigma: float, n: int = 10000, p_active: float = 1.0) -> np.ndarray:
    """
    Wide Receiver & Tight End distribution: Log-Normal distribution LogNormal(mu_log, sigma_log).
    Captures target volatility, explosive multi-TD upside, and long right tails.
      sigma_log = sqrt(ln(1 + (sigma / mu)^2))
      mu_log = ln(mu) - 0.5 * sigma_log^2
      Median = exp(mu_log)
      Mean E[X] = mu
    """
    mu = max(1.0, float(mu))
    sigma = max(0.5, float(sigma))

    var_ratio = (sigma / mu) ** 2
    sigma_log = np.sqrt(np.log(1.0 + var_ratio))
    mu_log = np.log(mu) - 0.5 * (sigma_log ** 2)

    samples = np.random.lognormal(mean=mu_log, sigma=sigma_log, size=n)

    if p_active < 1.0:
        availability = np.random.binomial(n=1, p=max(0.0, min(1.0, p_active)), size=n)
        samples = samples * availability
    return samples


def sample_player(
    position: str,
    mu: float,
    sigma: float,
    n: int = 10000,
    p_active: float = 1.0
) -> np.ndarray:
    """
    Sample player fantasy scores for n iterations based on position-specific distribution.
    If player is inactive (p_active <= 0.0 or mu <= 0.0), return zeros immediately.
    """
    if p_active <= 0.0 or mu <= 0.0:
        return np.zeros(n)

    pos = (position or "").upper()
    if pos == "QB":
        return sample_qb(mu, sigma, n=n, p_active=p_active)
    elif pos == "RB":
        return sample_rb(mu, sigma, n=n, p_active=p_active)
    elif pos in ["WR", "TE"]:
        return sample_wr_te(mu, sigma, n=n, p_active=p_active)
    else:
        # K, DEF, or generic: Normal truncated at 0
        return sample_qb(mu, sigma, n=n, p_active=p_active)


def summarize_distribution(samples: np.ndarray) -> Dict[str, float]:
    """Calculate summary statistics for a simulated distribution including P15 floor and P90 ceiling."""
    return {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "std": float(np.std(samples)),
        "p10": float(np.percentile(samples, 10)),
        "p15": float(np.percentile(samples, 15)),
        "p25": float(np.percentile(samples, 25)),
        "p75": float(np.percentile(samples, 75)),
        "p90": float(np.percentile(samples, 90)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }


def simulate_roster(
    roster: List[Dict[str, Any]],
    n_simulations: int = 10000
) -> Dict[str, Any]:
    """
    Simulate weekly fantasy points for an active roster across n iterations.
    Each player dict should have: name, position, mu, sigma, and optional p_active.
    """
    player_sims = {}
    total_scores = np.zeros(n_simulations, dtype=np.float64)

    for player in roster:
        name = player.get("name", "Unknown")
        pos = player.get("position", "WR")
        mu = float(player.get("mu", 12.0))
        sigma = float(player.get("sigma", 5.0))
        p_active = float(player.get("p_active", 1.0))

        sim = sample_player(pos, mu, sigma, n=n_simulations, p_active=p_active)
        player_sims[name] = sim
        total_scores += sim

    return {
        "player_simulations": player_sims,
        "team_scores": total_scores,
        "summary": summarize_distribution(total_scores)
    }


def simulate_matchup(
    team_a_roster: List[Dict[str, Any]],
    team_b_roster: List[Dict[str, Any]],
    n_simulations: int = 10000
) -> Dict[str, Any]:
    """
    Simulate head-to-head weekly matchup between Team A and Team B.
    Calculates exact win probability P(Win), margins of victory,
    and outcome distributions.
    """
    sim_a = simulate_roster(team_a_roster, n_simulations=n_simulations)
    sim_b = simulate_roster(team_b_roster, n_simulations=n_simulations)

    scores_a = sim_a["team_scores"]
    scores_b = sim_b["team_scores"]

    # Calculate Head-to-Head Win Probability
    wins_a = np.sum(scores_a > scores_b)
    ties = np.sum(scores_a == scores_b)
    p_win_a = float((wins_a + 0.5 * ties) / n_simulations)
    p_win_b = 1.0 - p_win_a

    # Margin of victory (Team A - Team B)
    margins = scores_a - scores_b

    return {
        "n_simulations": n_simulations,
        "p_win_a": p_win_a,
        "p_win_b": p_win_b,
        "team_a_scores": scores_a,
        "team_b_scores": scores_b,
        "team_a_summary": sim_a["summary"],
        "team_b_summary": sim_b["summary"],
        "team_a_players": sim_a["player_simulations"],
        "team_b_players": sim_b["player_simulations"],
        "margins": margins,
        "margin_summary": summarize_distribution(margins)
    }


def evaluate_start_sit_decision(
    candidate_a: Dict[str, Any],
    candidate_b: Dict[str, Any],
    base_team_roster: List[Dict[str, Any]],
    opponent_roster: List[Dict[str, Any]],
    slot_name: str = "FLEX",
    n_simulations: int = 10000,
    opponent_team: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate comparative Start/Sit decision between two candidates with:
      - Receiver Alignment Profiling (Slot Primary vs Boundary Primary vs Hybrid)
      - Opposing Defensive Funnel Calibration (Inside Funnel vs Outside Funnel)
      - Matchup-Dependent Objective Tuning:
        * Favorite (P(Win) > 60%): Maximize 15th percentile floor (P15) to eliminate failure modes.
        * Underdog (P(Win) < 40%): Maximize 90th percentile ceiling (P90) to spark an upset.
        * Toss-Up (40% - 60%): Maximize overall matchup win probability P(Win).
    """
    cand_a_eval = dict(candidate_a)
    cand_b_eval = dict(candidate_b)

    # 0. Defensive Opponent Identification & Scheme Modifiers
    opp_team_a = opponent_team or cand_a_eval.get("opponent_team") or cand_a_eval.get("opponent")
    if not opp_team_a and cand_a_eval.get("team"):
        p_team = str(cand_a_eval.get("team", "")).upper()
        opp_team_a = getattr(data_ingest, "CURRENT_NFL_SCHEDULE", {}).get(p_team, "")

    opp_team_b = opponent_team or cand_b_eval.get("opponent_team") or cand_b_eval.get("opponent")
    if not opp_team_b and cand_b_eval.get("team"):
        p_team = str(cand_b_eval.get("team", "")).upper()
        opp_team_b = getattr(data_ingest, "CURRENT_NFL_SCHEDULE", {}).get(p_team, "")

    # Apply defensive scheme & funnel modifiers if opponent team is identified and not already modified
    if opp_team_a and "scheme_notes" not in cand_a_eval:
        cand_a_eval = data_ingest.apply_defensive_scheme_modifiers(cand_a_eval, opponent_team=opp_team_a)
    if opp_team_b and "scheme_notes" not in cand_b_eval:
        cand_b_eval = data_ingest.apply_defensive_scheme_modifiers(cand_b_eval, opponent_team=opp_team_b)

    # Check for inactive / IR statuses
    is_inactive_a = bool(
        cand_a_eval.get("is_inactive", False)
        or cand_a_eval.get("p_active", 1.0) <= 0.0
        or cand_a_eval.get("slot") in ["IR", "IR_ELIGIBLE"]
        or str(cand_a_eval.get("injury_status", "")).upper() in ["IR", "INJURY_RESERVE", "OUT", "O", "PUP", "SUSPENDED", "DOUBTFUL"]
    )
    is_inactive_b = bool(
        cand_b_eval.get("is_inactive", False)
        or cand_b_eval.get("p_active", 1.0) <= 0.0
        or cand_b_eval.get("slot") in ["IR", "IR_ELIGIBLE"]
        or str(cand_b_eval.get("injury_status", "")).upper() in ["IR", "INJURY_RESERVE", "OUT", "O", "PUP", "SUSPENDED", "DOUBTFUL"]
    )

    if is_inactive_a:
        cand_a_eval["mu"] = 0.0
        cand_a_eval["sigma"] = 0.0
        cand_a_eval["p_active"] = 0.0
        cand_a_eval["is_inactive"] = True

    if is_inactive_b:
        cand_b_eval["mu"] = 0.0
        cand_b_eval["sigma"] = 0.0
        cand_b_eval["p_active"] = 0.0
        cand_b_eval["is_inactive"] = True

    # Receiver Alignment Profiles
    align_a = data_ingest.get_receiver_alignment(cand_a_eval.get("name", ""), cand_a_eval.get("position", "WR"))
    align_b = data_ingest.get_receiver_alignment(cand_b_eval.get("name", ""), cand_b_eval.get("position", "WR"))

    funnel_a = scheme_db.get_defensive_funnel_profile(opp_team_a or "DEF")
    funnel_b = scheme_db.get_defensive_funnel_profile(opp_team_b or "DEF")

    # Generate Actionable Funnel Advantage Note
    notes = []
    if align_a.get("alignment_tag") in ["Slot Primary", "Hybrid"] and funnel_a.get("is_inside_funnel", False):
        notes.append(f"Advantage {cand_a_eval['name']}: {align_a.get('alignment_tag', '').lower()} alignment ({align_a.get('slot_snap_pct', 0)*100:.0f}% slot) attacks {funnel_a.get('team', 'DEF')}'s vulnerable middle-of-field coverage (MOF EPA {funnel_a.get('mof_epa_allowed', 0):.2f}, Quick Pres {funnel_a.get('quick_pressure_rate', 0)*100:.1f}%)")
    elif align_a.get("alignment_tag") == "Boundary Primary" and funnel_a.get("is_outside_funnel", False):
        notes.append(f"Ceiling boost for {cand_a_eval['name']}: boundary alignment ({align_a.get('wide_snap_pct', 0)*100:.0f}% perimeter) attacks {funnel_a.get('team', 'DEF')}'s single-high boundary funnel")

    if align_b.get("alignment_tag") in ["Slot Primary", "Hybrid"] and funnel_b.get("is_inside_funnel", False):
        notes.append(f"Advantage {cand_b_eval['name']}: {align_b.get('alignment_tag', '').lower()} alignment ({align_b.get('slot_snap_pct', 0)*100:.0f}% slot) attacks {funnel_b.get('team', 'DEF')}'s vulnerable middle-of-field coverage (MOF EPA {funnel_b.get('mof_epa_allowed', 0):.2f}, Quick Pres {funnel_b.get('quick_pressure_rate', 0)*100:.1f}%)")
    elif align_b.get("alignment_tag") == "Boundary Primary" and funnel_b.get("is_outside_funnel", False):
        notes.append(f"Ceiling boost for {cand_b_eval['name']}: boundary alignment ({align_b.get('wide_snap_pct', 0)*100:.0f}% perimeter) attacks {funnel_b.get('team', 'DEF')}'s single-high boundary funnel")

    if not notes:
        funnel_advantage_note = f"Neutral coverage matchup: {cand_a_eval['name']} ({align_a['alignment_tag']}) vs {opp_team_a or 'DEF'} and {cand_b_eval['name']} ({align_b['alignment_tag']}) vs {opp_team_b or 'DEF'}."
    else:
        funnel_advantage_note = " | ".join(notes)

    # 1. Direct Head-to-Head Player Simulation
    sim_a = sample_player(
        cand_a_eval["position"],
        cand_a_eval["mu"],
        cand_a_eval["sigma"],
        n=n_simulations,
        p_active=cand_a_eval.get("p_active", 1.0)
    )
    sim_b = sample_player(
        cand_b_eval["position"],
        cand_b_eval["mu"],
        cand_b_eval["sigma"],
        n=n_simulations,
        p_active=cand_b_eval.get("p_active", 1.0)
    )

    p_outscore_a = float(np.mean(sim_a > sim_b) + 0.5 * np.mean(sim_a == sim_b))
    p_outscore_b = 1.0 - p_outscore_a

    # 2. Team Impact Simulation
    roster_with_a = [p for p in base_team_roster if p.get("slot") != slot_name] + [{**cand_a_eval, "slot": slot_name}]
    roster_with_b = [p for p in base_team_roster if p.get("slot") != slot_name] + [{**cand_b_eval, "slot": slot_name}]

    matchup_a = simulate_matchup(roster_with_a, opponent_roster, n_simulations=n_simulations)
    matchup_b = simulate_matchup(roster_with_b, opponent_roster, n_simulations=n_simulations)

    win_rate_a = matchup_a["p_win_a"]
    win_rate_b = matchup_b["p_win_a"]
    delta_win_rate = win_rate_a - win_rate_b

    # 3. Floor vs Ceiling Profile
    summary_a = summarize_distribution(sim_a)
    summary_b = summarize_distribution(sim_b)

    # 4. Matchup-Dependent Objective Tuning
    baseline_win_rate = max(win_rate_a, win_rate_b)

    # Absolute priority: Inactive / IR candidates must NEVER be started over active candidates
    if is_inactive_a and not is_inactive_b:
        tactical_mode = "INACTIVE / IR"
        tactical_objective = f"Avoid Inactive Player ({candidate_a['name']})"
        recommended = candidate_b["name"]
        inj_a = cand_a_eval.get("injury_status", "IR")
        rationale = f"🚨 DO NOT START {candidate_a['name']}! Player is designated {inj_a} (0% active probability) and will score 0.0 fantasy points. Start {candidate_b['name']}."

    elif is_inactive_b and not is_inactive_a:
        tactical_mode = "INACTIVE / IR"
        tactical_objective = f"Avoid Inactive Player ({candidate_b['name']})"
        recommended = candidate_a["name"]
        inj_b = cand_b_eval.get("injury_status", "IR")
        rationale = f"🚨 DO NOT START {candidate_b['name']}! Player is designated {inj_b} (0% active probability) and will score 0.0 fantasy points. Start {candidate_a['name']}."

    elif is_inactive_a and is_inactive_b:
        tactical_mode = "BOTH INACTIVE"
        tactical_objective = "Replace Both Inactive Candidates"
        recommended = candidate_a["name"]
        rationale = f"⚠️ Both {candidate_a['name']} and {candidate_b['name']} are designated INACTIVE/IR. Neither should be started."

    elif baseline_win_rate > 0.60:
        tactical_mode = "FAVORITE"
        tactical_objective = "Maximize P15 Floor (Protect Lead)"
        if summary_a["p15"] >= summary_b["p15"]:
            recommended = candidate_a["name"]
            rationale = f"As a strong favorite ({baseline_win_rate*100:.1f}% win prob), prioritize safe floor. Start {candidate_a['name']} with superior P15 floor ({summary_a['p15']:.1f} vs {summary_b['p15']:.1f})."
        else:
            recommended = candidate_b["name"]
            rationale = f"As a strong favorite ({baseline_win_rate*100:.1f}% win prob), prioritize safe floor. Start {candidate_b['name']} with superior P15 floor ({summary_b['p15']:.1f} vs {summary_a['p15']:.1f})."

    elif baseline_win_rate < 0.40:
        tactical_mode = "UNDERDOG"
        tactical_objective = "Maximize P90 Ceiling (Chase Upset)"
        if summary_a["p90"] >= summary_b["p90"]:
            recommended = candidate_a["name"]
            rationale = f"As an underdog ({baseline_win_rate*100:.1f}% win prob), chase explosive upside. Start {candidate_a['name']} with higher P90 ceiling ({summary_a['p90']:.1f} vs {summary_b['p90']:.1f})."
        else:
            recommended = candidate_b["name"]
            rationale = f"As an underdog ({baseline_win_rate*100:.1f}% win prob), chase explosive upside. Start {candidate_b['name']} with higher P90 ceiling ({summary_b['p90']:.1f} vs {summary_a['p90']:.1f})."

    else:
        tactical_mode = "TOSS-UP"
        tactical_objective = "Maximize Win Probability P(Win)"
        if win_rate_a >= win_rate_b:
            recommended = candidate_a["name"]
            rationale = f"In a contested matchup ({baseline_win_rate*100:.1f}% win prob), optimize head-to-head win probability. Start {candidate_a['name']} (+{delta_win_rate*100:+0.1f}% win leverage)."
        else:
            recommended = candidate_b["name"]
            rationale = f"In a contested matchup ({baseline_win_rate*100:.1f}% win prob), optimize head-to-head win probability. Start {candidate_b['name']} (+{-delta_win_rate*100:+0.1f}% win leverage)."

    return {
        "candidate_a": candidate_a["name"],
        "candidate_b": candidate_b["name"],
        "sim_a": sim_a,
        "sim_b": sim_b,
        "summary_a": summary_a,
        "summary_b": summary_b,
        "p_outscore_a": p_outscore_a,
        "p_outscore_b": p_outscore_b,
        "team_win_rate_a": win_rate_a,
        "team_win_rate_b": win_rate_b,
        "delta_win_rate": delta_win_rate,
        "tactical_mode": tactical_mode,
        "tactical_objective": tactical_objective,
        "tactical_rationale": rationale,
        "recommended": recommended,
        "align_a": align_a,
        "align_b": align_b,
        "funnel_profile_a": funnel_a,
        "funnel_profile_b": funnel_b,
        "funnel_profile": funnel_a,
        "funnel_advantage_note": funnel_advantage_note
    }


def find_roster_optimizations(
    starters: List[Dict[str, Any]],
    bench: List[Dict[str, Any]],
    opponent_roster: List[Dict[str, Any]],
    n_simulations: int = 5000
) -> Dict[str, Any]:
    """
    Roster Optimization & Win-Rate Maximizer.
    Audits the current starting lineup against all bench alternatives.
    Identifies any substitution that yields a net gain in win probability P(Win).
    Returns:
      {
        "current_win_rate": float,
        "optimal_win_rate": float,
        "max_gain": float,
        "is_optimal": bool,
        "recommended_swaps": List[Dict[str, Any]],
        "audit_table": pd.DataFrame
      }
    """
    base_matchup = simulate_matchup(starters, opponent_roster, n_simulations=n_simulations)
    current_p_win = base_matchup["p_win_a"]

    audit_rows = []

    for b in bench:
        b_name = b.get("name", "Unknown")
        b_pos = b.get("position", "WR").upper()
        b_mu = float(b.get("mu", 10.0))

        # Bench players who are inactive, on IR, or have zero projection must NEVER be recommended as starter upgrades
        b_inactive = bool(
            b.get("is_inactive", False)
            or b.get("p_active", 1.0) <= 0.0
            or b.get("slot") in ["IR", "IR_ELIGIBLE"]
            or str(b.get("injury_status", "")).upper() in ["IR", "INJURY_RESERVE", "OUT", "O", "PUP", "SUSPENDED", "DOUBTFUL"]
            or b_mu <= 0.0
        )
        if b_inactive:
            continue

        for s in starters:
            s_name = s.get("name", "Unknown")
            s_pos = s.get("position", "WR").upper()
            s_slot = s.get("slot", s_pos).upper()
            s_mu = float(s.get("mu", 10.0))

            s_inactive = bool(
                s.get("is_inactive", False)
                or s.get("p_active", 1.0) <= 0.0
                or s.get("slot") in ["IR", "IR_ELIGIBLE"]
                or str(s.get("injury_status", "")).upper() in ["IR", "INJURY_RESERVE", "OUT", "O", "PUP", "SUSPENDED", "DOUBTFUL"]
            )

            # Positional eligibility check:
            # OP allows QB, RB, WR, TE
            # FLEX allows RB, WR, TE
            # WR/TE allows WR, TE
            # RB/WR/TE allows RB, WR, TE
            is_eligible = (b_pos == s_pos) or (
                s_slot in ["FLEX", "WR/TE", "RB/WR/TE", "OP"] and (
                    b_pos in ["RB", "WR", "TE"] or (s_slot == "OP" and b_pos == "QB")
                )
            )

            if not is_eligible:
                continue

            # Simulate swapped lineup
            swapped_starters = [
                p if p.get("name") != s_name else {**b, "slot": s_slot}
                for p in starters
            ]
            swap_res = simulate_matchup(swapped_starters, opponent_roster, n_simulations=n_simulations)
            swap_p_win = swap_res["p_win_a"]
            delta_p_win = swap_p_win - current_p_win
            delta_pts = b_mu - s_mu

            if s_inactive:
                status = "🚨 INACTIVE STARTER (MUST REPLACE)"
                recommendation = f"Replace {s_name} ({s.get('injury_status', 'IR')}) with {b_name}"
            elif delta_p_win > 0.008:
                status = "🚀 UPGRADE AVAILABLE"
                recommendation = f"Start {b_name} over {s_name}"
            elif delta_p_win < -0.008:
                status = "✅ CURRENT STARTER OPTIMAL"
                recommendation = f"Keep {s_name} in lineup"
            else:
                status = "⚖️ LATERAL / TOSS-UP"
                recommendation = "Either option viable"

            audit_rows.append({
                "Bench Player": b_name,
                "Pos": b_pos,
                "Starter Compared": s_name,
                "Slot": s_slot,
                "Bench Proj": round(b_mu, 1),
                "Starter Proj": round(s_mu, 1),
                "Pt Delta": f"{delta_pts:+.1f}",
                "Win% Leverage": f"{delta_p_win * 100:+.1f}%",
                "Raw Delta Win": delta_p_win,
                "Status": status,
                "Recommendation": recommendation
            })

    audit_rows.sort(key=lambda x: x["Raw Delta Win"], reverse=True)
    df_audit = pd.DataFrame(audit_rows) if audit_rows else pd.DataFrame()

    recommended_swaps = [r for r in audit_rows if r["Raw Delta Win"] > 0.005]
    is_optimal = len(recommended_swaps) == 0
    max_gain = recommended_swaps[0]["Raw Delta Win"] if recommended_swaps else 0.0
    optimal_win_rate = current_p_win + max_gain

    return {
        "current_win_rate": current_p_win,
        "optimal_win_rate": optimal_win_rate,
        "max_gain": max_gain,
        "is_optimal": is_optimal,
        "recommended_swaps": recommended_swaps,
        "audit_table": df_audit
    }
