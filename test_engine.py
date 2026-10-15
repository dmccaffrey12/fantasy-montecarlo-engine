"""
test_engine.py
Automated Verification Test Suite for Weekly Fantasy Football Monte Carlo Engine.
Covers:
1. Data Ingestion (The Odds API, Polymarket Gamma API, Sleeper fallback, ESPN module).
2. Simulation Engine (QB Normal, RB Gamma, WR/TE Log-Normal, 10,000-run Monte Carlo).
3. VORP Engine (Dynamic Positional & Flex Baselines, Points VORP, Delta P(Win)).
"""

import os
import sys
import time
import numpy as np
import data_ingest
import simulation
import vorp
import pdf_parser
import scheme_db
import scoring_config
import trade_analyzer


def test_data_ingest():
    print("=== Testing 1: data_ingest.py ===")

    # Test 1.1: Key retrieval
    key = data_ingest.get_odds_api_key()
    assert key is not None and len(key) > 10, f"Failed to retrieve API key: {key}"
    print(f"  [PASS] The Odds API Key loaded: {key[:8]}...")

    # Test 1.2: Implied Team Totals Math
    games_mock = [{
        "id": "test_game_1",
        "home_team": "Buffalo Bills",
        "away_team": "Detroit Lions",
        "commence_time": "2026-09-18T00:15:00Z",
        "bookmakers": [{
            "markets": [
                {"key": "spreads", "outcomes": [{"name": "Buffalo Bills", "point": -4.5}]},
                {"key": "totals", "outcomes": [{"name": "Over", "point": 54.5}]}
            ]
        }]
    }]
    parsed = data_ingest.parse_nfl_games(games_mock)
    assert len(parsed) == 1
    # Home: (54.5 - (-4.5)) / 2 = 29.5, Away: (54.5 + (-4.5)) / 2 = 25.0
    assert parsed[0]["home_implied"] == 29.5, f"Expected 29.5, got {parsed[0]['home_implied']}"
    assert parsed[0]["away_implied"] == 25.0, f"Expected 25.0, got {parsed[0]['away_implied']}"
    print("  [PASS] Implied team totals calculated correctly (Bills 29.5, Lions 25.0).")

    # Test 1.3: American Odds to Probability
    prob_neg = data_ingest.american_odds_to_prob(-320)
    assert abs(prob_neg - (320 / 420)) < 0.01, f"Unexpected prob: {prob_neg}"
    prob_pos = data_ingest.american_odds_to_prob(+150)
    assert abs(prob_pos - (100 / 250)) < 0.01, f"Unexpected prob: {prob_pos}"
    print(f"  [PASS] American odds converted accurately (-320 -> {prob_neg:.1%}, +150 -> {prob_pos:.1%}).")

    # Test 1.4: Polymarket API Query
    poly_slug = "nba-will-the-mavericks-beat-the-grizzlies-by-more-than-5pt5-points-in-their-december-4-matchup"
    poly_res = data_ingest.fetch_polymarket_event(poly_slug)
    assert poly_res is not None, "Polymarket Gamma API returned None"
    assert "active_probability" in poly_res
    print(f"  [PASS] Polymarket Gamma API connected successfully: active_probability={poly_res['active_probability']:.2f}")

    # Test 1.5: Projection Parameter Synthesis
    mu_qb, sig_qb, p_qb = data_ingest.calculate_projection_params("QB", {"pass_yds": 250, "rush_yds": 30, "anytime_td_prob": 0.5})
    assert mu_qb > 15.0 and sig_qb > 3.0
    mu_rb, sig_rb, p_rb = data_ingest.calculate_projection_params("RB", {"rush_yds": 80, "receptions": 4, "anytime_td_prob": 0.6})
    assert mu_rb > 12.0 and sig_rb > 3.0
    print(f"  [PASS] Parameter synthesis: QB(mu={mu_qb}, sig={sig_qb}), RB(mu={mu_rb}, sig={sig_rb}).")

    # Test 1.6: De-Vig Anytime TD Implied Probabilities
    raw_market = {
        "Saquon Barkley": 0.65,
        "A.J. Brown": 0.52,
        "DeVonta Smith": 0.44,
        "Dallas Goedert": 0.38,
        "Jalen Hurts": 0.58,
        "Kenneth Gainwell": 0.22,
    }
    devigged_target = data_ingest.devig_anytime_td_probabilities(raw_market, realistic_td_total=2.1)
    sum_devigged = sum(devigged_target.values())
    assert abs(sum_devigged - 2.1) < 0.05, f"Expected de-vigged sum ~2.1, got {sum_devigged}"
    assert devigged_target["Saquon Barkley"] < raw_market["Saquon Barkley"]
    print(f"  [PASS] De-vigged Anytime TD Market: Raw sum {sum(raw_market.values()):.2f} normalized to {sum_devigged:.2f} TDs.")

    # Test 1.7: Format Distribution Columns
    import pandas as pd
    raw_df = pd.DataFrame([{"Player": "Test", "mu": 15.0, "sigma": 5.0, "p_active": 1.0}])
    formatted_df = data_ingest.format_distribution_columns(raw_df)
    assert "Projected (Med)" in formatted_df.columns
    assert "Floor (P10)" in formatted_df.columns
    assert "Ceiling (P90)" in formatted_df.columns
    assert formatted_df.iloc[0]["Floor (P10)"] == round(15.0 - 1.28 * 5.0, 1)
    assert formatted_df.iloc[0]["Ceiling (P90)"] == round(15.0 + 1.28 * 5.0, 1)
    print("  [PASS] format_distribution_columns creates 'Projected (Med)', 'Floor (P10)', and 'Ceiling (P90)'.")


def test_simulation():
    print("\n=== Testing 2: simulation.py ===")
    n = 10000

    # Test 2.1: QB Normal Distribution
    t0 = time.time()
    qb_sim = simulation.sample_qb(mu=20.0, sigma=5.0, n=n)
    dt_qb = time.time() - t0
    assert qb_sim.shape == (n,)
    assert abs(np.mean(qb_sim) - 20.0) < 0.3, f"QB mean error: {np.mean(qb_sim)}"
    assert np.all(qb_sim >= 0.0), "QB scores must be non-negative"
    print(f"  [PASS] QB Normal: mean={np.mean(qb_sim):.2f}, std={np.std(qb_sim):.2f} in {dt_qb*1000:.1f}ms")

    # Test 2.2: RB Gamma Distribution
    rb_sim = simulation.sample_rb(mu=16.0, sigma=6.4, n=n)
    assert rb_sim.shape == (n,)
    assert abs(np.mean(rb_sim) - 16.0) < 0.3, f"RB mean error: {np.mean(rb_sim)}"
    assert np.all(rb_sim >= 0.0), "RB scores must be non-negative"
    print(f"  [PASS] RB Gamma: mean={np.mean(rb_sim):.2f}, std={np.std(rb_sim):.2f}, max={np.max(rb_sim):.2f}")

    # Test 2.3: WR/TE Log-Normal Distribution
    wr_sim = simulation.sample_wr_te(mu=14.0, sigma=6.7, n=n)
    assert wr_sim.shape == (n,)
    assert abs(np.mean(wr_sim) - 14.0) < 0.4, f"WR mean error: {np.mean(wr_sim)}"
    # Verify right skew (max is significantly higher than mean + 3*std)
    assert np.max(wr_sim) > 40.0, f"Expected explosive ceiling, got max={np.max(wr_sim)}"
    print(f"  [PASS] WR Log-Normal: mean={np.mean(wr_sim):.2f}, max ceiling={np.max(wr_sim):.2f}")

    # Test 2.4: Polymarket Availability Bernoulli trials
    inj_sim = simulation.sample_wr_te(mu=15.0, sigma=6.0, n=n, p_active=0.50)
    zero_ratio = np.mean(inj_sim == 0.0)
    assert abs(zero_ratio - 0.50) < 0.03, f"Availability error: {zero_ratio}"
    print(f"  [PASS] Polymarket Bernoulli availability: {zero_ratio*100:.1f}% zero-score rate (target 50%).")

    # Test 2.5: 10,000 Matchup Simulation Speed & Convergence
    data = data_ingest.get_default_matchup_and_waivers()
    t_start = time.time()
    res = simulation.simulate_matchup(data["team_a_roster"], data["team_b_roster"], n_simulations=10000)
    total_time = (time.time() - t_start) * 1000
    assert 0.0 <= res["p_win_a"] <= 1.0
    assert abs(res["p_win_a"] + res["p_win_b"] - 1.0) < 1e-6
    print(f"  [PASS] 10,000 Matchup Run: Team A P(Win) = {res['p_win_a']*100:.1f}% in {total_time:.1f}ms")

    # Test 2.6: Roster Optimization & Swap Audit
    opt = simulation.find_roster_optimizations(data["team_a_roster"], data.get("team_a_bench", []), data["team_b_roster"], n_simulations=3000)
    assert "current_win_rate" in opt
    assert "optimal_win_rate" in opt
    assert "audit_table" in opt
    print(f"  [PASS] Roster Optimization Audit: Current P(Win)={opt['current_win_rate']*100:.1f}%, Optimal P(Win)={opt['optimal_win_rate']*100:.1f}% (+{opt['max_gain']*100:.1f}% max leverage).")


def test_vorp():
    print("\n=== Testing 3: vorp.py ===")
    data = data_ingest.get_default_matchup_and_waivers()

    # Test 3.1: Dynamic Baselines
    baselines = vorp.calculate_waiver_baselines(data["waiver_pool"], n_simulations=10000)
    for pos in ["QB", "RB", "WR", "TE", "FLEX"]:
        assert pos in baselines, f"Missing baseline for {pos}"
        assert baselines[pos]["baseline_mean"] > 0, f"Baseline <= 0 for {pos}"
    baseline_summary = [f"{k}={v['baseline_mean']:.1f} ({v['player_name']})" for k, v in baselines.items()]
    print(f"  [PASS] Dynamic Baselines computed: {', '.join(baseline_summary)}")

    # Test 3.2: Roster VORP and Win-Rate Leverage (Delta P(Win))
    df_vorp = vorp.calculate_roster_vorp(
        data["team_a_roster"],
        baselines,
        opponent_roster=data["team_b_roster"],
        n_simulations=5000
    )
    assert not df_vorp.empty
    assert "Points VORP" in df_vorp.columns
    assert "Win% VORP (Delta P(Win))" in df_vorp.columns
    top_player = df_vorp.iloc[0]
    print(f"  [PASS] Top VORP asset: {top_player['Player']} ({top_player['Pos']}) with VORP={top_player['Points VORP']:.2f}, Win% Leverage={top_player['Win% VORP (Delta P(Win))']}")

    # Test 3.3: Waiver Pool Rankings & Streamer OS
    df_waivers = vorp.rank_waiver_wire_pool(data["waiver_pool"], baselines, n_simulations=5000)
    assert not df_waivers.empty
    assert "Streamer OS (P≥12)" in df_waivers.columns
    print(f"  [PASS] Top Waiver Wire Target: {df_waivers.iloc[0]['Player']} ({df_waivers.iloc[0]['Pos']}) Proj {df_waivers.iloc[0]['Proj Pts']:.1f} pts, OS={df_waivers.iloc[0]['Streamer OS (P≥12)']}.")

    # Test 3.4: Targeted Waiver Roster Upgrades with 2-QB League Constraint
    df_upgrades = vorp.calculate_waiver_roster_upgrades(
        data.get("team_a_bench", []), data["waiver_pool"], full_roster=data.get("team_a_roster", []), n_simulations=3000
    )
    assert not df_upgrades.empty
    assert "Target Waiver Add" in df_upgrades.columns
    assert "Cut Candidate" in df_upgrades.columns
    assert "Net Proj Gain" in df_upgrades.columns
    # Ensure no QB is recommended when waiver QBs don't exceed the 2 rostered QBs (Goff 18.6, Dak 19.2)
    qb_adds = df_upgrades[df_upgrades["Pos"] == "QB"]
    assert qb_adds.empty, "Waiver QBs lower than rostered QBs should not be recommended"
    # Ensure WR/RB is never cut for a QB
    for _, row in df_upgrades.iterrows():
        if row["Pos"] == "QB":
            assert row["Cut Pos"] == "QB", f"QB waiver add cannot cut non-QB {row['Cut Candidate']} ({row['Cut Pos']})"
    top_rec = df_upgrades.iloc[0]
    print(f"  [PASS] Targeted Waiver Upgrades (2-QB Rule Enforced): Top Add '{top_rec['Target Waiver Add']}' ({top_rec['Pos']}) / Cut '{top_rec['Cut Candidate']}' (+{top_rec['Net Proj Gain']} pts gain). Sub-par QBs blocked from cutting skill players.")

    # Test 3.5: Elite QB Upgrade Scenario (comparing strictly against the 2 rostered QBs)
    waivers_with_elite = data["waiver_pool"] + [{"name": "Lamar Jackson", "position": "QB", "team": "BAL", "mu": 28.0, "sigma": 6.5, "p_active": 1.0}]
    df_elite_upgrades = vorp.calculate_waiver_roster_upgrades(
        data.get("team_a_bench", []), waivers_with_elite, full_roster=data.get("team_a_roster", []), n_simulations=3000
    )
    qb_elite_recs = df_elite_upgrades[df_elite_upgrades["Target Waiver Add"] == "Lamar Jackson"]
    assert not qb_elite_recs.empty, "Elite waiver QB should be recommended"
    assert qb_elite_recs.iloc[0]["Cut Candidate"] == "Jared Goff", f"Expected Cut Candidate 'Jared Goff', got {qb_elite_recs.iloc[0]['Cut Candidate']}"
    assert qb_elite_recs.iloc[0]["Cut Pos"] == "QB", "Cut position must be QB"
    print(f"  [PASS] Elite QB Waiver Upgrade: 'Lamar Jackson' compared strictly against rostered QBs -> Recommends cutting 'Jared Goff' (QB), preserving 2-QB limit.")

    # Test 3.6: Dynamic Suggested FAAB Calculation
    bid_streamer, pct_streamer, disp_streamer = vorp.calculate_suggested_faab(
        net_pts_gain=3.5, win_leverage=0.04, remaining_faab=80, priority_rank=1
    )
    assert bid_streamer > 0 and pct_streamer > 0
    assert "$" in disp_streamer and "%" in disp_streamer

    bid_stash, pct_stash, disp_stash = vorp.calculate_suggested_faab(
        net_pts_gain=0.0, remaining_faab=100, is_stash=True, contingency_ceiling=22.0
    )
    assert bid_stash >= 10 and pct_stash >= 10
    print(f"  [PASS] Dynamic FAAB calculation: Streamer={disp_streamer}, Contingent Stash={disp_stash}.")

    # Test 3.7: Atomic Transaction Pairs & Strategy Modes (Streamer vs Stash)
    df_streamers = vorp.calculate_waiver_roster_upgrades(
        data.get("team_a_bench", []), data["waiver_pool"], full_roster=data.get("team_a_roster", []), remaining_faab=75, mode="streamer", n_simulations=2000
    )
    assert not df_streamers.empty
    assert "Atomic Transaction Pair" in df_streamers.columns
    assert "Suggested FAAB" in df_streamers.columns
    assert "Projected (Med)" in df_streamers.columns
    assert "Floor (P10)" in df_streamers.columns
    assert "Ceiling (P90)" in df_streamers.columns
    assert "[Priority #1] Add" in df_streamers.iloc[0]["Atomic Transaction Pair"]
    assert "Suggested FAAB:" in df_streamers.iloc[0]["Atomic Transaction Pair"]

    df_stashes = vorp.calculate_waiver_roster_upgrades(
        data.get("team_a_bench", []), data["waiver_pool"], full_roster=data.get("team_a_roster", []), remaining_faab=75, mode="stash", n_simulations=2000
    )
    assert not df_stashes.empty
    assert "Net Ceiling" in df_stashes.iloc[0]["Atomic Transaction Pair"] or "Suggested FAAB" in df_stashes.iloc[0]["Atomic Transaction Pair"]
    clean_pair = df_streamers.iloc[0]["Atomic Transaction Pair"].encode("ascii", errors="replace").decode("ascii")
    print(f"  [PASS] Atomic Transaction Pairs generated: '{clean_pair}'.")


def test_advanced_features():
    print("\n=== Testing 4: Advanced Analytical Modules ===")

    # Test 4.1: Bayesian Usage vs. Efficiency Updating
    sample_wr = {"name": "Test Alpha WR", "position": "WR", "mu": 8.0, "sigma": 4.0}
    # Alpha usage: 90% snap, 28% target share, but only scored 8.0 pts due to TD bad luck
    mu_bayes, sig_bayes, diag = data_ingest.apply_bayesian_usage_shrinkage(
        sample_wr, snap_pct=0.90, target_share=0.28, prior_weight=0.70
    )
    assert mu_bayes > sample_wr["mu"], f"Expected Bayesian upward regression from alpha volume, got {mu_bayes}"
    print(f"  [PASS] Bayesian Shrinkage: Prior volume expectation ({diag['mu_volume_prior']:.1f}) regressed observed (8.0) -> mu_bayes={mu_bayes:.1f}")

    # Test 4.2: Scheme & Defensive Matchup Modifiers
    # Test TE against high MOF EPA defense (CAR has MOF EPA 0.58 > 0.50)
    sample_te = {"name": "Test TE", "position": "TE", "mu": 10.0, "sigma": 4.5}
    mod_te = data_ingest.apply_defensive_scheme_modifiers(sample_te, opponent_team="CAR")
    assert abs(mod_te["mu"] - 11.5) < 0.05, f"Expected 1.15x multiplier (11.5), got {mod_te['mu']}"
    print(f"  [PASS] Defensive Scheme (High MOF EPA): TE mu boosted 1.15x ({sample_te['mu']} -> {mod_te['mu']})")

    # Test RB against high Rush SR defense (NYG has Rush SR 0.475 > 0.45)
    sample_rb = {"name": "Test RB", "position": "RB", "mu": 14.0, "sigma": 5.6}
    mod_rb = data_ingest.apply_defensive_scheme_modifiers(sample_rb, opponent_team="NYG")
    assert mod_rb["mu"] > sample_rb["mu"] and mod_rb["sigma"] < sample_rb["sigma"]
    print(f"  [PASS] Defensive Scheme (High Rush SR): RB mu boosted ({mod_rb['mu']}) & sigma tightened ({mod_rb['sigma']})")

    # Test Early-Season Bayesian Shrinkage Damping (Weeks 1-3 cap matchup drift to +/- 8%)
    mod_te_early = data_ingest.apply_defensive_scheme_modifiers(sample_te, opponent_team="CAR", week=2)
    drift_pct = (mod_te_early["mu"] - sample_te["mu"]) / sample_te["mu"]
    assert drift_pct <= 0.0801, f"Expected early-season drift <= 8%, got {drift_pct:.1%}"
    assert "Early-Season Scheme Damping (Wk 2)" in mod_te_early["scheme_notes"]
    print(f"  [PASS] Early-Season Scheme Damping: Week 2 drift capped at {drift_pct*100:+.1f}% (<= 8.0%).")

    # Test 4.3: Matchup-Dependent Objective Tuning
    # Candidate A: High floor, lower ceiling (e.g. slot possession WR)
    cand_safe = {"name": "Safe Floor Play", "position": "WR", "mu": 12.0, "sigma": 3.0, "p_active": 1.0}
    # Candidate B: Low floor, explosive ceiling (e.g. deep threat WR)
    cand_boom = {"name": "Boom Bust Play", "position": "WR", "mu": 12.0, "sigma": 7.5, "p_active": 1.0}

    # Underdog setup: opponent heavily favored
    weak_roster = [{"name": f"P{i}", "position": "WR", "slot": f"WR{i}", "mu": 8.0, "sigma": 3.0} for i in range(6)]
    elite_opponent = [{"name": f"OP{i}", "position": "WR", "slot": f"WR{i}", "mu": 20.0, "sigma": 4.0} for i in range(7)]

    ss_underdog = simulation.evaluate_start_sit_decision(
        cand_safe, cand_boom, base_team_roster=weak_roster, opponent_roster=elite_opponent, n_simulations=5000
    )
    assert ss_underdog["tactical_mode"] == "UNDERDOG", f"Expected UNDERDOG mode, got {ss_underdog['tactical_mode']}"
    assert ss_underdog["recommended"] == "Boom Bust Play", f"Expected Boom candidate in Underdog mode, got {ss_underdog['recommended']}"
    print(f"  [PASS] Objective Tuning (Underdog): Tactical Mode = {ss_underdog['tactical_mode']} -> Recommended {ss_underdog['recommended']} (Chasing P90 Ceiling).")

    # Test 4.4: Contingency & Handcuff Bench Leverage
    test_handcuffs = [
        {"name": "Jerome Ford", "position": "RB", "team": "CLE", "mu": 11.5, "sigma": 4.6},
        {"name": "Jaylen Warren", "position": "RB", "team": "PIT", "mu": 10.2, "sigma": 4.1}
    ]
    df_cont = vorp.calculate_contingency_leverage(test_handcuffs, n_simulations=5000)
    assert not df_cont.empty
    assert "Contingency Index" in df_cont.columns
    assert "Contingency P90 Ceiling" in df_cont.columns
    assert df_cont.iloc[0]["Contingency Index"] > 0
    print(f"  [PASS] Contingency Leverage: Top Handcuff {df_cont.iloc[0]['Player']} (Index={df_cont.iloc[0]['Contingency Index']}, P90 Ceiling={df_cont.iloc[0]['Contingency P90 Ceiling']})")

    # Test 4.5: High-Probability Streamer Scoring (OS)
    sim_streamer = np.array([5.0, 8.0, 12.5, 14.0, 16.0, 7.0, 13.0, 6.0, 12.0, 4.0])
    os_score = vorp.calculate_outperformance_score(sim_streamer, threshold=12.0)
    assert abs(os_score - 50.0) < 1e-4, f"Expected 50.0%, got {os_score}"
    print(f"  [PASS] Streamer Outperformance Score (OS): P(Score >= 12.0) = {os_score:.1f}%")


def test_scheme_lab():
    print("\n=== Testing 5: Scheme Lab & Newsletter Storage (scheme_db.py & pdf_parser.py) ===")

    # Test 5.1: Sample PDF Generation & Parsing
    sample_pdf_path = "sample_sumersports_week3.pdf"
    if not os.path.exists(sample_pdf_path):
        pdf_parser.create_sample_sumersports_pdf(sample_pdf_path, week=3)

    parsed = pdf_parser.parse_sumersports_newsletter(sample_pdf_path, default_week=3)
    assert parsed["success"] is True, f"PDF parser failed: {parsed.get('error')}"
    assert parsed["week"] == 3, f"Expected Week 3, got {parsed['week']}"

    team_df = parsed["team_metrics"]
    player_df = parsed["player_metrics"]
    assert not team_df.empty, "Parsed team metrics should not be empty"
    assert not player_df.empty, "Parsed player metrics should not be empty"
    assert "quick_pressure_rate" in team_df.columns
    assert "mof_epa_allowed" in team_df.columns
    assert "rush_sr_allowed" in team_df.columns
    assert "target_share" in player_df.columns
    assert "yprr" in player_df.columns
    print(f"  [PASS] PDF Parser extracted {len(team_df)} team schemes and {len(player_df)} player records from Week 3.")

    # Test 5.2: Parquet Persistent Storage & Overwrite Logic
    # 1st save
    res1 = scheme_db.save_newsletter_metrics(team_df, player_df, week=3)
    assert res1["success"] is True
    total_teams_1 = len(scheme_db.load_scheme_history())

    # 2nd save for same week (overwrite verification: row count should remain constant)
    res2 = scheme_db.save_newsletter_metrics(team_df, player_df, week=3)
    assert res2["success"] is True
    total_teams_2 = len(scheme_db.load_scheme_history())
    assert total_teams_1 == total_teams_2, f"Expected idempotent overwrite (count {total_teams_1}), got {total_teams_2}"
    print(f"  [PASS] Persistent Storage Week Overwrite: verified idempotent upsert ({total_teams_2} total team rows).")

    # Test 5.3: Rolling 3-Week Moving Average Calculation
    cle_rolling = scheme_db.get_team_defense_rolling("CLE", window=3)
    assert cle_rolling["weeks_counted"] >= 1
    assert 0.10 <= cle_rolling["quick_pressure_rate"] <= 0.40
    assert 0.10 <= cle_rolling["mof_epa_allowed"] <= 0.80
    assert 0.25 <= cle_rolling["rush_sr_allowed"] <= 0.60
    assert cle_rolling["is_rolling"] is True
    print(f"  [PASS] Rolling 3-Wk Defense: CLE Quick Press={cle_rolling['quick_pressure_rate']*100:.1f}%, MOF EPA={cle_rolling['mof_epa_allowed']:.2f}, Rush SR={cle_rolling['rush_sr_allowed']*100:.1f}% ({cle_rolling['weeks_counted']} weeks).")

    # Test 5.4: Scheme Trend Detection (Delta vs Season Average)
    cle_trend = scheme_db.get_scheme_trend("CLE")
    assert cle_trend["team"] == "CLE"
    assert "delta_quick_pressure" in cle_trend
    assert "delta_mof_epa" in cle_trend
    assert "delta_rush_sr" in cle_trend
    assert "trend_summary" in cle_trend
    print(f"  [PASS] Scheme Trend: CLE Delta QP={cle_trend['delta_quick_pressure']*100:+.1f}%, Summary='{cle_trend['trend_summary']}'.")

    # Test 5.5: Player Weekly Usage Trajectory
    doug_hist = scheme_db.get_player_usage_history("Demario Douglas")
    assert not doug_hist.empty, "Expected player history for Demario Douglas"
    assert "target_share" in doug_hist.columns and "yprr" in doug_hist.columns
    latest_tgt = doug_hist.iloc[-1]["target_share"] * 100
    print(f"  [PASS] Player History: Demario Douglas tracked across {len(doug_hist)} weeks (Latest Target Share: {latest_tgt:.1f}%).")

    # Test 5.6: Dynamic Ingestion Integration with data_ingest
    dynamic_scheme = data_ingest.get_defensive_scheme("CLE", window=3)
    assert abs(dynamic_scheme["quick_pressure_rate"] - cle_rolling["quick_pressure_rate"]) < 1e-4
    print(f"  [PASS] Dynamic Scheme Ingestion: data_ingest dynamically queries scheme_db rolling metrics.")


def test_funnel_synergy_and_trade_scanner():
    print("\n=== Testing 6: Scheme-to-Alignment Funnel Synergy & Floor Surge Scanner ===")

    # Test 6.1: Receiver Alignment Tagging
    align_doug = data_ingest.get_receiver_alignment("DeMario Douglas", "WR")
    assert align_doug["alignment_tag"] == "Slot Primary", f"Expected Slot Primary for Douglas, got {align_doug['alignment_tag']}"
    assert align_doug["slot_snap_pct"] >= 0.60

    align_coker = data_ingest.get_receiver_alignment("Jalen Coker", "WR")
    assert align_coker["alignment_tag"] == "Hybrid", f"Expected Hybrid for Coker, got {align_coker['alignment_tag']}"

    align_london = data_ingest.get_receiver_alignment("Drake London", "WR")
    assert align_london["alignment_tag"] == "Boundary Primary", f"Expected Boundary Primary for London, got {align_london['alignment_tag']}"
    assert align_london["wide_snap_pct"] >= 0.60
    print(f"  [PASS] Alignment Tagging: Douglas={align_doug['alignment_tag']} ({align_doug['slot_snap_pct']*100:.0f}%), Coker={align_coker['alignment_tag']}, London={align_london['alignment_tag']} ({align_london['wide_snap_pct']*100:.0f}%).")

    # Test 6.2: Defensive Funnel Calibration
    funnel_car = scheme_db.get_defensive_funnel_profile("CAR")
    assert funnel_car["is_inside_funnel"] is True, f"Expected CAR to be Inside Funnel, got {funnel_car['funnel_classification']}"
    assert "Inside Funnel" in funnel_car["funnel_classification"]

    funnel_cle = scheme_db.get_defensive_funnel_profile("CLE")
    assert "funnel_classification" in funnel_cle
    print(f"  [PASS] Defensive Funnel Profiles: CAR='{funnel_car['funnel_classification']}', CLE='{funnel_cle['funnel_classification']}'.")

    # Test 6.3: Funnel Calibration Application in data_ingest
    wr_slot = {"name": "DeMario Douglas", "position": "WR", "team": "NE", "mu": 10.0, "sigma": 4.0}
    wr_mod_inside = data_ingest.apply_defensive_scheme_modifiers(wr_slot, opponent_team="CAR")
    assert wr_mod_inside["mu"] >= 11.5, f"Expected boosted mu >= 11.5, got {wr_mod_inside['mu']}"
    assert wr_mod_inside["sigma"] <= 3.6, f"Expected compressed sigma <= 3.6, got {wr_mod_inside['sigma']}"
    print(f"  [PASS] Scheme Modifiers Inside Funnel: Slot WR mu boosted ({wr_slot['mu']} -> {wr_mod_inside['mu']}) and sigma compressed ({wr_slot['sigma']} -> {wr_mod_inside['sigma']}).")

    # Test 6.4: Start/Sit Decision Engine Integration with Funnel Diagnostics
    cand_a = {"name": "DeMario Douglas", "position": "WR", "team": "NE", "mu": 10.0, "sigma": 4.0}
    cand_b = {"name": "Jalen Coker", "position": "WR", "team": "CAR", "mu": 10.0, "sigma": 4.0}
    base_team = [{"name": f"P{i}", "position": "WR", "slot": f"WR{i}", "mu": 10.0, "sigma": 3.0} for i in range(6)]
    opp_team = [{"name": f"OP{i}", "position": "WR", "slot": f"WR{i}", "mu": 10.0, "sigma": 3.0} for i in range(7)]

    ss_eval = simulation.evaluate_start_sit_decision(
        cand_a, cand_b, base_team_roster=base_team, opponent_roster=opp_team, n_simulations=3000, opponent_team="CAR"
    )
    assert "align_a" in ss_eval
    assert "align_b" in ss_eval
    assert "funnel_profile_a" in ss_eval
    assert "funnel_advantage_note" in ss_eval
    assert ss_eval["align_a"]["alignment_tag"] == "Slot Primary"
    print(f"  [PASS] Start/Sit Funnel Diagnostics: Returned alignment profiles and note: '{ss_eval['funnel_advantage_note'][:80]}...'")

    # Test 6.5: Floor Surge & Contrarian Buy-Low Scanner
    df_scanner = scheme_db.detect_floor_surges_and_buy_lows()
    assert not df_scanner.empty, "Expected non-empty scanner DataFrame"
    assert "Player" in df_scanner.columns
    assert "Trade Signal" in df_scanner.columns
    assert "Action" in df_scanner.columns
    assert "Trade & Strategy Rationale" in df_scanner.columns
    actions = df_scanner["Action"].tolist()
    assert any(a in ["BUY / TARGET", "STRONG BUY", "HOLD", "SELL / BENCH"] for a in actions)
    top_target = df_scanner.iloc[0]
    sig_clean = str(top_target['Trade Signal']).encode('ascii', errors='ignore').decode('ascii').strip()
    print(f"  [PASS] Floor Surge Scanner: Detected {len(df_scanner)} trade candidates. Top Pick: {top_target['Player']} ({top_target['Pos']}) Signal: {sig_clean} Action: {top_target['Action']}.")


def test_injury_and_ir_handling():
    print("\n=== Testing 7: Injury & IR Status Normalization ===")

    # Test 7.1: Status Parsing
    st_ir, p_ir, inact_ir = data_ingest.parse_player_injury_and_status("INJURY_RESERVE", "IR")
    assert st_ir == "IR" and p_ir == 0.0 and inact_ir is True, f"Failed IR parsing: {st_ir}, {p_ir}, {inact_ir}"

    st_out, p_out, inact_out = data_ingest.parse_player_injury_and_status("OUT", "BE")
    assert st_out == "OUT" and p_out == 0.0 and inact_out is True

    st_q, p_q, inact_q = data_ingest.parse_player_injury_and_status("QUESTIONABLE", "WR")
    assert st_q == "QUESTIONABLE" and p_q == 0.50 and inact_q is False

    st_act, p_act, inact_act = data_ingest.parse_player_injury_and_status("ACTIVE", "WR")
    assert st_act == "ACTIVE" and p_act == 1.0 and inact_act is False
    print("  [PASS] parse_player_injury_and_status correctly categorizes IR, OUT, Q, and ACTIVE.")

    # Test 7.2: Inactive Projection Parameters Zeroing
    mu_zero, sig_zero, p_zero = data_ingest.calculate_projection_params("WR", {"receptions": 5, "rec_yds": 60}, is_inactive=True)
    assert mu_zero == 0.0 and sig_zero == 0.0 and p_zero == 0.0
    print("  [PASS] calculate_projection_params(is_inactive=True) enforces (0.0, 0.0, 0.0).")

    # Test 7.3: sample_player with Inactive Zeroing
    sim_inactive = simulation.sample_player("WR", mu=0.0, sigma=0.0, p_active=0.0, n=5000)
    assert np.all(sim_inactive == 0.0)
    print("  [PASS] sample_player produces all 0.0s for inactive player.")

    # Test 7.4: Scheme Modifiers on Inactive Player
    kirk_profile = {
        "name": "Christian Kirk",
        "position": "WR",
        "team": "JAX",
        "mu": 0.0,
        "sigma": 0.0,
        "p_active": 0.0,
        "is_inactive": True,
        "injury_status": "IR",
        "slot": "IR"
    }
    mod_kirk = data_ingest.apply_defensive_scheme_modifiers(kirk_profile, opponent_team="CAR")
    assert mod_kirk["mu"] == 0.0 and mod_kirk["sigma"] == 0.0
    print("  [PASS] apply_defensive_scheme_modifiers never inflates inactive player mu.")

    # Test 7.5: Start/Sit Decision Engine Strictly Rejects IR Player (Kirk vs Coker)
    coker_profile = {
        "name": "Jalen Coker",
        "position": "WR",
        "team": "CAR",
        "mu": 8.7,
        "sigma": 3.8,
        "p_active": 0.50,
        "is_inactive": False,
        "injury_status": "QUESTIONABLE",
        "slot": "WR"
    }
    base_team = [{"name": f"P{i}", "position": "WR", "slot": f"WR{i}", "mu": 10.0, "sigma": 3.0} for i in range(6)]
    opp_team = [{"name": f"OP{i}", "position": "WR", "slot": f"WR{i}", "mu": 10.0, "sigma": 3.0} for i in range(7)]

    eval_kirk_coker = simulation.evaluate_start_sit_decision(
        kirk_profile,
        coker_profile,
        base_team_roster=base_team,
        opponent_roster=opp_team,
        n_simulations=3000
    )
    assert eval_kirk_coker["recommended"] == "Jalen Coker", f"Expected Coker, got {eval_kirk_coker['recommended']}"
    assert eval_kirk_coker["tactical_mode"] == "INACTIVE / IR"
    assert "DO NOT START Christian Kirk" in eval_kirk_coker["tactical_rationale"]
    clean_rat = eval_kirk_coker['tactical_rationale'][:60].encode('ascii', errors='ignore').decode('ascii')
    print(f"  [PASS] Start/Sit evaluation strictly rejects IR player ({clean_rat}...).")

    # Test 7.6: Roster Optimization Filters Out IR Bench Players and Flags Inactive Starters
    starters_with_kirk = base_team + [kirk_profile]
    bench_active = [coker_profile]
    opt_inactive = simulation.find_roster_optimizations(starters_with_kirk, bench_active, opp_team, n_simulations=2000)
    assert not opt_inactive["is_optimal"], "Should detect optimization because starter is inactive"
    assert any("INACTIVE STARTER" in sw["Status"] for sw in opt_inactive["recommended_swaps"]), "Must flag inactive starter"
    print("  [PASS] find_roster_optimizations flags inactive starter as 'INACTIVE STARTER (MUST REPLACE)'.")


def test_idiosyncratic_scoring():
    print("\n=== Testing 8: scoring_config.py & Idiosyncratic Scoring Engine ===")
    cfg = scoring_config.DEFAULT_SCORING

    # Test 8.1: Corroborated Scoring Rules
    assert cfg.pass_td == 6.0, "Passing TD must be 6.0 points"
    assert cfg.completion_bonus == 0.1, "Completion bonus must be 0.1"
    assert cfg.pass_td_40_bonus == 1.0, "40+ yd TD pass bonus must be 1.0"
    assert cfg.pass_td_50_bonus == 4.0, "50+ yd TD pass bonus must be 4.0 (stacks to 5.0)"
    assert cfg.pass_int == -1.0, "Interceptions must be -1.0"
    assert cfg.pass_300_bonus == 2.0, "300-399 yd pass bonus must be 2.0"
    assert cfg.pass_400_bonus == 3.0, "400+ yd pass bonus must be 3.0 (stacks to 5.0)"
    assert cfg.ppr == 1.0, "PPR must be 1.0"
    assert cfg.rush_100_bonus == 1.0, "100+ rush bonus must be 1.0"
    assert cfg.rush_200_bonus == 2.0, "200+ rush bonus must be 2.0 (stacks to 3.0)"
    assert cfg.rec_100_bonus == 1.0, "100+ rec bonus must be 1.0"
    assert cfg.rec_200_bonus == 2.0, "200+ rec bonus must be 2.0 (stacks to 3.0)"
    print("  [PASS] All idiosyncratic league scoring constants corroborated.")

    # Test 8.2: QB 6pt TD & Completion Elevation Math
    props_qb = {
        "pass_yds": 300.0,
        "pass_completions": 25.0,
        "pass_tds": 2.5,
        "pass_interceptions": 0.5,
        "rush_yds": 10.0
    }
    mu_idio, sig_idio, p_act, breakdown = scoring_config.calculate_player_mu_and_sigma("QB", props_qb)
    assert mu_idio > 30.0, f"Expected mu > 30.0 under 6pt pass TD & completions, got {mu_idio}"
    print(f"  [PASS] QB scoring math: {mu_idio:.2f} pts (Elevated by 6pt Pass TD & 0.1 comps).")

    # Test 8.3: Defensive Brackets Math
    pa_score_good = scoring_config.get_points_allowed_score(14.0)
    pa_score_bad = scoring_config.get_points_allowed_score(35.0)
    ya_score_good = scoring_config.get_yards_allowed_score(180.0)
    ya_score_bad = scoring_config.get_yards_allowed_score(420.0)
    assert pa_score_good == 1.0, f"Expected 1.0 for PA=14, got {pa_score_good}"
    assert pa_score_bad == -3.0, f"Expected -3.0 for PA=35, got {pa_score_bad}"
    assert ya_score_good == 3.0, f"Expected 3.0 for YA=180, got {ya_score_good}"
    assert ya_score_bad == -3.0, f"Expected -3.0 for YA=420, got {ya_score_bad}"
    print("  [PASS] D/ST Points Allowed and Yards Allowed tiered brackets match league rules.")

    # Test 8.4: Live Consensus Props Cache Integration
    props_cache = data_ingest.build_all_player_props_cache()
    assert len(props_cache) > 10, "Props cache should contain player lines"
    assert "jared goff" in props_cache, "Jared Goff should be in props cache"
    goff_props = props_cache["jared goff"]
    assert goff_props["pass_yds"] > 200.0
    print(f"  [PASS] Vegas Props Cache verified: Jared Goff Pass O/U {goff_props['pass_yds']:.1f} yds.")

    # Test 8.5: Dynamic Default Matchup with Idiosyncratic Projections
    matchup_data = data_ingest.get_default_matchup_and_waivers()
    team_a = matchup_data["team_a_roster"]
    assert len(team_a) == 9, "Should have 9 starters"
    assert team_a[0]["name"] == "Jared Goff"
    assert team_a[0]["prop_source"] in ["Vegas Consensus Props", "Hybrid (50% Vegas / 50% Athletic)"]
    assert team_a[0]["mu"] > 20.0, f"Goff projected {team_a[0]['mu']} pts (elevated under idiosyncratic scoring)"
    print(f"  [PASS] Default Matchup dynamically evaluated: {team_a[0]['name']} mu={team_a[0]['mu']} pts ({team_a[0]['prop_source']}).")

    # Test 8.6: Polymarket Availability Resolution
    p_prob, src = data_ingest.get_polymarket_player_probability("Drake London", "ACTIVE")
    assert p_prob == 1.0
    p_prob_q, src_q = data_ingest.get_polymarket_player_probability("Test Player", "QUESTIONABLE", default_p=0.50)
    assert p_prob_q == 0.50
    assert "Prior" in src_q or "Polymarket" in src_q
    print(f"  [PASS] Polymarket availability resolution operational: Active={p_prob*100:.0f}%, Questionable={p_prob_q*100:.0f}%.")

    # Test 8.7: Scoring Config Presets & Idiosyncratic Elevation
    cfg_idio = scoring_config.get_scoring_config_by_name("My League (Idiosyncratic 6pt Pass TD)")
    cfg_std = scoring_config.get_scoring_config_by_name("Standard Full PPR (4pt Pass TD)")
    assert cfg_idio.pass_td == 6.0 and cfg_idio.completion_bonus == 0.1
    assert cfg_std.pass_td == 4.0 and cfg_std.completion_bonus == 0.0

    matchup_idio = data_ingest.get_default_matchup_and_waivers(scoring_cfg=cfg_idio)
    matchup_std = data_ingest.get_default_matchup_and_waivers(scoring_cfg=cfg_std)
    goff_idio = matchup_idio["team_a_roster"][0]["mu"]
    goff_std = matchup_std["team_a_roster"][0]["mu"]
    assert goff_idio > goff_std + 5.0, f"Expected Goff idiosyncratic mu ({goff_idio}) to exceed standard 4pt mu ({goff_std}) by > 5 pts"
    print(f"  [PASS] Scoring Presets Verified: Goff Proj = {goff_idio:.2f} pts (Idiosyncratic 6pt + 0.1 comps) vs {goff_std:.2f} pts (Standard 4pt).")


def test_espn_credentials_and_waiver_isolation():
    print("\n=== Testing 9: ESPN Credentials Ingestion & Waiver Isolation ===")

    # Test 9.1: Recursive case-insensitive credential scanning and cookie unquoting
    mock_secrets_nested = {
        "espn": {
            "league_id": 998877,
            "year": 2025,
            "espn_s2": "mock_s2_token%3D",
            "swid": "%7BA1B2C3D4-E5F6-7890-ABCD-1234567890EF%7D",
            "team_name": "My True Fantasy Team"
        }
    }

    parsed = data_ingest.get_espn_credentials(secrets_dict=mock_secrets_nested)
    assert parsed["league_id"] == 998877, f"Expected 998877, got {parsed['league_id']}"
    assert parsed["year"] == 2025, f"Expected 2025, got {parsed['year']}"
    assert parsed["espn_s2"] == "mock_s2_token=", f"Expected unquoted s2, got {parsed['espn_s2']}"
    assert parsed["swid"] == "{A1B2C3D4-E5F6-7890-ABCD-1234567890EF}", f"Expected unquoted SWID with braces, got {parsed['swid']}"
    assert parsed["user_team_hint"] == "My True Fantasy Team"
    assert parsed["found_in_secrets"] is True
    print("  [PASS] Nested [espn] table credentials correctly extracted, unquoted, and formatted.")

    # Test 9.2: Flat uppercase credentials without curly braces on SWID
    mock_secrets_flat = {
        "ESPN_LEAGUE_ID": "123456",
        "ESPN_YEAR": "2026",
        "SWID": "F83A8BC9-1234-5678-90AB-CDEF12345678"
    }
    parsed_flat = data_ingest.get_espn_credentials(secrets_dict=mock_secrets_flat)
    assert parsed_flat["league_id"] == 123456
    assert parsed_flat["year"] == 2026
    assert parsed_flat["swid"] == "{F83A8BC9-1234-5678-90AB-CDEF12345678}"
    print("  [PASS] Flat uppercase credentials & naked SWID auto-wrapped in braces.")

    # Test 9.3: Waiver Pool Isolation (no demo players leaking into synced league)
    team_a_mock = {
        "team_name": "User Team",
        "starters": [
            {"name": "Josh Allen", "position": "QB", "slot": "QB", "mu": 24.0, "sigma": 5.0, "p_active": 1.0},
            {"name": "James Cook", "position": "RB", "slot": "RB1", "mu": 16.0, "sigma": 4.0, "p_active": 1.0},
            {"name": "Breece Hall", "position": "RB", "slot": "RB2", "mu": 17.0, "sigma": 4.0, "p_active": 1.0},
            {"name": "CeeDee Lamb", "position": "WR", "slot": "WR1", "mu": 18.0, "sigma": 4.5, "p_active": 1.0},
            {"name": "Amon-Ra St. Brown", "position": "WR", "slot": "WR2", "mu": 17.5, "sigma": 4.0, "p_active": 1.0},
            {"name": "Trey McBride", "position": "TE", "slot": "TE", "mu": 12.0, "sigma": 3.5, "p_active": 1.0},
            {"name": "Nico Collins", "position": "WR", "slot": "FLEX", "mu": 15.0, "sigma": 4.0, "p_active": 1.0}
        ],
        "bench": []
    }
    team_b_mock = {
        "team_name": "Opponent Team",
        "starters": [
            {"name": "Patrick Mahomes", "position": "QB", "slot": "QB", "mu": 22.0, "sigma": 5.0, "p_active": 1.0},
            {"name": "Jahmyr Gibbs", "position": "RB", "slot": "RB1", "mu": 17.0, "sigma": 4.0, "p_active": 1.0},
            {"name": "Kyren Williams", "position": "RB", "slot": "RB2", "mu": 16.5, "sigma": 4.0, "p_active": 1.0},
            {"name": "Justin Jefferson", "position": "WR", "slot": "WR1", "mu": 18.5, "sigma": 4.5, "p_active": 1.0},
            {"name": "Ja'Marr Chase", "position": "WR", "slot": "WR2", "mu": 18.0, "sigma": 4.5, "p_active": 1.0},
            {"name": "Sam LaPorta", "position": "TE", "slot": "TE", "mu": 11.5, "sigma": 3.5, "p_active": 1.0},
            {"name": "Drake London", "position": "WR", "slot": "FLEX", "mu": 14.0, "sigma": 3.5, "p_active": 1.0}
        ],
        "bench": []
    }

    # Live ESPN league with real free agents
    real_fa = [
        {"name": "Tyler Huntley", "position": "QB", "team": "MIA", "mu": 13.0, "sigma": 4.0, "p_active": 1.0},
        {"name": "Roschon Johnson", "position": "RB", "team": "CHI", "mu": 9.5, "sigma": 3.0, "p_active": 1.0}
    ]
    synced_matchup = data_ingest.build_matchup_from_espn_teams(team_a_mock, team_b_mock, free_agents_list=real_fa)
    synced_waivers = [p["name"] for p in synced_matchup["waiver_pool"]]
    assert "Tyler Huntley" in synced_waivers
    assert "Roschon Johnson" in synced_waivers
    assert "Rico Dowdle" not in synced_waivers, "Demo player Rico Dowdle should NOT be in synced waiver pool"
    assert "Quentin Johnston" not in synced_waivers, "Demo player Quentin Johnston should NOT be in synced waiver pool"
    print("  [PASS] Live ESPN waiver pool strictly isolated from demo players.")

    # Live ESPN league with empty free agents list (must not inject demo waivers!)
    empty_fa_matchup = data_ingest.build_matchup_from_espn_teams(team_a_mock, team_b_mock, free_agents_list=[])
    assert len(empty_fa_matchup["waiver_pool"]) == 0
    assert "Rico Dowdle" not in [p["name"] for p in empty_fa_matchup["waiver_pool"]]
    print("  [PASS] Empty ESPN free agent response produces empty waiver list without demo pollution.")


def test_archetype_volatility_and_underdog_accuracy():
    print("\n=== Testing 10: Receiver Archetype Volatility & Start/Sit Underdog Accuracy ===")

    # Test 10.1: Parameter synthesis for Slot Primary vs Boundary Primary vs Hybrid
    mu_doug, sig_doug, p_act, bk_doug = scoring_config.calculate_player_mu_and_sigma(
        "WR",
        props={"receptions": 4.8, "rec_yds": 52.5, "rush_yds": 2.0, "anytime_td_prob": 0.20},
        player_name="DeMario Douglas",
        alignment_tag="Slot Primary",
        adot=6.5
    )
    assert bk_doug["cv"] <= 0.35, f"Expected Slot WR CV <= 0.35, got {bk_doug['cv']}"
    assert "Slot" in bk_doug["volatility_class"]

    mu_coker, sig_coker, p_act, bk_coker = scoring_config.calculate_player_mu_and_sigma(
        "WR",
        props={"receptions": 3.8, "rec_yds": 46.5, "rush_yds": 0.0, "anytime_td_prob": 0.20},
        player_name="Jalen Coker",
        alignment_tag="Hybrid",
        adot=10.4
    )
    assert bk_coker["cv"] >= 0.48, f"Expected Coker CV >= 0.48, got {bk_coker['cv']}"

    # Verify Douglas has lower CV than Coker
    assert bk_doug["cv"] < bk_coker["cv"], f"Expected Douglas CV < Coker CV, got {bk_doug['cv']} vs {bk_coker['cv']}"
    print(f"  [PASS] Archetype CV Separation: DeMario Douglas CV={bk_doug['cv']} ({bk_doug['volatility_class']}) vs Jalen Coker CV={bk_coker['cv']} ({bk_coker['volatility_class']}).")

    # Test 10.2: Monte Carlo simulation tail distribution
    sim_doug = simulation.sample_wr_te(mu_doug, sig_doug, n=20000)
    sim_coker = simulation.sample_wr_te(mu_coker, sig_coker, n=20000)

    summary_doug = simulation.summarize_distribution(sim_doug)
    summary_coker = simulation.summarize_distribution(sim_coker)

    # Douglas should hold superior P10 floor
    assert summary_doug["p10"] > summary_coker["p10"], f"Expected Douglas P10 > Coker P10, got {summary_doug['p10']} vs {summary_coker['p10']}"
    # Coker should exhibit higher boom potential (max ceiling and boom probability >= 20 pts)
    assert summary_coker["p_boom_20"] >= summary_doug["p_boom_20"]
    assert summary_coker["max"] > summary_doug["max"]
    print(f"  [PASS] Distribution Tail Profiles: Douglas Floor (P10={summary_doug['p10']:.1f}) protects lead; Coker Max Ceiling ({summary_coker['max']:.1f} vs {summary_doug['max']:.1f}) and Boom P(>=20) ({summary_coker['p_boom_20']*100:.1f}% vs {summary_doug['p_boom_20']*100:.1f}%) unlock upside.")

    # Test 10.3: Contested Lineup Slot Underdog Logic
    starter_coker = {"name": "Jalen Coker", "position": "WR", "slot": "WR/TE", "mu": mu_coker, "sigma": sig_coker, "alignment_tag": "Hybrid"}
    bench_douglas = {"name": "DeMario Douglas", "position": "WR", "slot": "BE", "mu": mu_doug, "sigma": sig_doug, "alignment_tag": "Slot Primary"}

    contested = simulation.find_contested_lineup_slots(
        starters=[starter_coker],
        bench=[bench_douglas],
        team_a_win_rate=0.35,  # Underdog
        threshold=2.5
    )
    assert len(contested) == 1
    card = contested[0]
    # Douglas must NOT be branded an Underdog Ceiling Swing when Coker is the starter!
    assert card["badge"] != "⚡ Underdog Ceiling Swing", f"Douglas should NOT be labeled Underdog Ceiling Swing, got: {card['badge']}"
    assert "High-Floor Bench Alternative" in card["badge"] or "Preferred Ceiling Starter" in card["badge"]
    clean_badge = card['badge'].encode('ascii', errors='ignore').decode('ascii').strip()
    print(f"  [PASS] Contested Underdog Slot: Correctly identified {card['starter_name']} vs {card['challenger_name']} -> Badge: '{clean_badge}'.")


def test_copula_and_next_horizon():
    print("\n=== Testing 11: Gaussian Copula Covariance, Poisson Multi-TD, & Next-Horizon Modules ===")

    # Test 11.1: Poisson Multi-TD Conversion & Right-Skewed Milestone CDFs
    p_td = 0.50
    lam_td = scoring_config.anytime_prob_to_expected_tds(p_td)
    assert lam_td > p_td, f"Expected Poisson E[TD] > p_anytime for p=0.50, got {lam_td}"
    rush_ev = scoring_config.calculate_rush_milestone_ev(85.0)
    rec_ev = scoring_config.calculate_rec_milestone_ev(85.0)
    assert rush_ev > 0.25 and rec_ev > 0.25
    print(f"  [PASS] Poisson Multi-TD (p=50% -> E[TD]={lam_td:.3f}) & Gamma/Log-Normal Milestone CDFs (Rush EV=+{rush_ev:.2f}, Rec EV=+{rec_ev:.2f}).")

    # Test 11.2: Cholesky Gaussian Copula Stack Correlation (QB + WR Positive, Offense vs Opp D/ST Negative)
    corr_players = [
        {"name": "Jared Goff", "position": "QB", "team": "DET", "opponent": "GB", "mu": 24.0, "sigma": 5.5, "p_active": 1.0},
        {"name": "Amon-Ra St. Brown", "position": "WR", "team": "DET", "opponent": "GB", "mu": 18.0, "sigma": 6.5, "p_active": 1.0},
        {"name": "Packers D/ST", "position": "D/ST", "team": "GB", "opponent": "DET", "mu": 7.0, "sigma": 4.0, "p_active": 1.0},
    ]
    _, mat, corrs = simulation.simulate_correlated_players(corr_players, n_simulations=20000)
    emp_corr_stack = np.corrcoef(mat[0], mat[1])[0, 1]
    emp_corr_dst = np.corrcoef(mat[0], mat[2])[0, 1]
    assert emp_corr_stack > 0.25, f"Expected positive QB-WR stack correlation > 0.25, got {emp_corr_stack:.3f}"
    assert emp_corr_dst < -0.20, f"Expected negative Offense vs Opp D/ST correlation < -0.20, got {emp_corr_dst:.3f}"
    print(f"  [PASS] Cholesky Gaussian Copula verified: DET QB-WR empirical r={emp_corr_stack:+.3f}, DET QB vs GB D/ST r={emp_corr_dst:+.3f}.")

    # Test 11.3: Full 9-Starter ESPN Matchup Preservation (OP 2nd QB + D/ST)
    team_9_mock = {
        "team_name": "Full 9-Starter Team",
        "starters": [
            {"name": "Jared Goff", "position": "QB", "slot": "QB", "mu": 24.0, "sigma": 5.0, "p_active": 1.0},
            {"name": "Dak Prescott", "position": "QB", "slot": "OP", "mu": 23.0, "sigma": 5.0, "p_active": 1.0},
            {"name": "Saquon Barkley", "position": "RB", "slot": "RB", "mu": 21.0, "sigma": 6.0, "p_active": 1.0},
            {"name": "Breece Hall", "position": "RB", "slot": "RB", "mu": 17.0, "sigma": 5.5, "p_active": 1.0},
            {"name": "Drake London", "position": "WR", "slot": "WR", "mu": 16.0, "sigma": 6.0, "p_active": 1.0},
            {"name": "Jalen Coker", "position": "WR", "slot": "WR/TE", "mu": 11.0, "sigma": 5.0, "p_active": 1.0},
            {"name": "Trey McBride", "position": "TE", "slot": "TE", "mu": 14.0, "sigma": 5.0, "p_active": 1.0},
            {"name": "Tony Pollard", "position": "RB", "slot": "RB/WR/TE", "mu": 13.5, "sigma": 5.0, "p_active": 1.0},
            {"name": "Vikings D/ST", "position": "D/ST", "slot": "D/ST", "mu": 8.0, "sigma": 4.0, "p_active": 1.0},
        ],
        "bench": []
    }
    built_9 = data_ingest.build_matchup_from_espn_teams(team_9_mock, team_9_mock, free_agents_list=[])
    assert len(built_9["team_a_roster"]) == 9, f"Expected all 9 starters preserved (including OP & D/ST), got {len(built_9['team_a_roster'])}"
    slots_preserved = [p["slot"] for p in built_9["team_a_roster"]]
    assert "OP" in slots_preserved and "D/ST" in slots_preserved
    print(f"  [PASS] ESPN 9-Starter Lineup Integrity: Preserved all 9 starters ({', '.join(slots_preserved)}).")

    # Test 11.4: Game-Theoretic Matchup Inversion Solver
    data = data_ingest.get_default_matchup_and_waivers()
    inv = simulation.analyze_matchup_inversion(
        team_a_roster=data["team_a_roster"],
        team_a_bench=data["team_a_bench"],
        team_b_roster=data["team_b_roster"]
    )
    assert "dp_dsigma_pct" in inv
    assert "strategy_headline" in inv
    assert "strategy_summary" in inv
    assert "opp_profile" in inv
    clean_hl = inv["strategy_headline"].encode("ascii", errors="ignore").decode("ascii").strip()
    print(f"  [PASS] Game-Theoretic Matchup Inversion: Headline='{clean_hl}', dP(Win)/dSigma={inv['dp_dsigma_pct']:+.3f}%/pt.")

    # Test 11.5: Multi-Week Playoff & First-Round Bye Leverage Matrix
    pm = simulation.simulate_playoff_leverage_matrix(
        team_a_name=data["team_a_name"],
        team_a_roster=data["team_a_roster"],
        waiver_candidates=data["waiver_pool"][:5],
        current_week=6,
        total_reg_weeks=14,
        n_season_sims=1000,
        league_teams=[]
    )
    assert "leverage_table" in pm and not pm["leverage_table"].empty
    assert "1st-Rd Bye Odds Δ" in pm["leverage_table"].columns
    top_bye_cand = pm["leverage_table"].iloc[0]
    print(f"  [PASS] Playoff & Bye Leverage Matrix: Baseline P(Bye)={pm['base_p_bye']*100:.1f}%, Top Add '{top_bye_cand['Roster Addition']}' ({top_bye_cand['1st-Rd Bye Odds Δ']} Bye Leverage).")


def test_athletic_excel_and_hybrid_blending():
    print("\n=== Testing 12: The Athletic Multi-Block Excel Parser & 50/50 Hybrid Blending ===")

    # Test 12.1: Multi-Block Excel Parsing (Week6_Proj_1015.xlsx: 183 Skill Players + 28 D/STs with 4 Bye teams)
    summary = data_ingest.get_athletic_projection_summary()
    counts = summary["counts"]
    assert counts["QB"] == 28, f"Expected 28 QBs, got {counts['QB']}"
    assert counts["RB"] == 60, f"Expected 60 RBs, got {counts['RB']}"
    assert counts["WR"] == 70, f"Expected 70 WRs, got {counts['WR']}"
    assert counts["TE"] == 25, f"Expected 25 TEs, got {counts['TE']}"
    assert summary["skill_total"] == 183, f"Expected 183 skill players, got {summary['skill_total']}"
    assert summary["dst_total"] == 28, f"Expected 28 DSTs, got {summary['dst_total']}"
    print(f"  [PASS] Multi-Block Excel Parser ({summary['filename']}): Extracted {summary['skill_total']} Skill Players (28 QB, 60 RB, 70 WR, 25 TE) + {summary['dst_total']} D/STs.")

    # Test 12.2: Custom ESPN Multipliers on Raw Stat Columns (Ignoring Default 'Proj')
    dak_ath = data_ingest.get_matched_athletic_projection("Dak Prescott", position="QB")
    assert dak_ath["athletic_proj"] > 15.0
    expected_dak_mu = round(
        dak_ath["pass_yds"] * 0.04
        + dak_ath["pass_tds"] * 6.0
        + dak_ath["ints"] * (-1.0)
        + dak_ath["rush_yds"] * 0.10
        + dak_ath["rush_tds"] * 6.0,
        2,
    )
    assert abs(dak_ath["custom_mu"] - expected_dak_mu) < 0.05, f"Expected Dak custom_mu={expected_dak_mu}, got {dak_ath['custom_mu']}"
    print(f"  [PASS] Custom Stat Multiplier Scoring: Dak Prescott Default Proj={dak_ath['athletic_proj']} -> Custom League Mu={dak_ath['custom_mu']:.2f} pts.")

    # Test 12.3: 50% Vegas Implied / 50% Athletic Custom Mu Hybrid Blending
    props_cache = data_ingest.build_all_player_props_cache()
    dak_props = data_ingest.get_matched_player_props("Dak Prescott", props_cache, "QB")
    mu_blend, sig_blend, _, bk_blend = data_ingest.calculate_projection_params(
        position="QB",
        props=dak_props,
        player_name="Dak Prescott",
        return_breakdown=True
    )
    assert bk_blend["source"] == "Hybrid (50% Vegas / 50% Athletic)"
    expected_blend = round(0.50 * bk_blend["vegas_mu"] + 0.50 * bk_blend["athletic_custom_mu"], 2)
    assert abs(mu_blend - expected_blend) < 0.05, f"Expected 50/50 blend {expected_blend}, got {mu_blend}"
    print(f"  [PASS] 50/50 Hybrid Blending: Dak Vegas({bk_blend['vegas_mu']}) + Athletic({bk_blend['athletic_custom_mu']}) -> Blended Mu={mu_blend:.2f}, Calibrated Sigma={sig_blend:.2f}.")

    # Test 12.4: Universal Baseline for Players Without Vegas Props
    bowers_ath = data_ingest.get_matched_athletic_projection("Brock Bowers", position="TE")
    mu_bowers, sig_bowers, _, bk_bowers = data_ingest.calculate_projection_params(
        position="TE",
        props=None,
        player_name="Brock Bowers",
        return_breakdown=True
    )
    assert bk_bowers["source"] == "The Athletic Custom Baseline"
    assert abs(mu_bowers - bowers_ath["custom_mu"]) < 0.05
    print(f"  [PASS] Universal Athletic Baseline (No Vegas Props): Brock Bowers Mu={mu_bowers:.2f} pts, Sigma={sig_bowers:.2f} ({bk_bowers['source']}).")


def test_dual_newsletter_and_longitudinal_narratives():
    print("\n=== Testing 13: SumerSports Dual Newsletter Ingestion & Longitudinal Team Narratives ===")

    # Test 13.1: Document Type Detection (Preview vs Review on Real Newsletters)
    prev_path = os.path.join("newsletters", "week3_preview.pdf")
    rev_path = os.path.join("newsletters", "week3_review.pdf")
    if not os.path.exists(prev_path):
        pdf_parser.create_sample_preview_pdf(prev_path, week=3)
    if not os.path.exists(rev_path):
        pdf_parser.create_sample_sumersports_pdf(rev_path, week=3)
    det_prev = pdf_parser.detect_newsletter_type(prev_path)
    det_rev = pdf_parser.detect_newsletter_type(rev_path)
    assert det_prev["short_type"] == "PREVIEW", f"Expected PREVIEW, got {det_prev}"
    assert det_rev["short_type"] == "REVIEW", f"Expected REVIEW, got {det_rev}"
    print("  [PASS] Document Type Detection: Accurately classified Weekly Preview ('PREVIEW') and Weekly Review ('REVIEW') PDFs.")

    # Test 13.2: Preview & Review PDF Parsing (Real Embedded Image + Narrative Extraction)
    prev_parsed = pdf_parser.parse_preview_pdf(prev_path, week_num=3)
    rev_parsed = pdf_parser.parse_review_pdf(rev_path, week_num=3)
    assert not prev_parsed["preview_metrics"].empty
    assert "man_coverage_rate" in prev_parsed["preview_metrics"].columns
    assert "funnel_focus" in prev_parsed["preview_metrics"].columns
    assert not rev_parsed["team_metrics"].empty
    assert "pressure_rate" in rev_parsed["team_metrics"].columns
    print(f"  [PASS] Dual PDF Parser: Extracted {len(prev_parsed['preview_metrics'])} Preview rows and {len(rev_parsed['team_metrics'])} Review rows ({len(rev_parsed['player_metrics'])} player rows).")

    # Test 13.3: Chronological Batch Backfill & Dual Parquet + Narrative JSON Persistence
    batch_res = scheme_db.batch_ingest_newsletters(folder_path="newsletters")
    assert batch_res["files_processed"] >= 6, f"Expected at least 6 PDFs, got {batch_res['files_processed']}"
    assert os.path.exists(scheme_db.REVIEWS_PARQUET), "scheme_reviews.parquet missing"
    assert os.path.exists(scheme_db.PREVIEWS_PARQUET), "scheme_previews.parquet missing"
    assert os.path.exists(scheme_db.TEAM_NARRATIVES_JSON), "team_narratives.json missing"
    print(f"  [PASS] Batch Backfill Engine: Ingested {batch_res['files_processed']} PDFs chronologically ({batch_res['previews_upserted']} Previews, {batch_res['reviews_upserted']} Reviews) into Parquet & JSON repositories.")

    # Test 13.4: Expectation vs. Reality Delta & Structural Regime Tagging
    cle_delta = scheme_db.get_team_expectation_delta("CLE")
    assert "PASS_PROTECTION_COLLAPSE" in cle_delta["regime_tags"], f"Expected PASS_PROTECTION_COLLAPSE for CLE, got {cle_delta['regime_tags']}"
    assert cle_delta["prior_decay_multiplier"] < 1.0, "Expected accelerated Bayesian prior decay for CLE"

    car_delta = scheme_db.get_team_expectation_delta("CAR")
    assert "SLOT_FUNNEL_HIGH" in car_delta["regime_tags"], f"Expected SLOT_FUNNEL_HIGH for CAR, got {car_delta['regime_tags']}"
    assert car_delta["slot_te_floor_boost"] > 1.0 and car_delta["slot_te_ceiling_boost"] > 1.0
    print(f"  [PASS] Expectation vs. Reality Delta: CLE tags={cle_delta['regime_tags']} (Prior Decay={cle_delta['prior_decay_multiplier']}x), CAR tags={car_delta['regime_tags']} (Slot/TE Boost={car_delta['slot_te_ceiling_boost']}x).")

    # Test 13.5: Longitudinal Team Deep Dive & Accelerated Bayesian Shrinkage Integration
    cle_dive = scheme_db.get_team_longitudinal_deep_dive("CLE")
    assert not cle_dive["timeline_df"].empty and len(cle_dive["timeline_df"]) >= 3
    p_cle = {"name": "Amari Cooper", "position": "WR", "team": "CLE", "mu": 8.0, "sigma": 5.5}
    p_std = {"name": "Amari Cooper", "position": "WR", "team": "KC", "mu": 8.0, "sigma": 5.5}
    mu_cle, _, meta_cle = data_ingest.apply_bayesian_usage_shrinkage(
        p_cle, snap_pct=0.80, target_share=0.22, week=3
    )
    mu_std, _, meta_std = data_ingest.apply_bayesian_usage_shrinkage(
        p_std, snap_pct=0.80, target_share=0.22, week=3
    )
    assert meta_cle["prior_weight"] < meta_std["prior_weight"], "Expected CLE prior weight to decay faster than standard"
    assert mu_cle < mu_std, f"Expected CLE WR projection ({mu_cle}) to decay faster toward low observed (8.0) than standard ({mu_std})"
    print(f"  [PASS] Longitudinal Deep Dive & Bayesian Decay: CLE {len(cle_dive['timeline_df'])}-Week Timeline Built; CLE WR Prior Weight={meta_cle['prior_weight']} vs Standard={meta_std['prior_weight']} (Mu: {mu_cle:.2f} vs {mu_std:.2f}).")


def test_idiosyncratic_trade_analyzer():
    print("\n=== Testing 14: Idiosyncratic Trade Analyzer & Bilateral 9-Slot Lineup Engine ===")

    rosters = trade_analyzer.get_default_bodhizafa_trade_rosters()
    eatin_roster = rosters["Eatin TDs like Groceries"]
    bodhizafa_roster = rosters["BODHIZAFA"]

    eval_res = trade_analyzer.evaluate_trade_package(
        team_a_name="Eatin TDs like Groceries",
        team_a_roster=eatin_roster,
        team_b_name="BODHIZAFA",
        team_b_roster=bodhizafa_roster,
        team_a_gives_names=["DJ Moore", "Chase Brown", "Jalen Coker"],
        team_a_receives_names=["Jonathan Taylor", "Dontayvion Wicks", "Devaughn Vele", "Seth McGowan"],
        n_simulations=5000,
    )

    m = eval_res["metrics"]
    assert m["wk6_starter_delta"] > 6.5, f"Expected Wk6 starter delta > +6.5 pts, got {m['wk6_starter_delta']}"
    assert m["ros_model_effective_delta"] > 2.0, f"Expected positive ROS model effective delta, got {m['ros_model_effective_delta']}"
    assert abs(m["fs_incoming_active_vorp"] - 163.14) < 0.5, f"Expected Taylor FS active VORP ~163.14, got {m['fs_incoming_active_vorp']}"
    assert abs(m["fs_outgoing_active_vorp"] - 336.66) < 0.5, f"Expected Outgoing FS active VORP ~336.66, got {m['fs_outgoing_active_vorp']}"
    assert abs(m["fs_incoming_auc"] - 27.40) < 0.2, f"Expected Taylor FS Auction ~$27.40, got {m['fs_incoming_auc']}"
    assert abs(m["fs_outgoing_auc"] - 56.54) < 0.2, f"Expected Outgoing FS Auction ~$56.54, got {m['fs_outgoing_auc']}"
    assert m["bayes_ros_starter_delta"] < 0, f"Expected negative Bayesian 50/50 ROS starter delta for 4-for-3, got {m['bayes_ros_starter_delta']}"
    assert len(eval_res["synergies_a"]["handcuff_pairs"]) == 1, "Expected IND Backfield Handcuff pair (Taylor + McGowan)"
    assert eval_res["synergies_a"]["handcuff_pairs"][0]["handcuff_rb"] == "Seth McGowan"
    assert eval_res["dropped_players_a"][0]["name"] == "Rico Dowdle", f"Expected pre-existing bench drop 'Rico Dowdle', got {eval_res['dropped_players_a'][0]['name']}"

    clean_verdict = eval_res["verdict"].encode("ascii", errors="ignore").decode("ascii").strip()
    print(f"  [PASS] 4-for-3 Trade Evaluation: Verdict='{clean_verdict}' (Grade {eval_res['grade']}), Wk6 Delta={m['wk6_starter_delta']:+.2f}, FS Active VORP Delta={m['fs_net_active_vorp_delta']:+.1f} (${m['fs_net_auc_delta']:+.2f} Auc), Bayes 50/50 Delta={m['bayes_ros_starter_delta']:+.2f} PPG.")



if __name__ == "__main__":
    print("==================================================")
    print("RUNNING FANTASY MONTE CARLO ENGINE TEST SUITE")
    print("==================================================")
    test_data_ingest()
    test_simulation()
    test_vorp()
    test_advanced_features()
    test_scheme_lab()
    test_funnel_synergy_and_trade_scanner()
    test_injury_and_ir_handling()
    test_idiosyncratic_scoring()
    test_espn_credentials_and_waiver_isolation()
    test_archetype_volatility_and_underdog_accuracy()
    test_copula_and_next_horizon()
    test_athletic_excel_and_hybrid_blending()
    test_dual_newsletter_and_longitudinal_narratives()
    test_idiosyncratic_trade_analyzer()
    print("\n==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! (100% HEALTHY)")
    print("==================================================")




