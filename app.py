"""
app.py
Weekly Fantasy Football Monte Carlo Decision Engine.
Streamlit application featuring:
- Seamless ESPN League Integration: Auto-syncs active league, rosters, and live free agents.
- Head-to-Head Matchup Selector: Choose any weekly matchup or custom team vs team.
- 10,000-run Monte Carlo simulation with position-specific distribution models (QB Normal, RB Gamma, WR/TE Log-Normal).
- Interactive Plotly outcome curves (PDF, CDF, Win Margin).
- Start/Sit comparative decision engine with win-rate leverage.
- Dynamic weekly VORP rankings against real waiver wire baselines.
- The Odds API & Polymarket Gamma API crowd risk feeds.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde

import os
from typing import Dict, Any, List, Optional, Tuple
import data_ingest
import simulation
import vorp
import pdf_parser
import scheme_db
import scoring_config

# Page setup
st.set_page_config(
    page_title="NFL Fantasy Monte Carlo Engine",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .league-banner {
        background: linear-gradient(135deg, #064e3b 0%, #0f172a 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 14px 22px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .league-title {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.01em;
    }
    .league-subtitle {
        color: #34d399;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 4px 0;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.8rem;
    }
    .green-val { color: #10b981; }
    .blue-val { color: #38bdf8; }
    .amber-val { color: #f59e0b; }
    .purple-val { color: #c084fc; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# INITIALIZE / LOAD ESPN LEAGUE & SESSION STATE
# ---------------------------------------------------------------------------

if "espn_league" not in st.session_state:
    st.session_state.espn_league = None

if "espn_sync_error" not in st.session_state:
    st.session_state.espn_sync_error = None

if "espn_last_attempt" not in st.session_state:
    st.session_state.espn_last_attempt = None

if "selected_team_a" not in st.session_state:
    st.session_state.selected_team_a = None

if "selected_team_b" not in st.session_state:
    st.session_state.selected_team_b = None


# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------------
st.sidebar.title("🏈 Engine Controls")

if st.sidebar.button("🔄 Clear Cache & Re-Sync League", use_container_width=True):
    st.cache_data.clear()
    st.session_state.espn_league = None
    st.session_state.espn_sync_error = None
    st.session_state.espn_last_attempt = None
    st.session_state.selected_team_a = None
    st.session_state.selected_team_b = None
    st.rerun()

st.sidebar.markdown("### ⚙️ League Scoring Engine")
st.sidebar.success("🏆 **Active: Idiosyncratic League Rules**")
with st.sidebar.expander("📋 View League Scoring Settings", expanded=False):
    st.markdown("""
    - **Passing:** 6.0 pt TD · 0.1 pt/comp · 0.04 pt/yd · -1.0 INT
      - *Stacking Bonuses:* +1 (40+ yd TD), +4 (50+ yd TD), +2 (300+ yds), +3 (400+ yds)
    - **Rushing:** 6.0 pt TD · 0.10 pt/yd
      - *Stacking Bonuses:* +1 (40+ yd TD), +2 (50+ yd TD), +1 (100+ yds), +2 (200+ yds)
    - **Receiving:** 1.0 Full PPR · 6.0 pt TD · 0.10 pt/yd
      - *Stacking Bonuses:* +1 (40+ yd TD), +2 (50+ yd TD), +1 (100+ yds), +2 (200+ yds)
    - **Turnovers / Returns:** -2.0 Fumble Lost · 1 pt / 25 KR & PR yds (Offensive players)
    - **D/ST:** Sacks (+1), FF (+1), FR (+2), INT (+2), Safety (+2), PA & YA tiered brackets
    - **Kickers:** Excluded
    """)

scoring_format = st.sidebar.selectbox(
    "Scoring Format",
    ["My League (Idiosyncratic 6pt Pass TD)", "Standard Full PPR (4pt Pass TD)", "Standard Half PPR (0.5)"],
    index=0
)
ppr_val = 1.0 if ("Full" in scoring_format or "Idiosyncratic" in scoring_format) else (0.5 if "Half" in scoring_format else 0.0)
active_scoring_cfg = scoring_config.get_scoring_config_by_name(scoring_format)

n_iterations = st.sidebar.slider(
    "Monte Carlo Iterations",
    min_value=5000,
    max_value=25000,
    value=10000,
    step=1000,
    help="Number of simulations per matchup (10,000 recommended for high tail accuracy)."
)

# Ingest credentials from secrets (or environment variables)
espn_creds = data_ingest.get_espn_credentials()

# Auto-sync ESPN if credentials present and (not yet loaded OR league ID/year changed)
target_league_id = espn_creds.get("league_id")
target_year = espn_creds.get("year", 2024)
current_synced_id = st.session_state.espn_league.get("league_id") if st.session_state.espn_league else None
current_synced_year = st.session_state.espn_league.get("year") if st.session_state.espn_league else None

attempt_key = (target_league_id, target_year, scoring_format)

if target_league_id and (
    st.session_state.espn_league is None or
    current_synced_id != target_league_id or
    current_synced_year != target_year
) and st.session_state.espn_last_attempt != attempt_key:
    st.session_state.espn_last_attempt = attempt_key
    with st.spinner(f"Connecting to ESPN Fantasy League ({target_league_id}, Year {target_year})..."):
        st.cache_data.clear()
        sync_result = data_ingest.sync_espn_league(
            league_id=int(target_league_id),
            year=int(target_year),
            espn_s2=espn_creds.get("espn_s2"),
            swid=espn_creds.get("swid"),
            ppr=ppr_val,
            scoring_cfg=active_scoring_cfg,
            user_team_hint=espn_creds.get("user_team_hint")
        )
        if sync_result.get("success"):
            st.session_state.espn_league = sync_result
            st.session_state.espn_sync_error = None
            teams = sync_result.get("teams", [])
            user_t = sync_result.get("user_team_name")
            user_opp = sync_result.get("user_opp_name")
            league_title = f"🏆 {sync_result.get('league_name', 'ESPN League')}"
            st.session_state["active_data_source"] = league_title
            if user_t:
                st.session_state.selected_team_a = user_t
                st.session_state["team_a_select"] = user_t
                st.session_state.selected_team_b = user_opp or (teams[1]["team_name"] if len(teams) > 1 else teams[0]["team_name"])
                st.session_state["team_b_select"] = st.session_state.selected_team_b
            elif len(teams) >= 2:
                st.session_state.selected_team_a = teams[0]["team_name"]
                st.session_state["team_a_select"] = teams[0]["team_name"]
                st.session_state.selected_team_b = teams[1]["team_name"]
                st.session_state["team_b_select"] = teams[1]["team_name"]
            st.rerun()
        else:
            st.session_state.espn_league = None
            st.session_state.espn_sync_error = sync_result.get("error", "Unknown ESPN connection error")

st.sidebar.markdown("---")
st.sidebar.subheader("🏆 Active League Mode")

has_espn = st.session_state.espn_league is not None and st.session_state.espn_league.get("success")

mode_options = []
if has_espn:
    league_label = f"🏆 {st.session_state.espn_league.get('league_name', 'ESPN League')}"
    mode_options.append(league_label)
mode_options.append("🏈 Curated Demo Matchup (Bills vs Lions)")

# Ensure session state defaults to synced league if available
if has_espn:
    curr_choice = st.session_state.get("active_data_source")
    if curr_choice not in mode_options or curr_choice == "🏈 Curated Demo Matchup (Bills vs Lions)":
        st.session_state["active_data_source"] = league_label

default_mode_idx = 0
if "active_data_source" in st.session_state and st.session_state["active_data_source"] in mode_options:
    default_mode_idx = mode_options.index(st.session_state["active_data_source"])

active_mode_selection = st.sidebar.radio(
    "Data Source",
    mode_options,
    index=default_mode_idx,
    key="active_data_source"
)

APP_BUILD_VERSION = "2026.09.17-v2.1"
st.sidebar.caption(f"⚡ Engine Build: `{APP_BUILD_VERSION}`")

# ESPN Sync Management in Sidebar
with st.sidebar.expander("⚙️ ESPN Connection Settings", expanded=not has_espn):
    if espn_creds.get("found_in_secrets"):
        st.success(
            f"✓ **Secrets Ingested**: League `{espn_creds.get('league_id')}` · Year `{espn_creds.get('year')}`\n\n"
            + ("🔐 *Private Auth cookies active (espn_s2 & swid)*" if espn_creds.get("espn_s2") else "🌐 *Public League Mode (no cookies)*")
        )
    else:
        st.info("💡 To auto-connect, configure your league credentials in `.streamlit/secrets.toml`.")

    default_lid_val = str(espn_creds.get("league_id") or "")
    l_id = st.text_input("League ID", value=default_lid_val, help="ESPN numeric league ID from browser URL")
    default_yr_val = int(espn_creds.get("year") or 2024)
    yr = st.number_input("Year", min_value=2020, max_value=2026, value=default_yr_val)
    s2 = st.text_input("espn_s2", value=espn_creds.get("espn_s2") or "", type="password", help="espn_s2 cookie string for private leagues")
    swid = st.text_input("SWID", value=espn_creds.get("swid") or "", help="SWID cookie string e.g. {12345678-ABCD-...}")

    if st.button("🔄 Sync / Re-sync ESPN League", use_container_width=True):
        if not l_id.strip():
            st.error("Please enter a valid numeric League ID.")
        else:
            with st.spinner("Connecting to ESPN..."):
                sync_res = data_ingest.sync_espn_league(
                    league_id=int(l_id.strip()),
                    year=int(yr),
                    espn_s2=s2.strip() if s2 else None,
                    swid=swid.strip() if swid else None,
                    ppr=ppr_val,
                    scoring_cfg=active_scoring_cfg,
                    user_team_hint=espn_creds.get("user_team_hint")
                )
                if sync_res.get("success"):
                    st.session_state.espn_league = sync_res
                    st.session_state.espn_sync_error = None
                    st.session_state.espn_last_attempt = (int(l_id.strip()), int(yr), scoring_format)
                    teams = sync_res.get("teams", [])
                    user_t = sync_res.get("user_team_name")
                    user_opp = sync_res.get("user_opp_name")
                    league_title = f"🏆 {sync_res.get('league_name', 'ESPN League')}"
                    st.session_state["active_data_source"] = league_title
                    if user_t:
                        st.session_state.selected_team_a = user_t
                        st.session_state["team_a_select"] = user_t
                        st.session_state.selected_team_b = user_opp or (teams[1]["team_name"] if len(teams) > 1 else teams[0]["team_name"])
                        st.session_state["team_b_select"] = st.session_state.selected_team_b
                    elif len(teams) >= 2:
                        st.session_state.selected_team_a = teams[0]["team_name"]
                        st.session_state["team_a_select"] = teams[0]["team_name"]
                        st.session_state.selected_team_b = teams[1]["team_name"]
                        st.session_state["team_b_select"] = teams[1]["team_name"]
                    st.success(f"Connected: {sync_res.get('league_name')} ({len(teams)} teams)")
                    st.rerun()
                else:
                    st.session_state.espn_sync_error = sync_res.get("error")
                    st.error(f"Sync error: {sync_res.get('error')}")

# The Odds API Key in Sidebar
odds_api_key = data_ingest.get_odds_api_key()
with st.sidebar.expander("🔑 The Odds API Key"):
    user_key = st.text_input("API Key", value=odds_api_key, type="password")
    if user_key:
        odds_api_key = user_key


# ---------------------------------------------------------------------------
# PREPARE ACTIVE MATCHUP DATA
# ---------------------------------------------------------------------------
is_espn_active = has_espn and (active_mode_selection.startswith("🏆"))

# Display Sync Notice if live sync failed and app is falling back to demo
if st.session_state.get("espn_sync_error") and not has_espn:
    st.warning(
        f"⚠️ **ESPN League Live Sync Notice:** `{st.session_state.espn_sync_error}`\n\n"
        "The engine is currently running in **Curated Demo Matchup** mode because the live ESPN sync could not complete. "
        "Please check your credentials in `.streamlit/secrets.toml` (verify `league_id`, `year`, `espn_s2`, and `swid`) "
        "or adjust connection settings in the sidebar under **ESPN Connection Settings**."
    )

team_a_dict = {}
team_b_dict = {}

if is_espn_active:
    espn_info = st.session_state.espn_league
    teams_list = espn_info.get("teams", [])
    team_names = [t["team_name"] for t in teams_list]
    team_map = espn_info.get("team_map", {})
    fa_count = len(espn_info.get("free_agents", []))

    # Top League Banner
    st.markdown(f"""
    <div class="league-banner">
        <div>
            <div class="league-title">🏆 {espn_info.get('league_name')} ({espn_info.get('year')})</div>
            <div class="league-subtitle">● SYNCED LIVE &nbsp;·&nbsp; {len(teams_list)} TEAMS &nbsp;·&nbsp; {fa_count} WAIVER FREE AGENTS &nbsp;·&nbsp; WEEK {espn_info.get('current_week', 1)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Matchup Selection Bar
    st.markdown("### ⚔️ Select Matchup")
    m_type_col1, m_type_col2 = st.columns([1, 1])

    with m_type_col1:
        # Team A (User Team)
        default_a_idx = 0
        if st.session_state.get("team_a_select") in team_names:
            default_a_idx = team_names.index(st.session_state["team_a_select"])
        elif st.session_state.get("selected_team_a") in team_names:
            default_a_idx = team_names.index(st.session_state.selected_team_a)
        else:
            for idx, t in enumerate(team_names):
                if "eatin" in t.lower():
                    default_a_idx = idx
                    break

        selected_a = st.selectbox(
            "🏠 Team A (My Team)",
            team_names,
            index=default_a_idx,
            key="team_a_select"
        )
        st.session_state.selected_team_a = selected_a

    with m_type_col2:
        # Team B (Opponent Team)
        available_b_options = [t for t in team_names if t != selected_a] or team_names
        default_b_idx = 0
        if st.session_state.get("team_b_select") in available_b_options:
            default_b_idx = available_b_options.index(st.session_state["team_b_select"])
        elif st.session_state.get("selected_team_b") in available_b_options:
            default_b_idx = available_b_options.index(st.session_state.selected_team_b)
        else:
            for idx, t in enumerate(available_b_options):
                if "puka" in t.lower():
                    default_b_idx = idx
                    break

        selected_b = st.selectbox(
            "🚀 Team B (Opponent)",
            available_b_options,
            index=default_b_idx,
            key="team_b_select"
        )
        st.session_state.selected_team_b = selected_b

    # Build active matchup data from chosen ESPN teams
    team_a_dict = team_map.get(selected_a, teams_list[0])
    team_b_dict = team_map.get(selected_b, teams_list[1] if len(teams_list) > 1 else teams_list[0])

    current_data = data_ingest.build_matchup_from_espn_teams(
        team_a_dict,
        team_b_dict,
        free_agents_list=espn_info.get("free_agents", [])
    )

else:
    # Standalone / Curated User Mode
    st.markdown("""
    <div class="league-banner">
        <div>
            <div class="league-title">🏈 Curated Matchup: Eatin TDs like Groceries vs Puka Shell Queen</div>
            <div class="league-subtitle">ACTIVE ROSTER MODE &nbsp;·&nbsp; VEGAS PROPS &nbsp;·&nbsp; POLYMARKET RISK &nbsp;·&nbsp; DYNAMIC WAIVER WIRE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    current_data = data_ingest.get_default_matchup_and_waivers(ppr=ppr_val, scoring_cfg=active_scoring_cfg)


team_a_roster = current_data["team_a_roster"]
team_b_roster = current_data["team_b_roster"]
waiver_pool = current_data["waiver_pool"]
team_a_name = current_data["team_a_name"]
team_b_name = current_data["team_b_name"]
raw_bench_a = team_a_dict.get("bench", []) if is_espn_active else current_data.get("team_a_bench", [])
team_a_bench = [
    p for p in raw_bench_a
    if not p.get("is_inactive")
    and p.get("slot") not in ["IR", "IR_ELIGIBLE"]
    and float(p.get("p_active", 1.0)) > 0.0
    and float(p.get("mu", 0.0)) > 0.0
    and str(p.get("injury_status", "")).upper() not in ["IR", "INJURY_RESERVE", "OUT", "O", "PUP", "SUSPENDED", "DOUBTFUL"]
]


def find_contested_lineup_slots(
    starters: List[Dict[str, Any]],
    bench: List[Dict[str, Any]],
    team_a_win_rate: float,
    threshold: float = 2.5
) -> List[Dict[str, Any]]:
    """
    Surface Contested Lineup Slots: any active bench asset within 2.5 projected points of a starter,
    or higher (bench_mu >= starter_mu - 2.5).
    Pairs each contested starter with the challenger, calculating Win% Leverage, Floor/Ceiling delta,
    and assigning a dynamic contextual badge ('Safe Floor Play' vs 'Underdog Ceiling Swing').
    """
    contested = []
    seen_pairs = set()

    for starter in starters:
        s_pos = starter.get("position", "").upper()
        s_slot = starter.get("slot", s_pos).upper()
        s_mu = float(starter.get("mu", 0.0))
        s_sig = float(starter.get("sigma", 0.0))
        s_name = starter.get("name", "")

        # Compute starter Floor (P10) and Ceiling (P90)
        s_p10 = round(max(0.0, s_mu - 1.28 * s_sig), 1)
        s_p90 = round(max(0.0, s_mu + 1.28 * s_sig), 1)

        # Find eligible bench assets
        for b in bench:
            b_name = b.get("name", "")
            if b_name == s_name or b.get("is_inactive") or float(b.get("p_active", 1.0)) <= 0:
                continue
            b_pos = b.get("position", "").upper()
            b_mu = float(b.get("mu", 0.0))
            b_sig = float(b.get("sigma", 0.0))

            # Eligibility check
            is_eligible = False
            if "QB" in s_slot or s_pos == "QB":
                is_eligible = (b_pos == "QB")
            elif "TE" in s_slot and s_slot == "TE":
                is_eligible = (b_pos == "TE")
            elif "FLEX" in s_slot or "WR/TE" in s_slot or "RB/WR/TE" in s_slot:
                is_eligible = (b_pos in ["RB", "WR", "TE"])
            elif "RB" in s_slot or s_pos == "RB":
                is_eligible = (b_pos == "RB")
            elif "WR" in s_slot or s_pos == "WR":
                is_eligible = (b_pos == "WR")

            if not is_eligible:
                continue

            # Within 2.5 points or higher
            if b_mu >= (s_mu - threshold):
                pair_key = (s_name, b_name)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                b_p10 = round(max(0.0, b_mu - 1.28 * b_sig), 1)
                b_p90 = round(max(0.0, b_mu + 1.28 * b_sig), 1)

                delta_mu = round(b_mu - s_mu, 1)
                delta_floor = round(b_p10 - s_p10, 1)
                delta_ceiling = round(b_p90 - s_p90, 1)

                # Estimate win% leverage (~1.8% per net projected point)
                win_leverage = round(delta_mu * 0.018, 3)

                # Contextual badge assignment
                # Underdog regime (win_rate <= 48%): chasing upside & ceiling
                if team_a_win_rate <= 0.48:
                    if delta_ceiling > 0:
                        badge = "⚡ Underdog Ceiling Swing"
                        badge_color = "#f59e0b"
                        badge_bg = "rgba(245, 158, 11, 0.18)"
                        rationale = f"Trailing projection ({team_a_win_rate*100:.1f}% win rate). {b_name} elevates 90th percentile ceiling (+{delta_ceiling:+.1f} pts) to unlock upset win equity."
                    else:
                        badge = "🛡️ Safe Floor Play"
                        badge_color = "#10b981"
                        badge_bg = "rgba(16, 185, 129, 0.18)"
                        rationale = f"Starter {s_name} holds the superior floor (P10: {s_p10:.1f} vs {b_p10:.1f} pts)."
                # Favorite regime (win_rate >= 52%): protecting floor & minimizing variance
                elif team_a_win_rate >= 0.52:
                    if delta_floor >= 0:
                        badge = "🛡️ Safe Floor Play"
                        badge_color = "#10b981"
                        badge_bg = "rgba(16, 185, 129, 0.18)"
                        rationale = f"Projected favorite ({team_a_win_rate*100:.1f}% win rate). {b_name} raises lineup floor (+{delta_floor:+.1f} P10) to protect win equity."
                    else:
                        badge = "🛡️ Safe Floor Play"
                        badge_color = "#10b981"
                        badge_bg = "rgba(16, 185, 129, 0.18)"
                        rationale = f"Projected favorite ({team_a_win_rate*100:.1f}% win rate). Starter {s_name} insulates lead with higher floor (P10: {s_p10:.1f} vs {b_p10:.1f} pts)."
                else:
                    if delta_mu > 0:
                        badge = "🎯 Optimal Win% Pick"
                        badge_color = "#38bdf8"
                        badge_bg = "rgba(56, 189, 248, 0.18)"
                        rationale = f"Toss-up matchup. Subbing in {b_name} gains +{delta_mu:+.1f} expected points."
                    else:
                        badge = "🛡️ Safe Floor Play"
                        badge_color = "#10b981"
                        badge_bg = "rgba(16, 185, 129, 0.18)"
                        rationale = f"Toss-up matchup. Starter {s_name} projects for {s_mu:.1f} pts with higher floor."

                contested.append({
                    "slot": s_slot,
                    "starter": starter,
                    "challenger": b,
                    "starter_name": s_name,
                    "challenger_name": b_name,
                    "starter_mu": s_mu,
                    "challenger_mu": b_mu,
                    "starter_p10": s_p10,
                    "challenger_p10": b_p10,
                    "starter_p90": s_p90,
                    "challenger_p90": b_p90,
                    "delta_mu": delta_mu,
                    "delta_floor": delta_floor,
                    "delta_ceiling": delta_ceiling,
                    "win_leverage": win_leverage,
                    "badge": badge,
                    "badge_color": badge_color,
                    "badge_bg": badge_bg,
                    "rationale": rationale
                })

    contested.sort(key=lambda x: abs(x["delta_mu"]))
    return contested


# ---------------------------------------------------------------------------
# MAIN INTERFACE TABS
# ---------------------------------------------------------------------------
tab_matchup, tab_startsit, tab_vorp, tab_feeds, tab_scheme = st.tabs([
    "⚡ Sunday Morning Command Center",
    "⚖️ Start / Sit Comparative Engine",
    "📋 Tuesday Waiver & FAAB Claim Sheet",
    "📡 Vegas Odds & Polymarket Feed",
    "🔬 Scheme Lab & Newsletter Upload"
])


# ===========================================================================
# TAB 1: SUNDAY MORNING COMMAND CENTER
# ===========================================================================
with tab_matchup:
    # Run 10,000-run simulation
    with st.spinner(f"Running {n_iterations:,} Monte Carlo simulations for {team_a_name} vs {team_b_name}..."):
        matchup_result = simulation.simulate_matchup(
            team_a_roster,
            team_b_roster,
            n_simulations=n_iterations
        )

    p_win_a = matchup_result["p_win_a"]
    p_win_b = matchup_result["p_win_b"]
    a_summary = matchup_result["team_a_summary"]
    b_summary = matchup_result["team_b_summary"]
    m_summary = matchup_result["margin_summary"]
    ml_str = f"-{int(p_win_a/(1-p_win_a)*100)}" if p_win_a > 0.5 else f"+{int((1-p_win_a)/max(1e-4, p_win_a)*100)}"

    # Top KPI Metrics Bar
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{team_a_name} P(Win)</div>
            <div class="metric-value green-val">{p_win_a * 100:.1f}%</div>
            <div class="metric-sub">Implied Moneyline: {ml_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{team_a_name} Proj. Points</div>
            <div class="metric-value blue-val">{a_summary['mean']:.1f}</div>
            <div class="metric-sub">80% Conf: [{a_summary['p10']:.1f} - {a_summary['p90']:.1f}] · Med: {a_summary['median']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{team_b_name} Proj. Points</div>
            <div class="metric-value amber-val">{b_summary['mean']:.1f}</div>
            <div class="metric-sub">80% Conf: [{b_summary['p10']:.1f} - {b_summary['p90']:.1f}] · Med: {b_summary['median']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Expected Margin of Victory</div>
            <div class="metric-value purple-val">{m_summary['mean']:+0.1f}</div>
            <div class="metric-sub">Median Margin: {m_summary['median']:+0.1f} · Std: ±{m_summary['std']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Centered Win% Gauge + Clean Margin of Victory Distribution
    col_gauge, col_margin = st.columns([1, 1.85])

    with col_gauge:
        st.subheader("Win Probability Gauge")
        gauge_color = "#10b981" if p_win_a >= 0.5 else "#ef4444"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_win_a * 100,
            number={'suffix': "%", 'valueformat': ".1f", 'font': {'size': 44, 'color': gauge_color}},
            title={'text': f"<b>{team_a_name}</b><br><span style='font-size:0.82em;color:#94a3b8'>ML: {ml_str} · Exp. Margin: {m_summary['mean']:+0.1f}</span>", 'font': {'size': 16, 'color': '#f8fafc'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b", 'nticks': 6},
                'bar': {'color': gauge_color, 'thickness': 0.28},
                'bgcolor': "#1e293b",
                'borderwidth': 1,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.15)'},
                    {'range': [50, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 3},
                    'thickness': 0.75,
                    'value': 50.0
                }
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=20, r=20, t=35, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_margin:
        st.subheader("Margin of Victory Distribution")
        margins = matchup_result["margins"]
        fig_margin = go.Figure()
        losses = margins[margins < 0]
        fig_margin.add_trace(go.Histogram(
            x=losses,
            name=f"{team_b_name} Win Zone (<0)",
            marker_color="#ef4444",
            opacity=0.78,
            nbinsx=40
        ))
        wins = margins[margins >= 0]
        fig_margin.add_trace(go.Histogram(
            x=wins,
            name=f"{team_a_name} Win Zone (≥0)",
            marker_color="#10b981",
            opacity=0.78,
            nbinsx=40
        ))
        fig_margin.add_vline(x=0, line_width=3, line_color="#ffffff", line_dash="solid", annotation_text="TIE LINE (0)", annotation_position="top")
        fig_margin.add_vline(x=m_summary["mean"], line_width=2, line_color="#c084fc", line_dash="dash", annotation_text=f"Mean ({m_summary['mean']:+0.1f})")

        fig_margin.update_layout(
            barmode="overlay",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,41,59,0.5)",
            xaxis_title=f"Margin of Victory ({team_a_name} vs {team_b_name})",
            yaxis_title="Simulated Frequency (10,000 Runs)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=280,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_margin, use_container_width=True)

    # 2. Contested Lineup Slots & Pre-Paired Decision Cards
    st.markdown("---")
    st.markdown("### ⚔️ Contested Lineup Slots & Sunday Decision Cards")
    st.caption("Automatically surfaced borderline starter decisions: any bench asset within 2.5 projected points of a starter, or higher. Pre-paired with net Win% leverage, Floor/Ceiling delta, and contextual badges ('Safe Floor Play' vs 'Underdog Ceiling Swing').")

    contested_slots = find_contested_lineup_slots(team_a_roster, team_a_bench, p_win_a, threshold=2.5)

    if contested_slots:
        card_cols = st.columns(min(len(contested_slots), 2))
        for idx, item in enumerate(contested_slots):
            c_col = card_cols[idx % 2]
            with c_col:
                delta_sign = "+" if item["delta_mu"] >= 0 else ""
                delta_floor_sign = "+" if item["delta_floor"] >= 0 else ""
                delta_ceiling_sign = "+" if item["delta_ceiling"] >= 0 else ""
                win_lev_sign = "+" if item["win_leverage"] >= 0 else ""

                card_html = f"""
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                        <span style="font-weight: 800; font-size: 1.05rem; color: #f8fafc;">
                            Slot: <span style="color: #38bdf8;">{item['slot']}</span> &nbsp;·&nbsp; {item['starter_name']} vs {item['challenger_name']}
                        </span>
                        <span style="background: {item['badge_bg']}; color: {item['badge_color']}; border: 1px solid {item['badge_color']}; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 800;">
                            {item['badge']}
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 12px;">
                        <div style="background: #0f172a; padding: 10px 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                            <div style="font-size: 0.70rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Projected (Med)</div>
                            <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin-top: 2px;">{item['challenger_mu']:.1f} vs {item['starter_mu']:.1f}</div>
                            <div style="font-size: 0.75rem; color: {'#10b981' if item['delta_mu'] >= 0 else '#ef4444'}; font-weight: 700;">{delta_sign}{item['delta_mu']:.1f} pts</div>
                        </div>
                        <div style="background: #0f172a; padding: 10px 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                            <div style="font-size: 0.70rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Floor (P10)</div>
                            <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin-top: 2px;">{item['challenger_p10']:.1f} vs {item['starter_p10']:.1f}</div>
                            <div style="font-size: 0.75rem; color: {'#10b981' if item['delta_floor'] >= 0 else '#ef4444'}; font-weight: 700;">{delta_floor_sign}{item['delta_floor']:.1f} floor</div>
                        </div>
                        <div style="background: #0f172a; padding: 10px 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                            <div style="font-size: 0.70rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Ceiling (P90)</div>
                            <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin-top: 2px;">{item['challenger_p90']:.1f} vs {item['starter_p90']:.1f}</div>
                            <div style="font-size: 0.75rem; color: {'#10b981' if item['delta_ceiling'] >= 0 else '#ef4444'}; font-weight: 700;">{delta_ceiling_sign}{item['delta_ceiling']:.1f} ceiling</div>
                        </div>
                        <div style="background: #0f172a; padding: 10px 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                            <div style="font-size: 0.70rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Win% Leverage</div>
                            <div style="font-size: 1.15rem; font-weight: 800; color: {'#10b981' if item['win_leverage'] >= 0 else '#f59e0b'}; margin-top: 2px;">{win_lev_sign}{item['win_leverage']*100:.1f}%</div>
                            <div style="font-size: 0.75rem; color: #94a3b8;">Team Net Impact</div>
                        </div>
                    </div>
                    <div style="font-size: 0.82rem; color: #cbd5e1; background: rgba(15, 23, 42, 0.75); padding: 10px 14px; border-radius: 6px; border-left: 3px solid {item['badge_color']};">
                        💡 <b>Sunday Rationale:</b> {item['rationale']}
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.success(f"✅ **No Contested Slots:** All starters for **{team_a_name}** hold at least a +2.5 projected point cushion over your bench depth.")

    # 3. Collapsed "Advanced Model Diagnostics" Expander
    st.markdown("---")
    with st.expander("🔬 Advanced Model Diagnostics (KDE Density Curves, Cumulative Exceedance CDF & Defensive Scheme Breakdown)", expanded=False):
        col_chart1, col_chart2 = st.columns(2)
        scores_a = matchup_result["team_a_scores"]
        scores_b = matchup_result["team_b_scores"]

        with col_chart1:
            st.subheader("Probability Density Curves (KDE)")
            x_grid = np.linspace(min(scores_a.min(), scores_b.min()) - 10, max(scores_a.max(), scores_b.max()) + 10, 300)
            kde_a = gaussian_kde(scores_a)(x_grid)
            kde_b = gaussian_kde(scores_b)(x_grid)

            fig_kde = go.Figure()
            fig_kde.add_trace(go.Scatter(
                x=x_grid, y=kde_a,
                mode="lines",
                name=team_a_name,
                line=dict(color="#10b981", width=3),
                fill="tozeroy",
                fillcolor="rgba(16, 185, 129, 0.2)"
            ))
            fig_kde.add_trace(go.Scatter(
                x=x_grid, y=kde_b,
                mode="lines",
                name=team_b_name,
                line=dict(color="#f59e0b", width=3),
                fill="tozeroy",
                fillcolor="rgba(245, 158, 11, 0.2)"
            ))
            fig_kde.add_vline(x=a_summary["median"], line_width=2, line_dash="dash", line_color="#10b981", annotation_text=f"A Med {a_summary['median']:.1f}")
            fig_kde.add_vline(x=b_summary["median"], line_width=2, line_dash="dash", line_color="#f59e0b", annotation_text=f"B Med {b_summary['median']:.1f}")

            fig_kde.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,41,59,0.5)",
                xaxis_title="Weekly Fantasy Points",
                yaxis_title="Probability Density",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_kde, use_container_width=True)

        with col_chart2:
            st.subheader("Cumulative Win / Score Exceedance Curves (1 - CDF)")
            x_thresholds = np.linspace(min(scores_a.min(), scores_b.min()), max(scores_a.max(), scores_b.max()), 200)
            p_exceed_a = [np.mean(scores_a >= t) * 100 for t in x_thresholds]
            p_exceed_b = [np.mean(scores_b >= t) * 100 for t in x_thresholds]

            fig_cdf = go.Figure()
            fig_cdf.add_trace(go.Scatter(
                x=x_thresholds, y=p_exceed_a,
                mode="lines",
                name=f"{team_a_name} P(Score >= X)",
                line=dict(color="#10b981", width=3)
            ))
            fig_cdf.add_trace(go.Scatter(
                x=x_thresholds, y=p_exceed_b,
                mode="lines",
                name=f"{team_b_name} P(Score >= X)",
                line=dict(color="#f59e0b", width=3)
            ))
            fig_cdf.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,41,59,0.5)",
                xaxis_title="Score Threshold (Points)",
                yaxis_title="Probability of Scoring at least X (%)",
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_cdf, use_container_width=True)

        # Full Active Lineup Breakdown with Standardized Distribution Columns
        st.subheader(f"📋 Complete Starting Rosters with Floor / Median / Ceiling")
        col_la, col_lb = st.columns(2)
        with col_la:
            st.markdown(f"**{team_a_name} Starters**")
            df_a = pd.DataFrame(team_a_roster)
            cols_show_a = [c for c in ["slot", "name", "position", "mu", "sigma", "p_active", "prop_source", "injury_status"] if c in df_a.columns]
            df_display_a = df_a[cols_show_a].rename(columns={
                "slot": "Slot", "name": "Player", "position": "Pos", "p_active": "P(Active)", "prop_source": "Data Feed", "injury_status": "Status"
            })
            df_display_a = data_ingest.format_distribution_columns(df_display_a)
            st.dataframe(df_display_a, use_container_width=True, hide_index=True)

        with col_lb:
            st.markdown(f"**{team_b_name} Starters**")
            df_b = pd.DataFrame(team_b_roster)
            cols_show_b = [c for c in ["slot", "name", "position", "mu", "sigma", "p_active", "prop_source", "injury_status"] if c in df_b.columns]
            df_display_b = df_b[cols_show_b].rename(columns={
                "slot": "Slot", "name": "Player", "position": "Pos", "p_active": "P(Active)", "prop_source": "Data Feed", "injury_status": "Status"
            })
            df_display_b = data_ingest.format_distribution_columns(df_display_b)
            st.dataframe(df_display_b, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 2: START / SIT DECISION ENGINE
# ===========================================================================
with tab_startsit:
    st.subheader(f"Start / Sit Comparative Decision Engine ({team_a_name})")
    st.markdown("Compare two players on your roster or against waiver wire targets to evaluate outscore probability and team win-rate leverage.")

    col_ss1, col_ss2 = st.columns(2)

    # Candidate 1 options (all starters + bench on Team A if ESPN connected)
    all_team_a_players = team_a_roster
    if is_espn_active and "roster" in team_a_dict:
        all_team_a_players = team_a_dict["roster"]

    def format_player_display_label(p: Dict[str, Any]) -> str:
        name = p.get("name", "Unknown")
        pos = p.get("position", "WR")
        inj = str(p.get("injury_status", "")).upper()
        is_inact = (
            p.get("is_inactive", False)
            or p.get("p_active", 1.0) <= 0.0
            or p.get("slot") in ["IR", "IR_ELIGIBLE"]
            or inj in ["IR", "INJURY_RESERVE", "OUT", "O", "PUP", "SUSPENDED", "DOUBTFUL"]
        )
        if is_inact:
            badge = " [IR / OUT]" if inj in ["IR", "INJURY_RESERVE"] else f" [{inj or 'OUT'}]"
            return f"🚨 {name} ({pos}){badge}"
        elif inj in ["QUESTIONABLE", "Q"]:
            return f"⚠️ {name} ({pos}) [Q]"
        return f"{name} ({pos})"

    map_a = {p["name"]: format_player_display_label(p) for p in all_team_a_players}

    cand_a_options = [p["name"] for p in all_team_a_players]
    cand_a_idx = min(len(cand_a_options) - 1, 6)
    if "Jalen Coker" in cand_a_options:
        cand_a_idx = cand_a_options.index("Jalen Coker")

    with col_ss1:
        cand_a_name = st.selectbox(
            "Candidate 1 (Starter / Borderline Asset)",
            cand_a_options,
            index=cand_a_idx,
            format_func=lambda n: map_a.get(n, n)
        )
        cand_a = next(p for p in all_team_a_players if p["name"] == cand_a_name)

    # Candidate 2 options (bench challengers & waiver pool)
    challenger_candidates = [p for p in all_team_a_players if p["name"] != cand_a_name] + waiver_pool
    challenger_names = [p["name"] for p in challenger_candidates]
    map_b = {p["name"]: format_player_display_label(p) for p in challenger_candidates}

    cand_b_idx = 0
    if "Tony Pollard" in challenger_names:
        cand_b_idx = challenger_names.index("Tony Pollard")
    elif "DeMario Douglas" in challenger_names:
        cand_b_idx = challenger_names.index("DeMario Douglas")

    with col_ss2:
        cand_b_name = st.selectbox(
            "Candidate 2 (Bench Challenger / Waiver Target)",
            challenger_names,
            index=cand_b_idx,
            format_func=lambda n: map_b.get(n, n)
        )
        cand_b = next(p for p in challenger_candidates if p["name"] == cand_b_name)

    # Inactive Player Alert Warnings
    if cand_a.get("is_inactive") or cand_a.get("p_active", 1.0) <= 0.0 or cand_a.get("slot") in ["IR", "IR_ELIGIBLE"]:
        st.error(f"🚨 **CRITICAL LINEUP WARNING**: **{cand_a_name}** is designated **{cand_a.get('injury_status', 'IR')}** (0.0 projected pts, 0% active). This player CANNOT be started!")
    if cand_b.get("is_inactive") or cand_b.get("p_active", 1.0) <= 0.0 or cand_b.get("slot") in ["IR", "IR_ELIGIBLE"]:
        st.error(f"🚨 **CRITICAL LINEUP WARNING**: **{cand_b_name}** is designated **{cand_b.get('injury_status', 'IR')}** (0.0 projected pts, 0% active). This player CANNOT be started!")

    # Run Start/Sit Comparative Simulation
    with st.spinner("Analyzing Start/Sit distribution trade-offs..."):
        ss_result = simulation.evaluate_start_sit_decision(
            cand_a,
            cand_b,
            base_team_roster=team_a_roster,
            opponent_roster=team_b_roster,
            slot_name=cand_a.get("slot", "FLEX"),
            n_simulations=n_iterations
        )

    # Scheme Matchup & Defensive Funnel Breakdown
    funnel_a = ss_result.get("funnel_profile_a", {})
    funnel_b = ss_result.get("funnel_profile_b", {})
    align_a = ss_result.get("align_a", {})
    align_b = ss_result.get("align_b", {})
    funnel_note = ss_result.get("funnel_advantage_note", "")

    badge_color_a = "#10b981" if funnel_a.get("is_inside_funnel") else ("#f59e0b" if funnel_a.get("is_outside_funnel") else "#64748b")
    badge_color_b = "#10b981" if funnel_b.get("is_inside_funnel") else ("#f59e0b" if funnel_b.get("is_outside_funnel") else "#64748b")

    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid #3b82f6; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-weight: 800; color: #60a5fa; font-size: 1.0rem;">🎯 SCHEME MATCHUP & DEFENSIVE FUNNEL BREAKDOWN</span>
            <span style="font-size: 0.75rem; background: #1e3a8a; color: #93c5fd; padding: 2px 8px; border-radius: 4px; font-weight: 600;">SumerSports Synergy</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 10px;">
            <div style="background: #1e293b; padding: 10px 14px; border-radius: 6px; border-left: 3px solid #38bdf8;">
                <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">{cand_a_name} <span style="font-size: 0.8rem; color: #94a3b8; font-weight: normal;">({cand_a.get('position', 'WR')})</span></div>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 2px;">Alignment: <b style="color: #38bdf8;">{align_a.get('alignment_tag', 'WR')}</b> (Slot: {align_a.get('slot_snap_pct', 0)*100:.0f}% | Wide: {align_a.get('wide_snap_pct', 0)*100:.0f}%)</div>
                <div style="font-size: 0.82rem; color: #e2e8f0; margin-top: 4px;">Opponent: <b>{funnel_a.get('team', 'DEF')}</b> <span style="font-size: 0.75rem; background: {badge_color_a}22; color: {badge_color_a}; border: 1px solid {badge_color_a}; padding: 1px 6px; border-radius: 3px;">{funnel_a.get('funnel_classification', 'Balanced')}</span></div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">Opp MOF EPA: <b style="color: #f1f5f9;">{funnel_a.get('mof_epa_allowed', 0):.2f}</b> | Opp Quick Pres: <b style="color: #f1f5f9;">{funnel_a.get('quick_pressure_rate', 0)*100:.1f}%</b></div>
            </div>
            <div style="background: #1e293b; padding: 10px 14px; border-radius: 6px; border-left: 3px solid #c084fc;">
                <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">{cand_b_name} <span style="font-size: 0.8rem; color: #94a3b8; font-weight: normal;">({cand_b.get('position', 'WR')})</span></div>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 2px;">Alignment: <b style="color: #c084fc;">{align_b.get('alignment_tag', 'WR')}</b> (Slot: {align_b.get('slot_snap_pct', 0)*100:.0f}% | Wide: {align_b.get('wide_snap_pct', 0)*100:.0f}%)</div>
                <div style="font-size: 0.82rem; color: #e2e8f0; margin-top: 4px;">Opponent: <b>{funnel_b.get('team', 'DEF')}</b> <span style="font-size: 0.75rem; background: {badge_color_b}22; color: {badge_color_b}; border: 1px solid {badge_color_b}; padding: 1px 6px; border-radius: 3px;">{funnel_b.get('funnel_classification', 'Balanced')}</span></div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">Opp MOF EPA: <b style="color: #f1f5f9;">{funnel_b.get('mof_epa_allowed', 0):.2f}</b> | Opp Quick Pres: <b style="color: #f1f5f9;">{funnel_b.get('quick_pressure_rate', 0)*100:.1f}%</b></div>
            </div>
        </div>
        <div style="font-size: 0.85rem; color: #f1f5f9; background: #0f172a; padding: 8px 12px; border-radius: 6px; border: 1px solid #334155;">
            💡 <b>Scheme Calibration Recommendation:</b> {funnel_note}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tactical Objective Banner
    tac_mode = ss_result.get("tactical_mode", "TOSS-UP")
    tac_obj = ss_result.get("tactical_objective", "Maximize Win%")
    tac_bg = (
        "linear-gradient(135deg, #7f1d1d 0%, #1e293b 100%)" if "INACTIVE" in tac_mode
        else ("linear-gradient(135deg, #064e3b 0%, #1e293b 100%)" if tac_mode == "FAVORITE"
        else ("linear-gradient(135deg, #7c2d12 0%, #1e293b 100%)" if tac_mode == "UNDERDOG"
        else "linear-gradient(135deg, #1e3a8a 0%, #1e293b 100%)"))
    )
    tac_icon = "🚨" if "INACTIVE" in tac_mode else ("🛡️" if tac_mode == "FAVORITE" else ("⚡" if tac_mode == "UNDERDOG" else "⚖️"))

    st.markdown(f"""
    <div style="background: {tac_bg}; border: 1px solid #334155; border-radius: 10px; padding: 12px 18px; margin-bottom: 16px;">
        <span style="font-weight: 800; font-size: 1.05rem;">{tac_icon} {tac_mode} REGIME: {tac_obj}</span><br>
        <span style="color: #cbd5e1; font-size: 0.85rem;">{ss_result.get('tactical_rationale', '')}</span>
    </div>
    """, unsafe_allow_html=True)

    # Decision Scorecard
    sc_col1, sc_col2, sc_col3 = st.columns(3)
    with sc_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Head-to-Head Outscore %</div>
            <div class="metric-value blue-val">{ss_result['p_outscore_a'] * 100:.1f}%</div>
            <div class="metric-sub">{cand_a_name} vs {cand_b_name} ({ss_result['p_outscore_b']*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with sc_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Team Win% Impact</div>
            <div class="metric-value green-val">{ss_result['team_win_rate_a'] * 100:.1f}%</div>
            <div class="metric-sub">With {cand_a_name} vs {ss_result['team_win_rate_b']*100:.1f}% with {cand_b_name}</div>
        </div>
        """, unsafe_allow_html=True)

    with sc_col3:
        delta_str = f"{ss_result['delta_win_rate'] * 100:+0.1f}%"
        is_rec_a = ss_result["recommended"] == cand_a_name
        rec_color = "red-val" if "INACTIVE" in tac_mode else ("green-val" if is_rec_a else "amber-val")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Tuned Decision ({tac_mode})</div>
            <div class="metric-value {rec_color}">START {ss_result['recommended'].upper()}</div>
            <div class="metric-sub">Win% Delta: {delta_str}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Side-by-side distribution plots
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("Outcome Probability Densities")
        sim_a = ss_result["sim_a"]
        sim_b = ss_result["sim_b"]

        x_span = np.linspace(0, max(sim_a.max(), sim_b.max()) + 5, 200)
        kde_cand_a = gaussian_kde(sim_a)(x_span)
        kde_cand_b = gaussian_kde(sim_b)(x_span)

        fig_cand = go.Figure()
        fig_cand.add_trace(go.Scatter(
            x=x_span, y=kde_cand_a,
            mode="lines",
            name=f"{cand_a_name} ({cand_a.get('position')})",
            line=dict(color="#38bdf8", width=3),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.2)"
        ))
        fig_cand.add_trace(go.Scatter(
            x=x_span, y=kde_cand_b,
            mode="lines",
            name=f"{cand_b_name} ({cand_b.get('position')})",
            line=dict(color="#c084fc", width=3),
            fill="tozeroy",
            fillcolor="rgba(192, 132, 252, 0.2)"
        ))
        fig_cand.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,41,59,0.5)",
            xaxis_title="Fantasy Points",
            yaxis_title="Density",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_cand, use_container_width=True)

    with col_v2:
        st.subheader("Floor, Median & Ceiling Percentiles")
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=sim_a,
            name=cand_a_name,
            marker_color="#38bdf8",
            boxpoints=False
        ))
        fig_box.add_trace(go.Box(
            y=sim_b,
            name=cand_b_name,
            marker_color="#c084fc",
            boxpoints=False
        ))
        fig_box.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,41,59,0.5)",
            yaxis_title="Projected Points Distribution",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_box, use_container_width=True)

    sum_a = ss_result["summary_a"]
    sum_b = ss_result["summary_b"]
    p15_a = sum_a.get("p15", sum_a["p10"])
    p15_b = sum_b.get("p15", sum_b["p10"])
    st.info(
        f"**Tactical Objective ({tac_mode})**: "
        f"**{cand_a_name}** — P15 Floor: **{p15_a:.1f}** | Median: **{sum_a['median']:.1f}** | P90 Ceiling: **{sum_a['p90']:.1f}**\n\n"
        f"**{cand_b_name}** — P15 Floor: **{p15_b:.1f}** | Median: **{sum_b['median']:.1f}** | P90 Ceiling: **{sum_b['p90']:.1f}**\n\n"
        f"👉 **{ss_result.get('tactical_rationale')}**"
    )

    # Comprehensive Start/Sit Leverage Audit Table for Team A
    st.markdown("---")
    st.subheader(f"📋 Comprehensive Start/Sit Leverage Audit for {team_a_name}")
    st.caption("Every possible bench-to-starter substitution evaluated across Monte Carlo iterations to show exactly how each move impacts your team's win probability.")
    if team_a_bench:
        with st.spinner("Computing full start/sit leverage audit across all bench assets..."):
            opt_result = simulation.find_roster_optimizations(
                team_a_roster,
                team_a_bench,
                team_b_roster,
                n_simulations=min(3000, n_iterations)
            )
        if not opt_result["audit_table"].empty:
            st.dataframe(
                opt_result["audit_table"][["Bench Player", "Pos", "Starter Compared", "Slot", "Bench Proj", "Starter Proj", "Pt Delta", "Win% Leverage", "Status", "Recommendation"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No bench alternatives found to compare.")
    else:
        st.info("No active bench players available for comparison.")


# ===========================================================================
# TAB 3: TUESDAY WAIVER & FAAB CLAIM SHEET
# ===========================================================================
with tab_vorp:
    st.subheader(f"📋 Tuesday Waiver & FAAB Claim Sheet ({team_a_name})")
    st.markdown("Automated waiver priority queue. Recommendations are formatted as **Atomic Transaction Pairs** with dynamic FAAB bid sizing derived from marginal win% leverage, net projected points, and remaining team budget.")

    # Interactive Sizing & Strategy Controls
    w_col1, w_col2, w_col3 = st.columns([1.5, 1, 1])
    with w_col1:
        waiver_mode_label = st.radio(
            "Strategy Focus",
            ["⚡ 1-Week Spot Streamers", "💎 High-Upside Contingent Stashes"],
            horizontal=True,
            help="Spot Streamers optimize for immediate median points. Contingent Stashes rank by 90th-percentile injury contingency upside."
        )
        is_stash_selected = ("Stashes" in waiver_mode_label)
        mode_param = "stash" if is_stash_selected else "streamer"

    with w_col2:
        default_team_faab = int(team_a_dict.get("remaining_faab", 100)) if is_espn_active else 100
        remaining_faab = st.number_input(
            "💰 Remaining Team FAAB ($)",
            min_value=0,
            max_value=1000,
            value=default_team_faab,
            step=5,
            help="Total unspent FAAB budget used to dynamically size percentage-based waiver claims."
        )

    with w_col3:
        st.markdown(f"""
        <div class="metric-card" style="padding: 10px 14px;">
            <div class="metric-title">Active FAAB Pool</div>
            <div class="metric-value green-val" style="font-size: 1.6rem; margin: 2px 0;">${remaining_faab}</div>
            <div class="metric-sub">{"Default ($100)" if remaining_faab == 100 else f"${remaining_faab} budget"}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dynamic Replacement Baselines
    with st.spinner("Computing dynamic replacement baselines..."):
        baselines = vorp.calculate_waiver_baselines(waiver_pool, n_simulations=n_iterations)

    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
    for col, pos in zip([b_col1, b_col2, b_col3, b_col4, b_col5], ["QB", "RB", "WR", "TE", "FLEX"]):
        b_info = baselines.get(pos, {})
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{pos} BASELINE</div>
                <div class="metric-value blue-val">{b_info.get('baseline_mean', 0.0):.1f}</div>
                <div class="metric-sub">{b_info.get('player_name', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Evaluate Targeted Waiver Upgrades with Atomic Transaction Pairs
    all_team_a_players = (team_a_dict.get("roster", team_a_roster) if is_espn_active else team_a_roster) + team_a_bench
    team_a_qbs = [p for p in all_team_a_players if p.get("position", "").upper() == "QB"]
    qb_str = ", ".join([f"{p['name']} ({p.get('mu', 0):.1f} pts)" for p in team_a_qbs]) if team_a_qbs else "None"

    with st.spinner(f"Evaluating waiver wire pool under {waiver_mode_label}..."):
        df_upgrades = vorp.calculate_waiver_roster_upgrades(
            team_a_bench,
            waiver_pool,
            n_simulations=min(n_iterations, 3000),
            full_roster=all_team_a_players,
            max_roster_qbs=2,
            remaining_faab=int(remaining_faab),
            mode=mode_param
        )

    st.markdown("### ⚡ Atomic Claim Sheet (Priority Orders)")
    st.caption(f"Ordered queue of atomic Add ➔ Drop pairs with dynamically sized FAAB allocations ready for submission into your league manager. **🔒 2-QB Rule Enforced:** Quarterback claims are evaluated exclusively against rostered QBs ({qb_str}) to preserve the 2-QB limit.")

    if not df_upgrades.empty:
        # Copyable quick-claim text
        if "Atomic Transaction Pair" in df_upgrades.columns:
            claim_text = "\n".join(df_upgrades["Atomic Transaction Pair"].tolist())
            with st.expander("📋 Copyable Claim Queue (Plaintext)", expanded=False):
                st.code(claim_text, language="text")

        # Visual Atomic Claim Cards
        for idx, row in df_upgrades.iterrows():
            p_badge_bg = "#064e3b" if idx == 0 else ("#1e3a8a" if idx == 1 else "#1e293b")
            p_badge_border = "#10b981" if idx == 0 else ("#38bdf8" if idx == 1 else "#475569")
            gain_label = f"Net Ceiling +{row['Net P90 Gain']:.1f} pts" if is_stash_selected else f"Net Gain +{row['Net Proj Gain']:.1f} pts"
            gain_val = f"+{row['Net P90 Gain']:.1f}" if is_stash_selected else f"+{row['Net Proj Gain']:.1f}"

            st.markdown(f"""
            <div style="background: #1e293b; border: 1px solid {p_badge_border}; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="background: {p_badge_bg}; color: #f8fafc; border: 1px solid {p_badge_border}; padding: 3px 10px; border-radius: 6px; font-size: 0.82rem; font-weight: 800;">
                        Priority {row['Priority']}
                    </span>
                    <span style="margin-left: 12px; font-weight: 800; font-size: 1.05rem; color: #f8fafc;">
                        Add <span style="color: #10b981;">{row['Target Waiver Add']}</span> ({row['Pos']}) &nbsp;➔&nbsp; Drop <span style="color: #ef4444;">{row['Cut Candidate']}</span> ({row['Cut Pos']})
                    </span>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">
                        {row['Upgrade Rationale']}
                    </div>
                </div>
                <div style="display: flex; gap: 16px; align-items: center;">
                    <div style="text-align: right;">
                        <div style="font-size: 0.70rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">{gain_label.split()[0]} {gain_label.split()[1]}</div>
                        <div style="font-weight: 800; color: #10b981; font-size: 1.15rem;">
                            {gain_val} pts
                        </div>
                    </div>
                    <div style="background: #064e3b; border: 1px solid #10b981; padding: 8px 16px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 0.70rem; color: #a7f3d0; font-weight: 700; text-transform: uppercase;">Suggested FAAB</div>
                        <div style="font-size: 1.2rem; font-weight: 900; color: #ffffff;">{row['Suggested FAAB']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📋 View Detailed Upgrade Audit Table", expanded=False):
            disp_cols = [c for c in ["Priority", "Target Waiver Add", "Pos", "Cut Candidate", "Cut Pos", "Projected (Med)", "Floor (P10)", "Ceiling (P90)", "Net Proj Gain", "Net P90 Gain", "Suggested FAAB", "Upgrade Rationale"] if c in df_upgrades.columns]
            st.dataframe(df_upgrades[disp_cols], use_container_width=True, hide_index=True)
    else:
        st.info(f"Your bench on {team_a_name} is currently stronger than the available waiver wire pool at key positions.")

    st.markdown("---")

    # Active Roster VORP Table with Standardized Columns
    st.subheader(f"Active Roster VORP for {team_a_name}")
    with st.spinner("Evaluating active roster against waiver baselines..."):
        df_roster_vorp = vorp.calculate_roster_vorp(
            team_a_roster,
            baselines,
            opponent_roster=team_b_roster,
            n_simulations=n_iterations
        )
    disp_vorp_cols = [c for c in ["Player", "Pos", "Slot", "Projected (Med)", "Floor (P10)", "Ceiling (P90)", "Waiver Baseline", "Points VORP", "Win% VORP (Delta P(Win))"] if c in df_roster_vorp.columns]
    st.dataframe(df_roster_vorp[disp_vorp_cols], use_container_width=True, hide_index=True)

    # Waiver Wire Pool Rankings
    st.subheader(f"Top Waiver Wire Targets ({len(waiver_pool)} Available Free Agents)")
    df_waiver_rankings = vorp.rank_waiver_wire_pool(waiver_pool, baselines, n_simulations=n_iterations, roster_qbs=team_a_qbs, mode=mode_param)

    col_w1, col_w2 = st.columns([1.1, 0.9])
    with col_w1:
        disp_rank_cols = [c for c in ["Player", "Pos", "Team", "Projected (Med)", "Floor (P10)", "Ceiling (P90)", "Streamer OS (P≥12)", "Net Over Baseline", "2-QB Context"] if c in df_waiver_rankings.columns]
        st.dataframe(df_waiver_rankings[disp_rank_cols], use_container_width=True, hide_index=True)

    with col_w2:
        top_plot_waivers = df_waiver_rankings.head(15)
        plot_metric = "Ceiling (P90)" if is_stash_selected else "Projected (Med)"
        fig_waiver = px.bar(
            top_plot_waivers,
            x=plot_metric,
            y="Player",
            color="Pos",
            orientation="h",
            title=f"Top Pickups by {plot_metric}",
            color_discrete_map={"QB": "#38bdf8", "RB": "#10b981", "WR": "#c084fc", "TE": "#f59e0b"}
        )
        fig_waiver.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,41,59,0.5)",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_waiver, use_container_width=True)

    # Contingency & Handcuff Bench Value Table
    st.markdown("---")
    st.subheader("🛡️ Backup RB Contingency & Handcuff Leverage Index")
    st.caption("Quantifies the call option value of backup running backs: Contingency Index = P(Starter Injury) × Conditional RB1 Output. Ranked by 90th percentile contingency upside.")
    all_rb_pool = waiver_pool + (team_a_dict.get("roster", team_a_roster) if is_espn_active else team_a_roster)
    df_contingency = vorp.calculate_contingency_leverage(all_rb_pool, n_simulations=n_iterations)
    st.dataframe(df_contingency, use_container_width=True, hide_index=True)

    # Floor Surge & Contrarian Buy-Low Scanner
    st.markdown("---")
    st.subheader(f"📈 Floor Surge & Contrarian Buy-Low Scanner ({team_a_name})")
    st.caption(f"Scans multi-week SumerSports usage trajectories (Target Share velocity, YPRR efficiency, and aDOT) across the league to identify high-floor surging assets and contrarian buy-low trade candidates due for positive touchdown regression for **{team_a_name}**.")

    try:
        df_scanner = scheme_db.detect_floor_surges_and_buy_lows()
        if not df_scanner.empty:
            player_owner_map = {}
            if is_espn_active and st.session_state.get("espn_league"):
                for t in st.session_state.espn_league.get("teams", []):
                    t_title = t.get("team_name", "")
                    tag = f"👑 {t_title} (You)" if t_title == team_a_name else f"🏈 {t_title}"
                    for p in t.get("roster", []):
                        player_owner_map[p.get("name", "")] = tag

            def resolve_roster_status(name):
                if name in player_owner_map:
                    return player_owner_map[name]
                return "⚡ Available Free Agent" if is_espn_active else "Available / League"

            df_scanner["Roster Status"] = df_scanner["Player"].apply(resolve_roster_status)

            cols = ["Roster Status", "Player", "Team", "Pos", "Trade Signal", "Action", "Latest Tgt%", "Tgt% Delta", "Latest YPRR", "YPRR Delta", "aDOT", "Trade & Strategy Rationale"]
            display_cols = [c for c in cols if c in df_scanner.columns]

            st.dataframe(
                df_scanner[display_cols],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No player usage history found. Seed scheme history in Tab 5 or upload a SumerSports review.")
    except Exception as exc:
        st.warning(f"Could not load usage scanner: {exc}")



# ===========================================================================
# TAB 4: VEGAS ODDS & POLYMARKET FEED
# ===========================================================================
with tab_feeds:
    st.subheader("Vegas Betting Markets & Polymarket Crowd Probabilities")

    # Section 1: The Odds API Game Lines
    st.markdown("#### 🏈 NFL Spreads, Totals & Implied Team Totals")
    with st.spinner("Fetching NFL game lines from The Odds API..."):
        raw_odds = data_ingest.fetch_nfl_odds(api_key=odds_api_key)
        if raw_odds:
            parsed_games = data_ingest.parse_nfl_games(raw_odds)
            df_games = pd.DataFrame(parsed_games)
            st.dataframe(
                df_games[["away_team", "home_team", "spread", "total", "away_implied", "home_implied", "commence_time"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Live Vegas Odds not loaded. Check API Key or enjoy curated game data.")

    st.markdown("---")

    # Section 2: Polymarket Gamma API Inspector
    st.markdown("#### 🔮 Polymarket Gamma API Crowd Risk Inspector")
    st.caption("Query live crowd probability on player status, injury availability, and game props from Polymarket's Gamma API.")

    poly_col1, poly_col2 = st.columns([2, 1])
    with poly_col1:
        custom_slug = st.text_input(
            "Polymarket Event Slug",
            value="nfl-will-the-bills-beat-the-lions",
            help="Example: nfl-will-the-bills-beat-the-lions, or any active slug from gamma-api.polymarket.com/events"
        )
    with poly_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        query_poly = st.button("Query Polymarket Gamma API", use_container_width=True)

    if query_poly or custom_slug:
        with st.spinner("Connecting to Polymarket Gamma API..."):
            poly_result = data_ingest.fetch_polymarket_event(custom_slug)
            if poly_result:
                st.success(f"Connected: {poly_result.get('title')}")
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.metric("Crowd Implied Probability", f"{poly_result['active_probability'] * 100:.1f}%")
                with p_col2:
                    st.metric("Market Volume ($)", f"${poly_result.get('volume', 0):,.2f}")
            else:
                st.info(f"No active market found for slug '{custom_slug}'. Showing simulated crowd risk model (95% active probability).")

    # Section 3: Consensus Player Props
    st.markdown("---")
    st.markdown("#### 🎯 Ingested Vegas Consensus Player Props & Idiosyncratic Projections")
    st.caption("Live consensus player prop lines from The Odds API with expected fantasy points (μ) computed directly using your league's idiosyncratic scoring rules (6pt Pass TD, 0.1 comp, stacking milestone/long TD bonuses).")

    all_props_cache = data_ingest.build_all_player_props_cache(api_key=odds_api_key)
    prop_table_rows = []
    for p_name, p_props in list(all_props_cache.items())[:30]:
        if "dst" in p_name:
            continue
        p_title = p_name.title()
        if "pass_yds" in p_props:
            pos = "QB"
        elif "rush_yds" in p_props and p_props.get("rush_yds", 0) > 40:
            pos = "RB"
        else:
            pos = "WR"

        mu, sigma, p_act, breakdown = scoring_config.calculate_player_mu_and_sigma(
            position=pos,
            props=p_props,
            scoring_config=active_scoring_cfg
        )

        prop_table_rows.append({
            "Player": p_title,
            "Pos": pos,
            "Pass Yds O/U": f"{p_props.get('pass_yds', '-'):.1f}" if "pass_yds" in p_props else "-",
            "Completions": f"{p_props.get('pass_completions', '-'):.1f}" if "pass_completions" in p_props else "-",
            "Rush Yds O/U": f"{p_props.get('rush_yds', '-'):.1f}" if "rush_yds" in p_props else "-",
            "Receptions O/U": f"{p_props.get('receptions', '-'):.1f}" if "receptions" in p_props else "-",
            "Anytime TD%": f"{p_props.get('anytime_td_prob', 0)*100:.1f}%" if "anytime_td_prob" in p_props else "-",
            "Idiosyncratic Proj (μ)": f"{mu:.2f} pts",
            "Std Dev (σ)": f"±{sigma:.2f}"
        })

    if prop_table_rows:
        st.dataframe(pd.DataFrame(prop_table_rows), use_container_width=True, hide_index=True)

    # Section 4: 32-Team Defensive Scheme Profiles
    st.markdown("---")
    st.markdown("#### 🛡️ NFL 32-Team Defensive Scheme Profiles")
    st.caption("Quick Pressure Rate % (<2.5s), Middle of Field (MOF) EPA Allowed, and Rush Success Rate Allowed driving parameter adjustments.")
    scheme_rows = [
        {
            "Team": team_code,
            "Quick Pressure %": f"{metrics['quick_pressure_rate']*100:.1f}%",
            "MOF EPA Allowed": f"{metrics['mof_epa_allowed']:.2f}",
            "Rush SR Allowed": f"{metrics['rush_sr_allowed']*100:.1f}%",
            "Matchup Impact": (
                "High Pressure (QB checkdown shift)" if metrics['quick_pressure_rate'] > 0.20
                else ("High MOF EPA (TE/Slot boost 1.15x)" if metrics['mof_epa_allowed'] > 0.50
                else ("High Rush SR (RB floor tightened)" if metrics['rush_sr_allowed'] > 0.45 else "Neutral"))
            )
        }
        for team_code, metrics in data_ingest.NFL_DEFENSIVE_SCHEMES.items()
    ]
    st.dataframe(pd.DataFrame(scheme_rows), use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 5: SCHEME LAB & NEWSLETTER UPLOAD
# ===========================================================================
with tab_scheme:
    st.markdown("### 🔬 SumerSports Scheme Lab & Newsletter Ingestion")
    st.markdown(
        "Upload SumerSports *Stats & Scheme* weekly PDF reviews to extract team defensive schemes "
        "(Quick Pressure %, MOF EPA allowed, Rush Success Rate allowed) and player usage metrics "
        "(Target Share %, YPRR, aDOT). Ingested metrics update the 3-week rolling moving averages "
        "and dynamically modulate Monte Carlo simulation parameters."
    )

    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_pdf = st.file_uploader(
            "Upload SumerSports PDF Review",
            type=["pdf"],
            help="Select a weekly Stats & Scheme PDF review. The parser will extract team defensive schemes and player usage metrics."
        )
    with col_up2:
        tag_week = st.number_input("Target NFL Week", min_value=1, max_value=22, value=3, step=1)
        load_sample = st.button("📄 Load Sample Week 3 PDF", help="Load the bundled sample SumerSports Week 3 review for demonstration.")

    pdf_source = None
    if uploaded_pdf is not None:
        pdf_source = uploaded_pdf
    elif load_sample or st.session_state.get("sample_pdf_active", False):
        if load_sample:
            st.session_state["sample_pdf_active"] = True
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_sumersports_week3.pdf")
        if not os.path.exists(sample_path):
            pdf_parser.create_sample_sumersports_pdf(sample_path, week=int(tag_week))
        pdf_source = sample_path

    if pdf_source is not None:
        with st.spinner("Extracting scheme & usage tables via pdfplumber..."):
            parsed = pdf_parser.parse_sumersports_newsletter(pdf_source, default_week=int(tag_week))

        if parsed.get("success"):
            detected_week = parsed.get("week", tag_week)
            team_df = parsed.get("team_metrics", pd.DataFrame())
            player_df = parsed.get("player_metrics", pd.DataFrame())

            st.success(f"Successfully parsed PDF for Week {detected_week}: {len(team_df)} Team Schemes, {len(player_df)} Player Usages extracted.")

            col_btn1, col_btn2 = st.columns([2, 2])
            with col_btn1:
                if st.button("💾 Save / Overwrite Week in Scheme Database", type="primary", key="save_scheme_btn"):
                    save_res = scheme_db.save_newsletter_metrics(team_df, player_df, week=detected_week)
                    st.success(f"Successfully updated Week {detected_week} in persistent Parquet database ({save_res['team_rows']} teams, {save_res['player_rows']} players)!")
                    st.cache_data.clear()

            with col_btn2:
                if st.button("🔄 Reset Baseline Scheme History", help="Re-seed initial 3-week baseline for all 32 teams"):
                    scheme_db.init_default_scheme_history(force=True)
                    st.info("Re-seeded initial 3-week baseline for all 32 teams.")
                    st.cache_data.clear()

            preview_tab_team, preview_tab_player = st.tabs(["🛡️ Parsed Team Scheme Metrics", "🏃 Parsed Player Usage Metrics"])
            with preview_tab_team:
                if not team_df.empty:
                    st.dataframe(team_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No team scheme rows detected in this document.")

            with preview_tab_player:
                if not player_df.empty:
                    st.dataframe(player_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No player usage rows detected in this document.")
        else:
            st.error(f"Error parsing PDF: {parsed.get('error', 'Unknown parsing error')}")

    # Section: Scheme History & Trends
    st.markdown("---")
    st.markdown("### 📊 Scheme Trends & Usage Evolution")

    hist_tab_defense, hist_tab_player = st.tabs(["🛡️ Team Defense Trajectory", "🏃 Player Usage Evolution"])

    with hist_tab_defense:
        all_teams = scheme_db.get_all_teams_in_db()
        def_idx = all_teams.index("CLE") if "CLE" in all_teams else 0
        sel_team = st.selectbox("Select NFL Defense", all_teams, index=def_idx)

        rolling_stats = scheme_db.get_team_defense_rolling(sel_team, window=3)
        trend_stats = scheme_db.get_scheme_trend(sel_team)

        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        with t_col1:
            st.metric("3-Wk Quick Press %", f"{rolling_stats['quick_pressure_rate']*100:.1f}%", delta=f"{trend_stats['delta_quick_pressure']*100:+.1f}% vs Avg")
        with t_col2:
            st.metric("3-Wk MOF EPA Allowed", f"{rolling_stats['mof_epa_allowed']:.2f}", delta=f"{trend_stats['delta_mof_epa']:+.2f} vs Avg", delta_color="inverse")
        with t_col3:
            st.metric("3-Wk Rush SR Allowed", f"{rolling_stats['rush_sr_allowed']*100:.1f}%", delta=f"{trend_stats['delta_rush_sr']*100:+.1f}% vs Avg", delta_color="inverse")
        with t_col4:
            st.metric("Scheme Trend", trend_stats['trend_summary'])

        # Plotly chart: Team Defense over Weeks
        scheme_df = scheme_db.load_scheme_history()
        team_scheme_hist = scheme_df[scheme_df["team"] == sel_team].sort_values(by="week")
        if not team_scheme_hist.empty:
            fig_def = go.Figure()
            fig_def.add_trace(go.Scatter(
                x=team_scheme_hist["week"],
                y=team_scheme_hist["quick_pressure_rate"] * 100,
                mode="lines+markers",
                name="Quick Pressure % (<2.5s)",
                line=dict(color="#FF4B4B", width=3)
            ))
            fig_def.add_trace(go.Scatter(
                x=team_scheme_hist["week"],
                y=team_scheme_hist["rush_sr_allowed"] * 100,
                mode="lines+markers",
                name="Rush Success Rate Allowed %",
                line=dict(color="#00C853", width=2, dash="dot")
            ))
            fig_def.add_trace(go.Scatter(
                x=team_scheme_hist["week"],
                y=team_scheme_hist["mof_epa_allowed"],
                mode="lines+markers",
                name="MOF EPA Allowed (Right Axis)",
                yaxis="y2",
                line=dict(color="#FFA726", width=2, dash="dash")
            ))
            fig_def.update_layout(
                title=f"{sel_team} Defensive Scheme Trajectory Across Weeks",
                xaxis=dict(title="NFL Week Number", dtick=1),
                yaxis=dict(title="Rate Percentage (%)"),
                yaxis2=dict(title="EPA / Play", overlaying="y", side="right"),
                template="plotly_dark",
                height=380,
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_def, use_container_width=True)
        else:
            st.info(f"No historical records found for {sel_team}.")

    with hist_tab_player:
        all_players = scheme_db.get_all_players_in_db()
        if not all_players:
            all_players = ["CeeDee Lamb", "Demario Douglas", "Jalen Coker", "Justin Jefferson"]

        p_idx = all_players.index("Demario Douglas") if "Demario Douglas" in all_players else 0
        sel_player = st.selectbox(
            "Select Player for Usage Trajectory",
            all_players,
            index=p_idx
        )

        p_hist = scheme_db.get_player_usage_history(sel_player)
        if not p_hist.empty:
            pk1, pk2, pk3 = st.columns(3)
            with pk1:
                latest_tgt = p_hist.iloc[-1]["target_share"] * 100
                st.metric("Latest Target Share", f"{latest_tgt:.1f}%")
            with pk2:
                latest_yprr = p_hist.iloc[-1]["yprr"]
                st.metric("Latest YPRR", f"{latest_yprr:.2f}")
            with pk3:
                latest_adot = p_hist.iloc[-1]["adot"]
                st.metric("Latest aDOT", f"{latest_adot:.1f} yds")

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=p_hist["week"],
                y=p_hist["target_share"] * 100,
                mode="lines+markers",
                name="Target Share %",
                line=dict(color="#29B6F6", width=3)
            ))
            fig_p.add_trace(go.Scatter(
                x=p_hist["week"],
                y=p_hist["yprr"],
                mode="lines+markers",
                name="Yards Per Route Run (YPRR)",
                yaxis="y2",
                line=dict(color="#66BB6A", width=2, dash="dash")
            ))
            fig_p.update_layout(
                title=f"{sel_player} Usage & Efficiency Evolution",
                xaxis=dict(title="NFL Week Number", dtick=1),
                yaxis=dict(title="Target Share (%)"),
                yaxis2=dict(title="YPRR", overlaying="y", side="right"),
                template="plotly_dark",
                height=380,
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_p, use_container_width=True)

            st.dataframe(p_hist, use_container_width=True, hide_index=True)
        else:
            st.info(f"No historical records found for {sel_player}.")

    # Weekly Scheme Matchup Impact on Team A
    st.markdown("---")
    st.markdown(f"### 🛡️ Weekly Scheme Matchup Impact on {team_a_name}")
    st.caption(f"How opposing NFL defensive schemes (Quick Pressure %, MOF EPA allowed, Rush SR allowed) specifically modulate {team_a_name}'s players.")

    scheme_impact_rows = []
    for p in team_a_roster + team_a_bench:
        p_team = p.get("team", "NFL")
        opp_def = "CLE" if p_team in ["CIN", "BAL", "PIT"] else ("CAR" if p_team in ["ATL", "TB", "NO"] else ("NYG" if p_team in ["DAL", "PHI", "WAS"] else ("DET" if p_team in ["GB", "CHI", "MIN"] else "KC")))
        modified_p = data_ingest.apply_defensive_scheme_modifiers(p, opponent_team=opp_def)
        sch = data_ingest.get_defensive_scheme(opp_def)

        scheme_impact_rows.append({
            "Player": p.get("name"),
            "Pos": p.get("position"),
            "Slot": p.get("slot", "BE"),
            "NFL Team": p_team,
            "Opp Defense": opp_def,
            "Opp Quick Press%": f"{sch['quick_pressure_rate']*100:.1f}%",
            "Opp MOF EPA": f"{sch['mof_epa_allowed']:.2f}",
            "Opp Rush SR%": f"{sch['rush_sr_allowed']*100:.1f}%",
            "Base Proj": p.get("mu", 10.0),
            "Adjusted Proj": modified_p.get("mu", 10.0),
            "Scheme Impact Notes": modified_p.get("scheme_notes", "Neutral baseline matchup")
        })

    st.dataframe(pd.DataFrame(scheme_impact_rows), use_container_width=True, hide_index=True)
