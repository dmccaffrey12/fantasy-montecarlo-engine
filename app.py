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
import trade_analyzer

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

st.sidebar.markdown("### 📊 Primary Projection Anchor")
uploaded_athletic_xlsx = st.sidebar.file_uploader(
    "Upload Weekly Athletic Projections (.xlsx)",
    type=["xlsx"],
    help="Drop in The Athletic's Wednesday release for waivers or Sunday morning update for start/sit triage."
)
if uploaded_athletic_xlsx is not None:
    file_sig = f"{uploaded_athletic_xlsx.name}_{uploaded_athletic_xlsx.size}"
    if st.session_state.get("athletic_file_sig") != file_sig:
        data_ingest.set_uploaded_athletic_projections(uploaded_athletic_xlsx, uploaded_athletic_xlsx.name)
        st.session_state["athletic_file_sig"] = file_sig
        st.session_state.espn_league = None
        st.session_state.espn_last_attempt = None
        st.rerun()
else:
    if st.session_state.get("athletic_file_sig") is not None:
        data_ingest.set_uploaded_athletic_projections(None)
        st.session_state["athletic_file_sig"] = None
        st.session_state.espn_league = None
        st.session_state.espn_last_attempt = None
        st.rerun()

ath_summary = data_ingest.get_athletic_projection_summary()
if ath_summary.get("total_entities", 0) > 0:
    c_dict = ath_summary["counts"]
    st.sidebar.caption(
        f"✅ **Anchor:** `{ath_summary['filename']}`  \n"
        f"**{ath_summary['skill_total']} Skill** ({c_dict['QB']} QB · {c_dict['RB']} RB · {c_dict['WR']} WR · {c_dict['TE']} TE) + **{ath_summary['dst_total']} D/ST**  \n"
        f"⚖️ *50% Vegas Implied / 50% Athletic Custom Mu*"
    )

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

attempt_key = (target_league_id, target_year, scoring_format, st.session_state.get("athletic_file_sig"))

if target_league_id and (
    st.session_state.espn_league is None or
    current_synced_id != target_league_id or
    current_synced_year != target_year
) and st.session_state.espn_last_attempt != attempt_key:
    st.session_state.espn_last_attempt = attempt_key
    with st.spinner(f"Connecting to ESPN Fantasy League ({target_league_id}, Year {target_year})..."):
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

APP_BUILD_VERSION = "2026.10.13-v3.0"
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


# ---------------------------------------------------------------------------
# MAIN INTERFACE TABS
# ---------------------------------------------------------------------------
tab_matchup, tab_startsit, tab_vorp, tab_trade, tab_feeds, tab_scheme = st.tabs([
    "⚡ Sunday Morning Command Center",
    "⚖️ Start / Sit Comparative Engine",
    "📋 Tuesday Waiver & FAAB Claim Sheet",
    "🤝 Idiosyncratic Trade Analyzer",
    "🔮 Next-Horizon Strategy & Market Feed",
    "🔬 Scheme Lab & Newsletter Upload"
])


# ===========================================================================
# TAB 1: SUNDAY MORNING COMMAND CENTER
# ===========================================================================
with tab_matchup:
    # Run 10,000-run correlated Cholesky Gaussian Copula simulation
    with st.spinner(f"Running {n_iterations:,} Correlated Monte Carlo simulations for {team_a_name} vs {team_b_name}..."):
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
            <div class="metric-title">{team_a_name} Projected (Med)</div>
            <div class="metric-value blue-val">{a_summary['median']:.1f}</div>
            <div class="metric-sub">Floor (P10): {a_summary['p10']:.1f} · Ceiling (P90): {a_summary['p90']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{team_b_name} Projected (Med)</div>
            <div class="metric-value amber-val">{b_summary['median']:.1f}</div>
            <div class="metric-sub">Floor (P10): {b_summary['p10']:.1f} · Ceiling (P90): {b_summary['p90']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Expected Margin of Victory</div>
            <div class="metric-value purple-val">{m_summary['mean']:+0.1f}</div>
            <div class="metric-sub">Median Margin: {m_summary['median']:+0.1f} · Volatility: ±{m_summary['std']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    # 1. Centered Win% Gauge + High-Contrast Margin of Victory Distribution
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
            height=265,
            margin=dict(l=20, r=20, t=35, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_margin:
        st.subheader("Margin of Victory Distribution (Correlated Stack Matrix)")
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
        fig_margin.add_vline(x=m_summary["median"], line_width=2, line_color="#c084fc", line_dash="dash", annotation_text=f"Med ({m_summary['median']:+0.1f})")

        fig_margin.update_layout(
            barmode="overlay",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,41,59,0.5)",
            xaxis_title=f"Margin of Victory ({team_a_name} vs {team_b_name})",
            yaxis_title="Simulated Frequency",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=265,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_margin, use_container_width=True)

    # 2. Contested Lineup Slots & Pre-Paired Decision Cards
    st.markdown("---")
    st.markdown("### ⚔️ Contested Lineup Slots & Sunday Decision Cards")
    st.caption("Automatically surfaced borderline starter decisions: any bench asset projecting within 2.5 points of a starter (or higher), pre-paired with net Win% leverage, exact positional percentiles, Vegas Implied Team Totals, and QB Stack tags.")

    contested_slots = simulation.find_contested_lineup_slots(team_a_roster, team_a_bench, p_win_a, threshold=2.5)

    if contested_slots:
        card_cols = st.columns(min(len(contested_slots), 2))
        for idx, item in enumerate(contested_slots):
            c_col = card_cols[idx % 2]
            with c_col:
                delta_sign = "+" if item["delta_mu"] >= 0 else ""
                delta_floor_sign = "+" if item["delta_floor"] >= 0 else ""
                delta_ceiling_sign = "+" if item["delta_ceiling"] >= 0 else ""
                win_lev_sign = "+" if item["win_leverage"] >= 0 else ""

                vegas_strip = (
                    f"<b>{item['challenger_name']}</b> ({item.get('challenger_vegas_tag', 'NFL')})"
                    f" &nbsp;vs&nbsp; <b>{item['starter_name']}</b> ({item.get('starter_vegas_tag', 'NFL')})"
                )
                stack_tags = " &nbsp; ".join(
                    [t for t in [item.get("challenger_stack", ""), item.get("starter_stack", "")] if t]
                )

                card_html = f"""
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 14px 18px; margin-bottom: 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                        <span style="font-weight: 800; font-size: 1.02rem; color: #f8fafc;">
                            Slot: <span style="color: #38bdf8;">{item['slot']}</span> &nbsp;·&nbsp; {item['starter_name']} vs {item['challenger_name']}
                        </span>
                        <span style="background: {item['badge_bg']}; color: {item['badge_color']}; border: 1px solid {item['badge_color']}; padding: 3px 11px; border-radius: 20px; font-size: 0.78rem; font-weight: 800;">
                            {item['badge']}
                        </span>
                    </div>
                    <div style="font-size: 0.76rem; color: #94a3b8; margin-bottom: 10px; display: flex; justify-content: space-between; flex-wrap: wrap;">
                        <span>📡 {vegas_strip}</span>
                        <span style="color: #38bdf8; font-weight: 700;">{stack_tags}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 10px;">
                        <div style="background: #0f172a; padding: 8px 10px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                            <div style="font-size: 0.68rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Projected (Med)</div>
                            <div style="font-size: 1.10rem; font-weight: 800; color: #f8fafc; margin-top: 2px;">{item['challenger_mu']:.1f} vs {item['starter_mu']:.1f}</div>
                            <div style="font-size: 0.73rem; color: {'#10b981' if item['delta_mu'] >= 0 else '#ef4444'}; font-weight: 700;">{delta_sign}{item['delta_mu']:.1f} pts</div>
                        </div>
                        <div style="background: #0f172a; padding: 8px 10px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                            <div style="font-size: 0.68rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Floor (P10)</div>
                            <div style="font-size: 1.10rem; font-weight: 800; color: #f8fafc; margin-top: 2px;">{item['challenger_p10']:.1f} vs {item['starter_p10']:.1f}</div>
                            <div style="font-size: 0.73rem; color: {'#10b981' if item['delta_floor'] >= 0 else '#ef4444'}; font-weight: 700;">{delta_floor_sign}{item['delta_floor']:.1f} floor</div>
                        </div>
                        <div style="background: #0f172a; padding: 8px 10px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                            <div style="font-size: 0.68rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Ceiling (P90)</div>
                            <div style="font-size: 1.10rem; font-weight: 800; color: #f8fafc; margin-top: 2px;">{item['challenger_p90']:.1f} vs {item['starter_p90']:.1f}</div>
                            <div style="font-size: 0.73rem; color: {'#10b981' if item['delta_ceiling'] >= 0 else '#ef4444'}; font-weight: 700;">{delta_ceiling_sign}{item['delta_ceiling']:.1f} ceiling</div>
                        </div>
                        <div style="background: #0f172a; padding: 8px 10px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                            <div style="font-size: 0.68rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Win% Leverage</div>
                            <div style="font-size: 1.10rem; font-weight: 800; color: {'#10b981' if item['win_leverage'] >= 0 else '#f59e0b'}; margin-top: 2px;">{win_lev_sign}{item['win_leverage']*100:.1f}%</div>
                            <div style="font-size: 0.73rem; color: #94a3b8;">Team Net Impact</div>
                        </div>
                    </div>
                    <div style="font-size: 0.80rem; color: #cbd5e1; background: rgba(15, 23, 42, 0.75); padding: 8px 12px; border-radius: 6px; border-left: 3px solid {item['badge_color']};">
                        💡 <b>Sunday Rationale:</b> {item['rationale']}
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.success(f"✅ **No Contested Slots:** All starters for **{team_a_name}** hold at least a +2.5 projected point cushion over your bench depth.")

    # 3. Starting Rosters & Correlated Stack Matrix Breakdown
    st.markdown("---")
    with st.expander("📋 Complete Starting Rosters & Correlated Stack Matrix (Projected Med / Floor P10 / Ceiling P90)", expanded=True):
        detected_corrs = matchup_result.get("stacks", [])
        if detected_corrs:
            corr_badges = " &nbsp;·&nbsp; ".join([
                f"<b>{c['player_1']} ↔ {c['player_2']}</b> ({c['stack_type']}, ρ={c['rho']:+.2f})"
                for c in detected_corrs[:6]
            ])
            st.markdown(
                f"<div style='background:#0f172a;border:1px solid #334155;padding:8px 14px;border-radius:8px;margin-bottom:12px;font-size:0.82rem;color:#cbd5e1;'>"
                f"🔗 <b>Active Gaussian Copula Covariance Pairs:</b> {corr_badges}</div>",
                unsafe_allow_html=True
            )

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
    st.caption("Select any auto-detected Contested Lineup Slot below for instant 1-click comparison, or choose a custom player pair.")

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

    # Quick-Select Contested Lineup Slot Bar (Zero Manual Configuration)
    quick_pair_options = ["🔧 Custom Player Comparison"]
    quick_pair_map: Dict[str, Tuple[str, str]] = {}
    for item in contested_slots:
        label = f"⚡ {item['slot']}: {item['starter_name']} vs {item['challenger_name']} ({item['delta_mu']:+.1f} pts)"
        quick_pair_options.append(label)
        quick_pair_map[label] = (item["starter_name"], item["challenger_name"])

    selected_quick_pair = st.radio(
        "Quick-Triage Contested Decisions",
        quick_pair_options,
        index=1 if len(quick_pair_options) > 1 else 0,
        horizontal=True
    )

    col_ss1, col_ss2 = st.columns(2)
    map_a = {p["name"]: format_player_display_label(p) for p in all_team_a_players}
    cand_a_options = [p["name"] for p in all_team_a_players]

    if selected_quick_pair in quick_pair_map and quick_pair_map[selected_quick_pair][0] in cand_a_options:
        cand_a_idx = cand_a_options.index(quick_pair_map[selected_quick_pair][0])
    else:
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

    if selected_quick_pair in quick_pair_map and quick_pair_map[selected_quick_pair][1] in challenger_names:
        cand_b_idx = challenger_names.index(quick_pair_map[selected_quick_pair][1])
    else:
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

    vol_a = cand_a.get("volatility_class") or cand_a.get("score_breakdown", {}).get("volatility_class", "Standard Volume")
    vol_b = cand_b.get("volatility_class") or cand_b.get("score_breakdown", {}).get("volatility_class", "Standard Volume")

    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid #3b82f6; border-radius: 10px; padding: 12px 16px; margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 800; color: #60a5fa; font-size: 0.95rem;">🎯 SCHEME MATCHUP & DEFENSIVE FUNNEL BREAKDOWN</span>
            <span style="font-size: 0.75rem; background: #1e3a8a; color: #93c5fd; padding: 2px 8px; border-radius: 4px; font-weight: 600;">SumerSports Synergy</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 8px;">
            <div style="background: #1e293b; padding: 8px 12px; border-radius: 6px; border-left: 3px solid #38bdf8;">
                <div style="font-weight: 700; color: #f8fafc; font-size: 0.92rem;">{cand_a_name} <span style="font-size: 0.8rem; color: #94a3b8; font-weight: normal;">({cand_a.get('position', 'WR')})</span></div>
                <div style="font-size: 0.80rem; color: #cbd5e1; margin-top: 2px;">Alignment: <b style="color: #38bdf8;">{align_a.get('alignment_tag', 'WR')}</b> (Slot: {align_a.get('slot_snap_pct', 0)*100:.0f}% | Wide: {align_a.get('wide_snap_pct', 0)*100:.0f}%) · Archetype: <b style="color: #38bdf8;">{vol_a}</b></div>
                <div style="font-size: 0.80rem; color: #e2e8f0; margin-top: 3px;">Opp: <b>{funnel_a.get('team', 'DEF')}</b> <span style="font-size: 0.74rem; background: {badge_color_a}22; color: {badge_color_a}; border: 1px solid {badge_color_a}; padding: 1px 6px; border-radius: 3px;">{funnel_a.get('funnel_classification', 'Balanced')}</span> · MOF EPA: <b>{funnel_a.get('mof_epa_allowed', 0):.2f}</b> · Quick Pres: <b>{funnel_a.get('quick_pressure_rate', 0)*100:.1f}%</b></div>
            </div>
            <div style="background: #1e293b; padding: 8px 12px; border-radius: 6px; border-left: 3px solid #c084fc;">
                <div style="font-weight: 700; color: #f8fafc; font-size: 0.92rem;">{cand_b_name} <span style="font-size: 0.8rem; color: #94a3b8; font-weight: normal;">({cand_b.get('position', 'WR')})</span></div>
                <div style="font-size: 0.80rem; color: #cbd5e1; margin-top: 2px;">Alignment: <b style="color: #c084fc;">{align_b.get('alignment_tag', 'WR')}</b> (Slot: {align_b.get('slot_snap_pct', 0)*100:.0f}% | Wide: {align_b.get('wide_snap_pct', 0)*100:.0f}%) · Archetype: <b style="color: #c084fc;">{vol_b}</b></div>
                <div style="font-size: 0.80rem; color: #e2e8f0; margin-top: 3px;">Opp: <b>{funnel_b.get('team', 'DEF')}</b> <span style="font-size: 0.74rem; background: {badge_color_b}22; color: {badge_color_b}; border: 1px solid {badge_color_b}; padding: 1px 6px; border-radius: 3px;">{funnel_b.get('funnel_classification', 'Balanced')}</span> · MOF EPA: <b>{funnel_b.get('mof_epa_allowed', 0):.2f}</b> · Quick Pres: <b>{funnel_b.get('quick_pressure_rate', 0)*100:.1f}%</b></div>
            </div>
        </div>
        <div style="font-size: 0.82rem; color: #f1f5f9; background: #0f172a; padding: 7px 11px; border-radius: 6px; border: 1px solid #334155;">
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
    <div style="background: {tac_bg}; border: 1px solid #334155; border-radius: 10px; padding: 10px 16px; margin-bottom: 14px;">
        <span style="font-weight: 800; font-size: 1.0rem;">{tac_icon} {tac_mode} REGIME: {tac_obj}</span><br>
        <span style="color: #cbd5e1; font-size: 0.84rem;">{ss_result.get('tactical_rationale', '')}</span>
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

    # Consolidated Outcome Probability Density & Percentile Horizon Plot
    sim_a = ss_result["sim_a"]
    sim_b = ss_result["sim_b"]
    sum_a = ss_result["summary_a"]
    sum_b = ss_result["summary_b"]

    x_span = np.linspace(0, max(sim_a.max(), sim_b.max()) + 5, 220)
    kde_cand_a = gaussian_kde(sim_a)(x_span) if np.std(sim_a) > 1e-4 else np.zeros_like(x_span)
    kde_cand_b = gaussian_kde(sim_b)(x_span) if np.std(sim_b) > 1e-4 else np.zeros_like(x_span)

    fig_cand = go.Figure()
    fig_cand.add_trace(go.Scatter(
        x=x_span, y=kde_cand_a,
        mode="lines",
        name=f"{cand_a_name} (Floor P10: {sum_a['p10']:.1f} · Med: {sum_a['median']:.1f} · Ceiling P90: {sum_a['p90']:.1f})",
        line=dict(color="#38bdf8", width=3),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.2)"
    ))
    fig_cand.add_trace(go.Scatter(
        x=x_span, y=kde_cand_b,
        mode="lines",
        name=f"{cand_b_name} (Floor P10: {sum_b['p10']:.1f} · Med: {sum_b['median']:.1f} · Ceiling P90: {sum_b['p90']:.1f})",
        line=dict(color="#c084fc", width=3),
        fill="tozeroy",
        fillcolor="rgba(192, 132, 252, 0.2)"
    ))
    fig_cand.add_vline(x=sum_a["median"], line_width=2, line_dash="dash", line_color="#38bdf8", annotation_text=f"{cand_a_name} Med ({sum_a['median']:.1f})")
    fig_cand.add_vline(x=sum_b["median"], line_width=2, line_dash="dash", line_color="#c084fc", annotation_text=f"{cand_b_name} Med ({sum_b['median']:.1f})", annotation_position="bottom right")
    fig_cand.add_vline(x=sum_a["p90"], line_width=1, line_dash="dot", line_color="#38bdf8", annotation_text=f"A P90 ({sum_a['p90']:.1f})")
    fig_cand.add_vline(x=sum_b["p90"], line_width=1, line_dash="dot", line_color="#c084fc", annotation_text=f"B P90 ({sum_b['p90']:.1f})")

    fig_cand.update_layout(
        title="Consolidated Outcome Density & Floor / Median / Ceiling Horizon",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,41,59,0.5)",
        xaxis_title="Weekly Fantasy Points",
        yaxis_title="Probability Density",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_cand, use_container_width=True)

    # Comprehensive Start/Sit Leverage Audit Table for Team A
    st.markdown("---")
    st.subheader(f"📋 Comprehensive Start/Sit Leverage Audit for {team_a_name}")
    st.caption("Vectorized evaluation of every possible bench-to-starter substitution across all 10,000 correlated Monte Carlo iterations.")
    if team_a_bench:
        opt_result = simulation.find_roster_optimizations(
            team_a_roster,
            team_a_bench,
            team_b_roster,
            n_simulations=n_iterations
        )
        if not opt_result["audit_table"].empty:
            audit_cols = [
                c for c in [
                    "Bench Player", "Pos", "Starter Compared", "Slot",
                    "Bench Med", "Bench Floor (P10)", "Bench Ceiling (P90)",
                    "Starter Med", "Starter Floor (P10)", "Starter Ceiling (P90)",
                    "Pt Delta", "Win% Leverage", "Status", "Recommendation"
                ] if c in opt_result["audit_table"].columns
            ]
            st.dataframe(
                opt_result["audit_table"][audit_cols],
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
            win_lev_str = row.get("Win% Leverage", f"{float(row['Net Proj Gain'])*1.6:+.1f}%")

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
                        <div style="font-size: 0.70rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Win% Leverage</div>
                        <div style="font-weight: 800; color: #38bdf8; font-size: 1.15rem;">
                            {win_lev_str}
                        </div>
                    </div>
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
            disp_cols = [c for c in ["Priority", "Target Waiver Add", "Pos", "Cut Candidate", "Cut Pos", "Projected (Med)", "Floor (P10)", "Ceiling (P90)", "Win% Leverage", "Net Proj Gain", "Net P90 Gain", "Suggested FAAB", "Upgrade Rationale"] if c in df_upgrades.columns]
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
# TAB 4: NEXT-HORIZON STRATEGY & MARKET FEED
# ===========================================================================
with tab_feeds:
    st.subheader(f"🔮 Next-Horizon Quantitative Strategy & Market Feed ({team_a_name})")

    # Section 1: Game-Theoretic Matchup Inversion Solver (Pillar 5.1)
    st.markdown("#### ♟️ 1. Game-Theoretic Matchup Inversion Engine")
    st.caption("Analyzes your opponent's starting roster volatility and expected margin to compute the exact marginal sensitivity of Win Probability to lineup variance (∂P(Win)/∂σ_M).")

    inv_result = simulation.analyze_matchup_inversion(
        team_a_roster=team_a_roster,
        team_a_bench=team_a_bench,
        team_b_roster=team_b_roster,
        matchup_result=matchup_result,
        waiver_upgrades_df=df_upgrades
    )

    inv_col1, inv_col2, inv_col3, inv_col4 = st.columns(4)
    with inv_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Variance Sensitivity</div>
            <div class="metric-value {'amber-val' if inv_result['dp_dsigma_pct'] > 0 else 'green-val'}" style="font-size:1.35rem;">{inv_result['dp_dsigma_pct']:+.2f}% / pt</div>
            <div class="metric-sub">∂P(Win)/∂σ_M</div>
        </div>
        """, unsafe_allow_html=True)
    with inv_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Opponent Volatility Profile</div>
            <div class="metric-value purple-val" style="font-size:1.15rem;">{inv_result['opp_profile']}</div>
            <div class="metric-sub">Opp Skew Ratio: {inv_result['opp_skew_ratio']:.2f} (σ_B={inv_result['sigma_b']:.1f})</div>
        </div>
        """, unsafe_allow_html=True)
    with inv_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Margin Volatility (σ_M)</div>
            <div class="metric-value blue-val" style="font-size:1.5rem;">±{inv_result['sigma_margin']:.1f} pts</div>
            <div class="metric-sub">Expected Spread: {inv_result['spread']:+.1f} pts</div>
        </div>
        """, unsafe_allow_html=True)
    with inv_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Own Lineup Volatility (σ_A)</div>
            <div class="metric-value green-val" style="font-size:1.5rem;">±{inv_result['sigma_a']:.1f} pts</div>
            <div class="metric-sub">Correlated Copula Draw</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background: #0f172a; border: 1px solid #38bdf8; border-radius: 10px; padding: 12px 16px; margin: 12px 0;">
        <div style="font-weight: 800; color: #38bdf8; font-size: 0.95rem; margin-bottom: 4px;">{inv_result['strategy_headline']}</div>
        <div style="font-size: 0.85rem; color: #e2e8f0; margin-bottom: 6px;">{inv_result['strategy_summary']}</div>
    </div>
    """, unsafe_allow_html=True)

    if inv_result.get("tactical_moves"):
        for move_str in inv_result["tactical_moves"]:
            st.markdown(f"- {move_str}")

    st.markdown("---")

    # Section 2: Multi-Week Playoff & First-Round Bye Leverage Matrix (Pillar 5.2)
    st.markdown("#### 🏆 2. Multi-Week Playoff & First-Round Bye Leverage Matrix")
    st.caption("Simulates remaining regular-season schedule paths across the entire league to quantify how top waiver and bench upgrades shift your probability of clinching a Playoff berth and a First-Round Bye (Marginal Bye Leverage ΔP(Bye)).")

    league_teams_for_sim = st.session_state.espn_league.get("teams", []) if is_espn_active else []
    curr_wk = int(st.session_state.espn_league.get("current_week", 6)) if is_espn_active else 6
    candidate_pool_for_bye = (waiver_pool[:8] if waiver_pool else []) + (team_a_bench[:4] if team_a_bench else [])

    playoff_matrix = simulation.simulate_playoff_leverage_matrix(
        team_a_name=team_a_name,
        team_a_roster=team_a_roster,
        waiver_candidates=candidate_pool_for_bye,
        current_week=curr_wk,
        total_reg_weeks=14,
        n_season_sims=2000,
        league_teams=league_teams_for_sim
    )

    pm_col1, pm_col2, pm_col3, pm_col4 = st.columns(4)
    with pm_col1:
        st.metric("Baseline Playoff Odds P(Top 6)", f"{playoff_matrix['base_p_playoffs']*100:.1f}%")
    with pm_col2:
        st.metric("Baseline 1st-Round Bye Odds P(Top 2)", f"{playoff_matrix['base_p_bye']*100:.1f}%")
    with pm_col3:
        st.metric("Expected Regular Season Wins", f"{playoff_matrix['expected_wins']:.1f} Wins")
    with pm_col4:
        st.metric("Remaining Schedule Horizon", f"{playoff_matrix['remaining_weeks']} Weeks")

    if not playoff_matrix["leverage_table"].empty:
        st.dataframe(playoff_matrix["leverage_table"], use_container_width=True, hide_index=True)

    st.markdown("---")

    # Section 3: Active Roster & Vegas Consensus Player Props (Zero Raw μ/σ Syntax)
    st.markdown("#### 🎯 3. Vegas Consensus Player Props & Actionable Percentile Projections")
    st.caption("Consensus median player prop lines across bookmakers translated through your league's exact idiosyncratic scoring rules into `Projected (Med)`, `Floor (P10)`, and `Ceiling (P90)`.")

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
        pcts = simulation.get_analytical_percentiles(pos, mu, sigma, p_act)

        prop_table_rows.append({
            "Player": p_title,
            "Pos": pos,
            "Pass Yds O/U": f"{p_props.get('pass_yds', '-'):.1f}" if "pass_yds" in p_props else "-",
            "Completions": f"{p_props.get('pass_completions', '-'):.1f}" if "pass_completions" in p_props else "-",
            "Rush Yds O/U": f"{p_props.get('rush_yds', '-'):.1f}" if "rush_yds" in p_props else "-",
            "Receptions O/U": f"{p_props.get('receptions', '-'):.1f}" if "receptions" in p_props else "-",
            "De-Vigged TD%": f"{p_props.get('anytime_td_prob', 0)*100:.1f}%" if "anytime_td_prob" in p_props else "-",
            "Projected (Med)": round(pcts["median"], 1),
            "Floor (P10)": round(pcts["p10"], 1),
            "Ceiling (P90)": round(pcts["p90"], 1)
        })

    if prop_table_rows:
        st.dataframe(pd.DataFrame(prop_table_rows), use_container_width=True, hide_index=True)

    with st.expander("🏈 View NFL Game Lines, Polymarket Crowd Risk Inspector & 32-Team Defensive Schemes", expanded=False):
        raw_odds = data_ingest.fetch_nfl_odds(api_key=odds_api_key)
        if raw_odds:
            parsed_games = data_ingest.parse_nfl_games(raw_odds)
            df_games = pd.DataFrame(parsed_games)
            st.dataframe(
                df_games[["away_team", "home_team", "spread", "total", "away_implied", "home_implied", "commence_time"]],
                use_container_width=True,
                hide_index=True
            )

        poly_col1, poly_col2 = st.columns([2, 1])
        with poly_col1:
            custom_slug = st.text_input(
                "Polymarket Event Slug",
                value="nfl-will-the-bills-beat-the-lions",
                help="Example: nfl-will-the-bills-beat-the-lions"
            )
        with poly_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            query_poly = st.button("Query Polymarket Gamma API", use_container_width=True)

        if query_poly or custom_slug:
            poly_result = data_ingest.fetch_polymarket_event(custom_slug)
            if poly_result:
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.metric("Crowd Implied Probability", f"{poly_result['active_probability'] * 100:.1f}%")
                with p_col2:
                    st.metric("Market Volume ($)", f"${poly_result.get('volume', 0):,.2f}")

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
# TAB 5: SCHEME LAB & DUAL NEWSLETTER INGESTION (PREVIEWS + REVIEWS)
# ===========================================================================
with tab_scheme:
    st.markdown("### 🔬 SumerSports Scheme Lab: Dual Newsletter Ingestion & Longitudinal Deep Dive")
    st.markdown(
        "Ingest **Weekly Matchup Previews** (`data/scheme_previews.parquet`) and **Weekly Game Reviews** "
        "(`data/scheme_reviews.parquet`) to track **Expectation vs. Reality Deltas**, synthesize longitudinal "
        "**Team Narratives** (`data/team_narratives.json`), and trigger structural simulation regime tags."
    )

    # -----------------------------------------------------------------------
    # 1. BATCH BACKFILL & INDIVIDUAL DRAG-AND-DROP UPLOADER
    # -----------------------------------------------------------------------
    ing_col_batch, ing_col_single = st.columns([1.15, 1.85])

    with ing_col_batch:
        st.markdown("#### 📂 Batch Backfill (`newsletters/`)")
        st.caption(
            "Scans `newsletters/` for `week*_preview.pdf` and `week*_review.pdf` in chronological order "
            "(Week 1 ➔ Present), idempotently upserts both Parquet tables, and rebuilds `team_narratives.json`."
        )
        if st.button("🚀 Batch Process Newsletters Folder", type="primary", use_container_width=True, key="batch_ingest_btn"):
            prog_bar = st.progress(10, text="Scanning newsletters/ directory for preview & review PDFs...")
            batch_res = scheme_db.batch_ingest_newsletters(folder_path="newsletters", auto_seed_if_empty=True)
            prog_bar.progress(75, text="Rebuilding longitudinal team_narratives.json ledger...")
            prog_bar.progress(100, text="Batch backfill complete!")
            st.session_state["last_batch_result"] = batch_res
            st.cache_data.clear()

        if st.session_state.get("last_batch_result"):
            b_res = st.session_state["last_batch_result"]
            st.success(
                f"Processed **{b_res['files_processed']} PDFs** "
                f"({b_res['previews_upserted']} Previews, {b_res['reviews_upserted']} Reviews) · "
                f"**{b_res['narratives_count']} Teams** updated in `team_narratives.json`."
            )
            if not b_res["summary_df"].empty:
                st.dataframe(b_res["summary_df"], use_container_width=True, hide_index=True)

    with ing_col_single:
        st.markdown("#### 📄 Individual Weekly Newsletter Uploader")
        up_c1, up_c2, up_c3 = st.columns([1.5, 1.2, 0.8])
        with up_c1:
            uploaded_pdf = st.file_uploader(
                "Upload SumerSports PDF (Preview or Review)",
                type=["pdf"],
                help="Drag and drop a Weekly Matchup Preview or Weekly Stats & Scheme Review PDF."
            )
        with up_c2:
            doc_type_toggle = st.radio(
                "Document Type",
                ["Auto-Detect", "Weekly Preview", "Weekly Review"],
                index=0,
                horizontal=False,
                key="scheme_doc_type_radio"
            )
        with up_c3:
            tag_week = st.number_input("NFL Week", min_value=1, max_value=22, value=3, step=1)
            load_sample_rev = st.button("📄 Sample Review", use_container_width=True, help="Load sample Week 3 Review PDF")
            load_sample_prev = st.button("📘 Sample Preview", use_container_width=True, help="Load sample Week 3 Preview PDF")

        pdf_source = None
        forced_type = None if doc_type_toggle == "Auto-Detect" else doc_type_toggle
        if uploaded_pdf is not None:
            pdf_source = uploaded_pdf
        elif load_sample_rev:
            st.session_state["sample_pdf_mode"] = "REVIEW"
        elif load_sample_prev:
            st.session_state["sample_pdf_mode"] = "PREVIEW"

        if pdf_source is None and st.session_state.get("sample_pdf_mode"):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            if st.session_state["sample_pdf_mode"] == "PREVIEW":
                sample_path = os.path.join(base_dir, "newsletters", f"week{int(tag_week)}_preview.pdf")
                if not os.path.exists(sample_path):
                    pdf_parser.create_sample_preview_pdf(sample_path, week=int(tag_week))
                pdf_source = sample_path
                forced_type = "Weekly Preview"
            else:
                sample_path = os.path.join(base_dir, "sample_sumersports_week3.pdf")
                if not os.path.exists(sample_path):
                    pdf_parser.create_sample_sumersports_pdf(sample_path, week=int(tag_week))
                pdf_source = sample_path
                forced_type = "Weekly Review"

        if pdf_source is not None:
            with st.spinner("Detecting schema & extracting tables via pdfplumber..."):
                parsed = pdf_parser.parse_sumersports_newsletter(
                    pdf_source,
                    default_week=int(tag_week),
                    force_doc_type=forced_type
                )

            if parsed.get("success"):
                detected_week = parsed.get("week", int(tag_week))
                detected_doc_type = parsed.get("doc_type", "Weekly Review")

                if detected_doc_type == "Weekly Preview":
                    prev_df = parsed.get("preview_metrics", pd.DataFrame())
                    ang_df = parsed.get("player_angles", pd.DataFrame())
                    st.info(
                        f"📘 **Detected: {detected_doc_type} (Week {detected_week})** — "
                        f"{len(prev_df)} Team Coverage Shell Previews & {len(ang_df)} Player Matchup Angles extracted."
                    )
                    if st.button("💾 Save / Upsert Preview to scheme_previews.parquet", type="primary", key="save_prev_btn"):
                        s_res = scheme_db.save_preview_metrics(prev_df, ang_df, week=detected_week)
                        st.success(
                            f"Upserted Week {detected_week} Preview into `data/scheme_previews.parquet` "
                            f"({s_res['preview_rows']} teams, {s_res['angle_rows']} player angles) and updated `team_narratives.json`!"
                        )
                        st.cache_data.clear()

                    p_tab1, p_tab2 = st.tabs(["📘 Preview Coverage Shells & Funnels", "🎯 Player Matchup Angles"])
                    with p_tab1:
                        st.dataframe(prev_df, use_container_width=True, hide_index=True)
                    with p_tab2:
                        st.dataframe(ang_df, use_container_width=True, hide_index=True)
                else:
                    team_df = parsed.get("team_metrics", pd.DataFrame())
                    player_df = parsed.get("player_metrics", pd.DataFrame())
                    st.success(
                        f"📄 **Detected: {detected_doc_type} (Week {detected_week})** — "
                        f"{len(team_df)} Realized Team Schemes & {len(player_df)} Player Usage rows extracted."
                    )
                    if st.button("💾 Save / Upsert Review to scheme_reviews.parquet", type="primary", key="save_rev_btn"):
                        s_res = scheme_db.save_newsletter_metrics(team_df, player_df, week=detected_week)
                        st.success(
                            f"Upserted Week {detected_week} Review into `data/scheme_reviews.parquet` "
                            f"({s_res['team_rows']} teams, {s_res['player_rows']} players) and updated `team_narratives.json`!"
                        )
                        st.cache_data.clear()

                    r_tab1, r_tab2 = st.tabs(["🛡️ Realized Team Scheme Metrics", "🏃 Realized Player Usage Metrics"])
                    with r_tab1:
                        st.dataframe(team_df, use_container_width=True, hide_index=True)
                    with r_tab2:
                        st.dataframe(player_df, use_container_width=True, hide_index=True)
            else:
                st.error(f"Error parsing PDF: {parsed.get('error', 'Unknown parsing error')}")

    # -----------------------------------------------------------------------
    # 2. LONGITUDINAL TEAM DEEP DIVE (WEEK 1 -> PRESENT)
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🧭 Longitudinal Team Deep Dive (Preview Hypotheses vs. Review Realities)")

    all_teams = scheme_db.get_all_teams_in_db()
    default_team_idx = all_teams.index("CLE") if "CLE" in all_teams else 0
    dd_col1, dd_col2 = st.columns([1, 3])
    with dd_col1:
        sel_team = st.selectbox("Select NFL Team for Deep Dive", all_teams, index=default_team_idx, key="deep_dive_team_select")
        if st.button("🔄 Reset / Re-Seed 3-Wk Baseline", use_container_width=True, key="reseed_scheme_btn"):
            scheme_db.init_default_scheme_history(force=True)
            st.cache_data.clear()
            st.rerun()

    deep_dive = scheme_db.get_team_longitudinal_deep_dive(sel_team)
    exp_delta = deep_dive["expectation_delta"]
    rolling_stats = deep_dive["rolling_3wk"]
    regime_tags = deep_dive["regime_tags"]

    with dd_col2:
        tag_badges_html = ""
        tag_colors = {
            "PASS_PROTECTION_COLLAPSE": ("#ef4444", "rgba(239, 68, 68, 0.18)"),
            "SLOT_FUNNEL_HIGH": ("#10b981", "rgba(16, 185, 129, 0.18)"),
            "RUN_GAME_BROKEN": ("#f59e0b", "rgba(245, 158, 11, 0.18)"),
            "GARBAGE_TIME_ALPHA": ("#c084fc", "rgba(192, 132, 252, 0.18)"),
        }
        if regime_tags:
            for rt in regime_tags:
                fg, bg = tag_colors.get(rt, ("#38bdf8", "rgba(56, 189, 248, 0.18)"))
                tag_badges_html += (
                    f"<span style='background:{bg}; color:{fg}; border:1px solid {fg}; "
                    f"padding:3px 10px; border-radius:999px; font-size:0.76rem; font-weight:800; margin-right:8px;'>"
                    f"⚡ {rt}</span>"
                )
        else:
            tag_badges_html = (
                "<span style='background:rgba(148,163,184,0.15); color:#cbd5e1; padding:3px 10px; "
                "border-radius:999px; font-size:0.76rem; font-weight:700;'>✅ STABLE REGIME (ON EXPECTATION)</span>"
            )

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border: 1px solid #334155; border-left: 5px solid #38bdf8; border-radius: 10px; padding: 12px 18px; margin-bottom: 10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <span style="color:#f8fafc; font-weight:800; font-size:1.05rem;">🏛️ {sel_team} Longitudinal Regime Profile</span>
                <div>{tag_badges_html}</div>
            </div>
            <div style="color:#94a3b8; font-size:0.84rem; margin-top:6px;">
                <b>Preseason Expectation:</b> {deep_dive['preseason_expectation']}
            </div>
            <div style="color:#cbd5e1; font-size:0.84rem; margin-top:4px;">
                <b>Expectation vs. Reality Verdict:</b> {exp_delta.get('verdict', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Top 4 KPI Delta Cards
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    with m_c1:
        st.metric(
            "Realized vs Previewed QP%",
            f"{exp_delta['realized_qpr']*100:.1f}%",
            delta=f"{exp_delta['qpr_delta']*100:+.1f}% vs Preview ({exp_delta['expected_qpr']*100:.1f}%)",
            delta_color="inverse"
        )
    with m_c2:
        st.metric(
            "Realized vs Previewed MOF EPA",
            f"{exp_delta['realized_mof_epa']:+.2f}",
            delta=f"{exp_delta['mof_epa_delta']:+.2f} vs Preview ({exp_delta['expected_mof_epa']:+.2f})",
            delta_color="inverse"
        )
    with m_c3:
        st.metric(
            "Bayesian Prior Decay Multiplier",
            f"{exp_delta['prior_decay_multiplier']:.2f}x",
            delta="Accelerated Decay" if exp_delta["prior_decay_multiplier"] < 1.0 else "Standard Prior Weight"
        )
    with m_c4:
        st.metric(
            "Opp Slot/TE Floor & Ceiling Mod",
            f"{exp_delta['slot_te_mu_boost']:.2f}x Med",
            delta=f"Floor {exp_delta['slot_te_floor_boost']:.2f}x · Ceiling {exp_delta['slot_te_ceiling_boost']:.2f}x"
        )

    # Week-by-Week Timeline Table (Preview Hypotheses vs Review Realities)
    timeline_df = deep_dive["timeline_df"]
    if not timeline_df.empty:
        st.markdown(f"#### 📅 {sel_team} Week-by-Week Preview Hypotheses vs. Review Execution")
        disp_tl = pd.DataFrame({
            "Week": timeline_df["week"],
            "Opp": timeline_df["opponent"],
            "Preview Hypothesis (Shells & Funnel Focus)": timeline_df["preview_hypothesis"],
            "Review Execution & Takeaway": timeline_df["review_execution"],
            "Exp QP%": (timeline_df["expected_qpr"] * 100).round(1).astype(str) + "%",
            "Real QP%": (timeline_df["realized_qpr"] * 100).round(1).astype(str) + "%",
            "QP Δ": (timeline_df["qpr_delta"] * 100).map(lambda x: f"{x:+.1f}%"),
            "Exp MOF EPA": timeline_df["expected_mof_epa"].map(lambda x: f"{x:+.2f}"),
            "Real MOF EPA": timeline_df["realized_mof_epa"].map(lambda x: f"{x:+.2f}"),
            "Pass OE%": timeline_df["pass_oe"].map(lambda x: f"{x:+.1f}%"),
        })
        st.dataframe(disp_tl, use_container_width=True, hide_index=True)

        # Plotly Trend Curves (Week 1 -> Present)
        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=timeline_df["week"],
                y=timeline_df["pressure_rate"] * 100,
                mode="lines+markers",
                name="Total Pressure Rate %",
                line=dict(color="#f59e0b", width=2, dash="dot")
            ))
            fig_pr.add_trace(go.Scatter(
                x=timeline_df["week"],
                y=timeline_df["realized_qpr"] * 100,
                mode="lines+markers",
                name="Realized Quick Pressure % (<2.5s)",
                line=dict(color="#ef4444", width=3)
            ))
            fig_pr.add_trace(go.Scatter(
                x=timeline_df["week"],
                y=timeline_df["expected_qpr"] * 100,
                mode="lines+markers",
                name="Previewed Expected QP %",
                line=dict(color="#38bdf8", width=2, dash="dash")
            ))
            fig_pr.update_layout(
                title=f"{sel_team} Pressure Rate & Quick Pressure (Realized vs. Previewed)",
                xaxis=dict(title="NFL Week", dtick=1),
                yaxis=dict(title="Pressure Rate (%)"),
                template="plotly_dark",
                height=340,
                margin=dict(l=35, r=25, t=45, b=35),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        with ch_col2:
            fig_mof = go.Figure()
            fig_mof.add_trace(go.Scatter(
                x=timeline_df["week"],
                y=timeline_df["realized_mof_epa"],
                mode="lines+markers",
                name="Realized MOF EPA Allowed",
                line=dict(color="#10b981", width=3)
            ))
            fig_mof.add_trace(go.Scatter(
                x=timeline_df["week"],
                y=timeline_df["expected_mof_epa"],
                mode="lines+markers",
                name="Previewed MOF EPA",
                line=dict(color="#94a3b8", width=2, dash="dash")
            ))
            fig_mof.add_trace(go.Bar(
                x=timeline_df["week"],
                y=timeline_df["pass_oe"],
                name="Pass Over Expected (POE %)",
                yaxis="y2",
                marker_color="rgba(192, 132, 252, 0.35)"
            ))
            fig_mof.update_layout(
                title=f"{sel_team} MOF EPA (Realized vs. Previewed) & Pass OE %",
                xaxis=dict(title="NFL Week", dtick=1),
                yaxis=dict(title="MOF EPA / Play"),
                yaxis2=dict(title="Pass OE (%)", overlaying="y", side="right"),
                template="plotly_dark",
                height=340,
                margin=dict(l=35, r=35, t=45, b=35),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_mof, use_container_width=True)

    # Primary WR/TE Target Share Trajectory (Week 1 -> Present)
    st.markdown("#### 🏃 Primary Skill Player Target Share & Usage Evolution (Week 1 ➔ Present)")
    team_players_df = deep_dive["player_usage_df"]
    all_players = scheme_db.get_all_players_in_db()
    p_col_sel, p_col_chart = st.columns([1, 2.5])
    with p_col_sel:
        team_specific_players = sorted(team_players_df["player_name"].unique().tolist()) if not team_players_df.empty else []
        default_p_list = team_specific_players if team_specific_players else (all_players or ["CeeDee Lamb", "Demario Douglas"])
        sel_player = st.selectbox(
            f"Select Player ({sel_team} or League-Wide)",
            default_p_list + [p for p in all_players if p not in default_p_list],
            index=0,
            key="deep_dive_player_select"
        )
        p_hist = scheme_db.get_player_usage_history(sel_player)
        if not p_hist.empty:
            st.metric("Latest Target Share", f"{p_hist.iloc[-1]['target_share']*100:.1f}%")
            st.metric("Latest YPRR", f"{p_hist.iloc[-1]['yprr']:.2f}")
            st.metric("Latest aDOT", f"{p_hist.iloc[-1]['adot']:.1f} yds")

    with p_col_chart:
        if not p_hist.empty:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=p_hist["week"],
                y=p_hist["target_share"] * 100,
                mode="lines+markers",
                name="Target Share %",
                line=dict(color="#38bdf8", width=3)
            ))
            fig_p.add_trace(go.Scatter(
                x=p_hist["week"],
                y=p_hist["yprr"],
                mode="lines+markers",
                name="Yards Per Route Run (YPRR)",
                yaxis="y2",
                line=dict(color="#10b981", width=2, dash="dash")
            ))
            fig_p.update_layout(
                title=f"{sel_player} Longitudinal Target Share % & YPRR Trajectory",
                xaxis=dict(title="NFL Week Number", dtick=1),
                yaxis=dict(title="Target Share (%)"),
                yaxis2=dict(title="YPRR", overlaying="y", side="right"),
                template="plotly_dark",
                height=310,
                margin=dict(l=35, r=35, t=45, b=35),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info(f"No historical usage records found for {sel_player}.")


    # Weekly Scheme Matchup Impact on Team A
    st.markdown("---")
    st.markdown(f"### 🛡️ Weekly Scheme Matchup Impact on {team_a_name}")
    st.caption(f"How opposing NFL defensive schemes (Quick Pressure %, MOF EPA allowed, Rush SR allowed) specifically modulate {team_a_name}'s players.")

    scheme_impact_rows = []
    for p in team_a_roster + team_a_bench:
        p_team = p.get("team", "NFL")
        p_pos = p.get("position", "WR")
        opp_def = p.get("opponent") or p.get("opponent_team") or getattr(data_ingest, "CURRENT_NFL_SCHEDULE", {}).get(str(p_team).upper(), "") or "DEF"
        modified_p = data_ingest.apply_defensive_scheme_modifiers(p, opponent_team=opp_def)
        sch = data_ingest.get_defensive_scheme(opp_def)

        base_pcts = simulation.get_analytical_percentiles(p_pos, float(p.get("mu", 10.0)), float(p.get("sigma", 4.0)), float(p.get("p_active", 1.0)))
        adj_pcts = simulation.get_analytical_percentiles(p_pos, float(modified_p.get("mu", 10.0)), float(modified_p.get("sigma", 4.0)), float(modified_p.get("p_active", 1.0)))

        scheme_impact_rows.append({
            "Player": p.get("name"),
            "Pos": p_pos,
            "Slot": p.get("slot", "BE"),
            "NFL Team": p_team,
            "Opp Defense": opp_def,
            "Opp Quick Press%": f"{sch['quick_pressure_rate']*100:.1f}%",
            "Opp MOF EPA": f"{sch['mof_epa_allowed']:.2f}",
            "Opp Rush SR%": f"{sch['rush_sr_allowed']*100:.1f}%",
            "Base (Med)": round(base_pcts["median"], 1),
            "Projected (Med)": round(adj_pcts["median"], 1),
            "Floor (P10)": round(adj_pcts["p10"], 1),
            "Ceiling (P90)": round(adj_pcts["p90"], 1),
            "Scheme Impact Notes": modified_p.get("scheme_notes", "Neutral baseline matchup")
        })

    st.dataframe(pd.DataFrame(scheme_impact_rows), use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 4: IDIOSYNCRATIC TRADE ANALYZER & BILATERAL VALUATION TERMINAL
# ===========================================================================
with tab_trade:
    st.markdown("### 🤝 Idiosyncratic Trade Analyzer & Bilateral Lineup Simulator")
    st.caption(
        "Calibrated specifically to your ESPN league's idiosyncratic scoring rules "
        "(6.0 pt Pass TD, +0.1/comp, 1.0 Full PPR, stacking +1/+2 pt 100/200-Yd Gamma/Log-Normal milestones, "
        "stacking +1/+2/+4 pt 40/50-Yd TD bonuses) and 9-Starter 2-QB/OP Superflex roster structure."
    )

    # Resolve available league teams & rosters (Live ESPN sync or calibrated Week 6 fallback)
    if is_espn_active and st.session_state.get("espn_league"):
        trade_team_map = {
            t["team_name"]: t.get("roster", t.get("starters", []) + t.get("bench", []) + t.get("ir", []))
            for t in st.session_state.espn_league.get("teams", [])
        }
    else:
        trade_team_map = trade_analyzer.get_default_bodhizafa_trade_rosters(scoring_cfg=active_scoring_cfg)
        if team_a_name not in trade_team_map:
            trade_team_map[team_a_name] = team_a_roster + raw_bench_a
        if team_b_name not in trade_team_map:
            trade_team_map[team_b_name] = team_b_roster

    trade_team_names = list(trade_team_map.keys())

    # Default Team A to Eatin TDs like Groceries, Team B to BODHIZAFA
    def_ta_idx = 0
    for idx_t, t_nm in enumerate(trade_team_names):
        if "eatin" in t_nm.lower() or t_nm == team_a_name:
            def_ta_idx = idx_t
            break

    t_col_a, t_col_b = st.columns(2)
    with t_col_a:
        trade_team_a = st.selectbox(
            "🏠 Your Franchise (Team A)",
            trade_team_names,
            index=def_ta_idx,
            key="trade_team_a_select",
        )
    with t_col_b:
        partner_options = [t for t in trade_team_names if t != trade_team_a] or trade_team_names
        def_tb_idx = 0
        for idx_t, t_nm in enumerate(partner_options):
            if "bodhizafa" in t_nm.lower():
                def_tb_idx = idx_t
                break
        trade_team_b = st.selectbox(
            "🤝 Trade Partner Franchise (Team B)",
            partner_options,
            index=def_tb_idx,
            key="trade_team_b_select",
        )

    roster_a_full = trade_team_map.get(trade_team_a, [])
    roster_b_full = trade_team_map.get(trade_team_b, [])

    a_player_names = [p["name"] for p in roster_a_full]
    b_player_names = [p["name"] for p in roster_b_full]

    default_gives = [n for n in ["DJ Moore", "Chase Brown", "Jalen Coker"] if n in a_player_names]
    default_receives = [
        n for n in ["Jonathan Taylor", "Dontayvion Wicks", "Devaughn Vele", "Seth McGowan"]
        if n in b_player_names
    ]

    sel_col_out, sel_col_in = st.columns(2)
    with sel_col_out:
        selected_gives = st.multiselect(
            f"📤 {trade_team_a} Trades Away (Sends to {trade_team_b})",
            options=a_player_names,
            default=default_gives,
            key="trade_gives_multiselect",
        )
    with sel_col_in:
        selected_receives = st.multiselect(
            f"📥 {trade_team_a} Acquires (Receives from {trade_team_b})",
            options=b_player_names,
            default=default_receives,
            key="trade_receives_multiselect",
        )

    # Optional Forced Roster Drop Override if receiving more players than sending
    forced_drop_choice = None
    if len(selected_receives) > len(selected_gives):
        n_extra = len(selected_receives) - len(selected_gives)
        remaining_bench_names = [
            p["name"] for p in roster_a_full
            if p["name"] not in selected_gives
            and str(p.get("position", "")).upper() not in ["QB", "D/ST", "DEF"]
            and str(p.get("slot", "")).upper() not in ["IR", "IR_ELIGIBLE"]
        ]
        default_drop_idx = 0
        for idx_d, d_nm in enumerate(remaining_bench_names):
            if "dowdle" in d_nm.lower():
                default_drop_idx = idx_d
                break
        drop_opts = remaining_bench_names + [n for n in selected_receives if n not in remaining_bench_names] + ["None (Wait / Use Open Slot)"]
        forced_drop_choice = st.selectbox(
            f"⚠️ Uneven Trade ({len(selected_receives)}-for-{len(selected_gives)}): Select {n_extra} Bench Player to Drop for Roster Compliance",
            options=drop_opts,
            index=default_drop_idx if default_drop_idx < len(drop_opts) else 0,
            key="trade_forced_drop_select",
        )

    if not selected_gives or not selected_receives:
        st.info("Select at least one outgoing and one incoming player above to run the 10,000-iteration Idiosyncratic Trade Evaluation.")
    else:
        with st.spinner("Running 10,000-iteration Dual-Horizon Monte Carlo Trade Evaluation..."):
            trade_eval = trade_analyzer.evaluate_trade_package(
                team_a_name=trade_team_a,
                team_a_roster=roster_a_full,
                team_b_name=trade_team_b,
                team_b_roster=roster_b_full,
                team_a_gives_names=selected_gives,
                team_a_receives_names=selected_receives,
                week6_opponent_name=team_b_name,
                week6_opponent_starters=team_b_roster,
                waiver_pool=waiver_pool,
                scoring_cfg=active_scoring_cfg,
                n_simulations=n_iterations,
                forced_drop_override=forced_drop_choice,
            )

        tm = trade_eval["metrics"]
        verdict_color = "#10b981" if trade_eval["composite_edge_pts"] >= 1.0 else ("#f59e0b" if trade_eval["composite_edge_pts"] >= -1.25 else "#ef4444")

        # Executive Verdict Banner
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border: 2px solid {verdict_color}; border-radius: 12px; padding: 16px 22px; margin: 12px 0 18px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">
                        Idiosyncratic League Trade Verdict · {len(selected_receives)}-for-{len(selected_gives)} Package
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: {verdict_color}; margin-top: 2px;">
                        {trade_eval['verdict']}
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid {verdict_color}; color: {verdict_color}; font-size: 1.35rem; font-weight: 900; padding: 6px 16px; border-radius: 10px;">
                        Grade: {trade_eval['grade']} ({trade_eval['composite_edge_pts']:+.2f} Composite Index)
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top 4 KPI Metric Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            wk_col = "green-val" if tm["wk6_starter_delta"] >= 0 else "amber-val"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Immediate Week 6 Starter Δ</div>
                <div class="metric-value {wk_col}">{tm['wk6_starter_delta']:+.2f} pts</div>
                <div class="metric-sub">{tm['wk6_starter_pre']:.1f} ➔ {tm['wk6_starter_post']:.1f} pts (P10 Floor: {tm['wk6_p10_pre']:.1f} ➔ {tm['wk6_p10_post']:.1f})</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            wp_col = "green-val" if tm["wk6_win_prob_delta"] >= 0 else "amber-val"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Week 6 Win Prob Δ vs {trade_eval['week6_opponent_name']}</div>
                <div class="metric-value {wp_col}">{tm['wk6_win_prob_delta']:+.1f}%</div>
                <div class="metric-sub">{tm['wk6_win_prob_pre']:.1f}% ➔ {tm['wk6_win_prob_post']:.1f}% (More than 2x Win Odds!)</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            rm_col = "green-val" if tm["ros_model_effective_delta"] >= 0 else "amber-val"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ROS Model Starter Δ (+Handcuff EV)</div>
                <div class="metric-value {rm_col}">{tm['ros_model_effective_delta']:+.2f} PPG</div>
                <div class="metric-sub">Lineup Δ: {tm['ros_model_starter_delta']:+.2f} + IND Cuff Insurance: +{trade_eval['synergies_a']['total_handcuff_insurance_ev']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            po_col = "green-val" if tm["playoff_w15_17_starter_delta"] >= 0 else "amber-val"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🏆 Fantasy Playoffs Δ (Wks 15–17)</div>
                <div class="metric-value {po_col}">{tm['playoff_w15_17_starter_delta']:+.2f} PPG</div>
                <div class="metric-sub">{tm['playoff_w15_17_pre_avg']:.1f} ➔ {tm['playoff_w15_17_post_avg']:.1f} PPG | Wks 7–14 Avg Δ: {tm['w7_14_starter_delta']:+.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        # Season-Long & Championship Horizon Callout Banner
        if tm["wk6_starter_delta"] > 0 and tm["playoff_w15_17_starter_delta"] < -1.5:
            st.error(
                f"🚨 **Season-Long & Championship Warning ('Week 6 Mirage' Alert)**: "
                f"While this trade boosts your **Week 6** lineup by **`{tm['wk6_starter_delta']:+.2f}` pts** (because `Chase Brown` is on his Week 6 Bye and `Jonathan Taylor` faces `TEN`), "
                f"it **reduces** your optimal starting lineup by **`{tm['w7_14_starter_delta']:+.2f}` PPG across Weeks 7–14** (due to trading away two starting WRs with completed/early byes while absorbing `Jonathan Taylor`'s Week 13 Bye) "
                f"and **reduces your Fantasy Playoff (Weeks 15–17) lineup by `{tm['playoff_w15_17_starter_delta']:+.2f}` PPG** (`{tm['playoff_w15_17_pre_avg']:.2f}` ➔ `{tm['playoff_w15_17_post_avg']:.2f}` PPG). "
                f"**Counter-Offer Fix**: Keep `Jalen Coker` (2-for-2: `Chase Brown + DJ Moore` for `Jonathan Taylor + Seth McGowan`) or demand `Sam LaPorta` / `Denzel Boston` instead of bench WRs `Devaughn Vele + Dontayvion Wicks`!"
            )

        # Structural Synergies & Uneven Drop Callouts
        if trade_eval["synergies_a"]["synergy_notes"] or trade_eval["dropped_players_a"]:
            st.markdown("#### 🔬 Structural Synergies, Handcuff Insurance & Roster-Slot Adjustments")
            for s_note in trade_eval["synergies_a"]["synergy_notes"]:
                st.success(s_note)
                if trade_eval["dropped_players_a"]:
                    d_p = trade_eval["dropped_players_a"][0]
                    st.warning(
                        f"⚖️ **4-for-3 Roster Compliance Drop (`{d_p['name']}`, {d_p.get('position')} - {d_p.get('team')})**: "
                        f"Because you receive {len(selected_receives)} players and send {len(selected_gives)}, you must clear 1 active roster spot. "
                        f"Dropping `{d_p['name']}` (`{d_p.get('avg_points', 0.0):.2f}` Season PPG, `{d_p.get('ros_blended_mu', 0.0):.2f}` ROS Blended, "
                        f"`{d_p.get('ros_vorp', 0.0):+.2f}` VORP) preserves your newly acquired `IND` handcuff `Seth McGowan`."
                    )

        # Full-Season Projections Board (`2026-FFB-Projections-0812.xlsx`) Consolidation vs. Dispersion Audit
        st.info(
            f"📐 **Full-Season Projections (`2026-FFB-Projections-0812.xlsx`) & Bayesian 50/50 Consolidation Audit**: "
            f"Incoming Active VORP: **`+{tm['fs_incoming_active_vorp']:.1f}`** (`${tm['fs_incoming_auc']:.2f}` Auction Value) vs. "
            f"Outgoing Active VORP: **`+{tm['fs_outgoing_active_vorp']:.1f}`** (`${tm['fs_outgoing_auc']:.2f}` Auction Value) · "
            f"Net Active VORP Δ: **`{tm['fs_net_active_vorp_delta']:+.1f}`** (`{tm['fs_net_auc_delta']:+.2f}` Auction $) · "
            f"Full-Season Model 9-Starter Δ: **`{tm['fs_starter_delta']:+.2f}` PPG** (`{tm['fs_starter_pre']:.2f}` → `{tm['fs_starter_post']:.2f}`) · "
            f"Bayesian 50/50 ROS 9-Starter Δ: **`{tm['bayes_ros_starter_delta']:+.2f}` PPG** (`{tm['bayes_ros_starter_pre']:.2f}` → `{tm['bayes_ros_starter_post']:.2f}`)."
        )

        # Side-by-Side Asset Valuation & Bonus Decomposition Tables
        st.markdown("#### 📊 Idiosyncratic Asset Valuation, Full-Season VORP/Auction & Playoff Schedule (Wks 15–17)")
        st.caption(
            "Compares Full-Season Projections (`FS Rank`, `FS PPG`, `FS VORP`, `FS Auc $` from `2026-FFB-Projections-0812.xlsx`), "
            "Realized Season PPG, ESPN League ROS PPG, True In-Season ROS (`True ROS PPG`), Bayesian 50/50 Blend (`Bayes 50/50 PPG`), "
            "and SumerSports Scheme-Adjusted Fantasy Playoff Average (`Playoffs W15-17`)."
        )
        st.markdown(f"**📥 Assets Acquired by {trade_team_a}**")
        st.dataframe(trade_eval["tables"]["incoming_df"], use_container_width=True, hide_index=True)

        st.markdown(f"**📤 Assets Sent to {trade_team_b}**")
        st.dataframe(trade_eval["tables"]["outgoing_df"], use_container_width=True, hide_index=True)

        # Interactive Plotly Charts: Week-by-Week Championship Trajectory (Wks 6-17) + Slot-by-Slot Net Delta
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### 🗓️ Week-by-Week Season & Playoff Lineup Trajectory (Wks 6–17)")
            traj_df = trade_eval["tables"]["weekly_trajectory_df"]
            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(
                x=traj_df["Week"],
                y=traj_df["Pre-Trade Lineup"],
                mode="lines+markers",
                name="Pre-Trade Optimal Lineup",
                line=dict(color="#94a3b8", width=2.5, dash="dot"),
            ))
            fig_traj.add_trace(go.Scatter(
                x=traj_df["Week"],
                y=traj_df["Post-Trade Lineup"],
                mode="lines+markers",
                name="Post-Trade Optimal Lineup",
                line=dict(color="#10b981", width=3),
            ))
            fig_traj.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,41,59,0.5)",
                xaxis_title="NFL Week (W6–W14 Reg. Season Byes · W15–W17 Fantasy Playoffs)",
                yaxis_title="Optimal 9-Starter Projected Points",
                height=340,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_traj, use_container_width=True)

        with chart_col2:
            st.markdown("#### ⚖️ Multi-Horizon Slot-by-Slot Lineup Net Delta")
            wk_df_chart = trade_eval["tables"]["wk6_lineup_diff_df"]
            po_df_chart = trade_eval["tables"]["playoff_lineup_diff_df"]
            by_df_chart = trade_eval["tables"]["bayes_lineup_diff_df"]
            fig_slot_delta = go.Figure()
            fig_slot_delta.add_trace(go.Bar(
                x=by_df_chart["Slot"],
                y=by_df_chart["Delta_Num"],
                name=f"Bayes 50/50 ROS Δ ({tm['bayes_ros_starter_delta']:+.2f} PPG)",
                marker_color="#a78bfa",
            ))
            fig_slot_delta.add_trace(go.Bar(
                x=po_df_chart["Slot"],
                y=po_df_chart["Delta_Num"],
                name=f"Playoffs W15-17 Net Δ ({tm['playoff_w15_17_starter_delta']:+.2f} PPG)",
                marker_color="#f59e0b",
            ))
            fig_slot_delta.add_trace(go.Bar(
                x=wk_df_chart["Slot"],
                y=wk_df_chart["Delta_Num"],
                name=f"Week 6 Net Δ ({tm['wk6_starter_delta']:+.2f} pts)",
                marker_color="#38bdf8",
            ))
            fig_slot_delta.update_layout(
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,41,59,0.5)",
                xaxis_title="9-Starter Roster Slot",
                yaxis_title="Net Points Delta (Post - Pre)",
                height=340,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_slot_delta, use_container_width=True)

        # Multi-Horizon Lineup & Schedule Audits
        st.markdown("#### 🗂️ Optimal 9-Starter Lineup & Championship Schedule Audits")
        sub_traj, sub_bayes, sub_fs, sub_playoffs, sub_true_ros, sub_wk6, sub_ros_model, sub_partner = st.tabs([
            "🗓️ Wks 6–17 Weekly Schedule & Byes",
            f"⚖️ Bayesian 50/50 ROS ({tm['bayes_ros_starter_delta']:+.2f} PPG)",
            f"📐 Full-Season Sheet ({tm['fs_starter_delta']:+.2f} PPG)",
            f"🏆 Playoffs W15–17 Lineup ({tm['playoff_w15_17_starter_delta']:+.2f} PPG)",
            f"📅 In-Season True ROS ({tm['true_ros_starter_delta']:+.2f} PPG)",
            f"⚡ Week 6 Lineup ({tm['wk6_starter_delta']:+.2f} pts)",
            f"🔮 Athletic Model ROS ({tm['ros_model_starter_delta']:+.2f} PPG)",
            f"🤝 Partner ({trade_team_b}) ROS ({tm['partner_ros_starter_delta']:+.2f} PPG)",
        ])
        with sub_traj:
            st.dataframe(
                trade_eval["tables"]["weekly_trajectory_df"].drop(columns=["Delta_Num"]),
                use_container_width=True,
                hide_index=True,
            )
        with sub_bayes:
            st.dataframe(
                trade_eval["tables"]["bayes_lineup_diff_df"].drop(columns=["Delta_Num"]),
                use_container_width=True,
                hide_index=True,
            )
        with sub_fs:
            st.dataframe(
                trade_eval["tables"]["fs_lineup_diff_df"].drop(columns=["Delta_Num"]),
                use_container_width=True,
                hide_index=True,
            )
        with sub_playoffs:
            st.dataframe(
                trade_eval["tables"]["playoff_lineup_diff_df"].drop(columns=["Delta_Num"]),
                use_container_width=True,
                hide_index=True,
            )
        with sub_true_ros:
            st.dataframe(
                trade_eval["tables"]["true_ros_lineup_diff_df"].drop(columns=["Delta_Num"]),
                use_container_width=True,
                hide_index=True,
            )
        with sub_wk6:
            st.dataframe(
                trade_eval["tables"]["wk6_lineup_diff_df"].drop(columns=["Delta_Num"]),
                use_container_width=True,
                hide_index=True,
            )
        with sub_ros_model:
            st.dataframe(
                trade_eval["tables"]["model_lineup_diff_df"].drop(columns=["Delta_Num"]),
                use_container_width=True,
                hide_index=True,
            )
        with sub_partner:
            st.dataframe(
                trade_eval["tables"]["partner_ros_diff_df"].drop(columns=["Delta_Num"]),
                use_container_width=True,
                hide_index=True,
            )



