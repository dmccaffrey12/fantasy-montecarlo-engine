"""
pdf_parser.py
PDF Parser Module for SumerSports "Stats & Scheme" Weekly Reviews.
Uses pdfplumber to extract:
1. Team-level game metrics:
   - Team, Opponent, Week Number
   - Pressure Rate %, Quick Pressure %
   - MOF % (Middle of Field), MOF EPA
   - Pass OE %, EPA/DB, EPA/Rush, Rush Success Rate (SR)
2. Player receiving/rushing tables:
   - Player, Team, Position, Target %, YPRR, aDOT, Rush EPA, Rush SR
Includes helper to generate compliant sample PDFs for testing.
"""

import io
import os
import re
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import pdfplumber


def clean_num(val: Any, default: float = 0.0) -> float:
    """Helper to sanitize percentage strings and floats."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace("%", "").replace("+", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def parse_sumersports_newsletter(
    pdf_source: Union[str, io.BytesIO, Any],
    default_week: Optional[int] = None
) -> Dict[str, Any]:
    """
    Parse a SumerSports 'Stats & Scheme' PDF review using pdfplumber.
    Accepts file path, BytesIO, or Streamlit UploadedFile.

    Returns:
      {
        "success": True/False,
        "week": int,
        "team_metrics": pd.DataFrame,
        "player_metrics": pd.DataFrame,
        "raw_text_preview": str,
        "error": Optional[str]
      }
    """
    team_records = []
    player_records = []
    extracted_text_blocks = []
    detected_week = default_week or 1

    try:
        # Handle path vs file buffer
        if isinstance(pdf_source, str):
            pdf = pdfplumber.open(pdf_source)
        else:
            # BytesIO or Streamlit UploadedFile
            pdf = pdfplumber.open(pdf_source)

        with pdf:
            for page_idx, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                extracted_text_blocks.append(text)

                # Detect Week Number from text headers (e.g. "Week 3", "Week 03")
                if not default_week:
                    week_match = re.search(r"Week\s*(\d+)", text, re.IGNORECASE)
                    if week_match:
                        detected_week = int(week_match.group(1))

                # Extract Tables from Page
                tables = page.extract_tables() or []
                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    # Clean header row
                    header = [str(c).strip().lower() if c else "" for c in table[0]]
                    header_str = " ".join(header)

                    # Identify Player Usage Table First (contains "player")
                    if any(k in header_str for k in ["player", "target", "yprr", "adot"]):
                        p_col_map = {}
                        for idx, col_name in enumerate(header):
                            if "player" in col_name or "name" in col_name:
                                p_col_map["player"] = idx
                            elif "team" in col_name:
                                p_col_map["team"] = idx
                            elif "pos" in col_name:
                                p_col_map["position"] = idx
                            elif "tgt" in col_name or "target" in col_name:
                                p_col_map["target_pct"] = idx
                            elif "yprr" in col_name:
                                p_col_map["yprr"] = idx
                            elif "adot" in col_name:
                                p_col_map["adot"] = idx
                            elif "rush epa" in col_name:
                                p_col_map["rush_epa"] = idx
                            elif "rush sr" in col_name or "sr" in col_name:
                                p_col_map["rush_sr"] = idx

                        for row in table[1:]:
                            if not row or not any(row):
                                continue
                            p_name = str(row[p_col_map["player"]]).strip() if "player" in p_col_map and p_col_map["player"] < len(row) and row[p_col_map["player"]] else ""
                            if not p_name or p_name.lower() in ["player", "name", "total"]:
                                continue

                            p_team = str(row[p_col_map["team"]]).strip().upper() if "team" in p_col_map and p_col_map["team"] < len(row) and row[p_col_map["team"]] else "FA"
                            p_pos = str(row[p_col_map["position"]]).strip().upper() if "position" in p_col_map and p_col_map["position"] < len(row) and row[p_col_map["position"]] else "WR"

                            tgt_pct = clean_num(row[p_col_map["target_pct"]]) if "target_pct" in p_col_map and p_col_map["target_pct"] < len(row) else 15.0
                            if tgt_pct > 1.0:
                                tgt_pct = tgt_pct / 100.0

                            yprr = clean_num(row[p_col_map["yprr"]]) if "yprr" in p_col_map and p_col_map["yprr"] < len(row) else 1.65
                            adot = clean_num(row[p_col_map["adot"]]) if "adot" in p_col_map and p_col_map["adot"] < len(row) else 9.5
                            rush_epa = clean_num(row[p_col_map["rush_epa"]]) if "rush_epa" in p_col_map and p_col_map["rush_epa"] < len(row) else 0.0

                            p_rush_sr = clean_num(row[p_col_map["rush_sr"]]) if "rush_sr" in p_col_map and p_col_map["rush_sr"] < len(row) else 40.0
                            if p_rush_sr > 1.0:
                                p_rush_sr = p_rush_sr / 100.0

                            player_records.append({
                                "week": detected_week,
                                "player_name": p_name,
                                "team": p_team,
                                "position": p_pos,
                                "target_share": round(tgt_pct, 3),
                                "yprr": round(yprr, 2),
                                "adot": round(adot, 1),
                                "rush_epa": round(rush_epa, 2),
                                "rush_sr": round(p_rush_sr, 3)
                            })

                    # Identify Team Scheme Table
                    elif any(k in header_str for k in ["pressure", "quick press", "mof", "pass oe", "epa/db"]):
                        col_map = {}
                        for idx, col_name in enumerate(header):
                            if "team" in col_name and "opp" not in col_name:
                                col_map["team"] = idx
                            elif "opp" in col_name:
                                col_map["opponent"] = idx
                            elif "quick" in col_name or "qp" in col_name:
                                col_map["quick_pressure"] = idx
                            elif "pressure" in col_name or "press" in col_name:
                                col_map["pressure_rate"] = idx
                            elif "mof epa" in col_name or "mof_epa" in col_name:
                                col_map["mof_epa"] = idx
                            elif "mof" in col_name:
                                col_map["mof_pct"] = idx
                            elif "pass oe" in col_name or "poe" in col_name:
                                col_map["pass_oe"] = idx
                            elif "epa/db" in col_name or "epa db" in col_name:
                                col_map["epa_db"] = idx
                            elif "epa/rush" in col_name or "epa rush" in col_name:
                                col_map["epa_rush"] = idx
                            elif "rush sr" in col_name or "rush success" in col_name:
                                col_map["rush_sr"] = idx

                        for row in table[1:]:
                            if not row or not any(row):
                                continue
                            team_val = str(row[col_map["team"]]).strip() if "team" in col_map and col_map["team"] < len(row) and row[col_map["team"]] else ""
                            if not team_val or team_val.lower() in ["team", "average", "league"]:
                                continue

                            opp_val = str(row[col_map["opponent"]]).strip() if "opponent" in col_map and col_map["opponent"] < len(row) and row[col_map["opponent"]] else "OPP"
                            q_press = clean_num(row[col_map["quick_pressure"]]) if "quick_pressure" in col_map and col_map["quick_pressure"] < len(row) else 18.5
                            if q_press > 1.0:
                                q_press = q_press / 100.0

                            press_rt = clean_num(row[col_map["pressure_rate"]]) if "pressure_rate" in col_map and col_map["pressure_rate"] < len(row) else 32.0
                            if press_rt > 1.0:
                                press_rt = press_rt / 100.0

                            mof_p = clean_num(row[col_map["mof_pct"]]) if "mof_pct" in col_map and col_map["mof_pct"] < len(row) else 40.0
                            if mof_p > 1.0:
                                mof_p = mof_p / 100.0

                            mof_epa_val = clean_num(row[col_map["mof_epa"]]) if "mof_epa" in col_map and col_map["mof_epa"] < len(row) else 0.38
                            p_oe = clean_num(row[col_map["pass_oe"]]) if "pass_oe" in col_map and col_map["pass_oe"] < len(row) else 0.0
                            e_db = clean_num(row[col_map["epa_db"]]) if "epa_db" in col_map and col_map["epa_db"] < len(row) else 0.05
                            e_rush = clean_num(row[col_map["epa_rush"]]) if "epa_rush" in col_map and col_map["epa_rush"] < len(row) else -0.06

                            r_sr = clean_num(row[col_map["rush_sr"]]) if "rush_sr" in col_map and col_map["rush_sr"] < len(row) else 42.0
                            if r_sr > 1.0:
                                r_sr = r_sr / 100.0

                            team_records.append({
                                "week": detected_week,
                                "team": team_val.upper(),
                                "opponent": opp_val.upper(),
                                "pressure_rate": round(press_rt, 3),
                                "quick_pressure_rate": round(q_press, 3),
                                "mof_pct": round(mof_p, 3),
                                "mof_epa_allowed": round(mof_epa_val, 2),
                                "pass_oe": round(p_oe, 1),
                                "epa_db": round(e_db, 2),
                                "epa_rush": round(e_rush, 2),
                                "rush_sr_allowed": round(r_sr, 3)
                            })
                        p_col_map = {}
                        for idx, col_name in enumerate(header):
                            if "player" in col_name or "name" in col_name:
                                p_col_map["player"] = idx
                            elif "team" in col_name:
                                p_col_map["team"] = idx
                            elif "pos" in col_name:
                                p_col_map["position"] = idx
                            elif "tgt" in col_name or "target" in col_name:
                                p_col_map["target_pct"] = idx
                            elif "yprr" in col_name:
                                p_col_map["yprr"] = idx
                            elif "adot" in col_name:
                                p_col_map["adot"] = idx
                            elif "rush epa" in col_name:
                                p_col_map["rush_epa"] = idx
                            elif "rush sr" in col_name or "sr" in col_name:
                                p_col_map["rush_sr"] = idx

                        for row in table[1:]:
                            if not row or not any(row):
                                continue
                            p_name = str(row[p_col_map["player"]]).strip() if "player" in p_col_map and p_col_map["player"] < len(row) and row[p_col_map["player"]] else ""
                            if not p_name or p_name.lower() in ["player", "name", "total"]:
                                continue

                            p_team = str(row[p_col_map["team"]]).strip().upper() if "team" in p_col_map and p_col_map["team"] < len(row) and row[p_col_map["team"]] else "FA"
                            p_pos = str(row[p_col_map["position"]]).strip().upper() if "position" in p_col_map and p_col_map["position"] < len(row) and row[p_col_map["position"]] else "WR"

                            tgt_pct = clean_num(row[p_col_map["target_pct"]]) if "target_pct" in p_col_map and p_col_map["target_pct"] < len(row) else 15.0
                            if tgt_pct > 1.0:
                                tgt_pct = tgt_pct / 100.0

                            yprr = clean_num(row[p_col_map["yprr"]]) if "yprr" in p_col_map and p_col_map["yprr"] < len(row) else 1.65
                            adot = clean_num(row[p_col_map["adot"]]) if "adot" in p_col_map and p_col_map["adot"] < len(row) else 9.5
                            rush_epa = clean_num(row[p_col_map["rush_epa"]]) if "rush_epa" in p_col_map and p_col_map["rush_epa"] < len(row) else 0.0

                            p_rush_sr = clean_num(row[p_col_map["rush_sr"]]) if "rush_sr" in p_col_map and p_col_map["rush_sr"] < len(row) else 40.0
                            if p_rush_sr > 1.0:
                                p_rush_sr = p_rush_sr / 100.0

                            player_records.append({
                                "week": detected_week,
                                "player_name": p_name,
                                "team": p_team,
                                "position": p_pos,
                                "target_share": round(tgt_pct, 3),
                                "yprr": round(yprr, 2),
                                "adot": round(adot, 1),
                                "rush_epa": round(rush_epa, 2),
                                "rush_sr": round(p_rush_sr, 3)
                            })

        # Text Fallback Parsing if no tables extracted
        if not team_records and extracted_text_blocks:
            full_text = "\n".join(extracted_text_blocks)
            # Find lines formatted like: "BUF vs DET: Pressure 34%, QuickPress 22%, MOFEPA 0.35, RushSR 42%"
            pattern = re.compile(
                r"([A-Z]{2,3})\s+(?:vs|@)\s+([A-Z]{2,3})[^\n]*?Press(?:ure)?\s*[:=]?\s*(\d+\.?\d*)%?[^\n]*?Quick\s*[:=]?\s*(\d+\.?\d*)%?",
                re.IGNORECASE
            )
            for match in pattern.finditer(full_text):
                t1, t2, pr, qp = match.groups()
                team_records.append({
                    "week": detected_week,
                    "team": t1.upper(),
                    "opponent": t2.upper(),
                    "pressure_rate": round(float(pr) / 100.0 if float(pr) > 1.0 else float(pr), 3),
                    "quick_pressure_rate": round(float(qp) / 100.0 if float(qp) > 1.0 else float(qp), 3),
                    "mof_pct": 0.40,
                    "mof_epa_allowed": 0.38,
                    "pass_oe": 0.0,
                    "epa_db": 0.05,
                    "epa_rush": -0.05,
                    "rush_sr_allowed": 0.42
                })

        df_teams = pd.DataFrame(team_records) if team_records else pd.DataFrame(columns=[
            "week", "team", "opponent", "pressure_rate", "quick_pressure_rate",
            "mof_pct", "mof_epa_allowed", "pass_oe", "epa_db", "epa_rush", "rush_sr_allowed"
        ])
        df_players = pd.DataFrame(player_records) if player_records else pd.DataFrame(columns=[
            "week", "player_name", "team", "position", "target_share", "yprr", "adot", "rush_epa", "rush_sr"
        ])

        return {
            "success": True,
            "week": detected_week,
            "team_metrics": df_teams,
            "player_metrics": df_players,
            "raw_text_preview": "\n".join(extracted_text_blocks)[:1000] if extracted_text_blocks else ""
        }

    except Exception as e:
        return {
            "success": False,
            "week": default_week or 1,
            "team_metrics": pd.DataFrame(),
            "player_metrics": pd.DataFrame(),
            "raw_text_preview": "",
            "error": str(e)
        }


def create_sample_sumersports_pdf(output_path: str, week: int = 3) -> str:
    """
    Generate a compliant SumerSports 'Stats & Scheme' PDF review using reportlab.
    Useful for testing, offline demos, and verification suites.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    doc = SimpleDocTemplate(output_path, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#064e3b"),
        spaceAfter=6
    )
    elements.append(Paragraph(f"SumerSports NFL Stats & Scheme Review - Week {week}", title_style))

    sub_style = ParagraphStyle(
        "SubStyle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14
    )
    elements.append(Paragraph(f"Official Weekly Scheme Breakdown · Week {week} Review & Tendencies", sub_style))
    elements.append(Spacer(1, 10))

    # Section 1: Team Scheme Metrics Table
    elements.append(Paragraph("<b>Table 1: Team Game & Defensive Scheme Tendencies</b>", styles["Heading3"]))
    team_data = [
        ["Team", "Opp", "Pressure %", "Quick Press %", "MOF %", "MOF EPA", "Pass OE", "EPA/DB", "EPA/Rush", "Rush SR %"],
        ["CLE", "NYG", "38.2%", "24.5%", "39.0%", "0.28", "+4.2%", "+0.14", "-0.08", "38.0%"],
        ["PHI", "NO", "34.0%", "22.8%", "41.5%", "0.32", "-2.0%", "+0.08", "-0.04", "39.5%"],
        ["CAR", "LV", "21.5%", "13.0%", "45.0%", "0.58", "+6.5%", "-0.22", "-0.15", "48.5%"],
        ["TB", "DEN", "29.0%", "18.2%", "48.0%", "0.54", "+1.2%", "+0.02", "-0.06", "41.2%"],
        ["BUF", "JAX", "32.5%", "21.2%", "38.0%", "0.34", "+5.8%", "+0.28", "+0.02", "37.5%"],
        ["DET", "ARI", "33.0%", "20.5%", "40.0%", "0.38", "+3.0%", "+0.22", "+0.06", "36.2%"],
        ["NYG", "CLE", "30.0%", "18.5%", "43.0%", "0.47", "-1.5%", "-0.10", "-0.09", "47.5%"]
    ]
    t_team = Table(team_data, colWidths=[42, 38, 58, 68, 48, 55, 52, 52, 58, 58])
    t_team.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#064e3b")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
    ]))
    elements.append(t_team)
    elements.append(Spacer(1, 18))

    # Section 2: Player Usage Metrics Table
    elements.append(Paragraph("<b>Table 2: Key Player Usage & Efficiency Profiles</b>", styles["Heading3"]))
    player_data = [
        ["Player", "Team", "Pos", "Target %", "YPRR", "aDOT", "Rush EPA", "Rush SR %"],
        ["Amon-Ra St. Brown", "DET", "WR", "28.5%", "2.65", "8.2", "0.00", "0.0%"],
        ["Justin Jefferson", "MIN", "WR", "31.2%", "2.92", "11.8", "0.00", "0.0%"],
        ["James Cook", "BUF", "RB", "14.5%", "1.82", "1.5", "+0.12", "52.5%"],
        ["Jahmyr Gibbs", "DET", "RB", "18.0%", "2.10", "2.2", "+0.15", "54.0%"],
        ["Dalton Kincaid", "BUF", "TE", "22.0%", "2.05", "7.6", "0.00", "0.0%"],
        ["Brock Bowers", "LV", "TE", "24.5%", "2.35", "8.9", "0.00", "0.0%"],
        ["Zay Flowers", "BAL", "WR", "26.0%", "2.40", "9.1", "+0.04", "45.0%"],
        ["DeMario Douglas", "NE", "WR", "19.5%", "1.92", "6.4", "0.00", "0.0%"],
        ["Jalen Coker", "CAR", "WR", "16.0%", "1.85", "10.5", "0.00", "0.0%"]
    ]
    t_player = Table(player_data, colWidths=[110, 45, 40, 60, 55, 55, 60, 65])
    t_player.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
    ]))
    elements.append(t_player)

    doc.build(elements)
    return output_path
