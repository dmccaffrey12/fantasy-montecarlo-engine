"""
pdf_parser.py
Dual-Document PDF Parser Module for SumerSports "Stats & Scheme" Weekly Previews and Weekly Reviews.
Supports both:
  A) Real SumerSports Email Newsletter PDFs (`week1_preview.pdf` .. `week5_review.pdf` in `newsletters/`),
     combining coordinate-aware OCR (`winocr` + persistent `data/ocr_cache.json`) on embedded
     high-resolution Game Review Boxes, Game Preview Boxes, and Positional Leaderboards with
     NLP narrative text extraction (`The Sunday Stat Scroll`, `Questions We Want Answered`,
     `Three Takeaways`, `Film Favorites`, and `Under Center / Running the Ball Check-Ins`).
  B) Structured PDF table documents (`page.extract_tables()`) for synthetic/sample test PDFs.
"""

import hashlib
import io
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import pdfplumber

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OCR_CACHE_PATH = os.path.join(BASE_DIR, "data", "ocr_cache.json")

_OCR_CACHE_MEM: Optional[Dict[str, List[Dict[str, Any]]]] = None

TEAM_NORM: Dict[str, str] = {
    "HST": "HOU",
    "BLT": "BAL",
    "CLV": "CLE",
    "ARZ": "ARI",
    "LA": "LAR",
    "WSH": "WAS",
    "GNB": "GB",
    "KAN": "KC",
    "LVR": "LV",
    "NWE": "NE",
    "NOR": "NO",
    "SFO": "SF",
    "TAM": "TB",
    "CHl": "CHI",
    "CH1": "CHI",
    "PHl": "PHI",
    "PH1": "PHI",
    "MlA": "MIA",
    "M1A": "MIA",
    "MlN": "MIN",
    "M1N": "MIN",
    "ClN": "CIN",
    "C1N": "CIN",
}

VALID_NFL_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
}

FULL_TEAM_NAME_MAP: Dict[str, str] = {
    "arizona cardinals": "ARI", "cardinals": "ARI",
    "atlanta falcons": "ATL", "falcons": "ATL",
    "baltimore ravens": "BAL", "ravens": "BAL",
    "buffalo bills": "BUF", "bills": "BUF",
    "carolina panthers": "CAR", "panthers": "CAR",
    "chicago bears": "CHI", "bears": "CHI",
    "cincinnati bengals": "CIN", "bengals": "CIN",
    "cleveland browns": "CLE", "browns": "CLE",
    "dallas cowboys": "DAL", "cowboys": "DAL",
    "denver broncos": "DEN", "broncos": "DEN",
    "detroit lions": "DET", "lions": "DET",
    "green bay packers": "GB", "packers": "GB",
    "houston texans": "HOU", "texans": "HOU",
    "indianapolis colts": "IND", "colts": "IND",
    "jacksonville jaguars": "JAX", "jaguars": "JAX",
    "kansas city chiefs": "KC", "chiefs": "KC",
    "las vegas raiders": "LV", "raiders": "LV",
    "los angeles chargers": "LAC", "chargers": "LAC",
    "los angeles rams": "LAR", "rams": "LAR",
    "miami dolphins": "MIA", "dolphins": "MIA",
    "minnesota vikings": "MIN", "vikings": "MIN",
    "new england patriots": "NE", "patriots": "NE",
    "new orleans saints": "NO", "saints": "NO",
    "new york giants": "NYG", "giants": "NYG",
    "new york jets": "NYJ", "jets": "NYJ",
    "philadelphia eagles": "PHI", "eagles": "PHI",
    "pittsburgh steelers": "PIT", "steelers": "PIT",
    "san francisco 49ers": "SF", "49ers": "SF",
    "seattle seahawks": "SEA", "seahawks": "SEA",
    "tampa bay buccaneers": "TB", "buccaneers": "TB",
    "tennessee titans": "TEN", "titans": "TEN",
    "washington commanders": "WAS", "commanders": "WAS",
}

KNOWN_PLAYER_META: Dict[str, Tuple[str, str]] = {
    "christian watson": ("GB", "WR"),
    "demario douglas": ("NE", "WR"),
    "isaiah likely": ("NYG", "TE"),
    "mike evans": ("SF", "WR"),
    "sam laporta": ("DET", "TE"),
    "parker washington": ("JAX", "WR"),
    "matthew golden": ("GB", "WR"),
    "jameson williams": ("DET", "WR"),
    "trey mcbride": ("ARI", "TE"),
    "kyle williams": ("NE", "WR"),
    "jahmyr gibbs": ("DET", "RB"),
    "bucky irving": ("TB", "RB"),
    "jaylen warren": ("PIT", "RB"),
    "ollie gordon": ("MIA", "RB"),
    "christian mccaffrey": ("SF", "RB"),
    "d'andre swift": ("CHI", "RB"),
    "jacory croskey-merritt": ("WAS", "RB"),
    "braelon allen": ("NYJ", "RB"),
    "samaje perine": ("CIN", "RB"),
    "rhamondre stevenson": ("NE", "RB"),
    "drake london": ("ATL", "WR"),
    "tee higgins": ("CIN", "WR"),
    "george kittle": ("SF", "TE"),
    "ceedee lamb": ("DAL", "WR"),
    "ja'marr chase": ("CIN", "WR"),
    "brock bowers": ("LV", "TE"),
    "chris olave": ("NO", "WR"),
    "harold fannin": ("CLE", "TE"),
    "garrett wilson": ("NYJ", "WR"),
    "bijan robinson": ("ATL", "RB"),
    "brian robinson": ("WAS", "RB"),
    "bhayshul tuten": ("JAX", "RB"),
    "james cook": ("BUF", "RB"),
    "justice hill": ("BAL", "RB"),
    "travis etienne": ("JAX", "RB"),
    "jaxon smith-njigba": ("SEA", "WR"),
    "dontayvion wicks": ("PHI", "WR"),
    "amon-ra st. brown": ("DET", "WR"),
    "jaylen waddle": ("DEN", "WR"),
    "devaughn vele": ("DEN", "WR"),
    "dalton schultz": ("HOU", "TE"),
    "derrick henry": ("BAL", "RB"),
    "rachaad white": ("TB", "RB"),
    "tyjae spears": ("TEN", "RB"),
    "tony pollard": ("TEN", "RB"),
    "aaron jones": ("MIN", "RB"),
    "justin jefferson": ("MIN", "WR"),
    "jordan addison": ("MIN", "WR"),
    "t.j. hockenson": ("MIN", "TE"),
    "zay flowers": ("BAL", "WR"),
    "mark andrews": ("BAL", "TE"),
    "rashod bateman": ("BAL", "WR"),
    "dalton kincaid": ("BUF", "TE"),
    "khalil shakir": ("BUF", "WR"),
    "d.j. moore": ("BUF", "WR"),
    "nico collins": ("HOU", "WR"),
    "david montgomery": ("HOU", "RB"),
    "deebo samuel": ("SF", "WR"),
    "malik nabers": ("NYG", "WR"),
    "jalen coker": ("CAR", "WR"),
    "tetairoa mcmillan": ("CAR", "WR"),
    "brenton strange": ("JAX", "TE"),
    "kenneth walker": ("KC", "RB"),
    "kalif raymond": ("CHI", "WR"),
    "tre tucker": ("LV", "WR"),
    "ladd mcconkey": ("LAC", "WR"),
}


def norm_team(t: str) -> str:
    """Normalize SumerSports / OCR team abbreviations to standard 2/3-letter NFL codes."""
    if not t:
        return "NFL"
    t_s = t.strip()
    if t_s in TEAM_NORM:
        return TEAM_NORM[t_s]
    tu = t_s.upper()
    return TEAM_NORM.get(tu, tu)


def clean_num(val: Any, default: float = 0.0) -> float:
    """Helper to sanitize percentage strings, OCR artifacts, and floats."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val)
    s = re.sub(r"[0Oo]/[0Oo](?:i[lI]e)?", "%", s)
    s = re.sub(r"\(?#\d+\)?", "", s)
    s = re.sub(r"[@#]", "", s)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except (ValueError, TypeError):
            return default
    return default


def _load_ocr_cache() -> Dict[str, List[Dict[str, Any]]]:
    global _OCR_CACHE_MEM
    if _OCR_CACHE_MEM is not None:
        return _OCR_CACHE_MEM
    if os.path.exists(OCR_CACHE_PATH):
        try:
            with open(OCR_CACHE_PATH, "r", encoding="utf-8") as f:
                _OCR_CACHE_MEM = json.load(f)
                return _OCR_CACHE_MEM
        except Exception:
            pass
    _OCR_CACHE_MEM = {}
    return _OCR_CACHE_MEM


def _save_ocr_cache() -> None:
    global _OCR_CACHE_MEM
    if _OCR_CACHE_MEM is None:
        return
    try:
        os.makedirs(os.path.dirname(OCR_CACHE_PATH), exist_ok=True)
        with open(OCR_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_OCR_CACHE_MEM, f)
    except Exception:
        pass


def _get_ocr_items_for_image(img_info: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Tuple[int, int]]:
    """
    Retrieve normalized OCR lines `[{'nx': float, 'ny': float, 'text': str}, ...]`
    for a PDF image stream, using `data/ocr_cache.json` first and falling back to
    live Windows native OCR (`winocr`) if the image hash is new.
    """
    w, h = img_info.get("srcsize", (0, 0))
    if w < 1400 or h < 1400:
        return None, (w, h)
    try:
        raw_bytes = img_info["stream"].get_data()
    except Exception:
        return None, (w, h)

    hsh = hashlib.md5(raw_bytes).hexdigest()
    cache = _load_ocr_cache()
    if hsh in cache:
        return cache[hsh], (w, h)

    # Live OCR fallback via winocr + PIL
    try:
        from PIL import Image
        import winocr

        try:
            pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception:
            if w * h * 3 == len(raw_bytes):
                pil = Image.frombytes("RGB", (w, h), raw_bytes)
            elif w * h * 4 == len(raw_bytes):
                pil = Image.frombytes("RGBA", (w, h), raw_bytes).convert("RGB")
            else:
                return None, (w, h)

        if pil.width > 2400:
            scale = 2400.0 / pil.width
            pil = pil.resize((2400, int(pil.height * scale)), Image.Resampling.BILINEAR)

        res = winocr.recognize_pil_sync(pil, "en")
        pw, ph = pil.size
        items = []
        for ln in res.get("lines", []):
            words = ln.get("words", [])
            if not words:
                continue
            x0 = words[0]["bounding_rect"]["x"]
            y0 = words[0]["bounding_rect"]["y"]
            items.append({
                "nx": round(x0 / pw, 4),
                "ny": round(y0 / ph, 4),
                "text": ln.get("text", "").strip()
            })
        cache[hsh] = items
        _save_ocr_cache()
        return items, (w, h)
    except Exception:
        return None, (w, h)


def _open_pdf(pdf_source: Union[str, io.BytesIO, Any]):
    """Safely open a PDF from file path, bytes, BytesIO, or Streamlit UploadedFile."""
    if isinstance(pdf_source, (bytes, bytearray)):
        return pdfplumber.open(io.BytesIO(pdf_source))
    if hasattr(pdf_source, "seek"):
        try:
            pdf_source.seek(0)
        except Exception:
            pass
    return pdfplumber.open(pdf_source)


def detect_newsletter_type(
    pdf_source: Union[str, io.BytesIO, Any],
    default_week: Optional[int] = None
) -> Dict[str, Any]:
    """
    Determine whether a SumerSports PDF document is a "Weekly Preview" or "Weekly Review"
    based on filename hints, title headers, text markers, image aspect ratios, and table schemas.
    """
    detected_week = default_week or 1
    preview_score = 0
    review_score = 0
    signals: List[str] = []

    fname = ""
    if isinstance(pdf_source, str):
        fname = os.path.basename(pdf_source).lower()
    elif hasattr(pdf_source, "name"):
        fname = str(getattr(pdf_source, "name", "")).lower()

    has_file_week = False
    if fname:
        wk_file = re.search(r"week[_\-\s]*(\d+)", fname, re.IGNORECASE)
        if wk_file and not default_week:
            detected_week = int(wk_file.group(1))
            has_file_week = True
        if "preview" in fname:
            preview_score += 15
            signals.append(f"Filename contains 'preview' ({fname})")
        if "review" in fname:
            review_score += 15
            signals.append(f"Filename contains 'review' ({fname})")

    try:
        with _open_pdf(pdf_source) as pdf:
            for p_idx, page in enumerate(pdf.pages[:10]):
                text = page.extract_text() or ""
                text_lower = text.lower()

                if not default_week and not has_file_week and p_idx == 0:
                    wk_match = re.search(r"Week\s*(\d+)", text, re.IGNORECASE)
                    if wk_match:
                        detected_week = int(wk_match.group(1))

                if p_idx <= 2:
                    if any(k in text_lower for k in ["weekly preview", "matchup preview", "scheme preview", "questions we want answered"]):
                        preview_score += 5
                        signals.append("Header/Text contains Weekly Preview markers")
                    if any(k in text_lower for k in ["weekly review", "scheme review", "game review", "sunday stat scroll"]):
                        review_score += 5
                        signals.append("Header/Text contains Weekly Review markers")

                for img in getattr(page, "images", []):
                    w, h = img.get("srcsize", (0, 0))
                    if w >= 1500 and h >= 1500:
                        ratio = h / float(w)
                        if 1.40 <= ratio <= 1.75:
                            preview_score += 3
                            signals.append("Embedded image matches Preview Box aspect ratio")
                        elif 0.95 <= ratio <= 1.18:
                            review_score += 3
                            signals.append("Embedded image matches Review Box aspect ratio")

                tables = page.extract_tables() or []
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    header = " ".join([str(c).strip().lower() if c else "" for c in table[0]])
                    if any(k in header for k in ["cover 1/3", "man %", "zone %", "funnel", "explosive"]):
                        preview_score += 4
                        signals.append("Table schema matches Matchup Preview")
                    if any(k in header for k in ["pass oe", "epa/db", "quick press", "yprr", "adot"]):
                        review_score += 4
                        signals.append("Table schema matches Weekly Review")
    except Exception as e:
        signals.append(f"Fallback detection ({e})")

    is_preview = preview_score > review_score
    doc_type = "Weekly Preview" if is_preview else "Weekly Review"
    short_type = "PREVIEW" if is_preview else "REVIEW"
    total_pts = max(1, preview_score + review_score)
    conf = round(max(preview_score, review_score) / total_pts, 2)

    return {
        "doc_type": doc_type,
        "short_type": short_type,
        "week": detected_week,
        "confidence": conf,
        "signals": signals,
    }


def _parse_review_box_ocr(
    items: List[Dict[str, Any]],
    week_num: int,
    matchup_takeaways: Dict[str, str],
    week_takeaways: List[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract both team panels and 10 player rows from a SumerSports Game Review Box OCR item list."""
    left_pair = None
    right_pair = None
    for it in items:
        if it["ny"] < 0.06:
            m = re.search(r"([A-Za-z0-9]{2,3})\s+Offense\s+vs\.?\s+([A-Za-z0-9]{2,3})\s+Defense", it["text"], re.IGNORECASE)
            if m:
                t1, t2 = norm_team(m.group(1)), norm_team(m.group(2))
                if it["nx"] < 0.45:
                    left_pair = (t1, t2)
                else:
                    right_pair = (t1, t2)

    if not left_pair or not right_pair:
        left_off, right_off = None, None
        for it in items:
            m_o = re.match(r"^([A-Za-z0-9]{2,3})\s+Offense$", it["text"], re.IGNORECASE)
            if m_o:
                t_c = norm_team(m_o.group(1))
                if t_c in VALID_NFL_TEAMS:
                    if it["nx"] < 0.45 and not left_off:
                        left_off = t_c
                    elif it["nx"] >= 0.45 and not right_off:
                        right_off = t_c
        if left_off and right_off:
            left_pair = (left_off, right_off)
            right_pair = (right_off, left_off)
        elif left_pair and not right_pair:
            right_pair = (left_pair[1], left_pair[0])
        elif right_pair and not left_pair:
            left_pair = (right_pair[1], right_pair[0])

    if not left_pair or not right_pair:
        return [], []

    panels = [(0, left_pair[0], left_pair[1]), (1, right_pair[0], right_pair[1])]
    panel_metrics: Dict[str, Dict[str, Any]] = {}
    player_rows: List[Dict[str, Any]] = []

    for p_idx, off_t, def_t in panels:
        x_min = 0.0 if p_idx == 0 else 0.49
        x_max = 0.49 if p_idx == 0 else 1.0
        x_offset = 0.0 if p_idx == 0 else 0.495

        p_items = [it for it in items if x_min <= it["nx"] < x_max]

        def get_cell(ny_low: float, ny_high: float, col_idx: int) -> Optional[float]:
            col_ranges = [
                (0.11, 0.205),
                (0.205, 0.300),
                (0.300, 0.395),
                (0.395, 0.500),
            ]
            c_low, c_high = col_ranges[col_idx]
            candidates = [
                it for it in p_items
                if ny_low <= it["ny"] <= ny_high
                and c_low <= (it["nx"] - x_offset) < c_high
                and not any(s in it["text"].lower() for s in ["ile", "i1e", "epa"])
                and not re.search(r"\d+\s*(?:st|nd|rd|th)", it["text"].lower())
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda z: z["ny"])
            return clean_num(candidates[0]["text"], None)

        epa_play = get_cell(0.105, 0.148, 0)
        pass_oe = get_cell(0.105, 0.148, 2)
        epa_db = get_cell(0.325, 0.368, 0)
        epa_rush = get_cell(0.325, 0.368, 2)
        rush_sr = get_cell(0.325, 0.368, 3)
        pres = get_cell(0.435, 0.478, 0)
        q_pres = get_cell(0.435, 0.478, 1)
        mof_pct = get_cell(0.545, 0.588, 0)
        mof_epa = get_cell(0.545, 0.588, 1)

        raw_pres = (pres / 100.0 if pres and pres > 1.0 else (pres or 0.32))
        if not (0.10 <= raw_pres <= 0.60):
            raw_pres = 0.32
        raw_qpres = (q_pres / 100.0 if q_pres and q_pres > 1.0 else (q_pres or round(raw_pres * 0.52, 3)))
        if not (0.05 <= raw_qpres <= 0.32) or raw_qpres > raw_pres:
            raw_qpres = round(raw_pres * 0.52, 3)
        raw_rsr = (rush_sr / 100.0 if rush_sr and rush_sr > 1.0 else (rush_sr or 0.42))
        if not (0.15 <= raw_rsr <= 0.70):
            raw_rsr = 0.42

        panel_metrics[off_t] = {
            "week": week_num,
            "team": off_t,
            "opponent": def_t,
            "epa_play": epa_play if epa_play is not None else 0.0,
            "pass_oe": pass_oe if pass_oe is not None and abs(pass_oe) <= 45.0 else 0.0,
            "epa_db": epa_db if epa_db is not None and abs(epa_db) <= 3.0 else 0.0,
            "epa_rush": epa_rush if epa_rush is not None and abs(epa_rush) <= 3.0 else 0.0,
            "off_rush_sr": raw_rsr,
            "off_pressure_rate_allowed": raw_pres,
            "off_quick_pressure_allowed": raw_qpres,
            "off_mof_pct": (mof_pct / 100.0 if mof_pct and mof_pct > 1.0 else (mof_pct or 0.40)),
            "off_mof_epa": mof_epa if mof_epa is not None and abs(mof_epa) <= 3.0 else 0.35,
        }

        # Receiving / Pass Catching players
        rec_names = [
            it for it in p_items
            if 0.770 <= it["ny"] <= 0.865
            and (it["nx"] - x_offset) < 0.17
            and not any(h in it["text"].upper() for h in ["PLAYER", "RECEIVING", "CATCHING", "DESIGNED"])
            and len(it["text"]) >= 3
        ]
        rec_names.sort(key=lambda z: z["ny"])
        for rn in rec_names:
            ry = rn["ny"]
            row_nums = [
                it for it in p_items
                if abs(it["ny"] - ry) <= 0.015 and (it["nx"] - x_offset) >= 0.17
            ]
            tgt_val, yprr_val, adot_val = 0.18, 1.8, 8.5
            for r_it in row_nums:
                rx = r_it["nx"] - x_offset
                v = clean_num(r_it["text"], None)
                if v is None:
                    continue
                if 0.17 <= rx < 0.26:
                    tgt_val = v / 100.0 if v > 1.0 else v
                elif 0.26 <= rx < 0.33:
                    yprr_val = v if v <= 15.0 else v / 100.0
                elif 0.40 <= rx < 0.50:
                    adot_val = v if v <= 40.0 else v / 100.0

            p_clean_name = rn["text"].replace("McCaffrev", "McCaffrey").strip()
            known_pos = KNOWN_PLAYER_META.get(p_clean_name.lower(), (off_t, "WR"))[1]
            player_rows.append({
                "week": week_num,
                "player_name": p_clean_name,
                "team": off_t,
                "position": known_pos,
                "target_share": round(max(0.01, min(0.65, tgt_val)), 3),
                "yprr": round(max(0.0, min(12.0, yprr_val)), 2),
                "adot": round(max(-5.0, min(35.0, adot_val)), 1),
                "rush_epa": 0.0,
                "rush_sr": 0.0,
            })

        # Designed Rushing Stats players
        rush_names = [
            it for it in p_items
            if 0.920 <= it["ny"] <= 0.988
            and (it["nx"] - x_offset) < 0.17
            and not any(h in it["text"].upper() for h in ["PLAYER", "RUSHING", "DESIGNED"])
            and len(it["text"]) >= 3
        ]
        rush_names.sort(key=lambda z: z["ny"])
        for rhn in rush_names:
            ry = rhn["ny"]
            row_nums = [
                it for it in p_items
                if abs(it["ny"] - ry) <= 0.015 and (it["nx"] - x_offset) >= 0.17
            ]
            repa_val, rsr_val = -0.02, 0.40
            for r_it in row_nums:
                rx = r_it["nx"] - x_offset
                v = clean_num(r_it["text"], None)
                if v is None:
                    continue
                if 0.33 <= rx < 0.405:
                    repa_val = v if abs(v) <= 5.0 else -0.02
                elif 0.405 <= rx < 0.50:
                    rsr_val = v / 100.0 if v > 1.0 else v
            p_clean_name = rhn["text"].replace("McCaffrev", "McCaffrey").strip()
            player_rows.append({
                "week": week_num,
                "player_name": p_clean_name,
                "team": off_t,
                "position": "RB",
                "target_share": 0.08,
                "yprr": 1.10,
                "adot": 1.5,
                "rush_epa": round(repa_val, 2),
                "rush_sr": round(max(0.0, min(1.0, rsr_val)), 3),
            })

    team_rows: List[Dict[str, Any]] = []
    wk_tk = "; ".join(week_takeaways[:3]) if week_takeaways else f"Week {week_num} SumerSports Stats & Scheme Review"
    for off_t, m in panel_metrics.items():
        opp_t = m["opponent"]
        opp_m = panel_metrics.get(opp_t, m)
        # Combine offensive quick pressure allowed and defensive quick pressure generated
        comb_qp = round(0.5 * m["off_quick_pressure_allowed"] + 0.5 * opp_m["off_quick_pressure_allowed"], 3)
        comb_pr = round(0.5 * m["off_pressure_rate_allowed"] + 0.5 * opp_m["off_pressure_rate_allowed"], 3)
        mof_allowed = round(opp_m["off_mof_epa"], 2)
        rsr_allowed = round(opp_m["off_rush_sr"], 3)

        tk_text = matchup_takeaways.get(off_t) or matchup_takeaways.get(opp_t) or (
            f"{off_t} vs {opp_t}: Pass OE {m['pass_oe']:+.1f}%, EPA/DB {m['epa_db']:+.2f}, "
            f"Off Press Allowed {m['off_pressure_rate_allowed']*100:.1f}% (QP {m['off_quick_pressure_allowed']*100:.1f}%), "
            f"Def MOF EPA Allowed {mof_allowed:+.2f}, Rush SR Allowed {rsr_allowed*100:.1f}%."
        )

        team_rows.append({
            "week": week_num,
            "team": off_t,
            "opponent": opp_t,
            "pressure_rate": comb_pr,
            "quick_pressure_rate": comb_qp,
            "mof_pct": round(opp_m["off_mof_pct"], 3),
            "mof_epa_allowed": mof_allowed,
            "pass_oe": round(m["pass_oe"], 1),
            "epa_db": round(m["epa_db"], 2),
            "epa_rush": round(m["epa_rush"], 2),
            "rush_sr_allowed": rsr_allowed,
            "matchup_takeaway": tk_text,
            "week_takeaway": wk_tk,
        })

    return team_rows, player_rows


def _parse_leaderboard_box_ocr(items: List[Dict[str, Any]], week_num: int) -> List[Dict[str, Any]]:
    """Extract player rows from SumerSports Receiving or Rushing Leaderboard images."""
    header_text = " ".join([it["text"].lower() for it in items if it["ny"] < 0.16])
    is_rec = any(k in header_text for k in ["receiving", "pass catching"])
    is_rush = "rushing" in header_text
    if not is_rec and not is_rush:
        return []

    # Collect player names at nx in [0.05, 0.25], ny >= 0.22
    name_items = [
        it for it in items
        if 0.22 <= it["ny"] <= 0.96
        and 0.05 <= it["nx"] <= 0.24
        and not any(k in it["text"].upper() for k in ["PASS CATCHER", "RUSHER", "QUARTERBACK", "RANK", "SNF"])
    ]
    name_items.sort(key=lambda z: z["ny"])

    # Merge two-line wrapped player names (e.g. 'Amon-Ra St.' + 'Brown', 'Christian' + 'McCaffrey')
    merged_names: List[Dict[str, Any]] = []
    for it in name_items:
        txt = re.sub(r"^\d+\s+", "", it["text"]).strip()
        txt = txt.replace("McCaffrev", "McCaffrey").replace("Croskev-Merritt", "Croskey-Merritt")
        if not txt or len(txt) < 2:
            continue
        if merged_names and (it["ny"] - merged_names[-1]["ny"]) < 0.048:
            merged_names[-1]["text"] = f"{merged_names[-1]['text']} {txt}"
            merged_names[-1]["ny"] = 0.5 * (merged_names[-1]["ny"] + it["ny"])
        else:
            merged_names.append({"ny": it["ny"], "nx": it["nx"], "text": txt})

    rows: List[Dict[str, Any]] = []
    for mn in merged_names:
        p_name = mn["text"].strip()
        ry = mn["ny"]
        team, pos = KNOWN_PLAYER_META.get(p_name.lower(), ("NFL", "WR" if is_rec else "RB"))

        row_nums = [
            it for it in items
            if abs(it["ny"] - ry) <= 0.028 and it["nx"] >= 0.35
        ]
        row_nums.sort(key=lambda z: z["nx"])
        vals = [clean_num(r["text"], None) for r in row_nums]
        vals = [v for v in vals if v is not None]

        if is_rec:
            # Columns: ROUTES, TGT SHARE, REC, YPRR, TOTAL REC EPA
            tgt_share = 0.22
            yprr = 2.40
            for r in row_nums:
                v = clean_num(r["text"], None)
                if v is None:
                    continue
                if "%" in r["text"] or "0/0" in r["text"] or "O/o" in r["text"] or (0.45 <= r["nx"] <= 0.62 and v > 5.0):
                    tgt_share = v / 100.0 if v > 1.0 else v
                elif 0.70 <= r["nx"] <= 0.88 and 0.5 <= v <= 12.0:
                    yprr = v
            rows.append({
                "week": week_num,
                "player_name": p_name,
                "team": team,
                "position": pos,
                "target_share": round(max(0.05, min(0.60, tgt_share)), 3),
                "yprr": round(yprr, 2),
                "adot": 9.2,
                "rush_epa": 0.0,
                "rush_sr": 0.0,
            })
        elif is_rush:
            # Columns: ATT, YPC, SUCCESS %, MTF, RUSH EPA+
            rush_sr = 0.48
            rush_epa = 0.15
            for r in row_nums:
                v = clean_num(r["text"], None)
                if v is None:
                    continue
                if "%" in r["text"] or "0/0" in r["text"] or (0.55 <= r["nx"] <= 0.75 and v > 15.0):
                    rush_sr = v / 100.0 if v > 1.0 else v
            rows.append({
                "week": week_num,
                "player_name": p_name,
                "team": team,
                "position": "RB",
                "target_share": 0.10,
                "yprr": 1.25,
                "adot": 1.5,
                "rush_epa": rush_epa,
                "rush_sr": round(max(0.10, min(0.95, rush_sr)), 3),
            })

    return rows


def _parse_preview_box_ocr(
    items: List[Dict[str, Any]],
    week_num: int,
    matchup_notes: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Extract primary and mirror team preview records from a SumerSports Weekly Preview Box OCR item list."""
    off_t, def_t = None, None
    for it in items:
        if it["ny"] < 0.08:
            m = re.search(r"([A-Za-z0-9]{2,3})\s+Offense\s+vs\.?\s+([A-Za-z0-9]{2,3})\s+Defense", it["text"], re.IGNORECASE)
            if m:
                off_t = norm_team(m.group(1))
                def_t = norm_team(m.group(2))
                break
    if not off_t or not def_t:
        for it in items:
            if it["nx"] < 0.16:
                m_o = re.match(r"^([A-Za-z0-9]{2,3})\s+Offense", it["text"], re.IGNORECASE)
                m_d = re.match(r"^([A-Za-z0-9]{2,3})\s+Defense", it["text"], re.IGNORECASE)
                if m_o and not off_t:
                    off_t = norm_team(m_o.group(1))
                if m_d and not def_t:
                    def_t = norm_team(m_d.group(1))
    if not off_t or not def_t or off_t not in VALID_NFL_TEAMS or def_t not in VALID_NFL_TEAMS:
        return []

    def find_section_rows(keywords: List[str], default_off_y: float, default_def_y: float) -> Tuple[float, float]:
        sec_y = None
        for it in items:
            if it["nx"] < 0.20 and any(k in it["text"].lower() for k in keywords):
                sec_y = it["ny"]
                break
        if sec_y is None:
            return default_off_y, default_def_y
        y_off, y_def = None, None
        for it in items:
            if it["nx"] < 0.16 and sec_y + 0.025 <= it["ny"] <= sec_y + 0.16:
                if "offense" in it["text"].lower() and y_off is None:
                    y_off = it["ny"]
                elif "defense" in it["text"].lower() and y_def is None:
                    y_def = it["ny"]
        return (y_off or default_off_y), (y_def or default_def_y)

    def get_val_at_row(row_y: float, col_idx: int, total_cols: int = 6) -> Optional[float]:
        if total_cols == 6:
            col_ranges = [
                (0.14, 0.27),
                (0.27, 0.40),
                (0.40, 0.54),
                (0.54, 0.69),
                (0.69, 0.83),
                (0.83, 0.99),
            ]
        else:
            col_ranges = [
                (0.14, 0.29),
                (0.29, 0.45),
                (0.45, 0.61),
                (0.61, 0.78),
                (0.78, 0.99),
            ]
        c_low, c_high = col_ranges[col_idx]
        cands = [
            it for it in items
            if abs(it["ny"] - row_y) <= 0.018 and c_low <= it["nx"] < c_high
        ]
        if not cands:
            return None
        cands.sort(key=lambda z: abs(z["ny"] - row_y))
        return clean_num(cands[0]["text"], None)

    y_ovr_off, y_ovr_def = find_section_rows(["overall"], 0.140, 0.172)
    y_shl_off, y_shl_def = find_section_rows(["safety shell", "middle of field"], 0.530, 0.562)
    y_blz_off, y_blz_def = find_section_rows(["blitz", "pressure"], 0.661, 0.693)
    y_dep_off, y_dep_def = find_section_rows(["pass depth"], 0.791, 0.823)

    def_expl = get_val_at_row(y_ovr_def, 5, 6)
    off_expl = get_val_at_row(y_ovr_off, 5, 6)

    def_1high = get_val_at_row(y_shl_def, 0, 6)
    def_2high = get_val_at_row(y_shl_def, 2, 6)
    def_mof_epa = get_val_at_row(y_shl_def, 5, 6)

    off_1high = get_val_at_row(y_shl_off, 0, 6)
    off_2high = get_val_at_row(y_shl_off, 2, 6)
    off_mof_epa = get_val_at_row(y_shl_off, 5, 6)

    off_blitz = get_val_at_row(y_blz_off, 0, 5)
    off_press = get_val_at_row(y_blz_off, 3, 5)
    def_blitz = get_val_at_row(y_blz_def, 0, 5)
    def_press = get_val_at_row(y_blz_def, 3, 5)

    def_inter_epa = get_val_at_row(y_dep_def, 3, 6)
    def_deep_epa = get_val_at_row(y_dep_def, 5, 6)

    def build_rec(t_name: str, o_name: str, m_1h, m_2h, blz, prs_off, prs_def, mof_e, expl, inter_e, deep_e):
        man_rt = (m_1h / 100.0) if (m_1h is not None and 10.0 <= m_1h <= 90.0) else 0.45
        zone_rt = (m_2h / 100.0) if (m_2h is not None and 10.0 <= m_2h <= 90.0) else round(1.0 - man_rt, 3)
        blitz_rt = (blz / 100.0) if (blz is not None and 5.0 <= blz <= 65.0) else 0.25
        p_def = (prs_def / 100.0) if (prs_def is not None and 14.0 <= prs_def <= 58.0) else 0.33
        p_off = (prs_off / 100.0) if (prs_off is not None and 14.0 <= prs_off <= 58.0) else p_def
        exp_qpr = round(0.55 * (0.5 * p_off + 0.5 * p_def), 3)
        exp_mof = mof_e if mof_e is not None and abs(mof_e) <= 1.5 else 0.15
        expl_rt = (expl / 100.0) if (expl is not None and 3.0 <= expl <= 28.0) else 0.11

        if zone_rt >= 0.52 or exp_mof >= 0.15 or (inter_e is not None and inter_e >= 0.25):
            funnel = "Inside/Slot"
        elif man_rt >= 0.54 or (deep_e is not None and deep_e >= 0.35):
            funnel = "Boundary"
        else:
            funnel = "Balanced"

        narr_note = matchup_notes.get(t_name) or matchup_notes.get(o_name) or ""
        base_tk = (
            f"{t_name} vs {o_name}: 1-High {man_rt*100:.1f}% / 2-High {zone_rt*100:.1f}% | "
            f"Blitz {blitz_rt*100:.1f}% | Press {p_def*100:.1f}% | Exp MOF EPA {exp_mof:+.2f}"
        )
        if narr_note:
            base_tk = f"{narr_note} ({base_tk})"

        return {
            "week": week_num,
            "team": t_name,
            "opponent": o_name,
            "man_coverage_rate": round(man_rt, 3),
            "zone_coverage_rate": round(zone_rt, 3),
            "blitz_rate": round(blitz_rt, 3),
            "expected_quick_pressure_rate": exp_qpr,
            "expected_mof_epa": round(exp_mof, 2),
            "explosive_play_rate_allowed": round(expl_rt, 3),
            "funnel_focus": funnel,
            "tactical_takeaways": base_tk[:320],
        }

    return [
        build_rec(off_t, def_t, def_1high, def_2high, def_blitz, off_press, def_press, def_mof_epa, def_expl, def_inter_epa, def_deep_epa),
        build_rec(def_t, off_t, off_1high, off_2high, off_blitz, def_press, off_press, off_mof_epa, off_expl, None, None),
    ]


def _extract_narrative_matchups_and_takeaways(full_text: str) -> Tuple[Dict[str, str], List[str], List[Dict[str, Any]]]:
    """
    Parse SumerSports narrative text (`The Sunday Stat Scroll`, `Three Takeaways`,
    `Questions We Want Answered`, `Film Favorites`) to extract:
      - `matchup_takeaways`: Dict[team_abbr, takeaway_text]
      - `week_takeaways`: List[str]
      - `inline_player_ angles`: List[Dict]
    """
    matchup_takeaways: Dict[str, str] = {}
    week_takeaways: List[str] = []
    player_angles: List[Dict[str, Any]] = []

    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    for idx, ln in enumerate(lines):
        # Match game headers like:
        # 'Buffalo Bills 36, Houston Texans 31: Josh Allen starts off flaming hot.'
        # or 'Detroit Lions @ Buffalo Bills: How will the Lions base defense look...'
        if ":" in ln and len(ln) < 160:
            left_part, right_part = ln.split(":", 1)
            found_teams = []
            lp_lower = left_part.lower()
            for full_name, abbr in FULL_TEAM_NAME_MAP.items():
                if len(full_name) > 6 and full_name in lp_lower and abbr not in found_teams:
                    found_teams.append(abbr)
            if len(found_teams) >= 2:
                snippet = " ".join(lines[idx:min(len(lines), idx + 4)])
                for t_ab in found_teams[:2]:
                    matchup_takeaways[t_ab] = snippet[:300]

        # Capture overarching bullet takeaways
        if any(k in ln.lower() for k in ["epa/play", "pressure rate", "under center", "13 personnel", "success rate", "target share", "blitz"]):
            if 40 <= len(ln) <= 220 and len(week_takeaways) < 12:
                week_takeaways.append(ln)

        # Check for known players mentioned in narrative lines
        ln_lower = ln.lower()
        for p_name_lower, (p_team, p_pos) in KNOWN_PLAYER_META.items():
            if p_name_lower in ln_lower:
                context_str = " ".join(lines[max(0, idx - 1):min(len(lines), idx + 2)])
                align_adv = "Slot / Interior" if any(w in context_str.lower() for w in ["slot", "middle", "short", "inline"]) else (
                    "Boundary / Deep" if any(w in context_str.lower() for w in ["outside", "wide", "deep", "20+"]) else "Scheme Featured"
                )
                pers_note = "12/13 Heavy Personnel" if any(w in context_str.lower() for w in ["12 personnel", "13 personnel", "under center", "heavy"]) else "11 Personnel Primary"
                player_angles.append({
                    "player_name": p_name_lower.title().replace("Mccaffrey", "McCaffrey").replace("Mcbride", "McBride").replace("Mcconkey", "McConkey").replace("Mcmillan", "McMillan").replace("St. Brown", "St. Brown").replace("T.J.", "T.J.").replace("D.J.", "D.J."),
                    "team": p_team,
                    "position": p_pos,
                    "alignment_advantage": align_adv,
                    "personnel_note": pers_note,
                    "matchup_angle": context_str[:220],
                })

    return matchup_takeaways, week_takeaways, player_angles


def parse_review_pdf(
    pdf_source: Union[str, io.BytesIO, Any],
    week_num: Optional[int] = None
) -> Dict[str, Any]:
    """
    Parse a SumerSports "Weekly Review" PDF:
      1. Extracts embedded Game Review Boxes (`0.95 <= h/w <= 1.18`) and Positional Leaderboards
         (`0.60 <= h/w < 0.95`) via `winocr` + `data/ocr_cache.json`.
      2. Extracts PDF tables (`page.extract_tables()`) when present (e.g. `sample_sumersports_week3.pdf`).
      3. Extracts narrative matchup takeaways and overarching week takeaways from text.
    """
    team_records_by_team: Dict[str, Dict[str, Any]] = {}
    player_records_by_name: Dict[str, Dict[str, Any]] = {}
    extracted_text_blocks: List[str] = []
    detected_week = week_num or 1

    try:
        with _open_pdf(pdf_source) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    extracted_text_blocks.append(text)
                if not week_num:
                    wk_m = re.search(r"Week\s*(\d+)", text, re.IGNORECASE)
                    if wk_m:
                        detected_week = int(wk_m.group(1))

            full_text = "\n".join(extracted_text_blocks)
            matchup_takeaways, week_takeaways, _ = _extract_narrative_matchups_and_takeaways(full_text)

            # 1. Process Embedded Images (Game Review Boxes & Leaderboards)
            seen_hashes = set()
            for page in pdf.pages:
                for img in getattr(page, "images", []):
                    w, h = img.get("srcsize", (0, 0))
                    if w < 1400 or h < 1400:
                        continue
                    try:
                        hsh = hashlib.md5(img["stream"].get_data()).hexdigest()
                    except Exception:
                        continue
                    if hsh in seen_hashes:
                        continue
                    seen_hashes.add(hsh)

                    items, _ = _get_ocr_items_for_image(img)
                    if not items:
                        continue
                    ratio = h / float(w)
                    if 0.95 <= ratio <= 1.18:
                        t_rows, p_rows = _parse_review_box_ocr(items, detected_week, matchup_takeaways, week_takeaways)
                        for tr in t_rows:
                            team_records_by_team[tr["team"]] = tr
                        for pr in p_rows:
                            player_records_by_name[pr["player_name"].lower()] = pr
                    elif 0.60 <= ratio < 0.95:
                        lb_rows = _parse_leaderboard_box_ocr(items, detected_week)
                        for pr in lb_rows:
                            key = pr["player_name"].lower()
                            if key not in player_records_by_name:
                                player_records_by_name[key] = pr

            # 2. Process Standard PDF Tables (for synthetic / table-based PDFs)
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    header = [str(c).strip().lower() if c else "" for c in table[0]]
                    header_str = " ".join(header)

                    if any(k in header_str for k in ["player", "target", "yprr", "adot"]):
                        col_map = {}
                        for idx, col_name in enumerate(header):
                            if "player" in col_name or "name" in col_name:
                                col_map["player"] = idx
                            elif "team" in col_name:
                                col_map["team"] = idx
                            elif "pos" in col_name:
                                col_map["position"] = idx
                            elif "target" in col_name or "tgt" in col_name:
                                col_map["target_share"] = idx
                            elif "yprr" in col_name:
                                col_map["yprr"] = idx
                            elif "adot" in col_name:
                                col_map["adot"] = idx
                            elif "rush epa" in col_name or "epa/rush" in col_name:
                                col_map["rush_epa"] = idx
                            elif "rush sr" in col_name:
                                col_map["rush_sr"] = idx

                        for row in table[1:]:
                            if not row or not any(row):
                                continue
                            p_name = str(row[col_map["player"]]).strip() if "player" in col_map and col_map["player"] < len(row) and row[col_map["player"]] else ""
                            if not p_name or p_name.lower() in ["player", "name", "total"]:
                                continue
                            team_val = norm_team(str(row[col_map["team"]]).strip()) if "team" in col_map and col_map["team"] < len(row) and row[col_map["team"]] else "NFL"
                            pos_val = str(row[col_map["position"]]).strip().upper() if "position" in col_map and col_map["position"] < len(row) and row[col_map["position"]] else "WR"
                            tgt = clean_num(row[col_map["target_share"]], 0.0) if "target_share" in col_map and col_map["target_share"] < len(row) else 0.0
                            if tgt > 1.0:
                                tgt /= 100.0
                            yprr = clean_num(row[col_map["yprr"]], 0.0) if "yprr" in col_map and col_map["yprr"] < len(row) else 0.0
                            adot = clean_num(row[col_map["adot"]], 0.0) if "adot" in col_map and col_map["adot"] < len(row) else 0.0
                            r_epa = clean_num(row[col_map["rush_epa"]], 0.0) if "rush_epa" in col_map and col_map["rush_epa"] < len(row) else 0.0
                            r_sr = clean_num(row[col_map["rush_sr"]], 0.0) if "rush_sr" in col_map and col_map["rush_sr"] < len(row) else 0.0
                            if r_sr > 1.0:
                                r_sr /= 100.0

                            player_records_by_name[p_name.lower()] = {
                                "week": detected_week,
                                "player_name": p_name,
                                "team": team_val,
                                "position": pos_val,
                                "target_share": round(tgt, 3),
                                "yprr": round(yprr, 2),
                                "adot": round(adot, 1),
                                "rush_epa": round(r_epa, 2),
                                "rush_sr": round(r_sr, 3),
                            }

                    elif any(k in header_str for k in ["pressure", "quick", "mof", "rush sr", "pass oe", "epa/db"]):
                        col_map = {}
                        for idx, col_name in enumerate(header):
                            if "team" in col_name and "opp" not in col_name:
                                col_map["team"] = idx
                            elif "opp" in col_name:
                                col_map["opponent"] = idx
                            elif "quick" in col_name or "qp" in col_name:
                                col_map["quick_pressure"] = idx
                            elif "press" in col_name:
                                col_map["pressure"] = idx
                            elif "mof" in col_name and "epa" in col_name:
                                col_map["mof_epa"] = idx
                            elif "mof" in col_name and "%" in col_name:
                                col_map["mof_pct"] = idx
                            elif "mof" in col_name and "mof_epa" not in col_map:
                                col_map["mof_epa"] = idx
                            elif "pass oe" in col_name or "poe" in col_name:
                                col_map["pass_oe"] = idx
                            elif "epa/db" in col_name or "dropback" in col_name:
                                col_map["epa_db"] = idx
                            elif "epa/rush" in col_name or "rush epa" in col_name:
                                col_map["epa_rush"] = idx
                            elif "rush" in col_name and ("sr" in col_name or "success" in col_name):
                                col_map["rush_sr"] = idx
                            elif "takeaway" in col_name or "note" in col_name:
                                col_map["takeaway"] = idx

                        for row in table[1:]:
                            if not row or not any(row):
                                continue
                            team_val = norm_team(str(row[col_map["team"]]).strip()) if "team" in col_map and col_map["team"] < len(row) and row[col_map["team"]] else ""
                            if not team_val or team_val.lower() in ["team", "nfl", "average"]:
                                continue
                            opp_val = norm_team(str(row[col_map["opponent"]]).strip()) if "opponent" in col_map and col_map["opponent"] < len(row) and row[col_map["opponent"]] else "OPP"
                            q_press = clean_num(row[col_map["quick_pressure"]], 18.5) if "quick_pressure" in col_map and col_map["quick_pressure"] < len(row) else 18.5
                            if q_press > 1.0:
                                q_press /= 100.0
                            press_rt = clean_num(row[col_map["pressure"]], q_press * 1.65) if "pressure" in col_map and col_map["pressure"] < len(row) else (q_press * 1.65)
                            if press_rt > 1.0:
                                press_rt /= 100.0
                            mof_p = clean_num(row[col_map["mof_pct"]], 42.0) if "mof_pct" in col_map and col_map["mof_pct"] < len(row) else 42.0
                            if mof_p > 1.0:
                                mof_p /= 100.0
                            mof_epa_val = clean_num(row[col_map["mof_epa"]], 0.35) if "mof_epa" in col_map and col_map["mof_epa"] < len(row) else 0.35
                            p_oe = clean_num(row[col_map["pass_oe"]], 0.0) if "pass_oe" in col_map and col_map["pass_oe"] < len(row) else 0.0
                            e_db = clean_num(row[col_map["epa_db"]], 0.05) if "epa_db" in col_map and col_map["epa_db"] < len(row) else 0.05
                            e_rush = clean_num(row[col_map["epa_rush"]], -0.05) if "epa_rush" in col_map and col_map["epa_rush"] < len(row) else -0.05
                            r_sr = clean_num(row[col_map["rush_sr"]], 42.0) if "rush_sr" in col_map and col_map["rush_sr"] < len(row) else 42.0
                            if r_sr > 1.0:
                                r_sr /= 100.0
                            row_tk = str(row[col_map["takeaway"]]).strip() if "takeaway" in col_map and col_map["takeaway"] < len(row) and row[col_map["takeaway"]] else ""
                            if not row_tk:
                                row_tk = matchup_takeaways.get(team_val, f"Week {detected_week} Review: QP {q_press*100:.1f}%, MOF EPA {mof_epa_val:+.2f}")

                            team_records_by_team[team_val] = {
                                "week": detected_week,
                                "team": team_val,
                                "opponent": opp_val,
                                "pressure_rate": round(press_rt, 3),
                                "quick_pressure_rate": round(q_press, 3),
                                "mof_pct": round(mof_p, 3),
                                "mof_epa_allowed": round(mof_epa_val, 2),
                                "pass_oe": round(p_oe, 1),
                                "epa_db": round(e_db, 2),
                                "epa_rush": round(e_rush, 2),
                                "rush_sr_allowed": round(r_sr, 3),
                                "matchup_takeaway": row_tk,
                                "week_takeaway": "; ".join(week_takeaways[:3]) if week_takeaways else f"Week {detected_week} Realized Scheme Review",
                            }

        df_teams = pd.DataFrame(list(team_records_by_team.values())) if team_records_by_team else pd.DataFrame(columns=[
            "week", "team", "opponent", "pressure_rate", "quick_pressure_rate",
            "mof_pct", "mof_epa_allowed", "pass_oe", "epa_db", "epa_rush", "rush_sr_allowed",
            "matchup_takeaway", "week_takeaway"
        ])
        df_players = pd.DataFrame(list(player_records_by_name.values())) if player_records_by_name else pd.DataFrame(columns=[
            "week", "player_name", "team", "position", "target_share", "yprr", "adot", "rush_epa", "rush_sr"
        ])

        return {
            "success": True,
            "doc_type": "Weekly Review",
            "week": detected_week,
            "team_metrics": df_teams,
            "player_metrics": df_players,
            "matchup_takeaways": matchup_takeaways,
            "week_takeaways": week_takeaways,
            "raw_text_preview": "\n".join(extracted_text_blocks)[:1000] if extracted_text_blocks else ""
        }

    except Exception as e:
        return {
            "success": False,
            "doc_type": "Weekly Review",
            "week": week_num or 1,
            "team_metrics": pd.DataFrame(),
            "player_metrics": pd.DataFrame(),
            "matchup_takeaways": {},
            "week_takeaways": [],
            "raw_text_preview": "",
            "error": str(e)
        }


def parse_preview_pdf(
    pdf_source: Union[str, io.BytesIO, Any],
    week_num: Optional[int] = None
) -> Dict[str, Any]:
    """
    Parse a SumerSports "Weekly Preview" PDF:
      1. Extracts embedded Matchup Preview Boxes (`1.40 <= h/w <= 1.75`) via `winocr` + `data/ocr_cache.json`.
      2. Extracts narrative matchup hypotheses (`Questions We Want Answered`) and player matchup angles.
      3. Extracts standard PDF tables (`page.extract_tables()`) when present.
    """
    preview_by_team: Dict[str, Dict[str, Any]] = {}
    player_angles_by_name: Dict[str, Dict[str, Any]] = {}
    extracted_text_blocks: List[str] = []
    detected_week = week_num or 1

    try:
        with _open_pdf(pdf_source) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    extracted_text_blocks.append(text)
                if not week_num:
                    wk_m = re.search(r"Week\s*(\d+)", text, re.IGNORECASE)
                    if wk_m:
                        detected_week = int(wk_m.group(1))

            full_text = "\n".join(extracted_text_blocks)
            matchup_notes, _, narrative_angles = _extract_narrative_matchups_and_takeaways(full_text)
            for ang in narrative_angles:
                ang["week"] = detected_week
                player_angles_by_name[ang["player_name"].lower()] = ang

            # 1. Process Embedded Preview Box Images
            seen_hashes = set()
            for page in pdf.pages:
                for img in getattr(page, "images", []):
                    w, h = img.get("srcsize", (0, 0))
                    if w < 1400 or h < 1400:
                        continue
                    try:
                        hsh = hashlib.md5(img["stream"].get_data()).hexdigest()
                    except Exception:
                        continue
                    if hsh in seen_hashes:
                        continue
                    seen_hashes.add(hsh)

                    items, _ = _get_ocr_items_for_image(img)
                    if not items:
                        continue
                    ratio = h / float(w)
                    if 1.40 <= ratio <= 1.75:
                        recs = _parse_preview_box_ocr(items, detected_week, matchup_notes)
                        if recs:
                            # Primary box record always wins; mirror record fills in if not yet present
                            preview_by_team[recs[0]["team"]] = recs[0]
                            if recs[1]["team"] not in preview_by_team:
                                preview_by_team[recs[1]["team"]] = recs[1]

            # 2. Process Standard PDF Tables (for synthetic / table-based preview PDFs)
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    header = [str(c).strip().lower() if c else "" for c in table[0]]
                    header_str = " ".join(header)

                    if any(k in header_str for k in ["player", "angle", "alignment", "personnel"]):
                        col_map = {}
                        for idx, col_name in enumerate(header):
                            if "player" in col_name or "name" in col_name:
                                col_map["player"] = idx
                            elif "team" in col_name:
                                col_map["team"] = idx
                            elif "pos" in col_name:
                                col_map["position"] = idx
                            elif "align" in col_name:
                                col_map["alignment_advantage"] = idx
                            elif "personnel" in col_name:
                                col_map["personnel_note"] = idx
                            elif "angle" in col_name or "takeaway" in col_name or "note" in col_name:
                                col_map["matchup_angle"] = idx

                        for row in table[1:]:
                            if not row or not any(row):
                                continue
                            p_name = str(row[col_map["player"]]).strip() if "player" in col_map and col_map["player"] < len(row) and row[col_map["player"]] else ""
                            if not p_name or p_name.lower() in ["player", "name"]:
                                continue
                            p_team = norm_team(str(row[col_map["team"]]).strip()) if "team" in col_map and col_map["team"] < len(row) and row[col_map["team"]] else "NFL"
                            p_pos = str(row[col_map["position"]]).strip().upper() if "position" in col_map and col_map["position"] < len(row) and row[col_map["position"]] else "WR"
                            align_adv = str(row[col_map["alignment_advantage"]]).strip() if "alignment_advantage" in col_map and col_map["alignment_advantage"] < len(row) and row[col_map["alignment_advantage"]] else "Slot / Interior"
                            pers_note = str(row[col_map["personnel_note"]]).strip() if "personnel_note" in col_map and col_map["personnel_note"] < len(row) and row[col_map["personnel_note"]] else "11 Personnel Primary"
                            m_angle = str(row[col_map["matchup_angle"]]).strip() if "matchup_angle" in col_map and col_map["matchup_angle"] < len(row) and row[col_map["matchup_angle"]] else "Positive schematic leverage"

                            player_angles_by_name[p_name.lower()] = {
                                "week": detected_week,
                                "player_name": p_name,
                                "team": p_team,
                                "position": p_pos,
                                "alignment_advantage": align_adv,
                                "personnel_note": pers_note,
                                "matchup_angle": m_angle
                            }

                    elif any(k in header_str for k in ["cover", "man", "zone", "funnel", "blitz", "explosive", "opp"]):
                        col_map = {}
                        for idx, col_name in enumerate(header):
                            if "team" in col_name and "opp" not in col_name:
                                col_map["team"] = idx
                            elif "opp" in col_name:
                                col_map["opponent"] = idx
                            elif "man" in col_name or "cover 1" in col_name:
                                col_map["man_rate"] = idx
                            elif "zone" in col_name or "cover 2" in col_name:
                                col_map["zone_rate"] = idx
                            elif "blitz" in col_name:
                                col_map["blitz_rate"] = idx
                            elif "qp" in col_name or "quick" in col_name or "press" in col_name:
                                col_map["exp_qpr"] = idx
                            elif "mof" in col_name:
                                col_map["exp_mof_epa"] = idx
                            elif "explosive" in col_name or "exp play" in col_name:
                                col_map["explosive_rate"] = idx
                            elif "funnel" in col_name or "focus" in col_name:
                                col_map["funnel_focus"] = idx
                            elif "takeaway" in col_name or "tactical" in col_name or "note" in col_name:
                                col_map["takeaway"] = idx

                        for row in table[1:]:
                            if not row or not any(row):
                                continue
                            team_val = norm_team(str(row[col_map["team"]]).strip()) if "team" in col_map and col_map["team"] < len(row) and row[col_map["team"]] else ""
                            if not team_val or team_val.lower() in ["team", "average"]:
                                continue
                            opp_val = norm_team(str(row[col_map["opponent"]]).strip()) if "opponent" in col_map and col_map["opponent"] < len(row) and row[col_map["opponent"]] else "OPP"
                            man_rt = clean_num(row[col_map["man_rate"]], 28.0) if "man_rate" in col_map and col_map["man_rate"] < len(row) else 28.0
                            if man_rt > 1.0:
                                man_rt /= 100.0
                            zone_rt = clean_num(row[col_map["zone_rate"]], 1.0 - man_rt) if "zone_rate" in col_map and col_map["zone_rate"] < len(row) else (1.0 - man_rt)
                            if zone_rt > 1.0:
                                zone_rt /= 100.0
                            blitz_rt = clean_num(row[col_map["blitz_rate"]], 25.0) if "blitz_rate" in col_map and col_map["blitz_rate"] < len(row) else 25.0
                            if blitz_rt > 1.0:
                                blitz_rt /= 100.0
                            exp_qpr = clean_num(row[col_map["exp_qpr"]], 18.5) if "exp_qpr" in col_map and col_map["exp_qpr"] < len(row) else 18.5
                            if exp_qpr > 1.0:
                                exp_qpr /= 100.0
                            exp_mof = clean_num(row[col_map["exp_mof_epa"]], 0.36) if "exp_mof_epa" in col_map and col_map["exp_mof_epa"] < len(row) else 0.36
                            expl_rt = clean_num(row[col_map["explosive_rate"]], 11.5) if "explosive_rate" in col_map and col_map["explosive_rate"] < len(row) else 11.5
                            if expl_rt > 1.0:
                                expl_rt /= 100.0
                            funnel = str(row[col_map["funnel_focus"]]).strip() if "funnel_focus" in col_map and col_map["funnel_focus"] < len(row) and row[col_map["funnel_focus"]] else "Balanced"
                            tactical = str(row[col_map["takeaway"]]).strip() if "takeaway" in col_map and col_map["takeaway"] < len(row) and row[col_map["takeaway"]] else f"Attack {funnel} leverage against {opp_val}"

                            preview_by_team[team_val] = {
                                "week": detected_week,
                                "team": team_val,
                                "opponent": opp_val,
                                "man_coverage_rate": round(man_rt, 3),
                                "zone_coverage_rate": round(zone_rt, 3),
                                "blitz_rate": round(blitz_rt, 3),
                                "expected_quick_pressure_rate": round(exp_qpr, 3),
                                "expected_mof_epa": round(exp_mof, 2),
                                "explosive_play_rate_allowed": round(expl_rt, 3),
                                "funnel_focus": funnel,
                                "tactical_takeaways": tactical,
                            }

        df_previews = pd.DataFrame(list(preview_by_team.values())) if preview_by_team else pd.DataFrame(columns=[
            "week", "team", "opponent", "man_coverage_rate", "zone_coverage_rate",
            "blitz_rate", "expected_quick_pressure_rate", "expected_mof_epa",
            "explosive_play_rate_allowed", "funnel_focus", "tactical_takeaways"
        ])
        df_angles = pd.DataFrame(list(player_angles_by_name.values())) if player_angles_by_name else pd.DataFrame(columns=[
            "week", "player_name", "team", "position", "alignment_advantage", "personnel_note", "matchup_angle"
        ])

        return {
            "success": True,
            "doc_type": "Weekly Preview",
            "week": detected_week,
            "preview_metrics": df_previews,
            "player_angles": df_angles,
            "raw_text_preview": "\n".join(extracted_text_blocks)[:1000] if extracted_text_blocks else ""
        }

    except Exception as e:
        return {
            "success": False,
            "doc_type": "Weekly Preview",
            "week": week_num or 1,
            "preview_metrics": pd.DataFrame(),
            "player_angles": pd.DataFrame(),
            "raw_text_preview": "",
            "error": str(e)
        }


def parse_sumersports_newsletter(
    pdf_source: Union[str, io.BytesIO, Any],
    default_week: Optional[int] = None,
    force_doc_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Unified entry point for SumerSports PDF ingestion.
    Automatically detects whether the document is a Weekly Preview or Weekly Review
    (unless `force_doc_type` is specified) and dispatches to the appropriate parser.
    """
    detection = detect_newsletter_type(pdf_source, default_week=default_week)
    week_num = default_week or detection["week"]
    doc_type = force_doc_type or detection["doc_type"]

    if "preview" in str(doc_type).lower():
        res = parse_preview_pdf(pdf_source, week_num=week_num)
        res["detection"] = detection
        res.setdefault("team_metrics", pd.DataFrame())
        res.setdefault("player_metrics", pd.DataFrame())
        return res
    else:
        res = parse_review_pdf(pdf_source, week_num=week_num)
        res["detection"] = detection
        res.setdefault("preview_metrics", pd.DataFrame())
        res.setdefault("player_angles", pd.DataFrame())
        return res


def create_sample_sumersports_pdf(output_path: str, week: int = 3) -> str:
    """
    Generate a compliant SumerSports 'Stats & Scheme' Weekly Review PDF using reportlab.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    elements = []
    styles = getSampleStyleSheet()

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
        spaceAfter=10
    )
    elements.append(Paragraph(f"Official Weekly Scheme Breakdown · Week {week} Realized Review & Tendencies", sub_style))
    elements.append(Spacer(1, 8))

    w_offset = (week - 3) * 0.01
    elements.append(Paragraph("<b>Table 1: Team Game & Defensive Scheme Tendencies</b>", styles["Heading3"]))
    team_data = [
        ["Team", "Opp", "Pressure %", "Quick Press %", "MOF %", "MOF EPA", "Pass OE", "EPA/DB", "EPA/Rush", "Rush SR %"],
        ["CLE", "NYG", f"{(0.382 + w_offset)*100:.1f}%", f"{(0.245 + w_offset)*100:.1f}%", "39.0%", f"{0.28 + w_offset:.2f}", "+4.2%", "+0.14", "-0.12", "35.5%"],
        ["PHI", "NO", f"{(0.340 + w_offset)*100:.1f}%", f"{(0.228 + w_offset)*100:.1f}%", "41.5%", f"{0.32 + w_offset:.2f}", "-2.0%", "+0.08", "-0.04", "39.5%"],
        ["CAR", "LV", f"{(0.215 + w_offset)*100:.1f}%", f"{(0.130 + w_offset)*100:.1f}%", "45.0%", f"{0.58 + w_offset:.2f}", "+6.5%", "-0.22", "-0.15", "48.5%"],
        ["TB", "DEN", f"{(0.290 + w_offset)*100:.1f}%", f"{(0.182 + w_offset)*100:.1f}%", "48.0%", f"{0.54 + w_offset:.2f}", "+1.2%", "+0.02", "-0.06", "41.2%"],
        ["BUF", "JAX", f"{(0.325 + w_offset)*100:.1f}%", f"{(0.212 + w_offset)*100:.1f}%", "38.0%", f"{0.34 + w_offset:.2f}", "+5.8%", "+0.28", "+0.02", "37.5%"],
        ["DET", "ARI", f"{(0.330 + w_offset)*100:.1f}%", f"{(0.205 + w_offset)*100:.1f}%", "40.0%", f"{0.38 + w_offset:.2f}", "+3.0%", "+0.22", "+0.06", "36.2%"],
        ["NYG", "CLE", f"{(0.300 + w_offset)*100:.1f}%", f"{(0.185 + w_offset)*100:.1f}%", "43.0%", f"{0.47 + w_offset:.2f}", "-1.5%", "-0.10", "-0.09", "47.5%"]
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


def create_sample_preview_pdf(output_path: str, week: int = 3) -> str:
    """
    Generate a compliant SumerSports 'Weekly Preview' PDF using reportlab.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter, leftMargin=32, rightMargin=32, topMargin=36, bottomMargin=36)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontSize=19,
        leading=23,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=6
    )
    elements.append(Paragraph(f"SumerSports NFL Weekly Matchup Preview - Week {week}", title_style))

    sub_style = ParagraphStyle(
        "SubStyle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )
    elements.append(Paragraph(f"Pre-Game Coverage Shells, Funnel Matchup Focus & Tactical Hypotheses · Week {week} Preview", sub_style))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Table 1: Matchup Preview - Coverage Shells & Funnel Tendencies</b>", styles["Heading3"]))
    preview_data = [
        ["Team", "Opp", "Cover 1/3 Man %", "Cover 2/4/6 Zone %", "Blitz %", "Exp Quick Press %", "Exp MOF EPA", "Explosive %", "Funnel Focus", "Tactical Takeaway"],
        ["CLE", "NYG", "34.0%", "66.0%", "28.5%", "19.0%", "0.26", "10.2%", "Boundary", "Expect clean pocket & vertical play-action shots"],
        ["PHI", "NO", "26.0%", "74.0%", "21.0%", "20.5%", "0.30", "10.8%", "Inside/Slot", "Attack middle zone voids with Goedert & slot motion"],
        ["CAR", "LV", "22.0%", "78.0%", "24.0%", "14.5%", "0.42", "13.5%", "Inside/Slot", "Heavy Cover 3/4 shell funnels targets to interior slot"],
        ["TB", "DEN", "29.0%", "71.0%", "31.0%", "17.5%", "0.44", "12.4%", "Inside/Slot", "Hot-route slot answers against Denver simulated pressures"],
        ["BUF", "JAX", "38.0%", "62.0%", "26.0%", "19.5%", "0.32", "11.0%", "Inside/Slot", "Shakir & Kincaid exploit man-match linebackers"],
        ["DET", "ARI", "24.0%", "76.0%", "29.0%", "19.0%", "0.35", "12.8%", "Inside/Slot", "St. Brown & LaPorta option routes vs two-high shell"],
        ["NYG", "CLE", "41.0%", "59.0%", "32.0%", "17.0%", "0.38", "12.0%", "Boundary", "Nabers 1-on-1 perimeter isolation against single-high"]
    ]
    t_prev = Table(preview_data, colWidths=[34, 32, 54, 58, 40, 56, 48, 48, 56, 120])
    t_prev.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
    ]))
    elements.append(t_prev)
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("<b>Table 2: Player-Level Matchup Angles & Alignment Advantages</b>", styles["Heading3"]))
    angles_data = [
        ["Player", "Team", "Pos", "Alignment Advantage", "Personnel Note", "Matchup Angle"],
        ["Amon-Ra St. Brown", "DET", "WR", "Slot Primary (65%)", "11 Personnel Motion", "Attacks ARI middle zone voids"],
        ["DeMario Douglas", "NE", "WR", "Slot Primary (72%)", "11 Personnel Exclusive", "High-floor checkdown outlet vs blitz"],
        ["Drake London", "ATL", "WR", "Boundary X (72%)", "12 & 11 Personnel", "Perimeter size mismatch vs press corners"],
        ["Brock Bowers", "LV", "TE", "Inline / Slot Hybrid", "12 Personnel Featured", "Seam mismatch vs CAR linebackers"],
        ["Malik Nabers", "NYG", "WR", "Pre-Snap Motion Z", "11 Personnel Alpha", "High target share vs single-high man"]
    ]
    t_angles = Table(angles_data, colWidths=[95, 38, 32, 95, 95, 165])
    t_angles.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
    ]))
    elements.append(t_angles)

    doc.build(elements)
    return output_path


def seed_sample_newsletters_folder(folder_path: str = "newsletters", weeks: Optional[List[int]] = None) -> List[str]:
    """
    Populate `folder_path` with sample PDFs ONLY if the folder has no PDFs yet.
    Never overwrites existing user newsletter PDFs.
    """
    os.makedirs(folder_path, exist_ok=True)
    existing = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if existing:
        return existing

    target_weeks = weeks or [1, 2, 3]
    created_files: List[str] = []
    for w in target_weeks:
        prev_path = os.path.join(folder_path, f"week{w}_preview.pdf")
        rev_path = os.path.join(folder_path, f"week{w}_review.pdf")
        if not os.path.exists(prev_path):
            create_sample_preview_pdf(prev_path, week=w)
        if not os.path.exists(rev_path):
            create_sample_sumersports_pdf(rev_path, week=w)
        created_files.extend([prev_path, rev_path])
    return created_files
