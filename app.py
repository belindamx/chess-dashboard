"""
Streamlit dashboard for chess.com game history (user: belindafails)
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import urllib.request
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

APP_DIR       = Path(__file__).resolve().parent
PROCESSED_DIR = APP_DIR / "data" / "processed"
MIN_OPENING_GAMES = 20
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

ACCENT    = "#2ec4b6"
WIN_C     = "#52b788"
LOSS_C    = "#e76f51"
DRAW_C    = "#f4a261"
TC_COLORS = {"bullet": "#2ec4b6", "blitz": "#f4a261", "rapid": "#3B82F6", "daily": "#9CA3AF"}
PALETTE   = ["#2ec4b6", "#f4a261", "#3B82F6", "#9CA3AF"]
CLUSTER_C = ["#2ec4b6", "#f4a261", "#7C3AED", "#e76f51"]
N_CLUSTERS = 4
_N_CLUSTERS_FIT = 5

# css
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Press+Start+2P&display=swap');

html, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp { background: #e8f4f8; }
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1120px;
}

/* ── Sidebar — deep winter teal-navy ── */
[data-testid="stSidebar"] {
    background: #1e3848 !important;
    border-right: 2px solid #2ec4b6 !important;
}
[data-testid="stSidebar"] * { color: #d8eef4 !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: #26495e !important;
    border: 2px solid #3a6a80 !important;
    color: #eaf6fb !important;
    border-radius: 4px !important;
}
[data-testid="stSidebar"] label {
    color: #7ab8cc !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] hr { border-color: #2e5a70 !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    gap: 0;
    background: transparent !important;
    border-bottom: 3px solid #1a2e35;
    box-shadow: 0 1px 0 #2ec4b6;
}
[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #5a8a90 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 8px !important;
    font-weight: 400 !important;
    padding: 12px 22px !important;
    border-radius: 0 !important;
    letter-spacing: 0.5px !important;
    line-height: 2 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #1a2e35 !important;
    background: rgba(46,196,182,0.08) !important;
    border-bottom: 3px solid #2ec4b6 !important;
    margin-bottom: -3px !important;
}
[data-baseweb="tab-panel"] { padding-top: 24px !important; }

/* ── Multiselect tags ── */
[data-baseweb="tag"] { background: #b8e8e4 !important; color: #1a6b63 !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #26495e !important; color: #d8eef4 !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span { color: #d8eef4 !important; }

/* ── Slider ── */
[data-baseweb="slider"] [role="slider"] {
    background: #2ec4b6 !important;
    border-color: #2ec4b6 !important;
    box-shadow: none !important;
}
[data-baseweb="slider"] div[data-testid="stSliderTrackFill"] {
    background: #2ec4b6 !important;
}
[data-baseweb="slider"] > div > div:nth-child(2),
[data-baseweb="slider"] > div > div:nth-child(3) {
    background: #2ec4b6 !important;
}

/* ── Hide chrome ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stMetric"] { background: transparent !important; }
</style>

<style>
/* ── Hero — retro arcade terminal ── */
.hero-wrap {
    background: #0d2233;
    background-image:
        repeating-linear-gradient(
            0deg,
            rgba(0,0,0,0.18) 0px, rgba(0,0,0,0.18) 1px,
            transparent 1px, transparent 4px
        ),
        repeating-linear-gradient(
            90deg,
            rgba(46,196,182,0.025) 0px, rgba(46,196,182,0.025) 1px,
            transparent 1px, transparent 16px
        );
    border: 3px solid #2ec4b6;
    box-shadow:
        inset 0 0 0 1px rgba(46,196,182,0.15),
        0 4px 0 0 #1a2e35,
        0 0 24px rgba(46,196,182,0.08);
    border-radius: 0;
    padding: 22px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 22px;
    position: relative;
}
/* pixel corner brackets */
.hero-wrap::before {
    content: '';
    position: absolute;
    top: 5px; left: 5px; right: 5px; bottom: 5px;
    border-top: 1px solid rgba(46,196,182,0.18);
    border-left: 1px solid rgba(46,196,182,0.18);
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    top: 5px; left: 5px; right: 5px; bottom: 5px;
    border-bottom: 1px solid rgba(46,196,182,0.18);
    border-right: 1px solid rgba(46,196,182,0.18);
    pointer-events: none;
}
.hero-avatar { flex-shrink: 0; position: relative; z-index: 1; }
.hero-text { flex: 1; position: relative; z-index: 1; }
.hero-eyebrow {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 8px; font-weight: 400; color: #2ec4b6;
    letter-spacing: 2px; margin-bottom: 10px;
    text-shadow: 0 0 8px rgba(46,196,182,0.6);
}
.hero-name {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 19px; font-weight: 400; color: #e8f8f6; line-height: 1.5;
    text-shadow: 2px 2px 0 rgba(0,0,0,0.6), 0 0 16px rgba(46,196,182,0.25);
    letter-spacing: 0.5px;
}
.hero-accent { color: #2ec4b6; }
.hero-sub { font-size: 11px; color: #7ab8cc; margin-top: 12px; letter-spacing: 0.3px; }

/* ── Stat row ── */
.stat-row {
    display: flex; gap: 0;
    border: 2px solid #b8d4e0; border-radius: 4px;
    overflow: hidden; background: #ffffff; margin: 0 0 20px;
}
.stat-block {
    flex: 1; padding: 22px 24px;
    border-right: 2px solid #b8d4e0;
}
.stat-block:last-child { border-right: none; }
.stat-val { font-size: 30px; font-weight: 800; color: #1a2e35; line-height: 1; font-variant-numeric: tabular-nums; }
.stat-lbl {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 7px; font-weight: 400; color: #5a7a85;
    text-transform: uppercase; letter-spacing: 0.5px; margin-top: 8px;
    line-height: 1.8;
}
.stat-note { font-size: 11px; color: #5a7a85; margin-top: 4px; }

/* ── Section divider / header ── */
.bx-sh {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px; font-weight: 400; color: #1a2e35; margin: 0;
    padding-bottom: 10px;
    border-top: 2px solid #1a2e35;
    box-shadow: 0 -1px 0 #2ec4b6;
    padding-top: 12px;
    line-height: 1.8;
    letter-spacing: 0.3px;
}
.bx-ss { font-size: 11px; color: #5a7a85; margin: 2px 0 14px; }

/* ── At a glance — retro pixel-game dialog box ── */
.bx-glance-outer {
    position: relative;
    margin-bottom: 20px;
    /* stretch to match the rating journey chart height */
    min-height: 455px;
    display: flex;
    flex-direction: column;
}
/* Outer card: sharp corners, double-border NES-style with hard drop shadow */
.bx-glance-wrap {
    flex: 1;
    display: flex;
    flex-direction: column;
    border-radius: 0;
    border: 3px solid #1a2e35;
    box-shadow:
        inset 0 0 0 2px #2ec4b6,   /* inner teal ring */
        5px 5px 0 0 #1a2e35;        /* hard pixel drop shadow */
    background: #f0fafb;
    overflow: visible;
    position: relative;
}
/* Pixelated step-pointer at bottom-left (3-step staircase) */
.bx-glance-wrap::after {
    content: '';
    position: absolute;
    bottom: -14px; left: 16px;
    width: 0; height: 0;
    border-left:  14px solid transparent;
    border-right: 14px solid transparent;
    border-top:   14px solid #1a2e35;
}
.bx-glance-wrap::before {
    content: '';
    position: absolute;
    bottom: -10px; left: 19px;
    width: 0; height: 0;
    border-left:  11px solid transparent;
    border-right: 11px solid transparent;
    border-top:   11px solid #2ec4b6;
    z-index: 1;
}

/* Header — dark teal with CRT scanline texture + pixel grid overlay */
.bx-glance-header {
    position: relative; overflow: hidden;
    background: #133e45;
    background-image:
        repeating-linear-gradient(
            0deg,
            rgba(0,0,0,0.18) 0px, rgba(0,0,0,0.18) 1px,
            transparent 1px, transparent 4px
        ),
        repeating-linear-gradient(
            90deg,
            rgba(255,255,255,0.03) 0px, rgba(255,255,255,0.03) 1px,
            transparent 1px, transparent 8px
        );
    padding: 14px 18px 12px;
    border-bottom: 3px solid #1a2e35;
}
/* Teal pixel corner accents */
.bx-glance-header::before {
    content: '';
    position: absolute;
    top: 4px; left: 4px; right: 4px; bottom: 0;
    border-top: 2px solid rgba(46,196,182,0.35);
    border-left: 2px solid rgba(46,196,182,0.35);
    pointer-events: none;
}
.bx-glance-piece {
    font-size: 38px; line-height: 1; opacity: 0.12;
    position: absolute; right: 10px; bottom: -2px;
    user-select: none; color: #2ec4b6;
}
.bx-glance-eyebrow { display: none; }
.bx-glance-title {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 11px; color: #c8f0ec; line-height: 1.8;
    letter-spacing: 0.5px;
    text-shadow: 2px 2px 0 rgba(0,0,0,0.4);
}

/* Body rows — game stat entries */
.bx-glance-body {
    background: #f0fafb;
    padding: 4px 0;
    flex: 1;
    display: flex;
    flex-direction: column;
}
.bx-ic {
    flex: 1;
    padding: 10px 16px 10px 32px;
    border-bottom: 2px dashed #b8d4e0;
    display: flex; flex-direction: column;
    gap: 4px; justify-content: center;
    position: relative;
    transition: background 0.1s;
}
.bx-ic:last-child { border-bottom: none; }
/* ▶ game cursor — hidden by default, shown on hover */
.bx-ic::before {
    content: '▶';
    position: absolute;
    left: 12px; top: 50%;
    transform: translateY(-50%);
    font-size: 8px;
    color: transparent;
    transition: color 0.1s;
}
.bx-ic:hover { background: #d8f4f0; }
.bx-ic:hover::before { color: #2ec4b6; }

/* Label: Press Start 2P at 7px — retro stat name */
.bx-ic-lbl {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 7px; color: #5a8a90;
    text-transform: uppercase; letter-spacing: 0.5px;
    line-height: 1.6;
}
/* Value: Inter bold — stays readable */
.bx-ic-val { font-size: 14px; font-weight: 700; color: #1a2e35; }

/* ── Format card grid ── */
.fmt-grid { display: flex; gap: 12px; margin-bottom: 4px; }
.fmt-card {
    flex: 1; background: #ffffff; border: 2px solid #b8d4e0;
    border-radius: 4px; overflow: hidden;
}
.fmt-card-header {
    background: #d4eef2; padding: 10px 16px;
    border-bottom: 2px solid #b8d4e0;
    display: flex; align-items: center; gap: 8px;
}
.fmt-tc-icon { font-size: 18px; line-height: 1; }
.fmt-tc-name {
    font-size: 13px; font-weight: 800; color: #1a2e35;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.fmt-card-body { padding: 14px 16px; }
.fmt-best { font-size: 32px; font-weight: 800; color: #1a2e35; line-height: 1; }
.fmt-best-lbl { font-size: 10px; font-weight: 700; color: #5a7a85; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 3px; }
.fmt-divider { border: none; border-top: 1px solid #d4eef2; margin: 10px 0; }
.fmt-row { display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px; }
.fmt-row-lbl { font-size: 11px; color: #5a7a85; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; }
.fmt-row-val { font-size: 13px; font-weight: 700; color: #1a2e35; }

/* ── Cluster grid ── */
.cc-grid { display: flex; gap: 12px; margin-bottom: 20px; }
.cc {
    flex: 1; background: #ffffff; border: 2px solid #b8d4e0;
    border-radius: 4px; padding: 18px 16px;
}
.cc-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px;
    background: #2ec4b6; color: #ffffff;
    border-radius: 4px; font-size: 11px; font-weight: 800;
    margin-bottom: 12px;
}
.cc-num { font-size: 28px; font-weight: 800; color: #1a2e35; line-height: 1; }
.cc-unit { font-size: 10px; color: #5a7a85; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px; }
.cc-lbl { font-size: 12px; font-weight: 700; color: #1a2e35; margin: 10px 0 6px; line-height: 1.4; border-top: 1px solid #d4eef2; padding-top: 10px; }
.cc-meta { font-size: 11px; color: #5a7a85; line-height: 1.6; }

/* ── Opening list ── */
.op-wrap { background: #ffffff; border: 2px solid #b8d4e0; border-radius: 4px; padding: 4px 16px; }
.op-row { display: flex; align-items: center; padding: 10px 0; border-bottom: 1px solid #e0edf2; }
.op-row:last-child { border-bottom: none; }
.op-name-wrap { flex: 1; padding-right: 12px; }
.op-name-text { font-size: 12px; font-weight: 500; color: #1a2e35; line-height: 1.3; }
.op-bar-bg { background: #d4eef2; border-radius: 3px; height: 4px; margin-top: 4px; }
.op-pct { font-size: 14px; font-weight: 700; min-width: 40px; text-align: right; }
.op-n { font-size: 11px; color: #5a7a85; min-width: 44px; text-align: right; margin-left: 8px; }

/* ── Journey tab ── */
.journey-wrap {
    position: relative;
    padding: 40px 0 60px;
    overflow-x: auto;
}
.journey-player {
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.journey-player-label {
    font-size: 11px; font-weight: 700; color: #5a7a85;
    text-transform: uppercase; letter-spacing: 1px;
}
.journey-path {
    display: flex;
    align-items: flex-start;
    gap: 0;
    position: relative;
    padding-bottom: 16px;
    min-width: 700px;
}
.journey-node-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    position: relative;
}
/* dashed connector between nodes */
.journey-node-wrap:not(:last-child)::after {
    content: '';
    position: absolute;
    top: 30px;
    left: 50%;
    width: 100%;
    height: 0;
    border-top: 3px dashed #2ec4b6;
    z-index: 0;
}
.journey-node-wrap.locked:not(:last-child)::after {
    border-color: #aac8d0;
}
.journey-node {
    width: 60px; height: 60px;
    border-radius: 50%;
    background: #2ec4b6;
    border: 3px solid #1a9e92;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    position: relative;
    z-index: 1;
    box-shadow: 0 2px 8px rgba(46,196,182,0.3);
}
.journey-node.locked {
    background: #c8dde4;
    border-color: #aac8d0;
    opacity: 0.7;
    box-shadow: none;
}
.journey-year {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 10px;
    color: #1a2e35;
    margin-top: 10px;
    text-align: center;
}
.journey-year.locked {
    color: #5a7a85;
    opacity: 0.6;
}
.journey-desc {
    font-size: 11px;
    color: #1a2e35;
    margin-top: 6px;
    text-align: center;
    max-width: 110px;
    line-height: 1.5;
}
.journey-desc.locked {
    color: #5a7a85;
    opacity: 0.7;
    font-style: italic;
}
</style>
"""

# avatar
_AVATAR_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 32 32" shape-rendering="crispEdges" style="image-rendering:pixelated;display:block;">
  <!-- === BACK HAIR (drawn first, behind face) === -->
  <rect x="3"  y="7"  width="6"  height="23" fill="#3d1c02"/>
  <rect x="23" y="7"  width="6"  height="23" fill="#3d1c02"/>
  <!-- hair width at shoulder level -->
  <rect x="4"  y="22" width="5"  height="8"  fill="#3d1c02"/>
  <rect x="23" y="22" width="5"  height="8"  fill="#3d1c02"/>

  <!-- === HAIR CROWN === -->
  <rect x="10" y="0"  width="12" height="2"  fill="#3d1c02"/>
  <rect x="7"  y="2"  width="18" height="2"  fill="#3d1c02"/>
  <rect x="5"  y="4"  width="22" height="2"  fill="#3d1c02"/>

  <!-- === FACE (warm golden skin) === -->
  <!-- rounded face shape -->
  <rect x="9"  y="5"  width="14" height="1"  fill="#f0a050"/>
  <rect x="8"  y="6"  width="16" height="13" fill="#f0a050"/>
  <rect x="9"  y="19" width="14" height="1"  fill="#f0a050"/>
  <rect x="11" y="20" width="10" height="1"  fill="#f0a050"/>

  <!-- === HAIR FRONT — sweeps left across forehead === -->
  <!-- main left-side sweep -->
  <rect x="5"  y="5"  width="7"  height="7"  fill="#3d1c02"/>
  <!-- hair highlight strand -->
  <rect x="7"  y="6"  width="2"  height="4"  fill="#6b3510"/>
  <!-- right side thin framing -->
  <rect x="22" y="5"  width="5"  height="4"  fill="#3d1c02"/>
  <!-- front hair continues down left cheek -->
  <rect x="5"  y="12" width="4"  height="8"  fill="#3d1c02"/>
  <!-- right side framing lower -->
  <rect x="22" y="9"  width="5"  height="11" fill="#3d1c02"/>

  <!-- === EYEBROWS === -->
  <rect x="12" y="9"  width="4"  height="1"  fill="#2c1400"/>
  <rect x="18" y="9"  width="4"  height="1"  fill="#2c1400"/>

  <!-- === EYES === -->
  <!-- left eye -->
  <rect x="12" y="11" width="4"  height="4"  fill="#1a0a00"/>
  <rect x="13" y="12" width="2"  height="2"  fill="#5a2800"/>
  <rect x="15" y="11" width="1"  height="1"  fill="#ffffff"/>
  <!-- right eye -->
  <rect x="18" y="11" width="4"  height="4"  fill="#1a0a00"/>
  <rect x="19" y="12" width="2"  height="2"  fill="#5a2800"/>
  <rect x="21" y="11" width="1"  height="1"  fill="#ffffff"/>

  <!-- === NOSE (subtle) === -->
  <rect x="15" y="16" width="2"  height="1"  fill="#d08840"/>

  <!-- === BLUSH === -->
  <rect x="10" y="17" width="2"  height="1"  fill="#e07848"/>
  <rect x="20" y="17" width="2"  height="1"  fill="#e07848"/>

  <!-- === MOUTH === -->
  <rect x="13" y="18" width="1"  height="1"  fill="#b85c30"/>
  <rect x="18" y="18" width="1"  height="1"  fill="#b85c30"/>
  <rect x="14" y="19" width="4"  height="1"  fill="#b85c30"/>

  <!-- === NECK === -->
  <rect x="13" y="21" width="6"  height="2"  fill="#f0a050"/>

  <!-- === DARK HOODIE === -->
  <!-- shoulders -->
  <rect x="4"  y="23" width="24" height="2"  fill="#1e1e2e"/>
  <!-- body -->
  <rect x="5"  y="25" width="22" height="7"  fill="#1e1e2e"/>
  <!-- v-neck opening -->
  <rect x="14" y="23" width="4"  height="4"  fill="#2e2e42"/>
  <!-- pocket -->
  <rect x="11" y="28" width="10" height="4"  fill="#26263a"/>
  <!-- drawstrings -->
  <rect x="14" y="25" width="1"  height="5"  fill="#ccc498"/>
  <rect x="17" y="25" width="1"  height="5"  fill="#ccc498"/>
</svg>
"""
 
# altair
def _minimal_theme():
    return {"config": {
        "view": {"strokeWidth": 0, "fill": "transparent"},
        "background": "transparent",
        "axis": {
            "gridColor": "#d4eef2", "gridWidth": 1,
            "domainColor": "#b8d4e0", "tickColor": "#b8d4e0",
            "labelColor": "#5a7a85", "labelFontSize": 11,
            "titleColor": "#5a7a85", "titleFontSize": 11, "titleFontWeight": 600,
        },
        "legend": {
            "labelColor": "#5a7a85", "labelFontSize": 11,
            "titleColor": "#5a7a85", "titleFontSize": 11,
            "titleFontWeight": 600, "symbolSize": 80,
        },
    }}

alt.themes.register("minimal", _minimal_theme)
alt.themes.enable("minimal")

def fmt_k(n: float) -> str:
    n = int(round(float(n)))
    if n >= 1_000:
        return f"{n // 1000}k" if n % 1000 == 0 else f"{n / 1000:.1f}k"
    return f"{n:,}"


def hour_label(h) -> str:
    h = int(round(float(h))) % 24
    if h == 0: return "12 AM"
    if h < 12: return f"{h} AM"
    if h == 12: return "12 PM"
    return f"{h - 12} PM"

# html
def stat_row_html(stats: list) -> str:
    blocks = []
    for val, lbl, note in stats:
        note_html = f'<div class="stat-note">{note}</div>' if note else ""
        blocks.append(
            f'<div class="stat-block">'
            f'<div class="stat-val">{val}</div>'
            f'<div class="stat-lbl">{lbl}</div>'
            f'{note_html}</div>'
        )
    return f'<div class="stat-row">{"".join(blocks)}</div>'


def section_html(title: str, sub: str = "") -> str:
    sub_html = f'<div class="bx-ss">{sub}</div>' if sub else ""
    return f'<div class="bx-sh">{title}</div>{sub_html}'


def insight_cards_html(rows: list) -> str:
    parts = []
    for lbl, val in rows:
        parts.append(
            f'<div class="bx-ic">'
            f'<div class="bx-ic-lbl">{lbl}</div>'
            f'<div class="bx-ic-val">{val}</div>'
            f'</div>'
        )
    body = "".join(parts)
    return (
        f'<div class="bx-glance-outer">'
        f'<div class="bx-glance-wrap">'
        f'<div class="bx-glance-header">'
        f'<div class="bx-glance-eyebrow">chess.com</div>'
        f'<div class="bx-glance-title">At a glance</div>'
        f'<div class="bx-glance-piece">♛</div>'
        f'</div>'
        f'<div class="bx-glance-body">{body}</div>'
        f'</div>'
        f'</div>'
    )


def opening_list_html(rows: list, bar_color: str) -> str:
    items = []
    for name, pct, n in rows:
        bar_w = min(int(pct), 100)
        display = (name[:38] + "…") if len(name) > 40 else name
        items.append(
            f'<div class="op-row">'
            f'<div class="op-name-wrap">'
            f'<div class="op-name-text">{display}</div>'
            f'<div class="op-bar-bg">'
            f'<div style="width:{bar_w}%;background:{bar_color};border-radius:3px;height:4px;"></div>'
            f'</div></div>'
            f'<div class="op-pct" style="color:{bar_color};">{pct:.0f}%</div>'
            f'<div class="op-n">{n:,}g</div>'
            f'</div>'
        )
    return f'<div class="op-wrap">{"".join(items)}</div>'


def cluster_grid_html(agg: pd.DataFrame) -> str:
    cards = []
    for i, (_, row) in enumerate(agg.iterrows()):
        wr  = row.get("win_rate", np.nan)
        acc = row.get("avg_accuracy", np.nan)
        meta_lines = []
        if pd.notna(wr):  meta_lines.append(f'{wr * 100:.0f}% win rate')
        if pd.notna(acc): meta_lines.append(f'{acc:.1f}% accuracy')
        meta_str = "<br>".join(meta_lines)
        cards.append(
            f'<div class="cc">'
            f'<div class="cc-badge">{i + 1}</div>'
            f'<div class="cc-num">{fmt_k(row["games"])}</div>'
            f'<div class="cc-unit">games</div>'
            f'<div class="cc-lbl">{row["label"]}</div>'
            f'<div class="cc-meta">{meta_str}</div>'
            f'</div>'
        )
    return f'<div class="cc-grid">{"".join(cards)}</div>'


def _avatar_html(size: int = 130) -> str:
    """Prefer a real photo (avatar.jpg/png/gif); fall back to the pixel SVG."""
    import base64
    for candidate in ["avatar.jpg", "avatar.png", "avatar.gif",
                      "assets/avatar.png", "assets/avatar.jpg", "assets/avatar.gif"]:
        p = APP_DIR / candidate
        if p.exists():
            mime = "image/gif" if candidate.endswith(".gif") else (
                   "image/jpeg" if candidate.endswith(".jpg") else "image/png")
            b64 = base64.b64encode(p.read_bytes()).decode()
            return (
                f'<img src="data:{mime};base64,{b64}" '
                f'width="{size}" height="{size}" '
                f'style="display:block;object-fit:contain;position:relative;z-index:2;">'
            )
    return _AVATAR_SVG


def hero_html(username: str, span: int, k: dict) -> str:
    sub = "10 years of chess. Mostly bullet."
    return (
        f'<div class="hero-wrap">'
        f'<div class="hero-avatar">{_avatar_html()}</div>'
        f'<div class="hero-text">'
        f'<div class="hero-eyebrow">chess.com</div>'
        f'<div class="hero-name">{username} <span class="hero-accent">♟</span></div>'
        f'<div class="hero-sub">{sub}</div>'
        f'</div>'
        f'</div>'
    )


def journey_html() -> str:
    unlocked = [
        ("🏁", "2004", "First chess tournament. Lost every game."),
        ("⭐", "2007", "US Chess Federation Top 100 Girls Under 13"),
        ("💻", "2016", "Joined Chess.com"),
        ("📈", "2026", "32k+ games across 10 years"),
    ]
    locked = [
        ("🕐", "Next", "Slow down: Blitz &amp; Rapid"),
        ("🤝", "Next", "More over the board and variants: Bughouse, Hand &amp; Brain"),
    ]

    nodes = []
    for icon, year, desc in unlocked:
        nodes.append(
            f'<div class="journey-node-wrap">'
            f'<div class="journey-node">{icon}</div>'
            f'<div class="journey-year">{year}</div>'
            f'<div class="journey-desc">{desc}</div>'
            f'</div>'
        )
    for icon, year, desc in locked:
        nodes.append(
            f'<div class="journey-node-wrap locked">'
            f'<div class="journey-node locked">🔒</div>'
            f'<div class="journey-year locked">{year}</div>'
            f'<div class="journey-desc locked">{desc}</div>'
            f'</div>'
        )

    player_block = (
        f'<div class="journey-player">'
        f'{_avatar_html(size=52)}'
        f'<div class="journey-player-label">Player: belindafails</div>'
        f'</div>'
    )

    return (
        f'<div class="journey-wrap">'
        f'{player_block}'
        f'<div class="journey-path">{"".join(nodes)}</div>'
        f'</div>'
    )

# load data
@st.cache_data(show_spinner=False)
def load_data(path_str: str, mtime_ns: int) -> pd.DataFrame:
    wanted = {
        "end_date", "time_class", "time_control",
        "my_rating", "opponent_rating", "rating_diff",
        "my_result", "opponent_username", "game_url",
        "opening_name", "eco_code", "eco",
        "my_color", "white_accuracy", "black_accuracy",
    }
    df = pd.read_csv(path_str, usecols=lambda c: c in wanted)

    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    for col in ["my_rating", "opponent_rating", "rating_diff", "white_accuracy", "black_accuracy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "my_result" in df.columns:
        df["is_win"] = (df["my_result"] == "win").astype(int)
        draws = {"stalemate", "repetition", "insufficient", "timevsinsufficient", "agreed"}
        df["result_cat"] = df["my_result"].map(
            lambda r: "Win" if r == "win" else ("Draw" if r in draws else "Loss")
        )

    if "end_date" in df.columns and df["end_date"].notna().any():
        df["month"]       = df["end_date"].dt.to_period("M").astype(str)
        df["year"]        = df["end_date"].dt.year
        df["day_of_week"] = df["end_date"].dt.day_name()
        df["hour"]        = df["end_date"].dt.hour
        df = df.sort_values("end_date").reset_index(drop=True)

    on  = df.get("opening_name", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    ec  = df.get("eco_code",     pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    eco = df.get("eco",          pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    eco_clean = (
        eco.str.replace(r"https?://[^/]+/openings/", "", regex=True)
           .str.replace(r"[-_]", " ", regex=True)
           .str.strip()
    )
    df["opening_label"] = (
        on.where(on != "", ec)
          .where(ec != "", eco_clean)
          .where(eco_clean != "", "Unknown")
    )

    if "my_color" in df.columns:
        wa = df.get("white_accuracy", pd.Series([np.nan] * len(df)))
        ba = df.get("black_accuracy", pd.Series([np.nan] * len(df)))
        df["my_accuracy"] = wa.where(df["my_color"] == "white", ba)

    if "time_control" in df.columns:
        base = df["time_control"].astype(str).str.split("+").str[0]
        df["tc_secs"] = pd.to_numeric(
            base.str.replace(r"[^\d]", "", regex=True), errors="coerce"
        ).clip(upper=3600)

    return df


def get_csv_path(username: str) -> Path:
    return PROCESSED_DIR / f"{username}_games.csv"


_FORMAT_KEYS = {
    "chess_bullet": "Bullet",
    "chess_blitz":  "Blitz",
    "chess_rapid":  "Rapid",
}

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_player_stats(username: str) -> list:
    url = f"https://api.chess.com/pub/player/{username}/stats"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ChessExplorer/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            raw = json.loads(resp.read().decode())
    except Exception:
        return []

    rows = []
    for key, label in _FORMAT_KEYS.items():
        if key not in raw:
            continue
        d       = raw[key]
        best    = d.get("best", {}).get("rating")
        current = d.get("last", {}).get("rating")
        rec     = d.get("record", {})
        wins    = rec.get("win",  0)
        draws   = rec.get("draw", 0)
        losses  = rec.get("loss", 0)
        total   = wins + draws + losses
        if total == 0 and not best:
            continue
        rows.append({
            "format":  label,
            "best":    best,
            "current": current,
            "wins":    wins,
            "draws":   draws,
            "losses":  losses,
            "total":   total,
        })
    return rows


_TC_ICONS = {"Bullet": "♟", "Blitz": "♞", "Rapid": "♝", "Daily": "♜"}

def api_stats_html(rows: list) -> str:
    if not rows:
        return ""
    cards = []
    for r in rows:
        best_s    = str(r["best"])    if r["best"]    else "—"
        current_s = str(r["current"]) if r["current"] else "—"
        total_s   = f'{r["total"]:,}' if r["total"]   else "—"
        icon = _TC_ICONS.get(r["format"], "♟")
        cards.append(
            f'<div class="fmt-card">'
            f'<div class="fmt-card-header">'
            f'<span class="fmt-tc-icon">{icon}</span>'
            f'<span class="fmt-tc-name">{r["format"]}</span>'
            f'</div>'
            f'<div class="fmt-card-body">'
            f'<div class="fmt-best">{best_s}</div>'
            f'<div class="fmt-best-lbl">Peak rating</div>'
            f'<hr class="fmt-divider">'
            f'<div class="fmt-row"><span class="fmt-row-lbl">Current</span><span class="fmt-row-val">{current_s}</span></div>'
            f'<div class="fmt-row"><span class="fmt-row-lbl">Games</span><span class="fmt-row-val">{total_s}</span></div>'
            f'</div>'
            f'</div>'
        )
    return f'<div class="fmt-grid">{"".join(cards)}</div>'

# cache computations
@st.cache_data(show_spinner=False)
def compute_kpis(key: str, df: pd.DataFrame) -> dict:
    out: dict = {"total": len(df)}
    if "my_rating" in df.columns:
        valid = df.dropna(subset=["my_rating"])
        if len(valid):
            peak_idx = valid["my_rating"].idxmax()
            out["peak_rating"] = int(valid.loc[peak_idx, "my_rating"])
            if "time_class" in df.columns:
                out["peak_tc"] = valid.loc[peak_idx, "time_class"]
    if "time_class" in df.columns and "my_rating" in df.columns:
        last = (
            df.dropna(subset=["end_date", "my_rating", "time_class"])
              .groupby("time_class")["my_rating"].last().dropna()
        )
        if len(last):
            out["current_tc"]     = last.idxmax()
            out["current_rating"] = int(last[out["current_tc"]])
    if "my_accuracy" in df.columns:
        acc = df["my_accuracy"].dropna()
        if len(acc):
            out["avg_accuracy"]      = round(acc.mean(), 1)
            out["accuracy_coverage"] = round(len(acc) / len(df) * 100)
    return out


@st.cache_data(show_spinner=False)
def compute_opening_stats(key: str, df: pd.DataFrame, min_games: int):
    if "opening_label" not in df.columns or "is_win" not in df.columns:
        return None
    g = df.groupby("opening_label").agg(
        games=("is_win", "count"),
        win_rate=("is_win", "mean"),
    ).reset_index()
    g = g[g["games"] >= min_games].copy()
    # Drop bare ECO codes (e.g. "B12") and placeholder labels — not useful to display
    bad = {"Unknown", "Undefined", ""}
    g = g[~g["opening_label"].isin(bad)]
    g = g[~g["opening_label"].str.match(r"^[A-E]\d{2}$", na=False)]
    g["win_pct"] = (g["win_rate"] * 100).round(1)
    return g


@st.cache_data(show_spinner=False)
def compute_heatmap(key: str, df: pd.DataFrame):
    if not {"hour", "day_of_week"}.issubset(df.columns):
        return None
    g = df.groupby(["day_of_week", "hour"]).size().reset_index(name="games")
    g["day_short"] = pd.Categorical(
        g["day_of_week"].str[:3],
        categories=[d[:3] for d in DOW_ORDER],
        ordered=True,
    )
    return g


@st.cache_data(show_spinner=False)
def compute_clusters(key: str, df: pd.DataFrame):
    if not {"tc_secs", "hour"}.issubset(df.columns):
        return None
    feat = df[["tc_secs", "hour"]].dropna().copy()
    if len(feat) < _N_CLUSTERS_FIT * 20:
        return None

    X      = StandardScaler().fit_transform(feat)
    labels = KMeans(n_clusters=_N_CLUSTERS_FIT, random_state=42, n_init="auto").fit_predict(X)
    feat   = feat.copy()
    feat["cluster"] = labels

    for col in ["is_win", "time_class", "opening_label", "result_cat", "rating_diff", "my_accuracy"]:
        if col in df.columns:
            feat[col] = df.loc[feat.index, col].values

    agg_dict: dict = {
        "games":    ("tc_secs", "count"),
        "avg_tc":   ("tc_secs", "mean"),
        "avg_hour": ("hour",    "mean"),
    }
    if "is_win" in feat.columns:
        agg_dict["win_rate"] = ("is_win", "mean")
    if "my_accuracy" in feat.columns:
        agg_dict["avg_accuracy"] = ("my_accuracy", "mean")

    agg = feat.groupby("cluster").agg(**agg_dict).sort_values("games", ascending=False)

    def _label(row):
        tc = row["avg_tc"]
        hr = int(round(row["avg_hour"])) % 24
        tc_s = "Rapid" if tc >= 300 else ("Blitz" if tc >= 120 else "Bullet")
        hr_s = (
            "Night"     if (hr >= 21 or hr < 5) else
            "Morning"   if hr < 12 else
            "Afternoon" if hr < 17 else
            "Evening"
        )
        if hr == 0:    hr_fmt = "12 AM"
        elif hr < 12:  hr_fmt = f"{hr} AM"
        elif hr == 12: hr_fmt = "12 PM"
        else:          hr_fmt = f"{hr - 12} PM"
        return f"{tc_s} · {hr_s} · {hr_fmt}"

    agg["label"] = agg.apply(_label, axis=1)
    # Rapid clusters (≥300s) are too sparse to be interesting here
    agg = agg[agg["avg_tc"] < 300].head(N_CLUSTERS)
    feat["cluster_label"] = feat["cluster"].map(agg["label"])
    return feat, agg

# main
def main():
    st.set_page_config(page_title="Chess Analytics", layout="wide", page_icon="♟")
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="font-family:\'Press Start 2P\',monospace;font-size:10px;'
            'color:#2ec4b6;margin-bottom:4px;line-height:1.8;">♟ Chess Analytics</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        username = st.text_input("Username", value="belindafails").strip()
        st.divider()

    csv_path = get_csv_path(username)
    if not csv_path.exists():
        st.info(
            f"No processed CSV found for **{username}**.\n\n"
            "```bash\npython scripts/download_data.py\npython scripts/process_data.py\n```"
        )
        return

    mtime = csv_path.stat().st_mtime_ns
    df    = load_data(str(csv_path), mtime)
    if df is None or len(df) == 0:
        st.warning("CSV loaded but contains no rows.")
        return

    data_key = f"{username}:{mtime}"

    with st.sidebar:
        tc_opts = sorted(df["time_class"].dropna().unique()) if "time_class" in df.columns else []
        sel_tc  = st.multiselect("Time control", tc_opts, default=tc_opts)

        years = sorted(df["year"].dropna().unique().astype(int)) if "year" in df.columns else [2016, 2026]
        year_range = (
            st.slider("Year range", years[0], years[-1], (years[0], years[-1]))
            if len(years) >= 2 else (years[0], years[-1])
        )
        st.divider()
        st.caption(f"{len(df):,} total games loaded")

    fdf = df.copy()
    if sel_tc and "time_class" in fdf.columns:
        fdf = fdf[fdf["time_class"].isin(sel_tc)]
    if "year" in fdf.columns:
        fdf = fdf[fdf["year"].between(*year_range)]
    if len(fdf) == 0:
        st.warning("No games match the current filters.")
        return

    fkey = f"{data_key}:{tuple(sel_tc)}:{year_range}"

    df_years = sorted(df["year"].dropna().unique().astype(int)) if "year" in df.columns else []
    span     = df_years[-1] - df_years[0] if len(df_years) >= 2 else 1
    k        = compute_kpis(data_key, df)

    st.markdown(hero_html(username, span, k), unsafe_allow_html=True)

    st.markdown(stat_row_html([
        (fmt_k(k["total"]),                                           "total games",   None),
        (str(k["peak_rating"])    if "peak_rating"    in k else "—", "peak rating",   k.get("peak_tc", "").title() or None),
        (str(k["current_rating"]) if "current_rating" in k else "—", "current best",  k.get("current_tc", "").title() or None),
        (f'{k["avg_accuracy"]}%'  if "avg_accuracy"   in k else "—", "avg accuracy",  None),
    ]), unsafe_allow_html=True)

    t_over, t_habit, t_journey = st.tabs(["Overview", "Habits", "Journey"])

    api_stats = fetch_player_stats(username)

    # overview
    with t_over:
        col_ins, col_chart = st.columns([1, 2.2])

        with col_ins:
            ic_rows = []
            if "day_of_week" in fdf.columns:
                ic_rows.append(("Most active day",
                                fdf["day_of_week"].value_counts().index[0]))
            if "hour" in fdf.columns:
                ic_rows.append(("Peak hour",
                                hour_label(int(fdf["hour"].value_counts().index[0]))))
            if "time_class" in fdf.columns:
                tc_vc = fdf["time_class"].dropna().value_counts()
                ic_rows.append(("Favourite format",
                                f'{tc_vc.index[0].title()} · {fmt_k(tc_vc.iloc[0])} games'))
            if "hour" in fdf.columns:
                late = fdf["hour"].isin(list(range(22, 24)) + list(range(0, 5))).mean()
                ic_rows.append(("Late-night games", f'{late * 100:.1f}%  (10 PM – 5 AM)'))
            if "is_win" in fdf.columns and "opponent_rating" in fdf.columns:
                wins = fdf[fdf["is_win"] == 1].dropna(subset=["opponent_rating"])
                if len(wins):
                    top = wins.loc[wins["opponent_rating"].idxmax()]
                    ic_rows.append(("Biggest scalp",
                                    f'{top.get("opponent_username", "?")} ({int(top["opponent_rating"]):,})'))
            if ic_rows:
                st.markdown(insight_cards_html(ic_rows), unsafe_allow_html=True)

        with col_chart:
            st.markdown(section_html("Rating journey", "Bullet · Blitz · Rapid"),
                        unsafe_allow_html=True)
            if "end_date" in fdf.columns and "my_rating" in fdf.columns and "time_class" in fdf.columns:
                plot = (
                    fdf[fdf["time_class"].isin(["bullet", "blitz", "rapid"])]
                    .dropna(subset=["end_date", "my_rating"])
                )
                if len(plot):
                    chart = (
                        alt.Chart(plot)
                        .mark_line(strokeWidth=2, opacity=0.9)
                        .encode(
                            x=alt.X("end_date:T", title=None),
                            y=alt.Y("my_rating:Q", title="Rating",
                                    scale=alt.Scale(zero=False)),
                            color=alt.Color("time_class:N", title="Format",
                                            scale=alt.Scale(
                                                domain=["bullet", "blitz", "rapid"],
                                                range=["#2ec4b6", "#f4a261", "#3B82F6"],
                                            )),
                            tooltip=["end_date:T", "my_rating:Q", "time_class:N"],
                        )
                        .properties(height=360)
                    )
                    st.altair_chart(chart, use_container_width=True)

        if api_stats:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown(section_html("Ratings by format",
                                     "Live from Chess.com API · Best and current rating with all-time record"),
                        unsafe_allow_html=True)
            st.markdown(api_stats_html(api_stats), unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if "month" in fdf.columns:
            st.markdown(section_html("Volume over time", "Games per month"), unsafe_allow_html=True)
            monthly = fdf["month"].value_counts().sort_index().reset_index()
            monthly.columns = ["month", "games"]
            all_months = monthly["month"].tolist()
            tick_vals  = all_months[::6]
            bar = (
                alt.Chart(monthly)
                .mark_bar(color=ACCENT, opacity=0.8,
                          cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                .encode(
                    x=alt.X("month:O", title=None,
                             axis=alt.Axis(labelAngle=-45, labelFontSize=9, values=tick_vals)),
                    y=alt.Y("games:Q", title="Games"),
                    tooltip=["month:O", "games:Q"],
                )
                .properties(height=180)
            )
            st.altair_chart(bar, use_container_width=True)

    # habits
    with t_habit:
        st.markdown(
            section_html(
                "Play style clusters",
                f"KMeans (k={N_CLUSTERS}) on time-control length and hour · each cluster = a distinct play context",
            ),
            unsafe_allow_html=True,
        )
        result = compute_clusters(data_key, df)
        if result is None:
            st.info("Not enough data to cluster.")
        else:
            cdf, agg = result
            if "is_win" in cdf.columns:
                wr_map = cdf.groupby("cluster")["is_win"].mean()
                agg["win_rate"] = agg.index.map(lambda x: wr_map.get(x, np.nan))
            if "my_accuracy" in cdf.columns:
                acc_map = cdf.groupby("cluster")["my_accuracy"].mean()
                agg["avg_accuracy"] = agg.index.map(lambda x: acc_map.get(x, np.nan))
            st.markdown(cluster_grid_html(agg), unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        hm = compute_heatmap(fkey, fdf)
        if hm is not None and len(hm):
            st.markdown(section_html("When do you play most?",
                                     "Games by day and hour"),
                        unsafe_allow_html=True)
            dow_short = [d[:3] for d in DOW_ORDER]
            heat = (
                alt.Chart(hm)
                .mark_rect(cornerRadius=3)
                .encode(
                    x=alt.X("hour:O", title=None,
                             axis=alt.Axis(
                                 labelAngle=-45, labelFontSize=9,
                                 labelExpr="datum.value === 0 ? '12 AM' : datum.value < 12 ? datum.value + ' AM' : datum.value === 12 ? '12 PM' : (datum.value - 12) + ' PM'",
                             )),
                    y=alt.Y("day_short:O", sort=dow_short, title=None),
                    color=alt.Color("games:Q", title="Games",
                                    scale=alt.Scale(range=["#d4f0f4", "#a0dce8", "#60c8d0", "#2ec4b6", "#1a8c85"])),
                    tooltip=[
                        alt.Tooltip("day_of_week:N", title="Day"),
                        alt.Tooltip("hour:O",        title="Hour"),
                        alt.Tooltip("games:Q",       title="Games"),
                    ],
                )
                .properties(height=210)
            )
            st.altair_chart(heat, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(section_html("Games by hour"), unsafe_allow_html=True)
            if "hour" in fdf.columns:
                hc = fdf["hour"].value_counts().sort_index().reset_index()
                hc.columns = ["hour", "games"]
                all_h = pd.DataFrame({"hour": range(24)})
                hc    = all_h.merge(hc, on="hour", how="left").fillna(0)
                hc["games"]      = hc["games"].astype(int)
                hc["hour_label"] = hc["hour"].apply(lambda h: hour_label(int(h)))
                sort_order       = hc["hour_label"].tolist()
                bar = (
                    alt.Chart(hc)
                    .mark_bar(color=ACCENT, opacity=0.8,
                              cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                    .encode(
                        x=alt.X("hour_label:O", sort=sort_order, title=None,
                                 axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
                        y=alt.Y("games:Q", title="Games"),
                        tooltip=[alt.Tooltip("hour_label:O", title="Time"), "games:Q"],
                    )
                    .properties(height=190)
                )
                st.altair_chart(bar, use_container_width=True)

        with col2:
            st.markdown(section_html("Games by day"), unsafe_allow_html=True)
            if "day_of_week" in fdf.columns:
                dc = (
                    fdf["day_of_week"].value_counts()
                    .reindex(DOW_ORDER).fillna(0).reset_index()
                )
                dc.columns = ["day", "games"]
                bar = (
                    alt.Chart(dc)
                    .mark_bar(color=ACCENT, opacity=0.8,
                              cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                    .encode(
                        x=alt.X("day:O", sort=DOW_ORDER, title=None,
                                 axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
                        y=alt.Y("games:Q", title="Games"),
                        tooltip=["day:O", "games:Q"],
                    )
                    .properties(height=190)
                )
                st.altair_chart(bar, use_container_width=True)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.markdown(section_html("Games by year"), unsafe_allow_html=True)
        if "year" in fdf.columns:
            yc = fdf["year"].value_counts().sort_index().reset_index()
            yc.columns = ["year", "games"]
            bar = (
                alt.Chart(yc)
                .mark_bar(color=ACCENT, opacity=0.8,
                          cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                .encode(
                    x=alt.X("year:O", title=None,
                             axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
                    y=alt.Y("games:Q", title="Games"),
                    tooltip=["year:O", "games:Q"],
                )
                .properties(height=170)
            )
            st.altair_chart(bar, use_container_width=True)

    # journey
    with t_journey:
        st.markdown(section_html("Chess Journey", "Milestones unlocked and levels ahead"), unsafe_allow_html=True)
        st.markdown(journey_html(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
