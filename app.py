"""
Bankruptcy Risk Analyzer
Run: streamlit run app.py
"""

import os
import re
from io import BytesIO
from datetime import datetime
from urllib.parse import quote_plus, quote

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import httpx

APP_BUILD = "summary-strict-v6-en"

# ---------------------------------------------------------
# Embedded API key configuration
# ---------------------------------------------------------
# Paste your parser-api.com key between the quotation marks below.
# If you prefer environment variables, leave this value unchanged and set
# PARSER_API_KEY or ARBITR_API_KEY in your terminal/hosting environment.
EMBEDDED_PARSER_API_KEY = "b546b1b13b909730fe3b8a21603c6843"

# ---------------------------------------------------------
# Google Drive model configuration
# ---------------------------------------------------------
# The model file is intentionally not stored in GitHub because it is too large.
# Upload bankruptcy_rf_bundle.pkl to Google Drive, open sharing access
# as "Anyone with the link can view", and paste the share link below.
GOOGLE_DRIVE_MODEL_URL = "https://drive.google.com/file/d/16J9zdNbusXg_3vIuowZeKS8cLlfEyco_/view?usp=drive_link"

# The model will be downloaded once and cached locally under this path.
MODEL_CACHE_DIR = "model_cache"
MODEL_CACHE_FILENAME = "bankruptcy_rf_bundle.pkl"


# ---------------------------------------------------------
# Page configuration and styling
# ---------------------------------------------------------

st.set_page_config(
    page_title="Bankruptcy Risk Analyzer",
    page_icon="B",
    layout="wide"
)

st.markdown(
    """
    <style>
        :root {
            --page-bg: #dcefe8;
            --sidebar-bg: #247b65;
            --sidebar-bg-dark: #1f6b58;
            --text: #1f2937;
            --muted: #6b7280;
            --line: #e5e7eb;
            --panel: #ffffff;
            --panel-soft: #f8fafc;
            --accent: #247b65;
            --accent-dark: #1f6b58;
            --low-bg: #f0fdf4;
            --medium-bg: #fffbeb;
            --high-bg: #fef2f2;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--page-bg);
            color: var(--text);
            font-family: "Inter", "Segoe UI", sans-serif;
        }

        [data-testid="stAppViewContainer"] > .main {
            background: var(--page-bg);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
            padding-left: 2.2rem;
            padding-right: 2.2rem;
        }

        section[data-testid="stSidebar"] {
            background: var(--sidebar-bg);
            border-right: none;
            box-shadow: 0 22px 45px rgba(31, 107, 88, 0.22);
        }

        section[data-testid="stSidebar"] > div {
            background: var(--sidebar-bg);
            padding-top: 2rem;
        }

        section[data-testid="stSidebar"] * {
            color: rgba(255, 255, 255, 0.94) !important;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            color: rgba(255, 255, 255, 0.86) !important;
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.24);
        }

        section[data-testid="stSidebar"] [data-testid="stAlert"] {
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.20);
            border-radius: 14px;
        }

        /* Sidebar navigation as dashboard menu items */
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 14px;
            transition: 0.15s ease;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(255, 255, 255, 0.16) !important;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: rgba(255, 255, 255, 0.22) !important;
            border-color: rgba(255, 255, 255, 0.34) !important;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
            display: none !important;
        }

        section[data-testid="stSidebar"] [role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 13px;
            padding: 0.48rem 0.62rem;
            margin-bottom: 0.35rem;
        }

        .dashboard-panel {
            background: #ffffff;
            border-radius: 28px;
            padding: 1.65rem 1.8rem 2rem 1.8rem;
            box-shadow: 0 24px 55px rgba(15, 23, 42, 0.13);
            border: 1px solid rgba(255, 255, 255, 0.75);
            margin-bottom: 1.25rem;
        }

        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.45rem;
        }

        .search-pill {
            min-width: 270px;
            max-width: 430px;
            background: #f3f6f5;
            border: 1px solid #e7ecea;
            border-radius: 999px;
            padding: 0.72rem 1rem;
            color: #9ca3af;
            font-size: 0.88rem;
        }

        .status-pill {
            background: var(--accent);
            color: #ffffff;
            border-radius: 999px;
            padding: 0.65rem 1rem;
            font-weight: 650;
            font-size: 0.85rem;
            box-shadow: 0 10px 24px rgba(36, 123, 101, 0.25);
            white-space: nowrap;
        }

        .app-title {
            font-size: 2.15rem;
            font-weight: 720;
            letter-spacing: -0.04em;
            margin-bottom: 0.25rem;
            color: var(--text);
        }

        .app-subtitle {
            color: var(--muted);
            margin-bottom: 1.4rem;
            max-width: 780px;
            line-height: 1.55;
        }

        .section-label {
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: #728077;
            border-bottom: 1px solid var(--line);
            padding-bottom: 0.55rem;
            margin-top: 1.1rem;
            margin-bottom: 1rem;
        }

        .risk-card {
            border: 1px solid #e7ecea;
            border-radius: 18px;
            padding: 1.25rem 1.35rem;
            background: #ffffff;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        }

        .risk-high {
            border-color: #fecaca;
            background: var(--high-bg);
        }

        .risk-medium {
            border-color: #fde68a;
            background: var(--medium-bg);
        }

        .risk-low {
            border-color: #bbf7d0;
            background: #f1fbf6;
        }

        .risk-label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.76rem;
            margin-bottom: 0.4rem;
        }

        .risk-value {
            font-size: 2.05rem;
            font-weight: 720;
            letter-spacing: -0.04em;
            color: var(--text);
        }

        .summary-box {
            border: 1px solid #e7ecea;
            border-radius: 18px;
            background: #ffffff;
            padding: 1.05rem 1.2rem;
            margin-bottom: 1rem;
            line-height: 1.6;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.045);
        }

        .insight {
            border-left: 4px solid #d1d5db;
            border-radius: 14px;
            padding: 0.72rem 0.9rem;
            margin-bottom: 0.72rem;
            color: var(--text);
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
        }

        .insight-positive { border-left-color: #16a34a; background: #f0fdf4; }
        .insight-warning { border-left-color: #d97706; background: #fffbeb; }
        .insight-negative { border-left-color: #dc2626; background: #fef2f2; }
        .insight-neutral { border-left-color: #9ca3af; background: #f9fafb; }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e7ecea;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 0.8rem;
        }

        [data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 720;
        }

        div[data-testid="stTabs"] button {
            color: #374151;
            font-weight: 500;
            padding-top: 0.7rem;
            padding-bottom: 0.7rem;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom-color: var(--accent) !important;
        }

        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
            border-radius: 14px;
            border: 1px solid var(--accent);
            background: var(--accent);
            color: #ffffff;
            font-weight: 650;
            min-height: 2.75rem;
            box-shadow: 0 10px 24px rgba(36, 123, 101, 0.18);
        }

        div[data-testid="stButton"] > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {
            border-color: var(--accent-dark);
            background: var(--accent-dark);
            color: #ffffff;
        }

        [data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid #e7ecea;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.04);
        }

        .stNumberInput input,
        .stTextInput input {
            border-radius: 12px;
        }

        details {
            border-radius: 14px !important;
            border-color: #e7ecea !important;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        .conclusion-card {
            border: 1px solid #e7ecea;
            border-radius: 22px;
            background: #ffffff;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1.1rem;
            line-height: 1.65;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.06);
        }

        .factor-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin-bottom: 1.35rem;
        }

        .factor-card {
            border-radius: 20px;
            padding: 1.05rem 1.15rem;
            border: 1px solid #e7ecea;
            background: #ffffff;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.055);
            min-height: 124px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .factor-card-positive {
            background: #f0fdf4;
            border-color: #bbf7d0;
        }

        .factor-card-warning {
            background: #fffbeb;
            border-color: #fde68a;
        }

        .factor-card-negative {
            background: #fef2f2;
            border-color: #fecaca;
        }

        .factor-card-neutral {
            background: #f9fafb;
            border-color: #e5e7eb;
        }

        .factor-title {
            color: #6b7280;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
            font-weight: 750;
        }

        .factor-main {
            color: var(--text);
            font-size: 1.05rem;
            line-height: 1.25;
            font-weight: 720;
            margin-bottom: 0.35rem;
        }

        .factor-text {
            color: #4b5563;
            font-size: 0.92rem;
            line-height: 1.42;
        }

        .metrics-spacer {
            height: 1.25rem;
        }


        /* Clean sidebar menu: list-style navigation, no outlined pills */
        section[data-testid="stSidebar"] [data-testid="stRadio"] {
            margin-top: 0.25rem;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
            gap: 0.15rem;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            min-height: 2.45rem;
            display: flex !important;
            align-items: center !important;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
            display: none !important;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:last-child {
            width: 100%;
            padding: 0.62rem 0.85rem 0.62rem 0.95rem;
            border-radius: 0 !important;
            color: rgba(255,255,255,0.76) !important;
            font-weight: 500;
            position: relative;
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:last-child::before {
            content: "◦";
            font-size: 1.25rem;
            line-height: 1;
            opacity: 0.85;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) > div:last-child {
            background: rgba(20, 74, 62, 0.72) !important;
            color: #ffffff !important;
            font-weight: 700;
            box-shadow: inset 3px 0 0 rgba(255,255,255,0.85);
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover > div:last-child {
            background: rgba(255,255,255,0.10) !important;
            color: #ffffff !important;
        }

        section[data-testid="stSidebar"] h4 {
            margin-top: 0.6rem !important;
            margin-bottom: 0.25rem !important;
        }

        .factor-card-streamlit {
            border-radius: 20px;
            padding: 1.05rem 1.15rem;
            border: 1px solid #e7ecea;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.055);
            min-height: 136px;
            margin-bottom: 0.85rem;
        }

        .factor-card-streamlit.positive {
            background: #f0fdf4;
            border-color: #bbf7d0;
        }

        .factor-card-streamlit.warning {
            background: #fffbeb;
            border-color: #fde68a;
        }

        .factor-card-streamlit.negative {
            background: #fef2f2;
            border-color: #fecaca;
        }

        .factor-card-streamlit.neutral {
            background: #f9fafb;
            border-color: #e5e7eb;
        }

        .factor-card-streamlit .factor-title {
            color: #6b7280;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
            font-weight: 750;
        }

        .factor-card-streamlit .factor-main {
            color: #1f2937;
            font-size: 1.03rem;
            line-height: 1.25;
            font-weight: 720;
            margin-bottom: 0.4rem;
        }

        .factor-card-streamlit .factor-text {
            color: #4b5563;
            font-size: 0.9rem;
            line-height: 1.42;
        }

        .trend-card {
            background: rgba(255,255,255,0.72);
            border: 1px solid #d9e8e1;
            border-radius: 18px;
            padding: 1rem 1.05rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        }

        .trend-title {
            font-size: 0.9rem;
            font-weight: 750;
            color: #1f2937;
            margin-bottom: 0.35rem;
        }

        .trend-subtitle {
            font-size: 0.78rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.7rem;
            font-weight: 700;
        }

        .year-strip {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
        }

        .year-chip {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 0.45rem 0.65rem;
            min-width: 82px;
        }

        .year-chip .chip-year {
            font-size: 0.72rem;
            color: #6b7280;
            margin-bottom: 0.18rem;
            font-weight: 700;
        }

        .year-chip .chip-value {
            font-size: 0.9rem;
            color: #1f2937;
            font-weight: 700;
        }

        .case-card {
            background: #ffffff;
            border: 1px solid #deebe5;
            border-radius: 20px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.85rem;
            min-height: 182px;
        }

        .case-topline {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: flex-start;
            margin-bottom: 0.75rem;
        }

        .case-number {
            font-size: 1.02rem;
            font-weight: 760;
            color: #1f2937;
            line-height: 1.25;
        }

        .case-date {
            font-size: 0.82rem;
            color: #6b7280;
            white-space: nowrap;
            font-weight: 600;
        }

        .case-pill-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 0.75rem;
        }

        .case-pill {
            border-radius: 999px;
            padding: 0.28rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid #d8e5df;
            background: #f8fafc;
            color: #304150;
        }

        .case-pill.role-respondent, .case-pill.sev-warning {
            background: #fff7ed;
            border-color: #fdba74;
            color: #9a3412;
        }

        .case-pill.role-plaintiff, .case-pill.sev-positive {
            background: #f0fdf4;
            border-color: #86efac;
            color: #166534;
        }

        .case-pill.sev-negative {
            background: #fef2f2;
            border-color: #fca5a5;
            color: #991b1b;
        }

        .case-court {
            font-size: 0.9rem;
            color: #475569;
            line-height: 1.45;
        }

        .case-link {
            display: inline-flex;
            margin-top: 0.75rem;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: #eefcf4;
            border: 1px solid #86efac;
            color: #166534 !important;
            font-weight: 750;
            text-decoration: none !important;
            font-size: 0.82rem;
        }

        .ratio-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 0.95rem 1.05rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
        }

        .ratio-card.positive { border-left: 5px solid #16a34a; }
        .ratio-card.warning { border-left: 5px solid #d97706; }
        .ratio-card.negative { border-left: 5px solid #dc2626; }
        .ratio-card.neutral { border-left: 5px solid #9ca3af; }
        .ratio-card.no-data { border-left: 5px solid #cbd5e1; }

        .ratio-top {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 0.4rem;
        }

        .ratio-name {
            font-weight: 760;
            color: #1f2937;
            line-height: 1.25;
        }

        .ratio-value {
            font-weight: 760;
            color: #1f2937;
            white-space: nowrap;
        }

        .ratio-badge {
            display: inline-flex;
            width: fit-content;
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            font-size: 0.75rem;
            font-weight: 750;
            margin-bottom: 0.45rem;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            color: #475569;
        }

        .ratio-text {
            color: #4b5563;
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .structured-conclusion {
            background: #ffffff;
            border: 1px solid #e7ecea;
            border-radius: 24px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.06);
        }
        .conclusion-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.9rem; margin-top: 0.85rem; }
        .conclusion-block { border: 1px solid #e5e7eb; border-radius: 18px; background: #f8fafc; padding: 0.95rem 1rem; min-height: 142px; }
        .conclusion-block.ml { background: #f1fbf6; border-color: #bbf7d0; min-height: 220px; }
        .ml-result-value { color: #1f2937; font-size: 2.25rem; line-height: 1.05; font-weight: 820; letter-spacing: -0.045em; margin: 0.15rem 0 0.35rem 0; }
        .ml-result-text { color: #374151; font-size: 1.03rem; line-height: 1.45; font-weight: 560; margin-bottom: 0.75rem; }
        .ml-risk-chart { margin-top: 0.65rem; padding-top: 0.55rem; border-top: 1px solid rgba(148,163,184,0.26); }
        .ml-risk-chart-title { color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.68rem; font-weight: 800; margin-bottom: 0.35rem; }
        .ml-risk-chart svg { width: 100%; height: 125px; display: block; }
        .conclusion-block.context { background: #f8fafc; border-color: #dbe4ee; }
        .conclusion-block.attention { background: #fffdf2; border-color: #fde68a; }
        .conclusion-block.recommendation { background: #fff7ed; border-color: #fdba74; }
        .conclusion-kicker { color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem; font-weight: 800; margin-bottom: 0.45rem; }
        .conclusion-headline { color: #1f2937; font-size: 1.05rem; line-height: 1.32; font-weight: 780; margin-bottom: 0.45rem; }
        .conclusion-text { color: #4b5563; font-size: 0.92rem; line-height: 1.48; }
        .conclusion-list { display: flex; flex-direction: column; gap: 0.45rem; margin-top: 0.35rem; }
        .conclusion-factor { padding: 0.45rem 0.55rem; border-radius: 12px; background: rgba(255,255,255,0.62); border: 1px solid rgba(148,163,184,0.20); color: #374151; font-size: 0.88rem; line-height: 1.35; }
        .conclusion-factor.negative { border-left: 4px solid #dc2626; }
        .conclusion-factor.warning { border-left: 4px solid #d97706; }
        .conclusion-factor.positive { border-left: 4px solid #16a34a; }
        .conclusion-factor.neutral { border-left: 4px solid #9ca3af; }
        .context-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.55rem; margin-top: 0.35rem; }
        .context-item { background: rgba(255,255,255,0.72); border: 1px solid rgba(148,163,184,0.20); border-radius: 12px; padding: 0.5rem 0.6rem; }
        .context-label { color: #6b7280; font-size: 0.70rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 800; }
        .context-value { color: #1f2937; font-size: 0.93rem; font-weight: 760; margin-top: 0.15rem; }
        .role-context-panel { background: #ffffff; border: 1px solid #deebe5; border-radius: 18px; padding: 0.85rem 1rem; margin: 0.65rem 0 0.55rem 0; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.045); }
        .role-context-title { color: #1f2937; font-weight: 780; font-size: 0.98rem; margin-bottom: 0.22rem; }
        .role-context-text { color: #6b7280; font-size: 0.86rem; line-height: 1.45; }
        .role-focus-note { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 14px; padding: 0.65rem 0.8rem; color: #4b5563; font-size: 0.88rem; line-height: 1.45; margin-bottom: 0.85rem; }
        .kad-share-panel { background: #ffffff; border: 1px solid #e7ecea; border-radius: 22px; padding: 1rem 1.05rem; box-shadow: 0 12px 28px rgba(15, 23, 42, 0.055); margin-bottom: 1.1rem; }
        .kad-share-title { color: #1f2937; font-size: 1rem; font-weight: 780; margin-bottom: 0.35rem; }
        .kad-share-text { color: #4b5563; font-size: 0.9rem; line-height: 1.5; }
        @media (max-width: 1000px) { .conclusion-grid { grid-template-columns: 1fr; } .context-grid { grid-template-columns: 1fr; } }

        .case-link:hover {
            background: #dcfce7;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .dashboard-panel {
                border-radius: 20px;
                padding: 1.1rem;
            }
            .topbar {
                flex-direction: column;
                align-items: stretch;
            }
            .search-pill {
                min-width: auto;
                width: 100%;
            }
            .factor-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Model input fields
# ---------------------------------------------------------

# Required inputs for the ML model
ML_RAW_FIELDS = [
    ("Revenue", "2110"),
    ("Себестоимость продаж", "2120"),
    ("Прибыль (убыток) от продажи", "2200"),
    ("Чистая прибыль (убыток)", "2400"),
    ("Активы всего", "1600"),
    ("Оборотные активы", "1200"),
    ("Капитал и резервы", "1300"),
    ("Краткосрочные обязательства", "1500"),
    ("Дебиторская задолженность", "1230"),
    ("Кредиторская задолженность", "1520"),
    ("Нераспределенная прибыль (непокрытый убыток)", "1370"),
]

# Additional inputs used by classical bankruptcy models.
# If unavailable, the application applies fallback rules described in the code.
CLASSICAL_OPTIONAL_FIELDS = [
    ("Долгосрочные обязательства", "1400"),
    ("Долгосрочные заёмные средства", "1410"),
    ("Краткосрочные заёмные средства", "1510"),
    ("Прибыль до налогообложения", "2300"),
    ("Проценты к уплате", "2330"),
]

RAW_FIELDS = ML_RAW_FIELDS + CLASSICAL_OPTIONAL_FIELDS

FINANCIAL_MODEL_RAW_FIELDS = [field for field, _ in ML_RAW_FIELDS] + [
    "Возраст компании, лет"
]

MODEL_RAW_FIELDS = FINANCIAL_MODEL_RAW_FIELDS + [
    "ИДО"
]


# English interface labels. Internal dataframe column names stay in Russian because
# the trained model bundle and FNS parser use these names. Uploaded files may use
# either the original Russian names or the English labels below.
FIELD_LABELS_EN = {
    "Год": "Reporting year",
    "Выручка": "Revenue",
    "Себестоимость продаж": "Cost of sales",
    "Прибыль (убыток) от продажи": "Profit/loss from sales",
    "Чистая прибыль (убыток)": "Net profit/loss",
    "Активы всего": "Total assets",
    "Оборотные активы": "Current assets",
    "Капитал и резервы": "Equity and reserves",
    "Краткосрочные обязательства": "Short-term liabilities",
    "Дебиторская задолженность": "Accounts receivable",
    "Кредиторская задолженность": "Accounts payable",
    "Нераспределенная прибыль (непокрытый убыток)": "Retained earnings / accumulated loss",
    "Долгосрочные обязательства": "Long-term liabilities",
    "Долгосрочные заёмные средства": "Long-term borrowings",
    "Краткосрочные заёмные средства": "Short-term borrowings",
    "Прибыль до налогообложения": "Profit before tax",
    "Проценты к уплате": "Interest expense",
    "Возраст компании, лет": "Company age, years",
    "ИДО": "Due diligence index",
    "Наименование компании": "Company name",
    "bankruptcy_probability": "Bankruptcy probability",
    "predicted_class": "Predicted class",
    "risk_category": "Risk category",
}
COLUMN_ALIASES_TO_INTERNAL = {value: key for key, value in FIELD_LABELS_EN.items()}
COLUMN_ALIASES_TO_INTERNAL.update({
    "Year": "Год",
    "Company": "Наименование компании",
    "Company name": "Наименование компании",
    "DDI": "ИДО",
    "IDO": "ИДО",
})
RISK_LABELS_EN = {"Высокий": "High", "Средний": "Medium", "Низкий": "Low", "high": "High", "medium": "Medium", "low": "Low"}
ROLE_LABELS_EN = {"Клиент": "Client", "Поставщик": "Supplier", "Client": "Client", "Supplier": "Supplier"}


def ensure_dual_language_columns(df):
    """Keep both internal Russian columns and English UI aliases available.

    The trained model and parsed FNS data may use Russian names, while some
    visual blocks use English labels after the interface translation. This
    helper prevents KeyError after switching the UI language.
    """
    df = df.copy()
    for internal_name, english_name in FIELD_LABELS_EN.items():
        if internal_name in df.columns and english_name not in df.columns:
            df[english_name] = df[internal_name]
        if english_name in df.columns and internal_name not in df.columns:
            df[internal_name] = df[english_name]
    return df

def field_label(field):
    return FIELD_LABELS_EN.get(field, field)

def display_risk(value):
    return RISK_LABELS_EN.get(str(value), str(value))

def display_role(value):
    return ROLE_LABELS_EN.get(str(value), str(value))

def display_df(df):
    if df is None:
        return df
    return df.rename(columns={col: FIELD_LABELS_EN.get(col, col) for col in df.columns})


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def safe_div(a, b):
    return np.where((b != 0) & (~pd.isna(b)), a / b, np.nan)


def normalize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\xa0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df.rename(columns={col: COLUMN_ALIASES_TO_INTERNAL.get(col, col) for col in df.columns})
    df = ensure_dual_language_columns(df)
    return df


def format_money(value):
    if pd.isna(value):
        return "—"
    return f"{value:,.0f}".replace(",", " ")


def risk_category(probability):
    if probability >= 0.66:
        return "High", "high"
    if probability >= 0.33:
        return "Medium", "medium"
    return "Low", "low"


@st.cache_resource

def risk_class_name(probability):
    if probability >= 0.66:
        return "risk-high"
    if probability >= 0.33:
        return "risk-medium"
    return "risk-low"

def _extract_google_drive_file_id(url):
    """Extract a Google Drive file id from the most common sharing URL formats."""
    if not url or url == "PASTE_GOOGLE_DRIVE_MODEL_LINK_HERE":
        return None

    patterns = [
        r"/file/d/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
        r"/uc\?export=download&id=([A-Za-z0-9_-]+)",
        r"/open\?id=([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _get_google_drive_confirm_token(response):
    """Return the confirmation token used by Google Drive for large-file downloads."""
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    text_response = response.text if response.text else ""
    match = re.search(r"confirm=([0-9A-Za-z_\-]+)", text_response)
    if match:
        return match.group(1)
    return None


def _download_model_from_google_drive(url, output_path):
    """Download a model bundle from Google Drive and save it to output_path."""
    if not url or url == "PASTE_GOOGLE_DRIVE_MODEL_LINK_HERE":
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_id = _extract_google_drive_file_id(url)

    with httpx.Client(follow_redirects=True, timeout=180.0) as client:
        if file_id:
            base_url = "https://drive.google.com/uc"
            response = client.get(base_url, params={"export": "download", "id": file_id})
            token = _get_google_drive_confirm_token(response)
            if token:
                response = client.get(base_url, params={"export": "download", "id": file_id, "confirm": token})
        else:
            response = client.get(url)

        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        first_bytes = response.content[:1000].lower()
        if "text/html" in content_type or b"<html" in first_bytes:
            raise RuntimeError(
                "Google Drive returned an HTML page instead of the model file. "
                "Check that the file is shared as 'Anyone with the link can view'."
            )

        with open(output_path, "wb") as file:
            file.write(response.content)

    return output_path


def load_bundle(file_obj=None, model_kind="resident"):
    if file_obj is not None:
        return _fix_bundle(joblib.load(file_obj))

    if model_kind == "financial_only":
        candidates = [
            "bankruptcy_rf_financial_only_bundle.pkl",
            "bankruptcy_rf_financial_only.pkl",
            "model_financial_only.pkl",
        ]
    else:
        candidates = ["bankruptcy_rf_bundle.pkl", "model.pkl"]

    for filename in candidates:
        if os.path.exists(filename):
            return _fix_bundle(joblib.load(filename))

    if model_kind != "financial_only":
        cached_model_path = os.path.join(MODEL_CACHE_DIR, MODEL_CACHE_FILENAME)
        if os.path.exists(cached_model_path):
            return _fix_bundle(joblib.load(cached_model_path))

        try:
            downloaded_path = _download_model_from_google_drive(GOOGLE_DRIVE_MODEL_URL, cached_model_path)
            if downloaded_path and os.path.exists(downloaded_path):
                return _fix_bundle(joblib.load(downloaded_path))
        except Exception as exc:
            st.error(f"Could not download the model bundle from Google Drive: {exc}")

    return None


def _fix_bundle(bundle):
    """Recreate SimpleImputer for compatibility between scikit-learn versions."""
    if bundle is None:
        return None
    try:
        from sklearn.impute import SimpleImputer

        old_imputer = bundle["imputer"]
        new_imputer = SimpleImputer(strategy=old_imputer.strategy)
        new_imputer.statistics_ = old_imputer.statistics_
        new_imputer.n_features_in_ = old_imputer.n_features_in_
        if hasattr(old_imputer, "feature_names_in_"):
            new_imputer.feature_names_in_ = old_imputer.feature_names_in_
        new_imputer._fit_dtype = np.dtype(np.float64)
        new_imputer._fill_dtype = np.dtype(np.float64)
        new_imputer.indicator_ = None
        bundle["imputer"] = new_imputer
    except Exception as exc:
        print(f"[WARN] Could not recreate imputer: {exc}")
    return bundle


# ---------------------------------------------------------
# Feature engineering and prediction
# ---------------------------------------------------------

def build_features(df_raw):
    df = normalize_columns(df_raw)

    for column in MODEL_RAW_FIELDS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["roa"] = safe_div(df.get("Чистая прибыль (убыток)", 0), df.get("Активы всего", 0))
    df["roe"] = safe_div(df.get("Чистая прибыль (убыток)", 0), df.get("Капитал и резервы", 0))
    df["profitability_of_sales"] = safe_div(df.get("Чистая прибыль (убыток)", 0), df.get("Revenue", 0))
    df["operating_margin"] = safe_div(df.get("Прибыль (убыток) от продажи", 0), df.get("Revenue", 0))
    df["current_ratio"] = safe_div(df.get("Оборотные активы", 0), df.get("Краткосрочные обязательства", 0))
    df["receivables_turnover"] = safe_div(df.get("Revenue", 0), df.get("Дебиторская задолженность", 0))
    df["payables_turnover"] = safe_div(df.get("Себестоимость продаж", 0), df.get("Кредиторская задолженность", 0))
    df["asset_turnover"] = safe_div(df.get("Revenue", 0), df.get("Активы всего", 0))
    df["autonomy_ratio"] = safe_div(df.get("Капитал и резервы", 0), df.get("Активы всего", 0))
    df["retained_earnings_to_revenue"] = safe_div(
        df.get("Нераспределенная прибыль (непокрытый убыток)", 0),
        df.get("Revenue", 0)
    )
    df["log_assets"] = np.where(
        df.get("Активы всего", 0) > 0,
        np.log(df.get("Активы всего", np.nan)),
        np.nan
    )
    df["company_age"] = df.get("Company age, years", pd.Series(dtype=float))
    if "ИДО" in df.columns:
        df["ИДО"] = pd.to_numeric(df["ИДО"], errors="coerce")

    return df


def prepare_model_matrix(df_raw, bundle):
    df_features = build_features(df_raw)
    model_features = bundle["features"]
    X = df_features[model_features].copy()

    for column in model_features:
        if column in bundle["winsor_bounds"]:
            lower, upper = bundle["winsor_bounds"][column]
            X[column] = X[column].clip(lower=lower, upper=upper)

    X_imputed = pd.DataFrame(
        bundle["imputer"].transform(X),
        columns=model_features,
        index=X.index
    )
    return df_features, X_imputed


def predict(df_raw, bundle, threshold):
    df_features, X_imputed = prepare_model_matrix(df_raw, bundle)
    probabilities = bundle["model"].predict_proba(X_imputed)[:, 1]

    result = df_features.copy()
    result["bankruptcy_probability"] = probabilities
    result["predicted_class"] = (probabilities >= threshold).astype(int)
    result["risk_category"] = [risk_category(prob)[0] for prob in probabilities]

    return result


# ---------------------------------------------------------
# Data insights
# ---------------------------------------------------------

def compute_ratios(row):
    d = row.to_dict() if hasattr(row, "to_dict") else dict(row)

    total_assets = d.get("Активы всего", 0) or 0
    current_assets = d.get("Оборотные активы", 0) or 0
    current_liabilities = d.get("Краткосрочные обязательства", 0) or 0
    equity = d.get("Капитал и резервы", 0) or 0
    revenue = d.get("Revenue", 0) or 0
    net_profit = d.get("Чистая прибыль (убыток)", 0) or 0
    cost_of_sales = d.get("Себестоимость продаж", 0) or 0
    receivables = d.get("Дебиторская задолженность", 0) or 0
    payables = d.get("Кредиторская задолженность", 0) or 0
    long_term_liabilities = d.get("Долгосрочные обязательства", 0) or 0
    long_term_borrowings = d.get("Долгосрочные заёмные средства", 0) or 0
    short_term_borrowings = d.get("Краткосрочные заёмные средства", 0) or 0
    profit_before_tax = d.get("Прибыль до налогообложения", net_profit) or net_profit
    interest_expense = abs(d.get("Проценты к уплате", 0) or 0)
    operating_profit = d.get("Прибыль (убыток) от продажи", 0) or 0
    retained_earnings = d.get("Нераспределенная прибыль (непокрытый убыток)", 0) or 0

    total_liabilities = long_term_liabilities + current_liabilities
    total_debt = long_term_borrowings + short_term_borrowings
    if total_debt == 0:
        total_debt = total_liabilities
    ebit = profit_before_tax + interest_expense
    working_capital = current_assets - current_liabilities
    gross_profit = revenue - abs(cost_of_sales) if revenue else 0
    # Quick ratio: (current_assets - inventories). Inventories not available directly,
    # approximate as current_assets minus receivables remainder; use conservative proxy
    quick_assets = receivables  # most conservative: only receivables + cash (cash unknown)

    def _div(a, b):
        if b and b != 0 and not pd.isna(b):
            return a / b
        return np.nan

    ratios = {
        "ROA": row.get("roa"),
        "ROE": row.get("roe"),
        "Profitability of sales": row.get("profitability_of_sales"),
        "Operating margin": row.get("operating_margin"),
        "Gross margin": _div(gross_profit, revenue) if revenue else np.nan,
        "Current ratio": row.get("current_ratio"),
        "Quick ratio (conservative)": _div(quick_assets, current_liabilities),
        "Receivables turnover": row.get("receivables_turnover"),
        "Payables turnover": row.get("payables_turnover"),
        "Asset turnover": row.get("asset_turnover"),
        "Autonomy ratio": row.get("autonomy_ratio"),
        "Retained earnings to revenue": row.get("retained_earnings_to_revenue"),
        "Interest coverage (EBIT)": _div(ebit, interest_expense) if interest_expense > 0 else np.nan,
        "Net debt / EBIT": _div(total_debt, ebit) if ebit > 0 else np.nan,
        "Working capital": working_capital if total_assets else np.nan,
        "Working capital / Assets": _div(working_capital, total_assets),
        "Receivables / Revenue": _div(receivables, revenue) if revenue else np.nan,
        "Payables / Cost of sales": _div(payables, abs(cost_of_sales)) if cost_of_sales else np.nan,
        "Retained earnings / Assets": _div(retained_earnings, total_assets),
    }
    return ratios


def make_insight(text, severity="neutral"):
    return {"text": text, "severity": severity}


def assess_coefficient(name, value):
    """Return an English assessment and interpretation for the latest-year ratio table."""
    if pd.isna(value):
        return "No data", "The ratio could not be calculated from the available data."

    if name == "ROA":
        if value < -0.10:
            return "Negative", "Assets generate a substantial loss; operating performance is materially weak."
        if value < 0:
            return "Negative", "Assets generate a loss, so the company does not cover the cost of the capital employed."
        if value < 0.02:
            return "Warning", "Return on assets is very low and leaves a limited buffer against shocks."
        if value < 0.05:
            return "Neutral", "Return on assets is moderate and may be acceptable in capital-intensive sectors."
        return "Positive", "Return on assets is strong; the company uses its asset base efficiently."

    if name == "ROE":
        if value < 0:
            return "Negative", "Equity generates a negative return, which signals erosion of shareholder capital."
        if value < 0.05:
            return "Warning", "Return on equity is weak relative to the risk borne by capital providers."
        return "Positive", "Return on equity is positive and supports the financial profile."

    if name == "Profitability of sales":
        if value < 0:
            return "Negative", "The company records a net loss relative to revenue."
        if value < 0.02:
            return "Warning", "Net margin is narrow; pressure on prices or costs may move the result into loss."
        return "Positive", "Sales generate a positive net result."

    if name == "Operating margin":
        if value < -0.15:
            return "Negative", "Core operations are deeply loss-making."
        if value < 0:
            return "Negative", "Revenue does not cover operating expenses."
        if value < 0.05:
            return "Warning", "Operating margin is low and sensitive to cost or price changes."
        return "Positive", "Core operations generate a positive operating result."

    if name == "Gross margin":
        if value < 0:
            return "Negative", "Cost of sales exceeds revenue."
        if value < 0.15:
            return "Warning", "Gross margin is low and leaves a limited contribution to overheads and finance costs."
        if value < 0.30:
            return "Neutral", "Gross margin is moderate."
        return "Positive", "Gross margin is strong."

    if name == "Current ratio":
        if value < 0.5:
            return "Negative", "Current assets cover less than half of short-term liabilities."
        if value < 1:
            return "Negative", "Current assets do not cover short-term liabilities."
        if value < 1.5:
            return "Warning", "Liquidity is positive but the safety buffer is limited."
        if value < 3:
            return "Positive", "Current assets cover short-term liabilities with a reasonable buffer."
        return "Warning", "Very high liquidity may indicate inefficient working-capital allocation."

    if name == "Quick ratio (conservative)":
        if value < 0.5:
            return "Negative", "Quick assets are critically insufficient relative to current obligations."
        if value < 1:
            return "Warning", "Quick liquidity is limited."
        return "Positive", "Quick assets cover current obligations under the conservative proxy."

    if name == "Receivables turnover":
        if value < 1:
            return "Warning", "Receivables collection appears slow and may exceed one year on average."
        if value < 4:
            return "Neutral", "Receivables turnover is moderate."
        return "Positive", "Receivables are converted into cash relatively quickly."

    if name == "Payables turnover":
        if value < 1:
            return "Warning", "Accounts payable are high relative to annual cost of sales."
        return "Neutral", "The ratio describes the speed of supplier payments."

    if name == "Asset turnover":
        if value < 0.3:
            return "Warning", "Assets generate a low level of revenue."
        if value < 1:
            return "Neutral", "Revenue is below the asset base, which may be normal for capital-intensive firms."
        return "Positive", "Assets are efficiently involved in revenue generation."

    if name == "Autonomy ratio":
        if value < 0:
            return "Negative", "Equity is negative; liabilities exceed assets."
        if value < 0.10:
            return "Negative", "The equity buffer is critically low."
        if value < 0.30:
            return "Warning", "Dependence on borrowed capital is elevated."
        if value < 0.50:
            return "Neutral", "The capital structure shows moderate leverage."
        return "Positive", "At least half of assets are financed by equity."

    if name == "Retained earnings to revenue":
        if value < 0:
            return "Negative", "Accumulated financial result is negative."
        return "Positive", "Accumulated financial result is positive."

    if name == "Interest coverage (EBIT)":
        if value < 1:
            return "Negative", "EBIT does not cover interest expense."
        if value < 1.5:
            return "Negative", "Interest coverage is critically weak."
        if value < 2.5:
            return "Warning", "Interest coverage is limited."
        if value < 5:
            return "Positive", "Interest coverage is sufficient."
        return "Positive", "Interest coverage is strong."

    if name == "Net debt / EBIT":
        if value < 0:
            return "Neutral", "The ratio requires context because EBIT or net debt may be negative."
        if value < 2:
            return "Positive", "Debt burden is low relative to operating profit."
        if value < 4:
            return "Neutral", "Debt burden is moderate."
        if value < 6:
            return "Warning", "Debt burden is elevated."
        return "Negative", "Debt burden is excessive relative to EBIT."

    if name == "Working capital":
        if value < 0:
            return "Negative", "Working capital is negative."
        return "Positive", "Working capital is positive."

    if name == "Working capital / Assets":
        if value < -0.10:
            return "Negative", "Working-capital deficit is material relative to assets."
        if value < 0:
            return "Warning", "Working capital is negative."
        if value < 0.10:
            return "Neutral", "Working-capital buffer is limited."
        return "Positive", "Working-capital buffer is sufficient relative to assets."

    if name == "Receivables / Revenue":
        if value > 0.50:
            return "Negative", "Receivables are very high relative to revenue."
        if value > 0.25:
            return "Warning", "Receivables are elevated relative to revenue."
        return "Neutral", "Receivables are broadly proportional to revenue."

    if name == "Payables / Cost of sales":
        if value > 0.50:
            return "Warning", "Payables are high relative to cost of sales."
        return "Neutral", "Payables are broadly proportional to cost of sales."

    if name == "Retained earnings / Assets":
        if value < -0.30:
            return "Negative", "Accumulated losses exceed 30% of assets."
        if value < 0:
            return "Warning", "The company has accumulated losses."
        if value < 0.10:
            return "Neutral", "Accumulated result is limited."
        return "Positive", "Accumulated result is strong."

    return "Neutral", "The ratio has been calculated."


def generate_ml_summary(df_result):
    ordered = df_result.sort_values("Год")
    latest = ordered.iloc[-1]
    probability = latest["bankruptcy_probability"]
    risk_text, _ = risk_category(probability)
    parts = [
        f"Based on the data for {int(latest['Год'])}, the ML model estimates the bankruptcy probability at "
        f"{probability:.1%}, corresponding to the {risk_text.lower()} risk category."
    ]

    if len(ordered) > 1:
        first = ordered.iloc[0]
        change = latest["bankruptcy_probability"] - first["bankruptcy_probability"]
        if change >= 0.10:
            parts.append(
                f"Over {int(first['Год'])}–{int(latest['Год'])}, the predicted probability increased by {change:.1%}."
            )
        elif change <= -0.10:
            parts.append(
                f"Over {int(first['Год'])}–{int(latest['Год'])}, the predicted probability decreased by {abs(change):.1%}."
            )
        else:
            parts.append(
                f"Over {int(first['Год'])}–{int(latest['Год'])}, the predicted probability remained broadly stable."
            )

    return " ".join(parts)


def generate_classical_summary_text(classical_results):
    summary = summarize_classical_models(classical_results)
    if summary["available"] == 0:
        return "Classical bankruptcy models could not be calculated due to insufficient input data."

    high_models = [name for name, (_, _, risk) in classical_results.items() if risk == "high"]
    medium_models = [name for name, (_, _, risk) in classical_results.items() if risk == "medium"]
    low_models = [name for name, (_, _, risk) in classical_results.items() if risk == "low"]

    parts = [
        f"{summary['available']} classical models were calculated: "
        f"{summary['low']} indicate low risk, "
        f"{summary['medium']} indicate an intermediate assessment, "
        f"and {summary['high']} indicate high risk."
    ]

    if high_models:
        parts.append("High risk was identified by: " + ", ".join(high_models) + ".")
    if medium_models:
        parts.append("Intermediate assessment was given by: " + ", ".join(medium_models) + ".")
    if low_models:
        parts.append("Low risk was shown by: " + ", ".join(low_models) + ".")

    return " ".join(parts)


def generate_insights(df_result):
    """Generate concise English rule-based diagnostics for the dashboard."""
    if df_result.empty:
        return {"Current position": [], "Dynamics": []}

    ordered = df_result.sort_values("Год")
    latest = ordered.iloc[-1]
    current = []
    dynamics = []

    probability = latest["bankruptcy_probability"]
    risk_text, risk_key = risk_category(probability)
    current.append(make_insight(
        f"ML model: bankruptcy probability is {probability:.1%}, which corresponds to the {risk_text.lower()} risk category.",
        "negative" if risk_key == "high" else "warning" if risk_key == "medium" else "positive"
    ))

    def val(key, default=np.nan):
        v = latest.get(key, default)
        return v if pd.notna(v) else default

    equity = val("Капитал и резервы", 0)
    total_assets = val("Активы всего", 0)
    autonomy = val("autonomy_ratio")
    current_ratio = val("current_ratio")
    revenue = val("Выручка", 0)
    net_profit = val("Чистая прибыль (убыток)", 0)
    operating_profit = val("Прибыль (убыток) от продажи", 0)
    operating_margin = val("operating_margin")
    roa = val("roa")
    receivables_turnover = val("receivables_turnover")
    payables_turnover = val("payables_turnover")
    interest_expense = abs(val("Проценты к уплате", 0))
    profit_before_tax = val("Прибыль до налогообложения", net_profit)
    ebit = profit_before_tax + interest_expense

    if pd.notna(autonomy):
        if equity < 0 and total_assets > 0:
            current.append(make_insight("Negative equity: liabilities exceed assets, which is a strong solvency warning.", "negative"))
        elif autonomy < 0.10:
            current.append(make_insight(f"Autonomy ratio is critically low ({autonomy:.2f}); the equity buffer is very limited.", "negative"))
        elif autonomy < 0.30:
            current.append(make_insight(f"Autonomy ratio is {autonomy:.2f}; dependence on external financing is elevated.", "warning"))
        elif autonomy >= 0.50:
            current.append(make_insight(f"Autonomy ratio is {autonomy:.2f}; the capital structure has a strong equity base.", "positive"))

    if pd.notna(current_ratio):
        if current_ratio < 1:
            current.append(make_insight(f"Current ratio is {current_ratio:.2f}; current assets do not cover short-term liabilities.", "negative"))
        elif current_ratio < 1.5:
            current.append(make_insight(f"Current ratio is {current_ratio:.2f}; liquidity buffer is limited.", "warning"))
        elif current_ratio <= 3:
            current.append(make_insight(f"Current ratio is {current_ratio:.2f}; short-term obligations are covered by current assets.", "positive"))
        else:
            current.append(make_insight(f"Current ratio is {current_ratio:.2f}; very high liquidity may reflect inefficient working-capital allocation.", "warning"))

    if pd.notna(roa):
        if roa < 0:
            current.append(make_insight(f"ROA is {roa:.1%}; assets generate a loss.", "negative"))
        elif roa < 0.02:
            current.append(make_insight(f"ROA is {roa:.1%}; profitability is weak.", "warning"))
        elif roa >= 0.05:
            current.append(make_insight(f"ROA is {roa:.1%}; assets generate a solid return.", "positive"))

    if pd.notna(operating_margin):
        if operating_margin < 0:
            current.append(make_insight(f"Operating margin is {operating_margin:.1%}; core operations are loss-making.", "negative"))
        elif operating_margin < 0.05:
            current.append(make_insight(f"Operating margin is {operating_margin:.1%}; the operating buffer is narrow.", "warning"))
        else:
            current.append(make_insight(f"Operating margin is {operating_margin:.1%}; core operations generate a positive result.", "positive"))

    if operating_profit > 0 and net_profit < 0:
        current.append(make_insight("Earnings quality signal: operating profit is positive while net result is negative, indicating pressure from financing or non-operating items.", "warning"))
    elif operating_profit < 0 and net_profit > 0:
        current.append(make_insight("Earnings quality signal: net profit is supported by non-operating items while core operations are loss-making.", "warning"))

    if interest_expense > 0:
        if ebit <= 0:
            current.append(make_insight("Debt service warning: EBIT is negative while interest expense is present.", "negative"))
        else:
            icr = ebit / interest_expense
            if icr < 1.5:
                current.append(make_insight(f"Interest coverage is {icr:.1f}x; debt service capacity is critically weak.", "negative"))
            elif icr < 2.5:
                current.append(make_insight(f"Interest coverage is {icr:.1f}x; the debt service buffer is limited.", "warning"))
            elif icr >= 5:
                current.append(make_insight(f"Interest coverage is {icr:.1f}x; debt service capacity is strong.", "positive"))

    if pd.notna(receivables_turnover) and receivables_turnover > 0:
        days = 365 / receivables_turnover
        if days > 180:
            current.append(make_insight(f"Average receivables collection period is {days:.0f} days; a large part of revenue is locked in receivables.", "negative"))
        elif days > 90:
            current.append(make_insight(f"Average receivables collection period is {days:.0f} days; cash conversion is slower than desirable.", "warning"))
        else:
            current.append(make_insight(f"Average receivables collection period is {days:.0f} days; cash conversion is relatively fast.", "positive"))

    if pd.notna(payables_turnover) and payables_turnover > 0:
        days = 365 / payables_turnover
        if days > 180:
            current.append(make_insight(f"Average payables payment period is {days:.0f} days; supplier payments may be delayed.", "warning"))

    if len(ordered) > 1:
        first = ordered.iloc[0]
        probability_change = latest["bankruptcy_probability"] - first["bankruptcy_probability"]
        first_year = int(first["Год"])
        latest_year = int(latest["Год"])
        if probability_change >= 0.10:
            dynamics.append(make_insight(f"Bankruptcy probability increased by {probability_change:.1%} over {first_year}–{latest_year}.", "negative"))
        elif probability_change <= -0.10:
            dynamics.append(make_insight(f"Bankruptcy probability decreased by {abs(probability_change):.1%} over {first_year}–{latest_year}.", "positive"))
        else:
            dynamics.append(make_insight(f"Bankruptcy probability remained broadly stable over {first_year}–{latest_year}.", "neutral"))

        first_revenue = first.get("Revenue", np.nan)
        last_revenue = latest.get("Revenue", np.nan)
        if pd.notna(first_revenue) and pd.notna(last_revenue) and first_revenue != 0:
            rev_change = (last_revenue - first_revenue) / abs(first_revenue)
            if rev_change > 0.10:
                dynamics.append(make_insight(f"Revenue increased by {rev_change:.1%} over the analysed period.", "positive"))
            elif rev_change < -0.10:
                dynamics.append(make_insight(f"Revenue decreased by {abs(rev_change):.1%} over the analysed period.", "warning"))

    if not current:
        current.append(make_insight("No material comments were generated for the latest period.", "neutral"))
    if len(ordered) > 1 and not dynamics:
        dynamics.append(make_insight("No material changes were identified across the key indicators.", "neutral"))

    return {"Current position": current, "Dynamics": dynamics}

# ---------------------------------------------------------
# Classical bankruptcy models and integrated conclusion
# ---------------------------------------------------------

def run_classical_models(row):
    """Calculate classical bankruptcy models for one annual observation."""
    d = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    results = {}

    total_assets = d.get("Активы всего", 0)
    current_assets = d.get("Оборотные активы", 0)
    current_liabilities = d.get("Краткосрочные обязательства", 0)
    equity = d.get("Капитал и резервы", 0)
    revenue = d.get("Revenue", 0)
    net_profit = d.get("Чистая прибыль (убыток)", 0)
    retained_earnings = d.get("Нераспределенная прибыль (непокрытый убыток)", 0)
    long_term_liabilities = d.get("Долгосрочные обязательства", 0)
    long_term_borrowings = d.get("Долгосрочные заёмные средства", 0)
    short_term_borrowings = d.get("Краткосрочные заёмные средства", 0)
    profit_before_tax = d.get("Прибыль до налогообложения", net_profit)
    interest_expense = abs(d.get("Проценты к уплате", 0))
    total_liabilities = long_term_liabilities + current_liabilities
    ebit = profit_before_tax + interest_expense

    # Altman Z' score for private companies
    if total_assets:
        working_capital = current_assets - current_liabilities
        debt_for_altman = long_term_borrowings + short_term_borrowings
        if debt_for_altman == 0:
            debt_for_altman = total_liabilities if total_liabilities else 1
        score = (
            0.717 * (working_capital / total_assets)
            + 0.847 * (retained_earnings / total_assets)
            + 3.107 * (ebit / total_assets)
            + 0.420 * (equity / debt_for_altman)
            + 0.998 * (revenue / total_assets)
        )
        if score > 2.90:
            results["Altman Z'"] = (score, "Safe zone", "low")
        elif score > 1.23:
            results["Altman Z'"] = (score, "Grey zone", "medium")
        else:
            results["Altman Z'"] = (score, "Distress zone", "high")

    # Taffler model
    if total_assets and current_liabilities and total_liabilities:
        score = (
            0.53 * (profit_before_tax / current_liabilities)
            + 0.13 * (current_assets / total_liabilities)
            + 0.18 * (current_liabilities / total_assets)
            + 0.16 * (revenue / total_assets)
        )
        if score > 0.30:
            results["Taffler"] = (score, "Low risk", "low")
        elif score > 0.20:
            results["Taffler"] = (score, "Moderate risk", "medium")
        else:
            results["Taffler"] = (score, "High risk", "high")

    # Springate model
    if total_assets:
        working_capital = current_assets - current_liabilities
        score = (
            1.03 * (working_capital / total_assets)
            + 3.07 * (ebit / total_assets)
            + 0.66 * (profit_before_tax / (current_liabilities or 1))
            + 0.40 * (revenue / total_assets)
        )
        if score > 0.862:
            results["Springate"] = (score, "Stable position", "low")
        else:
            results["Springate"] = (score, "Bankruptcy risk", "high")

    # Saifullin–Kadykov model
    if total_assets and revenue and equity:
        non_current_assets = total_assets - current_assets
        score = (
            2 * ((equity - non_current_assets) / (current_assets or 1))
            + 0.1 * (current_assets / (current_liabilities or 1))
            + 0.08 * (revenue / total_assets)
            + 0.45 * (net_profit / (revenue or 1))
            + net_profit / (equity or 1)
        )
        if score >= 1:
            results["Saifullin–Kadykov"] = (score, "Satisfactory", "low")
        elif score >= 0.5:
            results["Saifullin–Kadykov"] = (score, "Unstable", "medium")
        else:
            results["Saifullin–Kadykov"] = (score, "Unsatisfactory", "high")

    # Beaver diagnostics
    if total_assets and total_liabilities:
        score = 0
        ratio_np_tl = net_profit / (total_liabilities or 1)
        ratio_np_ta = net_profit / (total_assets or 1)
        ratio_tl_ta = total_liabilities / total_assets
        ratio_wc_ta = (current_assets - current_liabilities) / total_assets
        ratio_ca_cl = current_assets / (current_liabilities or 1)

        if ratio_np_tl < 0.17:
            score += 2
        elif ratio_np_tl < 0.40:
            score += 1

        if ratio_np_ta < -0.15:
            score += 2
        elif ratio_np_ta < 0.02:
            score += 1

        if ratio_tl_ta > 0.80:
            score += 2
        elif ratio_tl_ta > 0.50:
            score += 1

        if ratio_wc_ta < 0.06:
            score += 2
        elif ratio_wc_ta < 0.30:
            score += 1

        if ratio_ca_cl < 1:
            score += 2
        elif ratio_ca_cl < 2:
            score += 1

        if score <= 3:
            results["Beaver"] = (score, "Normal position", "low")
        elif score <= 6:
            results["Beaver"] = (score, "Approx. 5 years before bankruptcy", "medium")
        else:
            results["Beaver"] = (score, "Approx. 1 year before bankruptcy", "high")

    return results


def summarize_classical_models(classical_results):
    """Return a compact summary of classical model outcomes."""
    if not classical_results:
        return {
            "available": 0,
            "low": 0,
            "medium": 0,
            "high": 0,
            "majority": "unavailable",
            "text": "Classical models could not be calculated due to insufficient data."
        }

    risks = [risk for _, _, risk in classical_results.values()]
    counts = {
        "available": len(risks),
        "low": risks.count("low"),
        "medium": risks.count("medium"),
        "high": risks.count("high"),
    }

    if counts["high"] > max(counts["low"], counts["medium"]):
        majority = "high"
        text_summary = f"{counts['high']} of {counts['available']} classical models indicate elevated risk."
    elif counts["low"] > max(counts["high"], counts["medium"]):
        majority = "low"
        text_summary = f"{counts['low']} of {counts['available']} classical models indicate a stable financial position."
    elif counts["medium"] > max(counts["high"], counts["low"]):
        majority = "medium"
        text_summary = f"{counts['medium']} of {counts['available']} classical models give an intermediate assessment."
    else:
        majority = "mixed"
        text_summary = "Classical models provide a mixed assessment."

    counts["majority"] = majority
    counts["text"] = text_summary
    return counts


def generate_integrated_conclusion(company_name, df_result, classical_results):
    """Build a compact English integrated credit assessment conclusion."""
    if df_result.empty:
        return ""

    ordered = df_result.sort_values("Год")
    latest = ordered.iloc[-1]
    latest_year = int(latest["Год"])
    probability = latest["bankruptcy_probability"]
    risk_text, risk_key = risk_category(probability)
    classical_summary = summarize_classical_models(classical_results)

    def val(key, default=0):
        v = latest.get(key, default)
        return v if pd.notna(v) else default

    name = company_name or "the analysed company"
    current_ratio = latest.get("current_ratio", np.nan)
    autonomy = latest.get("autonomy_ratio", np.nan)
    roa = latest.get("roa", np.nan)
    operating_margin = latest.get("operating_margin", np.nan)
    equity = val("Капитал и резервы")
    total_assets = val("Активы всего")
    net_profit = val("Чистая прибыль (убыток)")

    paragraphs = [
        f"The integrated assessment of {name} for {latest_year} gives a bankruptcy probability of {probability:.1%}, "
        f"which corresponds to the {risk_text.lower()} risk category."
    ]

    capital_parts = []
    if equity < 0 and total_assets > 0:
        capital_parts.append("equity is negative, so liabilities exceed assets")
    elif pd.notna(autonomy):
        if autonomy < 0.10:
            capital_parts.append(f"autonomy ratio is critically low ({autonomy:.2f})")
        elif autonomy < 0.30:
            capital_parts.append(f"autonomy ratio is below the comfortable zone ({autonomy:.2f})")
        elif autonomy >= 0.50:
            capital_parts.append(f"autonomy ratio is strong ({autonomy:.2f})")
    if capital_parts:
        paragraphs.append("Capital structure: " + "; ".join(capital_parts) + ".")

    liquidity_parts = []
    if pd.notna(current_ratio):
        if current_ratio < 1:
            liquidity_parts.append(f"current ratio is {current_ratio:.2f}, indicating a short-term liquidity deficit")
        elif current_ratio < 1.5:
            liquidity_parts.append(f"current ratio is {current_ratio:.2f}, indicating a limited liquidity buffer")
        else:
            liquidity_parts.append(f"current ratio is {current_ratio:.2f}, indicating sufficient current-asset coverage")
    if liquidity_parts:
        paragraphs.append("Liquidity: " + "; ".join(liquidity_parts) + ".")

    performance_parts = []
    if pd.notna(roa):
        if roa < 0:
            performance_parts.append(f"ROA is negative ({roa:.1%})")
        elif roa < 0.02:
            performance_parts.append(f"ROA is weak ({roa:.1%})")
        else:
            performance_parts.append(f"ROA is positive ({roa:.1%})")
    if pd.notna(operating_margin):
        if operating_margin < 0:
            performance_parts.append(f"operating margin is negative ({operating_margin:.1%})")
        elif operating_margin >= 0.05:
            performance_parts.append(f"operating margin is positive ({operating_margin:.1%})")
    if net_profit < 0:
        performance_parts.append("net result is negative")
    if performance_parts:
        paragraphs.append("Profitability: " + "; ".join(performance_parts) + ".")

    if classical_summary.get("available"):
        paragraphs.append(
            f"Classical diagnostics include {classical_summary['available']} models: "
            f"{classical_summary['low']} low-risk, {classical_summary['medium']} intermediate, "
            f"and {classical_summary['high']} high-risk outcomes."
        )

    if len(ordered) > 1:
        first = ordered.iloc[0]
        change = probability - first["bankruptcy_probability"]
        if change >= 0.05:
            paragraphs.append(f"Dynamics: predicted bankruptcy probability increased by {change:.1%} over the analysed period.")
        elif change <= -0.05:
            paragraphs.append(f"Dynamics: predicted bankruptcy probability decreased by {abs(change):.1%} over the analysed period.")
        else:
            paragraphs.append("Dynamics: predicted bankruptcy probability remained broadly stable over the analysed period.")

    return "\n\n".join(paragraphs)


def classical_results_table(classical_results):
    if not classical_results:
        return pd.DataFrame(columns=["Model", "Score", "Zone", "Risk level"])
    rows = []
    for model_name, (score, zone, risk) in classical_results.items():
        rows.append({
            "Model": model_name,
            "Score": score,
            "Zone": zone,
            "Risk level": {"low": "Low", "medium": "Medium", "high": "High"}[risk]
        })
    return pd.DataFrame(rows)


def plot_classical_models(classical_results):
    if not classical_results:
        return None
    risk_counts = {
        "Low": sum(1 for _, _, risk in classical_results.values() if risk == "low"),
        "Medium": sum(1 for _, _, risk in classical_results.values() if risk == "medium"),
        "High": sum(1 for _, _, risk in classical_results.values() if risk == "high"),
    }
    fig = go.Figure(
        go.Bar(
            x=list(risk_counts.keys()),
            y=list(risk_counts.values()),
            text=list(risk_counts.values()),
            textposition="outside"
        )
    )
    fig.update_layout(
        title="Распределение результатов классических моделей по уровням риска",
        yaxis_title="Количество моделей",
        height=320,
        margin=dict(l=40, r=20, t=50, b=35),
        plot_bgcolor="white"
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#f3f4f6", dtick=1)
    return fig

def plot_probability_dynamics(df_result):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_result["Год"].astype(str),
            y=df_result["bankruptcy_probability"] * 100,
            mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=8),
            name="Bankruptcy probability"
        )
    )
    fig.add_hline(y=33, line_dash="dash", line_color="#9ca3af")
    fig.add_hline(y=66, line_dash="dash", line_color="#9ca3af")
    fig.update_layout(
        title="Вероятность банкротства по годам",
        yaxis_title="Вероятность, %",
        xaxis_title="Year",
        yaxis_range=[0, 100],
        height=340,
        margin=dict(l=40, r=20, t=50, b=35),
        plot_bgcolor="white"
    )
    years = df_result["Год"].astype(str).tolist()
    fig.update_xaxes(showgrid=False, type="category", tickmode="array", tickvals=years, ticktext=years)
    fig.update_yaxes(gridcolor="#f3f4f6")
    return fig


def plot_revenue_profit(df_result):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_result["Год"].astype(str),
            y=df_result["Revenue"] / 1000,
            name="Revenue"
        )
    )
    fig.add_trace(
        go.Bar(
            x=df_result["Год"].astype(str),
            y=df_result["Чистая прибыль (убыток)"] / 1000,
            name="Net profit"
        )
    )
    fig.update_layout(
        title="Revenue and net profit",
        yaxis_title="тыс. руб.",
        barmode="group",
        height=340,
        margin=dict(l=40, r=20, t=50, b=35),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.18)
    )
    years = df_result["Год"].astype(str).tolist()
    fig.update_xaxes(showgrid=False, type="category", tickmode="array", tickvals=years, ticktext=years)
    fig.update_yaxes(gridcolor="#f3f4f6")
    return fig


def plot_ratio_dynamics(df_result):
    selected = {
        "Current ratio": "current_ratio",
        "Autonomy ratio": "autonomy_ratio",
        "ROA": "roa",
        "Operating margin": "operating_margin",
    }

    fig = go.Figure()
    for label, column in selected.items():
        fig.add_trace(
            go.Scatter(
                x=df_result["Год"].astype(str),
                y=df_result[column],
                mode="lines+markers",
                name=label
            )
        )

    fig.update_layout(
        title="Динамика ключевых коэффициентов",
        yaxis_title="Значение",
        xaxis_title="Year",
        height=360,
        margin=dict(l=40, r=20, t=50, b=35),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.18)
    )
    years = df_result["Год"].astype(str).tolist()
    fig.update_xaxes(showgrid=False, type="category", tickmode="array", tickvals=years, ticktext=years)
    fig.update_yaxes(gridcolor="#f3f4f6")
    return fig


def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    output.seek(0)
    return output


# ---------------------------------------------------------
# Input processing
# ---------------------------------------------------------

def example_values_for_period(period_index, year):
    """Return sample values for quick interface testing."""
    examples = [
        {
            "Revenue": 1200000,
            "Себестоимость продаж": 850000,
            "Прибыль (убыток) от продажи": 180000,
            "Чистая прибыль (убыток)": 120000,
            "Активы всего": 1500000,
            "Оборотные активы": 700000,
            "Капитал и резервы": 650000,
            "Краткосрочные обязательства": 420000,
            "Дебиторская задолженность": 210000,
            "Кредиторская задолженность": 170000,
            "Нераспределенная прибыль (непокрытый убыток)": 300000,
            "Company age, years": 8,
            "ИДО": 4,
            "Долгосрочные обязательства": 300000,
            "Долгосрочные заёмные средства": 180000,
            "Краткосрочные заёмные средства": 120000,
            "Прибыль до налогообложения": 145000,
            "Проценты к уплате": 12000,
        },
        {
            "Revenue": 1100000,
            "Себестоимость продаж": 830000,
            "Прибыль (убыток) от продажи": 130000,
            "Чистая прибыль (убыток)": 75000,
            "Активы всего": 1520000,
            "Оборотные активы": 620000,
            "Капитал и резервы": 560000,
            "Краткосрочные обязательства": 510000,
            "Дебиторская задолженность": 235000,
            "Кредиторская задолженность": 210000,
            "Нераспределенная прибыль (непокрытый убыток)": 260000,
            "Company age, years": 9,
            "ИДО": 9,
            "Долгосрочные обязательства": 350000,
            "Долгосрочные заёмные средства": 220000,
            "Краткосрочные заёмные средства": 150000,
            "Прибыль до налогообложения": 90000,
            "Проценты к уплате": 18000,
        },
        {
            "Revenue": 900000,
            "Себестоимость продаж": 760000,
            "Прибыль (убыток) от продажи": 65000,
            "Чистая прибыль (убыток)": 15000,
            "Активы всего": 1480000,
            "Оборотные активы": 500000,
            "Капитал и резервы": 360000,
            "Краткосрочные обязательства": 650000,
            "Дебиторская задолженность": 280000,
            "Кредиторская задолженность": 290000,
            "Нераспределенная прибыль (непокрытый убыток)": 90000,
            "Company age, years": 10,
            "ИДО": 16,
            "Долгосрочные обязательства": 420000,
            "Долгосрочные заёмные средства": 260000,
            "Краткосрочные заёмные средства": 210000,
            "Прибыль до налогообложения": 25000,
            "Проценты к уплате": 24000,
        },
        {
            "Revenue": 760000,
            "Себестоимость продаж": 730000,
            "Прибыль (убыток) от продажи": 12000,
            "Чистая прибыль (убыток)": -35000,
            "Активы всего": 1450000,
            "Оборотные активы": 420000,
            "Капитал и резервы": 180000,
            "Краткосрочные обязательства": 760000,
            "Дебиторская задолженность": 315000,
            "Кредиторская задолженность": 360000,
            "Нераспределенная прибыль (непокрытый убыток)": -40000,
            "Company age, years": 11,
            "ИДО": 24,
            "Долгосрочные обязательства": 510000,
            "Долгосрочные заёмные средства": 320000,
            "Краткосрочные заёмные средства": 260000,
            "Прибыль до налогообложения": -28000,
            "Проценты к уплате": 32000,
        },
        {
            "Revenue": 650000,
            "Себестоимость продаж": 690000,
            "Прибыль (убыток) от продажи": -50000,
            "Чистая прибыль (убыток)": -110000,
            "Активы всего": 1400000,
            "Оборотные активы": 350000,
            "Капитал и резервы": -50000,
            "Краткосрочные обязательства": 850000,
            "Дебиторская задолженность": 340000,
            "Кредиторская задолженность": 430000,
            "Нераспределенная прибыль (непокрытый убыток)": -160000,
            "Company age, years": 12,
            "ИДО": 31,
            "Долгосрочные обязательства": 600000,
            "Долгосрочные заёмные средства": 360000,
            "Краткосрочные заёмные средства": 310000,
            "Прибыль до налогообложения": -95000,
            "Проценты к уплате": 39000,
        },
    ]
    values = examples[min(period_index, len(examples) - 1)].copy()
    values["Год"] = int(year)
    return values


def fill_example_data(period_index, year, include_ido=True):
    """Populate Streamlit session state for one reporting period."""
    values = example_values_for_period(period_index, year)
    for field, _ in ML_RAW_FIELDS:
        st.session_state[f"{period_index}_{year}_{field}"] = float(values[field])
    st.session_state[f"{period_index}_{year}_age"] = float(values["Company age, years"])
    if include_ido:
        st.session_state[f"{period_index}_{year}_ido"] = float(values["ИДО"])
    for field, _ in CLASSICAL_OPTIONAL_FIELDS:
        st.session_state[f"{period_index}_{year}_{field}"] = float(values[field])


def build_manual_input(include_ido=True):
    header_left, header_right = st.columns([1, 2])
    with header_left:
        number_of_years = st.selectbox(
            "Number of reporting years",
            [1, 2, 3, 4, 5],
            index=0
        )
    with header_right:
        st.caption(
            "For each block below, specify the actual reporting year. "
            "If several years are analysed, the model calculates probability separately for each year."
        )

    records = []
    tabs = (
        st.tabs([f"Period {i + 1}" for i in range(number_of_years)])
        if number_of_years > 1
        else [st.container()]
    )
    default_years = list(range(datetime.now().year - number_of_years + 1, datetime.now().year + 1))

    for index, container in enumerate(tabs):
        with container:
            st.markdown('<div class="section-label">Reporting period</div>', unsafe_allow_html=True)
            year = st.number_input(
                "Reporting year",
                min_value=2000,
                max_value=datetime.now().year,
                value=default_years[index],
                step=1,
                key=f"reporting_year_{index}"
            )

            if st.button(
                "Fill with sample data",
                key=f"fill_example_{index}_{year}",
                use_container_width=False
            ):
                fill_example_data(index, year, include_ido=include_ido)
                st.rerun()

            record = {"Год": int(year)}

            st.markdown('<div class="section-label">Required ML model inputs</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            for i, (field, code) in enumerate(ML_RAW_FIELDS):
                with cols[i % 3]:
                    record[field] = st.number_input(
                        f"{field_label(field)} ({code})",
                        value=0.0,
                        step=1000.0,
                        format="%.0f",
                        key=f"{index}_{year}_{field}"
                    )

            st.markdown('<div class="section-label">Company parameters</div>', unsafe_allow_html=True)
            if include_ido:
                age_col, ido_col = st.columns(2)
                with age_col:
                    record["Возраст компании, лет"] = st.number_input(
                        "Company age, years",
                        min_value=0.0,
                        max_value=200.0,
                        value=5.0,
                        key=f"{index}_{year}_age"
                    )
                with ido_col:
                    record["ИДО"] = st.number_input(
                        "Due diligence index",
                        min_value=0.0,
                        value=5.0,
                        step=1.0,
                        key=f"{index}_{year}_ido"
                    )
            else:
                record["Возраст компании, лет"] = st.number_input(
                    "Company age, years",
                    min_value=0.0,
                    max_value=200.0,
                    value=5.0,
                    key=f"{index}_{year}_age"
                )

            with st.expander("Additional inputs for classical models", expanded=False):
                extra_cols = st.columns(3)
                for i, (field, code) in enumerate(CLASSICAL_OPTIONAL_FIELDS):
                    with extra_cols[i % 3]:
                        record[field] = st.number_input(
                            f"{field_label(field)} ({code})",
                            value=0.0,
                            step=1000.0,
                            format="%.0f",
                            key=f"{index}_{year}_{field}"
                        )

            records.append(record)

    return pd.DataFrame(records)



def build_company_template(include_ido=True):
    """Template for one company with one or several annual observations."""
    required_fields = MODEL_RAW_FIELDS if include_ido else FINANCIAL_MODEL_RAW_FIELDS
    internal_columns = ["Год"] + required_fields + [field for field, _ in CLASSICAL_OPTIONAL_FIELDS]
    example = {
        "Год": datetime.now().year - 1,
        "Revenue": 1000000,
        "Себестоимость продаж": 700000,
        "Прибыль (убыток) от продажи": 150000,
        "Чистая прибыль (убыток)": 100000,
        "Активы всего": 1200000,
        "Оборотные активы": 600000,
        "Капитал и резервы": 500000,
        "Краткосрочные обязательства": 350000,
        "Дебиторская задолженность": 180000,
        "Кредиторская задолженность": 140000,
        "Нераспределенная прибыль (непокрытый убыток)": 220000,
        "Возраст компании, лет": 8,
        "ИДО": 5,
        "Долгосрочные обязательства": 250000,
        "Долгосрочные заёмные средства": 150000,
        "Краткосрочные заёмные средства": 100000,
        "Прибыль до налогообложения": 125000,
        "Проценты к уплате": 10000,
    }
    row = {field_label(column): example.get(column, "") for column in internal_columns}
    return pd.DataFrame([row])


def build_portfolio_template(include_ido=True):
    """Template for several companies; one row equals one company-year observation."""
    company_df = build_company_template(include_ido=include_ido)
    company_df.insert(0, "Company name", "Example LLC")
    return company_df


def render_file_requirements(template_type="company", include_ido=True):
    if template_type == "company":
        st.markdown('<div class="section-label">File structure for single-company analysis</div>', unsafe_allow_html=True)
        st.caption(
            "One row corresponds to one reporting year of one company. "
            "The file may contain one or several years for the same company."
        )
        template_df = build_company_template(include_ido=include_ido)
        filename = "company_analysis_template.xlsx" if include_ido else "nonresident_company_analysis_template.xlsx"
    else:
        st.markdown('<div class="section-label">File structure for batch prediction</div>', unsafe_allow_html=True)
        st.caption(
            "One row corresponds to one company in one reporting year. "
            "Use separate rows for different companies."
        )
        template_df = build_portfolio_template(include_ido=include_ido)
        filename = "portfolio_prediction_template.xlsx" if include_ido else "nonresident_portfolio_prediction_template.xlsx"

    rows = [
        {"Column": field_label("Год"), "Status": "Required", "Purpose": "Reporting year"},
        *[
            {"Column": field_label(field), "Status": "Required", "Purpose": "ML model input"}
            for field, _ in ML_RAW_FIELDS
        ],
        {"Column": field_label("Возраст компании, лет"), "Status": "Required", "Purpose": "ML model input"},
        *([{"Column": field_label("ИДО"), "Status": "Required", "Purpose": "Resident model behavioural input"}] if include_ido else []),
        *[
            {"Column": field_label(field), "Status": "Optional", "Purpose": "Classical models"}
            for field, _ in CLASSICAL_OPTIONAL_FIELDS
        ],
    ]

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("**Example row:**")
    st.dataframe(template_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Excel template",
        data=to_excel(template_df),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=False
    )


def read_uploaded_file(uploaded_file, required_fields=None):
    filename = uploaded_file.name.lower()
    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=";", encoding="cp1251")

    df = normalize_columns(df)

    if "Год" not in df.columns:
        raise ValueError("The file must contain the column 'Reporting year' or 'Год'.")

    if required_fields is None:
        required_fields = MODEL_RAW_FIELDS
    missing_columns = [column for column in required_fields if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "The file is missing required columns: " + ", ".join(field_label(c) for c in missing_columns)
        )

    return df


# ---------------------------------------------------------
# Application
# ---------------------------------------------------------


def render_conclusion_card(title, text):
    html_text = text.replace("\n\n", "<br><br>")
    st.markdown(
        f"""
        <div class="conclusion-card">
            <div class="section-label" style="margin-top:0;">{title}</div>
            {html_text}
        </div>
        """,
        unsafe_allow_html=True
    )



def _html_escape(text):
    text = "" if text is None else str(text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_ml_probability_svg(result_df):
    """Return a compact inline SVG for ML bankruptcy probability dynamics."""
    if result_df is None or result_df.empty or "bankruptcy_probability" not in result_df.columns:
        return ""

    df = result_df[["Год", "bankruptcy_probability"]].copy()
    df["Год"] = pd.to_numeric(df["Год"], errors="coerce")
    df["bankruptcy_probability"] = pd.to_numeric(df["bankruptcy_probability"], errors="coerce")
    df = df.dropna(subset=["Год", "bankruptcy_probability"]).sort_values("Год")
    if df.empty:
        return ""

    width, height = 520, 150
    left, right, top, bottom = 42, 18, 16, 32
    plot_w = width - left - right
    plot_h = height - top - bottom
    years = df["Год"].astype(int).astype(str).tolist()
    values = (df["bankruptcy_probability"].clip(0, 1) * 100).tolist()

    if len(values) == 1:
        xs = [left + plot_w / 2]
    else:
        xs = [left + i * plot_w / (len(values) - 1) for i in range(len(values))]
    ys = [top + plot_h - (v / 100) * plot_h for v in values]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    def y_for(level):
        return top + plot_h - (level / 100) * plot_h

    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="#247b65" />'
        f'<text x="{x:.1f}" y="{y - 9:.1f}" text-anchor="middle" font-size="10" font-weight="700" fill="#1f2937">{v:.1f}%</text>'
        for x, y, v in zip(xs, ys, values)
    )
    labels = "".join(
        f'<text x="{x:.1f}" y="{height - 9}" text-anchor="middle" font-size="10" fill="#6b7280">{_html_escape(year)}</text>'
        for x, year in zip(xs, years)
    )
    thresholds = "".join(
        f'<line x1="{left}" y1="{y_for(level):.1f}" x2="{width-right}" y2="{y_for(level):.1f}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 4" />'
        f'<text x="{left-8}" y="{y_for(level)+3:.1f}" text-anchor="end" font-size="9" fill="#9ca3af">{level}%</text>'
        for level in (33, 66)
    )

    # Keep this fragment on a single line. Streamlit Markdown may treat a multiline
    # SVG fragment inside a larger HTML block as preformatted text and print the
    # remaining HTML instead of rendering it.
    return (
        f'<div class="ml-risk-chart">'
        f'<div class="ml-risk-chart-title">ML risk dynamics</div>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="ML bankruptcy probability dynamics">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="rgba(255,255,255,0.42)" />'
        f'{thresholds}'
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width-right}" y2="{top + plot_h}" stroke="#d1d5db" stroke-width="1" />'
        f'<polyline points="{points}" fill="none" stroke="#247b65" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" />'
        f'{circles}'
        f'{labels}'
        f'</svg>'
        f'</div>'
    )


COUNTERPARTY_ROLES = ["Client", "Supplier"]


def get_counterparty_focus_text(counterparty_role):
    """Explain how the selected counterparty role changes the analytical focus."""
    if display_role(counterparty_role) == "Supplier":
        return (
            "For a supplier, the focus is on operating stability, liquidity, payables pressure, "
            "debt service capacity and legal claims. These factors indicate the risk of delivery disruption, "
            "payment delays and dependence on refinancing."
        )
    return (
        "For a client, the focus is on solvency, receivables quality, collection speed, "
        "cash flow, sales profitability and legal pressure. These factors indicate the risk of non-payment "
        "and deterioration of payment discipline."
    )


def render_counterparty_role_selector():
    """Render a compact role selector above the model input table."""
    saved_role = display_role(st.session_state.get("counterparty_role", "Client"))
    default_index = COUNTERPARTY_ROLES.index(saved_role) if saved_role in COUNTERPARTY_ROLES else 0
    st.markdown(
        """
        <div class="role-context-panel">
            <div class="role-context-title">Counterparty role in the analysis</div>
            <div class="role-context-text">The role does not change the ML probability itself. It changes the set of factors displayed in the summary and analytical blocks.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    role = st.radio(
        "Counterparty role",
        COUNTERPARTY_ROLES,
        index=default_index,
        horizontal=True,
        key="counterparty_role"
    )
    st.markdown(
        f'<div class="role-focus-note">{_html_escape(get_counterparty_focus_text(role))}</div>',
        unsafe_allow_html=True
    )
    return role


def _item_text_for_role(item):
    if isinstance(item, dict):
        return f"{item.get('title', '')} {item.get('text', '')}".lower()
    return str(item).lower()


def factor_is_globally_excluded(item):
    """Hide factors that should not appear anywhere in the service summary cards."""
    text = _item_text_for_role(item)
    excluded_terms = [
        "долговая нагрузка",
        "снижение стоимости активов",
        "обязательства / активы",
        "обязательства/активы",
        "total liabilities / assets",
        "liabilities / assets",
        "liabilities/assets",
        "debt / equity",
        "debt/equity",
        "роль ответчика",
        "статус фнс",
        "банкротные дела",
        "судебная активность",
        "ответчик",
    ]
    return any(term in text for term in excluded_terms)


def factor_is_allowed_for_risk(item, risk_key):
    """Keep summary cards consistent with the final ML conclusion."""
    if not isinstance(item, dict):
        return True
    if factor_is_globally_excluded(item):
        return False
    severity = item.get("severity", "neutral")
    if risk_key == "low":
        # A low-risk ML verdict should not be visually dominated by red cards.
        # Red factors remain available in detailed analytical tabs, but the summary
        # explains the low-risk conclusion through positive signals and yellow zones.
        return severity in {"positive", "warning", "neutral"}
    return True


def factor_is_excluded_for_role(item, counterparty_role):
    """Hard-exclude factors that are globally disabled or built for the opposite counterparty role."""
    if factor_is_globally_excluded(item):
        return True
    text = _item_text_for_role(item)
    if display_role(counterparty_role) == "Supplier":
        excluded_terms = [
            "дебитор", "инкассац", "клиентской базы", "revenue stuffing",
            "платежеспособность клиента", "покупател"
        ]
        return any(term in text for term in excluded_terms)
    if display_role(counterparty_role) == "Client":
        excluded_terms = [
            "срок оплаты кредитор", "оплаты кредитор", "кредиторская задолженность значительна",
            "расчёты с поставщиками", "расчеты с поставщиками", "поставщикам"
        ]
        return any(term in text for term in excluded_terms)
    return False


def factor_matches_counterparty_role(item, counterparty_role):
    """Soft-match factors that are especially useful for the selected counterparty role."""
    text = _item_text_for_role(item)
    common_terms = [
        "ml-модель", "банкрот", "судеб", "ответчик", "фнс", "статус", "чистый денежный поток",
        "ликвид", "кассов", "обязательств", "капитал", "актив", "техническая несостоятельность",
        "накопленный", "процент", "автоном"
    ]
    if any(term in text for term in common_terms):
        return True
    if display_role(counterparty_role) == "Supplier":
        supplier_terms = [
            "кредитор", "поставщик", "рабочий капитал", "current ratio", "debt", "equity",
            "обслуживание долга", "рефинансирован", "задерживает расч", "операционная устойчивость"
        ]
        return any(term in text for term in supplier_terms)
    if display_role(counterparty_role) == "Client":
        client_terms = [
            "дебитор", "инкассац", "выруч", "продаж", "марж", "прибыл", "roa", "roe",
            "оборот", "качество прибыли", "денежн", "расчетной дисциплины"
        ]
        return any(term in text for term in client_terms)
    return True


def get_severity_priority_for_risk(risk_key=None):
    """Align the order of factors in the summary with the final ML risk class."""
    if risk_key == "low":
        # For low bankruptcy risk, the summary should first explain what supports the conclusion,
        # then show yellow attention factors, and only after that residual negative signals.
        return {"positive": 0, "warning": 1, "neutral": 2, "negative": 3}
    if risk_key == "medium":
        # For medium risk, the dashboard should emphasize ambiguous yellow signals first.
        return {"warning": 0, "negative": 1, "positive": 2, "neutral": 3}
    # For high risk or unknown risk, keep the conservative credit-analysis order.
    return {"negative": 0, "warning": 1, "neutral": 2, "positive": 3}


def prioritize_items_for_role(items, counterparty_role, severity_priority=None):
    """Sort items by counterparty relevance and by severity order aligned with the ML conclusion."""
    if severity_priority is None:
        severity_priority = get_severity_priority_for_risk(None)
    if not counterparty_role:
        return sorted(
            list(items),
            key=lambda item: severity_priority.get(item.get("severity", "neutral"), 2) if isinstance(item, dict) else 2
        )

    primary, secondary, excluded = [], [], []
    for item in items:
        if factor_is_excluded_for_role(item, counterparty_role):
            excluded.append(item)
        elif factor_matches_counterparty_role(item, counterparty_role):
            primary.append(item)
        else:
            secondary.append(item)

    def sort_key(item):
        if isinstance(item, dict):
            return severity_priority.get(item.get("severity", "neutral"), 2)
        return 2

    return sorted(primary, key=sort_key) + sorted(secondary, key=sort_key) + sorted(excluded, key=sort_key)


def summarize_kad_case_shares(cases):
    """Return mutually exclusive KAD case buckets for a pie/donut chart."""
    buckets = {"Respondent": 0, "Claimant": 0, "Claimant and respondent": 0, "Undefined role": 0}
    for case in cases or []:
        role = str(case.get("role") or "")
        is_respondent = "Respondent" in role or "Ответчик" in role or "ответчик" in role
        is_plaintiff = "Claimant" in role or "Истец" in role
        if is_respondent and is_plaintiff:
            buckets["Claimant and respondent"] += 1
        elif is_respondent:
            buckets["Respondent"] += 1
        elif is_plaintiff:
            buckets["Claimant"] += 1
        else:
            buckets["Undefined role"] += 1
    return {key: value for key, value in buckets.items() if value > 0}


def create_kad_case_share_chart(cases, title="Case share by company role"):
    shares = summarize_kad_case_shares(cases)
    if not shares:
        return None
    return create_donut_chart(list(shares.keys()), list(shares.values()), title)


def render_kad_share_panel(kad_data, counterparty_role=None):
    """Bring the KAD Arbitr case-share pie chart into the summary page."""
    if not kad_data:
        return
    cases = kad_data.get("cases") or []
    if not cases:
        return
    respondent = sum(1 for c in cases if "Respondent" in str(c.get("role", "")) or "Ответчик" in str(c.get("role", "")) or "ответчик" in str(c.get("role", "")))
    plaintiff = sum(1 for c in cases if "Claimant" in str(c.get("role", "")) or "Истец" in str(c.get("role", "")))
    bankruptcy = sum(1 for c in cases if c.get("case_type_label") in {"Bankruptcy", "Банкротное"})
    role_phrase = "supplier" if display_role(counterparty_role) == "Supplier" else "client"
    chart = create_kad_case_share_chart(cases)
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(
            f"""
            <div class="kad-share-panel">
                <div class="kad-share-title">KAD Arbitr: case share and legal context</div>
                <div class="kad-share-text">
                    Total cases found: <b>{len(cases)}</b>.<br>
                    Respondent role: <b>{respondent}</b>; claimant role: <b>{plaintiff}</b>; bankruptcy cases: <b>{bankruptcy}</b>.<br><br>
                    For the assessment of a {role_phrase}, the share of respondent cases is especially relevant because it may indicate pressure from counterparties and creditors.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with right:
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})


def get_summary_factor_section_title(risk_key):
    if risk_key == "low":
        return "Positive factors and attention areas"
    if risk_key == "medium":
        return "Priority attention factors"
    return "Priority risk factors"


def build_summary_recommendation(latest, probability, classical_summary, summary_items):
    _, risk_key = risk_category(probability)
    negative_count = sum(1 for item in summary_items if item.get("severity") == "negative")
    warning_count = sum(1 for item in summary_items if item.get("severity") == "warning")
    if risk_key == "high" or negative_count >= 2:
        return "Enhanced monitoring is recommended: request updated statements, review liquidity, legal claims and refinancing sources."
    if risk_key == "medium" or warning_count >= 2 or classical_summary.get("high", 0) > 0:
        return "Targeted review of attention areas is recommended: liquidity, capitalization, cash-flow dynamics and legal pressure. Update the assessment when new statements become available."
    return "Standard monitoring is recommended. Reassessment is appropriate after material changes in revenue, profit, cash flow, court activity or liability structure."



def _is_missing_number(value):
    return value is None or pd.isna(value)


def _fmt_decimal(value, digits=2):
    if _is_missing_number(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_mln(value):
    if _is_missing_number(value):
        return "—"
    return f"{float(value):,.2f}".replace(",", " ")


def _latest_fns_report(fns_data):
    reports = (fns_data or {}).get("reports") or []
    if not reports:
        return None
    return sorted(reports, key=lambda x: str(x.get("period") or ""), reverse=True)[0]


def _previous_fns_report(fns_data):
    reports = (fns_data or {}).get("reports") or []
    if len(reports) < 2:
        return None
    ordered = sorted(reports, key=lambda x: str(x.get("period") or ""), reverse=True)
    return ordered[1]


def _coalesce_number(*values):
    for value in values:
        if not _is_missing_number(value):
            return float(value)
    return np.nan


def _summary_severity(severity, risk_key):
    """Keep the summary visually aligned with the final ML conclusion."""
    if risk_key == "low" and severity == "negative":
        return "warning"
    return severity


SUMMARY_ALLOWED_FACTOR_TITLES = {
    "Autonomy ratio",
    "Current liquidity",
    "Average receivables collection period",
    "Average payables payment period",
    "Net profit",
    "Net cash flow",
    "Revenue dynamics",
}


def is_allowed_summary_factor(item):
    """Hard allow-list for the Summary page factor cards.

    This prevents older generic insights from leaking into the Summary block: no KAD role
    card, no Debt/Equity card, no status-only card, and no liabilities/assets card.
    """
    if not isinstance(item, dict):
        return False
    title = str(item.get("title") or "").replace("\xa0", " ").strip()
    allowed = {str(x).replace("\xa0", " ").strip() for x in SUMMARY_ALLOWED_FACTOR_TITLES}
    if title not in allowed:
        return False
    return not factor_is_globally_excluded(item)


def _make_summary_item(title, text, severity, risk_key, order):
    return {
        "title": title,
        "text": text,
        "severity": _summary_severity(severity, risk_key),
        "_summary_order": order,
    }


def _safe_turnover_days(turnover):
    if _is_missing_number(turnover) or turnover <= 0:
        return np.nan
    return 365 / turnover


def _severity_autonomy(value):
    if _is_missing_number(value):
        return "neutral"
    if value < 0.10:
        return "negative"
    if value < 0.30:
        return "warning"
    return "positive"


def _severity_current_ratio(value):
    if _is_missing_number(value):
        return "neutral"
    if value < 1.0:
        return "negative"
    if value < 1.5:
        return "warning"
    if value <= 3.0:
        return "positive"
    return "warning"


def _severity_collection_days(days, counterparty_role):
    if _is_missing_number(days):
        return "neutral"
    if display_role(counterparty_role) == "Supplier":
        if days > 180:
            return "warning"
        if days > 90:
            return "warning"
        return "positive"
    if days > 180:
        return "negative"
    if days > 90:
        return "warning"
    return "positive"


def _severity_profit(value):
    if _is_missing_number(value):
        return "neutral"
    return "positive" if value >= 0 else "negative"


def _severity_cash_flow(value):
    if _is_missing_number(value):
        return "neutral"
    return "positive" if value >= 0 else "negative"


def _severity_revenue_change(value):
    if _is_missing_number(value):
        return "neutral"
    if value >= 10:
        return "positive"
    if value >= 0:
        return "positive"
    if value <= -10:
        return "negative"
    return "warning"


def build_core_summary_items(result_df, fns_data=None, max_items=6, counterparty_role=None, risk_key=None):
    """Build the Summary block from a fixed set of business-relevant factors."""
    if result_df is None or result_df.empty:
        return []

    ordered = result_df.copy()
    ordered["Год"] = pd.to_numeric(ordered["Год"], errors="coerce")
    ordered = ordered.dropna(subset=["Год"]).sort_values("Год")
    if ordered.empty:
        return []

    latest = ordered.iloc[-1]
    previous = ordered.iloc[-2] if len(ordered) > 1 else None
    latest_fns = _latest_fns_report(fns_data)

    items = []

    autonomy = _coalesce_number(
        latest.get("autonomy_ratio"),
        latest_fns.get("autonomy_ratio") if latest_fns else np.nan,
    )
    if not _is_missing_number(autonomy):
        text_value = f"Autonomy ratio: {_fmt_decimal(autonomy, 2)}. Equity covers approximately {autonomy * 100:.0f}% of assets."
    else:
        text_value = "There is not enough data to calculate the autonomy ratio."
    items.append(_make_summary_item("Autonomy ratio", text_value, _severity_autonomy(autonomy), risk_key, 1))

    current_ratio = _coalesce_number(
        latest.get("current_ratio"),
        latest_fns.get("current_ratio") if latest_fns else np.nan,
    )
    if not _is_missing_number(current_ratio):
        if current_ratio < 1:
            text_value = f"Current liquidity: {_fmt_decimal(current_ratio, 2)}. Current assets are below short-term liabilities."
        elif current_ratio < 1.5:
            text_value = f"Current liquidity: {_fmt_decimal(current_ratio, 2)}. Obligations are covered, with a limited safety buffer."
        elif current_ratio <= 3:
            text_value = f"Current liquidity: {_fmt_decimal(current_ratio, 2)}. Current assets cover short-term liabilities."
        else:
            text_value = f"Current liquidity: {_fmt_decimal(current_ratio, 2)}. Working capital may be concentrated in current assets."
    else:
        text_value = "There is not enough data to calculate current liquidity."
    items.append(_make_summary_item("Current liquidity", text_value, _severity_current_ratio(current_ratio), risk_key, 2))

    if display_role(counterparty_role) == "Supplier":
        days = _safe_turnover_days(latest.get("payables_turnover"))
        if not _is_missing_number(days):
            text_value = f"Average payables payment period: {days:.0f} days. The metric reflects the speed of settlements with suppliers."
        else:
            text_value = "There is not enough data to calculate the average payables payment period."
        items.append(_make_summary_item("Average payables payment period", text_value, _severity_collection_days(days, counterparty_role), risk_key, 3))
    else:
        days = _safe_turnover_days(latest.get("receivables_turnover"))
        if not _is_missing_number(days):
            text_value = f"Average receivables collection period: {days:.0f} days. The metric reflects how quickly sales are converted into cash."
        else:
            text_value = "There is not enough data to calculate the average receivables collection period."
        items.append(_make_summary_item("Average receivables collection period", text_value, _severity_collection_days(days, counterparty_role), risk_key, 3))

    profit_mln = _coalesce_number(
        latest_fns.get("profit_mln_rub") if latest_fns else np.nan,
        latest.get("Чистая прибыль (убыток)") / 1000 if not _is_missing_number(latest.get("Чистая прибыль (убыток)")) else np.nan,
    )
    if not _is_missing_number(profit_mln):
        text_value = f"Net result: {_fmt_mln(profit_mln)} million RUB."
    else:
        text_value = "Net profit was not found in the available data."
    items.append(_make_summary_item("Net profit", text_value, _severity_profit(profit_mln), risk_key, 4))

    net_cf_mln = _coalesce_number(latest_fns.get("net_cash_flow_mln_rub") if latest_fns else np.nan)
    if not _is_missing_number(net_cf_mln):
        if net_cf_mln >= 0:
            text_value = f"Net cash flow is positive: {_fmt_mln(net_cf_mln)} million RUB."
        else:
            text_value = f"Net cash flow is negative: {_fmt_mln(net_cf_mln)} million RUB."
    else:
        text_value = "Net cash flow was not found in the available statements."
    items.append(_make_summary_item("Net cash flow", text_value, _severity_cash_flow(net_cf_mln), risk_key, 5))

    revenue_change = _coalesce_number(latest_fns.get("revenue_change_pct") if latest_fns else np.nan)
    if _is_missing_number(revenue_change) and previous is not None:
        cur_rev = latest.get("Revenue")
        prev_rev = previous.get("Revenue")
        if not _is_missing_number(cur_rev) and not _is_missing_number(prev_rev) and prev_rev != 0:
            revenue_change = (cur_rev - prev_rev) / abs(prev_rev) * 100
    if not _is_missing_number(revenue_change):
        direction = "increased" if revenue_change >= 0 else "decreased"
        text_value = f"Revenue {direction} by {abs(revenue_change):.1f}% compared with the previous period."
    else:
        text_value = "There is not enough data to assess revenue dynamics."
    items.append(_make_summary_item("Revenue dynamics", text_value, _severity_revenue_change(revenue_change), risk_key, 6))

    items = [item for item in items if not factor_is_globally_excluded(item)]

    severity_priority = get_severity_priority_for_risk(risk_key)
    items = sorted(
        items,
        key=lambda item: (
            severity_priority.get(item.get("severity", "neutral"), 2),
            item.get("_summary_order", 99),
        )
    )

    clean_items = []
    for item in items[:max_items]:
        item = dict(item)
        item.pop("_summary_order", None)
        clean_items.append(item)
    return clean_items


def select_summary_items(insights=None, fns_data=None, kad_data=None, max_items=6, counterparty_role=None, risk_key=None, result_df=None):
    """Select factor cards for Summary from the fixed allowed set only.

    Important: this function deliberately ignores generic ML/FNS/KAD insight lists.
    Generic lists produced the old cards (debt load, Debt/Equity, respondent role,
    bankruptcy cases, FNS status). The Summary must only show the agreed factors:
    autonomy, current liquidity, receivables/payables days, net profit, net cash flow,
    and revenue dynamics.
    """
    items = build_core_summary_items(
        result_df=result_df,
        fns_data=fns_data,
        max_items=max_items,
        counterparty_role=counterparty_role,
        risk_key=risk_key,
    )
    clean = []
    for item in items:
        if is_allowed_summary_factor(item):
            clean.append(item)
    return clean[:max_items]

def render_structured_conclusion(company_name, result_df, classical_summary, summary_items, fns_data=None, kad_data=None, counterparty_role=None):
    ordered = result_df.sort_values("Год")
    latest = ordered.iloc[-1]
    latest_year = int(latest["Год"])
    probability = latest["bankruptcy_probability"]
    risk_text, risk_key = risk_category(probability)
    name = company_name or "the company"

    ml_headline = f"{probability:.1%} — {risk_text.lower()} risk"
    ml_chart_html = build_ml_probability_svg(ordered)
    if len(ordered) > 1:
        first = ordered.iloc[0]
        change = probability - first["bankruptcy_probability"]
        if change >= 0.05:
            ml_text = f"Probability increased by {change:.1%} over {int(first['Год'])}–{latest_year}."
        elif change <= -0.05:
            ml_text = f"Probability decreased by {abs(change):.1%} over {int(first['Год'])}–{latest_year}."
        else:
            ml_text = f"Probability remained broadly stable over {int(first['Год'])}–{latest_year}."
    else:
        ml_text = f"The estimate is based on data for {latest_year}."

    if risk_key == "low":
        factor_block_kicker = "Low-risk support"
        factor_block_headline = "Positive factors and attention areas"
        attention_items = [item for item in summary_items if item.get("severity") in ["positive", "warning"]]
        empty_factor_text = "No material positive factors or attention areas were identified from the current data."
    elif risk_key == "medium":
        factor_block_kicker = "Attention areas"
        factor_block_headline = "Factors requiring review"
        attention_items = [item for item in summary_items if item.get("severity") in ["warning", "negative", "positive"]]
        empty_factor_text = "No material attention areas were identified from the current data."
    else:
        factor_block_kicker = "Risk factors"
        factor_block_headline = "Negative factors and attention areas"
        attention_items = [item for item in summary_items if item.get("severity") in ["negative", "warning"]]
        empty_factor_text = "No material negative factors were identified from the current data."

    if not attention_items:
        attention_items = summary_items[:3]
    attention_items = attention_items[:3]
    if attention_items:
        attention_html = '<div class="conclusion-list">' + ''.join([
            f'<div class="conclusion-factor {item.get("severity", "neutral")}"><b>{_html_escape(item.get("title", "Factor"))}</b><br>{_html_escape(item.get("text", ""))}</div>'
            for item in attention_items
        ]) + '</div>'
    else:
        attention_html = f'<div class="conclusion-text">{_html_escape(empty_factor_text)}</div>'

    context_items = []
    if classical_summary.get("available", 0):
        context_items.append(("Classical models", f"{classical_summary['high']} high / {classical_summary['medium']} medium / {classical_summary['low']} low"))
    if fns_data:
        org = fns_data.get("organization") or {}
        reports = fns_data.get("reports") or []
        status = org.get("status") or "—"
        last_period = max([r.get("period") for r in reports if r.get("period") is not None], default="—")
        context_items.append(("FNS", f"{status}; statements through {last_period}"))
    if kad_data:
        cases = kad_data.get("cases") or []
        bankruptcies = sum(1 for c in cases if c.get("case_type_label") in {"Bankruptcy", "Банкротное"})
        context_items.append(("KAD Arbitr", f"cases: {len(cases)}, bankruptcy: {bankruptcies}"))
    if not context_items:
        context_items.append(("Source", "financial data and ML model"))

    context_html = '<div class="context-grid">' + ''.join([
        f'<div class="context-item"><div class="context-label">{_html_escape(label)}</div><div class="context-value">{_html_escape(value)}</div></div>'
        for label, value in context_items
    ]) + '</div>'

    role_focus = get_counterparty_focus_text(counterparty_role) if counterparty_role else "The assessment is based on financial, legal and external-source indicators without a separate counterparty role."
    role_headline = f"Focus: {display_role(counterparty_role).lower()}" if counterparty_role else "Assessment focus"

    st.markdown(f"""
        <div class="structured-conclusion">
            <div class="section-label" style="margin-top:0;">Integrated conclusion</div>
            <div class="conclusion-text" style="margin-bottom:0.8rem;">The analysis of {_html_escape(name)} is based on the latest available period: {latest_year}. The summary below shows the model result, attention areas, source context and selected counterparty focus.</div>
            <div class="conclusion-grid">
                <div class="conclusion-block ml">
                    <div class="conclusion-kicker">ML model result</div>
                    <div class="ml-result-value">{_html_escape(ml_headline)}</div>
                    <div class="ml-result-text">{_html_escape(ml_text)}</div>
                    {ml_chart_html}
                </div>
                <div class="conclusion-block attention">
                    <div class="conclusion-kicker">{_html_escape(factor_block_kicker)}</div>
                    <div class="conclusion-headline">{_html_escape(factor_block_headline)}</div>
                    {attention_html}
                </div>
                <div class="conclusion-block context">
                    <div class="conclusion-kicker">Assessment context</div>
                    <div class="conclusion-headline">Inputs used in the analysis</div>
                    {context_html}
                </div>
                <div class="conclusion-block recommendation">
                    <div class="conclusion-kicker">Counterparty focus</div>
                    <div class="conclusion-headline">{_html_escape(role_headline)}</div>
                    <div class="conclusion-text">{_html_escape(role_focus)}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def split_factor_text(text):
    """Split a long rule-based comment into compact dashboard title and explanation."""
    if ":" in text:
        main, detail = text.split(":", 1)
        return main.strip(), detail.strip()
    if "." in text:
        main, detail = text.split(".", 1)
        return main.strip(), detail.strip()
    return text.strip(), ""


def split_factor_text(text):
    """Split a long rule-based comment into compact dashboard title and explanation."""
    if ":" in text:
        main, detail = text.split(":", 1)
        return main.strip(), detail.strip()
    if "." in text:
        main, detail = text.split(".", 1)
        return main.strip(), detail.strip()
    return text.strip(), ""


def render_factor_cards(insights, limit=6, counterparty_role=None):
    current = insights.get("Current position", insights.get("Текущее состояние", []))
    source_cards = []

    for item in current:
        text_value = item["text"] if isinstance(item, dict) else str(item)
        severity = item["severity"] if isinstance(item, dict) else "neutral"
        if "ML model" in text_value or "ML-модель" in text_value:
            continue
        source_cards.append({"text": text_value, "severity": severity})

    cards = prioritize_items_for_role(source_cards, counterparty_role)
    cards = [item for item in cards if not factor_is_excluded_for_role(item, counterparty_role)]

    if not cards:
        cards = [{"text": "No material negative factors were identified for the latest period.", "severity": "positive"}]

    cards = [(item.get("text", ""), item.get("severity", "neutral")) for item in cards[:limit]]
    title_map = {
        "positive": "Positive factor",
        "warning": "Attention area",
        "negative": "Risk factor",
        "neutral": "Neutral factor",
    }

    for row_start in range(0, len(cards), 2):
        cols = st.columns(2)
        row_cards = cards[row_start:row_start + 2]

        for col, (text_value, severity) in zip(cols, row_cards):
            main, detail = split_factor_text(text_value)
            with col:
                st.markdown(
                    f"""
                    <div class="factor-card-streamlit {severity}">
                        <div class="factor-title">{title_map.get(severity, "Factor")}</div>
                        <div class="factor-main">{main}</div>
                        <div class="factor-text">{detail}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def render_ratio_cards(latest_ratios):
    """Render latest-year coefficients as readable cards instead of a narrow table."""
    severity_map = {
        "Positive": "positive",
        "Warning": "warning",
        "Negative": "negative",
        "Neutral": "neutral",
        "No data": "no-data",
    }

    for i in range(0, len(latest_ratios), 2):
        cols = st.columns(2)
        for col, (_, row) in zip(cols, latest_ratios.iloc[i:i+2].iterrows()):
            assessment = str(row.get("Assessment", row.get("Оценка", "Neutral")))
            css_class = severity_map.get(assessment, "neutral")
            value = row.get("Value", row.get("Значение", np.nan))
            value_text = "—" if pd.isna(value) else f"{value:.4f}"
            ratio_name = row.get("Indicator", row.get("Показатель", ""))
            explanation = row.get("Interpretation", row.get("Расшифровка", ""))

            with col:
                st.markdown(
                    f"""
                    <div class="ratio-card {css_class}">
                        <div class="ratio-top">
                            <div class="ratio-name">{ratio_name}</div>
                            <div class="ratio-value">{value_text}</div>
                        </div>
                        <div class="ratio-badge">{assessment}</div>
                        <div class="ratio-text">{explanation}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def render_key_metrics(latest, latest_probability, latest_risk_text, latest_classical_summary):
    risk_class = risk_class_name(latest_probability)
    st.markdown('<div class="section-label">Key indicators</div>', unsafe_allow_html=True)
    left, right = st.columns([1.25, 2.4])
    with left:
        st.markdown(
            f"""
            <div class="risk-card {risk_class}">
                <div class="risk-label">Bankruptcy probability</div>
                <div class="risk-value">{latest_probability:.1%}</div>
                <div>Risk category: {latest_risk_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("Assessment year", int(latest["Год"]))
        c2.metric("Classical models", latest_classical_summary.get("available", 0))
        c3.metric("High-risk models", latest_classical_summary.get("high", 0))

    st.markdown('<div class="metrics-spacer"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# External sources: FNS and KAD Arbitr
# ---------------------------------------------------------
NALOG_BASE_URL = "https://parser-api.com/parser/nalog_bo_api"
ARBITR_BASE_URL = "https://parser-api.com/parser/arbitr_api"


def external_api_key():
    embedded_key = (EMBEDDED_PARSER_API_KEY or "").strip()
    if embedded_key and embedded_key != "PASTE_YOUR_API_KEY_HERE":
        return embedded_key
    return os.getenv("PARSER_API_KEY") or os.getenv("ARBITR_API_KEY") or ""


def clean_inn(inn):
    inn = "".join(ch for ch in str(inn or "") if ch.isdigit())
    if len(inn) not in (10, 12):
        raise ValueError("TIN must contain 10 or 12 digits")
    return inn


def api_get(url, params, timeout=60):
    key = external_api_key()
    if not key:
        raise RuntimeError("API key was not found. Paste it into EMBEDDED_PARSER_API_KEY near the top of app.py, or set PARSER_API_KEY/ARBITR_API_KEY.")
    params = {k: v for k, v in params.items() if v not in (None, "")}
    params["key"] = key
    with httpx.Client(timeout=timeout) as client:
        r = client.get(url, params=params, headers={"Accept": "application/json"})
    if r.status_code != 200:
        raise RuntimeError(f"The API returned error {r.status_code}: {r.text[:500]}")
    return r.json()


def to_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def find_row(rows, code):
    for row in rows or []:
        if str(row.get("code")) == str(code):
            return row
    return None


def row_val(rows, code, field="current"):
    row = find_row(rows, code)
    return to_float(row.get(field) if row else None)


def mln(value):
    return None if value is None else round(value / 1000, 2)


def pchg(cur, prev):
    if cur is None or prev in (None, 0):
        return None
    return round((cur - prev) / abs(prev) * 100, 2)


def div(a, b):
    if a is None or b in (None, 0):
        return None
    return round(a / b, 4)


def classifier_text(value):
    if isinstance(value, dict):
        return value.get("name") or value.get("id") or "—"
    return value or "—"


def normalize_fns_report(report):
    fin = report.get("financial_result") or []
    bal = report.get("balance") or []
    funds = report.get("funds_movement") or []

    revenue = to_float(report.get("gain_sum")) or row_val(fin, 2110)
    revenue_prev = row_val(fin, 2110, "previous")
    cost = row_val(fin, 2120)
    sales_profit = row_val(fin, 2200)
    profit_before_tax = row_val(fin, 2300)
    interest_expense = row_val(fin, 2330)
    profit = row_val(fin, 2400)
    profit_prev = row_val(fin, 2400, "previous") or row_val(fin, 2300, "previous")

    assets = to_float(report.get("actives")) or row_val(bal, 1600)
    assets_prev = row_val(bal, 1600, "previous")
    current_assets = row_val(bal, 1200)
    receivables = row_val(bal, 1230)
    equity = row_val(bal, 1300)
    retained_earnings = row_val(bal, 1370)
    long_term_liabilities = row_val(bal, 1400)
    long_term_borrowings = row_val(bal, 1410)
    short_term_liabilities = row_val(bal, 1500)
    short_term_borrowings = row_val(bal, 1510)
    payables = row_val(bal, 1520)

    # Cash flow statement values (form codes):
    # 4100 — net cash flow from operating activities; 4200 — investing activities;
    # 4300 — financing activities; 4400 — total net cash flow for the period;
    # 4450 — cash at beginning of period; 4500 — cash at end of period.
    operating_cash_flow = row_val(funds, 4100)
    investing_cash_flow = row_val(funds, 4200)
    financing_cash_flow = row_val(funds, 4300)
    net_cash_flow = row_val(funds, 4400)
    cash_beginning = row_val(funds, 4450)
    cash_end = row_val(funds, 4500)

    liabilities = None if long_term_liabilities is None and short_term_liabilities is None else (long_term_liabilities or 0) + (short_term_liabilities or 0)

    return {
        "period": report.get("period"),
        "published_date": report.get("published_date"),
        "url": report.get("url"),
        "required_audit": bool(report.get("required_audit")),
        "audit_report": report.get("audit_report"),

        # Raw values for model prefill
        "Revenue": revenue,
        "Себестоимость продаж": cost,
        "Прибыль (убыток) от продажи": sales_profit,
        "Чистая прибыль (убыток)": profit,
        "Активы всего": assets,
        "Оборотные активы": current_assets,
        "Капитал и резервы": equity,
        "Краткосрочные обязательства": short_term_liabilities,
        "Дебиторская задолженность": receivables,
        "Кредиторская задолженность": payables,
        "Нераспределенная прибыль (непокрытый убыток)": retained_earnings,
        "Долгосрочные обязательства": long_term_liabilities,
        "Долгосрочные заёмные средства": long_term_borrowings,
        "Краткосрочные заёмные средства": short_term_borrowings,
        "Прибыль до налогообложения": profit_before_tax,
        "Проценты к уплате": interest_expense,

        # Display values
        "revenue_mln_rub": mln(revenue),
        "profit_mln_rub": mln(profit),
        "sales_profit_mln_rub": mln(sales_profit),
        "assets_mln_rub": mln(assets),
        "current_assets_mln_rub": mln(current_assets),
        "equity_mln_rub": mln(equity),
        "liabilities_mln_rub": mln(liabilities),
        "short_term_liabilities_mln_rub": mln(short_term_liabilities),
        "operating_cash_flow_mln_rub": mln(operating_cash_flow),
        "investing_cash_flow_mln_rub": mln(investing_cash_flow),
        "financing_cash_flow_mln_rub": mln(financing_cash_flow),
        "net_cash_flow_mln_rub": mln(net_cash_flow),
        "cash_beginning_mln_rub": mln(cash_beginning),
        "cash_end_mln_rub": mln(cash_end),
        "revenue_change_pct": pchg(revenue, revenue_prev),
        "profit_change_pct": pchg(profit, profit_prev),
        "assets_change_pct": pchg(assets, assets_prev),
        "net_margin": div(profit, revenue),
        "return_on_assets": div(profit, assets),
        "autonomy_ratio": div(equity, assets),
        "current_ratio": div(current_assets, short_term_liabilities),
        "debt_to_assets": div(liabilities, assets),
    }


def parse_year(value):
    try:
        return int(str(value)[:4])
    except Exception:
        return None


def parse_registration_year(org):
    for key in ["registration_date", "status_date", "ogrn_date"]:
        value = org.get(key)
        if value:
            year = parse_year(value)
            if year:
                return year
    return None


def build_model_input_from_fns_reports(reports, org=None, kad_data=None, max_years=3):
    """Build model input rows from FNS accounting reports."""
    if not reports:
        return pd.DataFrame()

    org = org or {}
    reg_year = parse_registration_year(org)
    ordered = sorted(reports, key=lambda x: str(x.get("period") or ""), reverse=True)[:max_years]
    ordered = sorted(ordered, key=lambda x: str(x.get("period") or ""))

    # IRO/IDO proxy stays conservative: zero unless later user adjusts it manually.
    ido_value = 0.0
    records = []
    for item in ordered:
        year = parse_year(item.get("period")) or datetime.now().year
        record = {"Год": year}
        for field, _ in ML_RAW_FIELDS:
            record[field] = item.get(field) if item.get(field) is not None else 0.0
        for field, _ in CLASSICAL_OPTIONAL_FIELDS:
            record[field] = item.get(field) if item.get(field) is not None else 0.0
        record["Возраст компании, лет"] = float(max(year - reg_year, 0)) if reg_year else 5.0
        record["ИДО"] = ido_value
        records.append(record)

    return pd.DataFrame(records)


def build_fns_insights(org, reports):
    items = []
    if reports:
        latest = reports[0]
        items.append({"severity": "positive", "title": "Financial statements found", "text": f"Available periods: {len(reports)}. Latest period: {latest.get('period')}."})

        if latest.get("revenue_change_pct") is not None:
            ch = latest["revenue_change_pct"]
            items.append({"severity": "positive" if ch > 10 else "warning" if ch < -10 else "neutral", "title": "Revenue dynamics", "text": f"Revenue change: {ch:.1f}%."})

        if latest.get("profit_mln_rub") is not None:
            p = latest["profit_mln_rub"]
            items.append({"severity": "positive" if p >= 0 else "warning", "title": "Net profit", "text": f"Net result: {p:,.2f} million RUB."})

        net_cf = latest.get("net_cash_flow_mln_rub")
        if net_cf is not None and not pd.isna(net_cf):
            if net_cf >= 0:
                items.append({"severity": "positive", "title": "Net cash flow", "text": f"Net cash flow is positive: {net_cf:,.2f} million RUB."})
            else:
                items.append({"severity": "negative", "title": "Net cash flow", "text": f"Net cash flow is negative: {net_cf:,.2f} million RUB."})

            if len(reports) > 1:
                previous = reports[1]
                previous_net_cf = previous.get("net_cash_flow_mln_rub")
                if previous_net_cf is not None and not pd.isna(previous_net_cf):
                    cf_change = net_cf - previous_net_cf
                    if cf_change > 0:
                        items.append({"severity": "positive", "title": "Net cash flow dynamics", "text": f"Net cash flow increased by {cf_change:,.2f} million RUB versus the previous period."})
                    elif cf_change < 0:
                        items.append({"severity": "warning" if net_cf >= 0 else "negative", "title": "Net cash flow dynamics", "text": f"Net cash flow decreased by {abs(cf_change):,.2f} million RUB versus the previous period."})
                    else:
                        items.append({"severity": "neutral", "title": "Net cash flow dynamics", "text": "Net cash flow did not change versus the previous period."})
    else:
        items.append({"severity": "warning", "title": "Financial statements not found", "text": "The API did not return accounting statements."})
    return items

@st.cache_data(ttl=1800, show_spinner=False)
def load_fns(inn):
    inn = clean_inn(inn)
    search = api_get(f"{NALOG_BASE_URL}/search", {"inn": inn})
    items = search.get("items") or []
    if not items:
        return {"found": False, "inn": inn, "organization": {}, "reports": [], "insights": [{"severity":"warning","title":"Not found","text":"No organization was found for the specified TIN."}]}
    exact = [x for x in items if str(x.get("inn")) == inn]
    selected = next((x for x in (exact or items) if x.get("bfo")), (exact or items)[0])
    org_id = selected.get("id")
    details = api_get(f"{NALOG_BASE_URL}/details", {"id": org_id}) if org_id is not None else {"organization": selected, "reports": []}
    org = {k: v for k, v in (details.get("organization") or selected).items() if k != "id"}
    raw_reports = details.get("reports") or []
    reports = [normalize_fns_report(r) for r in raw_reports]
    reports.sort(key=lambda x: str(x.get("period") or ""), reverse=True)
    return {"found": True, "inn": inn, "organization": org, "reports": reports, "insights": build_fns_insights(org, reports)}


def party_has_inn(parties, inn):
    return any(inn in str(p) for p in (parties or []))


def normalize_case(item, inn):
    plaintiffs = item.get("Plaintiffs") or []
    respondents = item.get("Respondents") or []
    role = "Undefined"
    if party_has_inn(plaintiffs, inn):
        role = "Claimant"
    if party_has_inn(respondents, inn):
        role = "Respondent" if role == "Undefined" else role + " / Respondent"
    ctype = item.get("CaseType") or ""
    label = {"A":"Administrative", "А":"Administrative", "B":"Bankruptcy", "Б":"Bankruptcy", "G":"Civil", "Г":"Civil"}.get(ctype, ctype or "Not specified")
    case_id = item.get("CaseId")
    case_number = item.get("CaseNumber")
    if case_id:
        case_url = f"https://kad.arbitr.ru/Card/{case_id}"
    elif case_number:
        case_url = f"https://kad.arbitr.ru/?caseNumber={quote_plus(str(case_number))}"
    else:
        case_url = "https://kad.arbitr.ru/"
    return {
        "case_id": case_id,
        "case_number": case_number,
        "case_url": case_url,
        "start_date": item.get("StartDate"),
        "role": role,
        "case_type_label": label,
        "court": item.get("Court") or "Not specified"
    }


def build_kad_insights(cases, pages_count=0):
    total = len(cases)
    bankruptcy = sum(1 for c in cases if c.get("case_type_label") in {"Bankruptcy", "Банкротное"})
    if total == 0:
        return [{"severity": "positive", "title": "No arbitration cases found", "text": "No court activity was identified for the specified TIN."}]
    items = [{"severity": "warning" if bankruptcy else "neutral", "title": "Court activity", "text": f"Cases found: {total}. Search result pages: {pages_count}. Company roles are shown in the donut chart."}]
    return items

@st.cache_data(ttl=1800, show_spinner=False)
def load_kad(inn, pages_limit=3):
    inn = clean_inn(inn)
    payload = api_get(f"{ARBITR_BASE_URL}/search", {"Inn": inn})
    pages_count = int(payload.get("PagesCount") or 0)
    cases = [normalize_case(x, inn) for x in (payload.get("Cases") or [])]
    for page in range(2, min(pages_count, pages_limit) + 1):
        extra = api_get(f"{ARBITR_BASE_URL}/search", {"Inn": inn, "page": page})
        cases.extend(normalize_case(x, inn) for x in (extra.get("Cases") or []))
    return {"inn": inn, "pages_count": pages_count, "cases": cases, "insights": build_kad_insights(cases, pages_count)}


FNS_METRIC_CONFIG = [
    ("revenue_mln_rub", "Revenue, million RUB", "money"),
    ("profit_mln_rub", "Net profit, million RUB", "money"),
    ("sales_profit_mln_rub", "Profit from sales, million RUB", "money"),
    ("assets_mln_rub", "Assets, million RUB", "money"),
    ("current_assets_mln_rub", "Current assets, million RUB", "money"),
    ("equity_mln_rub", "Equity, million RUB", "money"),
    ("liabilities_mln_rub", "Liabilities, million RUB", "money"),
    ("short_term_liabilities_mln_rub", "Short-term liabilities, million RUB", "money"),
    ("net_cash_flow_mln_rub", "Net cash flow, million RUB", "money"),
    ("operating_cash_flow_mln_rub", "Operating cash flow, million RUB", "money"),
    ("investing_cash_flow_mln_rub", "Investing cash flow, million RUB", "money"),
    ("financing_cash_flow_mln_rub", "Financing cash flow, million RUB", "money"),
    ("cash_end_mln_rub", "Cash at period end, million RUB", "money"),
    ("revenue_change_pct", "Revenue change, %", "pct"),
    ("profit_change_pct", "Profit change, %", "pct"),
    ("assets_change_pct", "Assets change, %", "pct"),
    ("net_margin", "Net margin", "ratio"),
    ("return_on_assets", "ROA", "ratio"),
    ("autonomy_ratio", "Autonomy ratio", "ratio"),
    ("current_ratio", "Current ratio", "ratio"),
    ("debt_to_assets", "Debt / Assets", "ratio"),
]

def format_display_value(value, kind="plain"):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if kind == "money":
        return f"{value:,.2f}".replace(",", " ")
    if kind == "pct":
        return f"{value:.1f}%"
    if kind == "ratio":
        return f"{value:.2f}"
    return str(value)

def create_mini_trend_chart(years, values, color="#247b65", title=None):
    years = [str(year) for year in years]
    valid = [(str(y), v) for y, v in zip(years, values) if v is not None and not pd.isna(v)]
    fig = go.Figure()
    if valid:
        x, y = zip(*valid)
        fig.add_trace(go.Scatter(
            x=list(x), y=list(y), mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=7, color=color),
            fill="tozeroy", fillcolor="rgba(36,123,101,0.10)",
            hovertemplate="%{x}: %{y:.2f}<extra></extra>"
        ))

    layout = dict(
        margin=dict(l=0, r=8, t=68 if title else 8, b=30),
        height=285 if title else 155,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            title=None,
            tickfont=dict(size=10, color="#64748b"),
            tickmode="array",
            tickvals=years,
            ticktext=years,
            type="category",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            zeroline=False,
            title=None,
            tickfont=dict(size=10, color="#64748b")
        ),
        showlegend=False,
    )
    if title:
        layout["title"] = dict(text=str(title), font=dict(size=15, color="#1f2937"), x=0, xanchor="left")
    fig.update_layout(**layout)
    return fig


def create_bar_chart(labels, values, title="", color="#247b65"):
    labels = [str(label) for label in labels]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=color, text=values, textposition="outside"))
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=36, b=0),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=None,
            showgrid=False,
            type="category",
            tickmode="array",
            tickvals=labels,
            ticktext=labels,
        ),
        yaxis=dict(title=None, gridcolor="rgba(148,163,184,0.18)")
    )
    return fig

def create_donut_chart(labels, values, title=""):
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.6, textinfo="label+percent"))
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=36, b=0),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )
    return fig

def render_fns_trend_matrix(reports):
    if not reports:
        st.info("Financial statements were not found.")
        return
    ordered = sorted(reports, key=lambda x: str(x.get("period") or ""))
    years = [str(r.get("period") or "—") for r in ordered]
    for key, label, kind in FNS_METRIC_CONFIG:
        values = [r.get(key) for r in ordered]
        if all(v is None or (isinstance(v, float) and pd.isna(v)) for v in values):
            continue
        left, middle, right = st.columns([1.7, 2.9, 1.8])
        latest_value = next((v for v in reversed(values) if v is not None and not pd.isna(v)), None)
        with left:
            st.markdown(f'<div class="trend-card"><div class="trend-subtitle">Indicator</div><div class="trend-title">{label}</div><div class="factor-text">Latest value: <b>{format_display_value(latest_value, kind)}</b></div></div>', unsafe_allow_html=True)
        with middle:
            chips = "".join([f'<div class="year-chip"><div class="chip-year">{year}</div><div class="chip-value">{format_display_value(value, kind)}</div></div>' for year, value in zip(years, values)])
            st.markdown(f'<div class="trend-card"><div class="trend-subtitle">Periods</div><div class="year-strip">{chips}</div></div>', unsafe_allow_html=True)
        with right:
            st.plotly_chart(create_mini_trend_chart(years, values), use_container_width=True, config={"displayModeBar": False})


def render_kad_case_cards(cases, limit=12):
    if not cases:
        st.info("Cases were not found.")
        return
    ordered = sorted(cases, key=lambda item: str(item.get("start_date") or ""), reverse=True)
    shown = ordered[:limit]
    for i in range(0, len(shown), 2):
        cols = st.columns(2)
        for col, case in zip(cols, shown[i:i+2]):
            role = case.get("role") or "Undefined"
            case_type = case.get("case_type_label") or "Not specified"
            role_class = "role-respondent" if "Respondent" in role or "Ответчик" in role else "role-plaintiff" if "Claimant" in role or "Истец" in role else ""
            sev_class = "sev-negative" if case_type in {"Bankruptcy", "Банкротное"} else "sev-warning" if ("Respondent" in role or "Ответчик" in role) else "sev-positive" if ("Claimant" in role or "Истец" in role) else ""
            case_number = str(case.get("case_number") or "")
            case_id = case.get("case_id")
            if case_id:
                detail_url = f"https://kad.arbitr.ru/Card/{case_id}"
            elif case_number:
                detail_url = f"https://kad.arbitr.ru/?CaseNumber={quote(case_number)}"
            else:
                detail_url = "https://kad.arbitr.ru/"
            html = (
                f'<div class="case-card">'
                f'<div class="case-topline"><div class="case-number">{case_number or "No number"}</div><div class="case-date">{case.get("start_date") or "—"}</div></div>'
                f'<div class="case-pill-row"><span class="case-pill {role_class}">{role}</span><span class="case-pill {sev_class}">{case_type}</span></div>'
                f'<div class="case-court"><b>Court:</b> {case.get("court") or "Not specified"}</div>'
                f'<a class="case-link" href="{detail_url}" target="_blank" rel="noopener noreferrer">Details</a>'
                "</div>"
            )
            with col:
                st.markdown("".join(html), unsafe_allow_html=True)
    if len(ordered) > limit:
        with st.expander(f"Show {len(ordered) - limit} more cases"):
            for case in ordered[limit:]:
                role = case.get("role") or "Undefined"
                case_type = case.get("case_type_label") or "Not specified"
                st.markdown(f"**{case.get('case_number') or 'No number'}** · {case.get('start_date') or '—'} · {role} · {case_type} · {case.get('court') or 'Not specified'}")


def render_kad_charts(cases):
    if not cases:
        return
    df = pd.DataFrame(cases)
    if "start_date" in df.columns:
        df["year"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year
    else:
        df["year"] = None
    df = df.dropna(subset=["year"])
    if not df.empty:
        by_year = df.groupby("year").size().reset_index(name="count").sort_values("year")
        by_year["year"] = by_year["year"].astype(int).astype(str)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_bar_chart(by_year["year"].tolist(), by_year["count"].tolist(), "Cases by year"), use_container_width=True, config={"displayModeBar": False})
        role_counts = pd.Series({"Claimant": int(df["role"].fillna("").str.contains("Claimant|Истец", regex=True).sum()), "Respondent": int(df["role"].fillna("").str.contains("Respondent|Ответчик", regex=True).sum())})
        type_counts = df["case_type_label"].fillna("Not specified").value_counts()
        with col2:
            if role_counts.sum() > 0:
                st.plotly_chart(create_donut_chart(role_counts.index.tolist(), role_counts.values.tolist(), "Roles in cases"), use_container_width=True, config={"displayModeBar": False})
            elif len(type_counts) > 0:
                st.plotly_chart(create_donut_chart(type_counts.index.tolist(), type_counts.values.tolist(), "Case types"), use_container_width=True, config={"displayModeBar": False})


def render_source_cards(items, limit=6, summary_mode=False):
    if not items:
        st.info("No insights were generated.")
        return

    raw_cards = list(items)
    cards = []
    for item in raw_cards:
        if not isinstance(item, dict):
            continue
        if summary_mode:
            if not is_allowed_summary_factor(item):
                continue
        else:
            if factor_is_globally_excluded(item):
                continue
        cards.append(item)

    if not cards:
        st.info("No factors were generated for the selected data set.")
        return

    title_map = {"positive":"Positive signal", "warning":"Attention area", "negative":"Risk factor", "neutral":"Information"}
    shown = 0
    for i in range(0, len(cards), 2):
        if shown >= limit:
            break
        cols = st.columns(2)
        for col, item in zip(cols, cards[i:i+2]):
            if shown >= limit:
                break
            if summary_mode and not is_allowed_summary_factor(item):
                continue
            sev = item.get("severity", "neutral")
            with col:
                st.markdown(f"""
                <div class="factor-card-streamlit {sev}">
                    <div class="factor-title">{title_map.get(sev, 'Information')}</div>
                    <div class="factor-main">{item.get('title','')}</div>
                    <div class="factor-text">{item.get('text','')}</div>
                </div>
                """, unsafe_allow_html=True)
            shown += 1


def render_cash_flow_summary(reports):
    if not reports:
        return
    latest = sorted(reports, key=lambda x: str(x.get("period") or ""), reverse=True)[0]
    net_cf = latest.get("net_cash_flow_mln_rub")
    if net_cf is None or pd.isna(net_cf):
        return
    item = {
        "severity": "positive" if net_cf >= 0 else "negative",
        "title": "Net cash flow",
        "text": f"{net_cf:,.2f} million RUB for the latest period."
    }
    render_source_cards([item], limit=1)


def render_fns_page(data):
    if not data:
        st.info("Enter a TIN and click 'Load FNS and KAD'. FNS data will appear here.")
        return
    org = data.get("organization") or {}
    reports = data.get("reports") or []
    st.markdown('<div class="section-label">Organization profile</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", org.get("status") or "—")
    c2.metric("TIN", org.get("inn") or data.get("inn") or "—")
    c3.metric("OGRN", org.get("ogrn") or "—")
    st.markdown(f"""<div class="summary-box"><b>{org.get('short_name') or org.get('full_name') or 'Organization'}</b><br>Registration: {org.get('registration_date') or org.get('status_date') or '—'}<br>OKVED: {classifier_text(org.get('okved2'))}<br>Region: {org.get('region') or '—'}</div>""", unsafe_allow_html=True)
    st.markdown('<div class="section-label">FNS insights</div>', unsafe_allow_html=True)
    render_source_cards(data.get("insights") or [], limit=10)
    if reports:
        ordered = sorted(reports, key=lambda x: str(x.get("period") or ""))
        years = [str(r.get("period") or "—") for r in ordered]
        rev = [r.get("revenue_mln_rub") for r in ordered]
        profit = [r.get("profit_mln_rub") for r in ordered]
        assets = [r.get("assets_mln_rub") for r in ordered]
        cash_flow = [r.get("net_cash_flow_mln_rub") for r in ordered]
        st.markdown('<div class="section-label">Indicator trend charts</div>', unsafe_allow_html=True)
        ch1, ch2, ch3 = st.columns(3)
        with ch1:
            fig = create_mini_trend_chart(years, rev, color="#247b65", title="Revenue, million RUB")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with ch2:
            fig = create_mini_trend_chart(years, profit, color="#16a34a", title="Net profit, million RUB")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with ch3:
            fig = create_mini_trend_chart(years, assets, color="#0f766e", title="Assets, million RUB")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if any(v is not None and not pd.isna(v) for v in cash_flow):
            fig = create_mini_trend_chart(years, cash_flow, color="#7c3aed", title="Net cash flow, million RUB")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="section-label">Reporting by periods: transposed view</div>', unsafe_allow_html=True)
    render_fns_trend_matrix(reports)


def render_kad_page(data):
    if not data:
        st.info("Enter a TIN and click 'Load FNS and KAD'. KAD Arbitr data will appear here.")
        return
    cases = data.get("cases") or []
    respondent = sum(1 for c in cases if "Respondent" in c.get("role", "") or "Ответчик" in c.get("role", ""))
    plaintiff = sum(1 for c in cases if "Claimant" in c.get("role", "") or "Истец" in c.get("role", ""))
    bankruptcy = sum(1 for c in cases if c.get("case_type_label") in {"Bankruptcy", "Банкротное"})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total cases", len(cases))
    c2.metric("Respondent", respondent)
    c3.metric("Claimant", plaintiff)
    c4.metric("Bankruptcy", bankruptcy)
    st.markdown('<div class="section-label">KAD Arbitr insights</div>', unsafe_allow_html=True)
    render_source_cards(data.get("insights") or [])
    st.markdown('<div class="section-label">Court activity charts</div>', unsafe_allow_html=True)
    render_kad_charts(cases)
    st.markdown('<div class="section-label">Case cards</div>', unsafe_allow_html=True)
    render_kad_case_cards(cases)


def external_summary_text(fns_data, kad_data):
    parts = []
    if fns_data:
        org = fns_data.get("organization") or {}
        reports = fns_data.get('reports') or []
        if reports and reports[0].get('net_cash_flow_mln_rub') is not None:
            parts.append(f"FNS: status — {org.get('status') or 'not specified'}, reporting periods — {len(reports)}, net cash flow — {reports[0].get('net_cash_flow_mln_rub'):,.2f} million RUB.")
        else:
            parts.append(f"FNS: status — {org.get('status') or 'not specified'}, reporting periods — {len(reports)}.")
    if kad_data:
        cases = kad_data.get("cases") or []
        resp = sum(1 for c in cases if "Respondent" in c.get("role", "") or "Ответчик" in c.get("role", ""))
        bankr = sum(1 for c in cases if c.get("case_type_label") in {"Bankruptcy", "Банкротное"})
        parts.append(f"KAD Arbitr: cases found — {len(cases)}, respondent role — {resp}, bankruptcy cases — {bankr}.")
    return "\n\n".join(parts)


def make_input_signature(df, analysis_scope, input_method):
    """Build a stable signature of the currently displayed model input.
    Used to prevent stale results from previous runs being shown after the user changes the table, analysis type or input source.
    """
    if df is None or df.empty:
        return None
    normalized = df.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    for column in normalized.columns:
        normalized[column] = normalized[column].astype(str).fillna("")
    return {
        "analysis_scope": analysis_scope,
        "input_method": input_method,
        "columns": list(normalized.columns),
        "data": normalized.to_json(orient="split", force_ascii=False),
    }


def main():
    with st.sidebar:
        st.markdown("### Bankruptcy Risk Analyzer")
        st.caption("Single-company report or batch prediction")
        st.caption(f"Summary build: {APP_BUILD}")
        st.markdown("---")

        analysis_scope = st.radio(
            "Analysis type",
            ["Russian resident", "Non-resident"],
            help="For Russian residents, FNS and KAD Arbitr sources are available. For non-residents, the service uses the financial model without Russian external sources."
        )
        is_resident = analysis_scope == "Russian resident"

        st.markdown("---")
        mode = st.radio("Mode", ["Single-company report", "Batch company prediction"])

        st.markdown("---")
        st.markdown("#### Navigation")
        navigation_items = ["Summary", "ML forecast", "Classical models", "Insights", "Data"]
        if is_resident:
            navigation_items = ["Summary", "ML forecast", "Classical models", "Insights", "FNS", "KAD Arbitr", "Data"]
        report_section = st.radio(
            "Navigation",
            navigation_items,
            label_visibility="collapsed"
        )

        # Model is loaded silently to keep the sidebar clean.
        model_kind = "resident" if is_resident else "financial_only"
        bundle = load_bundle(model_kind=model_kind)
        threshold = float(bundle.get("threshold", 0.50)) if bundle is not None else 0.50

    st.markdown(
        """
        <div class="dashboard-panel">
            <div class="topbar">
                <div class="search-pill">Search company, scenario or report...</div>
                <div class="status-pill">Bankruptcy Risk Dashboard</div>
            </div>
            <div class="app-title">Bankruptcy Risk Analyzer</div>
            <div class="app-subtitle">
                Classical financial diagnostics, ML forecast and analytical insights based on annual company data.
                For non-residents, the service uses a limited mode without FNS and KAD Arbitr.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if mode == "Single-company report":
        st.markdown('<div class="section-label">Single-company report</div>', unsafe_allow_html=True)
        st.caption(
            "This mode analyses one company. Data can be entered manually "
            "or uploaded as a file for one or several reporting years."
        )

        company_name = st.text_input(
            "Company name",
            value=st.session_state.get("company_name", ""),
            placeholder="Company name"
        )
        fns_data = None
        kad_data = None
        fns_prefill_df = pd.DataFrame()

        if is_resident:
            st.markdown('<div class="section-label">External sources</div>', unsafe_allow_html=True)
            external_inn = st.text_input("TIN for loading FNS and KAD Arbitr", placeholder="7707083893")
            load_sources = st.button("Load FNS and KAD", use_container_width=True)
            if load_sources:
                try:
                    with st.spinner("Loading FNS data..."):
                        st.session_state["fns_data"] = load_fns(external_inn)
                    with st.spinner("Loading KAD Arbitr data..."):
                        st.session_state["kad_data"] = load_kad(external_inn)
                    fns_loaded = st.session_state.get("fns_data") or {}
                    org_loaded = fns_loaded.get("organization") or {}
                    auto_name = org_loaded.get("short_name") or org_loaded.get("full_name")
                    if auto_name and not company_name:
                        st.session_state["company_name"] = auto_name
                    st.session_state["fns_prefill_df"] = build_model_input_from_fns_reports(
                        fns_loaded.get("reports") or [],
                        org_loaded,
                        st.session_state.get("kad_data")
                    )
                    # External data have changed, so previous forecast must not be shown.
                    st.session_state.pop("result_df", None)
                    st.session_state.pop("current_input_signature", None)
                    st.session_state.pop("result_input_signature", None)
                    st.success("External-source data has been loaded. Model input fields are filled from FNS statements where available.")
                except Exception as exc:
                    st.error(str(exc))
            fns_data = st.session_state.get("fns_data")
            kad_data = st.session_state.get("kad_data")
            fns_prefill_df = st.session_state.get("fns_prefill_df", pd.DataFrame())
        else:
            st.info("Non-resident mode is selected: FNS and KAD Arbitr are disabled. The forecast is based on the financial model without the due diligence index.")

        input_options = ["Enter manually", "Upload file"]
        if is_resident and isinstance(fns_prefill_df, pd.DataFrame) and not fns_prefill_df.empty:
            input_options = ["Data from FNS"] + input_options

        input_method = st.radio(
            "Input method",
            input_options,
            horizontal=True
        )

        counterparty_role = render_counterparty_role_selector()

        if input_method == "Data from FNS":
            st.caption(
                "The table below shows financial data automatically collected from FNS statements. "
                "All values can be edited before running the model."
            )

            st.markdown('<div class="section-label">Editable data for model calculation</div>', unsafe_allow_html=True)
            edited_df = st.data_editor(
                fns_prefill_df.copy(),
                column_config={col: st.column_config.NumberColumn(field_label(col)) for col in fns_prefill_df.columns if col != "Год"},
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="fns_editable_model_input"
            )
            input_df = edited_df.copy()
        elif input_method == "Enter manually":
            input_df = build_manual_input(include_ido=is_resident)
        else:
            render_file_requirements("company", include_ido=is_resident)
            uploaded_file = st.file_uploader(
                "Excel or CSV file with single-company data",
                type=["xlsx", "xls", "csv"]
            )
            input_df = pd.DataFrame()
            if uploaded_file is not None:
                try:
                    required_fields = MODEL_RAW_FIELDS if is_resident else FINANCIAL_MODEL_RAW_FIELDS
                    input_df = read_uploaded_file(uploaded_file, required_fields=required_fields)
                    st.dataframe(display_df(input_df), use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.error(str(exc))

        if input_df.empty:
            return

        # Ensure numeric years and values before calculation.
        input_df = input_df.copy()
        if "Год" in input_df.columns:
            input_df["Год"] = pd.to_numeric(input_df["Год"], errors="coerce")
        for column in input_df.columns:
            if column != "Год":
                input_df[column] = pd.to_numeric(input_df[column], errors="coerce").fillna(0)
        input_df = input_df.dropna(subset=["Год"]).sort_values("Год").reset_index(drop=True)

        if not any(input_df["Активы всего"].fillna(0) > 0):
            st.info("Enter data for at least one year.")
            return

        # If the displayed/edited input table changes, remove old results.
        # This prevents showing a stale forecast for 2026 after the user loads FNS data ending in 2023.
        current_signature = make_input_signature(input_df, analysis_scope, input_method)
        if st.session_state.get("current_input_signature") != current_signature:
            st.session_state["current_input_signature"] = current_signature
            st.session_state.pop("result_df", None)

        if st.button("Calculate", type="primary", use_container_width=True):
            if bundle is None:
                st.error("To calculate the forecast, place the model bundle file in the service folder. For non-residents, bankruptcy_rf_financial_only_bundle.pkl is required.")
                return
            result = predict(input_df, bundle, threshold)
            result["Год"] = pd.to_numeric(result["Год"], errors="coerce")
            result = result.dropna(subset=["Год"]).sort_values("Год").reset_index(drop=True)
            st.session_state["result_df"] = result
            st.session_state["company_name"] = company_name
            st.session_state["result_input_signature"] = current_signature

        if "result_df" not in st.session_state:
            st.info("Data has been loaded or changed. Click Calculate to update the forecast and charts.")
            return

        # Do not show results calculated for a different table or analysis type.
        if st.session_state.get("result_input_signature") != current_signature:
            st.session_state.pop("result_df", None)
            st.info("Data has changed. Click Calculate to update the forecast and charts.")
            return

        result_df = st.session_state["result_df"].copy()
        result_df["Год"] = pd.to_numeric(result_df["Год"], errors="coerce")
        result_df = result_df.dropna(subset=["Год"]).sort_values("Год").reset_index(drop=True)
        latest = result_df.iloc[-1]
        latest_probability = latest["bankruptcy_probability"]
        latest_risk_text, latest_risk_key = risk_category(latest_probability)

        classical_by_year = {
            int(row["Год"]): run_classical_models(row)
            for _, row in result_df.iterrows()
        }
        latest_classical = classical_by_year.get(int(latest["Год"]), {})
        latest_classical_summary = summarize_classical_models(latest_classical)
        integrated_conclusion = generate_integrated_conclusion(
            company_name=st.session_state.get("company_name", company_name),
            df_result=result_df,
            classical_results=latest_classical
        )

        render_key_metrics(
            latest=latest,
            latest_probability=latest_probability,
            latest_risk_text=latest_risk_text,
            latest_classical_summary=latest_classical_summary
        )

        insights = generate_insights(result_df)

        if report_section == "Summary":
            summary_items = select_summary_items(
                insights,
                fns_data=fns_data,
                kad_data=kad_data,
                max_items=6,
                counterparty_role=counterparty_role,
                risk_key=latest_risk_key,
                result_df=result_df
            )
            render_structured_conclusion(
                company_name=st.session_state.get("company_name", company_name),
                result_df=result_df,
                classical_summary=latest_classical_summary,
                summary_items=summary_items,
                fns_data=fns_data,
                kad_data=kad_data,
                counterparty_role=counterparty_role
            )

            if is_resident and kad_data:
                st.markdown('<div class="section-label">KAD Arbitr: case share</div>', unsafe_allow_html=True)
                render_kad_share_panel(kad_data, counterparty_role=counterparty_role)

            st.markdown(f'<div class="section-label">{get_summary_factor_section_title(latest_risk_key)}</div>', unsafe_allow_html=True)
            render_source_cards(summary_items, limit=6, summary_mode=True)

        elif report_section == "ML forecast":
            st.markdown('<div class="summary-box">' + generate_ml_summary(result_df) + '</div>', unsafe_allow_html=True)

            if len(result_df) > 1:
                st.plotly_chart(plot_probability_dynamics(result_df), use_container_width=True)

            probability_table = result_df[
                ["Год", "bankruptcy_probability", "predicted_class", "risk_category"]
            ].copy()
            probability_table["bankruptcy_probability"] = probability_table[
                "bankruptcy_probability"
            ].map(lambda value: f"{value:.1%}")
            probability_table["risk_category"] = probability_table["risk_category"].map(display_risk)

            st.dataframe(
                probability_table.rename(
                    columns={
                        "Год": "Year",
                        "bankruptcy_probability": "Bankruptcy probability",
                        "predicted_class": "Class",
                        "risk_category": "Risk category"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

        elif report_section == "Classical models":
            if latest_classical:
                st.markdown(
                    '<div class="summary-box">' + generate_classical_summary_text(latest_classical) + '</div>',
                    unsafe_allow_html=True
                )
                st.markdown('<div class="section-label">Latest-year results</div>', unsafe_allow_html=True)
                classical_df = classical_results_table(latest_classical)
                st.dataframe(
                    classical_df.style.format({"Score": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True
                )

                classical_plot = plot_classical_models(latest_classical)
                if classical_plot is not None:
                    st.plotly_chart(classical_plot, use_container_width=True)

                if len(classical_by_year) > 1:
                    st.markdown('<div class="section-label">Results by year</div>', unsafe_allow_html=True)
                    rows = []
                    for year, results in classical_by_year.items():
                        summary = summarize_classical_models(results)
                        rows.append({
                            "Year": year,
                            "Models available": summary["available"],
                            "Low risk": summary["low"],
                            "Medium risk": summary["medium"],
                            "High risk": summary["high"],
                        })
                    st.dataframe(pd.DataFrame(rows).sort_values("Year"), use_container_width=True, hide_index=True)
            else:
                st.info("Classical models were not calculated due to insufficient input data.")

        elif report_section == "Insights":
            st.markdown('<div class="section-label">Current position</div>', unsafe_allow_html=True)
            render_factor_cards(insights, limit=6, counterparty_role=counterparty_role)

            if len(result_df) > 1:
                st.markdown('<div class="section-label">Dynamics</div>', unsafe_allow_html=True)
                for insight in insights["Dynamics"]:
                    st.markdown(
                        f'<div class="insight insight-{insight["severity"]}">{insight["text"]}</div>',
                        unsafe_allow_html=True
                    )

            if len(result_df) > 1:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_revenue_profit(result_df), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_ratio_dynamics(result_df), use_container_width=True)

            latest_ratios = pd.DataFrame(
                compute_ratios(latest).items(),
                columns=["Indicator", "Value"]
            )
            latest_ratios[["Assessment", "Interpretation"]] = latest_ratios.apply(
                lambda row: pd.Series(assess_coefficient(row["Indicator"], row["Value"])),
                axis=1
            )
            st.markdown('<div class="section-label">Latest-year ratios</div>', unsafe_allow_html=True)
            render_ratio_cards(latest_ratios)

        elif report_section == "FNS":
            render_fns_page(fns_data)

        elif report_section == "KAD Arbitr":
            render_kad_page(kad_data)

        elif report_section == "Data":
            display_columns = ["Год"] + MODEL_RAW_FIELDS + [
                "bankruptcy_probability",
                "risk_category"
            ]
            available_columns = [column for column in display_columns if column in result_df.columns]

            st.dataframe(
                display_df(result_df[available_columns]),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download results",
                to_excel(result_df),
                "bankruptcy_risk_results.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    else:
        st.markdown('<div class="section-label">Batch company prediction</div>', unsafe_allow_html=True)
        st.caption(
            "This mode evaluates multiple companies from one file. "
            "One row corresponds to one company in one reporting year."
        )

        if bundle is None:
            st.error("The model bundle must be loaded to run batch prediction.")
            return

        render_file_requirements("portfolio", include_ido=is_resident)
        uploaded_file = st.file_uploader("Excel or CSV file with data for multiple companies", type=["xlsx", "xls", "csv"])
        if uploaded_file is None:
            return

        try:
            required_fields = MODEL_RAW_FIELDS if is_resident else FINANCIAL_MODEL_RAW_FIELDS
            input_df = read_uploaded_file(uploaded_file, required_fields=required_fields)
        except Exception as exc:
            st.error(str(exc))
            return

        st.dataframe(display_df(input_df.head(10)), use_container_width=True, hide_index=True)

        if st.button("Run batch prediction", type="primary", use_container_width=True):
            result_df = predict(input_df, bundle, threshold)

            total = len(result_df)
            high_count = (result_df["risk_category"] == "High").sum()
            medium_count = (result_df["risk_category"] == "Medium").sum()
            low_count = (result_df["risk_category"] == "Low").sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total observations", total)
            c2.metric("High risk", high_count)
            c3.metric("Medium risk", medium_count)
            c4.metric("Low risk", low_count)

            result_view = result_df[
                ["Год", "bankruptcy_probability", "predicted_class", "risk_category"]
            ].copy()
            result_view["bankruptcy_probability"] = result_view[
                "bankruptcy_probability"
            ].map(lambda value: f"{value:.1%}")
            result_view["risk_category"] = result_view["risk_category"].map(display_risk)

            st.dataframe(
                display_df(result_view),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download results",
                to_excel(result_df),
                "batch_predictions.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
