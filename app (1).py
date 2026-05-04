"""
📉 Bankruptcy Analyzer — Анализ банкротства
Объединённое приложение:
  • ML-модель (Random Forest) из обученного бандла
  • Классические модели (Альтман, Таффлер, Спрингейт, Сайфуллин-Кадыков, Бивер)
  • Автозагрузка данных с bo.nalog.ru / kad.arbitr.ru по ИНН
  • Загрузка файла (Excel/CSV) для пакетного прогноза
  • Инсайты и рекомендации

Запуск:
    pip install streamlit requests beautifulsoup4 plotly pandas scikit-learn joblib openpyxl
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import time
import json
import joblib
import os
from io import BytesIO
from datetime import datetime

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ═══════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Bankruptcy Analyzer",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-family: 'Courier New', monospace;
        font-size: 32px;
        font-weight: 700;
    }
    .metric-label { font-size: 13px; color: #6c757d; margin-top: 4px; }
    .insight-box-danger {
        background: #f8d7da; border-left: 4px solid #ef4444;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
    }
    .insight-box-warn {
        background: #fef3cd; border-left: 4px solid #f59e0b;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
    }
    .insight-box-ok {
        background: #d1e7dd; border-left: 4px solid #10b981;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
    }
    .insight-box {
        background: #e8f4fd; border-left: 4px solid #3b82f6;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "ru-RU,ru;q=0.9",
}


def safe_div(num, den):
    return np.where((den != 0) & (~pd.isna(den)), num / den, np.nan)


def to_num(val):
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return val
    try:
        s = str(val).replace(" ", "").replace("\xa0", "").replace(",", ".")
        s = re.sub(r"[^\d.\-]", "", s)
        return float(s) if s else 0
    except (ValueError, TypeError):
        return 0


def safe_get(url, **kwargs):
    kwargs.setdefault("timeout", 15)
    kwargs.setdefault("headers", HEADERS)
    for attempt in range(2):
        try:
            r = requests.get(url, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == 1:
                return None
            time.sleep(1)
    return None


def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
    output.seek(0)
    return output


# ═══════════════════════════════════════════════════════════════
# ML-МОДЕЛЬ
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_bundle(file_obj=None):
    if file_obj is not None:
        return joblib.load(file_obj)
    for name in ["bankruptcy_rf_bundle.pkl", "model.pkl"]:
        if os.path.exists(name):
            return joblib.load(name)
    return None


def normalize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.str.replace('\ufeff', '', regex=False)
        .str.replace('\xa0', ' ', regex=False)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    return df


def build_features(df_raw):
    df = normalize_columns(df_raw)
    numeric_cols = [
        'Выручка', 'Активы всего', 'Чистая прибыль (убыток)',
        'Капитал и резервы', 'Краткосрочные обязательства',
        'Дебиторская задолженность', 'Кредиторская задолженность',
        'Оборотные активы', 'Себестоимость продаж',
        'Прибыль (убыток) от продажи',
        'Нераспределенная прибыль (непокрытый убыток)',
        'Возраст компании, лет', 'ИДО',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['roa'] = safe_div(df.get('Чистая прибыль (убыток)', 0), df.get('Активы всего', 0))
    df['roe'] = safe_div(df.get('Чистая прибыль (убыток)', 0), df.get('Капитал и резервы', 0))
    df['profitability_of_sales'] = safe_div(df.get('Чистая прибыль (убыток)', 0), df.get('Выручка', 0))
    df['operating_margin'] = safe_div(df.get('Прибыль (убыток) от продажи', 0), df.get('Выручка', 0))
    df['current_ratio'] = safe_div(df.get('Оборотные активы', 0), df.get('Краткосрочные обязательства', 0))
    df['receivables_turnover'] = safe_div(df.get('Выручка', 0), df.get('Дебиторская задолженность', 0))
    df['payables_turnover'] = safe_div(df.get('Себестоимость продаж', 0), df.get('Кредиторская задолженность', 0))
    df['asset_turnover'] = safe_div(df.get('Выручка', 0), df.get('Активы всего', 0))
    df['autonomy_ratio'] = safe_div(df.get('Капитал и резервы', 0), df.get('Активы всего', 0))
    df['retained_earnings_to_revenue'] = safe_div(
        df.get('Нераспределенная прибыль (непокрытый убыток)', 0), df.get('Выручка', 0)
    )
    if 'Активы всего' in df.columns:
        df['log_assets'] = np.where(df['Активы всего'] > 0, np.log(df['Активы всего']), np.nan)
    else:
        df['log_assets'] = np.nan
    df['company_age'] = df.get('Возраст компании, лет', pd.Series(dtype=float))
    return df


def predict_companies(df_raw, bundle, threshold=0.5):
    df_feat = build_features(df_raw)
    features = bundle['features']
    winsor_bounds = bundle['winsor_bounds']
    imputer = bundle['imputer']
    model = bundle['model']

    X = df_feat[features].copy()
    for col in features:
        if col in winsor_bounds:
            lo, hi = winsor_bounds[col]
            X[col] = X[col].clip(lower=lo, upper=hi)

    X_imp = pd.DataFrame(imputer.transform(X), columns=features, index=X.index)
    proba = model.predict_proba(X_imp)[:, 1]
    pred = (proba >= threshold).astype(int)

    result = df_feat.copy()
    result['bankruptcy_probability'] = proba
    result['predicted_class'] = pred
    result['risk_category'] = pd.cut(
        proba, bins=[-0.01, 0.33, 0.66, 1.0],
        labels=['Низкий', 'Средний', 'Высокий']
    )

    insights = []
    for _, row in result.iterrows():
        reasons = []
        if pd.notna(row.get('ИДО')) and row.get('ИДО', 10) < 2:
            reasons.append("Низкий индекс должной осмотрительности")
        if pd.notna(row.get('autonomy_ratio')) and row['autonomy_ratio'] < 0.2:
            reasons.append("Слабая структура капитала (автономия < 0.2)")
        if pd.notna(row.get('current_ratio')) and row['current_ratio'] < 1:
            reasons.append("Низкая ликвидность (< 1)")
        if pd.notna(row.get('roa')) and row['roa'] < 0:
            reasons.append("Отрицательная рентабельность активов")
        if pd.notna(row.get('retained_earnings_to_revenue')) and row['retained_earnings_to_revenue'] < 0:
            reasons.append("Отрицательная нераспределённая прибыль")
        if pd.notna(row.get('payables_turnover')) and row['payables_turnover'] < 1:
            reasons.append("Медленная оборачиваемость кредиторки")
        if not reasons:
            reasons.append("Существенных красных флагов не обнаружено")
        insights.append("; ".join(reasons))
    result['insights'] = insights
    return result


# ═══════════════════════════════════════════════════════════════
# ЗАГРУЗКА ДАННЫХ (bo.nalog.ru / kad.arbitr.ru)
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def bo_search(inn):
    try:
        r = safe_get("https://bo.nalog.ru/advanced-search/organizations/search",
                     params={"query": inn, "page": "0"})
        if r is None:
            return None
        data = r.json()
        if not data.get("content"):
            return None
        org = data["content"][0]
        return {
            "id": org.get("id"), "name": org.get("shortName") or org.get("fullName", ""),
            "fullName": org.get("fullName", ""), "inn": org.get("inn", inn),
            "ogrn": org.get("ogrn", ""), "status": org.get("statusName", ""),
            "regDate": org.get("registrationDate", ""), "address": org.get("address", ""),
            "okved": org.get("okvedName", ""),
        }
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def bo_get_periods(org_id):
    try:
        r = safe_get(f"https://bo.nalog.ru/nbo/organizations/{org_id}/bfo")
        if r is None:
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _parse_lines(data):
    lines = {}
    if isinstance(data, list):
        for item in data:
            code = str(item.get("code") or item.get("lineCode") or "")
            if code:
                lines[code] = {
                    "cur": to_num(item.get("currentValue") or item.get("endValue")),
                    "prev": to_num(item.get("previousValue") or item.get("startValue")),
                }
    elif isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, dict):
                code = str(val.get("code") or key)
                lines[code] = {
                    "cur": to_num(val.get("currentValue") or val.get("endValue")),
                    "prev": to_num(val.get("previousValue") or val.get("startValue")),
                }
    return lines


@st.cache_data(ttl=600, show_spinner=False)
def bo_get_financials(org_id, detail_id):
    """Баланс + ОФР → dict с русскими ключами (совместимо с ML-моделью)."""
    base = "https://bo.nalog.ru"
    result = {}

    bal_map = {
        "1600": "Активы всего", "1200": "Оборотные активы",
        "1100": "Внеоборотные активы", "1500": "Краткосрочные обязательства",
        "1400": "Долгосрочные обязательства", "1300": "Капитал и резервы",
        "1370": "Нераспределенная прибыль (непокрытый убыток)",
        "1410": "Долгосрочные заёмные средства", "1510": "Краткосрочные заёмные средства",
        "1230": "Дебиторская задолженность", "1520": "Кредиторская задолженность",
        "1250": "Денежные средства", "1210": "Запасы", "1150": "Основные средства",
    }
    try:
        r = safe_get(f"{base}/nbo/organizations/{org_id}/bfo/{detail_id}/balance")
        if r is not None:
            lines = _parse_lines(r.json())
            for code, field in bal_map.items():
                if code in lines:
                    result[field] = lines[code]["cur"]
    except Exception as e:
        st.warning(f"Баланс: {e}")

    ofr_map = {
        "2110": "Выручка", "2120": "Себестоимость продаж",
        "2200": "Прибыль (убыток) от продажи",
        "2300": "Прибыль до налогообложения",
        "2400": "Чистая прибыль (убыток)", "2330": "Проценты к уплате",
    }
    try:
        r = safe_get(f"{base}/nbo/organizations/{org_id}/bfo/{detail_id}/financialResult")
        if r is not None:
            lines = _parse_lines(r.json())
            for code, field in ofr_map.items():
                if code in lines:
                    result[field] = lines[code]["cur"]
    except Exception as e:
        st.warning(f"ОФР: {e}")

    return result


@st.cache_data(ttl=600, show_spinner=False)
def kad_search(inn):
    headers = {**HEADERS, "Content-Type": "application/json", "Accept": "application/json",
               "Origin": "https://kad.arbitr.ru", "Referer": "https://kad.arbitr.ru/",
               "x-requested-with": "XMLHttpRequest"}
    try:
        requests.get("https://kad.arbitr.ru", headers=HEADERS, timeout=10)
    except Exception:
        pass
    payload = {"Page": 1, "Count": 25, "Courts": [], "DateFrom": None, "DateTo": None,
               "Sides": [{"Name": inn, "Type": -1, "ExactMatch": False}],
               "Judges": [], "CaseNumbers": [], "WithVKSInstances": False}
    try:
        r = requests.post("https://kad.arbitr.ru/Kad/SearchInstances",
                          json=payload, headers=headers, timeout=25)
        if "json" in r.headers.get("Content-Type", ""):
            res = r.json().get("Result", {})
            items = res.get("Items", [])
            total = res.get("TotalCount", len(items))
        else:
            items, total = [], 0

        cases, bankruptcy, as_resp = [], 0, 0
        for item in items:
            is_b = item.get("CaseType") == "B" or "банкрот" in str(item).lower()
            if is_b: bankruptcy += 1
            respondents = item.get("Respondents", [])
            is_r = any(inn in str(rr.get("Inn", "")) for rr in respondents) if isinstance(respondents, list) else False
            if is_r: as_resp += 1
            pl = ", ".join(p.get("Name", "") for p in item.get("Plaintiffs", []) if isinstance(item.get("Plaintiffs"), list))
            re_ = ", ".join(p.get("Name", "") for p in respondents if isinstance(respondents, list))
            cid = item.get("CaseId", "")
            cases.append({"number": item.get("CaseNumber", ""),
                          "url": f"https://kad.arbitr.ru/Card/{cid}" if cid else "",
                          "date": item.get("Date", ""), "court": item.get("CourtName", ""),
                          "plaintiff": pl[:100], "respondent": re_[:100],
                          "is_respondent": is_r, "is_bankruptcy": is_b})
        return {"total": total, "cases": cases, "bankruptcy": bankruptcy, "as_respondent": as_resp}
    except Exception as e:
        return {"total": 0, "cases": [], "bankruptcy": 0, "as_respondent": 0, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# КЛАССИЧЕСКИЕ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════

def altman_z(d):
    ta = d.get("Активы всего", 0)
    if not ta: return None
    wc = d.get("Оборотные активы", 0) - d.get("Краткосрочные обязательства", 0)
    ebit = d.get("Прибыль до налогообложения", 0) + abs(d.get("Проценты к уплате", 0))
    debt = (d.get("Долгосрочные заёмные средства", 0) + d.get("Краткосрочные заёмные средства", 0)) or 1
    z = 0.717*(wc/ta) + 0.847*(d.get("Нераспределенная прибыль (непокрытый убыток)", 0)/ta) + \
        3.107*(ebit/ta) + 0.420*(d.get("Капитал и резервы", 0)/debt) + 0.998*(d.get("Выручка", 0)/ta)
    if z > 2.9: return z, "Безопасная зона", "low"
    if z > 1.23: return z, "Серая зона", "medium"
    return z, "Зона бедствия", "high"

def taffler(d):
    ta = d.get("Активы всего", 0)
    cl = d.get("Краткосрочные обязательства", 0)
    tl = d.get("Долгосрочные обязательства", 0) + cl
    if not ta or not cl or not tl: return None
    pbt = d.get("Прибыль до налогообложения", d.get("Чистая прибыль (убыток)", 0))
    z = 0.53*(pbt/cl) + 0.13*(d.get("Оборотные активы",0)/tl) + 0.18*(cl/ta) + 0.16*(d.get("Выручка",0)/ta)
    if z > 0.3: return z, "Низкий риск", "low"
    if z > 0.2: return z, "Умеренный риск", "medium"
    return z, "Высокий риск", "high"

def springate(d):
    ta = d.get("Активы всего", 0)
    if not ta: return None
    wc = d.get("Оборотные активы", 0) - d.get("Краткосрочные обязательства", 0)
    ebit = d.get("Прибыль до налогообложения", 0) + abs(d.get("Проценты к уплате", 0))
    pbt = d.get("Прибыль до налогообложения", d.get("Чистая прибыль (убыток)", 0))
    z = 1.03*(wc/ta) + 3.07*(ebit/ta) + 0.66*(pbt/(d.get("Краткосрочные обязательства",0) or 1)) + 0.4*(d.get("Выручка",0)/ta)
    if z > 0.862: return z, "Устойчивое положение", "low"
    return z, "Риск банкротства", "high"

def saifullin(d):
    ta = d.get("Активы всего", 0); eq = d.get("Капитал и резервы", 0); sa = d.get("Выручка", 0)
    if not ta or not sa or not eq: return None
    ca = d.get("Оборотные активы", 0); cl = d.get("Краткосрочные обязательства", 0)
    nca = d.get("Внеоборотные активы", 0) or (ta - ca); np_ = d.get("Чистая прибыль (убыток)", 0)
    R = 2*((eq-nca)/(ca or 1)) + 0.1*(ca/(cl or 1)) + 0.08*(sa/ta) + 0.45*(np_/(sa or 1)) + np_/(eq or 1)
    if R >= 1: return R, "Удовлетворительное", "low"
    if R >= 0.5: return R, "Неустойчивое", "medium"
    return R, "Неудовлетворительное", "high"

def beaver_m(d):
    ta = d.get("Активы всего", 0); cl = d.get("Краткосрочные обязательства", 0)
    tl = d.get("Долгосрочные обязательства", 0) + cl
    if not ta or not tl: return None
    np_ = d.get("Чистая прибыль (убыток)", 0); ca = d.get("Оборотные активы", 0)
    s = 0
    bv = np_/(tl or 1)
    if bv < 0.17: s += 2
    elif bv < 0.4: s += 1
    if np_/(ta or 1) < -0.15: s += 2
    elif np_/(ta or 1) < 0.02: s += 1
    if tl/ta > 0.8: s += 2
    elif tl/ta > 0.5: s += 1
    if (ca-cl)/ta < 0.06: s += 2
    elif (ca-cl)/ta < 0.3: s += 1
    if ca/(cl or 1) < 1: s += 2
    elif ca/(cl or 1) < 2: s += 1
    if s <= 3: return s, "Нормальное положение", "low"
    if s <= 6: return s, "5 лет до банкротства", "medium"
    return s, "1 год до банкротства", "high"


# ═══════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════

def risk_color(r): return {"low": "#10b981", "medium": "#f59e0b", "high": "#ef4444"}.get(r, "#999")
def risk_emoji(r): return {"low": "✅", "medium": "⚡", "high": "⚠️"}.get(r, "❓")

def make_gauge(score, mn, mx, title, risk):
    if not HAS_PLOTLY: return None
    c = risk_color(risk)
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score,
        title={"text": title, "font": {"size": 14}},
        number={"font": {"size": 24, "color": c}},
        gauge={"axis": {"range": [mn, mx]}, "bar": {"color": c, "thickness": 0.7},
               "steps": [{"range": [mn, mn+(mx-mn)*0.33], "color": "rgba(239,68,68,0.08)"},
                         {"range": [mn+(mx-mn)*0.33, mn+(mx-mn)*0.66], "color": "rgba(245,158,11,0.08)"},
                         {"range": [mn+(mx-mn)*0.66, mx], "color": "rgba(16,185,129,0.08)"}]}))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=10))
    return fig

def make_proba_gauge(proba):
    if not HAS_PLOTLY: return None
    c = "#ef4444" if proba > 0.66 else "#f59e0b" if proba > 0.33 else "#10b981"
    fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=proba*100,
        title={"text": "Вероятность банкротства (ML)", "font": {"size": 16}},
        number={"suffix": "%", "font": {"size": 40, "color": c}},
        delta={"reference": 50, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": c, "thickness": 0.8},
               "steps": [{"range": [0, 33], "color": "rgba(16,185,129,0.12)"},
                         {"range": [33, 66], "color": "rgba(245,158,11,0.12)"},
                         {"range": [66, 100], "color": "rgba(239,68,68,0.12)"}],
               "threshold": {"line": {"color": "#333", "width": 3}, "thickness": 0.8, "value": 50}}))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=20))
    return fig


def calc_age(reg_date):
    if not reg_date: return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return round((datetime.now() - datetime.strptime(str(reg_date)[:10], fmt)).days / 365.25, 1)
        except ValueError:
            continue
    return None


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    with st.sidebar:
        st.markdown("## 📉 Bankruptcy Analyzer")
        st.markdown("---")
        mode = st.radio("Режим", ["🔍 Анализ по ИНН", "📁 Пакетный (файл)"], index=0)
        st.markdown("---")
        st.markdown("#### 🤖 ML-модель")
        model_file = st.file_uploader("Загрузите `.pkl`", type=["pkl"])
        bundle = load_bundle(model_file)
        if bundle:
            st.success("✅ Модель загружена")
            threshold = st.slider("Порог", 0.10, 0.90, bundle.get("threshold", 0.5), 0.05)
        else:
            st.info("Загрузите .pkl файл")
            threshold = 0.5
        st.markdown("---")
        st.caption("[bo.nalog.ru](https://bo.nalog.ru) · [КАД Арбитр](https://kad.arbitr.ru) · [Checko](https://checko.ru)")

    # ══════════════════════════════════════════════════════════
    # РЕЖИМ 1: ПО ИНН
    # ══════════════════════════════════════════════════════════
    if "ИНН" in mode:
        st.title("🔍 Анализ компании по ИНН")

        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1: inn = st.text_input("ИНН", max_chars=12, placeholder="7707049388")
        with c2: ido = st.number_input("ИДО (0–10)", 0.0, 10.0, 5.0, 0.1, help="Индекс должной осмотрительности")
        with c3: age_over = st.number_input("Возраст (лет)", 0.0, 200.0, 0.0, 1.0, help="0 = авто")
        with c4:
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("🚀 Анализ", type="primary", use_container_width=True)

        if run and inn and len(inn) >= 10:

            # ── Компания ──
            org = None
            fns = {}
            api_available = True

            with st.spinner("🔍 Ищу компанию на bo.nalog.ru..."):
                org = bo_search(inn)

            if not org:
                api_available = False
                st.warning("⚠️ bo.nalog.ru недоступен (таймаут). Сервер Streamlit Cloud не может подключиться к российским госсервисам. Введите данные вручную или загрузите файл в пакетном режиме.")
                st.info(f"🔗 Проверьте компанию вручную: [bo.nalog.ru](https://bo.nalog.ru) · [Checko](https://checko.ru/company/{inn}) · [Rusprofile](https://www.rusprofile.ru/search?query={inn})")
                org = {"id": None, "name": st.text_input("Введите название компании", value=""), "inn": inn,
                       "ogrn": "", "status": "", "regDate": "", "address": "", "okved": ""}

            age = age_over if age_over > 0 else calc_age(org.get("regDate"))

            if org.get("name"):
                st.markdown(f"### 🏢 {org['name']}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ИНН", org.get("inn", inn))
            m2.metric("ОГРН", org.get("ogrn") or "—")
            m3.metric("Статус", org.get("status") or "—")
            m4.metric("Возраст", f"{age:.0f} лет" if age else "—")

            if api_available and org.get("fullName"):
                with st.expander("📄 Подробности"):
                    st.write(f"**Название:** {org.get('fullName','—')}")
                    st.write(f"**Адрес:** {org.get('address','—')}")
                    st.write(f"**ОКВЭД:** {org.get('okved','—')}")
                    st.write(f"**Ссылки:** [Checko](https://checko.ru/company/{inn}) · [Rusprofile](https://www.rusprofile.ru/search?query={inn})")

            # ── Финансы ──
            if api_available and org.get("id"):
                with st.spinner("📊 Загружаю отчётность..."):
                    periods = bo_get_periods(org["id"])

                if periods:
                    p_map = {str(p.get("period") or p.get("year")): p for p in periods}
                    sel = st.selectbox("📅 Период", list(p_map.keys()))
                    did = p_map[sel].get("id")

                    with st.spinner(f"📥 Данные за {sel}..."):
                        fns = bo_get_financials(org["id"], did)

                    if fns:
                        st.success(f"✅ Загружено {len(fns)} показателей за {sel} г.")
                    else:
                        st.warning("Финансовые данные не загрузились — введите вручную ниже")
                else:
                    st.warning("Периоды отчётности не найдены — введите данные вручную")

            # ── Ручной ввод (всегда показываем, с предзаполнением если данные есть) ──
            with st.expander("📋 Финансовые показатели (тыс. руб.) — редактирование", expanded=not bool(fns)):
                manual_fields = [
                    ("Активы всего", "1600"), ("Оборотные активы", "1200"),
                    ("Краткосрочные обязательства", "1500"), ("Долгосрочные обязательства", "1400"),
                    ("Капитал и резервы", "1300"), ("Выручка", "2110"),
                    ("Чистая прибыль (убыток)", "2400"), ("Себестоимость продаж", "2120"),
                    ("Прибыль (убыток) от продажи", "2200"), ("Прибыль до налогообложения", "2300"),
                    ("Нераспределенная прибыль (непокрытый убыток)", "1370"),
                    ("Дебиторская задолженность", "1230"), ("Кредиторская задолженность", "1520"),
                    ("Проценты к уплате", "2330"),
                    ("Долгосрочные заёмные средства", "1410"), ("Краткосрочные заёмные средства", "1510"),
                ]
                mcols = st.columns(3)
                for i, (field, code) in enumerate(manual_fields):
                    with mcols[i % 3]:
                        val = fns.get(field, 0.0)
                        fns[field] = st.number_input(f"{field} ({code})", value=float(val), step=1000.0,
                                                     format="%.0f", key=f"man_{field}")

            if not fns or not fns.get("Активы всего"):
                st.info("Заполните финансовые показатели для запуска моделей")
                return

            # ── Вкладки ──
            arb = {"total": 0, "cases": [], "bankruptcy": 0, "as_respondent": 0}
            t_ml, t_cls, t_arb, t_ins = st.tabs(["🤖 ML-модель", "📊 Классические", "⚖️ Арбитраж", "💡 Инсайты"])

            # ML
            with t_ml:
                if not bundle:
                    st.warning("⬆️ Загрузите .pkl в боковой панели")
                else:
                    row = dict(fns)
                    if age: row["Возраст компании, лет"] = age
                    row["ИДО"] = ido
                    df_in = pd.DataFrame([row])
                    try:
                        res = predict_companies(df_in, bundle, threshold)
                        prob = res['bankruptcy_probability'].iloc[0]
                        pred = res['predicted_class'].iloc[0]
                        rcat = res['risk_category'].iloc[0]
                        ins = res['insights'].iloc[0]

                        fig = make_proba_gauge(prob)
                        if fig: st.plotly_chart(fig, use_container_width=True)

                        a, b, c_ = st.columns(3)
                        a.metric("Вероятность", f"{prob:.1%}")
                        b.metric("Класс", "🔴 Банкрот" if pred else "🟢 Не банкрот")
                        c_.metric("Риск", rcat)

                        st.markdown("**Причины:**")
                        for r in ins.split("; "):
                            cl = "insight-box-danger" if any(w in r for w in ["Отрицат", "Низк", "Слаб", "Медлен"]) else "insight-box-ok"
                            st.markdown(f'<div class="{cl}">• {r}</div>', unsafe_allow_html=True)

                        with st.expander("📊 Признаки модели"):
                            fv = res[bundle['features']].iloc[0]
                            st.dataframe(pd.DataFrame({"Признак": bundle['features'],
                                                       "Значение": [f"{v:.4f}" if pd.notna(v) else "—" for v in fv]}),
                                         use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Ошибка: {e}"); st.exception(e)

            # Классические
            with t_cls:
                models = {"Альтман Z'": altman_z(fns), "Таффлер": taffler(fns),
                          "Спрингейт": springate(fns), "Сайфуллин-Кадыков": saifullin(fns), "Бивер": beaver_m(fns)}
                risks = [r[2] for r in models.values() if r]
                if risks:
                    rc = {k: risks.count(k) for k in ("high","medium","low")}
                    ov = "high" if rc["high"]>=3 or (rc["high"]>=1 and rc["medium"]>=2) else "low" if rc["low"]>=3 else "medium"
                    st.markdown(f'<div class="metric-card" style="border-left:4px solid {risk_color(ov)}">'
                                f'<div class="metric-value" style="color:{risk_color(ov)}">{risk_emoji(ov)} '
                                f'{"ВЫСОКИЙ" if ov=="high" else "СРЕДНИЙ" if ov=="medium" else "НИЗКИЙ"} РИСК</div>'
                                f'<div class="metric-label">Консенсус {len(risks)} моделей: 🔴{rc["high"]} 🟡{rc["medium"]} 🟢{rc["low"]}</div>'
                                f'</div>', unsafe_allow_html=True)

                gp = {"Альтман Z'": (-2,5), "Таффлер": (-0.5,1.5), "Спрингейт": (-1,3),
                      "Сайфуллин-Кадыков": (-1,3), "Бивер": (0,10)}
                cols = st.columns(3)
                for i, (nm, rv) in enumerate(models.items()):
                    with cols[i % 3]:
                        if rv is None: st.warning(f"**{nm}:** мало данных"); continue
                        s, z, rk = rv
                        mn, mx = gp.get(nm, (0,5))
                        fig = make_gauge(s, mn, mx, nm, rk)
                        if fig: st.plotly_chart(fig, use_container_width=True)
                        st.markdown(f"**{z}** — {risk_emoji(rk)} {s:.3f}")

            # Арбитраж
            with t_arb:
                arb = {"total": 0, "cases": [], "bankruptcy": 0, "as_respondent": 0}
                try:
                    with st.spinner("⚖️ Загружаю дела..."):
                        arb = kad_search(inn)
                except Exception:
                    arb["error"] = "Таймаут подключения к kad.arbitr.ru"
                if arb.get("error"):
                    st.warning(f"⚠️ {arb['error']}")
                    st.info(f"kad.arbitr.ru может быть недоступен из-за геоблокировки. [Проверьте вручную →](https://kad.arbitr.ru/)")
                else:
                    a1, a2, a3 = st.columns(3)
                    a1.metric("Всего", arb["total"]); a2.metric("🔴 Банкротные", arb["bankruptcy"])
                    a3.metric("⚠️ Ответчик", arb["as_respondent"])
                    if arb["bankruptcy"] > 0: st.error("⚠️ Обнаружены банкротные дела!")
                    for cs in arb["cases"][:15]:
                        bm = " 🔴 **БАНКРОТ**" if cs["is_bankruptcy"] else ""
                        rm = " | 📌 Ответчик" if cs["is_respondent"] else ""
                        lnk = f"[{cs['number']}]({cs['url']})" if cs["url"] else cs["number"]
                        st.markdown(f"**{lnk}**{bm}{rm} — {cs['date']} | {cs['court']}")
                    if arb["total"] == 0: st.success("✅ Дел не найдено")

            # Инсайты
            with t_ins:
                ta = fns.get("Активы всего", 0); ca = fns.get("Оборотные активы", 0)
                cl = fns.get("Краткосрочные обязательства", 0)
                eq = fns.get("Капитал и резервы", 0); sa = fns.get("Выручка", 0)
                np_ = fns.get("Чистая прибыль (убыток)", 0)
                tl = fns.get("Долгосрочные обязательства", 0) + cl

                cr = ca/(cl or 1); lev = tl/(ta or 1); ros = np_/(sa or 1) if sa else 0; aut = eq/(ta or 1) if ta else 0

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Ликвидность", f"{cr:.2f}", "≥2" if cr>=2 else "<2", delta_color="normal" if cr>=2 else "inverse")
                k2.metric("Леверидж", f"{lev:.0%}", "<50%" if lev<0.5 else ">50%", delta_color="normal" if lev<0.5 else "inverse")
                k3.metric("ROS", f"{ros:.1%}", "+" if ros>0 else "−", delta_color="normal" if ros>0 else "inverse")
                k4.metric("Автономия", f"{aut:.2f}", "≥0.5" if aut>=0.5 else "<0.5", delta_color="normal" if aut>=0.5 else "inverse")

                il = []
                if cr<1: il.append(("danger",f"Ликвидность {cr:.2f} < 1 — не покрывает обязательства"))
                elif cr<2: il.append(("warn",f"Ликвидность {cr:.2f} — ниже рекомендуемой 2.0"))
                else: il.append(("ok",f"Ликвидность {cr:.2f} — хорошо"))
                if lev>0.8: il.append(("danger",f"Долговая нагрузка {lev:.0%} — критическая"))
                elif lev>0.5: il.append(("warn",f"Долговая нагрузка {lev:.0%} — контролируйте"))
                else: il.append(("ok",f"Долговая нагрузка {lev:.0%} — безопасная"))
                if np_<0: il.append(("danger","Компания убыточна"))
                elif ros<0.05: il.append(("warn",f"Низкая рентабельность ({ros:.1%})"))
                else: il.append(("ok",f"Рентабельность {ros:.1%}"))
                if aut<0.2: il.append(("danger",f"Автономия {aut:.2f} — зависимость от долга"))
                if (ca-cl)<0: il.append(("danger","Отрицательный рабочий капитал"))
                if arb.get("bankruptcy",0)>0: il.append(("danger",f"Банкротные дела: {arb['bankruptcy']}"))

                for lv, tx in il:
                    em = {"danger":"🔴","warn":"🟡","ok":"🟢"}.get(lv,"ℹ️")
                    st.markdown(f'<div class="insight-box-{lv}">{em} {tx}</div>', unsafe_allow_html=True)

        elif run:
            st.warning("Введите ИНН (10–12 цифр)")

    # ══════════════════════════════════════════════════════════
    # РЕЖИМ 2: ПАКЕТНЫЙ
    # ══════════════════════════════════════════════════════════
    else:
        st.title("📁 Пакетный анализ из файла")
        data_file = st.file_uploader("📄 Excel / CSV", type=["xlsx", "xls", "csv"])

        if not bundle:
            st.warning("⬆️ Загрузите .pkl модель в боковой панели")
        elif data_file:
            try:
                nm = data_file.name.lower()
                if nm.endswith(('.xlsx','.xls')):
                    df_in = pd.read_excel(data_file)
                else:
                    try: df_in = pd.read_csv(data_file)
                    except: data_file.seek(0); df_in = pd.read_csv(data_file, sep=';', encoding='cp1251')

                st.write("### 📋 Предпросмотр")
                st.dataframe(df_in.head(), use_container_width=True)

                missing = [c for c in bundle.get('raw_input_cols',[]) if c not in normalize_columns(df_in).columns]
                if missing:
                    st.error("❌ Недостающие столбцы:"); st.write(missing)
                else:
                    res = predict_companies(df_in, bundle, threshold)
                    st.write("### 📊 Результаты")
                    show = [c for c in ['Наименование','Регистрационный номер','bankruptcy_probability',
                                        'predicted_class','risk_category','insights'] if c in res.columns]
                    st.dataframe(res[show] if show else res.head(), use_container_width=True)

                    st.write("### 📈 Распределение")
                    st.bar_chart(res['risk_category'].value_counts())

                    s1, s2, s3 = st.columns(3)
                    tot = len(res); hi = (res['risk_category']=='Высокий').sum()
                    s1.metric("Компаний", tot)
                    s2.metric("🔴 Высокий риск", f"{hi} ({hi/tot*100:.0f}%)" if tot else "0")
                    s3.metric("Ср. вероятность", f"{res['bankruptcy_probability'].mean():.1%}")

                    st.download_button("📥 Скачать Excel", to_excel_bytes(res),
                                       "bankruptcy_predictions.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка: {e}"); st.exception(e)
        else:
            st.info("Загрузите файл с данными")
            if bundle:
                st.markdown("**Ожидаемые столбцы:**")
                for i, c in enumerate(bundle.get('raw_input_cols', []), 1):
                    st.write(f"{i}. `{c}`")


if __name__ == "__main__":
    main()
