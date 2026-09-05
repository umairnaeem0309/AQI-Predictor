"""
AQI Theme and Color Utility

Centralized color scheme, design tokens, and CSS design system for AirPulse dashboard.
All pages import `get_dashboard_css()` once at render time to inject the full token set.
"""

from typing import Dict, Tuple

# ── US EPA AQI Color Palette ──────────────────────────────────────────────────
AQI_COLORS = {
    "good": "#00C853",
    "moderate": "#FFD600",
    "unhealthy_sensitive": "#FF6D00",
    "unhealthy": "#D50000",
    "very_unhealthy": "#6A1B9A",
    "hazardous": "#4A0010",
}

# Lighter background tints for cards / stripes
AQI_BG_COLORS = {
    "good": "#E8F5E9",
    "moderate": "#FFFDE7",
    "unhealthy_sensitive": "#FFF3E0",
    "unhealthy": "#FFEBEE",
    "very_unhealthy": "#F3E5F5",
    "hazardous": "#FCE4EC",
}

# Category display names
AQI_CATEGORY_NAMES = {
    "good": "Good",
    "moderate": "Moderate",
    "unhealthy_sensitive": "Unhealthy for Sensitive Groups",
    "unhealthy": "Unhealthy",
    "very_unhealthy": "Very Unhealthy",
    "hazardous": "Hazardous",
}

AQI_CATEGORY_SHORT = {
    "good": "Good",
    "moderate": "Moderate",
    "unhealthy_sensitive": "USG",
    "unhealthy": "Unhealthy",
    "very_unhealthy": "Very Unhealthy",
    "hazardous": "Hazardous",
}

# ── Chart Color Palette ───────────────────────────────────────────────────────
CHART_COLORS = {
    "primary": "#1E88E5",
    "secondary": "#43A047",
    "tertiary": "#FB8C00",
    "quaternary": "#E53935",
    "purple": "#8E24AA",
    "teal": "#00897B",
    "background": "#FFFFFF",
    "text": "#1A1A2E",
    "grid": "#E8ECF0",
    "surface": "#F8FAFC",
}

# City colors for multi-city charts
CITY_COLORS = {
    "Karachi": "#1E88E5",
    "Lahore": "#43A047",
    "Islamabad": "#FB8C00",
}


# ── Helper Functions ──────────────────────────────────────────────────────────

def get_aqi_color(aqi_value: int) -> str:
    """Get AQI hex color from value (0-500)."""
    if aqi_value <= 50:
        return AQI_COLORS["good"]
    elif aqi_value <= 100:
        return AQI_COLORS["moderate"]
    elif aqi_value <= 150:
        return AQI_COLORS["unhealthy_sensitive"]
    elif aqi_value <= 200:
        return AQI_COLORS["unhealthy"]
    elif aqi_value <= 300:
        return AQI_COLORS["very_unhealthy"]
    else:
        return AQI_COLORS["hazardous"]


def get_aqi_bg_color(aqi_value: int) -> str:
    """Get AQI background tint color from value."""
    if aqi_value <= 50:
        return AQI_BG_COLORS["good"]
    elif aqi_value <= 100:
        return AQI_BG_COLORS["moderate"]
    elif aqi_value <= 150:
        return AQI_BG_COLORS["unhealthy_sensitive"]
    elif aqi_value <= 200:
        return AQI_BG_COLORS["unhealthy"]
    elif aqi_value <= 300:
        return AQI_BG_COLORS["very_unhealthy"]
    else:
        return AQI_BG_COLORS["hazardous"]


def get_aqi_category(aqi_value: int) -> str:
    """Get AQI category full display name from value."""
    if aqi_value <= 50:
        return AQI_CATEGORY_NAMES["good"]
    elif aqi_value <= 100:
        return AQI_CATEGORY_NAMES["moderate"]
    elif aqi_value <= 150:
        return AQI_CATEGORY_NAMES["unhealthy_sensitive"]
    elif aqi_value <= 200:
        return AQI_CATEGORY_NAMES["unhealthy"]
    elif aqi_value <= 300:
        return AQI_CATEGORY_NAMES["very_unhealthy"]
    else:
        return AQI_CATEGORY_NAMES["hazardous"]


def get_aqi_category_short(aqi_value: int) -> str:
    """Get short AQI category name (for badges/chips)."""
    if aqi_value <= 50:
        return AQI_CATEGORY_SHORT["good"]
    elif aqi_value <= 100:
        return AQI_CATEGORY_SHORT["moderate"]
    elif aqi_value <= 150:
        return AQI_CATEGORY_SHORT["unhealthy_sensitive"]
    elif aqi_value <= 200:
        return AQI_CATEGORY_SHORT["unhealthy"]
    elif aqi_value <= 300:
        return AQI_CATEGORY_SHORT["very_unhealthy"]
    else:
        return AQI_CATEGORY_SHORT["hazardous"]


def get_aqi_category_key(aqi_value: int) -> str:
    """Get AQI category dict key from value."""
    if aqi_value <= 50:
        return "good"
    elif aqi_value <= 100:
        return "moderate"
    elif aqi_value <= 150:
        return "unhealthy_sensitive"
    elif aqi_value <= 200:
        return "unhealthy"
    elif aqi_value <= 300:
        return "very_unhealthy"
    else:
        return "hazardous"


def get_city_color(city: str) -> str:
    """Get brand color for a city."""
    return CITY_COLORS.get(city, CHART_COLORS["primary"])


def render_aqi_badge(aqi_value: int, category: str, size: str = "md") -> str:
    """
    Return an HTML AQI severity badge string for use in st.markdown().

    Args:
        aqi_value: Numeric AQI value
        category: Display category name
        size: 'sm', 'md', or 'lg'

    Returns:
        HTML string for the badge
    """
    color = get_aqi_color(aqi_value)
    bg = get_aqi_bg_color(aqi_value)

    size_map = {
        "sm": ("0.65rem", "3px 8px", "0.7rem"),
        "md": ("0.8rem", "4px 12px", "0.85rem"),
        "lg": ("1rem", "6px 18px", "1.1rem"),
    }
    font_size, padding, num_size = size_map.get(size, size_map["md"])

    return (
        f'<span style="'
        f'background:{bg};color:{color};border:1.5px solid {color};'
        f'border-radius:20px;padding:{padding};font-size:{font_size};'
        f'font-weight:700;letter-spacing:0.3px;display:inline-block;'
        f'line-height:1.4;white-space:nowrap;">'
        f'<span style="font-size:{num_size};font-weight:800;">{aqi_value}</span>'
        f'&nbsp;·&nbsp;{category}'
        f'</span>'
    )


def get_plotly_template() -> dict:
    """
    Return a consistent Plotly layout dict to apply to all charts.
    Call fig.update_layout(**get_plotly_template()) after creating a figure.
    """
    return {
        "template": "plotly_white",
        "font": {"family": "Inter, system-ui, -apple-system, sans-serif", "color": "#1A1A2E"},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(248,250,252,0.6)",
        "margin": {"l": 16, "r": 16, "t": 48, "b": 16},
        "xaxis": {"gridcolor": "#E8ECF0", "showgrid": True, "zeroline": False},
        "yaxis": {"gridcolor": "#E8ECF0", "showgrid": True, "zeroline": False},
        "hoverlabel": {
            "bgcolor": "#1A1A2E",
            "font_color": "#FFFFFF",
            "bordercolor": "#1A1A2E",
            "font_size": 13,
        },
        "legend": {
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "#E8ECF0",
            "borderwidth": 1,
            "font": {"size": 12},
        },
    }


# ── Global CSS Design System ──────────────────────────────────────────────────

def get_dashboard_css() -> str:
    """
    Return the full AirPulse CSS design system as a <style> block.

    Injected once per page render via st.markdown(get_dashboard_css(), unsafe_allow_html=True).
    Defines: CSS custom properties, layout utilities, KPI blocks, AQI badges,
    status indicators, chart card wrappers, sidebar brand, and micro-animations.
    """
    return """
<style>
/* ── 1. CSS Custom Properties (Design Tokens) ─────────────────────────── */
:root {
  --brand-blue:        #1E88E5;
  --brand-blue-dark:   #1565C0;
  --brand-blue-light:  #E3F2FD;
  --surface-0:         #FFFFFF;
  --surface-1:         #F8FAFC;
  --surface-2:         #F0F4F8;
  --border-color:      #E2E8F0;
  --text-primary:      #1A1A2E;
  --text-secondary:    #64748B;
  --text-muted:        #94A3B8;
  --radius-sm:         6px;
  --radius-md:         10px;
  --radius-lg:         16px;
  --shadow-sm:         0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
  --shadow-md:         0 4px 12px rgba(0,0,0,0.10), 0 2px 4px rgba(0,0,0,0.06);
  --shadow-lg:         0 10px 30px rgba(0,0,0,0.12), 0 4px 8px rgba(0,0,0,0.08);

  /* AQI Tier Colors */
  --aqi-good:          #00C853;
  --aqi-good-bg:       #E8F5E9;
  --aqi-moderate:      #FFD600;
  --aqi-moderate-bg:   #FFFDE7;
  --aqi-usg:           #FF6D00;
  --aqi-usg-bg:        #FFF3E0;
  --aqi-unhealthy:     #D50000;
  --aqi-unhealthy-bg:  #FFEBEE;
  --aqi-vunhealthy:    #6A1B9A;
  --aqi-vunhealthy-bg: #F3E5F5;
  --aqi-hazardous:     #4A0010;
  --aqi-hazardous-bg:  #FCE4EC;

  /* Status Colors */
  --status-ok:         #00C853;
  --status-warning:    #FF9800;
  --status-error:      #D50000;
  --status-info:       #1E88E5;
}

/* ── 2. Global Resets & Typography ──────────────────────────────────────── */
.stApp {
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Smooth transitions for all interactive elements */
button, .stButton > button, input, select {
  transition: all 0.18s ease !important;
}

/* ── 3. Sidebar Branding ─────────────────────────────────────────────────── */
.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 0 18px 0;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 16px;
}
.sidebar-brand-icon {
  font-size: 2rem;
  line-height: 1;
}
.sidebar-brand-text {
  display: flex;
  flex-direction: column;
}
.sidebar-brand-name {
  font-size: 1.25rem;
  font-weight: 800;
  color: var(--brand-blue);
  letter-spacing: -0.3px;
  line-height: 1.2;
}
.sidebar-brand-sub {
  font-size: 0.7rem;
  color: var(--text-muted);
  letter-spacing: 0.5px;
  text-transform: uppercase;
  font-weight: 500;
}

/* Sidebar city dot strip */
.city-strip {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 0;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 12px;
}
.city-strip-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--text-secondary);
}
.city-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 6px;
  flex-shrink: 0;
}

/* Sidebar status pill */
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.2px;
}
.status-pill-ok    { background: #E8F5E9; color: #2E7D32; }
.status-pill-warn  { background: #FFF3E0; color: #E65100; }
.status-pill-error { background: #FFEBEE; color: #C62828; }
.status-pill-info  { background: #E3F2FD; color: #1565C0; }

/* ── 4. Page Hero ─────────────────────────────────────────────────────────── */
.page-hero {
  background: var(--surface-0);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 24px 28px;
  margin-bottom: 20px;
  box-shadow: var(--shadow-sm);
  border-left: 4px solid var(--brand-blue);
}
.page-hero-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.3px;
  margin: 0 0 4px 0;
  line-height: 1.2;
}
.page-hero-sub {
  font-size: 0.85rem;
  color: var(--text-secondary);
  font-weight: 400;
  margin: 0;
}

/* ── 5. Card / Container Utilities ──────────────────────────────────────── */
.card {
  background: var(--surface-0);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 16px 18px;
  box-shadow: var(--shadow-sm);
}

.chart-card {
  background: var(--surface-0);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 4px;
  box-shadow: var(--shadow-sm);
  margin-bottom: 12px;
}

/* ── 6. KPI / Forecast Metric Blocks ────────────────────────────────────── */
.kpi-block {
  background: var(--surface-0);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 16px 18px 14px;
  box-shadow: var(--shadow-sm);
  position: relative;
  overflow: hidden;
  transition: box-shadow 0.18s ease, transform 0.18s ease;
}
.kpi-block:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}
.kpi-block::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 4px; height: 100%;
  border-radius: var(--radius-md) 0 0 var(--radius-md);
}
.kpi-stripe-good::before           { background: var(--aqi-good); }
.kpi-stripe-moderate::before       { background: var(--aqi-moderate); }
.kpi-stripe-usg::before            { background: var(--aqi-usg); }
.kpi-stripe-unhealthy::before      { background: var(--aqi-unhealthy); }
.kpi-stripe-very-unhealthy::before { background: var(--aqi-vunhealthy); }
.kpi-stripe-hazardous::before      { background: var(--aqi-hazardous); }

.kpi-label {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: var(--text-muted);
  margin-bottom: 6px;
}
.kpi-value {
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: -1px;
  line-height: 1;
  margin-bottom: 6px;
}
.kpi-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.3px;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.kpi-delta {
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-top: 4px;
}
.kpi-ci {
  font-size: 0.70rem;
  color: var(--text-secondary);
  margin-top: 2px;
  font-weight: 500;
}

/* AQI color classes for KPI badge backgrounds */
.badge-good           { background: var(--aqi-good-bg);       color: #1B5E20; }
.badge-moderate       { background: var(--aqi-moderate-bg);   color: #827717; }
.badge-usg            { background: var(--aqi-usg-bg);        color: #BF360C; }
.badge-unhealthy      { background: var(--aqi-unhealthy-bg);  color: #B71C1C; }
.badge-very-unhealthy { background: var(--aqi-vunhealthy-bg); color: #4A148C; }
.badge-hazardous      { background: var(--aqi-hazardous-bg);  color: var(--aqi-hazardous); }

/* ── 7. Status / Health Indicator Dots ──────────────────────────────────── */
.status-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 6px;
  vertical-align: middle;
  flex-shrink: 0;
}
.dot-ok      { background: var(--status-ok); }
.dot-warning { background: var(--status-warning); }
.dot-error   { background: var(--status-error); }

@keyframes pulse-ok {
  0%   { box-shadow: 0 0 0 0 rgba(0,200,83,0.5); }
  70%  { box-shadow: 0 0 0 6px rgba(0,200,83,0); }
  100% { box-shadow: 0 0 0 0 rgba(0,200,83,0); }
}
@keyframes pulse-warning {
  0%   { box-shadow: 0 0 0 0 rgba(255,152,0,0.5); }
  70%  { box-shadow: 0 0 0 6px rgba(255,152,0,0); }
  100% { box-shadow: 0 0 0 0 rgba(255,152,0,0); }
}
@keyframes pulse-error {
  0%   { box-shadow: 0 0 0 0 rgba(213,0,0,0.5); }
  70%  { box-shadow: 0 0 0 6px rgba(213,0,0,0); }
  100% { box-shadow: 0 0 0 0 rgba(213,0,0,0); }
}

.dot-ok.animated      { animation: pulse-ok 2s infinite; }
.dot-warning.animated { animation: pulse-warning 2s infinite; }
.dot-error.animated   { animation: pulse-error 1.5s infinite; }

/* Health card */
.health-card {
  background: var(--surface-0);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 14px 16px;
  box-shadow: var(--shadow-sm);
  display: flex;
  align-items: center;
  gap: 10px;
}
.health-card-icon { font-size: 1.4rem; }
.health-card-body { flex: 1; }
.health-card-label {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: var(--text-muted);
}
.health-card-value {
  font-size: 1rem;
  font-weight: 700;
  color: var(--text-primary);
}

/* ── 8. Alert Cards ──────────────────────────────────────────────────────── */
.alert-card {
  border-radius: var(--radius-md);
  padding: 14px 16px;
  margin-bottom: 10px;
  border: 1px solid;
  display: flex;
  align-items: flex-start;
  gap: 12px;
  box-shadow: var(--shadow-sm);
}
.alert-card-icon { font-size: 1.4rem; flex-shrink: 0; line-height: 1; }
.alert-card-body { flex: 1; }
.alert-card-city { font-size: 1rem; font-weight: 700; margin-bottom: 2px; }
.alert-card-aqi  { font-size: 0.8rem; font-weight: 600; margin-bottom: 4px; }
.alert-card-rec  { font-size: 0.76rem; color: var(--text-secondary); }

.alert-critical { background: #FFEBEE; border-color: #D50000; }
.alert-warning  { background: #FFF3E0; border-color: #FF6D00; }
.alert-caution  { background: #FFFDE7; border-color: #FFD600; }

/* ── 9. Toolbar / Control Row ────────────────────────────────────────────── */
.toolbar {
  background: var(--surface-1);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 12px 16px;
  margin-bottom: 16px;
  box-shadow: var(--shadow-sm);
}

/* ── 10. Info Strip ──────────────────────────────────────────────────────── */
.info-strip {
  background: var(--brand-blue-light);
  border-left: 3px solid var(--brand-blue);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  padding: 8px 14px;
  font-size: 0.8rem;
  color: var(--brand-blue-dark);
  font-weight: 500;
  margin-top: 8px;
}

/* ── 11. Drift Badge Pills ───────────────────────────────────────────────── */
.drift-pill {
  display: inline-block;
  background: var(--aqi-usg-bg);
  color: #BF360C;
  border: 1px solid #FF6D00;
  border-radius: 20px;
  padding: 2px 10px;
  font-size: 0.72rem;
  font-weight: 600;
  margin: 3px 3px 3px 0;
  font-family: monospace;
}
.drift-pill-ok {
  background: var(--aqi-good-bg);
  color: #1B5E20;
  border-color: var(--aqi-good);
}

/* ── 12. AQI Reference Color Swatch ─────────────────────────────────────── */
.aqi-swatch {
  display: inline-block;
  width: 14px; height: 14px;
  border-radius: 3px;
  vertical-align: middle;
  margin-right: 4px;
  border: 1px solid rgba(0,0,0,0.1);
}

/* ── 13. SHAP Analysis Callout ──────────────────────────────────────────── */
.shap-callout {
  background: #EEF2FF;
  border-left: 3px solid #4F46E5;
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  padding: 10px 14px;
  font-size: 0.8rem;
  color: #3730A3;
  line-height: 1.5;
}

/* ── 14. Metric Delta Override ───────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--surface-0);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 12px 14px !important;
  box-shadow: var(--shadow-sm);
}
[data-testid="metric-container"]:hover {
  box-shadow: var(--shadow-md);
}
[data-testid="stMetricValue"] {
  font-weight: 800;
  font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: var(--text-muted);
}

/* ── 15. Streamlit Component Polish ─────────────────────────────────────── */
/* Tabs */
[data-testid="stTabs"] [data-testid="stHorizontalBlock"] button {
  font-weight: 600 !important;
}

/* Expander */
.streamlit-expanderHeader {
  font-weight: 600;
  font-size: 0.88rem;
}

/* Dataframe */
[data-testid="stDataFrame"] {
  border-radius: var(--radius-md);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}

/* Sidebar */
[data-testid="stSidebar"] {
  border-right: 1px solid var(--border-color);
}

/* Primary button */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--brand-blue) 0%, var(--brand-blue-dark) 100%);
  border: none;
  border-radius: var(--radius-sm);
  font-weight: 600;
  letter-spacing: 0.2px;
  box-shadow: 0 2px 6px rgba(30,136,229,0.35);
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 4px 12px rgba(30,136,229,0.45);
  transform: translateY(-1px);
}
.stButton > button[kind="secondary"] {
  border-radius: var(--radius-sm);
  font-weight: 600;
}

/* Spinner / Status labels */
.stStatus {
  border-radius: var(--radius-md);
}

/* ── 16. Scrollbar Polish ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface-2); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
</style>
"""
