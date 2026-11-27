"""Airtel Telecom Analytics - Domain D: Customer Service dashboard.

Reader  : Head of Customer Service
Decision: prioritise service effort and understand ticket patterns and
          ticket-status relationships.

Every displayed number comes from a SQL query run through q().  Pandas is only
used to hold query results for plotting - no metric is calculated in Pandas.
"""

import math
import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
_env_db = os.environ.get("AIRTEL_DB_PATH")
_db_candidates = [
    Path(_env_db) if _env_db else None,
    ROOT / "data" / "raw" / "airtel.db",
    ROOT / "notebook" / "airtel.db",
    ROOT / "data" / "airtel.db",
]
DB_PATH = next((path for path in _db_candidates if path and path.exists()), None)
PERIOD = "Jan-Jun 2025"
MIN_GROUP = 100  # groups smaller than this show counts only, no rate

RED, DARK, GREY = "#E31E24", "#B5121B", "#FF9A9E"
MAROON = "#8B0000"
SUCCESS_GREEN = "#159447"

st.set_page_config(page_title="Airtel Customer Service", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
header[data-testid="stHeader"], footer {display:none;}
.stApp {background:#F5F6F8; color:#222222;}
.block-container {padding:0.25rem 0.75rem 0; max-width:100%;}
[data-testid="stVerticalBlock"] {gap:0.2rem;}
[data-testid="stSidebar"] {min-width:175px; max-width:195px; background:#242424;}
[data-testid="stSidebar"] > div {padding-top:0.45rem; background:#242424;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {color:#FFFFFF;}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background:#FFFFFF; border-color:#D9DDE3; border-radius:8px; color:#222222;
}
[data-testid="stSidebar"] [data-baseweb="select"] input {color:#222222;}
[data-testid="stSidebar"] [data-baseweb="select"] svg {fill:#B5121B;}
div[data-testid="stSelectbox"] label p {font-size:0.74rem; font-weight:600;}
div[data-testid="stCaption"] {margin:0; padding:0; line-height:1;}
div[data-testid="stCaption"] p {font-size:0.58rem; margin:0; color:#666666;}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background:#FFFFFF; border-color:#D9DDE3; border-radius:8px;
}
div[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
    border-color:#E31E24; box-shadow:0 0 0 1px #E31E24;
}
button[kind="primary"] {background:#E31E24; border-color:#E31E24; color:#FFFFFF;
                         border-radius:8px;}
button[kind="secondary"] {background:#E5E7EB; border-color:#D9DDE3; color:#222222;
                           border-radius:8px;}
.banner {display:flex; align-items:center; gap:14px; background:#8B0000;
         border-bottom:3px solid #E31E24; border-radius:9px; padding:6px 12px;
         box-shadow:0 2px 8px rgba(34,34,34,0.10);}
.logo {color:#FFFFFF; font-weight:800; font-size:1.45rem; letter-spacing:-1px;
       border-right:2px solid rgba(255,255,255,0.45); padding-right:14px; line-height:1.1;}
.ttl {font-size:1.1rem; font-weight:700; line-height:1.1; color:#FFFFFF;}
.sub {font-size:0.72rem; color:#FCE8E9;}
.card {border:1px solid #F3C4C7; border-left:4px solid #E31E24; border-radius:9px;
       padding:6px 9px; min-height:76px; background:#FFFFFF; margin-bottom:0.2rem;
       box-shadow:0 2px 7px rgba(34,34,34,0.06);}
.c-t {font-size:0.74rem; font-weight:600; color:#222222; display:flex;
      justify-content:space-between;}
.c-v {font-size:1.35rem; font-weight:700; line-height:1.15; color:#222222;}
.c-c {font-size:0.62rem; color:#666666; line-height:1.1;}
.tag {background:#FCE8E9; border-radius:5px; padding:0 5px; font-size:0.6rem;
      color:#8B0000;}
.info {background:#FFFFFF; border:1px solid #D9DDE3; border-left:4px solid #E31E24;
       padding:3px 8px; font-size:0.76rem; border-radius:6px; color:#222222;}
.note {font-size:0.62rem; color:#666666;}
div[role="tablist"] {border-bottom:1px solid #D9DDE3; gap:4px;}
div[data-testid="stTab"] {padding:2px 14px; height:32px; border-radius:6px 6px 0 0;
                          background:#FFFFFF; color:#222222;
                          border-bottom:2px solid transparent;}
div[data-testid="stTab"] p {color:#222222;}
div[data-testid="stTab"][aria-selected="true"] {
    background:#E31E24; color:#FFFFFF; border-bottom:2px solid #B5121B;
}
div[data-testid="stTab"][aria-selected="true"] p {color:#FFFFFF;}
div[data-testid="stTab"]:hover {color:#B5121B;}
[data-testid="stPlotlyChart"] {background:#FFFFFF; border-radius:8px;}
</style>
""", unsafe_allow_html=True)

if DB_PATH is None:
    searched = "<br>".join(str(path) for path in _db_candidates if path)
    st.error("Database not found. Searched:<br>" + searched +
             "<br>Copy airtel.db into `data/raw/` or set `AIRTEL_DB_PATH`. "
             "See README.md.")
    st.stop()


@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True,
                           check_same_thread=False)
    conn.create_function("sqrt", 1, lambda v: None if v is None else math.sqrt(v))
    conn.create_function("two_tailed_p", 1,
                         lambda z: None if z is None else math.erfc(abs(z) / math.sqrt(2)))
    return conn


@st.cache_data(show_spinner=False)
def q(sql, params=None):
    """Run a SQL query and return the result as a DataFrame."""
    return pd.read_sql_query(sql, get_conn(), params=params or {})


# ----------------------------------------------------------------------------
# SQL building blocks (all dashboard metrics are defined here)
# ----------------------------------------------------------------------------
FILTER = ("(:conn = 'All' OR c.connection_type = :conn) "
          "AND (:reg = 'All' OR ci.region = :reg)")
SUBSCRIBERS = ("customers c JOIN circles ci ON ci.circle_id = c.circle_id "
               "JOIN v_complaint_customer v ON v.customer_id = c.customer_id")
TICKETS = ("complaints t JOIN customers c ON c.customer_id = t.customer_id "
           "JOIN circles ci ON ci.circle_id = c.circle_id")

SQL_TOTAL = "SELECT COUNT(*) AS n FROM complaints"

# T7: two-tailed two-proportion z-test on churn, 'One open' vs 'All closed'.
SQL_T7 = """
WITH g AS (
  SELECT CASE WHEN v.unresolved_count = 1 THEN 'open1'
              WHEN v.complaint_count > 0 AND v.unresolved_count = 0 THEN 'closed' END AS grp,
         COUNT(*) AS n, SUM(c.is_churned) AS x
  FROM customers c JOIN v_complaint_customer v ON v.customer_id = c.customer_id
  GROUP BY 1),
a AS (SELECT o.n AS n1, o.x AS x1, k.n AS n2, k.x AS x2
      FROM g o, g k WHERE o.grp = 'open1' AND k.grp = 'closed'),
s AS (SELECT *, 1.0*x1/n1 AS p1, 1.0*x2/n2 AS p2, 1.0*(x1+x2)/(n1+n2) AS pp FROM a)
SELECT n1, x1, n2, x2, ROUND(100*p1, 1) AS rate1, ROUND(100*p2, 1) AS rate2,
       two_tailed_p((p1 - p2) / sqrt(pp*(1-pp)*(1.0/n1 + 1.0/n2))) AS p_value
FROM s
"""

SQL_OPEN = f"""
SELECT n, closed_n, open_n, ROUND(100.0*open_n/n, 1) AS open_pct,
       ROUND(100.0*closed_n/n, 1) AS closed_pct
FROM (SELECT COUNT(*) AS n, COALESCE(SUM(t.is_resolved), 0) AS closed_n,
             COALESCE(SUM(1 - t.is_resolved), 0) AS open_n
      FROM {TICKETS} WHERE {FILTER})
"""

SQL_CHURN = f"""
WITH g AS (
  SELECT CASE WHEN v.complaint_count = 0 THEN 'Never'
              WHEN v.unresolved_count = 0 THEN 'All closed'
              WHEN v.unresolved_count = 1 THEN 'One open' END AS grp,
         COUNT(*) AS n, SUM(c.is_churned) AS x
  FROM {SUBSCRIBERS} WHERE {FILTER} GROUP BY 1),
r AS (
  SELECT grp, n, x,
         CASE WHEN n >= {MIN_GROUP} THEN ROUND(100.0*x/n, 1) END AS rate
  FROM g WHERE grp IS NOT NULL)
SELECT grp, n, x, rate,
       CASE WHEN rate IS NOT NULL AND rate = MAX(rate) OVER () THEN 1 ELSE 0 END AS is_top
FROM r
ORDER BY CASE grp WHEN 'Never' THEN 1 WHEN 'All closed' THEN 2 ELSE 3 END
"""

SQL_CHURN_REF = f"""
SELECT COUNT(*) AS n, SUM(c.is_churned) AS x,
       ROUND(100.0*SUM(c.is_churned)/COUNT(*), 1) AS rate,
       SUM(CASE WHEN v.unresolved_count >= 2 THEN 1 ELSE 0 END) AS excluded
FROM {SUBSCRIBERS} WHERE {FILTER}
"""

SQL_CLOSURE = f"""
SELECT category, COUNT(*) AS n, SUM(t.is_resolved) AS closed,
       ROUND(100.0*SUM(t.is_resolved)/COUNT(*), 1) AS share
FROM {TICKETS} WHERE {FILTER}
  AND (:chart_conn = 'All' OR c.connection_type = :chart_conn)
  AND (:chart_cat = 'All' OR t.category = :chart_cat)
GROUP BY category ORDER BY share ASC
"""

SQL_CIRCLE = f"""
WITH r AS (
  SELECT ci.circle_name AS circle, COUNT(DISTINCT c.customer_id) AS subs,
         COUNT(t.complaint_id) AS tickets,
         ROUND(100.0*COUNT(t.complaint_id)/COUNT(DISTINCT c.customer_id), 1) AS rate
  FROM customers c JOIN circles ci ON ci.circle_id = c.circle_id
       LEFT JOIN complaints t ON t.customer_id = c.customer_id
  WHERE {FILTER}
    AND (:chart_conn = 'All' OR c.connection_type = :chart_conn)
    AND (:chart_circle = 'All' OR ci.circle_name = :chart_circle)
  GROUP BY ci.circle_id)
SELECT circle, subs, tickets, rate,
       circle || '  ' || rate || ' (' || tickets || '/' || subs || ')' AS label,
       CASE WHEN rate = MAX(rate) OVER () THEN 1 ELSE 0 END AS is_top
FROM r ORDER BY rate DESC
"""

SQL_CIRCLE_REF = f"""
SELECT ROUND(100.0*COUNT(t.complaint_id)/COUNT(DISTINCT c.customer_id), 1) AS rate,
       COUNT(t.complaint_id) AS tickets, COUNT(DISTINCT c.customer_id) AS subs
FROM customers c JOIN circles ci ON ci.circle_id = c.circle_id
     LEFT JOIN complaints t ON t.customer_id = c.customer_id
WHERE {FILTER}
"""

SQL_CATEGORIES = "SELECT DISTINCT category FROM complaints ORDER BY category"

SQL_D_CARDS = f"""
SELECT n, total, closed_n, open_n,
       ROUND(100.0*n/total, 1) AS pct_of_all,
       ROUND(100.0*closed_n/n, 1) AS closed_pct,
       ROUND(100.0*open_n/n, 1) AS open_pct
FROM (SELECT COUNT(*) AS total,
             COALESCE(SUM(t.category = :cat), 0) AS n,
             COALESCE(SUM(CASE WHEN t.category = :cat AND t.is_resolved = 1 THEN 1 ELSE 0 END), 0) AS closed_n,
             COALESCE(SUM(CASE WHEN t.category = :cat AND t.is_resolved = 0 THEN 1 ELSE 0 END), 0) AS open_n
      FROM {TICKETS} WHERE {FILTER})
"""

SQL_D_TREND = f"""
WITH m AS (SELECT DISTINCT substr(complaint_date, 1, 7) AS ym FROM complaints),
f AS (SELECT t.complaint_id, substr(t.complaint_date, 1, 7) AS ym
      FROM {TICKETS} WHERE {FILTER}
        AND (:trend_conn = 'All' OR c.connection_type = :trend_conn)
        AND (:trend_cat = 'All' OR t.category = :trend_cat)),
r AS (SELECT m.ym,
             substr('JanFebMarAprMayJunJulAugSepOctNovDec',
                    (CAST(substr(m.ym, 6, 2) AS INT) - 1) * 3 + 1, 3) AS month,
             COUNT(f.complaint_id) AS tickets
      FROM m LEFT JOIN f ON f.ym = m.ym
      WHERE (:trend_month = 'All' OR m.ym = :trend_month)
      GROUP BY m.ym)
SELECT ym, month, tickets,
       CASE WHEN tickets = MAX(tickets) OVER () THEN 1 ELSE 0 END AS is_peak
FROM r ORDER BY ym
"""

SQL_D_LEAVE = f"""
SELECT CASE WHEN t.is_resolved = 1 THEN 'Closed' ELSE 'Open' END AS status,
       COUNT(*) AS n, SUM(c.is_churned) AS left_n, SUM(1 - c.is_churned) AS stayed_n,
       CASE WHEN COUNT(*) >= {MIN_GROUP}
            THEN ROUND(100.0*SUM(c.is_churned)/COUNT(*), 1) END AS rate
FROM {TICKETS} WHERE {FILTER} AND (:cat = 'All' OR t.category = :cat)
GROUP BY status ORDER BY status DESC
"""

SQL_D_CIRCLE = f"""
SELECT ci.circle_name AS circle, COUNT(t.complaint_id) AS tickets
FROM customers c JOIN circles ci ON ci.circle_id = c.circle_id
LEFT JOIN complaints t
       ON t.customer_id = c.customer_id
      AND (:cat = 'All' OR t.category = :cat)
      AND (:circle_status = 'All'
           OR (:circle_status = 'Open' AND t.is_resolved = 0)
           OR (:circle_status = 'Closed' AND t.is_resolved = 1))
WHERE {FILTER}
GROUP BY ci.circle_id
ORDER BY tickets DESC, ci.circle_name
"""


# ----------------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------------
def card(title, value, caption, fixed=False):
    tag = '<span class="tag">FIXED - ignores filters</span>' if fixed else ""
    st.markdown(f'<div class="card"><div class="c-t"><span>{title}</span>{tag}</div>'
                f'<div class="c-v">{value}</div><div class="c-c">{caption}</div></div>',
                unsafe_allow_html=True)


def style(fig, height, title, xtitle=None, ytitle=None):
    fig.update_layout(
        height=height, margin=dict(l=5, r=5, t=30, b=5), showlegend=False,
        title=dict(text=title, font=dict(size=11, color="#222222"), x=0,
                   xanchor="left", y=0.97),
        font=dict(size=10, color="#222222"),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        xaxis=dict(title=xtitle, showgrid=True, gridcolor="#E7E9ED",
                   linecolor="#D9DDE3", tickfont=dict(size=9, color="#222222"),
                   title_font=dict(color="#222222"), automargin=True),
        yaxis=dict(title=ytitle, showgrid=True, gridcolor="#E7E9ED",
                   linecolor="#D9DDE3", tickfont=dict(size=9, color="#222222"),
                   title_font=dict(color="#222222"), automargin=True))
    return fig


def show(fig):
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )


CHART_TYPES = [
    "Bar Chart",
    "Horizontal Bar Chart",
    "Line Chart",
    "Area Chart",
    "Pie Chart",
    "Donut Chart",
    "Scatter Plot",
]
TREND_CHART_TYPES = [
    "Bar Chart",
    "Horizontal Bar Chart",
    "Line Chart",
    "Area Chart",
    "Scatter Plot",
]
COUNT_CHART_TYPES = [
    "Bar Chart",
    "Horizontal Bar Chart",
    "Pie Chart",
    "Donut Chart",
    "Scatter Plot",
]


def make_categorical_chart(chart_type, df, x_col, y_col, title, x_title=None,
                           y_title=None, height=360, text_col=None,
                           reference=None, reference_orientation="h",
                           highlight_col=None, hover_template=None):
    """Build a chart while preserving the dashboard's existing data and labels."""
    if text_col is not None:
        labels = df[text_col].tolist()
    else:
        labels = df[y_col].tolist()

    highlight = (
        df[highlight_col].fillna(0).astype(int).tolist()
        if highlight_col is not None
        else [0] * len(df)
    )
    colors = [RED if is_highlight else GREY for is_highlight in highlight]

    if chart_type in ("Pie Chart", "Donut Chart"):
        fig = px.pie(
            df,
            names=x_col,
            values=y_col,
            hole=0.55 if chart_type == "Donut Chart" else 0,
            color=x_col,
            color_discrete_sequence=[RED, DARK, GREY, "#8B0000", "#FCE8E9", "#666666"],
        )
        fig.update_traces(
            textinfo="label+percent",
            texttemplate="%{label}<br>%{percent}",
            hovertemplate=hover_template or "%{label}: %{value}<extra></extra>",
        )
    elif chart_type == "Horizontal Bar Chart":
        fig = go.Figure(go.Bar(
            y=df[x_col],
            x=df[y_col],
            orientation="h",
            marker_color=colors,
            text=labels,
            textposition="inside",
            insidetextanchor="start",
            hovertemplate=hover_template or "%{y}: %{x}<extra></extra>",
        ))
    elif chart_type == "Scatter Plot":
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            text=labels,
            color=highlight_col if highlight_col is not None else None,
            color_discrete_map={1: RED, 0: GREY},
        )
        fig.update_traces(
            marker=dict(size=10),
            textposition="top center",
            hovertemplate=hover_template,
        )
    elif chart_type == "Line Chart":
        fig = go.Figure(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines+markers+text",
            text=labels,
            textposition="top center",
            line=dict(color=DARK, width=3),
            marker=dict(size=9, color=colors),
            hovertemplate=hover_template or "%{x}: %{y}<extra></extra>",
        ))
    elif chart_type == "Area Chart":
        fig = go.Figure(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines+markers+text",
            text=labels,
            textposition="top center",
            line=dict(color=DARK, width=2),
            marker=dict(size=8, color=colors),
            fill="tozeroy",
            fillcolor="rgba(227,30,36,0.16)",
            hovertemplate=hover_template or "%{x}: %{y}<extra></extra>",
        ))
    else:
        fig = go.Figure(go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker_color=colors,
            text=labels,
            textposition="outside",
            cliponaxis=False,
            hovertemplate=hover_template or "%{x}: %{y}<extra></extra>",
        ))

    if reference is not None and chart_type not in ("Pie Chart", "Donut Chart", "Scatter Plot"):
        if chart_type == "Horizontal Bar Chart":
            fig.add_vline(x=reference, line_dash="dash", line_color=DARK)
        elif reference_orientation == "v":
            fig.add_vline(x=reference, line_dash="dash", line_color=DARK)
        else:
            fig.add_hline(y=reference, line_dash="dash", line_color=DARK)

    style(fig, height, title, x_title, y_title)
    fig.update_layout(
        showlegend=(chart_type in ("Pie Chart", "Donut Chart")),
        hoverlabel=dict(bgcolor="white"),
    )
    return fig


def make_status_chart(chart_type, df, title, height=340):
    if chart_type in ("Pie Chart", "Donut Chart"):
        fig = px.pie(
            df,
            names="status",
            values="n",
            hole=0.55 if chart_type == "Donut Chart" else 0,
            color_discrete_sequence=[DARK, RED],
        )
        fig.update_traces(
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} tickets (%{percent})<extra></extra>",
        )
    elif chart_type == "Horizontal Bar Chart":
        fig = go.Figure()
        fig.add_bar(y=df["status"], x=df["stayed_n"], orientation="h",
                    name="Stayed", marker_color=GREY,
                    text=df["stayed_n"].map(fmt), textposition="outside",
                    hovertemplate="%{y}: %{x} stayed<extra></extra>")
        fig.add_bar(y=df["status"], x=df["left_n"], orientation="h",
                    name="Left", marker_color=RED,
                    text=df["left_n"].map(fmt), textposition="outside",
                    hovertemplate="%{y}: %{x} left<extra></extra>")
        fig.update_layout(barmode="group")
    elif chart_type == "Scatter Plot":
        fig = px.scatter(
            df,
            x="stayed_n",
            y="left_n",
            text="status",
            size="n",
            color="status",
            color_discrete_map={"Closed": DARK, "Open": RED},
        )
        fig.update_traces(textposition="top center",
                          hovertemplate="%{text}<br>Stayed: %{x}<br>Left: %{y}<extra></extra>")
    elif chart_type == "Line Chart":
        fig = go.Figure()
        fig.add_scatter(x=df["status"], y=df["stayed_n"], mode="lines+markers",
                        name="Stayed", line=dict(color=GREY),
                        hovertemplate="%{x}: %{y} stayed<extra></extra>")
        fig.add_scatter(x=df["status"], y=df["left_n"], mode="lines+markers",
                        name="Left", line=dict(color=RED),
                        hovertemplate="%{x}: %{y} left<extra></extra>")
    elif chart_type == "Area Chart":
        fig = go.Figure()
        fig.add_scatter(x=df["status"], y=df["stayed_n"], mode="lines+markers",
                        name="Stayed", line=dict(color=GREY), fill="tozeroy",
                        hovertemplate="%{x}: %{y} stayed<extra></extra>")
        fig.add_scatter(x=df["status"], y=df["left_n"], mode="lines+markers",
                        name="Left", line=dict(color=RED),
                        hovertemplate="%{x}: %{y} left<extra></extra>")
    else:
        fig = go.Figure()
        fig.add_bar(x=df["status"], y=df["stayed_n"], name="Stayed",
                    marker_color=GREY, text=df["stayed_n"].map(fmt),
                    textposition="outside",
                    hovertemplate="%{x}: %{y} stayed<extra></extra>")
        fig.add_bar(x=df["status"], y=df["left_n"], name="Left",
                    marker_color=RED, text=df["left_n"].map(fmt),
                    textposition="outside",
                    hovertemplate="%{x}: %{y} left<extra></extra>")
        fig.update_layout(barmode="group")

    style(
        fig,
        height,
        title,
        None if chart_type in ("Pie Chart", "Donut Chart", "Horizontal Bar Chart") else "Ticket status",
        None if chart_type in ("Pie Chart", "Donut Chart") else "Ticket count",
    )
    fig.update_layout(
        showlegend=(chart_type not in ("Pie Chart", "Donut Chart")),
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=1,
                    xanchor="right", font=dict(size=10)),
    )
    return fig


def make_channel_chart(chart_type, df, title, height=185):
    if chart_type in ("Pie Chart", "Donut Chart"):
        totals = df.assign(total=df["closed_n"] + df["open_n"])
        fig = px.pie(
            totals,
            names="channel",
            values="total",
            hole=0.55 if chart_type == "Donut Chart" else 0,
            color_discrete_sequence=[RED, DARK, GREY, "#8B0000"],
        )
        fig.update_traces(
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} tickets (%{percent})<extra></extra>",
        )
    elif chart_type == "Horizontal Bar Chart":
        fig = go.Figure()
        fig.add_bar(y=df["channel"], x=df["closed_n"], orientation="h",
                    name="Closed", marker_color=DARK,
                    text=df["closed_n"].map(fmt), textposition="inside",
                    hovertemplate="%{y}: %{x} closed<extra></extra>")
        fig.add_bar(y=df["channel"], x=df["open_n"], orientation="h",
                    name="Open", marker_color=RED,
                    text=df["open_n"].map(fmt), textposition="inside",
                    hovertemplate="%{y}: %{x} open<extra></extra>")
        fig.update_layout(barmode="stack")
    elif chart_type == "Scatter Plot":
        fig = px.scatter(
            df,
            x="closed_n",
            y="open_n",
            text="channel",
            size="n",
            color_discrete_sequence=[RED],
        )
        fig.update_traces(textposition="top center",
                          hovertemplate="%{text}<br>Closed: %{x}<br>Open: %{y}<extra></extra>")
    elif chart_type == "Line Chart":
        fig = go.Figure()
        fig.add_scatter(x=df["channel"], y=df["closed_n"], mode="lines+markers",
                        name="Closed", line=dict(color=DARK),
                        hovertemplate="%{x}: %{y} closed<extra></extra>")
        fig.add_scatter(x=df["channel"], y=df["open_n"], mode="lines+markers",
                        name="Open", line=dict(color=RED),
                        hovertemplate="%{x}: %{y} open<extra></extra>")
    elif chart_type == "Area Chart":
        fig = go.Figure()
        fig.add_scatter(x=df["channel"], y=df["closed_n"], mode="lines+markers",
                        name="Closed", line=dict(color=DARK), fill="tozeroy",
                        hovertemplate="%{x}: %{y} closed<extra></extra>")
        fig.add_scatter(x=df["channel"], y=df["open_n"], mode="lines+markers",
                        name="Open", line=dict(color=RED),
                        hovertemplate="%{x}: %{y} open<extra></extra>")
    else:
        fig = go.Figure()
        fig.add_bar(x=df["channel"], y=df["closed_n"], name="Closed",
                    marker_color=DARK, text=df["closed_n"].map(fmt),
                    textposition="inside",
                    hovertemplate="%{x}: %{y} closed<extra></extra>")
        fig.add_bar(x=df["channel"], y=df["open_n"], name="Open",
                    marker_color=RED, text=df["open_n"].map(fmt),
                    textposition="inside",
                    hovertemplate="%{x}: %{y} open<extra></extra>")
        fig.update_layout(barmode="stack")

    style(fig, height, title,
          None if chart_type in ("Pie Chart", "Donut Chart", "Horizontal Bar Chart") else "Channel",
          None if chart_type in ("Pie Chart", "Donut Chart") else "Tickets")
    fig.update_layout(
        showlegend=(chart_type not in ("Pie Chart", "Donut Chart")),
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=1,
                    xanchor="right", font=dict(size=10)),
    )
    return fig


def fmt(n):
    return f"{int(n):,}"


def rate_text(rate, x, n):
    if pd.isna(rate):
        return f"n={fmt(n)}<br>(rate omitted, n<{MIN_GROUP})"
    return f"{rate:.1f}%<br>{fmt(x)} of {fmt(n)}"


# ----------------------------------------------------------------------------
# Filters (sidebar) and banner
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filters")
    conn_type = st.selectbox("Connection type", ["All", "Prepaid", "Postpaid"])
    region = st.selectbox("Region", ["All", "North", "South", "East", "West"])
    st.caption("Filters update every card and chart except those labelled FIXED.")

P = {"conn": conn_type, "reg": region}
CHART_CATEGORIES = ["All"] + q(SQL_CATEGORIES).category.tolist()
_actual_connections = q(
    "SELECT DISTINCT connection_type FROM customers "
    "WHERE connection_type IS NOT NULL ORDER BY connection_type"
).connection_type.tolist()
CHART_CONNECTIONS = ["All"] + _actual_connections
CHART_CUSTOMER_STATUSES = ["All"] + q(
    "SELECT DISTINCT CASE WHEN is_churned = 1 THEN 'Churned' ELSE 'Active' END AS status "
    "FROM customers ORDER BY status"
).status.tolist()
CHART_TICKET_STATUSES = ["All"] + q(
    "SELECT DISTINCT CASE WHEN is_resolved = 1 THEN 'Closed' ELSE 'Open' END AS status "
    "FROM complaints ORDER BY status"
).status.tolist()
CHART_LEAVING_STATUSES = ["All"] + q(
    "SELECT DISTINCT CASE WHEN is_churned = 1 THEN 'Left' ELSE 'Stayed' END AS status "
    "FROM customers ORDER BY status"
).status.tolist()
CHART_HISTORY = ["All"] + q(
    "SELECT DISTINCT CASE "
    "WHEN v.complaint_count = 0 THEN 'Never' "
    "WHEN v.unresolved_count = 0 THEN 'All closed' "
    "WHEN v.unresolved_count = 1 THEN 'One open' END AS history "
    "FROM customers c JOIN v_complaint_customer v ON v.customer_id = c.customer_id "
    "WHERE v.complaint_count = 0 OR v.unresolved_count IN (0, 1) "
    "ORDER BY CASE history WHEN 'Never' THEN 1 "
    "WHEN 'All closed' THEN 2 ELSE 3 END"
).history.tolist()
CHART_MONTHS = ["All"] + q(
    "SELECT DISTINCT substr(complaint_date, 1, 7) AS month "
    "FROM complaints ORDER BY month"
).month.tolist()


def compact_filter(label, options, key, default="All"):
    index = options.index(default) if default in options else 0
    st.caption(label)
    return st.selectbox(
        label,
        options,
        index=index,
        key=key,
        label_visibility="collapsed",
    )


def circle_filter_options(chart_connection):
    if conn_type != "All" and chart_connection not in ("All", conn_type):
        return ["All"]
    effective_connection = chart_connection if chart_connection != "All" else conn_type
    circles = q(
        "SELECT DISTINCT ci.circle_name AS circle "
        "FROM customers c JOIN circles ci ON ci.circle_id = c.circle_id "
        "WHERE (:conn = 'All' OR c.connection_type = :conn) "
        "AND (:reg = 'All' OR ci.region = :reg) ORDER BY ci.circle_name",
        {"conn": effective_connection, "reg": region},
    ).circle.tolist()
    return ["All"] + circles


if (conn_type, region) == ("All", "All"):
    SCOPE = "All connections | All regions"
else:
    SCOPE = f"{conn_type} connections | {region} region"

st.markdown('<div class="banner"><div class="logo">airtel</div><div>'
            '<div class="ttl">Airtel Telecom Analytics</div>'
            '<div class="sub">Domain D - Customer Service</div></div></div>',
            unsafe_allow_html=True)

DEFINITIONS = f"""
**Ticket** = one complaint record (not a unique subscriber). Period: {PERIOD}.
**Open / Closed** = `is_resolved` 0 / 1. **Closure share** = closed tickets / all tickets
in the group. **Tickets per 100 subscribers** = tickets / subscribers in the circle x 100
(subscribers = all customers in the dataset). **Complaint history** (per subscriber):
Never = 0 tickets; All closed = 1+ tickets, none open; One open = exactly 1 open ticket
(subscribers with 2+ open tickets are not in these three groups). **Churn** =
`is_churned` = 1. Groups under {MIN_GROUP} show counts only. All relationships are
**associations, not causation.**
"""


def info_line(text):
    c1, c2 = st.columns([9, 1.2])
    c1.markdown(f'<div class="info">{text}</div>', unsafe_allow_html=True)
    with c2.popover("Definitions"):
        st.markdown(DEFINITIONS)


tab_over, tab_detail = st.tabs(["Overview", "Detail"])

# ----------------------------------------------------------------------------
# OVERVIEW
# ----------------------------------------------------------------------------
with tab_over:
    info_line("This dashboard helps Head of Customer Service prioritise service effort "
              "and understand ticket patterns and ticket-status relationships. "
              f"<b>Showing: {SCOPE}</b>")

    total = q(SQL_TOTAL).iloc[0]
    t7 = q(SQL_T7).iloc[0]
    op = q(SQL_OPEN, P).iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Total tickets", fmt(total.n),
             f"Ticket records, {PERIOD}. All regions and connection types.", fixed=True)
    with c2:
        p = t7.p_value
        p_txt = "< 0.001" if p < 0.001 else f"{p:.4f}"
        card("T7: open ticket vs churn", f"p = {p_txt}",
             f"Two-tailed z-test: churn {t7.rate1}% (one open, n={fmt(t7.n1)}) vs "
             f"{t7.rate2}% (all closed, n={fmt(t7.n2)}). Association, not causation.",
             fixed=True)
    with c3:
        card("Open tickets", f"{fmt(op.open_n)} ({op.open_pct}%)",
             f"Open / all {fmt(op.n)} tickets in current selection, {PERIOD}.")

    left, mid, right = st.columns([5, 3.2, 4])

    with left:  # hero chart
        ch = q(SQL_CHURN, P)
        ref = q(SQL_CHURN_REF, P).iloc[0]
        history_col, customer_col, type_col = st.columns(
            [1, 1, 1.4], gap="small"
        )
        with history_col:
            history_filter = compact_filter(
                "Complaint history",
                CHART_HISTORY,
                "overview_churn_history",
            )
        with customer_col:
            customer_filter = compact_filter(
                "Customer status",
                CHART_CUSTOMER_STATUSES,
                "overview_churn_customer_status",
            )
        with type_col:
            churn_chart_type = compact_filter(
                "Chart type",
                CHART_TYPES,
                "overview_churn_chart_type",
            )

        if history_filter != "All":
            ch = ch[ch.grp == history_filter].copy()

        if ch.empty:
            st.info("No subscriber data for the selected complaint-history filter.")
        else:
            if customer_filter == "All":
                ch_plot = ch.assign(
                    chart_value=ch.rate.fillna(0),
                    chart_text=[
                        rate_text(r, x, n) for r, x, n in zip(ch.rate, ch.x, ch.n)
                    ],
                )
                has_rate = ch.rate.notna().any()
                top = ch.loc[ch.rate.idxmax()] if has_rate else ch.iloc[0]
                churn_title = (
                    f"Churn is highest for '{top.grp}' subscribers ({top.rate}%)"
                    if has_rate
                    else f"{top.grp} subscribers (insufficient data for rate)"
                )
                churn_y_title = "Churn rate (% of subscribers)"
                churn_reference = ref.rate
                churn_hover = "%{x}: %{y:.1f}%<extra></extra>"
            else:
                selected_count = (
                    ch.x if customer_filter == "Churned" else ch.n - ch.x
                )
                ch_plot = ch.assign(
                    chart_value=selected_count,
                    chart_text=selected_count.map(fmt),
                )
                ch_plot["is_top"] = (
                    ch_plot.chart_value == ch_plot.chart_value.max()
                ).astype(int)
                top = ch_plot.loc[ch_plot.chart_value.idxmax()]
                churn_title = f"{customer_filter} subscribers by complaint history"
                churn_y_title = f"{customer_filter} subscribers"
                churn_reference = None
                churn_hover = "%{x}: %{y} subscribers<extra></extra>"

            fig = make_categorical_chart(
                churn_chart_type,
                ch_plot,
                "grp",
                "chart_value",
                churn_title,
                "Complaint history",
                churn_y_title,
                text_col="chart_text",
                reference=churn_reference,
                reference_orientation="h",
                highlight_col="is_top",
                hover_template=churn_hover,
            )
            if customer_filter == "All":
                rate_max = ch.rate.max()
                if pd.isna(rate_max):
                    rate_max = 0
                fig.update_layout(
                    title=dict(
                        text=(f"{churn_title}"
                              f"<br><span style='font-size:10px'>Dashed = all subscribers "
                              f"{ref.rate}% ({fmt(ref.x)} of {fmt(ref.n)}); excludes "
                              f"{fmt(ref.excluded)} subscribers with 2+ open tickets</span>")
                    )
                )
                fig.update_yaxes(range=[0, max(rate_max, ref.rate) * 1.15])
            else:
                fig.update_yaxes(
                    range=[0, max(ch_plot.chart_value.max(), 1) * 1.15]
                )
            show(fig)

    with mid:
        category_col, connection_col, type_col = st.columns(
            [1, 1, 1.4], gap="small"
        )
        with category_col:
            closure_category = compact_filter(
                "Complaint category",
                CHART_CATEGORIES,
                "overview_closure_category",
            )
        with connection_col:
            closure_connection = compact_filter(
                "Connection type",
                CHART_CONNECTIONS,
                "overview_closure_connection",
                conn_type,
            )
        with type_col:
            closure_chart_type = compact_filter(
                "Chart type",
                CHART_TYPES,
                "overview_closure_chart_type",
            )
        cl = q(SQL_CLOSURE, {
            **P,
            "chart_conn": closure_connection,
            "chart_cat": closure_category,
        })
        if cl.empty:
            st.info("No closure data for the selected filters.")
        else:
            low = cl.iloc[0]
            effective_connection = (
                closure_connection if closure_connection != "All" else conn_type
            )
            closure_overall = q(
                SQL_OPEN, {**P, "conn": effective_connection}
            ).iloc[0]
            cl_plot = cl.assign(
                chart_text=[
                    f"{s}% ({fmt(c)}/{fmt(n)})"
                    for s, c, n in zip(cl.share, cl.closed, cl.n)
                ],
                is_low=[1 if i == 0 else 0 for i in range(len(cl))],
            )
            fig = make_categorical_chart(
                closure_chart_type,
                cl_plot,
                "category",
                "share",
                f"{low.category} tickets close least ({low.share}%)",
                "Category",
                "Closure share (%)",
                text_col="chart_text",
                reference=closure_overall.closed_pct,
                reference_orientation="v",
                highlight_col="is_low",
                hover_template="%{x}: %{y:.1f}% closed<extra></extra>",
            )
            fig.update_layout(
                title=dict(
                    text=(f"{low.category} tickets close least ({low.share}%)"
                          f"<br><span style='font-size:10px'>Closure share = closed/tickets; "
                          f"dashed = overall {closure_overall.closed_pct}%</span>")
                )
            )
            fig.update_xaxes(range=[0, 100])
            show(fig)

    with right:
        circle_connection_col, circle_col, type_col = st.columns(
            [1, 1, 1.4], gap="small"
        )
        with circle_connection_col:
            circle_connection = compact_filter(
                "Connection type",
                CHART_CONNECTIONS,
                "overview_circle_connection",
                conn_type,
            )
        with circle_col:
            circle_filter = compact_filter(
                "Circle",
                circle_filter_options(circle_connection),
                "overview_circle_name",
            )
        with type_col:
            circle_chart_type = compact_filter(
                "Chart type",
                CHART_TYPES,
                "overview_circle_chart_type",
            )
        ci = q(SQL_CIRCLE, {
            **P,
            "chart_conn": circle_connection,
            "chart_circle": circle_filter,
        })
        if ci.empty:
            st.info("No circle data for the selected filters.")
        else:
            effective_connection = (
                circle_connection if circle_connection != "All" else conn_type
            )
            cref = q(
                SQL_CIRCLE_REF, {**P, "conn": effective_connection}
            ).iloc[0]
            top_c = ci.iloc[0]
            fig = make_categorical_chart(
                circle_chart_type,
                ci,
                "label",
                "rate",
                f"{top_c.circle} files most tickets per 100 subscribers",
                "Circle",
                "Tickets per 100 subscribers",
                text_col="label",
                reference=cref.rate,
                reference_orientation="v",
                highlight_col="is_top",
                hover_template="%{x}: %{y:.1f}<extra></extra>",
            )
            fig.update_layout(
                title=dict(
                    text=(f"{top_c.circle} files most tickets per 100 subscribers"
                          f"<br><span style='font-size:10px'>Per 100 subs: tickets/subscribers; "
                          f"dashed = overall {cref.rate}</span>")
                )
            )
            fig.update_xaxes(range=[0, ci.rate.max() * 1.05])
            show(fig)

# ----------------------------------------------------------------------------
# DETAIL
# ----------------------------------------------------------------------------
with tab_detail:
    cat = st.selectbox("Complaint category", CHART_CATEGORIES[1:],
                       key="category")
    PC = {**P, "cat": cat}
    dc = q(SQL_D_CARDS, PC).iloc[0]
    k1, k2, k3 = st.columns(3, gap="small")
    with k1:
        card("Total tickets", fmt(dc.n),
             f"{cat} tickets: {dc.pct_of_all}% of {fmt(dc.total)} in selection, {PERIOD}.")
    with k2:
        card("Closed tickets", f"{fmt(dc.closed_n)} ({dc.closed_pct}%)",
             f"Closure share = closed / {fmt(dc.n)} {cat} tickets.")
    with k3:
        card("Open tickets", f"{fmt(dc.open_n)} ({dc.open_pct}%)",
             f"Open / {fmt(dc.n)} {cat} tickets.")

    info_line(f"Detail for <b>{cat}</b> complaints: ticket volume, closure and the "
              f"association with leaving. <b>Showing: {SCOPE}</b>")

    trend_col, status_col = st.columns(2, gap="small")
    with trend_col:
        trend_month_col, trend_category_col, trend_connection_col, trend_type_col = (
            st.columns([1, 1.1, 1.1, 1.7], gap="small")
        )
        with trend_month_col:
            trend_month = compact_filter(
                "Month", CHART_MONTHS, "detail_trend_month"
            )
        with trend_category_col:
            trend_category = compact_filter(
                "Category",
                CHART_CATEGORIES,
                "detail_trend_category",
                cat,
            )
        with trend_connection_col:
            trend_connection = compact_filter(
                "Connection",
                CHART_CONNECTIONS,
                "detail_trend_connection",
                conn_type,
            )
        with trend_type_col:
            trend_chart_type = compact_filter(
                "Chart type",
                TREND_CHART_TYPES,
                "detail_trend_chart_type",
            )

        tr = q(SQL_D_TREND, {
            **P,
            "trend_month": trend_month,
            "trend_cat": trend_category,
            "trend_conn": trend_connection,
        })
        peak = tr[tr.is_peak == 1].iloc[0]
        tr_plot = tr.assign(chart_text=tr.tickets.map(fmt))
        trend_category_label = (
            trend_category if trend_category != "All" else "All categories"
        )
        fig = make_categorical_chart(
            trend_chart_type,
            tr_plot,
            "month",
            "tickets",
            f"{trend_category_label} tickets peaked in {peak.month} "
            f"({fmt(peak.tickets)} tickets)",
            "Month (2025)",
            "Ticket count",
            height=170,
            text_col="chart_text",
            highlight_col="is_peak",
            hover_template="%{x}: %{y} tickets<extra></extra>",
        )
        fig.update_yaxes(range=[0, tr.tickets.max() * 1.25])
        show(fig)
    with status_col:
        status_filter_col, leaving_filter_col, status_category_col, status_type_col = (
            st.columns([1, 1.1, 1.1, 1.7], gap="small")
        )
        with status_filter_col:
            ticket_status_filter = compact_filter(
                "Ticket status",
                CHART_TICKET_STATUSES,
                "detail_status_ticket_status",
            )
        with leaving_filter_col:
            leaving_status_filter = compact_filter(
                "Leaving status",
                CHART_LEAVING_STATUSES,
                "detail_status_leaving_status",
            )
        with status_category_col:
            status_category = compact_filter(
                "Category",
                CHART_CATEGORIES,
                "detail_status_category",
                cat,
            )
        with status_type_col:
            status_chart_type = compact_filter(
                "Chart type",
                CHART_TYPES,
                "detail_status_chart_type",
            )

        lv = q(SQL_D_LEAVE, {**P, "cat": status_category})
        if ticket_status_filter != "All":
            lv = lv[lv.status == ticket_status_filter].copy()

        status_category_label = (
            status_category if status_category != "All" else "All categories"
        )
        if leaving_status_filter == "All":
            fig = make_status_chart(
                status_chart_type,
                lv,
                f"Open vs closed {status_category_label} tickets: "
                "who stayed and who left?",
                height=170,
            )
            if not lv.empty:
                fig.update_yaxes(range=[0, max(lv.stayed_n.max(), lv.left_n.max()) * 1.2])
        elif not lv.empty:
            value_column = "left_n" if leaving_status_filter == "Left" else "stayed_n"
            status_plot = lv.assign(
                chart_value=lv[value_column],
                chart_text=lv[value_column].map(fmt),
                is_open=[1 if value == "Open" else 0 for value in lv.status],
            )
            fig = make_categorical_chart(
                status_chart_type,
                status_plot,
                "status",
                "chart_value",
                f"{leaving_status_filter} customers by ticket status",
                "Ticket status",
                f"{leaving_status_filter} customers (tickets)",
                height=170,
                text_col="chart_text",
                highlight_col="is_open",
                hover_template="%{x}: %{y} tickets<extra></extra>",
            )
            fig.update_yaxes(
                range=[0, max(status_plot.chart_value.max(), 1) * 1.2]
            )
        else:
            st.info("No ticket data for the selected filters.")
            fig = None
        if fig is not None:
            show(fig)

    circle_col, selected_status_col = st.columns(2, gap="small")
    with circle_col:
        circle_category_col, circle_status_col, circle_type_col = st.columns(
            [1, 1.1, 1.7], gap="small"
        )
        with circle_category_col:
            circle_category = compact_filter(
                "Category",
                CHART_CATEGORIES,
                "detail_circle_category",
                cat,
            )
        with circle_status_col:
            circle_status = compact_filter(
                "Ticket status",
                CHART_TICKET_STATUSES,
                "detail_circle_ticket_status",
            )
        with circle_type_col:
            detail_circle_chart_type = compact_filter(
                "Chart type",
                COUNT_CHART_TYPES,
                "detail_circle_chart_type",
                "Horizontal Bar Chart",
            )
        circle_tickets = q(SQL_D_CIRCLE, {
            **P,
            "cat": circle_category,
            "circle_status": circle_status,
        })
        if circle_tickets.empty:
            st.info("No ticket data for the selected filters.")
        else:
            circle_tickets = circle_tickets.assign(
                ticket_label=circle_tickets.tickets.map(fmt),
                is_top=[
                    1 if index == 0 else 0 for index in range(len(circle_tickets))
                ],
            )
            category_title = (
                circle_category if circle_category != "All" else "All categories"
            )
            status_title = (
                "" if circle_status == "All" else f" ({circle_status.lower()} tickets)"
            )
            fig = make_categorical_chart(
                detail_circle_chart_type,
                circle_tickets,
                "circle",
                "tickets",
                f"{category_title} ticket volume by circle{status_title}",
                "Circle",
                "Tickets",
                height=150,
                text_col="ticket_label",
                highlight_col="is_top",
                hover_template="%{x}: %{y} tickets<extra></extra>",
            )
            show(fig)
    with selected_status_col:
        selected_category_col, selected_ticket_status_col, selected_type_col = (
            st.columns([1.1, 1, 1.7], gap="small")
        )
        with selected_category_col:
            selected_status_category = compact_filter(
                "Complaint category",
                CHART_CATEGORIES,
                "detail_selected_status_category",
                cat,
            )
        with selected_ticket_status_col:
            selected_ticket_status = compact_filter(
                "Ticket status",
                CHART_TICKET_STATUSES,
                "detail_selected_status_ticket_status",
            )
        with selected_type_col:
            selected_status_chart_type = compact_filter(
                "Chart type",
                COUNT_CHART_TYPES,
                "detail_selected_status_chart_type",
            )

        selected_status_data = q(
            SQL_D_LEAVE,
            {**P, "cat": selected_status_category},
        )
        if selected_ticket_status != "All":
            selected_status_data = selected_status_data[
                selected_status_data.status == selected_ticket_status
            ].copy()

        if selected_status_data.empty:
            st.info("No ticket data for the selected filters.")
        else:
            selected_status_data = selected_status_data.assign(
                status_label=selected_status_data.n.map(fmt),
                is_open=(selected_status_data.status == "Open").astype(int),
            )
            status_category_label = (
                selected_status_category
                if selected_status_category != "All"
                else "All categories"
            )
            selected_status_title = f"Selected Category Status: {status_category_label}"
            if selected_ticket_status != "All":
                selected_status_title += f" ({selected_ticket_status} tickets)"
            fig = make_categorical_chart(
                selected_status_chart_type,
                selected_status_data,
                "status",
                "n",
                selected_status_title,
                "Ticket status",
                "Ticket count",
                height=150,
                text_col="status_label",
                highlight_col="is_open",
                hover_template="%{x}: %{y} tickets<extra></extra>",
            )
            show(fig)

st.markdown(f'<div class="note">Period {PERIOD}. Ticket = complaint record. '
            'Rates under 100 records are shown as counts only. '
            'All patterns are associations, not causes.</div>', unsafe_allow_html=True)
