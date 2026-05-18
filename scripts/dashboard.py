## scripts/dashboard.py — Fixed: chart and box always match
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

C = {"teal":"#1D9E75","blue":"#378ADD","amber":"#BA7517","purple":"#534AB7"}
MONTHS = ["","Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Load model once ───────────────────────────────────
with open("data/gold/model_best.pkl",     "rb") as f: model    = pickle.load(f)
with open("data/gold/model_features.pkl", "rb") as f: features = pickle.load(f)

# ── Load CSVs ─────────────────────────────────────────
def load_data():
    gold = pd.read_csv("data/gold/fact_tourism_monthly.csv")
    pred = pd.read_csv("data/gold/predictions.csv")
    fi   = pd.read_csv("data/gold/feature_importance.csv")
    gold["date"] = pd.to_datetime(gold["date"])
    pred["date"] = pd.to_datetime(pred["date"])
    return gold, pred, fi

# ── Find column by keyword ────────────────────────────
def find_col(df, *keywords):
    for col in df.columns:
        if all(k.lower() in col.lower() for k in keywords):
            return col
    return None

# ── Run model for one month row ───────────────────────
def predict_one(m, l12, l24, nt, nr, sr_r, sr_t):
    row = {
        "lag_12":               l12,
        "lag_24":               l24,
        "ratio_1224":           l12 / (l24 + 1),
        "month":                m,
        "month_sin":            np.sin(2 * np.pi * m / 12),
        "month_cos":            np.cos(2 * np.pi * m / 12),
        "is_dry":               1 if m in [1,2,3,4,11,12] else 0,
        "is_peak":              1 if m in [11,12,1] else 0,
        "is_covid":             0,
        "nat_temp":             nt,
        "nat_rain":             nr,
        "total_rain_siem_reap": sr_r,
        "avg_temp_siem_reap":   sr_t,
        "avg_temp":             nt,
        "total_rain":           nr,
        "prev_month_arrivals":  l12,
    }
    df  = pd.DataFrame([row])
    ok  = [f for f in features if f in df.columns]
    return int(model.predict(df[ok])[0])

# ── Core: compute all 12 months, return plain lists ───
def compute_all_months(gold, sel_month, nat_temp, nat_rain, sr_rain, sr_temp):
    """
    Returns (labels, actuals, preds, live_pred, lag12_sel)
    - labels  : ["Jan 2026", ..., "Dec 2026"]
    - actuals : 2025 actual arrivals for each month
    - preds   : 2026 predicted arrivals for each month (model output)
    - live_pred : prediction for sel_month (same as preds[sel_month-1])
    - lag12_sel : 2025 actual for sel_month
    """
    # Weather fallback averages per month
    temp_cols = [c for c in gold.columns if "temp" in c.lower() and "avg" in c.lower()]
    rain_cols = [c for c in gold.columns if "rain" in c.lower() and "total" in c.lower()]
    sr_temp_c = find_col(gold, "siem_reap", "temp") or find_col(gold, "siem", "temp")
    sr_rain_c = find_col(gold, "siem_reap", "rain") or find_col(gold, "siem", "rain")

    def month_avg(col):
        if col and col in gold.columns:
            return gold.groupby("month")[col].mean()
        return pd.Series(0.0, index=range(1,13))

    nat_temp_avg = gold.groupby("month")[temp_cols].mean().mean(axis=1) if temp_cols else pd.Series(32.0, index=range(1,13))
    nat_rain_avg = gold.groupby("month")[rain_cols].mean().mean(axis=1) if rain_cols else pd.Series(100.0, index=range(1,13))
    sr_temp_avg  = month_avg(sr_temp_c)
    sr_rain_avg  = month_avg(sr_rain_c)

    labels, actuals, preds = [], [], []

    for m in range(1, 13):
        r25 = gold[(gold["year"]==2025) & (gold["month"]==m)]
        r24 = gold[(gold["year"]==2024) & (gold["month"]==m)]
        l12 = int(r25["arrivals"].values[0]) if len(r25)>0 else 500000
        l24 = int(r24["arrivals"].values[0]) if len(r24)>0 else 480000

        # Only the selected month uses slider values
        if m == sel_month:
            nt, nr, sr_r, sr_t = nat_temp, nat_rain, sr_rain, sr_temp
        else:
            nt   = float(nat_temp_avg.get(m, 32.0))
            nr   = float(nat_rain_avg.get(m, 100.0))
            sr_r = float(sr_rain_avg.get(m, 80.0))
            sr_t = float(sr_temp_avg.get(m, 32.0))

        p = predict_one(m, l12, l24, nt, nr, sr_r, sr_t)

        labels.append(f"{MONTHS[m]}")
        actuals.append(l12)
        preds.append(p)

    live_pred  = preds[sel_month - 1]
    lag12_sel  = actuals[sel_month - 1]
    return labels, actuals, preds, live_pred, lag12_sel

# ── Build forecast figure from precomputed lists ──────
def make_forecast_fig(labels, actuals, preds, sel_month, live_pred, lag12_sel):
    chg   = (live_pred - lag12_sel) / lag12_sel * 100
    col_a = [C["amber"] if i+1==sel_month else C["blue"]  for i in range(12)]
    col_p = ["#E8A000"  if i+1==sel_month else C["teal"]  for i in range(12)]

    fig = go.Figure()
    fig.add_bar(x=labels, y=actuals, name="2025 Actual",
                marker_color=col_a, opacity=0.85)
    fig.add_bar(x=labels, y=preds,   name="2026 Predicted",
                marker_color=col_p)
    fig.add_annotation(
        x=f"{MONTHS[sel_month]} 2026",
        y=max(live_pred, lag12_sel),
        text=(f"<b>{MONTHS[sel_month]} 2026</b><br>"
              f"{live_pred:,}<br>{chg:+.1f}% vs 2025"),
        showarrow=True, arrowhead=2, arrowcolor=C["amber"],
        font=dict(color=C["amber"], size=11),
        bgcolor="white", bordercolor=C["amber"], borderwidth=1, ay=-55
    )
    fig.update_layout(
        barmode="group", plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=55, b=30, l=10, r=10),
        xaxis=dict(tickangle=-30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        title=dict(
            text=(f"{MONTHS[sel_month]} 2026: predicted {live_pred:,} "
                  f"({chg:+.1f}% vs 2025)"),
            font=dict(size=12, color=C["amber"]), x=0.5
        )
    )
    return fig

# ── Weather scatter ───────────────────────────────────
def make_weather(gold, city="siem_reap"):
    nc = gold[~gold["year"].isin([2020,2021])].copy()
    rain_col = find_col(nc, city, "rain") or find_col(nc, city.split("_")[0], "rain")
    temp_col = find_col(nc, city, "temp") or find_col(nc, city.split("_")[0], "temp")
    hover    = [c for c in ["date","year","month"] if c in nc.columns]
    cm       = {"Dry":"#F4A460","Wet":"#4169E1"}

    def empty_fig(msg):
        f = go.Figure()
        f.add_annotation(text=msg, xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        f.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        return f

    if rain_col and "season" in nc.columns:
        fig_r = px.scatter(nc, x=rain_col, y="arrivals", color="season",
                           trendline="ols", hover_data=hover, color_discrete_map=cm)
        fig_r.update_xaxes(title_text="Rainfall (mm)")
    else:
        fig_r = empty_fig(f"No rain column for {city}")

    if temp_col and "season" in nc.columns:
        fig_t = px.scatter(nc, x=temp_col, y="arrivals", color="season",
                           trendline="ols", hover_data=hover, color_discrete_map=cm)
        fig_t.update_xaxes(title_text="Temperature (°C)")
    else:
        fig_t = empty_fig(f"No temp column for {city}")

    for f in [fig_r, fig_t]:
        f.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                        margin=dict(t=10,b=10))
        f.update_yaxes(title_text="Arrivals")
    return fig_r, fig_t


# ═══════════════════════════════════════════════════════
# APP + LAYOUT
# ═══════════════════════════════════════════════════════
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.FLATLY],
                title="Cambodia Tourism Analytics")

app.layout = dbc.Container([
    dcc.Interval(id="interval", interval=30_000, n_intervals=0),

    # ── Header ────────────────────────────────────────
    dbc.Row([dbc.Col([
        html.H2("Cambodia Tourism + Weather Analytics",
                className="text-white mb-0 fw-bold"),
        html.P("Open-Meteo · World Bank CSV · MOT PDF · Kafka · PySpark",
               className="text-white-50 mb-0 small"),
    ])], className="py-3 px-4 mb-3",
        style={"background":"#1a1a2e","borderRadius":"12px"}),

    dbc.Row(id="kpi-cards", className="mb-3 g-3"),

    dbc.Tabs([

        # ── TAB 1: Overview ───────────────────────────
        dbc.Tab(label="Overview", children=[
            dbc.Row([dbc.Col(dbc.Card([
                dbc.CardHeader("Annual Tourist Arrivals 2012-2025"),
                dbc.CardBody(dcc.Graph(id="annual-bar"))
            ], className="shadow-sm border-0 mt-3"))]),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Monthly Arrivals Trend 2012-2025"),
                    dbc.CardBody(dcc.Graph(id="trend-chart"))
                ], className="shadow-sm border-0 mt-3"), width=8),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Dry vs Wet Season"),
                    dbc.CardBody(dcc.Graph(id="season-chart"))
                ], className="shadow-sm border-0 mt-3"), width=4),
            ]),
        ]),

        # ── TAB 2: Weather ────────────────────────────
        dbc.Tab(label="Weather Analysis", children=[
            dbc.Row([dbc.Col([
                html.Label("Select City:", className="fw-semibold mt-3"),
                dcc.Dropdown(id="city-dd", clearable=False, className="mb-3",
                    value="siem_reap",
                    options=[
                        {"label":"Phnom Penh",    "value":"phnom_penh"},
                        {"label":"Siem Reap",     "value":"siem_reap"},
                        {"label":"Sihanoukville", "value":"sihanoukville"},
                    ]),
            ], width=4)]),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("Rainfall vs Arrivals"),
                    dbc.CardBody(dcc.Graph(id="rain-chart"))],
                    className="shadow-sm border-0"), width=6),
                dbc.Col(dbc.Card([dbc.CardHeader("Temperature vs Arrivals"),
                    dbc.CardBody(dcc.Graph(id="temp-chart"))],
                    className="shadow-sm border-0"), width=6),
            ], className="mt-2"),
            dbc.Row([dbc.Col(dbc.Card([
                dbc.CardHeader("Feature Importance"),
                dbc.CardBody(dcc.Graph(id="fi-chart"))
            ], className="shadow-sm border-0 mt-3"))]),
            dbc.Row([dbc.Col(dbc.Alert([
                html.H6("Key Finding", className="alert-heading"),
                html.P("lag_12 (same month last year) explains ~80% of arrivals. "
                       "Weather adds less than 2% to model accuracy.", className="mb-0")
            ], color="info", className="mt-3"))]),
        ]),

        # ── TAB 3: ML Prediction ──────────────────────
        dbc.Tab(label="ML Prediction", children=[
            dbc.Row([
                dbc.Col([dbc.Card([
                    dbc.CardHeader("Input Parameters"),
                    dbc.CardBody([
                        html.Label("Month:"),
                        dcc.Slider(id="sl-month", min=1, max=12, step=1, value=6,
                            marks={i:m for i,m in enumerate(
                                ["J","F","M","A","M","J",
                                 "J","A","S","O","N","D"],1)},
                            className="mb-4"),
                        html.Label("Avg Temperature (°C):"),
                        dcc.Slider(id="sl-temp", min=25, max=38, step=0.5, value=32,
                            marks={25:"25°",30:"30°",35:"35°",38:"38°"},
                            className="mb-4"),
                        html.Label("Total Rainfall (mm):"),
                        dcc.Slider(id="sl-rain", min=0, max=600, step=10, value=100,
                            marks={0:"0",200:"200",400:"400",600:"600"},
                            className="mb-4"),
                        html.Label("Siem Reap Rain (mm):"),
                        dcc.Slider(id="sl-sr-rain", min=0, max=500, step=10, value=80,
                            marks={0:"0",200:"200",400:"400",500:"500"},
                            className="mb-4"),
                        html.Label("Siem Reap Temp (°C):"),
                        dcc.Slider(id="sl-sr-temp", min=25, max=38, step=0.5, value=32,
                            marks={25:"25°",30:"30°",35:"35°",38:"38°"},
                            className="mb-4"),
                        html.Div(id="pred-out"),
                    ])
                ], className="shadow-sm border-0 mt-3")], width=4),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("2025 Actual vs 2026 Predicted — All 12 Months"),
                        dbc.CardBody(dcc.Graph(id="fc-chart"))
                    ], className="shadow-sm border-0 mt-3"),
                    dbc.Card([
                        dbc.CardHeader("Model Validation — 2019 Test Set"),
                        dbc.CardBody(dcc.Graph(id="val-chart"))
                    ], className="shadow-sm border-0 mt-3"),
                ], width=8),
            ]),
        ]),
    ]),

    html.Hr(className="mt-4"),
    html.P("Cambodia Tourism Analytics | ITC Data Engineering | "
           "Kafka + PySpark + Gold Layer + RandomForest",
           className="text-muted text-center small pb-2"),

], fluid=True, className="px-4 py-3",
   style={"backgroundColor":"#f0f2f6","minHeight":"100vh"})


# ═══════════════════════════════════════════════════════
# CALLBACK 1 — KPI Cards
# ═══════════════════════════════════════════════════════
@app.callback(Output("kpi-cards","children"), Input("interval","n_intervals"))
def refresh_kpis(_):
    try:
        gold, _, _ = load_data()
        a24 = int(gold[gold["year"]==2024]["arrivals"].sum())
        a23 = int(gold[gold["year"]==2023]["arrivals"].sum())
        a25 = int(gold[gold["year"]==2025]["arrivals"].sum())
        g   = (a24-a23)/a23*100
    except Exception as e:
        return [dbc.Col(dbc.Alert(f"Error: {e}", color="danger"))]
    return [
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H6("2024 Total Arrivals",className="text-muted mb-1 small"),
            html.H4(f"{a24/1e6:.2f}M",className="fw-bold mb-0",style={"color":C["teal"]}),
            html.Small(f"+{g:.1f}% vs 2023",className="text-success fw-semibold"),
        ])],className="shadow-sm border-0"),width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H6("2025 Total",className="text-muted mb-1 small"),
            html.H4(f"{a25/1e6:.2f}M",className="fw-bold mb-0",style={"color":C["blue"]}),
            html.Small("Latest available data",className="text-muted"),
        ])],className="shadow-sm border-0"),width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H6("All-Time Record",className="text-muted mb-1 small"),
            html.H4("708,038",className="fw-bold mb-0",style={"color":C["amber"]}),
            html.Small("December 2019",className="text-muted"),
        ])],className="shadow-sm border-0"),width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H6("ML Model R²",className="text-muted mb-1 small"),
            html.H4("0.8906",className="fw-bold mb-0",style={"color":C["purple"]}),
            html.Small("RandomForest · MAE=15,755",className="text-muted"),
        ])],className="shadow-sm border-0"),width=3),
    ]


# ═══════════════════════════════════════════════════════
# CALLBACK 2 — Overview
# ═══════════════════════════════════════════════════════
@app.callback(
    Output("annual-bar",   "figure"),
    Output("trend-chart",  "figure"),
    Output("season-chart", "figure"),
    Input("interval","n_intervals")
)
def refresh_overview(_):
    def empty(msg=""):
        f = go.Figure()
        if msg:
            f.add_annotation(text=msg, xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
        f.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        return f
    try:
        gold, _, _ = load_data()
    except Exception as e:
        return empty(str(e)), empty(str(e)), empty(str(e))

    nc     = gold[~gold["year"].isin([2020,2021])]
    annual = gold.groupby("year")["arrivals"].sum().reset_index()
    cb     = ["#E24B4A" if y in [2020,2021] else
               C["teal"] if y>=2022 else C["blue"] for y in annual["year"]]

    fig_a = go.Figure(go.Bar(
        x=annual["year"], y=annual["arrivals"], marker_color=cb,
        text=(annual["arrivals"]/1e6).round(2).astype(str)+"M",
        textposition="outside"))
    fig_a.add_annotation(x=2020, y=annual["arrivals"].max()*0.25,
        text="COVID", showarrow=True, arrowhead=2, font=dict(color="red"))
    fig_a.update_layout(plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20,b=10,l=10,r=20),
        xaxis=dict(tickmode="linear"), yaxis_title="Arrivals")

    fig_t = px.line(gold, x="date", y="arrivals",
                    color_discrete_sequence=[C["blue"]])
    fig_t.add_vrect(x0="2020-01-01", x1="2022-06-01", fillcolor="red", opacity=0.08,
                    annotation_text="COVID", annotation_position="top left")
    fig_t.update_layout(plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10,b=10,l=10,r=10),
        xaxis_title="Date", yaxis_title="Monthly Arrivals")

    if "season" in nc.columns:
        s = nc.groupby("season")["arrivals"].sum().reset_index()
        fig_s = px.pie(s, names="season", values="arrivals", hole=0.4, color="season",
                       color_discrete_map={"Dry":"#F4A460","Wet":"#4169E1"})
    else:
        fig_s = empty("No season column")
    fig_s.update_layout(margin=dict(t=10,b=10,l=10,r=10))

    return fig_a, fig_t, fig_s


# ═══════════════════════════════════════════════════════
# CALLBACK 3 — Weather
# ═══════════════════════════════════════════════════════
@app.callback(
    Output("rain-chart","figure"),
    Output("temp-chart","figure"),
    Output("fi-chart",  "figure"),
    Input("city-dd",  "value"),
    Input("interval", "n_intervals")
)
def refresh_weather(city, _):
    def empty(msg=""):
        f = go.Figure()
        if msg:
            f.add_annotation(text=msg, xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
        f.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        return f
    try:
        gold, _, fi = load_data()
    except Exception as e:
        return empty(str(e)), empty(str(e)), empty(str(e))

    fig_r, fig_t = make_weather(gold, city)

    top = fi.head(10).sort_values("importance")
    fig_fi = px.bar(top, x="importance", y="feature", orientation="h",
                    color="importance", color_continuous_scale="Teal",
                    text=top["importance"].round(3))
    fig_fi.update_traces(textposition="outside")
    fig_fi.update_layout(plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10,b=10,l=10,r=10), coloraxis_showscale=False)
    return fig_r, fig_t, fig_fi


# ═══════════════════════════════════════════════════════
# CALLBACK 4 — ML chart + prediction box
# Single callback → chart and box ALWAYS use identical numbers
# ═══════════════════════════════════════════════════════
@app.callback(
    Output("fc-chart",  "figure"),
    Output("val-chart", "figure"),
    Output("pred-out",  "children"),
    Input("interval",    "n_intervals"),
    Input("sl-month",    "value"),
    Input("sl-temp",     "value"),
    Input("sl-rain",     "value"),
    Input("sl-sr-rain",  "value"),
    Input("sl-sr-temp",  "value"),
)
def refresh_ml(_, month, nat_temp, nat_rain, sr_rain, sr_temp):
    def empty(msg=""):
        f = go.Figure()
        if msg:
            f.add_annotation(text=msg, xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
        f.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        return f

    try:
        gold, pred, _ = load_data()

        # ── ONE call computes everything ──────────────
        labels, actuals, preds, live_pred, lag12_sel = compute_all_months(
            gold, month, nat_temp, nat_rain, sr_rain, sr_temp)

        # ── Bar chart built from same lists ───────────
        fig_fc = make_forecast_fig(
            labels, actuals, preds, month, live_pred, lag12_sel)

    except Exception as e:
        fig_fc    = empty(f"Forecast error: {e}")
        live_pred = 0
        lag12_sel = 1

    # ── Validation chart ──────────────────────────────
    try:
        fig_val = go.Figure()
        fig_val.add_scatter(x=pred["date"], y=pred["actual"], name="Actual",
            mode="lines+markers", line=dict(color=C["blue"], width=2))
        fig_val.add_scatter(x=pred["date"], y=pred["predicted"], name="Predicted",
            mode="lines+markers", line=dict(color=C["teal"], width=2, dash="dash"),
            marker=dict(symbol="diamond"))
        fig_val.update_layout(plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=30,b=10,l=10,r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            title=dict(text="R²=0.89 | MAE=15,755 | Avg error 2.9%",
                       font=dict(size=12, color=C["teal"]), x=0.5))
    except Exception as e:
        fig_val = empty(f"Validation error: {e}")

    # ── Prediction box — same live_pred as bar chart ──
    try:
        r24   = gold[(gold["year"]==2024) & (gold["month"]==month)]
        lag24 = int(r24["arrivals"].values[0]) if len(r24)>0 else 480000
        chg   = (live_pred - lag12_sel) / lag12_sel * 100
        color = "success" if chg >= 0 else "danger"
        pred_box = dbc.Alert([
            html.Small(f"Predicted arrivals — 2026 {MONTHS[month]}",
                       className="d-block text-center mb-1"),
            html.H3(f"{live_pred:,}", className="fw-bold text-center mb-1"),
            html.H5(f"{chg:+.1f}% vs 2025", className="text-center mb-1"),
            html.Hr(className="my-2"),
            html.Small(f"lag_12 (2025): {lag12_sel:,}", className="d-block text-center"),
            html.Small(f"lag_24 (2024): {lag24:,}",     className="d-block text-center"),
        ], color=color, className="mt-3")
    except Exception as e:
        pred_box = dbc.Alert(f"Error: {e}", color="warning", className="mt-3")

    return fig_fc, fig_val, pred_box


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, port=8050)