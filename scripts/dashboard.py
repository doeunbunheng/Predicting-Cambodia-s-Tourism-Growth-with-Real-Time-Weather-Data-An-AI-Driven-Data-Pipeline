## scripts/dashboard.py — Fixed Version
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── LOAD DATA ─────────────────────────────────────────
gold  = pd.read_csv("data/gold/fact_tourism_monthly.csv")
pred  = pd.read_csv("data/gold/predictions.csv")
p2026 = pd.read_csv("data/gold/predictions_2026.csv")
fi    = pd.read_csv("data/gold/feature_importance.csv")
gold["date"] = pd.to_datetime(gold["date"])
pred["date"] = pd.to_datetime(pred["date"])

with open("data/gold/model_best.pkl",     "rb") as f: model    = pickle.load(f)
with open("data/gold/model_features.pkl", "rb") as f: features = pickle.load(f)

# ── KPI ───────────────────────────────────────────────
arr_2024 = int(gold[gold["year"]==2024]["arrivals"].sum())
arr_2023 = int(gold[gold["year"]==2023]["arrivals"].sum())
arr_2025 = int(gold[gold["year"]==2025]["arrivals"].sum())
growth   = (arr_2024 - arr_2023) / arr_2023 * 100
annual   = gold.groupby("year")["arrivals"].sum().reset_index()
nc       = gold[~gold["year"].isin([2020,2021])]

C = {"teal":"#1D9E75","blue":"#378ADD","amber":"#BA7517","purple":"#534AB7"}

# ══════════════════════════════════════════════════════
# PRE-COMPUTE ALL STATIC CHARTS AT STARTUP
# ══════════════════════════════════════════════════════

# Annual bar
colors_bar = ["#E24B4A" if y in [2020,2021] else
               C["teal"] if y >= 2022 else C["blue"]
               for y in annual["year"]]
fig_annual = go.Figure(go.Bar(
    x=annual["year"], y=annual["arrivals"],
    marker_color=colors_bar,
    text=(annual["arrivals"]/1e6).round(2).astype(str)+"M",
    textposition="outside"
))
fig_annual.add_annotation(x=2020,y=1500000,text="🦠 COVID",showarrow=True,arrowhead=2,font=dict(color="red"))
fig_annual.add_annotation(x=2024,y=7300000,text="✅ Recovery",showarrow=True,arrowhead=2,font=dict(color=C["teal"]))
fig_annual.update_layout(plot_bgcolor="white",paper_bgcolor="white",
    margin=dict(t=20,b=10,l=10,r=20),
    xaxis=dict(tickmode="linear"),yaxis_title="Arrivals")

# Monthly trend
fig_trend = px.line(gold, x="date", y="arrivals", color_discrete_sequence=[C["blue"]])
fig_trend.add_vrect(x0="2020-01-01",x1="2022-06-01",fillcolor="red",opacity=0.08,
                    annotation_text="COVID",annotation_position="top left")
fig_trend.update_layout(plot_bgcolor="white",paper_bgcolor="white",
    margin=dict(t=10,b=10,l=10,r=10),xaxis_title="Date",yaxis_title="Monthly Arrivals")

# Season pie
s = nc.groupby("season")["arrivals"].sum().reset_index()
fig_season = px.pie(s, names="season", values="arrivals", hole=0.4,
                    color="season",
                    color_discrete_map={"Dry":"#F4A460","Wet":"#4169E1"})
fig_season.update_layout(margin=dict(t=10,b=10,l=10,r=10))

# Feature importance
top = fi.head(10).sort_values("importance")
fig_fi = px.bar(top, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="Teal",
                text=top["importance"].round(3))
fig_fi.update_traces(textposition="outside")
fig_fi.update_layout(plot_bgcolor="white",paper_bgcolor="white",
    margin=dict(t=10,b=10,l=10,r=10),coloraxis_showscale=False)

# Forecast 2026
fig_forecast = go.Figure()
fig_forecast.add_bar(x=p2026["date"],y=p2026["actual_2025"],
                     name="2025 Actual",marker_color=C["blue"])
fig_forecast.add_bar(x=p2026["date"],y=p2026["predicted_2026"],
                     name="2026 Predicted",marker_color=C["teal"])
fig_forecast.update_layout(barmode="group",plot_bgcolor="white",paper_bgcolor="white",
    margin=dict(t=10,b=10,l=10,r=10),
    legend=dict(orientation="h",yanchor="bottom",y=1.02))

# Validation chart
fig_val = go.Figure()
fig_val.add_scatter(x=pred["date"],y=pred["actual"],name="Actual",
                    mode="lines+markers",line=dict(color=C["blue"],width=2))
fig_val.add_scatter(x=pred["date"],y=pred["predicted"],name="Predicted",
                    mode="lines+markers",line=dict(color=C["teal"],width=2,dash="dash"),
                    marker=dict(symbol="diamond"))
fig_val.update_layout(plot_bgcolor="white",paper_bgcolor="white",
    margin=dict(t=30,b=10,l=10,r=10),
    legend=dict(orientation="h",yanchor="bottom",y=1.02),
    title=dict(text="R²=0.89 | MAE=15,755 | Avg error 2.9%",
               font=dict(size=12,color=C["teal"]),x=0.5))

# Default weather charts (Siem Reap)
def make_weather(city="siem_reap"):
    r = nc.copy()
    fig_r = px.scatter(r,x=f"total_rain_{city}",y="arrivals",color="season",
                       trendline="ols",hover_data=["date","year","month"],
                       color_discrete_map={"Dry":"#F4A460","Wet":"#4169E1"})
    fig_r.update_layout(plot_bgcolor="white",paper_bgcolor="white",
        margin=dict(t=10,b=10),xaxis_title="Rainfall (mm)",yaxis_title="Arrivals")
    fig_t = px.scatter(r,x=f"avg_temp_{city}",y="arrivals",color="season",
                       trendline="ols",hover_data=["date","year","month"],
                       color_discrete_map={"Dry":"#F4A460","Wet":"#4169E1"})
    fig_t.update_layout(plot_bgcolor="white",paper_bgcolor="white",
        margin=dict(t=10,b=10),xaxis_title="Temperature (°C)",yaxis_title="Arrivals")
    return fig_r, fig_t

fig_rain, fig_temp = make_weather()

# ── APP ───────────────────────────────────────────────
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.FLATLY],
                title="Cambodia Tourism Analytics")

# ── LAYOUT ────────────────────────────────────────────
app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H2("🇰🇭 Cambodia Tourism + Weather Analytics",
                    className="text-white mb-0 fw-bold"),
            html.P("Open-Meteo API · World Bank CSV · MOT Official PDF · Kafka · PySpark",
                   className="text-white-50 mb-0 small"),
        ])
    ], className="py-3 px-4 mb-3",
       style={"background":"#1a1a2e","borderRadius":"12px"}),

    # KPI Cards
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H6("2024 Total Arrivals",className="text-muted mb-1 small"),
            html.H4(f"{arr_2024/1e6:.2f}M",className="fw-bold mb-0",style={"color":C["teal"]}),
            html.Small(f"+{growth:.1f}% vs 2023",className="text-success fw-semibold"),
        ])],className="shadow-sm border-0"),width=3),

        dbc.Col(dbc.Card([dbc.CardBody([
            html.H6("2025 Jan-Sep Total",className="text-muted mb-1 small"),
            html.H4(f"{arr_2025/1e6:.2f}M",className="fw-bold mb-0",style={"color":C["blue"]}),
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
    ], className="mb-3 g-3"),

    # Tabs
    dbc.Tabs([

        # TAB 1: OVERVIEW
        dbc.Tab(label=" Overview", children=[
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Annual Tourist Arrivals 2012-2025"),
                    dbc.CardBody(dcc.Graph(figure=fig_annual,id="annual-bar"))
                ],className="shadow-sm border-0 mt-3")),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Monthly Arrivals Trend 2012-2025"),
                    dbc.CardBody(dcc.Graph(figure=fig_trend,id="trend-chart"))
                ],className="shadow-sm border-0 mt-3"),width=8),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Dry vs Wet Season"),
                    dbc.CardBody(dcc.Graph(figure=fig_season,id="season-chart"))
                ],className="shadow-sm border-0 mt-3"),width=4),
            ]),
        ]),

        # TAB 2: WEATHER
        dbc.Tab(label=" Weather Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select City:",className="fw-semibold mt-3"),
                    dcc.Dropdown(
                        id="city-dd",
                        options=[
                            {"label":"Phnom Penh",    "value":"phnom_penh"},
                            {"label":"Siem Reap",     "value":"siem_reap"},
                            {"label":"Sihanoukville", "value":"sihanoukville"},
                        ],
                        value="siem_reap", clearable=False, className="mb-3"
                    ),
                ],width=4),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Rainfall vs Arrivals"),
                    dbc.CardBody(dcc.Graph(figure=fig_rain,id="rain-chart"))
                ],className="shadow-sm border-0"),width=6),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Temperature vs Arrivals"),
                    dbc.CardBody(dcc.Graph(figure=fig_temp,id="temp-chart"))
                ],className="shadow-sm border-0"),width=6),
            ],className="mt-2"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Feature Importance — What drives tourist arrivals?"),
                    dbc.CardBody(dcc.Graph(figure=fig_fi,id="fi-chart"))
                ],className="shadow-sm border-0 mt-3")),
            ]),
            dbc.Row([
                dbc.Col(dbc.Alert([
                    html.H6(" Key Finding",className="alert-heading"),
                    html.P("lag_12 (same month last year) explains 80% of arrivals — "
                           "tourists follow strong seasonal patterns regardless of weather. "
                           "Weather features add <2% to model accuracy.",className="mb-0")
                ],color="info",className="mt-3")),
            ]),
        ]),

        # TAB 3: PREDICTION
        dbc.Tab(label=" ML Prediction", children=[
            dbc.Row([
                # Input
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(" Input Parameters"),
                        dbc.CardBody([
                            html.Label("Month (1=Jan ... 12=Dec):"),
                            dcc.Slider(id="sl-month",min=1,max=12,step=1,value=6,
                                marks={i:m for i,m in enumerate(
                                    ["","J","F","M","A","M","J",
                                     "J","A","S","O","N","D"],1)},
                                className="mb-4"),
                            html.Label("Avg Temperature (°C):"),
                            dcc.Slider(id="sl-temp",min=25,max=38,step=0.5,value=32,
                                marks={25:"25°",30:"30°",35:"35°",38:"38°"},
                                className="mb-4"),
                            html.Label("Total Rainfall (mm):"),
                            dcc.Slider(id="sl-rain",min=0,max=600,step=10,value=100,
                                marks={0:"0",200:"200",400:"400",600:"600"},
                                className="mb-4"),
                            html.Label("Siem Reap Rain (mm):"),
                            dcc.Slider(id="sl-sr-rain",min=0,max=500,step=10,value=80,
                                marks={0:"0",200:"200",400:"400",500:"500"},
                                className="mb-4"),
                            html.Label("Siem Reap Temp (°C):"),
                            dcc.Slider(id="sl-sr-temp",min=25,max=38,step=0.5,value=32,
                                marks={25:"25°",30:"30°",35:"35°",38:"38°"},
                                className="mb-4"),
                            html.Div(id="pred-out"),
                        ])
                    ],className="shadow-sm border-0 mt-3")
                ],width=4),

                # Output
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("2025 Actual vs 2026 Predicted"),
                        dbc.CardBody(dcc.Graph(figure=fig_forecast,id="fc-chart"))
                    ],className="shadow-sm border-0 mt-3"),
                    dbc.Card([
                        dbc.CardHeader("Model Validation — 2019 Test Set"),
                        dbc.CardBody(dcc.Graph(figure=fig_val,id="val-chart"))
                    ],className="shadow-sm border-0 mt-3"),
                ],width=8),
            ]),
        ]),

    ]),

    html.Hr(className="mt-4"),
    html.P("🇰🇭 Cambodia Tourism + Weather Analytics | ITC Data Engineering | "
           "Kafka + PySpark + Gold Layer + RandomForest",
           className="text-muted text-center small pb-2"),

],fluid=True,className="px-4 py-3",
  style={"backgroundColor":"#f0f2f6","minHeight":"100vh"})


# ══════════════════════════════════════════════════════
# CALLBACKS (interactive only)
# ══════════════════════════════════════════════════════

@app.callback(
    Output("rain-chart","figure"),
    Output("temp-chart","figure"),
    Input("city-dd","value")
)
def update_weather(city):
    return make_weather(city)


@app.callback(
    Output("pred-out","children"),
    Input("sl-month","value"),
    Input("sl-temp","value"),
    Input("sl-rain","value"),
    Input("sl-sr-rain","value"),
    Input("sl-sr-temp","value"),
)
def predict(month, nat_temp, nat_rain, sr_rain, sr_temp):
    r25  = gold[(gold["year"]==2025)&(gold["month"]==month)]
    r24  = gold[(gold["year"]==2024)&(gold["month"]==month)]
    lag12 = int(r25["arrivals"].values[0]) if len(r25)>0 else 500000
    lag24 = int(r24["arrivals"].values[0]) if len(r24)>0 else 480000

    row = {
        "lag_12":               lag12,
        "lag_24":               lag24,
        "ratio_1224":           lag12/(lag24+1),
        "month":                month,
        "month_sin":            np.sin(2*np.pi*month/12),
        "month_cos":            np.cos(2*np.pi*month/12),
        "is_dry":               1 if month in [1,2,3,4,11,12] else 0,
        "is_peak":              1 if month in [11,12,1] else 0,
        "is_covid":             0,
        "nat_temp":             nat_temp,
        "nat_rain":             nat_rain,
        "total_rain_siem_reap": sr_rain,
        "avg_temp_siem_reap":   sr_temp,
    }
    df_row = pd.DataFrame([row])
    avail  = [f for f in features if f in df_row.columns]
    p      = int(model.predict(df_row[avail])[0])
    chg    = (p - lag12) / lag12 * 100
    color  = "success" if chg >= 0 else "danger"
    months = ["","Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    return dbc.Alert([
        html.Small(f"Predicted arrivals — 2026 {months[month]}",
                   className="d-block text-center mb-1"),
        html.H3(f"{p:,}", className="fw-bold text-center mb-1"),
        html.H5(f"{chg:+.1f}% vs 2025",className="text-center mb-1"),
        html.Hr(className="my-2"),
        html.Small(f"lag_12 (2025): {lag12:,}", className="d-block text-center"),
        html.Small(f"lag_24 (2024): {lag24:,}", className="d-block text-center"),
    ], color=color, className="mt-3")


if __name__ == "__main__":
    app.run(debug=False, port=8050)