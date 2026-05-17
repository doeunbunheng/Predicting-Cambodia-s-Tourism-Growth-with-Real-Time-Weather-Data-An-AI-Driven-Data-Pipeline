## scripts/spark_gold.py
## Joins Silver weather + tourism → Star Schema Gold layer
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 55)
print(" GOLD LAYER — Star Schema")
print("=" * 55)

# ── READ SILVER ───────────────────────────────────────
print("\n Reading Silver tables...")

weather_silver = pd.read_csv("data/silver/weather_silver.csv")
tourism_silver = pd.read_csv("data/silver/tourism_silver.csv")

print(f"    Weather Silver : {len(weather_silver)} rows")
print(f"    Tourism Silver : {len(tourism_silver)} rows")

# ── USE FULL HISTORICAL WEATHER ───────────────────────
# Silver weather only has live data → use historical CSV
weather_full = pd.read_csv("data/raw/weather_monthly.csv")
weather_full = weather_full.rename(columns={
    "avg_temp_max":  "avg_temp",
    "total_rain_mm": "total_rain",
    "avg_humidity":  "avg_humidity",
    "avg_wind":      "avg_wind"
})
print(f"    Weather Full   : {len(weather_full)} rows (historical)")

# ── PIVOT WEATHER → WIDE FORMAT ───────────────────────
print("\n Pivoting weather to wide format (3 cities as columns)...")

weather_pivot = weather_full.pivot_table(
    index=["year","month"],
    columns="city",
    values=["avg_temp","total_rain","avg_humidity"],
    aggfunc="mean"
).reset_index()

# Flatten column names
weather_pivot.columns = [
    f"{col[0]}_{col[1].replace(' ','_').lower()}"
    if col[1] else col[0]
    for col in weather_pivot.columns
]
print(f"    Pivoted shape  : {weather_pivot.shape}")

# ── JOIN TOURISM + WEATHER ────────────────────────────
print("\n Joining tourism + weather on year + month...")

tourism_silver["date"] = pd.to_datetime(tourism_silver["date"])

gold = pd.merge(
    tourism_silver[["date","year","month","arrivals","source"]],
    weather_pivot,
    on=["year","month"],
    how="inner"
)
gold = gold.sort_values(["year","month"]).reset_index(drop=True)
print(f"    Gold joined    : {len(gold)} rows")

# ── ADD DIMENSION COLUMNS ─────────────────────────────
print("\n Adding dimension columns...")

# Season (Cambodia: Dry Nov-Apr, Wet May-Oct)
gold["season"] = gold["month"].apply(
    lambda m: "Dry" if m in [11,12,1,2,3,4] else "Wet"
)

# Quarter
gold["quarter"] = gold["month"].apply(
    lambda m: f"Q{(m-1)//3+1}"
)

# COVID flag
gold["is_covid"] = gold["year"].apply(
    lambda y: 1 if y in [2020,2021] else 0
)

# Previous month arrivals (lag feature for ML)
gold["prev_month_arrivals"] = gold["arrivals"].shift(1).fillna(0).astype(int)

print(f"    Columns added  : season, quarter, is_covid, prev_month_arrivals")

# ── CREATE DIMENSION TABLES ───────────────────────────
print("\n Creating dimension tables...")

# dim_time
dim_time = gold[["year","month","quarter","season"]].drop_duplicates()
dim_time["time_id"] = range(1, len(dim_time)+1)
dim_time = dim_time[["time_id","year","month","quarter","season"]]

# dim_purpose (from MOT report)
dim_purpose = pd.DataFrame([
    (1, "Holiday",  "Leisure tourism"),
    (2, "Business", "Business travel"),
    (3, "Others",   "Other purposes"),
], columns=["purpose_id","purpose_name","description"])

# dim_transport (from MOT report)
dim_transport = pd.DataFrame([
    (1, "Air",       "Phnom Penh (PNH)"),
    (2, "Air",       "Siem Reap (SAI)"),
    (3, "Air",       "Sihanoukville (KOS)"),
    (4, "Land",      "Land border crossing"),
    (5, "Waterways", "River/sea entry"),
], columns=["transport_id","mode","entry_point"])

print(f"    dim_time      : {len(dim_time)} rows")
print(f"    dim_purpose   : {len(dim_purpose)} rows")
print(f"    dim_transport : {len(dim_transport)} rows")

# ── SAVE GOLD ─────────────────────────────────────────
os.makedirs("data/gold", exist_ok=True)

gold.to_csv("data/gold/fact_tourism_monthly.csv",  index=False)
dim_time.to_csv("data/gold/dim_time.csv",          index=False)
dim_purpose.to_csv("data/gold/dim_purpose.csv",    index=False)
dim_transport.to_csv("data/gold/dim_transport.csv",index=False)

print(f"\n{'=' * 55}")
print(f" GOLD LAYER DONE — Star Schema created!")
print(f"{'=' * 55}")
print(f"   fact_tourism_monthly.csv : {len(gold)} rows")
print(f"   dim_time.csv             : {len(dim_time)} rows")
print(f"   dim_purpose.csv          : {len(dim_purpose)} rows")
print(f"   dim_transport.csv        : {len(dim_transport)} rows")
print(f"\n Fact table columns:")
print(f"   {list(gold.columns)}")
print(f"\n Sample Gold rows:")
print(gold[["date","year","month","arrivals",
            "season","avg_temp_phnom_penh",
            "total_rain_siem_reap"]].head(5).to_string(index=False))
