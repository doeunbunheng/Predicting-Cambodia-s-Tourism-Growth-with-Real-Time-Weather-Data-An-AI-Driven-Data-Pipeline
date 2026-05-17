## scripts/spark_silver.py
## Reads Bronze Parquet → Cleans → Saves Silver
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 55)
print("  SILVER LAYER — Clean + Transform")
print("=" * 55)

# ── READ BRONZE ───────────────────────────────────────
print("\n Reading Bronze tables...")

try:
    weather_bronze = pd.read_parquet("data/bronze/weather")
    print(f"    Weather Bronze : {len(weather_bronze)} rows")
except:
    print("     Bronze weather empty — using raw CSV instead")
    weather_bronze = pd.read_csv("data/raw/weather_monthly.csv")

try:
    tourism_bronze = pd.read_parquet("data/bronze/tourism")
    print(f"    Tourism Bronze : {len(tourism_bronze)} rows")
except:
    print("     Bronze tourism empty — using raw CSV instead")
    tourism_bronze = pd.read_csv("data/raw/tourism_monthly.csv")

# ── CLEAN WEATHER ─────────────────────────────────────
print("\n Cleaning weather data...")

weather_silver = weather_bronze.copy()

# Drop nulls
before = len(weather_silver)
weather_silver = weather_silver.dropna()
print(f"   Dropped {before - len(weather_silver)} null rows")

# Fix types
weather_silver["temperature"] = pd.to_numeric(
    weather_silver["temperature"], errors="coerce")
weather_silver["rain_mm"]     = pd.to_numeric(
    weather_silver["rain_mm"], errors="coerce")
weather_silver["humidity"]    = pd.to_numeric(
    weather_silver["humidity"], errors="coerce")

# Add date parts
if "timestamp" in weather_silver.columns:
    weather_silver["datetime"] = pd.to_datetime(
        weather_silver["timestamp"], errors="coerce")
    weather_silver["year"]  = weather_silver["datetime"].dt.year
    weather_silver["month"] = weather_silver["datetime"].dt.month

# Remove duplicates
weather_silver = weather_silver.drop_duplicates()
print(f"    Weather Silver : {len(weather_silver)} rows")

# ── CLEAN TOURISM ─────────────────────────────────────
print("\n🧹 Cleaning tourism data...")

tourism_silver = tourism_bronze.copy()

# Drop nulls
before = len(tourism_silver)
tourism_silver = tourism_silver.dropna(subset=["arrivals"])
print(f"   Dropped {before - len(tourism_silver)} null rows")

# Fix types
tourism_silver["arrivals"] = pd.to_numeric(
    tourism_silver["arrivals"], errors="coerce").astype(int)
tourism_silver["year"]  = pd.to_numeric(
    tourism_silver["year"], errors="coerce").astype(int)
tourism_silver["month"] = pd.to_numeric(
    tourism_silver["month"], errors="coerce").astype(int)

# Parse date
tourism_silver["date"] = pd.to_datetime(
    tourism_silver["date"], errors="coerce")

# Remove duplicates — keep latest per year+month
tourism_silver = tourism_silver.drop_duplicates(
    subset=["year","month"], keep="last")
tourism_silver = tourism_silver.sort_values(
    ["year","month"]).reset_index(drop=True)

print(f"    Tourism Silver : {len(tourism_silver)} rows")

# ── AGGREGATE WEATHER MONTHLY ─────────────────────────
print("\n Aggregating weather to monthly...")

if "city" in weather_silver.columns and "year" in weather_silver.columns:
    weather_monthly = weather_silver.groupby(
        ["city","year","month"]
    ).agg(
        avg_temp     = ("temperature", "mean"),
        total_rain   = ("rain_mm",     "sum"),
        avg_humidity = ("humidity",    "mean")
    ).reset_index()
    print(f"    Weather monthly: {len(weather_monthly)} rows")
else:
    weather_monthly = pd.read_csv("data/raw/weather_monthly.csv")
    weather_monthly = weather_monthly.rename(columns={
        "avg_temp_max":  "avg_temp",
        "total_rain_mm": "total_rain",
    })
    print(f"    Using CSV weather: {len(weather_monthly)} rows")

# ── SAVE SILVER ───────────────────────────────────────
os.makedirs("data/silver", exist_ok=True)

weather_monthly.to_csv("data/silver/weather_silver.csv",  index=False)
tourism_silver.to_csv("data/silver/tourism_silver.csv",   index=False)

print(f"\n{'=' * 55}")
print(f" SILVER LAYER DONE!")
print(f"   data/silver/weather_silver.csv : {len(weather_monthly)} rows")
print(f"   data/silver/tourism_silver.csv : {len(tourism_silver)} rows")
print(f"\n Tourism Silver sample:")
print(tourism_silver[["date","year","month","arrivals","source"]].head(5).to_string(index=False))
print(f"\n Weather Silver sample:")
print(weather_monthly.head(5).to_string(index=False))
print(f"\n Ready for Gold layer!")