## scripts/fetch_weather.py
import requests
import pandas as pd

CITIES = {
    "Phnom Penh":    {"lat": 11.56, "lon": 104.92},
    "Siem Reap":     {"lat": 13.36, "lon": 103.86},
    "Sihanoukville": {"lat": 10.63, "lon": 103.52},
}

def fetch_city_weather(city, lat, lon):
    print(f" Fetching weather for {city}...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": "2012-01-01",
        "end_date":   "2025-12-31",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "relative_humidity_2m_mean"
        ],
        "timezone": "Asia/Phnom_Penh"
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print(f"   Error: {r.status_code} - {r.text}")
        return None

    daily = r.json()["daily"]
    df = pd.DataFrame(daily)
    df.rename(columns={"time": "date"}, inplace=True)
    df["city"] = city
    print(f"   Got {len(df)} daily rows")
    return df

# ── Fetch all 3 cities ──────────────────────────────
all_dfs = []
for city, coords in CITIES.items():
    df = fetch_city_weather(city, coords["lat"], coords["lon"])
    if df is not None:
        all_dfs.append(df)

weather_raw = pd.concat(all_dfs, ignore_index=True)

# ── Aggregate daily → monthly ───────────────────────
weather_raw["date"]  = pd.to_datetime(weather_raw["date"])
weather_raw["year"]  = weather_raw["date"].dt.year
weather_raw["month"] = weather_raw["date"].dt.month

weather_monthly = weather_raw.groupby(
    ["city", "year", "month"]
).agg(
    avg_temp_max  = ("temperature_2m_max",          "mean"),
    avg_temp_min  = ("temperature_2m_min",          "mean"),
    total_rain_mm = ("precipitation_sum",           "sum"),
    avg_wind      = ("windspeed_10m_max",           "mean"),
    avg_humidity  = ("relative_humidity_2m_mean",   "mean")
).reset_index()

# ── Save to CSV ─────────────────────────────────────
weather_monthly.to_csv("data/raw/weather_monthly.csv", index=False)

print(f"   Shape     : {weather_monthly.shape}")
print(f"   Years     : {weather_monthly['year'].min()} → {weather_monthly['year'].max()}")
print(f"   Cities    : {weather_monthly['city'].unique().tolist()}")
print(f"   Saved to  : data/raw/weather_monthly.csv")
print("\nPreview:")
print(weather_monthly.head(6).to_string(index=False))