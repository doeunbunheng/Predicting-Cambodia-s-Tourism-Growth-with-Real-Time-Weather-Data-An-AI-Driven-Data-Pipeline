## scripts/verify_data.py
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 55)
print(" VERIFY DATA — Check both CSVs")
print("=" * 55)

# ── LOAD BOTH ─────────────────────────────────────────
print("\n Loading weather data...")
weather = pd.read_csv("data/raw/weather_monthly.csv")
print(f"   Shape   : {weather.shape}")
print(f"   Columns : {list(weather.columns)}")
print(f"   Years   : {weather['year'].min()} → {weather['year'].max()}")
print(f"   Cities  : {weather['city'].unique().tolist()}")
print(f"   Nulls   : {weather.isnull().sum().sum()}")

print("\n Loading tourism data...")
tourism = pd.read_csv("data/raw/tourism_monthly.csv")
print(f"   Shape   : {tourism.shape}")
print(f"   Columns : {list(tourism.columns)}")
print(f"   Years   : {tourism['year'].min()} → {tourism['year'].max()}")
print(f"   Nulls   : {tourism.isnull().sum().sum()}")

# ── JOIN ──────────────────────────────────────────────
print("\n Joining on year + month...")
weather_pp = weather[weather["city"] == "Phnom Penh"].copy()

joined = pd.merge(
    tourism[["date","year","month","arrivals"]],
    weather_pp[["year","month","avg_temp_max",
                "total_rain_mm","avg_humidity"]],
    on=["year","month"],
    how="inner"
)
joined = joined.sort_values(["year","month"]).reset_index(drop=True)
joined.to_csv("data/raw/joined_preview.csv", index=False)
print(f"   Joined rows : {len(joined)}")
print(f"    Saved → data/raw/joined_preview.csv")

# ── CHECKS ────────────────────────────────────────────
print("\n Sanity checks...")
passed = 0

# Check 1
if len(joined) >= 60:
    print(f"    Check 1 — Rows: {len(joined)} (expect ≥ 60)")
    passed += 1
else:
    print(f"    Check 1 — Too few rows: {len(joined)}")

# Check 2
nulls = joined.isnull().sum().sum()
if nulls == 0:
    print(f"    Check 2 — No nulls")
    passed += 1
else:
    print(f"    Check 2 — Found {nulls} nulls")

# Check 3
min_arr = joined["arrivals"].min()
if min_arr < 20000:
    print(f"    Check 3 — COVID dip visible: min={min_arr:,}")
    passed += 1
else:
    print(f"    Check 3 — COVID dip missing: {min_arr:,}")

# Check 4
avg_temp = joined["avg_temp_max"].mean()
if 25 <= avg_temp <= 40:
    print(f"    Check 4 — Avg temp: {avg_temp:.1f}°C")
    passed += 1
else:
    print(f"    Check 4 — Temp odd: {avg_temp:.1f}°C")

# Check 5
wet  = joined[joined["month"].isin([6,7,8,9])]["total_rain_mm"].mean()
dry  = joined[joined["month"].isin([1,2,3])]["total_rain_mm"].mean()
if wet > dry:
    print(f"    Check 5 — Wet={wet:.0f}mm > Dry={dry:.0f}mm")
    passed += 1
else:
    print(f"    Check 5 — Seasonality wrong")

# ── PREVIEW ───────────────────────────────────────────
print(f"\n Sample joined rows:")
print(joined[["date","arrivals","avg_temp_max","total_rain_mm"]].head(5).to_string(index=False))

print(f"\n Annual arrivals:")
for year, total in joined.groupby("year")["arrivals"].sum().items():
    bar = "" * (total // 500000)
    print(f"   {year}: {total:>10,}  {bar}")

# ── RESULT ────────────────────────────────────────────
print(f"\n{'=' * 55}")
if passed == 5:
    print(f" ALL 5 CHECKS PASSED — DATA READY!")
    print(f"   weather_monthly.csv : {weather.shape[0]} rows")
    print(f"   tourism_monthly.csv : {tourism.shape[0]} rows")
    print(f"   joined_preview.csv  : {len(joined)} rows")
    print(f"\n Ready for Kafka producers!")
else:
    print(f"  {passed}/5 checks passed — fix issues above")
print(f"{'=' * 55}")