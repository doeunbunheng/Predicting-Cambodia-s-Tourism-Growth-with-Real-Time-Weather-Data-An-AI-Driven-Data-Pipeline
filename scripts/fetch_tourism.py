## scripts/fetch_tourism.py
## Source 1: World Bank CSV (downloaded) → 2012-2020
## Source 2: MOT Official PDF Report → 2021-2025
import pandas as pd
import os

# Always run from project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working dir: {os.getcwd()}")

print("=" * 55)
print(" TOURISM DATA — World Bank CSV + MOT Report")
print("=" * 55)

# ── READ WORLD BANK CSV ───────────────────────────────
print("\n Reading World Bank CSV file...")

wb_csv = "data/raw/API_ST.INT.ARVL_DS2_en_csv_v2_126984.csv"

# Read CSV — skip first 4 header rows
df_raw = pd.read_csv(wb_csv, skiprows=4)

# Filter only Cambodia (KHM)
cambodia = df_raw[df_raw["Country Code"] == "KHM"].copy()
print(f" Found Cambodia row!")

# Extract years 2012-2020
years = [str(y) for y in range(2012, 2021)]
annual_rows = []
for year in years:
    value = cambodia[year].values[0]
    if pd.notna(value) and value > 0:
        annual_rows.append({
            "year":     int(year),
            "arrivals": int(value),
            "source":   "World Bank CSV (ST.INT.ARVL)"
        })

annual_df = pd.DataFrame(annual_rows)
print(f" Extracted {len(annual_df)} annual records (2012-2020)")
print(annual_df.to_string(index=False))

# Save annual
annual_df.to_csv("data/raw/tourism_annual_worldbank.csv", index=False)
print(" Saved → data/raw/tourism_annual_worldbank.csv")

# ── EXPAND ANNUAL → MONTHLY ───────────────────────────
print("\n Expanding annual → monthly...")

# Real Cambodia seasonal weights from MOT 2024 report
monthly_share = {
     1: 0.0980,  2: 0.0905,  3: 0.0958,
     4: 0.0814,  5: 0.0716,  6: 0.0681,
     7: 0.0760,  8: 0.0787,  9: 0.0687,
    10: 0.0729, 11: 0.0912, 12: 0.1071,
}

wb_rows = []
for _, row in annual_df.iterrows():
    for month, share in monthly_share.items():
        wb_rows.append({
            "date":     f"{int(row['year'])}-{month:02d}-01",
            "year":     int(row["year"]),
            "month":    month,
            "arrivals": int(row["arrivals"] * share),
            "source":   "World Bank CSV (monthly expanded)"
        })

wb_monthly = pd.DataFrame(wb_rows)
wb_monthly["date"] = pd.to_datetime(wb_monthly["date"])

# ── ADD MOT REAL MONTHLY DATA 2021-2024 ──────────────
print(" Adding MOT official monthly data 2021-2024...")

mot_rows = pd.DataFrame([
    # 2021 — COVID recovery (from MOT PDF)
    ("2021-01-01",2021, 1,  20567), ("2021-02-01",2021, 2,  20580),
    ("2021-03-01",2021, 3,  29754), ("2021-04-01",2021, 4,  11938),
    ("2021-05-01",2021, 5,   8757), ("2021-06-01",2021, 6,  10964),
    ("2021-07-01",2021, 7,   9984), ("2021-08-01",2021, 8,   9163),
    ("2021-09-01",2021, 9,   9967), ("2021-10-01",2021,10,  12759),
    ("2021-11-01",2021,11,  18933), ("2021-12-01",2021,12,  33129),
    # 2022 — strong recovery
    ("2022-01-01",2022, 1,  44910), ("2022-02-01",2022, 2,  50411),
    ("2022-03-01",2022, 3,  64225), ("2022-04-01",2022, 4,  81939),
    ("2022-05-01",2022, 5, 101979), ("2022-06-01",2022, 6, 163298),
    ("2022-07-01",2022, 7, 236697), ("2022-08-01",2022, 8, 254813),
    ("2022-09-01",2022, 9, 267500), ("2022-10-01",2022,10, 310182),
    ("2022-11-01",2022,11, 338101), ("2022-12-01",2022,12, 362571),
    # 2023 — full recovery
    ("2023-01-01",2023, 1, 402943), ("2023-02-01",2023, 2, 434503),
    ("2023-03-01",2023, 3, 454093), ("2023-04-01",2023, 4, 430129),
    ("2023-05-01",2023, 5, 442114), ("2023-06-01",2023, 6, 416150),
    ("2023-07-01",2023, 7, 457412), ("2023-08-01",2023, 8, 464637),
    ("2023-09-01",2023, 9, 424901), ("2023-10-01",2023,10, 480330),
    ("2023-11-01",2023,11, 510231), ("2023-12-01",2023,12, 535788),
    # 2024 — growth
    ("2024-01-01",2024, 1, 540023), ("2024-02-01",2024, 2, 448551),
    ("2024-03-01",2024, 3, 594103), ("2024-04-01",2024, 4, 533932),
    ("2024-05-01",2024, 5, 524390), ("2024-06-01",2024, 6, 525498),
    ("2024-07-01",2024, 7, 575733), ("2024-08-01",2024, 8, 548828),
    ("2024-09-01",2024, 9, 509474), ("2024-10-01",2024,10, 574206),
    ("2024-11-01",2024,11, 626319), ("2024-12-01",2024,12, 699068),
     # ── 2025 ── latest MOT report
    ("2025-01-01",2025, 1, 611894), ("2025-02-01",2025, 2, 652091),
    ("2025-03-01",2025, 3, 574223), ("2025-04-01",2025, 4, 565243),
    ("2025-05-01",2025, 5, 547373), ("2025-06-01",2025, 6, 413560),
    ("2025-07-01",2025, 7, 347492), ("2025-08-01",2025, 8, 338351),
    ("2025-09-01",2025, 9, 327239), ("2025-10-01",2025,10, 574206),
    ("2025-11-01",2025,11, 422766), ("2025-12-01",2025,12, 396911),
], columns=["date","year","month","arrivals"])
mot_rows["date"]   = pd.to_datetime(mot_rows["date"])
mot_rows["source"] = "MOT Official PDF Report"

# ── COMBINE BOTH ──────────────────────────────────────
tourism = pd.concat([wb_monthly, mot_rows], ignore_index=True)
tourism = tourism.drop_duplicates(subset=["year","month"], keep="last")
tourism = tourism.sort_values(["year","month"]).reset_index(drop=True)

# ── SAVE FINAL CSV ────────────────────────────────────
tourism.to_csv("data/raw/tourism_monthly.csv", index=False)

# ── FINAL REPORT ──────────────────────────────────────
print(f"\n{'=' * 55}")
print(f"   Source 1 : World Bank CSV → 2012-2020 annual")
print(f"   Source 2 : MOT Official PDF → 2021-2025 monthly")
print(f"   Rows     : {len(tourism)}")
print(f"   Range    : {tourism['date'].min().strftime('%Y-%m')} → {tourism['date'].max().strftime('%Y-%m')}")
print(f"   Max      : {tourism['arrivals'].max():,}")
print(f"   Min      : {tourism['arrivals'].min():,}")
print(f"\n Records by source:")
print(tourism.groupby("source")["arrivals"].count().rename("months").to_string())
print(f"\n Last 6 rows:")
print(tourism[["date","year","month","arrivals","source"]].tail(6).to_string(index=False))