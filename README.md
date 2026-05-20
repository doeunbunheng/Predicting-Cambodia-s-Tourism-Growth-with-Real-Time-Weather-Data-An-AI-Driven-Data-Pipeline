# 🇰🇭 Predicting Cambodia's Tourism Growth with Real-Time Weather Data

**An AI-Driven End-to-End Data Engineering Pipeline**

> Built by ITC Data Engineering students — Group Project 2026
> Department of Applied Mathematics and Statistics, Institute of Technology of Cambodia

---

##  What Is This Project?

Tourism contributes about **12% of Cambodia's GDP**. We asked a simple question:

> *Can real-time weather data help us predict how many tourists will arrive next month?*

To answer this, we built a **complete data engineering pipeline** from scratch — collecting live weather data every 30 minutes, streaming it through Apache Kafka, processing it with PySpark, training a Machine Learning model, and displaying everything in an interactive dashboard.

The answer? **Weather has less than 2% impact on arrivals.** The same month last year explains 80% of the prediction. But the pipeline we built to discover this is the real achievement.

---

##  Team Members

| Name | Student ID | Role |
|------|-----------|------|
| Chheng Sothean     | e20220686 | Team Leader — Architecture |
| Chiv Mengchou      | e20221028 | Data Engineer — APIs |
| Chho Sengmeng      | e20220296 | Streaming Engineer — Kafka |
| Choub Botumraksa   | e20221709 | PySpark Engineer — ETL |
| Din Reaksa         | e20221070 | DW Engineer — Gold Layer |
| Doeun Bunheng      | e20221528 | ML Engineer + App Dev |
| Mon Sreylin        | e20221701 | Demo Lead |

---

##  System Architecture

```
Open-Meteo API          World Bank CSV + MOT PDF
(live every 30 min)     (2012 - 2025 monthly)
        │                        │
        ▼                        ▼
  weather_producer.py    tourism_producer.py
  (real-time stream)     (Event Replay)
        │                        │
        └─────────┬──────────────┘
                  ▼
          Apache Kafka 4.2.0
       ┌──────────────────────┐
       │  weather-topic       │
       │  tourism-topic       │
       └──────────┬───────────┘
                  ▼
         PySpark readStream
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
    Bronze Layer       Bronze Layer
   (raw Parquet)        (raw Parquet)
        │
        ▼
    Silver Layer
   (cleaned CSV)
        │
        ▼
    Gold Layer — Star Schema
   fact_tourism_monthly.csv
   dim_time / dim_transport / dim_purpose
        │
   ┌────┴────┐
   ▼         ▼
ML Model   Dashboard
R²=0.89   Plotly Dash
```

---

##  Data Sources

### 1. Weather Data — Open-Meteo API
- **URL:** https://api.open-meteo.com
- **Free, no API key required**
- 3 cities: Phnom Penh, Siem Reap, Sihanoukville
- Columns: temperature, rainfall, humidity, wind speed
- Coverage: 2012–2025 (504 rows = 3 cities × 168 months)

### 2. Tourism Data — World Bank + MOT PDF
- **World Bank CSV:** Annual international arrivals 2012–2018
- **MOT Official PDF:** Exact monthly arrivals 2019–2025
- Source: Cambodia Ministry of Tourism Statistics Report
- Coverage: 168 months (Jan 2012 – Dec 2025)

---

##  Tech Stack

| Component | Technology |
|-----------|------------|
| Message Broker | Apache Kafka 4.2.0 (KRaft mode) |
| Stream Processing | PySpark 3.3.0 Structured Streaming |
| Data Storage | Parquet (Bronze) → CSV (Silver, Gold) |
| Machine Learning | RandomForest, GradientBoosting (scikit-learn) |
| Dashboard | Plotly Dash + Bootstrap |
| Weather API | Open-Meteo (free, no key) |
| Language | Python 3.10 |

---

## 📁 Project Structure

```
Cambodai_Tourism_Weather_Analysis/
│
├── data/
│   ├── raw/
│   │   ├── weather_monthly.csv          ← 504 rows, 3 cities, 2012-2025
│   │   ├── tourism_monthly.csv          ← 168 rows, Jan 2012 – Dec 2025
│   │   ├── tourism_mot_raw.pdf          ← MOT official report
│   │   ├── API_ST.INT.ARVL_DS2_en_csv_v2_126984.csv  ← World Bank
│   │   └── joined_preview.csv
│   │
│   ├── bronze/
│   │   ├── weather/                     ← raw Parquet from Kafka
│   │   └── tourism/                     ← raw Parquet from Kafka
│   │
│   ├── silver/
│   │   ├── weather_silver.csv           ← cleaned weather
│   │   └── tourism_silver.csv           ← cleaned tourism
│   │
│   └── gold/
│       ├── fact_tourism_monthly.csv     ← main fact table (168 rows)
│       ├── dim_time.csv
│       ├── dim_transport.csv
│       ├── dim_purpose.csv
│       ├── model_best.pkl               ← trained RandomForest
│       ├── model_features.pkl
│       ├── predictions.csv              ← 2019 test results
│       ├── predictions_2026.csv         ← 2026 forecast
│       └── feature_importance.csv
│
└── scripts/
    ├── fetch_weather.py                 ← Step 1: collect weather
    ├── fetch_tourism.py                 ← Step 1: collect tourism
    ├── verify_data.py                   ← Step 2: quality checks
    ├── tourism_producer.py              ← Step 3: stream to Kafka
    ├── weather_producer.py              ← Step 3: stream to Kafka
    ├── spark_bronze.py                  ← Step 4: PySpark Bronze
    ├── spark_silver.py                  ← Step 5: PySpark Silver
    ├── spark_gold.py                    ← Step 6: Gold + Star Schema
    ├── ml_model.py                      ← Step 7: Train ML model
    └── dashboard.py                     ← Step 8: Plotly Dash app
```

---

##  How to Run

### Prerequisites

- Python 3.10
- Java 17 (Eclipse Adoptium JDK)
- Apache Kafka 4.2.0
- Git Bash (Windows)

### Step 1 — Clone and set up environment

```bash
git clone https://github.com/doeunbunheng/Predicting-Cambodia-s-Tourism-Growth-with-Real-Time-Weather-Data-An-AI-Driven-Data-Pipeline.git
cd Cambodai_Tourism_Weather_Analysis

python -m venv venv
source venv/Scripts/activate        # Windows
source venv/bin/activate            # Mac/Linux

pip install -r requirements.txt
```

### Step 2 — Collect data

```bash
python scripts/fetch_weather.py     # → data/raw/weather_monthly.csv
python scripts/fetch_tourism.py     # → data/raw/tourism_monthly.csv
python scripts/verify_data.py       # → all 5 checks must pass
```

### Step 3 — Start Kafka

```bash
# Set Java (Windows)
export JAVA_HOME="/c/Program Files/Eclipse Adoptium/jdk-17.0.18.8-hotspot"
export PATH="$JAVA_HOME/bin:$PATH"

# Start Kafka server (keep this terminal open)
cd /c/kafka_2.13-4.2.0
./bin/kafka-server-start.sh config/server.properties
```

Wait for `INFO Kafka Server started`

### Step 4 — Create Kafka topics

```bash
# New terminal
export JAVA_HOME="/c/Program Files/Eclipse Adoptium/jdk-17.0.18.8-hotspot"
export PATH="$JAVA_HOME/bin:$PATH"
cd /c/kafka_2.13-4.2.0

./bin/kafka-topics.sh --create --topic weather-topic \
  --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

./bin/kafka-topics.sh --create --topic tourism-topic \
  --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

./bin/kafka-topics.sh --list --bootstrap-server localhost:9092
```

### Step 5 — Run producers

```bash
# Terminal 3 — tourism (Event Replay)
python scripts/tourism_producer.py

# Terminal 4 — weather (live API)
python scripts/weather_producer.py
```

### Step 6 — Run PySpark pipeline

```bash
python scripts/spark_bronze.py   # run 2 minutes then Ctrl+C
python scripts/spark_silver.py
python scripts/spark_gold.py
```

### Step 7 — Train ML model

```bash
python scripts/ml_model.py
```

Expected result:
```
 Best model : RandomForest
   R²         : 0.8906
   MAE        : 15,755 arrivals
   Avg error  : 2.9%
```

### Step 8 — Launch dashboard

```bash
python scripts/dashboard.py
```

Open Chrome → `http://localhost:8050`

---

##  Dashboard Features

The Plotly Dash dashboard has **4 tabs**:

| Tab | What it shows |
|-----|---------------|
|  Live Weather | Real-time temperature, humidity, rainfall — 3 cities — updates every 30 seconds |
|  Overview | Annual arrivals 2012–2025, COVID impact, monthly trend, season pie |
|  Weather Analysis | Scatter charts, feature importance, weather correlation |
|  ML Prediction | Interactive sliders → predict 2026 arrivals, model validation |

---

##  ML Model Results

| Model | R² | MAE | Avg Error |
|-------|----|-----|-----------|
| **RandomForest** | **0.8906** | **15,755** | **2.9%** |
| GradientBoosting | 0.87 | 18,200 | 3.3% |

**Train:** 2014–2018 (pre-COVID clean data)
**Test:** 2019 (held-out year)

### Key Finding

```
Feature Importance:
  lag_12 (same month last year)   → 80.8%  ← dominant predictor
  lag_24 (same month 2 years ago) → 17.4%
  is_covid                        →  5.6%
  weather features combined       →  <2%
```

**Insight:** Cambodia tourists book flights 2–3 months in advance.
By arrival time, they cannot change plans regardless of weather.
Seasonal momentum drives arrivals — not current weather.

### 2026 Predictions (Jan–Jun)

| Month | 2025 Actual | 2026 Predicted | Change |
|-------|-------------|----------------|--------|
| Jan | 611,894 | 619,691 | +1.3% |
| Feb | 652,091 | 535,398 | -17.8% |
| Mar | 574,223 | 616,586 | +7.4% |
| Apr | 565,243 | 554,969 | -1.8% |
| May | 547,373 | 518,637 | -5.2% |
| Jun | 413,560 | 449,079 | +8.6% |

---

##  Data Quality

`verify_data.py` runs **5 checks** before any data enters Kafka:

| Check | Description | Result |
|-------|-------------|--------|
|  Row count | At least 60 rows | Pass |
|  No nulls | Zero missing values | Pass |
|  COVID dip | 2020 arrivals < 20% of 2019 | Pass |
|  Temperature range | 25°C – 40°C Cambodia range | Pass |
|  Seasonality | Wet season rain > dry season | Pass |

---

##  Troubleshooting

**Kafka won't start — AccessDeniedException:**
```bash
# Run CMD as Administrator, then:
rmdir /s /q C:\tmp

# Then in Git Bash:
./bin/kafka-storage.sh random-uuid
./bin/kafka-storage.sh format -t YOUR_UUID -c config/server.properties
./bin/kafka-server-start.sh config/server.properties
```

**Producer error — NoBrokersAvailable:**
```bash
# Kafka is not running. Start it first (Step 3 above)
netstat -an | grep 9092   # check if Kafka is listening
```

**Dashboard shows no charts:**
```bash
# Dashboard works without Kafka — just run:
python scripts/dashboard.py
# It reads directly from data/gold/ CSV files
```

---

##  License

This project is licensed under the MIT License.

---

## Acknowledgements

- [Open-Meteo](https://open-meteo.com) — free weather API
- [World Bank Open Data](https://data.worldbank.org) — tourism statistics
- [Ministry of Tourism Cambodia](https://www.tourism.gov.kh) — official monthly reports
- Institute of Technology of Cambodia — Data Engineering course

---

*Built with ❤️ by ITC Data Engineering Group — May 2026*
