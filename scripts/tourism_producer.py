## scripts/tourism_producer.py
import pandas as pd
import json
import time
from kafka import KafkaProducer
from datetime import datetime
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 50)
print("  TOURISM PRODUCER STARTED")
print("   Topic  : tourism-topic")
print("   Method : Event Replay 1 record/sec")
print("   Source : World Bank + MOT PDF")
print("=" * 50)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    api_version=(3, 7, 0),
    request_timeout_ms=30000,
    connections_max_idle_ms=60000
)
print("\n Connected to Kafka!")

df = pd.read_csv("data/raw/tourism_monthly.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["year","month"]).reset_index(drop=True)

print(f" Loaded {len(df)} monthly records")
print(f"   Range: {df['date'].min().strftime('%Y-%m')} → {df['date'].max().strftime('%Y-%m')}")
print(f"\n Streaming to Kafka...\n")

for idx, row in df.iterrows():
    msg = {
        "date":      row["date"].strftime("%Y-%m-%d"),
        "year":      int(row["year"]),
        "month":     int(row["month"]),
        "arrivals":  int(row["arrivals"]),
        "source":    str(row["source"]),
        "timestamp": datetime.now().isoformat()
    }
    producer.send("tourism-topic", value=msg)
    print(f"    [{idx+1:03d}/{len(df)}] "
          f"{msg['date']} → {msg['arrivals']:>8,} arrivals")
    time.sleep(1)

producer.flush()
producer.close()
print(f"\n ALL {len(df)} records sent to tourism-topic!")