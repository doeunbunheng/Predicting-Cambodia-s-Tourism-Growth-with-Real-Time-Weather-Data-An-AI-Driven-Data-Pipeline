## scripts/weather_producer.py
import requests
import json
import time
import random
from kafka import KafkaProducer
from datetime import datetime
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 50)
print("  WEATHER PRODUCER STARTED")
print("   Topic  : weather-topic")
print("   Source : Open-Meteo API")
print("   Poll   : every 30 seconds")
print("=" * 50)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    api_version=(3, 7, 0),
    request_timeout_ms=30000,
    connections_max_idle_ms=60000
)
print("\n Connected to Kafka!")

CITIES = {
    "Phnom Penh":    {"lat": 11.56, "lon": 104.92},
    "Siem Reap":     {"lat": 13.36, "lon": 103.86},
    "Sihanoukville": {"lat": 10.63, "lon": 103.52},
}

DEFAULTS = {
    "Phnom Penh":    {"temperature": 34.0, "rain_mm": 0.0, "wind_kmh": 12.0, "humidity": 68},
    "Siem Reap":     {"temperature": 33.0, "rain_mm": 0.0, "wind_kmh": 10.0, "humidity": 65},
    "Sihanoukville": {"temperature": 32.0, "rain_mm": 0.0, "wind_kmh": 15.0, "humidity": 75},
}

round_num = 0

while True:
    round_num += 1
    print(f"\n Round {round_num} — {datetime.now().strftime('%H:%M:%S')}")

    for city, coords in CITIES.items():
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":  coords["lat"],
                    "longitude": coords["lon"],
                    "current": ["temperature_2m","precipitation",
                                "windspeed_10m","relative_humidity_2m"],
                    "timezone": "Asia/Phnom_Penh"
                },
                timeout=15
            )
            current = r.json()["current"]
            msg = {
                "city":        city,
                "timestamp":   datetime.now().isoformat(),
                "temperature": current["temperature_2m"],
                "rain_mm":     current["precipitation"],
                "wind_kmh":    current["windspeed_10m"],
                "humidity":    current["relative_humidity_2m"],
                "source":      "Open-Meteo Live API"
            }
            print(f"    {city}: {msg['temperature']}°C  "
                  f"rain={msg['rain_mm']}mm  "
                  f"humidity={msg['humidity']}%  [LIVE]")

        except Exception:
            base = DEFAULTS[city]
            msg = {
                "city":        city,
                "timestamp":   datetime.now().isoformat(),
                "temperature": round(base["temperature"] + random.uniform(-1.5, 1.5), 1),
                "rain_mm":     round(random.uniform(0, 2), 1),
                "wind_kmh":    round(base["wind_kmh"] + random.uniform(-2, 2), 1),
                "humidity":    base["humidity"] + random.randint(-5, 5),
                "source":      "Simulated (network timeout)"
            }
            print(f"     {city}: {msg['temperature']}°C  "
                  f"rain={msg['rain_mm']}mm  [SIMULATED]")

        producer.send("weather-topic", value=msg)

    producer.flush()
    print(f"    Waiting 30 seconds...")
    time.sleep(30)