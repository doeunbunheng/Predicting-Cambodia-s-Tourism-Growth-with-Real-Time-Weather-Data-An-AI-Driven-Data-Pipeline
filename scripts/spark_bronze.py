## scripts/spark_bronze.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import *
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 55)
print(" PYSPARK BRONZE LAYER")
print("   Reading : weather-topic + tourism-topic")
print("   Writing : data/bronze/weather + data/bronze/tourism")
print("=" * 55)

# ── Spark Session (no Delta — use Parquet) ────────────
spark = SparkSession.builder \
    .appName("CambodiaBronze") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
    .config("spark.sql.streaming.checkpointFileManagerClass",
            "org.apache.spark.sql.execution.streaming.FileSystemBasedCheckpointFileManager") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("\n Spark Session started!")

# ── Schemas ───────────────────────────────────────────
weather_schema = StructType([
    StructField("city",        StringType()),
    StructField("timestamp",   StringType()),
    StructField("temperature", DoubleType()),
    StructField("rain_mm",     DoubleType()),
    StructField("wind_kmh",    DoubleType()),
    StructField("humidity",    IntegerType()),
    StructField("source",      StringType()),
])

tourism_schema = StructType([
    StructField("date",      StringType()),
    StructField("year",      IntegerType()),
    StructField("month",     IntegerType()),
    StructField("arrivals",  IntegerType()),
    StructField("source",    StringType()),
    StructField("timestamp", StringType()),
])

# ── Read from Kafka ───────────────────────────────────
print(" Connecting to Kafka topics...")

weather_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "weather-topic") \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load() \
    .select(from_json(
        col("value").cast("string"),
        weather_schema
    ).alias("d")).select("d.*")

tourism_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "tourism-topic") \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load() \
    .select(from_json(
        col("value").cast("string"),
        tourism_schema
    ).alias("d")).select("d.*")

print(" Connected to both topics!")

# ── Write to Bronze as Parquet ────────────────────────
print(" Writing streams to Bronze (Parquet)...")

weather_query = weather_stream.writeStream \
    .format("parquet") \
    .option("checkpointLocation", "data/bronze/weather_ckpt") \
    .option("path", "data/bronze/weather") \
    .outputMode("append") \
    .trigger(processingTime="10 seconds") \
    .start()

tourism_query = tourism_stream.writeStream \
    .format("parquet") \
    .option("checkpointLocation", "data/bronze/tourism_ckpt") \
    .option("path", "data/bronze/tourism") \
    .outputMode("append") \
    .trigger(processingTime="10 seconds") \
    .start()

print("\n BRONZE LAYER RUNNING!")
print("   data/bronze/weather  ← weather records")
print("   data/bronze/tourism  ← tourism records")
print("\n Let it run 2 minutes → then press Ctrl+C")

spark.streams.awaitAnyTermination()