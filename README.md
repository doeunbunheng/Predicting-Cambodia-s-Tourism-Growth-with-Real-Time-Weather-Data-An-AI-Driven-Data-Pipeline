# Predicting Cambodia's Tourism Growth with Real-Time Weather Data: An AI-Driven Data Pipeline

## Project Overview

This project aims to predict tourism arrivals in Cambodia by integrating real-time weather data and historical tourism data. The solution utilizes a full-stack data pipeline powered by **PySpark**, **Kafka**, **Delta Lake**, and **Machine Learning** to forecast future tourism trends. Weather forecasts are sourced from **Open-Meteo**, while tourism data is obtained from **World Bank** and official **Cambodian government reports**.

## Objective

- Develop an end-to-end data pipeline to collect, process, and analyze real-time weather data and tourism arrivals data.
- Utilize machine learning models, such as **Random Forest** and **Linear Regression**, to predict future tourism arrivals.
- Present insights via a **Power BI dashboard** and an **interactive Streamlit app**.

## Tech Stack

- **Kafka**: Real-time data streaming platform.
- **PySpark**: Large-scale data processing and analysis.
- **Delta Lake**: Reliable and scalable data storage in the data lake.
- **Machine Learning**: **Random Forest** and **Linear Regression** for forecasting.
- **Power BI**: For data visualization and insights.
- **Streamlit**: For creating interactive dashboards.
- **Python**: For data extraction, transformation, and pipeline integration.

## Features

### Data Collection:
- Collect real-time weather data for three Cambodian cities (Phnom Penh, Siem Reap, and Sihanoukville) using the **Open-Meteo API**.
- Collect monthly tourism arrivals data from **World Bank** (2012-2018) and **Ministry of Tourism (MOT)** official reports (2019-2025).

### Data Processing:
- **Bronze Layer**: Raw data ingestion and storage.
- **Silver Layer**: Cleaned and transformed data (e.g., removing duplicates, fixing formats).
- **Gold Layer**: Aggregated data for analysis and machine learning.

### Real-Time Forecasting:
- Use **PySpark** to read and process data from Kafka in real time for both weather and tourism data.
- Train machine learning models to predict future tourism arrivals based on historical data and weather forecasts.

### Visualization:
- **Power BI** dashboard to visualize tourism trends, weather correlations, and predictions.
- **Streamlit** app to display real-time weather data, predictive analytics, and key performance indicators (KPIs).

## Getting Started

### Prerequisites

1. **Kafka**: Ensure you have Kafka set up on your machine. If not, follow the [Kafka installation guide](https://kafka.apache.org/quickstart).
2. **Python Environment**: Set up a Python virtual environment and install the required dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    venv\Scripts\activate     # For Windows
    pip install -r requirements.txt
    ```

### Running the Application

1. **Start Kafka**:
   - Create the necessary Kafka topics (`weather-topic` and `tourism-topic`):

     ```bash
     kafka-topics.sh --create --topic weather-topic --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
     kafka-topics.sh --create --topic tourism-topic --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
     ```

2. **Run Data Producers**:
   - Start the weather and tourism data producers:

     ```bash
     python scripts/weather_producer.py   # Starts producing weather data
     python scripts/tourism_producer.py  # Starts producing tourism data
     ```

3. **Run Spark Bronze Layer**:
   - Ingest raw data from Kafka topics into the Bronze Layer:

     ```bash
     python scripts/spark_bronze.py
     ```

4. **Run Spark Silver Layer**:
   - Clean and transform the raw data in the Bronze Layer:

     ```bash
     python scripts/spark_silver.py
     ```

5. **Machine Learning and Forecasting**:
   - Train machine learning models to predict tourism arrivals based on processed data:

     ```bash
     python scripts/ml_model.py
     ```

6. **Visualization**:
   - Open the **Power BI** dashboard to visualize trends and predictions.
   - Launch the **Streamlit app** for real-time interactive analysis:

     ```bash
     streamlit run app.py
     ```

## File Structure

- **scripts/**: Contains all Python scripts for data collection, processing, and machine learning.
  - `fetch_weather.py`: Fetches real-time weather data.
  - `fetch_tourism.py`: Fetches tourism data from World Bank and MOT.
  - `spark_bronze.py`: Ingests raw data from Kafka into the Bronze Layer (Parquet).
  - `spark_silver.py`: Cleans and transforms data for the Silver Layer.
  - `ml_model.py`: Trains machine learning models to predict tourism arrivals.
  - `app.py`: Streamlit app for real-time interactive dashboard.

- **data/**: Stores raw and processed data.
  - `raw/`: Stores raw data files (CSV, JSON, etc.).
  - `bronze/`: Stores raw Parquet data in the Bronze Layer.
  - `silver/`: Stores cleaned and transformed Parquet data in the Silver Layer.
  - `gold/`: Stores aggregated and optimized Parquet data for analysis.

## Contributing

Feel free to fork this project and contribute by opening pull requests. If you encounter any issues or have suggestions for improvements, please open an issue on the [GitHub repository](https://github.com/your-repo).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
