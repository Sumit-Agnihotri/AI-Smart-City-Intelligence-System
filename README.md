# 🌆 AI Smart City Intelligence System

## 📌 Project Overview

The **AI Smart City Intelligence System** is an AI-powered traffic monitoring, congestion analysis, and forecasting platform designed to help smart cities analyze and predict traffic conditions efficiently.

This project combines:

- Machine Learning
- Deep Learning
- Data Analysis
- Interactive Visualization
- Real-Time Prediction Systems

to create a modern intelligent traffic analytics dashboard.

The system can:
- analyze historical traffic data
- forecast vehicle traffic
- classify congestion levels
- visualize city traffic patterns
- assist future smart city planning

---

# 🎯 Project Goals

The primary goals of this project are:

✅ Build a complete end-to-end AI project  
✅ Learn practical Machine Learning workflows  
✅ Create deployable AI dashboards  
✅ Simulate real-world Smart City analytics  
✅ Build a portfolio-ready AI application  
✅ Understand traffic pattern analysis using AI  

---

# 🚀 Key Features

## 📈 Traffic Forecasting
Predict future traffic volume using Machine Learning models.

---

## 🚦 Congestion Classification
Classify traffic conditions into:
- 🟢 Low Congestion
- 🟡 Medium Congestion
- 🔴 High Congestion

---

## 📊 Interactive Dashboard
Built with Streamlit and Plotly for real-time interaction.

Features include:
- dropdown filters
- sliders
- live predictions
- dynamic charts
- heatmaps
- analytics panels

---

## 🔥 Traffic Heatmap
Visual representation of traffic intensity based on:
- time
- weekday
- traffic density

---

## 📅 Traffic Trend Analysis
Analyze:
- hourly traffic
- daily traffic
- monthly traffic
- weekday traffic

---

## 🚥 Junction Comparison
Compare average vehicle traffic across multiple city junctions.

---

## 📄 Dataset Preview
Inspect filtered traffic records directly from the dashboard.

---

# 🧠 Machine Learning Models Used

## 1️⃣ Random Forest Regressor
Used for:
- traffic volume prediction

Advantages:
- handles non-linear data well
- strong baseline model
- good performance on tabular data

---

## 2️⃣ XGBoost Regressor
Used for:
- advanced traffic forecasting experiments

Advantages:
- high performance
- boosting-based learning
- strong prediction capability

---

## 3️⃣ TensorFlow LSTM
Used for:
- sequence-based traffic forecasting

Advantages:
- deep learning model
- handles time-series patterns
- captures sequential dependencies

---

## 4️⃣ Random Forest Classifier
Used for:
- congestion classification

Predicts:
- Low
- Medium
- High traffic congestion

---

# 📊 Dashboard Visualizations

The dashboard contains:

## 🚦 Traffic by Hour
Line chart showing average traffic across different hours.

---

## 📈 Daily Traffic Trend
Traffic changes over days.

---

## 🚗 Vehicle Distribution
Histogram visualization of vehicle counts.

---

## 📅 Weekday Traffic Analysis
Traffic comparison across weekdays.

---

## 📆 Monthly Traffic Trend
Monthly vehicle movement analysis.

---

## 🚥 Junction Comparison
Bar chart comparing traffic across junctions.

---

## 🔥 Traffic Density Heatmap
Heatmap showing traffic intensity distribution.

---

## ⚙️ PySpark Pipeline

The repository now includes a local-first PySpark pipeline in `pyspark_pipeline/` that can:

- clean the raw traffic CSV
- create lag and rolling-window features
- export a feature table
- train Spark ML regression and congestion models
- save a JSON run summary for reproducibility

### Why it helps

This pipeline gives the project a more professional and scalable data-processing layer. It reduces manual preprocessing work, makes the feature-generation step reproducible, and prepares the codebase for larger traffic datasets if the project is extended later. It also strengthens the academic value of the project because you can show both a normal Python-based workflow and a Spark-based big-data workflow, which is useful for report writing, viva questions, and future deployment work.

Run it with:

```bash
python -m pyspark_pipeline.traffic_pipeline
```

Use `--no-models` if you only want feature generation.

---

# 🏗️ Project Architecture

```text
User Input
     ↓
Dashboard Filters
     ↓
Feature Engineering
     ↓
Machine Learning Models
     ↓
Prediction System
     ↓
Interactive Visualizations
     ↓
Traffic Insights
