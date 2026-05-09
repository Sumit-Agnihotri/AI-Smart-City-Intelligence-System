import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Page Config
st.set_page_config(
    page_title="AI Smart City Dashboard",
    layout="wide"
)

# Title
st.title("🌆 AI Smart City Traffic Dashboard")
model_results = pd.DataFrame({
    'Model': [
        'Random Forest',
        'XGBoost',
        'TensorFlow LSTM'
    ],

    'MAE': [
        0.43,
        0.81,
        2.87
    ],

    'MSE': [
        3.50,
        3.23,
        25.28
    ],

    'R² Score': [
        0.96,
        0.97,
        0.75
    ]
})

# Load Dataset
try:
    df = pd.read_csv("data/clean_traffic.csv")
except:
    df = pd.read_csv("../data/clean_traffic.csv")

try:
    traffic_model = joblib.load(
        "models/traffic_model.pkl"
    )

except:
    traffic_model = joblib.load(
        "../models/traffic_model.pkl"
    )

try:
    congestion_model = joblib.load(
        "models/congestion_model.pkl"
    )

except:
    congestion_model = joblib.load(
        "../models/congestion_model.pkl"
    )

# Convert DateTime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Feature Engineering
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month
df['Weekday'] = df['DateTime'].dt.weekday

# Sidebar
st.sidebar.header("Dashboard Filters")
st.sidebar.subheader("🚀 Traffic Prediction")

selected_junction = st.sidebar.selectbox(
    "Select Junction",
    sorted(df['Junction'].unique())
)

input_hour = st.sidebar.slider(
    "Select Hour",
    0,
    23,
    12
)

input_day = st.sidebar.slider(
    "Select Day",
    1,
    31,
    15
)

input_month = st.sidebar.slider(
    "Select Month",
    1,
    12,
    6
)

input_weekday = st.sidebar.slider(
    "Select Weekday",
    0,
    6,
    3
)

# Filter Data
filtered_df = df[df['Junction'] == selected_junction]

# Metrics Section
col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Vehicles",
    round(filtered_df['Vehicles'].mean(), 2)
)

col2.metric(
    "Maximum Vehicles",
    int(filtered_df['Vehicles'].max())
)

col3.metric(
    "Minimum Vehicles",
    int(filtered_df['Vehicles'].min())
)
if st.sidebar.button("Predict Traffic"):

    latest_vehicle = filtered_df['Vehicles'].iloc[-1]

    input_data = pd.DataFrame({
        'Junction': [selected_junction],
        'Hour': [input_hour],
        'Day': [input_day],
        'Month': [input_month],
        'Weekday': [input_weekday],
        'Lag_1': [latest_vehicle],
        'Lag_2': [latest_vehicle],
        'Lag_3': [latest_vehicle],
        'Rolling_Mean_3': [latest_vehicle]
    })

    prediction = traffic_model.predict(input_data)
    congestion_prediction = congestion_model.predict(
    input_data[['Junction', 'Hour', 'Day', 'Month', 'Weekday']]
    )

    congestion_status = congestion_prediction[0]

    st.success(
        f"🚗 Predicted Traffic Volume: {prediction[0]:.2f} vehicles")
    if congestion_status == "Low":
        st.success("🟢 Low Traffic Congestion")

    elif congestion_status == "Medium":
        st.warning("🟡 Medium Traffic Congestion")

    else:
        st.error("🔴 High Traffic Congestion")

# Traffic by Hour
st.subheader("🚦 Traffic by Hour")

hourly_traffic = (
    filtered_df
    .groupby('Hour')['Vehicles']
    .mean()
    .reset_index()
)

fig_hour = px.line(
    hourly_traffic,
    x='Hour',
    y='Vehicles',
    markers=True,
    title='Average Traffic by Hour'
)

st.plotly_chart(fig_hour, use_container_width=True)

# Daily Traffic Trend
st.subheader("📈 Daily Traffic Trend")

daily_traffic = (
    filtered_df
    .groupby('Day')['Vehicles']
    .mean()
    .reset_index()
)

fig_day = px.line(
    daily_traffic,
    x='Day',
    y='Vehicles',
    markers=True,
    title='Average Daily Traffic'
)

st.plotly_chart(fig_day, use_container_width=True)

# Vehicle Distribution
st.subheader("🚗 Vehicle Distribution")

fig_hist = px.histogram(
    filtered_df,
    x='Vehicles',
    nbins=30,
    title='Vehicle Distribution'
)

st.plotly_chart(fig_hist, use_container_width=True)

# Traffic by Weekday
st.subheader("📅 Traffic by Weekday")

weekday_traffic = (
    filtered_df
    .groupby('Weekday')['Vehicles']
    .mean()
    .reset_index()
)

fig_weekday = px.bar(
    weekday_traffic,
    x='Weekday',
    y='Vehicles',
    title='Average Traffic by Weekday'
)

st.plotly_chart(fig_weekday, use_container_width=True)

# Monthly Traffic
st.subheader("📆 Monthly Traffic")

monthly_traffic = (
    filtered_df
    .groupby('Month')['Vehicles']
    .mean()
    .reset_index()
)

fig_month = px.line(
    monthly_traffic,
    x='Month',
    y='Vehicles',
    markers=True,
    title='Monthly Traffic Trend'
)

st.plotly_chart(fig_month, use_container_width=True)

# Junction Comparison
st.subheader("🚥 Junction Comparison")

junction_traffic = (
    df.groupby('Junction')['Vehicles']
    .mean()
    .reset_index()
)

fig_junction = px.bar(
    junction_traffic,
    x='Junction',
    y='Vehicles',
    color='Junction',
    title='Average Traffic by Junction'
)

st.plotly_chart(fig_junction, use_container_width=True)

# Heatmap
st.subheader("🔥 Traffic Heatmap")

heatmap_data = (
    filtered_df
    .groupby(['Hour', 'Weekday'])['Vehicles']
    .mean()
    .reset_index()
)

fig_heatmap = px.density_heatmap(
    heatmap_data,
    x='Hour',
    y='Weekday',
    z='Vehicles',
    title='Traffic Density Heatmap'
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# Model Comparison
st.subheader("🤖 Model Performance Comparison")

st.dataframe(model_results)

# Model Comparison
st.subheader("🤖 Model Performance Comparison")

st.dataframe(model_results)

fig_models = px.bar(
    model_results,
    x='Model',
    y='R² Score',
    color='Model',
    title='Model Accuracy Comparison'
)

st.plotly_chart(
    fig_models,
    use_container_width=True
)

# Dataset Preview
st.subheader("📄 Dataset Preview")

preview_df = filtered_df.head(20).copy()

preview_df['DateTime'] = preview_df['DateTime'].astype(str)

st.dataframe(preview_df)