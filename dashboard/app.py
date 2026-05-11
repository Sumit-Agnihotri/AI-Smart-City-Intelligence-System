import gradio as gr
import pandas as pd
import plotly.express as px
import joblib
import os


# --- Load data and models (robust path handling) ---
def _read_csv_try(paths):
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"None of paths exist: {paths}")


try:
    df = _read_csv_try([
        os.path.join('data', 'clean_traffic.csv'),
        os.path.join('..', 'data', 'clean_traffic.csv'),
    ])
except Exception as e:
    raise


def _load_model_try(paths):
    for p in paths:
        if os.path.exists(p):
            return joblib.load(p)
    return None


traffic_model = _load_model_try([
    os.path.join('models', 'traffic_model.pkl'),
    os.path.join('..', 'models', 'traffic_model.pkl'),
])

congestion_model = _load_model_try([
    os.path.join('models', 'congestion_model.pkl'),
    os.path.join('..', 'models', 'congestion_model.pkl'),
])


# Basic model results table (kept from original app)
model_results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'TensorFlow LSTM'],
    'MAE': [0.43, 0.81, 2.87],
    'MSE': [3.50, 3.23, 25.28],
    'R² Score': [0.96, 0.97, 0.75]
})

# Convert DateTime and add features
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month
df['Weekday'] = df['DateTime'].dt.weekday


def predict_and_plots(junction, hour, day, month, weekday):
    filtered_df = df[df['Junction'] == junction]

    # Metrics
    avg_veh = round(filtered_df['Vehicles'].mean(), 2)
    max_veh = int(filtered_df['Vehicles'].max())
    min_veh = int(filtered_df['Vehicles'].min())

    # Prepare input for model (use last observed as lag features)
    latest_vehicle = filtered_df['Vehicles'].iloc[-1]
    input_data = pd.DataFrame({
        'Junction': [junction],
        'Hour': [hour],
        'Day': [day],
        'Month': [month],
        'Weekday': [weekday],
        'Lag_1': [latest_vehicle],
        'Lag_2': [latest_vehicle],
        'Lag_3': [latest_vehicle],
        'Rolling_Mean_3': [latest_vehicle]
    })

    prediction_text = 'Model not loaded'
    congestion_text = 'Model not loaded'

    if traffic_model is not None:
        pred = traffic_model.predict(input_data)
        prediction_text = f"🚗 Predicted Traffic Volume: {pred[0]:.2f} vehicles"

    if congestion_model is not None:
        congestion_pred = congestion_model.predict(
            input_data[['Junction', 'Hour', 'Day', 'Month', 'Weekday']]
        )
        status = congestion_pred[0]
        if status == 'Low':
            congestion_text = '🟢 Low Traffic Congestion'
        elif status == 'Medium':
            congestion_text = '🟡 Medium Traffic Congestion'
        else:
            congestion_text = '🔴 High Traffic Congestion'

    # Plots
    hourly_traffic = filtered_df.groupby('Hour')['Vehicles'].mean().reset_index()
    fig_hour = px.line(hourly_traffic, x='Hour', y='Vehicles', markers=True, title='Average Traffic by Hour')

    daily_traffic = filtered_df.groupby('Day')['Vehicles'].mean().reset_index()
    fig_day = px.line(daily_traffic, x='Day', y='Vehicles', markers=True, title='Average Daily Traffic')

    fig_hist = px.histogram(filtered_df, x='Vehicles', nbins=30, title='Vehicle Distribution')

    weekday_traffic = filtered_df.groupby('Weekday')['Vehicles'].mean().reset_index()
    fig_weekday = px.bar(weekday_traffic, x='Weekday', y='Vehicles', title='Average Traffic by Weekday')

    monthly_traffic = filtered_df.groupby('Month')['Vehicles'].mean().reset_index()
    fig_month = px.line(monthly_traffic, x='Month', y='Vehicles', markers=True, title='Monthly Traffic Trend')

    junction_traffic = df.groupby('Junction')['Vehicles'].mean().reset_index()
    fig_junction = px.bar(junction_traffic, x='Junction', y='Vehicles', color='Junction', title='Average Traffic by Junction')

    heatmap_data = filtered_df.groupby(['Hour', 'Weekday'])['Vehicles'].mean().reset_index()
    fig_heatmap = px.density_heatmap(heatmap_data, x='Hour', y='Weekday', z='Vehicles', title='Traffic Density Heatmap')

    preview_df = filtered_df.head(20).copy()
    preview_df['DateTime'] = preview_df['DateTime'].astype(str)

    metrics = {
        'avg': avg_veh,
        'max': max_veh,
        'min': min_veh
    }

    return (
        prediction_text,
        congestion_text,
        metrics,
        fig_hour,
        fig_day,
        fig_hist,
        fig_weekday,
        fig_month,
        fig_junction,
        fig_heatmap,
        model_results,
        preview_df
    )


def get_junctions():
    def _py_val(x):
        try:
            return x.item()
        except Exception:
            return x

    return [_py_val(x) for x in sorted(df['Junction'].unique())]


with gr.Blocks(title="AI Smart City Traffic Dashboard") as demo:
    gr.Markdown("# 🌆 AI Smart City Traffic Dashboard")
    with gr.Row():
        with gr.Column(scale=1):
            junction = gr.Dropdown(choices=get_junctions(), label="Select Junction")
            hour = gr.Slider(minimum=0, maximum=23, step=1, value=12, label="Hour")
            day = gr.Slider(minimum=1, maximum=31, step=1, value=15, label="Day")
            month = gr.Slider(minimum=1, maximum=12, step=1, value=6, label="Month")
            weekday = gr.Slider(minimum=0, maximum=6, step=1, value=3, label="Weekday")
            predict_btn = gr.Button("Predict Traffic")
            avg_box = gr.Textbox(label="Average Vehicles")
            max_box = gr.Textbox(label="Maximum Vehicles")
            min_box = gr.Textbox(label="Minimum Vehicles")
        with gr.Column(scale=2):
            out_pred = gr.Textbox(label="Prediction")
            out_cong = gr.Textbox(label="Congestion Status")
            tab = gr.Tabs()
            with tab:
                with gr.TabItem('Hour'):
                    plot_hour = gr.Plot(label='Traffic by Hour')
                with gr.TabItem('Daily'):
                    plot_day = gr.Plot(label='Daily Traffic')
                with gr.TabItem('Distribution'):
                    plot_hist = gr.Plot(label='Vehicle Distribution')
                with gr.TabItem('Weekday'):
                    plot_weekday = gr.Plot(label='Traffic by Weekday')
                with gr.TabItem('Monthly'):
                    plot_month = gr.Plot(label='Monthly Traffic')
                with gr.TabItem('Junctions'):
                    plot_junction = gr.Plot(label='Junction Comparison')
                with gr.TabItem('Heatmap'):
                    plot_heat = gr.Plot(label='Heatmap')
                with gr.TabItem('Model Results'):
                    table_models = gr.Dataframe(value=model_results)
                with gr.TabItem('Dataset Preview'):
                    table_preview = gr.Dataframe()

    def _on_predict(junction, hour, day, month, weekday):
        (ptext, ctext, metrics, fh, fd, fhist, fweek, fmonth, fjunc, fheat, mres, preview) = predict_and_plots(junction, hour, day, month, weekday)
        return (
            ptext,
            ctext,
            str(metrics['avg']),
            str(metrics['max']),
            str(metrics['min']),
            fh,
            fd,
            fhist,
            fweek,
            fmonth,
            fjunc,
            fheat,
            mres,
            preview
        )

    predict_btn.click(
        _on_predict,
        inputs=[junction, hour, day, month, weekday],
        outputs=[
            out_pred,
            out_cong,
            avg_box,
            max_box,
            min_box,
            plot_hour,
            plot_day,
            plot_hist,
            plot_weekday,
            plot_month,
            plot_junction,
            plot_heat,
            table_models,
            table_preview,
        ]
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)