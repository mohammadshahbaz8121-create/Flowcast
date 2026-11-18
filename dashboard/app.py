
from pathlib import Path
from datetime import datetime, date, time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FlowCast | Smart Traffic Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "flowcast_historical_features.csv"
REGRESSION_MODEL_PATH = BASE_DIR / "models" / "best_traffic_model.pkl"
CLASSIFIER_MODEL_PATH = BASE_DIR / "models" / "best_congestion_classifier.pkl"


# =========================================================
# PREMIUM DARK THEME
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080F1D;
    color: #E8F0FA;
}

[data-testid="stSidebar"] {
    background: #0D1728;
    border-right: 1px solid #1C2C43;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3 {
    color: #F1F6FF !important;
}

p, label, span {
    color: #B8C7DA;
}

.hero {
    background: linear-gradient(120deg, #10243B, #102D40, #123B4C);
    border: 1px solid #24475C;
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 35px;
    font-weight: 800;
    color: #F4FAFF;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 14px;
    color: #A9C6D8;
    margin-top: 8px;
}

.eyebrow {
    color: #43D9E8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.kpi {
    background: #101D30;
    border: 1px solid #20334C;
    border-radius: 15px;
    padding: 20px;
    min-height: 120px;
}

.kpi-label {
    font-size: 12px;
    color: #9DB0C8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.kpi-value {
    font-size: 28px;
    color: #F4FAFF;
    font-weight: 800;
    margin-top: 12px;
}

.kpi-note {
    font-size: 11px;
    color: #43D9E8;
    margin-top: 5px;
}

.section-note {
    color: #8FA4BD;
    font-size: 13px;
}

div[data-testid="stMetric"] {
    background: #101D30;
    border: 1px solid #20334C;
    padding: 15px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: 1px solid #20334C;
}

.stTabs [data-baseweb="tab"] {
    background: #101D30;
    border-radius: 8px 8px 0 0;
    padding: 12px 18px;
    color: #B8C7DA;
}

.stTabs [aria-selected="true"] {
    background: #15394A !important;
    color: #43D9E8 !important;
}

div[data-testid="stForm"] {
    background: #101D30;
    padding: 22px;
    border: 1px solid #20334C;
    border-radius: 15px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #16B9C9;
    color: #07121F;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    padding: 10px 20px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #43D9E8;
    color: #07121F;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #20334C;
    border-radius: 12px;
}

hr {
    border-color: #20334C;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA AND MODELS
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )

    df["traffic_volume"] = pd.to_numeric(
        df["traffic_volume"], errors="coerce"
    )

    df["avg_speed"] = pd.to_numeric(
        df["avg_speed"], errors="coerce"
    )

    df["road_id"] = df["road_id"].astype(str)
    df["weather_condition"] = df["weather_condition"].astype(str)

    return df


@st.cache_resource
def load_models():
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
    classifier_model = joblib.load(CLASSIFIER_MODEL_PATH)

    return regression_model, classifier_model


try:
    df = load_data()
    regression_model, classifier_model = load_models()

except Exception as error:
    st.error(f"Unable to load FlowCast resources: {error}")
    st.info(
        "Check that the dataset and both trained models exist "
        "in their expected project folders."
    )
    st.stop()


# =========================================================
# VALIDATE MODEL FEATURES
# =========================================================

MODEL_FEATURES = [
    "road_id",
    "hour",
    "minute",
    "day_of_week",
    "month",
    "is_weekend",
    "is_morning_rush",
    "is_evening_rush",
    "temperature",
    "rainfall",
    "visibility",
    "weather_condition",
    "public_holiday",
    "event_flag",
    "roadwork_flag",
    "traffic_volume_lag_1",
    "traffic_volume_lag_2",
    "traffic_volume_lag_3",
    "avg_speed_lag_1",
    "avg_speed_lag_2",
    "traffic_rolling_mean_3",
    "traffic_rolling_mean_6",
    "traffic_change"
]

for model in [regression_model, classifier_model]:
    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != MODEL_FEATURES:
            st.error("Model feature names do not match the dashboard.")
            st.stop()


# =========================================================
# GENERAL HELPERS
# =========================================================

def format_number(value):
    return f"{value:,.0f}"


def make_chart(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101D30",
        plot_bgcolor="#101D30",
        font=dict(color="#C5D4E5", family="Inter"),
        height=height,
        margin=dict(l=15, r=15, t=45, b=15),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=1.12
        )
    )

    fig.update_xaxes(
        gridcolor="#20334C",
        zerolinecolor="#20334C"
    )

    fig.update_yaxes(
        gridcolor="#20334C",
        zerolinecolor="#20334C"
    )

    return fig


def get_congestion_color(level):
    colors = {
        "Free-flow": "#25C997",
        "Moderate": "#F2C14E",
        "Heavy": "#F28C45",
        "Severe": "#F05265"
    }

    return colors.get(level, "#43D9E8")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("""
<div style="padding: 15px 5px 25px 5px;">
    <div style="font-size: 28px; font-weight: 800; color: #43D9E8;">
        FLOWCAST
    </div>
    <div style="font-size: 11px; letter-spacing: 2px; color: #91A9C4;">
        SMART CITY INTELLIGENCE
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Traffic Analytics",
        "Congestion Analysis",
        "ML Predictions",
        "Model Performance",
        "Data Explorer"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### DATA INFORMATION")
st.sidebar.write(f"Records: {len(df):,}")
st.sidebar.write(f"Road segments: {df['road_id'].nunique()}")
st.sidebar.write(
    f"Period: {df['datetime'].min():%d %b %Y} – "
    f"{df['datetime'].max():%d %b %Y}"
)

st.sidebar.markdown("---")
st.sidebar.caption("FlowCast | ML Traffic Intelligence")
st.sidebar.caption("Historical dataset: 2025")


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">INTELLIGENT TRAFFIC ANALYTICS</div>
    <div class="hero-title">FlowCast</div>
    <div class="hero-subtitle">
        Smart Traffic Flow Prediction & Congestion Intelligence
        <br>
        Machine learning insights for smarter urban mobility.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.subheader("System Overview")

    total_records = len(df)
    total_roads = df["road_id"].nunique()
    avg_volume = df["traffic_volume"].mean()
    avg_speed = df["avg_speed"].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">Traffic Records</div>
            <div class="kpi-value">{total_records:,}</div>
            <div class="kpi-note">Historical observations</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">Road Segments</div>
            <div class="kpi-value">{total_roads}</div>
            <div class="kpi-note">Monitored road sections</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">Average Traffic</div>
            <div class="kpi-value">{avg_volume:,.0f}</div>
            <div class="kpi-note">Recorded traffic volume</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">Average Speed</div>
            <div class="kpi-value">{avg_speed:.1f}</div>
            <div class="kpi-note">Recorded average speed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    left, right = st.columns(2)

    with left:
        st.markdown("### Traffic Volume Trend")

        trend = (
            df.dropna(subset=["datetime", "traffic_volume"])
            .groupby(df["datetime"].dt.date)["traffic_volume"]
            .mean()
            .reset_index(name="traffic_volume")
        )

        trend.columns = ["Date", "Traffic Volume"]

        fig = px.line(
            trend,
            x="Date",
            y="Traffic Volume",
            markers=True
        )

        fig.update_traces(line_color="#43D9E8")
        st.plotly_chart(make_chart(fig), use_container_width=True)

    with right:
        st.markdown("### Congestion Distribution")

        congestion = (
            df["congestion_level"]
            .value_counts()
            .reset_index()
        )

        congestion.columns = ["Congestion", "Records"]

        fig = px.pie(
            congestion,
            names="Congestion",
            values="Records",
            hole=0.58,
            color="Congestion",
            color_discrete_map={
                "Free-flow": "#25C997",
                "Moderate": "#F2C14E",
                "Heavy": "#F28C45",
                "Severe": "#F05265"
            }
        )

        st.plotly_chart(make_chart(fig), use_container_width=True)

    st.markdown("### Congestion Summary")

    summary = (
        df.groupby("congestion_level")
        .agg(
            Records=("traffic_volume", "count"),
            Average_Traffic=("traffic_volume", "mean"),
            Average_Speed=("avg_speed", "mean")
        )
        .reset_index()
    )

    st.dataframe(
        summary.round(2),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# TRAFFIC ANALYTICS
# =========================================================

elif page == "Traffic Analytics":

    st.subheader("Traffic Analytics")
    st.markdown(
        '<p class="section-note">Explore traffic patterns across road segments.</p>',
        unsafe_allow_html=True
    )

    roads = sorted(df["road_id"].dropna().unique())

    selected_road = st.selectbox(
        "Select Road Segment",
        ["All Roads"] + roads
    )

    filtered = df.copy()

    if selected_road != "All Roads":
        filtered = filtered[
            filtered["road_id"] == selected_road
        ]

    left, right = st.columns(2)

    with left:
        st.markdown("### Traffic by Hour")

        hourly = (
            filtered.groupby("hour")["traffic_volume"]
            .mean()
            .reset_index()
        )

        fig = px.area(
            hourly,
            x="hour",
            y="traffic_volume",
            markers=True
        )

        fig.update_traces(
            line_color="#43D9E8",
            fillcolor="rgba(67,217,232,0.12)"
        )

        fig.update_xaxes(title="Hour of Day")
        fig.update_yaxes(title="Average Traffic Volume")

        st.plotly_chart(make_chart(fig), use_container_width=True)

    with right:
        st.markdown("### Average Speed by Hour")

        speed = (
            filtered.groupby("hour")["avg_speed"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            speed,
            x="hour",
            y="avg_speed",
            markers=True
        )

        fig.update_traces(line_color="#F2C14E")
        fig.update_xaxes(title="Hour of Day")
        fig.update_yaxes(title="Average Speed")

        st.plotly_chart(make_chart(fig), use_container_width=True)

    st.markdown("### Traffic Volume by Road")

    road_comparison = (
        filtered.groupby(["road_id", "road_name"])["traffic_volume"]
        .mean()
        .reset_index()
        .sort_values("traffic_volume", ascending=False)
    )

    fig = px.bar(
        road_comparison,
        x="road_id",
        y="traffic_volume",
        hover_name="road_name"
    )

    fig.update_traces(marker_color="#43D9E8")
    st.plotly_chart(make_chart(fig), use_container_width=True)

    st.markdown("### Weather and Traffic")

    weather = (
        filtered.groupby("weather_condition")["traffic_volume"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        weather,
        x="weather_condition",
        y="traffic_volume",
        color="weather_condition"
    )

    st.plotly_chart(make_chart(fig), use_container_width=True)


# =========================================================
# CONGESTION ANALYSIS
# =========================================================

elif page == "Congestion Analysis":

    st.subheader("Congestion Intelligence")

    congestion_order = [
        "Free-flow",
        "Moderate",
        "Heavy",
        "Severe"
    ]

    congestion_counts = (
        df["congestion_level"]
        .value_counts()
        .reindex(congestion_order, fill_value=0)
        .reset_index()
    )

    congestion_counts.columns = ["Level", "Records"]

    left, right = st.columns(2)

    with left:
        st.markdown("### Congestion Levels")

        fig = px.bar(
            congestion_counts,
            x="Level",
            y="Records",
            color="Level",
            color_discrete_map={
                "Free-flow": "#25C997",
                "Moderate": "#F2C14E",
                "Heavy": "#F28C45",
                "Severe": "#F05265"
            }
        )

        st.plotly_chart(make_chart(fig), use_container_width=True)

    with right:
        st.markdown("### Congestion by Hour")

        hourly_congestion = (
            df.groupby(["hour", "congestion_level"])
            .size()
            .reset_index(name="Records")
        )

        fig = px.bar(
            hourly_congestion,
            x="hour",
            y="Records",
            color="congestion_level",
            category_orders={
                "congestion_level": congestion_order
            },
            color_discrete_map={
                "Free-flow": "#25C997",
                "Moderate": "#F2C14E",
                "Heavy": "#F28C45",
                "Severe": "#F05265"
            }
        )

        st.plotly_chart(make_chart(fig), use_container_width=True)

    st.markdown("### Congestion by Road")

    road_congestion = (
        df.groupby(["road_id", "congestion_level"])
        .size()
        .reset_index(name="Records")
    )

    fig = px.bar(
        road_congestion,
        x="road_id",
        y="Records",
        color="congestion_level",
        category_orders={
            "congestion_level": congestion_order
        },
        color_discrete_map={
            "Free-flow": "#25C997",
            "Moderate": "#F2C14E",
            "Heavy": "#F28C45",
            "Severe": "#F05265"
        }
    )

    st.plotly_chart(make_chart(fig, 450), use_container_width=True)

    st.markdown("### Congestion Classification Method")

    st.write("""
    Congestion labels in this project are derived from the
    traffic volume-to-capacity ratio:

    - Free-flow: below 0.50
    - Moderate: 0.50 to below 0.80
    - Heavy: 0.80 to below 1.00
    - Severe: 1.00 or higher
    """)


# =========================================================
# ML PREDICTIONS
# =========================================================

elif page == "ML Predictions":

    st.subheader("Machine Learning Prediction Center")

    st.markdown("""
    <p class="section-note">
    Enter a prediction time, road conditions and the six previous
    traffic readings. Both trained models will use the same 23 features.
    </p>
    """, unsafe_allow_html=True)

    st.info(
        "This is a historical-data-based prediction demonstration. "
        "Enter six consecutive previous traffic readings, each 30 minutes "
        "apart, in newest-to-oldest order."
    )

    road_ids = sorted(df["road_id"].dropna().unique())

    road_names = (
        df.drop_duplicates("road_id")
        .set_index("road_id")["road_name"]
        .to_dict()
    )

    with st.form("prediction_form"):

        st.markdown("### 1. Road and Prediction Time")

        col1, col2, col3 = st.columns(3)

        with col1:
            road_id = st.selectbox(
                "Road Segment",
                road_ids
            )

        with col2:
            prediction_date = st.date_input(
                "Prediction Date",
                value=date(2025, 6, 1)
            )

        with col3:
            prediction_time = st.time_input(
                "Prediction Time",
                value=time(8, 0)
            )

        st.caption(
            f"Selected road: {road_names.get(road_id, road_id)}"
        )

        st.markdown("---")
        st.markdown("### 2. Weather and Road Conditions")

        col1, col2, col3 = st.columns(3)

        with col1:
            weather_condition = st.selectbox(
                "Weather",
                [
                    "clear",
                    "cloudy",
                    "overcast",
                    "rain",
                    "fog",
                    "rainy",
                    "foggy"
                ]
            )

            temperature = st.number_input(
                "Temperature (°C)",
                value=28.0,
                min_value=-20.0,
                max_value=60.0
            )

        with col2:
            rainfall = st.number_input(
                "Rainfall",
                value=0.0,
                min_value=0.0,
                max_value=500.0
            )

            visibility = st.number_input(
                "Visibility",
                value=10000.0,
                min_value=0.0,
                max_value=100000.0
            )

        with col3:
            public_holiday = st.selectbox(
                "Public Holiday",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            event_flag = st.selectbox(
                "Special Event",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            roadwork_flag = st.selectbox(
                "Roadwork",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        st.markdown("---")
        st.markdown("### 3. Previous Traffic Readings")

        st.caption(
            "Enter readings from most recent to oldest. "
            "All six traffic readings should be exactly 30 minutes apart."
        )

        volume_cols = st.columns(3)

        volumes = []

        volume_labels = [
            "Latest volume (30 min ago)",
            "2nd latest (60 min ago)",
            "3rd latest (90 min ago)",
            "4th latest (120 min ago)",
            "5th latest (150 min ago)",
            "6th latest (180 min ago)"
        ]

        for i, label in enumerate(volume_labels):
            with volume_cols[i % 3]:
                value = st.number_input(
                    label,
                    min_value=0.0,
                    value=100.0,
                    step=10.0,
                    key=f"volume_{i}"
                )
                volumes.append(value)

        speed_cols = st.columns(2)

        with speed_cols[0]:
            speed_lag_1 = st.number_input(
                "Latest average speed",
                min_value=0.0,
                value=45.0,
                max_value=200.0
            )

        with speed_cols[1]:
            speed_lag_2 = st.number_input(
                "Previous average speed",
                min_value=0.0,
                value=48.0,
                max_value=200.0
            )

        submitted = st.form_submit_button(
            "Generate Traffic Prediction",
            use_container_width=True
        )

    if submitted:

        prediction_datetime = datetime.combine(
            prediction_date,
            prediction_time
        )

        hour = prediction_datetime.hour
        minute = prediction_datetime.minute
        day_of_week = prediction_datetime.weekday()
        month = prediction_datetime.month

        is_weekend = int(day_of_week >= 5)
        is_morning_rush = int(7 <= hour <= 10)
        is_evening_rush = int(16 <= hour <= 19)

        # Inputs are newest to oldest.
        lag_1, lag_2, lag_3, lag_4, lag_5, lag_6 = volumes

        rolling_mean_3 = float(np.mean([lag_1, lag_2, lag_3]))
        rolling_mean_6 = float(np.mean(volumes))

        traffic_change = lag_1 - lag_2

        prediction_features = pd.DataFrame([{
            "road_id": str(road_id),
            "hour": hour,
            "minute": minute,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "is_morning_rush": is_morning_rush,
            "is_evening_rush": is_evening_rush,
            "temperature": temperature,
            "rainfall": rainfall,
            "visibility": visibility,
            "weather_condition": str(weather_condition),
            "public_holiday": int(public_holiday),
            "event_flag": int(event_flag),
            "roadwork_flag": int(roadwork_flag),
            "traffic_volume_lag_1": lag_1,
            "traffic_volume_lag_2": lag_2,
            "traffic_volume_lag_3": lag_3,
            "avg_speed_lag_1": speed_lag_1,
            "avg_speed_lag_2": speed_lag_2,
            "traffic_rolling_mean_3": rolling_mean_3,
            "traffic_rolling_mean_6": rolling_mean_6,
            "traffic_change": traffic_change
        }], columns=MODEL_FEATURES)

        try:
            predicted_volume = float(
                regression_model.predict(prediction_features)[0]
            )

            predicted_congestion = str(
                classifier_model.predict(prediction_features)[0]
            )

            st.markdown("---")
            st.markdown("## Prediction Results")

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div class="kpi">
                    <div class="kpi-label">Predicted Traffic Volume</div>
                    <div class="kpi-value">
                        {predicted_volume:,.0f}
                    </div>
                    <div class="kpi-note">
                        Regression model output
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                color = get_congestion_color(predicted_congestion)

                st.markdown(f"""
                <div class="kpi">
                    <div class="kpi-label">Predicted Congestion</div>
                    <div class="kpi-value" style="color:{color};">
                        {predicted_congestion}
                    </div>
                    <div class="kpi-note">
                        Classification model output
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Prediction Input Summary")

            st.dataframe(
                prediction_features.T.rename(
                    columns={0: "Input Value"}
                ),
                use_container_width=True
            )

            st.caption(
                "These predictions depend on the values entered and "
                "the historical patterns learned by the trained models. "
                "They are not live sensor readings."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.subheader("Machine Learning Model Performance")

    st.markdown(
        '<p class="section-note">Final evaluation results from the saved models.</p>',
        unsafe_allow_html=True
    )

    st.markdown("### Traffic Volume Regression")

    c1, c2, c3 = st.columns(3)

    c1.metric("R² Score", "0.9502")
    c2.metric("MAE", "43.4246")
    c3.metric("RMSE", "63.8286")

    st.markdown("### Congestion Classification")

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", "85.50%")
    c2.metric("Weighted F1", "85.69%")
    c3.metric("Macro F1", "73.70%")

    st.markdown("---")
    st.markdown("### Model Comparison")

    comparison = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R² Score"],
        "Baseline Random Forest": [59.0566, 93.3983, 0.8933],
        "Historical Feature Random Forest": [
            43.4246,
            63.8286,
            0.9502
        ]
    })

    st.dataframe(
        comparison.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Improvement from Historical Features")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE Reduction", "26.5%")
    c2.metric("RMSE Reduction", "31.7%")
    c3.metric("R² Improvement", "+0.0569")

    st.markdown("---")

    st.markdown("### Model Information")

    st.write("""
    **Regression:** Historical Feature Random Forest

    **Classification:** Historical Feature Random Forest

    **Input features:** 23

    **Training approach:** Historical traffic features with
    chronological train-test separation.

    The reported results are evaluation metrics, not guarantees
    of performance on new or live traffic data.
    """)


# =========================================================
# DATA EXPLORER
# =========================================================

elif page == "Data Explorer":

    st.subheader("Traffic Dataset Explorer")

    st.markdown(
        '<p class="section-note">Inspect, filter and download the processed dataset.</p>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        selected_roads = st.multiselect(
            "Filter by Road",
            sorted(df["road_id"].dropna().unique()),
            default=[]
        )

    with c2:
        selected_congestion = st.multiselect(
            "Filter by Congestion",
            sorted(df["congestion_level"].dropna().unique()),
            default=[]
        )

    filtered = df.copy()

    if selected_roads:
        filtered = filtered[
            filtered["road_id"].isin(selected_roads)
        ]

    if selected_congestion:
        filtered = filtered[
            filtered["congestion_level"].isin(selected_congestion)
        ]

    st.metric("Filtered Records", f"{len(filtered):,}")

    st.dataframe(
        filtered.head(1000),
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Dataset",
        data=csv_data,
        file_name="flowcast_filtered_data.csv",
        mime="text/csv"
    )

    st.markdown("### Dataset Summary")

    st.dataframe(
        filtered.describe(include="all").T,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:15px; color:#71869F;">
    <div style="font-size:14px; font-weight:700; color:#43D9E8;">
        FLOWCAST
    </div>
    <div style="font-size:11px;">
        Smart Traffic Flow Prediction System |
        Machine Learning & Data Analytics
    </div>
    <div style="font-size:10px; margin-top:5px;">
        Historical dataset: January – May 2025
    </div>
</div>
""", unsafe_allow_html=True)
