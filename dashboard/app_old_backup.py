
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="FlowCast | Smart Traffic Analytics",
    page_icon="🚦",
    layout="wide"
)

# -----------------------------------
# PROJECT PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "flowcast_historical_features.csv"
)

# -----------------------------------
# LOAD DATA
# -----------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["traffic_volume"] = pd.to_numeric(
        df["traffic_volume"],
        errors="coerce"
    )

    return df.dropna(
        subset=["datetime", "traffic_volume"]
    )


try:
    df = load_data()

except Exception as error:
    st.error(f"Unable to load FlowCast data: {error}")
    st.stop()


# -----------------------------------
# HEADER
# -----------------------------------

st.title("🚦 FlowCast")

st.subheader(
    "Smart Traffic Flow Prediction & Analytics"
)

st.markdown(
    """
    Monitor traffic patterns, understand congestion,
    and explore road-level traffic behaviour.
    """
)

st.divider()


# -----------------------------------
# SIDEBAR FILTERS
# -----------------------------------

st.sidebar.title("Dashboard Filters")

roads = sorted(
    df["road_id"].dropna().unique()
)

selected_road = st.sidebar.selectbox(
    "Select Road",
    ["All Roads"] + list(roads)
)

min_date = df["date"].min().date()
max_date = df["date"].max().date()

selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)


# -----------------------------------
# APPLY FILTERS
# -----------------------------------

filtered_df = df.copy()

if selected_road != "All Roads":
    filtered_df = filtered_df[
        filtered_df["road_id"] == selected_road
    ]

if isinstance(selected_dates, (tuple, list)):

    if len(selected_dates) == 2:

        start_date, end_date = selected_dates

        filtered_df = filtered_df[
            (filtered_df["date"].dt.date >= start_date)
            & (filtered_df["date"].dt.date <= end_date)
        ]

    elif len(selected_dates) == 1:

        filtered_df = filtered_df[
            filtered_df["date"].dt.date == selected_dates[0]
        ]


if filtered_df.empty:

    st.warning(
        "No traffic data available for these filters."
    )

    st.stop()


# -----------------------------------
# TRAFFIC OVERVIEW
# -----------------------------------

st.subheader("📊 Traffic Overview")

col1, col2, col3, col4 = st.columns(4)

total_readings = len(filtered_df)

average_volume = filtered_df[
    "traffic_volume"
].mean()

average_speed = (
    filtered_df["avg_speed"].mean()
    if "avg_speed" in filtered_df.columns
    else None
)

road_count = filtered_df[
    "road_id"
].nunique()


col1.metric(
    "Total Readings",
    f"{total_readings:,}"
)

col2.metric(
    "Average Traffic Volume",
    f"{average_volume:.1f}"
)

col3.metric(
    "Average Speed",
    f"{average_speed:.1f}"
    if pd.notna(average_speed)
    else "N/A"
)

col4.metric(
    "Road Segments",
    f"{road_count}"
)


st.divider()


# -----------------------------------
# TRAFFIC TRENDS
# -----------------------------------

st.subheader("📈 Traffic Trends")

hourly_data = (
    filtered_df
    .groupby("hour", as_index=False)["traffic_volume"]
    .mean()
    .sort_values("hour")
)

fig_hourly = px.line(
    hourly_data,
    x="hour",
    y="traffic_volume",
    markers=True,
    title="Average Traffic Volume by Hour",
    labels={
        "hour": "Hour of Day",
        "traffic_volume": "Average Traffic Volume"
    }
)

fig_hourly.update_layout(
    xaxis=dict(dtick=1),
    template="plotly_white"
)

st.plotly_chart(
    fig_hourly,
    use_container_width=True
)


# DAILY TRAFFIC TREND

daily_data = (
    filtered_df
    .groupby("date", as_index=False)["traffic_volume"]
    .mean()
)

fig_daily = px.line(
    daily_data,
    x="date",
    y="traffic_volume",
    markers=True,
    title="Daily Average Traffic Volume",
    labels={
        "date": "Date",
        "traffic_volume": "Average Traffic Volume"
    }
)

fig_daily.update_layout(
    template="plotly_white"
)

st.plotly_chart(
    fig_daily,
    use_container_width=True
)


st.divider()


# -----------------------------------
# CONGESTION ANALYSIS
# -----------------------------------

st.subheader("🚥 Congestion Analysis")

if "congestion_level" in filtered_df.columns:

    congestion_counts = (
        filtered_df["congestion_level"]
        .value_counts()
        .reset_index()
    )

    congestion_counts.columns = [
        "Congestion Level",
        "Count"
    ]

    col1, col2 = st.columns(2)

    with col1:

        fig_pie = px.pie(
            congestion_counts,
            names="Congestion Level",
            values="Count",
            hole=0.45,
            title="Congestion Distribution"
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with col2:

        fig_bar = px.bar(
            congestion_counts,
            x="Congestion Level",
            y="Count",
            title="Congestion Level Counts",
            text="Count"
        )

        fig_bar.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )


st.divider()


# -----------------------------------
# WEATHER ANALYSIS
# -----------------------------------

st.subheader("🌦️ Weather Analysis")

if "weather_condition" in filtered_df.columns:

    weather_data = (
        filtered_df["weather_condition"]
        .value_counts()
        .reset_index()
    )

    weather_data.columns = [
        "Weather Condition",
        "Readings"
    ]

    fig_weather = px.bar(
        weather_data,
        x="Weather Condition",
        y="Readings",
        title="Weather Conditions Across Selected Readings",
        text="Readings"
    )

    st.plotly_chart(
        fig_weather,
        use_container_width=True
    )


st.divider()


# -----------------------------------
# ROAD-WISE TRAFFIC ANALYSIS
# -----------------------------------

st.subheader("🛣️ Road-wise Traffic Analysis")

road_data = (
    filtered_df
    .groupby("road_id", as_index=False)["traffic_volume"]
    .mean()
    .sort_values(
        "traffic_volume",
        ascending=False
    )
)

fig_road = px.bar(
    road_data,
    x="road_id",
    y="traffic_volume",
    title="Average Traffic Volume by Road",
    labels={
        "road_id": "Road Segment",
        "traffic_volume": "Average Traffic Volume"
    }
)

st.plotly_chart(
    fig_road,
    use_container_width=True
)


st.divider()


# -----------------------------------
# TRAFFIC DATA EXPLORER
# -----------------------------------

st.subheader("📋 Traffic Data Explorer")

display_columns = [
    col for col in [
        "road_id",
        "date",
        "time",
        "traffic_volume",
        "avg_speed",
        "congestion_level",
        "weather_condition"
    ]
    if col in filtered_df.columns
]

st.dataframe(
    filtered_df[display_columns].head(1000),
    use_container_width=True,
    hide_index=True
)


# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

st.caption(
    "FlowCast | Smart Traffic Flow Prediction System"
)

st.caption(
    "Analytics are based on the available historical traffic dataset (January–May 2025)."
)
