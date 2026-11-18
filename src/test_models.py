
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REGRESSION_PATH = ROOT / "models" / "best_traffic_model.pkl"
CLASSIFIER_PATH = ROOT / "models" / "best_congestion_classifier.pkl"

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


def create_test_input():
    values = {
        "road_id": "ROAD_001",
        "hour": 8,
        "minute": 30,
        "day_of_week": 2,
        "month": 3,
        "is_weekend": 0,
        "is_morning_rush": 1,
        "is_evening_rush": 0,
        "temperature": 28.0,
        "rainfall": 0.0,
        "visibility": 10.0,
        "weather_condition": "Clear",
        "public_holiday": 0,
        "event_flag": 0,
        "roadwork_flag": 0,
        "traffic_volume_lag_1": 850,
        "traffic_volume_lag_2": 820,
        "traffic_volume_lag_3": 800,
        "avg_speed_lag_1": 35.0,
        "avg_speed_lag_2": 38.0,
        "traffic_rolling_mean_3": 823.33,
        "traffic_rolling_mean_6": 810.0,
        "traffic_change": 30
    }

    return pd.DataFrame([values], columns=MODEL_FEATURES)


def test_model_files_exist():
    assert REGRESSION_PATH.exists()
    assert CLASSIFIER_PATH.exists()


def test_models_load():
    regression_model = joblib.load(REGRESSION_PATH)
    classifier_model = joblib.load(CLASSIFIER_PATH)

    assert regression_model is not None
    assert classifier_model is not None


def test_model_features_match_dashboard():
    regression_model = joblib.load(REGRESSION_PATH)
    classifier_model = joblib.load(CLASSIFIER_PATH)

    assert list(regression_model.feature_names_in_) == MODEL_FEATURES
    assert list(classifier_model.feature_names_in_) == MODEL_FEATURES


def test_input_has_correct_features():
    test_input = create_test_input()

    assert list(test_input.columns) == MODEL_FEATURES
    assert test_input.shape == (1, 23)


def test_regression_prediction():
    regression_model = joblib.load(REGRESSION_PATH)
    test_input = create_test_input()

    prediction = regression_model.predict(test_input)[0]

    assert np.isfinite(float(prediction))


def test_classifier_prediction():
    classifier_model = joblib.load(CLASSIFIER_PATH)
    test_input = create_test_input()

    prediction = classifier_model.predict(test_input)[0]

    assert prediction in classifier_model.classes_


def test_both_models_accept_same_input():
    regression_model = joblib.load(REGRESSION_PATH)
    classifier_model = joblib.load(CLASSIFIER_PATH)

    test_input = create_test_input()

    regression_prediction = regression_model.predict(test_input)[0]
    classification_prediction = classifier_model.predict(test_input)[0]

    assert np.isfinite(float(regression_prediction))
    assert classification_prediction in classifier_model.classes_
