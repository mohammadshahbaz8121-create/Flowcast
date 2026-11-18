
# FlowCast – Smart Traffic Flow Prediction System

<p align="center">
  <strong>An ML-powered traffic analytics and congestion prediction platform</strong>
</p>

FlowCast is a machine learning-based traffic analytics system designed to analyze historical traffic patterns, predict traffic volume, and classify road congestion levels.

The project combines data preprocessing, feature engineering, machine learning, model evaluation, and an interactive Streamlit dashboard to transform historical traffic data into meaningful insights.

---

## Table of Contents

- Overview
- Key Features
- Dashboard Modules
- Machine Learning Models
- Model Performance
- System Architecture
- Technologies Used
- Dataset Information
- Project Structure
- Installation
- Running the Application
- Model Validation
- Limitations
- Future Improvements
- Author

---

## Overview

Traffic congestion is a major challenge in urban transportation. Understanding traffic patterns can help identify congestion trends and support better traffic analysis.

FlowCast uses historical traffic observations and engineered features to:

- Estimate traffic volume.
- Identify road congestion levels.
- Analyze historical traffic patterns.
- Explore relationships between traffic and weather conditions.
- Present model predictions through an interactive dashboard.

The system is developed using Python and machine learning libraries, with Streamlit providing the user interface.

## Key Features

- Historical traffic data analysis.
- Traffic volume prediction using regression.
- Congestion level classification.
- Historical traffic feature engineering.
- Weather and calendar-based analysis.
- Interactive traffic analytics dashboard.
- Model performance evaluation.
- Interactive prediction interface.
- Data exploration and visualization.

## Dashboard Modules

FlowCast includes six interactive dashboard modules.

### 1. Overview
Provides a general summary of traffic data and important traffic indicators.

### 2. Traffic Analytics
Explores historical traffic volume patterns and traffic trends.

### 3. Congestion Analysis
Analyzes congestion levels and their distribution across road segments.

### 4. ML Predictions
Provides an interactive interface for generating traffic volume and congestion predictions.

### 5. Model Performance
Displays machine learning evaluation results and performance metrics.

### 6. Data Explorer
Allows users to explore the underlying traffic dataset.

---

## Machine Learning Models

FlowCast implements two primary machine learning tasks.

### 1. Traffic Volume Prediction

**Problem Type:** Regression

The regression model estimates traffic volume using historical traffic observations and engineered features.

Evaluation metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### 2. Congestion Level Classification

**Problem Type:** Multi-class Classification

The classification model predicts one of four congestion categories:

| Class | Description |
|---|---|
| Free-flow | Low congestion |
| Moderate | Moderate traffic conditions |
| Heavy | High traffic congestion |
| Severe | Very high traffic congestion |

The congestion target is derived using the traffic volume-to-road capacity ratio.

---

## Model Performance

The following results were obtained during chronological holdout validation.

### Regression Performance

| Metric | Result |
|---|---:|
| MAE | 43.4246 |
| RMSE | 63.8286 |
| R² Score | 0.9502 |

### Classification Performance

| Metric | Result |
|---|---:|
| Accuracy | 85.50% |
| Weighted F1 Score | 85.69% |
| Macro F1 Score | 73.70% |

### Validation Dataset

| Model | Training Records | Testing Records |
|---|---:|---:|
| Regression | 137,691 | 34,423 |
| Classification | 120,061 | 30,016 |

These results represent performance on the project's historical holdout dataset. They should not be interpreted as guaranteed performance on new roads or future traffic conditions.

---

## System Architecture

The project follows a structured machine learning workflow.

```text
          Historical Traffic Data
                    |
                    v
             Data Processing
                    |
                    v
             Data Cleaning
                    |
                    v
            Feature Engineering
                    |
                    v
             Model Training
                    |
          +---------+---------+
          |                   |
          v                   v
    Regression Model    Classification Model
          |                   |
          +---------+---------+
                    |
                    v
              Model Validation
                    |
                    v
             Saved ML Models
                    |
                    v
          Streamlit Dashboard
                    |
                    v
          Interactive Predictions
```

---

## Technologies Used

### Programming Language
- Python 3.11

### Machine Learning and Data Science
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Data Visualization
- Plotly
- Matplotlib

### Dashboard Development
- Streamlit

### Development Environment
- Visual Studio Code
- PowerShell
- Git and GitHub

---

## Dataset Information

The project uses historical traffic sensor observations covering:

- **Period:** January 1, 2025 – May 31, 2025
- **Road segments:** 25
- **Traffic observations:** Approximately 178,000
- **Traffic interval:** 30 minutes
- **Additional information:** Weather and calendar features

Weather observations were aligned with the traffic time intervals during preprocessing.

The dataset was cleaned and transformed before model training.

---

## Project Structure

```text
Flowcast/
│
├── dashboard/
│   ├── app.py
│   └── app_old_backup.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── check_setup.py
│   ├── inspect_data.py
│   ├── analyze_quality.py
│   ├── clean_data.py
│   ├── features.py
│   ├── explore_data.py
│   ├── prepare_data.py
│   ├── check_calendar.py
│   ├── train_model.py
│   ├── feature_importance.py
│   ├── compare_models.py
│   ├── train_classifier.py
│   ├── analyze_congestion.py
│   ├── compare_classifiers.py
│   ├── traffic_history_features.py
│   ├── finalize_congestion_model.py
│   ├── finalize_regression_model.py
│   ├── validate_predictions.py
│   ├── check_traffic_intervals.py
│   └── check_weather_join.py
│
├── tests/
│
├── requirements.txt
├── .gitignore
├── README.md
├── inspection_output.txt
└── quality_report.txt
```

**Note:** The trained model files and datasets are not necessarily included in the GitHub repository. Their distribution and setup instructions will be provided separately.

---

## Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the actual repository URL.

### 2. Navigate to the Project

```bash
cd Flowcast
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Configure the Project Files

Ensure the required processed data and trained model files are available in their expected directories.

---

## Running the Application

Start the FlowCast dashboard using:

```bash
streamlit run dashboard/app.py
```

The application will start locally. Open the local URL displayed in your terminal to access the dashboard.

---

## Model Validation

The project includes a validation script:

```bash
python src/validate_predictions.py
```

The script evaluates the saved regression and classification models and generates validation reports.

Generated reports:

```text
reports/
├── regression_validation.csv
└── classification_validation.csv
```

---

## Limitations

- The system is developed using historical traffic data from January to May 2025.
- Predictions depend on the availability and quality of input features.
- Historical model performance does not guarantee future prediction accuracy.
- Weather observations are aligned to 30-minute traffic intervals.
- Congestion classes are derived from traffic volume relative to road capacity.
- The system is an experimental analytics and prediction platform, not a production traffic management system.
- Independent validation on additional datasets and road networks is still required.

---

## Future Improvements

- Integrate real-time traffic data sources.
- Explore live weather API integration.
- Improve prediction performance using additional machine learning approaches.
- Explore time-series forecasting models.
- Expand the dataset to additional road networks.
- Improve model monitoring and evaluation.
- Deploy the dashboard to a cloud platform.
- Develop a more comprehensive automated testing suite.

---

## Author

**Mohammed Shahbaz**

Machine Learning and Python Project

**Project:** FlowCast – Smart Traffic Flow Prediction System

---

## Acknowledgement

FlowCast was developed as a practical machine learning project to explore traffic analytics, feature engineering, predictive modeling, and interactive data visualization.

---

<p align="center">
  <strong>FlowCast – Turning Historical Traffic Data into Meaningful Insights.</strong>
</p>
