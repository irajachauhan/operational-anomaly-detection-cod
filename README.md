# Operational Anomaly Detection and COD Soft Sensing

Machine learning pipeline for **COD prediction and operational anomaly detection** using daily wastewater treatment plant data from the Eastern Treatment Plant, Melbourne.

**Roll Number:** 230107055

## 🔍 Project Overview

The project combines:

* **COD prediction** using Linear Regression, Random Forest, and XGBoost
* **Point anomaly detection** using Isolation Forest
* **Temporal anomaly detection** using a 7-day LSTM Autoencoder
* **SHAP** for explaining XGBoost COD predictions
* Interactive **Streamlit dashboard** for prediction and anomaly analysis

A time-ordered **70/10/20 train-validation-test split** was used without random shuffling.

## 📊 Dataset

* **1,382 daily records**
* **1 January 2014 – 27 June 2019**
* Plant and weather variables including inflow, outflow, energy, ammonia, nitrogen, temperature, humidity, precipitation, visibility, and wind
* Target: **COD (mg/L)**

Feature engineering includes cyclic month encoding, lag features, rolling averages, and interaction terms.

## 📈 Results

### COD Prediction — Test Set

| Model             |       MAE |      RMSE |        R² |
| ----------------- | --------: | --------: | --------: |
| Linear Regression |     72.69 |     92.11 |     0.509 |
| Random Forest     |     58.47 | **75.81** | **0.668** |
| XGBoost           | **58.03** |     76.32 |     0.663 |

Both tree-based models substantially outperform the linear baseline, with XGBoost achieving the lowest MAE.

### Anomaly Detection

| Result                |  Days |
| --------------------- | ----: |
| Both methods          |    47 |
| Isolation Forest only |    53 |
| LSTM only             |   184 |
| Neither               | 1,097 |

Isolation Forest identifies unusual individual operating points, while the LSTM Autoencoder detects unusual **7-day temporal patterns**.

## 🖥️ Streamlit Dashboard

The dashboard provides:

* 🧪 COD prediction with SHAP explanation
* 🚨 Isolation Forest point anomaly detection
* 🧠 LSTM-based temporal anomaly detection
* 📊 Historical anomaly visualization

Make sure uploaded CSV contains exactly these columns and 7 rows of data to successfully run the LSTM Autoencoder anomaly detection:

avg_outflow, avg_inflow, total_grid, Am, BOD, COD, TN, T, TM, Tm, SLP, H, PP, VV, V, VM, VG, year, month, day


Run locally:

```bash
pip install -r requirements.txt
streamlit run deployment/app.py
```

## 📁 Repository Structure

```text
operational-anomaly-detection-cod/
├── deployment/     # Streamlit app + trained models
├── notebooks/      # Complete ML workflow
├── data/           # Project dataset
├── plots/          # Exported visualizations
├── report/         # Final project report
├── requirements.txt
├── .python-version
└── README.md
```

## 📄 Project Files

* [Model development notebook](notebooks/230107055.ipynb)
* [Final project report](report/230107055.docx)
* [Visualizations](plots/)
* [Streamlit application](deployment/app.py)
