# 🔧 Anomaly Detection — SECOM Manufacturing

> Real-time anomaly detection in semiconductor manufacturing 
> using Isolation Forest on 590 sensor readings, deployed 
> as an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Complete-success)

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Algorithm | Isolation Forest (Unsupervised) |
| Precision | **0.911** |
| Recall | **0.146** |
| Sensors used | **297** (filtered from 590) |
| Samples | **1,567 products** |
| Real failure rate | **6.6%** |

> SECOM is a notoriously challenging dataset where failures 
> are statistically similar to normal products. High precision 
> (91%) makes it practical for prioritizing quality inspections 
> without overwhelming teams with false alarms.

---

## 🎯 Business Context

This project mirrors real challenges in consumer electronics 
and semiconductor quality engineering:

- **Traditional QA** flags failures only after they occur
- **This model detects anomalous behavior in real time** 
  across 297 active sensor readings
- Enables quality teams to act **proactively** — scheduling 
  inspections before failures reach end users
- Directly applicable to **Sony-style product reliability 
  workflows** where sensor data is continuously monitored

---

## 🖥️ Interactive App

Real-time anomaly detection dashboard built with Streamlit.

**Features:**
- Adjust contamination threshold live with a slider — 
  model recalculates instantly
- PCA visualization comparing model predictions vs real labels
- Anomaly score distribution by product class
- Filterable table of all detected anomalies with scores

**Run locally:**
```bash
git clone https://github.com/lincereal/anomaly-detection-secom
cd anomaly-detection-secom
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

anomaly-detection-secom/
│
├── data/
│   ├── secom.data                  # Raw sensor readings (590)
│   ├── secom_labels.data           # Product labels (-1=OK, 1=FAIL)
│   ├── X_processed.csv             # Cleaned & normalized data
│   ├── y_labels.csv                # Labels aligned to processed data
│   └── predictions.csv             # Model predictions & scores
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb   # EDA, missing values, distributions
│   ├── 02_preprocessing.ipynb          # Cleaning, imputation, normalization
│   ├── 03_anomaly_detection.ipynb      # Isolation Forest training & PCA
│   └── 04_evaluation.ipynb             # Dashboard, PR curve, top sensors
│
├── src/
│   └── model/
│       ├── isolation_forest.pkl    # Trained Isolation Forest
│       ├── scaler.pkl              # StandardScaler
│       ├── pca_model.pkl           # PCA (297 → 2 dimensions)
│       └── feature_cols.pkl        # Active sensor columns
│
├── app.py                          # Streamlit interactive dashboard
├── requirements.txt
└── README.md

---

## 🔬 Methodology

### 1. Exploratory Analysis
- Identified severe class imbalance (6.6% failures)
- Mapped missing values across 590 sensors
- Compared sensor distributions between OK vs FAIL products

### 2. Preprocessing
- Removed sensors with >50% missing values (590 → 386)
- Imputed remaining missing values with median
- Removed zero-variance sensors (386 → 297)
- Applied StandardScaler normalization

### 3. Anomaly Detection
- Trained Isolation Forest with contamination=0.15
- Generated anomaly scores for all 1,567 products
- Reduced to 2D via PCA for visualization

### 4. Evaluation
- Precision-Recall curve analysis
- Confusion matrix vs real labels
- Identified top 15 sensors driving anomaly detection

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Scikit-learn** — Isolation Forest, PCA, preprocessing
- **Streamlit** — Interactive web dashboard
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualization
- **Plotly** — Interactive charts

---

## 🧠 Key Insights

**Why Isolation Forest?**
Unlike supervised models, Isolation Forest requires no 
labeled failure examples — it detects statistically rare 
points by measuring how easily they can be isolated. 
This is critical in manufacturing where failure examples 
are scarce.

**Why high precision matters here:**
In a real factory, a false alarm triggers a costly 
inspection line halt. A 91% precision rate means 9 out 
of 10 flagged products genuinely warrant inspection — 
making this model operationally viable.

**The Precision-Recall tradeoff:**
The contamination slider in the app lets users control 
this tradeoff in real time — higher contamination catches 
more failures but generates more false alarms.

---

## 🔗 Related Project

**[Predictive Failure Detection — NASA Turbofan](https://github.com/lincereal/predictive-failure-detection)**  
XGBoost model predicting Remaining Useful Life (RUL) 
of industrial components. R² = 0.80 | MAE = 13 cycles.

---

## 👤 Author

**Aldo Yamil Avila Carrillo**  
Quality Engineer → ML Engineer  
4+ years at Sony & LG Electronics  
Master's in Artificial Intelligence

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/TU_USUARIO)
[![GitHub](https://img.shields.io/badge/GitHub-lincereal-black)](https://github.com/lincereal)