# End-to-End Sales Forecasting & Demand Intelligence System

Internship Project — Week 3 & Week 4

## What This Project Does

This project builds a sales forecasting and demand intelligence system for a retail
company, using four years of Superstore sales data. It covers exploratory data
analysis, time series decomposition, forecasting with three different models,
anomaly detection, product demand segmentation, and a deployed interactive
dashboard.

## Folder Contents

| File | Description |
|---|---|
| `analysis.ipynb` | Main Jupyter Notebook with all 8 tasks — EDA, decomposition, forecasting (SARIMA, Prophet, XGBoost), anomaly detection, clustering, and notes/explanations throughout |
| `train.csv` | Superstore sales dataset (Order Date, Ship Date, Region, Category, Sub-Category, Sales, etc.) |
| `vgsales.csv` | Supplementary video game sales dataset, used to test the anomaly detection method on a second, unrelated dataset |
| `app.py` | Streamlit dashboard — Sales Overview, Forecast Explorer, Anomaly Report, and Product Demand Segments pages |
| `requirements.txt` | Python libraries needed to run the notebook and the dashboard |
| `summary.pdf` | 2-page executive business report, written for a non-technical audience (Head of Supply Chain / CFO) |
| `charts/` | Saved PNG images of the key charts from the notebook |

## How to Run the Notebook

1. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
2. Open `analysis.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run all cells from top to bottom. `train.csv` and `vgsales.csv` should be in
   the same folder as the notebook.

## How to Run the Dashboard Locally

1. Make sure `app.py` and `train.csv` are in the same folder.
2. Open a terminal (Anaconda Prompt, or your system terminal — not a Jupyter
   cell) in that folder.
3. Run:
   ```
   streamlit run app.py
   ```
4. This opens the dashboard in your browser at `http://localhost:8501`.

## Key Results

- **Best forecasting model:** SARIMA, with the lowest error of the three models
  tested (about 15% average error, vs. 22% for Prophet and 32% for XGBoost).
- **Seasonality:** March, September, and November consistently show higher sales
  every year.
- **Anomaly detection:** Two methods were used — Isolation Forest (catches the
  most extreme spikes/drops overall) and a rolling Z-score (catches sudden local
  changes). Both agree on the biggest spikes, like March 2015 and December 2018.
- **Product segmentation:** Products were grouped into three demand clusters —
  High Growth/Premium Niche, Low Volume/Stable Demand, and High Volume/Volatile
  Demand — each with a different recommended stocking strategy.

## Notes & Limitations

- The forecasting models are trained on historical patterns, so they assume the
  future will look broadly similar to the past four years. Sudden market shifts
  (new competitors, supply disruptions, etc.) won't be reflected until they show
  up in new sales data.
- `requirements.txt` includes Prophet, XGBoost, pmdarima, and seaborn for the
  notebook — the deployed dashboard (`app.py`) only actually needs pandas, numpy,
  matplotlib, statsmodels, scikit-learn, and streamlit.


