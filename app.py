import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")


# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df[['Order Date', 'Ship Date']] = df[['Order Date', 'Ship Date']].apply(
        pd.to_datetime, dayfirst=True, errors='coerce'
    )
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    return df


df_store = load_data()

page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Segments"]
)

# ==================================================
# PAGE 1 — SALES OVERVIEW
# ==================================================
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    st.subheader("Total Sales by Year")
    yearly_sales = df_store.groupby('Year')['Sales'].sum()
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(yearly_sales.index.astype(str), yearly_sales.values, color='steelblue')
    ax1.set_ylabel("Sales")
    st.pyplot(fig1)

    st.subheader("Monthly Sales Trend")
    monthly_sales = df_store.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(monthly_sales.index, monthly_sales.values, color='green')
    ax2.set_ylabel("Sales")
    st.pyplot(fig2)

    st.subheader("Sales by Region and Category")
    region_filter = st.selectbox("Select Region", options=["All"] + list(df_store['Region'].unique()))
    category_filter = st.selectbox("Select Category", options=["All"] + list(df_store['Category'].unique()))

    filtered = df_store.copy()
    if region_filter != "All":
        filtered = filtered[filtered['Region'] == region_filter]
    if category_filter != "All":
        filtered = filtered[filtered['Category'] == category_filter]

    st.write(f"Total Sales: {filtered['Sales'].sum():.2f}")
    cat_sales = filtered.groupby('Category')['Sales'].sum()
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(cat_sales.index, cat_sales.values, color='orange')
    st.pyplot(fig3)


# ==================================================
# PAGE 2 — FORECAST EXPLORER (SARIMA — your best model)
# ==================================================
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Model used: SARIMA — selected as best performer (lowest MAE/RMSE/MAPE in your model comparison)")

    view_type = st.selectbox("View by", ["Category", "Region"])

    if view_type == "Category":
        options = list(df_store['Category'].unique())
    else:
        options = list(df_store['Region'].unique())

    selected_value = st.selectbox(f"Select {view_type}", options)
    horizon = st.slider("Forecast Horizon (months ahead)", 1, 3, 3)

    filtered = df_store[df_store[view_type] == selected_value]
    monthly = filtered.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum()

    train = monthly[:-3]
    test = monthly[-3:]

    model = SARIMAX(train, order=(2, 1, 0), seasonal_order=(1, 0, 0, 12)).fit(disp=False)
    forecast_test = model.get_forecast(steps=3).predicted_mean
    forecast_test.index = test.index

    mae = mean_absolute_error(test, forecast_test)
    rmse = mean_squared_error(test, forecast_test) ** 0.5

    final_model = SARIMAX(monthly, order=(2, 1, 0), seasonal_order=(1, 0, 0, 12)).fit(disp=False)
    future_forecast = final_model.get_forecast(steps=horizon).predicted_mean

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly.values, label='Actual', color='blue')
    ax.plot(future_forecast.index, future_forecast.values, label='Forecast', color='red', marker='o')
    ax.set_title(f"{horizon}-Month Forecast for {selected_value}")
    ax.legend()
    st.pyplot(fig)

    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")


# ==================================================
# PAGE 3 — ANOMALY REPORT
# ==================================================
elif page == "Anomaly Report":
    st.title("Anomaly Report")

    df_weekly = df_store.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum().reset_index()

    # Isolation Forest (global context)
    iso = IsolationForest(contamination=0.05, random_state=42)
    df_weekly['iso_anomaly'] = iso.fit_predict(df_weekly[['Sales']])
    df_weekly['iso_anomaly'] = df_weekly['iso_anomaly'].map({1: 'Normal', -1: 'Anomaly'})

    # Z-score on rolling 4-week window (local context), same as your notebook
    df_weekly['rolling_mean'] = df_weekly['Sales'].shift(1).rolling(window=4).mean()
    df_weekly['rolling_std'] = df_weekly['Sales'].shift(1).rolling(window=4).std()
    df_weekly['z_score'] = (df_weekly['Sales'] - df_weekly['rolling_mean']) / df_weekly['rolling_std']
    df_weekly['zscore_anomaly'] = df_weekly['z_score'].abs() > 2
    df_weekly['zscore_anomaly'] = df_weekly['zscore_anomaly'].map({True: 'Anomaly', False: 'Normal'})

    method_choice = st.radio("Detection Method", ["Isolation Forest", "Z-Score"])
    flag_col = 'iso_anomaly' if method_choice == "Isolation Forest" else 'zscore_anomaly'
    anomalies = df_weekly[df_weekly[flag_col] == 'Anomaly']

    st.subheader(f"Weekly Sales — {method_choice} Anomalies")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_weekly['Order Date'], df_weekly['Sales'], label='Weekly Sales', color='blue')
    ax.scatter(anomalies['Order Date'], anomalies['Sales'], color='red', label='Anomaly', s=60, zorder=5)
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Detected Anomaly Weeks")
    st.dataframe(anomalies[['Order Date', 'Sales']].reset_index(drop=True))
    st.write(f"**Total anomalies flagged:** {len(anomalies)} out of {len(df_weekly)} weeks")

    common = set(df_weekly[df_weekly['iso_anomaly'] == 'Anomaly']['Order Date']) & \
             set(df_weekly[df_weekly['zscore_anomaly'] == 'Anomaly']['Order Date'])
    st.write(f"**Weeks flagged by both methods:** {len(common)}")


# ==================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ==================================================
elif page == "Product Segments":
    st.title("Product Demand Segments")

    # Aggregate at Sub-Category level (same features as your notebook)
    seg = df_store.groupby('Sub-Category').agg(
        total_sales=('Sales', 'sum'),
        avg_order_value=('Sales', 'mean')
    ).reset_index()

    yearly_sub = df_store.groupby(['Sub-Category', 'Year'])['Sales'].sum().reset_index()

    def growth_rate(group):
        group = group.sort_values('Year')
        first = group['Sales'].iloc[0]
        last = group['Sales'].iloc[-1]
        return (last - first) / first

    growth = yearly_sub.groupby('Sub-Category').apply(growth_rate).reset_index()
    growth.columns = ['Sub-Category', 'growth_rate']
    seg = seg.merge(growth, on='Sub-Category')

    monthly_sub = df_store.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='ME')])['Sales'].sum().reset_index()
    volatility = monthly_sub.groupby('Sub-Category')['Sales'].std().reset_index()
    volatility.columns = ['Sub-Category', 'volatility']
    seg = seg.merge(volatility, on='Sub-Category')

    features = ['total_sales', 'avg_order_value', 'growth_rate', 'volatility']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(seg[features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    seg['cluster'] = kmeans.fit_predict(X_scaled)

    cluster_labels = {
        0: 'High Growth, Premium Niche',
        1: 'Low Volume, Stable Demand',
        2: 'High Volume, Volatile Demand'
    }
    seg['cluster_label'] = seg['cluster'].map(cluster_labels)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    seg['pca1'] = pca_result[:, 0]
    seg['pca2'] = pca_result[:, 1]

    st.subheader("Cluster Visualization (PCA)")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    for cluster_num, color in colors.items():
        cluster_data = seg[seg['cluster'] == cluster_num]
        ax.scatter(cluster_data['pca1'], cluster_data['pca2'],
                   label=cluster_labels[cluster_num], color=color, s=100)
    for _, row in seg.iterrows():
        ax.annotate(row['Sub-Category'], (row['pca1'], row['pca2']), fontsize=8, xytext=(5, 5),
                    textcoords='offset points')
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Sub-Category → Demand Cluster")
    st.dataframe(
        seg[['Sub-Category', 'total_sales', 'avg_order_value', 'growth_rate', 'volatility', 'cluster_label']]
        .sort_values('cluster_label')
        .reset_index(drop=True)
    )

    st.subheader("Recommended Stocking Strategy")
    st.markdown("""
    - **High Growth, Premium Niche** (e.g. Copiers): low volume, high value, fast-growing.
      Keep smaller stock levels but reorder frequently — each unit has high revenue impact.
    - **Low Volume, Stable Demand** (e.g. Accessories, Paper, Labels): predictable demand.
      Standard reorder points, moderate inventory, no close monitoring needed.
    - **High Volume, Volatile Demand** (e.g. Binders, Chairs, Phones): high sales but
      unpredictable swings and flat/declining growth. Needs safety stock buffers and
      closer inventory monitoring.
    """)