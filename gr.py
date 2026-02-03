# This cell writes a deployable Streamlit app + requirements + Procfile based on the notebook contents
app_code = """import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# xgboost is optional in some deploy targets; we'll show a helpful error if missing
try:
    import xgboost as xgb
except Exception:
    xgb = None

st.set_page_config(page_title='Stock Forecasting Dashboard', layout='wide')

st.title('Apple Stock Forecasting - ARIMA, SARIMAX, XGBoost')
st.write('Upload a CSV with at least Date and Close columns. The app will clean the data and run forecasts.')

@st.cache_data
def load_and_clean_data_from_file(uploaded_file):
    df_local = pd.read_csv(uploaded_file)
    df_local.columns = [c.strip() for c in df_local.columns]
    if 'Date' not in df_local.columns:
        raise ValueError('Missing Date column')
    if 'Close' not in df_local.columns:
        raise ValueError('Missing Close column')

    df_local['Date'] = pd.to_datetime(df_local['Date'], errors='coerce')
    df_local = df_local.dropna(subset=['Date', 'Close']).copy()
    df_local['Close'] = pd.to_numeric(df_local['Close'], errors='coerce')
    df_local = df_local.dropna(subset=['Close']).copy()

    df_local = df_local.sort_values('Date')
    df_local = df_local.set_index('Date')
    df_local = df_local[~df_local.index.duplicated(keep='first')]
    return df_local

@st.cache_data
def create_ml_features(df_local, lags=10):
    feat_df = df_local.copy()
    for lag in range(1, lags + 1):
        feat_df['lag_' + str(lag)] = feat_df['Close'].shift(lag)
    feat_df['rolling_mean_7'] = feat_df['Close'].shift(1).rolling(7).mean()
    feat_df['rolling_std_7'] = feat_df['Close'].shift(1).rolling(7).std()
    feat_df = feat_df.dropna().copy()
    return feat_df

def plot_history(df_local):
    fig = plt.figure(figsize=(12, 4))
    plt.plot(df_local.index, df_local['Close'], color='black', linewidth=1)
    plt.title('Close Price History')
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.grid(True, alpha=0.2)
    st.pyplot(fig)


def fit_arima_forecast(train_series, steps, order=(5, 1, 0)):
    model = ARIMA(train_series, order=order)
    fit_res = model.fit()
    fc = fit_res.forecast(steps=steps)
    return fc


def fit_sarimax_forecast(train_series, steps, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    fit_res = model.fit(disp=False)
    fc = fit_res.forecast(steps=steps)
    return fc


def xgb_recursive_forecast(df_local, steps, lags=10):
    if xgb is None:
        raise RuntimeError('xgboost is not installed in this environment')

    feat_df = create_ml_features(df_local, lags=lags)
    X = feat_df[[c for c in feat_df.columns if c.startswith('lag_') or c.startswith('rolling_')]]
    y = feat_df['Close']

    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    xgb_model.fit(X, y)

    # Recursive forecast by appending predictions
    temp_df = df_local.copy()
    preds = []
    for _ in range(steps):
        feat_now = create_ml_features(temp_df, lags=lags)
        X_last = feat_now[[c for c in feat_now.columns if c.startswith('lag_') or c.startswith('rolling_')]].tail(1)
        next_pred = float(xgb_model.predict(X_last)[0])
        next_date = temp_df.index.max() + pd.Timedelta(days=1)
        temp_df.loc[next_date, 'Close'] = next_pred
        preds.append(next_pred)

    future_index = pd.date_range(start=df_local.index.max() + pd.Timedelta(days=1), periods=steps, freq='D')
    return pd.Series(preds, index=future_index)


uploaded_file = st.file_uploader('Upload CSV', type=['csv'])

with st.sidebar:
    st.header('Settings')
    forecast_horizon = st.slider('Forecast horizon (days)', min_value=7, max_value=90, value=30, step=1)
    train_split = st.slider('Train split', min_value=0.6, max_value=0.95, value=0.8, step=0.05)

    st.subheader('ARIMA')
    arima_p = st.number_input('p', min_value=0, max_value=10, value=5, step=1)
    arima_d = st.number_input('d', min_value=0, max_value=2, value=1, step=1)
    arima_q = st.number_input('q', min_value=0, max_value=10, value=0, step=1)

    st.subheader('SARIMAX')
    sar_p = st.number_input('SAR p', min_value=0, max_value=5, value=1, step=1)
    sar_d = st.number_input('SAR d', min_value=0, max_value=2, value=1, step=1)
    sar_q = st.number_input('SAR q', min_value=0, max_value=5, value=1, step=1)
    seas_p = st.number_input('Seasonal P', min_value=0, max_value=3, value=1, step=1)
    seas_d = st.number_input('Seasonal D', min_value=0, max_value=2, value=1, step=1)
    seas_q = st.number_input('Seasonal Q', min_value=0, max_value=3, value=1, step=1)
    seas_m = st.number_input('Seasonal period m', min_value=2, max_value=365, value=12, step=1)

    st.subheader('XGBoost')
    use_xgb = st.checkbox('Run XGBoost forecast', value=True)
    xgb_lags = st.slider('XGBoost lags', min_value=3, max_value=30, value=10, step=1)


if uploaded_file is None:
    st.info('Upload a CSV file to begin. Expected columns: Date, Close')
    st.stop()

try:
    df = load_and_clean_data_from_file(uploaded_file)
except Exception as e:
    st.error('Failed to load data: ' + str(e))
    st.stop()

st.subheader('Data preview')
st.dataframe(df.tail(20), use_container_width=True)

st.subheader('Close price history')
plot_history(df)

# Train/test split for evaluation on the last chunk
n_total = len(df)
n_train = int(n_total * float(train_split))
train_series = df['Close'].iloc[:n_train]
test_series = df['Close'].iloc[n_train:]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('#### ARIMA forecast')
    try:
        arima_fc = fit_arima_forecast(train_series, steps=len(test_series), order=(int(arima_p), int(arima_d), int(arima_q)))
        arima_mse = float(np.mean((arima_fc.values - test_series.values) ** 2))
        st.write('Test MSE: ' + str(round(arima_mse, 4)))
    except Exception as e:
        arima_fc = None
        st.write('ARIMA failed: ' + str(e))

with col2:
    st.markdown('#### SARIMAX forecast')
    try:
        sarimax_fc = fit_sarimax_forecast(
            train_series,
            steps=len(test_series),
            order=(int(sar_p), int(sar_d), int(sar_q)),
            seasonal_order=(int(seas_p), int(seas_d), int(seas_q), int(seas_m))
        )
        sarimax_mse = float(np.mean((sarimax_fc.values - test_series.values) ** 2))
        st.write('Test MSE: ' + str(round(sarimax_mse, 4)))
    except Exception as e:
        sarimax_fc = None
        st.write('SARIMAX failed: ' + str(e))

with col3:
    st.markdown('#### XGBoost forecast')
    if use_xgb:
        try:
            xgb_fc_test = xgb_recursive_forecast(df.iloc[:n_train][['Close']], steps=len(test_series), lags=int(xgb_lags))
            xgb_mse = float(np.mean((xgb_fc_test.values - test_series.values) ** 2))
            st.write('Test MSE: ' + str(round(xgb_mse, 4)))
        except Exception as e:
            xgb_fc_test = None
            st.write('XGBoost failed: ' + str(e))
    else:
        xgb_fc_test = None
        st.write('Skipped')

# Plot backtest
st.subheader('Backtest plot')
fig_bt = plt.figure(figsize=(12, 4))
plt.plot(df.index, df['Close'], label='Actual', color='black', linewidth=1)
if arima_fc is not None:
    plt.plot(test_series.index, arima_fc.values, label='ARIMA', alpha=0.8)
if sarimax_fc is not None:
    plt.plot(test_series.index, sarimax_fc.values, label='SARIMAX', alpha=0.8)
if xgb_fc_test is not None:
    plt.plot(test_series.index, xgb_fc_test.values, label='XGBoost', alpha=0.8)
plt.legend()
plt.grid(True, alpha=0.2)
st.pyplot(fig_bt)

# Future forecast
st.subheader('Future forecast')
last_close = float(df['Close'].iloc[-1])
st.write('Last close: ' + str(last_close))

future_dates = pd.date_range(start=df.index.max() + pd.Timedelta(days=1), periods=int(forecast_horizon), freq='D')

future_fig = plt.figure(figsize=(12, 4))
plt.plot(df.index[-180:], df['Close'].tail(180), label='History (last 180 days)', color='black', alpha=0.7)

try:
    arima_future = fit_arima_forecast(df['Close'], steps=int(forecast_horizon), order=(int(arima_p), int(arima_d), int(arima_q)))
    plt.plot(future_dates, arima_future.values, label='ARIMA future')
except Exception:
    pass

try:
    sarimax_future = fit_sarimax_forecast(
        df['Close'],
        steps=int(forecast_horizon),
        order=(int(sar_p), int(sar_d), int(sar_q)),
        seasonal_order=(int(seas_p), int(seas_d), int(seas_q), int(seas_m))
    )
    plt.plot(future_dates, sarimax_future.values, label='SARIMAX future')
except Exception:
    pass

if use_xgb:
    try:
        xgb_future = xgb_recursive_forecast(df[['Close']], steps=int(forecast_horizon), lags=int(xgb_lags))
        plt.plot(xgb_future.index, xgb_future.values, label='XGBoost future')
    except Exception:
        pass

plt.legend()
plt.grid(True, alpha=0.2)
st.pyplot(future_fig)
"""

reqs = """streamlit==1.36.0
pandas>=2.0
numpy>=1.23
matplotlib>=3.7
seaborn>=0.12
statsmodels>=0.14
scikit-learn>=1.3
xgboost>=1.7
"""

procfile = """web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
"""

dockerfile = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)
with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write(reqs)
with open('Procfile', 'w', encoding='utf-8') as f:
    f.write(procfile)
with open('Dockerfile', 'w', encoding='utf-8') as f:
    f.write(dockerfile)

print('app.py')
print('requirements.txt')
print('Procfile')
print('Dockerfile')