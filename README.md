# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq
from prophet import Prophet
import pmdarima as pm
import plotly.express as px
import io
from datetime import datetime, timedelta

st.set_page_config(page_title="Company Supply–Demand Analyzer", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Helper functions & cache
# -------------------------
@st.cache_data(show_spinner=False)
def fetch_stock_history(ticker: str, start="2015-01-01"):
    t = yf.Ticker(ticker)
    hist = t.history(start=start, auto_adjust=False)
    if hist is None or hist.empty:
        return None
    hist = hist.reset_index()
    return hist

@st.cache_data(show_spinner=False)
def fetch_financials(ticker: str):
    t = yf.Ticker(ticker)
    try:
        income = t.financials
        balance = t.balance_sheet
        cashflow = t.cashflow
    except Exception:
        income, balance, cashflow = None, None, None
    return {"income": income, "balance_sheet": balance, "cashflow": cashflow}

@st.cache_data(show_spinner=False)
def fetch_google_trends(keywords, timeframe='today 5-y', geo='US'):
    pytrends = TrendReq(hl='en-US', tz=0)
    try:
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop='')
        df = pytrends.interest_over_time()
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.drop(columns=['isPartial'], errors='ignore')
    # Aggregate keywords into a single demand index
    df['demand_index'] = df.sum(axis=1)
    df = df[['demand_index']]
    return df

def extract_inventory_timeseries(balance_sheet):
    if balance_sheet is None:
        return None
    try:
        # yfinance dataframes typically use row labels like 'Inventory'
        # balance_sheet columns are timestamps; transpose for easier access
        bs = balance_sheet.copy()
        # Try several possible labels in the index
        possible_keys = ['Inventory', 'Inventories', 'TotalInventory', 'TotalInventories']
        for key in possible_keys:
            if key in bs.index:
                inv = bs.loc[key].T.reset_index()
                inv.columns = ['date', 'inventory']
                inv['date'] = pd.to_datetime(inv['date'])
                inv = inv.sort_values('date')
                return inv
        # fallback: try to find any numeric row that likely matches inventory by heuristics (small values relative)
    except Exception:
        return None
    return None

def build_demand_proxy(revenue_df, trends_df):
    # revenue_df expected columns: ['date','revenue'] monthly or quarterly
    # trends_df index is datetime index with column 'demand_index' (daily weekly granularity)
    if revenue_df is None and trends_df is None:
        return None
    # normalize sources and combine
    pieces = []
    if revenue_df is not None:
        rev = revenue_df.copy()
        rev['date'] = pd.to_datetime(rev['date'])
        rev_month = rev.set_index('date').resample('M').sum().fillna(method='ffill')
        rev_month = rev_month.rename(columns={'revenue':'revenue_value'})
        rev_month['revenue_norm'] = rev_month['revenue_value'] / (rev_month['revenue_value'].max() or 1)
        pieces.append(rev_month[['revenue_norm']])
    if trends_df is not None:
        tr = trends_df.copy()
        tr_month = tr.resample('M').mean()
        tr_month['tr_norm'] = tr_month['demand_index'] / (tr_month['demand_index'].max() or 1)
        pieces.append(tr_month[['tr_norm']])
    # merge
    if not pieces:
        return None
    merged = pd.concat(pieces, axis=1).fillna(method='ffill').fillna(0)
    # Weighted combination: revenue (if exists) heavier weight (0.6) else trend only
    weights = []
    if 'revenue_norm' in merged.columns and 'tr_norm' in merged.columns:
        merged['demand_proxy'] = 0.6 * merged['revenue_norm'] + 0.4 * merged['tr_norm']
    elif 'revenue_norm' in merged.columns:
        merged['demand_proxy'] = merged['revenue_norm']
    else:
        merged['demand_proxy'] = merged['tr_norm']
    demand_df = merged[['demand_proxy']].reset_index().rename(columns={'index':'date'})
    return demand_df

def prophet_forecast_df(df_ts, periods=12, freq='M'):
    df = df_ts.rename(columns={'date':'ds','y':'y'}) if 'date' in df_ts.columns else df_ts.copy()
    # Standardize expected columns for prophet: ds, y
    if 'ds' not in df.columns:
        if 'date' in df.columns:
            df = df.rename(columns={'date':'ds'})
        else:
            raise ValueError("No date column found for Prophet input")
    if 'y' not in df.columns:
        # try to find a numeric column to use as y
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df['y'] = df[numeric_cols[0]]
        else:
            raise ValueError("No numeric column found for Prophet input")
    df = df.dropna(subset=['ds','y']).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Too little data for forecasting")
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return forecast, m

def arima_forecast_series(series, periods=12):
    # series is a pandas Series with DatetimeIndex
    try:
        model = pm.auto_arima(series, seasonal=True, m=12, error_action='ignore', suppress_warnings=True)
        fc, conf = model.predict(n_periods=periods, return_conf_int=True)
        idx_start = series.index[-1] + pd.offsets.MonthEnd(1)
        idx = pd.date_range(start=idx_start, periods=periods, freq='M')
        fc_series = pd.Series(fc, index=idx)
        conf_df = pd.DataFrame(conf, index=idx, columns=['yhat_lower','yhat_upper'])
        return fc_series, conf_df
    except Exception as e:
        return None, None

def safe_df_to_csv_bytes(df: pd.DataFrame):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

# -------------------------
# Streamlit UI
# -------------------------
st.title("📊 Company Supply–Demand Analyzer")
st.markdown(
    "Analyze **supply** (inventory proxies) and **demand** (revenue & Google Trends) for *any public company* by ticker. "
    "Upload SKU sales for deeper SKU-level forecasts."
)

with st.sidebar:
    st.header("Analyze company")
    ticker = st.text_input("Ticker (e.g. WMT, AAPL, AMZN)", value="WMT")
    start_date = st.date_input("Historical start date", value=datetime.today() - timedelta(days=365*5))
    forecast_months = st.slider("Forecast horizon (months)", min_value=3, max_value=36, value=12)
    timeframe_trends = st.selectbox("Google Trends timeframe", options=['today 5-y','today 12-m','today 3-m','today 90-d','today 7-d'], index=0)
    geo = st.text_input("Trends geo (country code)", value="US")
    st.markdown("---")
    st.header("SKU-level (optional)")
    sku_upload = st.file_uploader("Upload CSV with columns: date,sku,sales", type=['csv'])
    public_only = st.checkbox("Public-data mode (no SKU processing)", value=False)
    analyze_btn = st.button("Analyze")

# Placeholders for outputs
main_col, right_col = st.columns([3,1])

if analyze_btn and ticker:
    with st.spinner(f"Fetching data for {ticker}..."):
        # 1) stock history
        stock_hist = fetch_stock_history(ticker, start=start_date.strftime("%Y-%m-%d"))
        fin = fetch_financials(ticker)

        # attempt to extract revenue from yfinance income statement
        revenue_df = None
        try:
            inc = fin.get('income')
            if inc is not None and not inc.empty:
                # yfinance returns a DataFrame where index are financial lines and columns are dates
                inc_t = inc.T.reset_index().rename(columns={'index':'date'})
                # try to pick a revenue-like row
                revenue_col = None
                for label in ['TotalRevenue','Total Revenue','Revenues','TotalRevenues','Revenue']:
                    if label in inc_t.columns:
                        revenue_col = label
                        break
                if revenue_col is None:
                    # If no standard label, heuristically pick the numeric column with largest values
                    numeric_cols = inc_t.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        revenue_col = numeric_cols[0]
                if revenue_col:
                    revenue_df = inc_t[['date', revenue_col]].rename(columns={revenue_col:'revenue'})
                    revenue_df['date'] = pd.to_datetime(revenue_df['date'])
                    # Some yfinance revenue values are in raw dollars per quarter; keep as-is
        except Exception:
            revenue_df = None

        # 2) inventory timeseries
        inv_ts = extract_inventory_timeseries(fin.get('balance_sheet'))

        # 3) Google Trends
        try:
            trends_df = fetch_google_trends([ticker, f"{ticker} stock", f"{ticker} company"], timeframe=timeframe_trends, geo=geo)
        except Exception:
            trends_df = None

    # Layout: show key cards
    with main_col:
        st.subheader(f"{ticker.upper()} overview")
        cols = st.columns(3)
        if stock_hist is not None and not stock_hist.empty:
            latest = stock_hist.iloc[-1]
            price = latest['Close']
            change = latest['Close'] - stock_hist.iloc[-2]['Close'] if len(stock_hist) > 1 else 0.0
            cols[0].metric("Latest close", f"${price:,.2f}", f"{change:,.2f}")
        else:
            cols[0].write("No stock history")

        if revenue_df is not None and not revenue_df.empty:
            latest_rev = revenue_df.sort_values('date').iloc[-1]['revenue']
            cols[1].metric("Most recent revenue (reported)", f"${latest_rev:,.0f}")
        else:
            cols[1].write("Revenue not available")

        if inv_ts is not None and not inv_ts.empty:
            latest_inv = inv_ts.sort_values('date').iloc[-1]['inventory']
            cols[2].metric("Most recent inventory (balance sheet)", f"${latest_inv:,.0f}")
        else:
            cols[2].write("Inventory not available")

        st.markdown("---")

        # Plots: stock history
        if stock_hist is not None and not stock_hist.empty:
            fig_px = px.line(stock_hist, x='Date', y='Close', title=f"{ticker.upper()} Stock Price")
            st.plotly_chart(fig_px, use_container_width=True)
        else:
            st.info("No stock history to plot.")

        # Demand proxy build
        demand_df = build_demand_proxy(revenue_df, trends_df)
        if demand_df is not None:
            st.subheader("Demand proxy (revenue + Google Trends)")
            fig_d = px.line(demand_df, x='date', y='demand_proxy', title="Demand proxy (normalized)", labels={'demand_proxy':'demand proxy'})
            st.plotly_chart(fig_d, use_container_width=True)
            # Forecast demand
            st.markdown("**Demand forecast**")
            try:
                dfp = demand_df.rename(columns={'date':'ds','demand_proxy':'y'})[['ds','y']].dropna()
                if len(dfp) >= 24:
                    fc, m = prophet_forecast_df(dfp, periods=forecast_months, freq='M')
                    fc_short = fc[['ds','yhat','yhat_lower','yhat_upper']].tail(forecast_months)
                    fig_fc = px.line(fc, x='ds', y='yhat', title="Demand forecast (Prophet)")
                    fig_fc.add_scatter(x=fc['ds'], y=fc['yhat_lower'], mode='lines', opacity=0.2, name='lower')
                    fig_fc.add_scatter(x=fc['ds'], y=fc['yhat_upper'], mode='lines', opacity=0.2, name='upper')
                    st.plotly_chart(fig_fc, use_container_width=True)
                    # Provide download
                    st.download_button("Download demand forecast CSV", data=safe_df_to_csv_bytes(fc_short), file_name=f"{ticker}_demand_forecast.csv")
                    demand_forecast_df = fc_short
                else:
                    # fallback ARIMA
                    series = pd.Series(dfp['y'].values, index=pd.DatetimeIndex(dfp['ds']))
                    series.index.freq = 'M'
                    fc_series, conf_df = arima_forecast_series(series, periods=forecast_months)
                    if fc_series is not None:
                        dfc = pd.DataFrame({'ds':fc_series.index, 'yhat': fc_series.values})
                        dfc['yhat_lower'] = conf_df['yhat_lower'].values
                        dfc['yhat_upper'] = conf_df['yhat_upper'].values
                        st.line_chart(dfc.set_index('ds')['yhat'])
                        st.download_button("Download demand forecast CSV", data=safe_df_to_csv_bytes(dfc), file_name=f"{ticker}_demand_forecast_arima.csv")
                        demand_forecast_df = dfc
                    else:
                        st.warning("Demand forecasting failed.")
            except Exception as e:
                st.error(f"Demand forecasting error: {e}")
        else:
            st.info("Not enough public signals to build a demand proxy. Provide SKU sales for deeper forecasting.")

        # Inventory forecast & supply-demand ratio
        if inv_ts is not None and not inv_ts.empty and (('demand_forecast_df' in locals() or 'demand_forecast_df' in globals())):
            st.subheader("Inventory (supply) and supply–demand gap")
            inv = inv_ts.copy()
            inv = inv.rename(columns={'date':'ds','inventory':'y'})
            try:
                if len(inv) >= 12:
                    inv_fc, _ = prophet_forecast_df(inv[['ds','y']], periods=forecast_months, freq='M')
                    inv_fc_short = inv_fc[['ds','yhat']].tail(forecast_months).rename(columns={'yhat':'inventory_forecast'})
                    # Merge demand forecast
                    df_dem = demand_forecast_df.copy()
                    df_dem['ds'] = pd.to_datetime(df_dem['ds'])
                    df_dem = df_dem.rename(columns={'ds':'ds','yhat':'yhat'}) if 'yhat' in df_dem.columns else df_dem.rename(columns={'ds':'ds','y':'yhat'})
                    merged = pd.merge(df_dem, inv_fc_short, how='left', left_on='ds', right_on='ds')
                    merged['supply_demand_ratio'] = merged['inventory_forecast'] / merged['yhat'].replace({0:np.nan})
                    # show chart of ratio
                    fig_ratio = px.line(merged, x='ds', y='supply_demand_ratio', title='Forecasted Supply/Demand Ratio (inventory / demand proxy)')
                    st.plotly_chart(fig_ratio, use_container_width=True)
                    st.download_button("Download supply-demand CSV", data=safe_df_to_csv_bytes(merged), file_name=f"{ticker}_supply_demand_gap.csv")
                else:
                    st.info("Not enough inventory history to do reliable inventory forecasting.")
            except Exception as e:
                st.error(f"Inventory forecast error: {e}")

        # SKU-level forecasting (optional)
        if (not public_only) and (sku_upload is not None):
            st.subheader("SKU-level forecasts (uploaded CSV)")
            try:
                sku_df = pd.read_csv(sku_upload, parse_dates=['date'])
                # Expect columns: date, sku, sales
                required_cols = {'date','sku','sales'}
                if not required_cols.issubset(set(sku_df.columns)):
                    st.error("SKU CSV must have columns: date,sku,sales")
                else:
                    st.info("Running SKU-level forecasts (Prophet preferred; ARIMA fallback). This can take time for many SKUs.")
                    sku_out = []
                    for sku, g in sku_df.groupby('sku'):
                        ts = g.set_index('date').resample('M')['sales'].sum().fillna(0)
                        if len(ts.dropna()) < 12:
                            continue
                        dfp = ts.reset_index().rename(columns={'date':'ds','sales':'y'})
                        try:
                            fc, _ = prophet_forecast_df(dfp, periods=forecast_months, freq='M')
                            out = fc[['ds','yhat']].tail(forecast_months).copy()
                            out['sku'] = sku
                            sku_out.append(out)
                        except Exception:
                            fc_ser, conf = arima_forecast_series(ts, periods=forecast_months)
                            if fc_ser is not None:
                                out = pd.DataFrame({'ds':fc_ser.index,'yhat':fc_ser.values})
                                out['sku'] = sku
                                sku_out.append(out)
                    if sku_out:
                        sku_out_df = pd.concat(sku_out, ignore_index=True)
                        st.dataframe(sku_out_df.head(200))
                        st.download_button("Download SKU forecasts CSV", data=safe_df_to_csv_bytes(sku_out_df), file_name=f"{ticker}_sku_forecasts.csv")
                    else:
                        st.info("No SKU forecasts created (insufficient SKU history).")
            except Exception as e:
                st.error(f"SKU processing error: {e}")

    with right_col:
        st.subheader("Quick data & tips")
        st.markdown(
            "- **Tip:** If inventory isn't available via yfinance, upload the company 10-K XBRL or parse the filings for better inventory history.\n"
            "- **Tip:** Supply–demand accuracy improves massively with SKU-level sales and supplier lead-time data.\n"
            "- **CSV downloads** are available for demand forecasts and supply–demand gap.\n"
        )
        st.markdown("### Raw data")
        if stock_hist is not None:
            if st.button("Show raw stock history"):
                st.dataframe(stock_hist.tail(200))
            st.download_button("Download stock history CSV", data=safe_df_to_csv_bytes(stock_hist), file_name=f"{ticker}_stock_history.csv")
        if revenue_df is not None:
            st.download_button("Download extracted revenue CSV", data=safe_df_to_csv_bytes(revenue_df), file_name=f"{ticker}_revenue.csv")
        if inv_ts is not None:
            st.download_button("Download inventory timeseries", data=safe_df_to_csv_bytes(inv_ts), file_name=f"{ticker}_inventory.csv")
        if trends_df is not None:
            tr_dl = trends_df.reset_index().rename(columns={'index':'date'})
            st.download_button("Download Google Trends CSV", data=safe_df_to_csv_bytes(tr_dl), file_name=f"{ticker}_trends.csv")

else:
    st.info("Enter a ticker and press Analyze. Upload a SKU CSV if you want SKU-level forecasts.")

st.markdown("---")
st.caption("Built with Streamlit • Uses public signals (yfinance, Google Trends) as proxies. Forecasts use Prophet/ARIMA. Interpret results as directional guidance — for operational ordering you need SKU on-hand, lead times, and supplier reliability data.")
