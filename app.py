"""
🇵🇰 PSX Stock Market Prediction App
Streamlit web application for Pakistan Stock Exchange analysis & prediction.
University Project - Stock Price Prediction using Linear Regression
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from predictor import get_stock_list, predict_stock_prices, PSX_STOCKS, get_sectors

# ─── Page Config ───
st.set_page_config(
    page_title="PSX Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS for Premium Dark Look ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
div[data-testid="stMetric"] {
    background: rgba(17,24,39,0.7);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
}
div[data-testid="stMetric"] label { font-size: 12px !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 800 !important; }
.signal-box {
    padding: 12px 24px; border-radius: 12px; text-align: center;
    font-size: 20px; font-weight: 800; margin: 8px 0;
}
.green-bg { background: rgba(0,230,118,0.15); border: 1px solid rgba(0,230,118,0.3); color: #00e676; }
.red-bg { background: rgba(255,23,68,0.15); border: 1px solid rgba(255,23,68,0.3); color: #ff1744; }
.yellow-bg { background: rgba(255,215,64,0.15); border: 1px solid rgba(255,215,64,0.3); color: #ffd740; }
.info-card {
    background: rgba(17,24,39,0.7); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 20px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


def create_price_chart(df, predicted_dates, predicted_prices, ticker_name):
    """Create main price chart with predictions."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.75, 0.25],
        subplot_titles=("", "Volume")
    )
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Price",
        increasing_line_color='#00e676', decreasing_line_color='#ff1744'
    ), row=1, col=1)
    # SMA lines
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name="SMA 20",
            line=dict(color='#448aff', width=1.5)), row=1, col=1)
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name="SMA 50",
            line=dict(color='#ffd740', width=1.5)), row=1, col=1)
    # Prediction line
    pred_df = pd.DataFrame({'Date': predicted_dates, 'Price': predicted_prices})
    last_hist = pd.DataFrame({'Date': [df['Date'].iloc[-1]], 'Price': [float(df['Close'].iloc[-1])]})
    pred_df = pd.concat([last_hist, pred_df], ignore_index=True)
    fig.add_trace(go.Scatter(
        x=pred_df['Date'], y=pred_df['Price'], name="🔮 Prediction",
        line=dict(color='#b388ff', width=3, dash='dot'),
        mode='lines+markers', marker=dict(size=4)
    ), row=1, col=1)
    # Volume
    colors = ['#00e676' if c >= o else '#ff1744' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name="Volume",
        marker_color=colors, opacity=0.5), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", height=600,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False,
        font=dict(family="Inter"), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    return fig


def create_rsi_chart(df):
    """Create RSI indicator chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI",
        line=dict(color='#b388ff', width=2), fill='tozeroy',
        fillcolor='rgba(179,136,255,0.1)'))
    fig.add_hline(y=70, line_dash="dash", line_color="#ff1744", opacity=0.5,
        annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="#00e676", opacity=0.5,
        annotation_text="Oversold (30)")
    fig.update_layout(
        template="plotly_dark", height=250,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        yaxis=dict(range=[0, 100])
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    return fig


def create_macd_chart(df):
    """Create MACD indicator chart."""
    fig = go.Figure()
    colors = ['#00e676' if v >= 0 else '#ff1744' for v in df['MACD_Histogram']]
    fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Histogram'], name="Histogram",
        marker_color=colors, opacity=0.5))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name="MACD",
        line=dict(color='#448aff', width=2)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name="Signal",
        line=dict(color='#ff6e40', width=2)))
    fig.update_layout(
        template="plotly_dark", height=250,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    return fig


def create_bollinger_chart(df):
    """Create Bollinger Bands chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name="Upper Band",
        line=dict(color='rgba(255,23,68,0.4)', width=1)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name="Lower Band",
        line=dict(color='rgba(0,230,118,0.4)', width=1),
        fill='tonexty', fillcolor='rgba(68,138,255,0.05)'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close",
        line=dict(color='#448aff', width=2)))
    fig.update_layout(
        template="plotly_dark", height=300,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    return fig


def create_accuracy_chart(y_test, y_pred):
    """Create model accuracy comparison chart."""
    fig = go.Figure()
    x_axis = list(range(len(y_test)))
    fig.add_trace(go.Scatter(x=x_axis, y=y_test, name="Actual Price",
        line=dict(color='#00e676', width=2)))
    fig.add_trace(go.Scatter(x=x_axis, y=y_pred, name="Predicted Price",
        line=dict(color='#ff6e40', width=2, dash='dot')))
    fig.update_layout(
        template="plotly_dark", height=300,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Test Data Points", yaxis_title="Price (PKR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    return fig


# ═══════════════════════════════════════════
#              SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🇵🇰 PSX Predictor")
    st.markdown("---")

    # Stock selection
    stocks = get_stock_list()
    stock_options = {s["display"]: s["ticker"] for s in stocks}
    selected_display = st.selectbox("📊 Select Stock", list(stock_options.keys()))
    selected_ticker = stock_options[selected_display]

    st.markdown("---")

    # Parameters
    period = st.selectbox("📅 Historical Period", ["6mo", "1y", "2y", "5y"], index=2)
    pred_days = st.slider("🔮 Prediction Days", 7, 90, 30)

    st.markdown("---")

    # Filter by sector
    st.markdown("### 🏭 Sectors")
    for sector in get_sectors():
        count = sum(1 for s in PSX_STOCKS.values() if s["sector"] == sector)
        st.markdown(f"- **{sector}** ({count} stocks)")

    st.markdown("---")
    predict_btn = st.button("🚀 Run Prediction", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("⚠️ Disclaimer: This is a university project. Predictions are for educational purposes only. Not financial advice.")


# ═══════════════════════════════════════════
#              MAIN CONTENT
# ═══════════════════════════════════════════
st.markdown("# 📈 Pakistan Stock Exchange Predictor")
st.markdown("##### Linear Regression based stock price prediction & technical analysis")
st.markdown("---")

if predict_btn:
    with st.spinner(f"⏳ Fetching data & analyzing {selected_display}..."):
        result = predict_stock_prices(selected_ticker, prediction_days=pred_days, period=period)

    if "error" in result:
        st.error(f"❌ {result['error']}")
    else:
        # ── Store result in session ──
        st.session_state['result'] = result

# ── Show results if available ──
if 'result' in st.session_state:
    r = st.session_state['result']
    df = r['df_analysis']

    # ── Header Info ──
    st.markdown(f"## {r['name']}  `{r['ticker']}`")
    st.markdown(f"**Sector:** {r['sector']}")

    # ── Signal Badge ──
    sig = r['signal']
    if "Buy" in sig:
        css_class = "green-bg"
    elif "Sell" in sig:
        css_class = "red-bg"
    else:
        css_class = "yellow-bg"
    st.markdown(f'<div class="signal-box {css_class}">{sig}</div>', unsafe_allow_html=True)

    # ── Key Metrics Row ──
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Current Price", f"PKR {r['current_price']:,.2f}",
              f"{r['day_change']:+.2f} ({r['day_change_pct']:+.2f}%)")
    c2.metric("Predicted Price", f"PKR {r['predicted_price']:,.2f}",
              f"{r['price_change_pct']:+.2f}%")
    c3.metric("52W High", f"PKR {r['high_52w']:,.2f}")
    c4.metric("52W Low", f"PKR {r['low_52w']:,.2f}")
    c5.metric("RSI (14)", f"{r['rsi']}" if r['rsi'] else "N/A")
    c6.metric("Avg Volume", f"{r['avg_volume']:,}")

    st.markdown("---")

    # ═══ PRICE CHART WITH PREDICTION ═══
    st.markdown("### 📊 Price Chart & Prediction")
    fig_price = create_price_chart(df, r['predicted_dates'], r['predicted_prices'], r['name'])
    st.plotly_chart(fig_price, use_container_width=True)

    # ═══ TECHNICAL ANALYSIS ═══
    st.markdown("### 📐 Technical Analysis")
    tab1, tab2, tab3 = st.tabs(["📉 RSI", "📊 MACD", "📈 Bollinger Bands"])

    with tab1:
        st.plotly_chart(create_rsi_chart(df), use_container_width=True)
        rsi_val = r['rsi']
        if rsi_val:
            if rsi_val > 70:
                st.warning(f"RSI = {rsi_val} → **Overbought** (stock may be overvalued)")
            elif rsi_val < 30:
                st.success(f"RSI = {rsi_val} → **Oversold** (stock may be undervalued)")
            else:
                st.info(f"RSI = {rsi_val} → **Neutral** range")

    with tab2:
        st.plotly_chart(create_macd_chart(df), use_container_width=True)
        if r['macd'] and r['macd_signal_line']:
            if r['macd'] > r['macd_signal_line']:
                st.success(f"MACD ({r['macd']:.4f}) > Signal ({r['macd_signal_line']:.4f}) → **Bullish**")
            else:
                st.error(f"MACD ({r['macd']:.4f}) < Signal ({r['macd_signal_line']:.4f}) → **Bearish**")

    with tab3:
        st.plotly_chart(create_bollinger_chart(df), use_container_width=True)

    st.markdown("---")

    # ═══ MOVING AVERAGES TABLE ═══
    st.markdown("### 📋 Moving Averages Summary")
    ma_data = {
        "Indicator": ["SMA 20", "SMA 50", "SMA 200", "Current Price"],
        "Value (PKR)": [
            f"{r['sma_20']:,.2f}" if r['sma_20'] else "N/A",
            f"{r['sma_50']:,.2f}" if r['sma_50'] else "N/A",
            f"{r['sma_200']:,.2f}" if r['sma_200'] else "N/A",
            f"{r['current_price']:,.2f}"
        ],
        "Signal": [
            "✅ Bullish" if r['sma_20'] and r['current_price'] > r['sma_20'] else "❌ Bearish",
            "✅ Bullish" if r['sma_50'] and r['current_price'] > r['sma_50'] else "❌ Bearish",
            "✅ Bullish" if r['sma_200'] and r['current_price'] > r['sma_200'] else "❌ Bearish",
            "—"
        ]
    }
    st.dataframe(pd.DataFrame(ma_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ═══ MODEL ACCURACY ═══
    st.markdown("### 🤖 Model Performance (Linear Regression)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"PKR {r['mae']:,.2f}", help="Mean Absolute Error - avg prediction error")
    m2.metric("RMSE", f"PKR {r['rmse']:,.2f}", help="Root Mean Squared Error")
    m3.metric("R² Score", f"{r['r2_score']:.4f}", help="1.0 = perfect, 0 = random")
    m4.metric("MAPE", f"{r['mape']:.2f}%", help="Mean Absolute Percentage Error")

    st.markdown("#### Actual vs Predicted (Test Data)")
    fig_acc = create_accuracy_chart(r['y_test'], r['y_pred_test'])
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("---")

    # ═══ PREDICTION TABLE ═══
    st.markdown("### 🔮 Predicted Prices (Next Trading Days)")
    pred_table = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d (%A)") for d in r['predicted_dates']],
        "Predicted Price (PKR)": [f"{p:,.2f}" for p in r['predicted_prices']],
        "Change from Today": [f"{((p - r['current_price'])/r['current_price']*100):+.2f}%"
                              for p in r['predicted_prices']]
    })
    st.dataframe(pred_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ═══ RAW DATA DOWNLOAD ═══
    st.markdown("### 💾 Download Data")
    col_a, col_b = st.columns(2)
    with col_a:
        csv_hist = r['df_historical'].to_csv(index=False)
        st.download_button("📥 Download Historical Data (CSV)", csv_hist,
                          f"{r['ticker']}_historical.csv", "text/csv")
    with col_b:
        csv_pred = pred_table.to_csv(index=False)
        st.download_button("📥 Download Predictions (CSV)", csv_pred,
                          f"{r['ticker']}_predictions.csv", "text/csv")

else:
    # ── Welcome Screen ──
    st.markdown("")
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown("""
        <div style="text-align:center; padding:60px 0;">
            <div style="font-size:72px; margin-bottom:16px;">📈</div>
            <h2 style="margin-bottom:8px;">Welcome to PSX Predictor</h2>
            <p style="color:#94a3b8; font-size:16px;">
                Select a stock from the sidebar and click <b>Run Prediction</b><br>
                to see analysis, charts, and price predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Show available stocks ──
    st.markdown("### 📋 Available PSX Stocks")
    stock_df = pd.DataFrame(get_stock_list())
    stock_df = stock_df[['name', 'ticker', 'sector']]
    stock_df.columns = ['Company Name', 'Ticker', 'Sector']
    st.dataframe(stock_df, use_container_width=True, hide_index=True)

# ── Footer ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:13px;'>"
    "🎓 University Project — PSX Stock Predictor using Linear Regression | "
    "Built with Python, Streamlit, scikit-learn & Plotly"
    "</div>",
    unsafe_allow_html=True
)
