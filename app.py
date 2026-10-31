import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

# --- Configuration ---
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide", page_icon="📊")

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def get_stock_data(ticker, period="1y"):
    """Fetch stock data with error handling - NO CACHING"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if not data.empty:
            data = data.copy()
            # Remove timezone info to avoid serialization issues
            if hasattr(data.index, 'tz_localize'):
                data.index = data.index.tz_localize(None)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def get_stock_info(ticker):
    """Fetch stock info - NO CACHING"""
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception as e:
        st.error(f"Error fetching stock info for {ticker}: {str(e)}")
        return {}

def get_stock_news(ticker, limit=5):
    """Fetch recent news for a stock ticker - FIXED VERSION"""
    try:
        stock = yf.Ticker(ticker)
        # Try to get news from yfinance
        try:
            news = stock.news
        except:
            # If that fails, try alternative approach
            news = []
            
        articles = []
        if news and isinstance(news, list) and len(news) > 0:
            for item in news[:limit]:
                try:
                    # More robust data extraction
                    title = item.get("title", "")
                    if not title or title.strip() == "":
                        continue
                        
                    # Handle different timestamp formats
                    published_time = item.get("providerPublishTime")
                    if published_time:
                        try:
                            if isinstance(published_time, (int, float)):
                                published = datetime.fromtimestamp(published_time).strftime("%Y-%m-%d %H:%M")
                            else:
                                published = str(published_time)
                        except:
                            published = "Unknown date"
                    else:
                        published = "Unknown date"
                    
                    article = {
                        "title": title.strip(),
                        "publisher": item.get("publisher", "Unknown"),
                        "link": item.get("link", "#"),
                        "published": published,
                        "summary": item.get("summary", "")[:200] + "..." if item.get("summary") and len(item.get("summary", "")) > 200 else item.get("summary", "")
                    }
                    articles.append(article)
                except Exception as e:
                    continue
        
        # If no news found, try alternative news source
        if not articles:
            try:
                articles = get_alternative_news(ticker, limit)
            except:
                pass
                
        return articles
    except Exception as e:
        st.warning(f"Error fetching news for {ticker}: {str(e)}")
        return []

def get_alternative_news(ticker, limit=5):
    """Alternative news source using web scraping (fallback)"""
    try:
        # This is a simple fallback - you might want to implement a more robust solution
        # For now, return empty list to avoid errors
        return []
    except:
        return []

def calculate_technical_indicators(data):
    """Calculate technical indicators without TA-Lib dependency - NO CACHING"""
    if data is None or data.empty:
        return data
    
    try:
        # Make a copy to avoid modifying original
        data = data.copy()
        
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        
        return data
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return data

def analyze_news_sentiment(articles):
    """Enhanced sentiment analysis for news articles"""
    if not articles:
        return 0, "No data", {}
    
    positive_words = [
        'buy', 'bull', 'bullish', 'up', 'gain', 'gains', 'rise', 'rises', 'rising', 
        'strong', 'strength', 'beat', 'beats', 'growth', 'high', 'surge', 'rally', 
        'rallies', 'outperform', 'upgrade', 'upgraded', 'positive', 'boost', 'boosts', 
        'record', 'peak', 'soar', 'soars', 'excellent', 'outstanding', 'success', 
        'profit', 'profits', 'revenue', 'expand', 'expansion', 'breakthrough'
    ]
    
    negative_words = [
        'sell', 'bear', 'bearish', 'down', 'fall', 'falls', 'falling', 'decline', 
        'declines', 'weak', 'weakness', 'miss', 'misses', 'loss', 'losses', 'low', 
        'drop', 'drops', 'crash', 'crashes', 'underperform', 'downgrade', 'downgraded', 
        'negative', 'concern', 'concerns', 'risk', 'risks', 'plunge', 'plunges', 
        'disappointing', 'warning', 'cut', 'reduce', 'layoff', 'lawsuit'
    ]
    
    sentiment_scores = []
    sentiment_breakdown = {'positive': 0, 'neutral': 0, 'negative': 0}
    
    for article in articles:
        title_lower = article.get('title', '').lower()
        summary_lower = article.get('summary', '').lower()
        text_to_analyze = title_lower + ' ' + summary_lower
        
        pos_count = sum(1 for word in positive_words if word in text_to_analyze)
        neg_count = sum(1 for word in negative_words if word in text_to_analyze)
        
        score = pos_count - neg_count
        sentiment_scores.append(score)
        
        if score > 0:
            sentiment_breakdown['positive'] += 1
        elif score < 0:
            sentiment_breakdown['negative'] += 1
        else:
            sentiment_breakdown['neutral'] += 1
    
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    
    if avg_sentiment > 0.3:
        sentiment_label = "Positive"
    elif avg_sentiment < -0.3:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    
    return avg_sentiment, sentiment_label, sentiment_breakdown

def get_financial_metrics(stock_info):
    """Extract key financial metrics from stock info"""
    metrics = {
        "Market Cap": stock_info.get("marketCap"),
        "Enterprise Value": stock_info.get("enterpriseValue"),
        "P/E Ratio (TTM)": stock_info.get("trailingPE"),
        "Forward P/E": stock_info.get("forwardPE"),
        "PEG Ratio": stock_info.get("pegRatio"),
        "Price-to-Sales": stock_info.get("priceToSalesTrailing12Months"),
        "Price-to-Book": stock_info.get("priceToBook"),
        "EPS (TTM)": stock_info.get("trailingEps"),
        "Forward EPS": stock_info.get("forwardEps"),
        "Dividend Yield": stock_info.get("dividendYield"),
        "Beta": stock_info.get("beta"),
        "52-Week High": stock_info.get("fiftyTwoWeekHigh"),
        "52-Week Low": stock_info.get("fiftyTwoWeekLow"),
        "50-Day Average": stock_info.get("fiftyDayAverage"),
        "200-Day Average": stock_info.get("twoHundredDayAverage"),
        "Shares Outstanding": stock_info.get("sharesOutstanding"),
        "Float Shares": stock_info.get("floatShares"),
        "Revenue (TTM)": stock_info.get("totalRevenue"),
        "Gross Margins": stock_info.get("grossMargins"),
        "Operating Margins": stock_info.get("operatingMargins"),
        "Profit Margins": stock_info.get("profitMargins"),
        "Return on Equity": stock_info.get("returnOnEquity"),
        "Return on Assets": stock_info.get("returnOnAssets"),
        "Debt-to-Equity": stock_info.get("debtToEquity"),
        "Current Ratio": stock_info.get("currentRatio"),
        "Quick Ratio": stock_info.get("quickRatio")
    }
    return metrics

def format_large_number(num):
    """Format large numbers for display"""
    if num is None:
        return "N/A"
    if abs(num) >= 1e12:
        return f"${num/1e12:.2f}T"
    elif abs(num) >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

def format_percentage(num):
    """Format percentages for display"""
    if num is None:
        return "N/A"
    return f"{num*100:.2f}%" if abs(num) < 1 else f"{num:.2f}%"

def format_ratio(num):
    """Format ratios for display"""
    if num is None:
        return "N/A"
    return f"{num:.2f}"

# --- Main App ---
st.title("Stock Analysis Dashboard")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("🔧 Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL", help="e.g., AAPL, MSFT, GOOGL").upper()
period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line", "Area"], index=0)

# Fetch data
if ticker:
    with st.spinner(f"Loading data for {ticker}..."):
        data = get_stock_data(ticker, period)
        stock_info = get_stock_info(ticker)
    
    if data is not None and not data.empty:
        # Calculate technical indicators
        data = calculate_technical_indicators(data)
        
        # Get company name
        company_name = stock_info.get('longName', ticker)
        
        # --- Header with key metrics ---
        current_price = data['Close'].iloc[-1]
        prev_close = stock_info.get('previousClose', data['Close'].iloc[-2] if len(data) > 1 else current_price)
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label=f"**{company_name}** ({ticker})",
                value=f"${current_price:.2f}",
                delta=f"{change:+.2f} ({change_pct:+.2f}%)"
            )
        
        with col2:
            volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            volume_change = ((volume - avg_volume) / avg_volume) * 100 if avg_volume != 0 else 0
            st.metric(
                label="Volume",
                value=f"{volume:,.0f}",
                delta=f"{volume_change:+.1f}% vs 20-day avg"
            )
        
        with col3:
            market_cap = stock_info.get('marketCap')
            st.metric(
                label="Market Cap",
                value=format_large_number(market_cap) if market_cap else "N/A"
            )
        
        with col4:
            pe_ratio = stock_info.get('trailingPE')
            st.metric(
                label="P/E Ratio (TTM)",
                value=format_ratio(pe_ratio) if pe_ratio else "N/A"
            )
        
        # --- Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Technical Analysis", "Fundamentals", "News & Research"])
        
        # --- Tab 1: Price Chart ---
        with tab1:
            st.subheader("📈 Price Chart")
            
            # Chart options
            col1, col2 = st.columns([3, 1])
            
            with col2:
                show_volume = st.checkbox("Show Volume", value=True)
                show_sma = st.checkbox("Show Moving Averages", value=True)
                show_bollinger = st.checkbox("Show Bollinger Bands", value=False)
            
            with col1:
                fig = go.Figure()
                
                if chart_type == "Candlestick":
                    fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name=ticker
                    ))
                elif chart_type == "Line":
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name=f'{ticker} Close',
                        line=dict(color='blue', width=2)
                    ))
                else:  # Area
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        fill='tonexty',
                        mode='lines',
                        name=f'{ticker} Close',
                        line=dict(color='blue', width=1)
                    ))
                
                # Add moving averages
                if show_sma:
                    for ma_period, color in [(20, 'orange'), (50, 'red'), (200, 'purple')]:
                        if f'SMA_{ma_period}' in data.columns:
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data[f'SMA_{ma_period}'],
                                mode='lines',
                                name=f'SMA {ma_period}',
                                line=dict(color=color, width=1)
                            ))
                
                # Add Bollinger Bands
                if show_bollinger and 'BB_Upper' in data.columns:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['BB_Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['BB_Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)'
                    ))
                
                fig.update_layout(
                    title=f'{ticker} Stock Price',
                    yaxis_title='Price ($)',
                    xaxis_title='Date',
                    height=500,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                if show_volume:
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name='Volume',
                        marker_color='lightblue'
                    ))
                    if 'Volume_SMA' in data.columns:
                        fig_vol.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Volume_SMA'],
                            mode='lines',
                            name='Volume SMA (20)',
                            line=dict(color='red', width=2)
                        ))
                    
                    fig_vol.update_layout(
                        title='Volume',
                        yaxis_title='Volume',
                        xaxis_title='Date',
                        height=300,
                        showlegend=True
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)
        
        # --- Tab 2: Technical Analysis ---
        with tab2:
            st.subheader("Technical Indicators")
            
            # RSI
            if 'RSI' in data.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    current_rsi = data['RSI'].iloc[-1]
                    rsi_signal = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
                    rsi_color = "negative" if current_rsi < 30 else "positive" if current_rsi > 70 else "neutral"
                    
                    st.markdown(f"**RSI (14-day): <span class='{rsi_color}'>{current_rsi:.2f} ({rsi_signal})</span>**", unsafe_allow_html=True)
                    
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig_rsi.update_layout(title="RSI (Relative Strength Index)", yaxis_title="RSI", height=300)
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    # MACD
                    if 'MACD' in data.columns:
                        current_macd = data['MACD'].iloc[-1]
                        current_signal = data['MACD_Signal'].iloc[-1]
                        macd_signal = "Bullish" if current_macd > current_signal else "Bearish"
                        macd_color = "positive" if current_macd > current_signal else "negative"
                        
                        st.markdown(f"**MACD: <span class='{macd_color}'>{current_macd:.4f} ({macd_signal})</span>**", unsafe_allow_html=True)
                        
                        fig_macd = go.Figure()
                        fig_macd.add_trace(go.Scatter(
                            x=data.index,
                            y=data['MACD'],
                            mode='lines',
                            name='MACD',
                            line=dict(color='blue', width=2)
                        ))
                        fig_macd.add_trace(go.Scatter(
                            x=data.index,
                            y=data['MACD_Signal'],
                            mode='lines',
                            name='Signal',
                            line=dict(color='red', width=2)
                        ))
                        fig_macd.add_trace(go.Bar(
                            x=data.index,
                            y=data['MACD_Histogram'],
                            name='Histogram',
                            marker_color='gray',
                            opacity=0.6
                        ))
                        fig_macd.update_layout(title="MACD", yaxis_title="MACD", height=300)
                        st.plotly_chart(fig_macd, use_container_width=True)
            
            # Technical Summary
            st.subheader("📋 Technical Summary")
            technical_signals = []
            
            if 'RSI' in data.columns:
                rsi_val = data['RSI'].iloc[-1]
                if rsi_val < 30:
                    technical_signals.append("🟢 RSI indicates oversold condition (potential buy signal)")
                elif rsi_val > 70:
                    technical_signals.append("🔴 RSI indicates overbought condition (potential sell signal)")
                else:
                    technical_signals.append("🟡 RSI in neutral zone")
            
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1]:
                    technical_signals.append("🟢 MACD above signal line (bullish)")
                else:
                    technical_signals.append("🔴 MACD below signal line (bearish)")
            
            # Moving Average Analysis
            current_price = data['Close'].iloc[-1]
            if 'SMA_20' in data.columns and data['SMA_20'].notna().any():
                sma20 = data['SMA_20'].dropna().iloc[-1] if len(data['SMA_20'].dropna()) > 0 else None
                if sma20:
                    if current_price > sma20:
                        technical_signals.append(f"🟢 Price above 20-day SMA (${sma20:.2f})")
                    else:
                        technical_signals.append(f"🔴 Price below 20-day SMA (${sma20:.2f})")
            
            if 'SMA_200' in data.columns and data['SMA_200'].notna().any():
                sma200_data = data['SMA_200'].dropna()
                if len(sma200_data) > 0:
                    sma200 = sma200_data.iloc[-1]
                    if current_price > sma200:
                        technical_signals.append(f"🟢 Price above 200-day SMA (${sma200:.2f}) - Long-term uptrend")
                    else:
                        technical_signals.append(f"🔴 Price below 200-day SMA (${sma200:.2f}) - Long-term downtrend")
                else:
                    technical_signals.append("Insufficient data for 200-day SMA analysis")
            
            for signal in technical_signals:
                st.write(signal)
        
        # --- Tab 3: Fundamentals ---
        with tab3:
            st.subheader("Fundamental Analysis")
            
            if stock_info:
                # Company Overview
                st.write("### Company Overview")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Company Name:** {stock_info.get('longName', 'N/A')}")
                    st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
                    st.write(f"**Country:** {stock_info.get('country', 'N/A')}")
                    st.write(f"**Website:** {stock_info.get('website', 'N/A')}")
                
                with col2:
                    st.write(f"**Full Time Employees:** {stock_info.get('fullTimeEmployees', 'N/A'):,}" if stock_info.get('fullTimeEmployees') else "**Full Time Employees:** N/A")
                    st.write(f"**Exchange:** {stock_info.get('exchange', 'N/A')}")
                    st.write(f"**Currency:** {stock_info.get('currency', 'N/A')}")
                
                # Business Summary
                if stock_info.get('longBusinessSummary'):
                    st.write("### Business Summary")
                    st.write(stock_info['longBusinessSummary'][:500] + "..." if len(stock_info['longBusinessSummary']) > 500 else stock_info['longBusinessSummary'])
                
                # Financial Metrics
                st.write("### Key Financial Metrics")
                metrics = get_financial_metrics(stock_info)
                
                # Organize metrics into categories
                valuation_metrics = {
                    "Market Cap": format_large_number(metrics.get("Market Cap")),
                    "Enterprise Value": format_large_number(metrics.get("Enterprise Value")),
                    "P/E Ratio (TTM)": format_ratio(metrics.get("P/E Ratio (TTM)")),
                    "Forward P/E": format_ratio(metrics.get("Forward P/E")),
                    "PEG Ratio": format_ratio(metrics.get("PEG Ratio")),
                    "Price-to-Sales": format_ratio(metrics.get("Price-to-Sales")),
                    "Price-to-Book": format_ratio(metrics.get("Price-to-Book"))
                }
                
                profitability_metrics = {
                    "Revenue (TTM)": format_large_number(metrics.get("Revenue (TTM)")),
                    "EPS (TTM)": format_ratio(metrics.get("EPS (TTM)")),
                    "Forward EPS": format_ratio(metrics.get("Forward EPS")),
                    "Gross Margins": format_percentage(metrics.get("Gross Margins")),
                    "Operating Margins": format_percentage(metrics.get("Operating Margins")),
                    "Profit Margins": format_percentage(metrics.get("Profit Margins")),
                    "Return on Equity": format_percentage(metrics.get("Return on Equity")),
                    "Return on Assets": format_percentage(metrics.get("Return on Assets"))
                }
                
                financial_health = {
                    "Current Ratio": format_ratio(metrics.get("Current Ratio")),
                    "Quick Ratio": format_ratio(metrics.get("Quick Ratio")),
                    "Debt-to-Equity": format_ratio(metrics.get("Debt-to-Equity")),
                    "Beta": format_ratio(metrics.get("Beta"))
                }
                
                share_info = {
                    "Shares Outstanding": f"{metrics.get('Shares Outstanding'):,.0f}" if metrics.get('Shares Outstanding') else "N/A",
                    "Float Shares": f"{metrics.get('Float Shares'):,.0f}" if metrics.get('Float Shares') else "N/A",
                    "52-Week High": f"${metrics.get('52-Week High'):.2f}" if metrics.get('52-Week High') else "N/A",
                    "52-Week Low": f"${metrics.get('52-Week Low'):.2f}" if metrics.get('52-Week Low') else "N/A",
                    "50-Day Average": f"${metrics.get('50-Day Average'):.2f}" if metrics.get('50-Day Average') else "N/A",
                    "200-Day Average": f"${metrics.get('200-Day Average'):.2f}" if metrics.get('200-Day Average') else "N/A",
                    "Dividend Yield": format_percentage(metrics.get("Dividend Yield"))
                }
                
                # Display metrics in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Valuation Metrics")
                    for key, value in valuation_metrics.items():
                        st.write(f"**{key}:** {value}")
                    
                    st.write("#### Financial Health")
                    for key, value in financial_health.items():
                        st.write(f"**{key}:** {value}")
                
                with col2:
                    st.write("#### Profitability Metrics")
                    for key, value in profitability_metrics.items():
                        st.write(f"**{key}:** {value}")
                    
                    st.write("#### Share Information")
                    for key, value in share_info.items():
                        st.write(f"**{key}:** {value}")
            else:
                st.warning("Unable to fetch fundamental data for this ticker.")
        
        # --- Tab 4: News & Research (FIXED VERSION) ---
        with tab4:
            st.subheader("📰 Latest News & Analysis")
            
            # Fetch news with better error handling
            news_articles = []
            with st.spinner("Loading news..."):
                try:
                    news_articles = get_stock_news(ticker, limit=8)
                except Exception as e:
                    st.warning(f"Unable to fetch news at this time: {str(e)}")
            
            if news_articles:
                st.write(f"### Recent Headlines for {ticker} ({len(news_articles)} articles found)")
                
                # Display articles
                for i, article in enumerate(news_articles, 1):
                    title = article.get('title', '').strip()
                    if not title:
                        continue
                        
                    # Create expandable article view
                    with st.expander(f"{i}. {title[:80]}..." if len(title) > 80 else f"{i}. {title}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Title:** {title}")
                            st.markdown(f"**Publisher:** {article.get('publisher', 'Unknown')}")
                            st.markdown(f"**Published:** {article.get('published', 'Unknown date')}")
                            
                            # Show summary if available
                            summary = article.get('summary', '').strip()
                            if summary:
                                st.markdown(f"**Summary:** {summary}")
                        
                        with col2:
                            link = article.get('link', '')
                            if link and link != "#" and link.startswith('http'):
                                st.markdown(f"[Read Full Article]({link})")
                            else:
                                st.markdown("*Link unavailable*")
                
                # News Sentiment Analysis
                st.write("### News Sentiment Analysis")
                
                try:
                    avg_sentiment, sentiment_label, sentiment_breakdown = analyze_news_sentiment(news_articles)
                    
                    # Display sentiment metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        sentiment_color = "🟢" if sentiment_label == "Positive" else "🔴" if sentiment_label == "Negative" else "🟡"
                        st.metric(
                            label="Overall Sentiment",
                            value=f"{sentiment_color} {sentiment_label}",
                            delta=f"Score: {avg_sentiment:.2f}"
                        )
                    
                    with col2:
                        total_articles = len(news_articles)
                        positive_pct = (sentiment_breakdown['positive'] / total_articles * 100) if total_articles > 0 else 0
                        st.metric(
                            label="Positive Articles",
                            value=f"{sentiment_breakdown['positive']}/{total_articles}",
                            delta=f"{positive_pct:.1f}%"
                        )
                    
                    with col3:
                        negative_pct = (sentiment_breakdown['negative'] / total_articles * 100) if total_articles > 0 else 0
                        st.metric(
                            label="Negative Articles",
                            value=f"{sentiment_breakdown['negative']}/{total_articles}",
                            delta=f"{negative_pct:.1f}%"
                        )
                    
                    # Sentiment Distribution Chart
                    if sum(sentiment_breakdown.values()) > 0:
                        sentiment_labels = ['Positive', 'Neutral', 'Negative']
                        sentiment_counts = [
                            sentiment_breakdown['positive'],
                            sentiment_breakdown['neutral'],
                            sentiment_breakdown['negative']
                        ]
                        
                        # Only show non-zero values
                        filtered_labels = []
                        filtered_counts = []
                        colors = []
                        color_map = {'Positive': '#28a745', 'Neutral': '#6c757d', 'Negative': '#dc3545'}
                        
                        for label, count in zip(sentiment_labels, sentiment_counts):
                            if count > 0:
                                filtered_labels.append(f"{label} ({count})")
                                filtered_counts.append(count)
                                colors.append(color_map[label])
                        
                        if filtered_counts:
                            fig_sentiment = go.Figure(data=[go.Pie(
                                labels=filtered_labels,
                                values=filtered_counts,
                                marker_colors=colors,
                                textinfo='label+percent',
                                textposition='outside'
                            )])
                            
                            fig_sentiment.update_layout(
                                title="News Sentiment Distribution",
                                showlegend=True,
                                height=400
                            )
                            st.plotly_chart(fig_sentiment, use_container_width=True)
                    
                    # Sentiment insights
                    st.write("#### Sentiment Insights")
                    if avg_sentiment > 0.5:
                        st.success("**Strong Positive Sentiment**: News coverage is overwhelmingly positive, which could indicate strong market confidence.")
                    elif avg_sentiment > 0:
                        st.info("**Mild Positive Sentiment**: News coverage leans positive, suggesting cautious optimism.")
                    elif avg_sentiment < -0.5:
                        st.error("**Strong Negative Sentiment**: News coverage is predominantly negative, which may indicate market concerns.")
                    elif avg_sentiment < 0:
                        st.warning("**Mild Negative Sentiment**: News coverage leans negative, suggesting some market caution.")
                    else:
                        st.info("**Neutral Sentiment**: News coverage is balanced, indicating stable market perception.")
                
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {str(e)}")
                    
            else:
                st.info(f"No recent news articles found for {ticker}")
                
                st.write("### External News Sources")
                st.markdown("Try checking these financial news sources for the latest updates:")
                
                # Create clickable links to external sources
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Yahoo Finance**")
                    st.markdown(f"[Quote](https://finance.yahoo.com/quote/{ticker}/)")
                    st.markdown(f"[News](https://finance.yahoo.com/quote/{ticker}/news/)")
                    st.markdown(f"[Charts](https://finance.yahoo.com/quote/{ticker}/chart/)")
                
                with col2:
                    st.markdown(f"**Google Finance**")
                    st.markdown(f"[Overview](https://www.google.com/finance/quote/{ticker}:NASDAQ)")
                    st.markdown(f"**SEC Filings**")
                    st.markdown(f"[EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker})")
                
                with col3:
                    st.markdown(f"**Other Sources**")
                    st.markdown(f"[MarketWatch](https://www.marketwatch.com/investing/stock/{ticker.lower()})")
                    st.markdown(f"[Seeking Alpha](https://seekingalpha.com/symbol/{ticker})")
                    st.markdown(f"[Finviz](https://finviz.com/quote.ashx?t={ticker})")
            
            # Market Analysis Section
            st.write("### Quick Market Analysis")
            
            try:
                # Calculate market statistics
                returns = data['Close'].pct_change().dropna()
                if len(returns) > 0:
                    volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Annualized Volatility", f"{volatility:.1f}%")
                    
                    with col2:
                        # Calculate period return
                        period_return = ((current_price / data['Close'].iloc[0]) - 1) * 100
                        st.metric(f"{period.upper()} Return", f"{period_return:+.1f}%")
                    
                    with col3:
                        # Sharpe ratio (simplified, assuming 0% risk-free rate)
                        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    
                    with col4:
                        # Max drawdown
                        rolling_max = data['Close'].expanding().max()
                        drawdown = (data['Close'] - rolling_max) / rolling_max
                        max_drawdown = drawdown.min() * 100
                        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
                    
                    # Risk Assessment
                    st.write("#### Risk Assessment")
                    
                    risk_factors = []
                    
                    # Volatility assessment
                    if volatility > 40:
                        risk_factors.append("🔴 **High Volatility**: Very volatile stock with significant price swings")
                    elif volatility > 25:
                        risk_factors.append("🟡 **Moderate Volatility**: Moderately volatile stock")
                    else:
                        risk_factors.append("🟢 **Low Volatility**: Relatively stable stock")
                    
                    # Beta assessment
                    beta = stock_info.get('beta')
                    if beta:
                        if beta > 1.5:
                            risk_factors.append(f"🔴 **High Beta** ({beta:.2f}): More volatile than market")
                        elif beta > 1.0:
                            risk_factors.append(f"🟡 **Above Market Beta** ({beta:.2f}): Slightly more volatile than market")
                        elif beta > 0.5:
                            risk_factors.append(f"🟢 **Below Market Beta** ({beta:.2f}): Less volatile than market")
                        else:
                            risk_factors.append(f"🟢 **Low Beta** ({beta:.2f}): Much less volatile than market")
                    
                    # Display risk factors
                    for factor in risk_factors:
                        st.write(factor)
                        
                    # Trading volume analysis
                    recent_volume = data['Volume'].iloc[-5:].mean()  # 5-day average
                    historical_volume = data['Volume'].mean()
                    volume_ratio = recent_volume / historical_volume if historical_volume != 0 else 1
                    
                    st.write("#### Volume Analysis")
                    if volume_ratio > 1.5:
                        st.write("🟢 **High Volume Activity**: Recent trading volume is significantly above average")
                    elif volume_ratio > 1.2:
                        st.write("🟡 **Above Average Volume**: Recent trading volume is moderately elevated")
                    elif volume_ratio < 0.7:
                        st.write("🔴 **Low Volume Activity**: Recent trading volume is below average")
                    else:
                        st.write("⚪ **Normal Volume**: Recent trading volume is within normal range")
                    
                else:
                    st.warning("Insufficient data for market analysis")
                    
            except Exception as e:
                st.error(f"Error calculating market analysis: {str(e)}")
    
    else:
        st.error(f"Unable to fetch data for ticker: {ticker}")
        st.info("Please check that the ticker symbol is correct and try again.")

else:
    # Welcome screen
    st.info("Welcome to the Stock Analysis Dashboard!")
    st.markdown("""
    **Getting Started:**
    1. Enter a stock ticker symbol in the sidebar (e.g., AAPL, MSFT, GOOGL)
    2. Select your preferred time period and chart type
    3. Explore the four main sections:
       - **Price Chart**: Interactive price charts with technical indicators
       - **Technical Analysis**: RSI, MACD, and other technical indicators
       - **Fundamentals**: Company information and financial metrics
       - **News & Research**: Latest news with sentiment analysis
    
    **Features:**
    - Real-time stock data and news
    - Advanced technical indicators
    - Comprehensive fundamental analysis
    - News sentiment analysis
    - Risk assessment tools
    """)
    
    # Show some example tickers
    st.write("### Popular Tickers to Try:")
    example_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    
    cols = st.columns(4)
    for i, ticker_example in enumerate(example_tickers):
        with cols[i % 4]:
            if st.button(ticker_example, key=f"example_{ticker_example}"):
                st.rerun()
