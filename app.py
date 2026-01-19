"""
Stock News Sentiment Analyzer with Zapier Integration
======================================================
A real-world ML project demonstrating:
- Free data sources (Yahoo Finance, NewsAPI)
- Machine Learning (Sentiment Analysis, Trend Prediction)
- Zapier automation integration via webhooks
- Deployable on Hugging Face Spaces

Author: Your Name
License: MIT
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from transformers import pipeline
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

# Zapier Webhook URL (User will add their own)
ZAPIER_WEBHOOK_URL = None  # Set via environment variable or UI

# Free News API alternatives (no key required for demo)
FREE_NEWS_SOURCES = {
    "Google News RSS": "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
    "Reddit Finance": "https://www.reddit.com/r/stocks/search.json?q={query}&limit=10&sort=new"
}

# ============================================
# MACHINE LEARNING MODELS
# ============================================

# Load sentiment analysis model (FinBERT for financial sentiment)
print("Loading ML models...")
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="ProsusAI/finbert",
        top_k=None
    )
except:
    # Fallback to general sentiment model
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

print("Models loaded successfully!")

# ============================================
# DATA FETCHING FUNCTIONS
# ============================================

def fetch_stock_data(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance (FREE)"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        df.reset_index(inplace=True)
        df['Ticker'] = ticker
        return df
    except Exception as e:
        return pd.DataFrame()

def fetch_stock_news(ticker: str) -> list:
    """Fetch news from Yahoo Finance (FREE)"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news[:10] if news else []
    except:
        return []

def fetch_reddit_sentiment(query: str) -> list:
    """Fetch posts from Reddit Finance subreddits (FREE)"""
    try:
        headers = {'User-Agent': 'StockSentimentBot/1.0'}
        url = f"https://www.reddit.com/r/stocks/search.json?q={query}&limit=10&sort=new"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            posts = []
            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                posts.append({
                    'title': post_data.get('title', ''),
                    'score': post_data.get('score', 0),
                    'created': datetime.fromtimestamp(post_data.get('created_utc', 0)),
                    'source': 'Reddit'
                })
            return posts
    except:
        pass
    return []

# ============================================
# SENTIMENT ANALYSIS
# ============================================

def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of a single text"""
    try:
        result = sentiment_analyzer(text[:512])  # Limit text length
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                # Multi-label output
                sentiments = {item['label']: item['score'] for item in result[0]}
            else:
                # Single label output
                sentiments = {result[0]['label']: result[0]['score']}
            return sentiments
    except Exception as e:
        return {"error": str(e)}
    return {"neutral": 1.0}

def analyze_news_batch(news_items: list) -> pd.DataFrame:
    """Analyze sentiment for multiple news items"""
    results = []
    for item in news_items:
        title = item.get('title', '') or item.get('content', {}).get('title', '')
        if not title:
            continue
        
        sentiment = analyze_sentiment(title)
        
        # Determine overall sentiment
        if 'positive' in sentiment:
            score = sentiment.get('positive', 0) - sentiment.get('negative', 0)
            label = 'Positive' if score > 0.1 else ('Negative' if score < -0.1 else 'Neutral')
        elif 'POSITIVE' in sentiment:
            score = sentiment.get('POSITIVE', 0) - sentiment.get('NEGATIVE', 0)
            label = 'Positive' if score > 0.1 else ('Negative' if score < -0.1 else 'Neutral')
        else:
            label = list(sentiment.keys())[0] if sentiment else 'Neutral'
            score = list(sentiment.values())[0] if sentiment else 0
        
        results.append({
            'Title': title[:100] + '...' if len(title) > 100 else title,
            'Sentiment': label,
            'Confidence': abs(score) if isinstance(score, (int, float)) else 0.5,
            'Source': item.get('source', 'Yahoo Finance')
        })
    
    return pd.DataFrame(results)

# ============================================
# MACHINE LEARNING: PRICE PREDICTION
# ============================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare technical indicators as ML features"""
    if df.empty or len(df) < 20:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Technical indicators
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=5).std()
    df['Price_Change'] = df['Close'].diff()
    
    # Target: Will price go up tomorrow?
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df.dropna()

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_prediction_model(df: pd.DataFrame) -> tuple:
    """Train a simple Random Forest model for price direction prediction"""
    features = ['SMA_5', 'SMA_20', 'RSI', 'Daily_Return', 'Volatility']
    
    if df.empty or len(df) < 30:
        return None, None, "Insufficient data for training"
    
    X = df[features].values
    y = df['Target'].values
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    accuracy = model.score(X_test_scaled, y_test)
    
    # Predict next day
    last_features = scaler.transform(X[-1:])
    next_day_pred = model.predict_proba(last_features)[0]
    
    return model, {
        'accuracy': accuracy,
        'up_probability': next_day_pred[1] if len(next_day_pred) > 1 else 0.5,
        'down_probability': next_day_pred[0] if len(next_day_pred) > 1 else 0.5
    }, None

# ============================================
# ZAPIER INTEGRATION
# ============================================

def send_to_zapier(webhook_url: str, data: dict) -> str:
    """Send data to Zapier webhook for automation"""
    if not webhook_url or webhook_url.strip() == "":
        return "❌ No webhook URL provided"
    
    try:
        response = requests.post(
            webhook_url,
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        if response.status_code == 200:
            return "✅ Successfully sent to Zapier!"
        else:
            return f"⚠️ Zapier returned status: {response.status_code}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def create_zapier_payload(ticker: str, sentiment_summary: dict, prediction: dict) -> dict:
    """Create structured payload for Zapier automation"""
    return {
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "analysis_type": "stock_sentiment",
        "sentiment": {
            "overall": sentiment_summary.get('overall', 'neutral'),
            "positive_count": sentiment_summary.get('positive', 0),
            "negative_count": sentiment_summary.get('negative', 0),
            "neutral_count": sentiment_summary.get('neutral', 0),
            "avg_confidence": sentiment_summary.get('avg_confidence', 0)
        },
        "ml_prediction": {
            "direction": "UP" if prediction.get('up_probability', 0.5) > 0.5 else "DOWN",
            "confidence": max(prediction.get('up_probability', 0.5), prediction.get('down_probability', 0.5)),
            "model_accuracy": prediction.get('accuracy', 0)
        },
        "alert_level": "HIGH" if sentiment_summary.get('negative', 0) > sentiment_summary.get('positive', 0) else "NORMAL"
    }

# ============================================
# VISUALIZATION
# ============================================

def create_stock_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create interactive stock price chart"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Add moving averages if available
    if 'SMA_5' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_5'], name='SMA 5', line=dict(color='orange')))
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='blue')))
    
    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_sentiment_chart(sentiment_df: pd.DataFrame) -> go.Figure:
    """Create sentiment distribution chart"""
    if sentiment_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No sentiment data", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    sentiment_counts = sentiment_df['Sentiment'].value_counts()
    
    colors = {'Positive': '#00CC96', 'Negative': '#EF553B', 'Neutral': '#636EFA'}
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        marker_colors=[colors.get(s, '#888888') for s in sentiment_counts.index],
        hole=0.4
    )])
    
    fig.update_layout(
        title='News Sentiment Distribution',
        template='plotly_white',
        height=300
    )
    
    return fig

# ============================================
# MAIN ANALYSIS FUNCTION
# ============================================

def analyze_stock(ticker: str, zapier_webhook: str = "") -> tuple:
    """Main analysis function that combines all features"""
    
    ticker = ticker.upper().strip()
    
    if not ticker:
        return (
            "Please enter a stock ticker",
            pd.DataFrame(),
            go.Figure(),
            go.Figure(),
            "",
            ""
        )
    
    # 1. Fetch Stock Data
    stock_df = fetch_stock_data(ticker, period="3mo")
    if stock_df.empty:
        return (
            f"❌ Could not fetch data for {ticker}. Please check the ticker symbol.",
            pd.DataFrame(),
            go.Figure(),
            go.Figure(),
            "",
            ""
        )
    
    # 2. Prepare ML features and train model
    ml_df = prepare_features(stock_df)
    model, prediction, error = train_prediction_model(ml_df)
    
    # 3. Fetch and analyze news
    news_items = fetch_stock_news(ticker)
    reddit_posts = fetch_reddit_sentiment(ticker)
    
    # Combine news sources
    all_news = []
    for item in news_items:
        all_news.append({
            'title': item.get('title', ''),
            'source': 'Yahoo Finance'
        })
    for post in reddit_posts:
        all_news.append({
            'title': post.get('title', ''),
            'source': 'Reddit'
        })
    
    sentiment_df = analyze_news_batch(all_news)
    
    # 4. Calculate sentiment summary
    sentiment_summary = {
        'positive': len(sentiment_df[sentiment_df['Sentiment'] == 'Positive']) if not sentiment_df.empty else 0,
        'negative': len(sentiment_df[sentiment_df['Sentiment'] == 'Negative']) if not sentiment_df.empty else 0,
        'neutral': len(sentiment_df[sentiment_df['Sentiment'] == 'Neutral']) if not sentiment_df.empty else 0,
        'avg_confidence': sentiment_df['Confidence'].mean() if not sentiment_df.empty else 0
    }
    
    # Determine overall sentiment
    if sentiment_summary['positive'] > sentiment_summary['negative']:
        sentiment_summary['overall'] = 'POSITIVE'
    elif sentiment_summary['negative'] > sentiment_summary['positive']:
        sentiment_summary['overall'] = 'NEGATIVE'
    else:
        sentiment_summary['overall'] = 'NEUTRAL'
    
    # 5. Create visualizations
    stock_chart = create_stock_chart(ml_df if not ml_df.empty else stock_df, ticker)
    sentiment_chart = create_sentiment_chart(sentiment_df)
    
    # 6. Generate summary
    current_price = stock_df['Close'].iloc[-1] if not stock_df.empty else 0
    price_change = ((stock_df['Close'].iloc[-1] / stock_df['Close'].iloc[0]) - 1) * 100 if len(stock_df) > 1 else 0
    
    summary = f"""
## 📊 Analysis Summary for {ticker}

### Stock Data
- **Current Price:** ${current_price:.2f}
- **Period Change:** {price_change:+.2f}%
- **Data Points:** {len(stock_df)} days

### 🤖 ML Prediction (Next Day)
- **Direction:** {'📈 UP' if prediction and prediction.get('up_probability', 0.5) > 0.5 else '📉 DOWN'}
- **Confidence:** {prediction.get('up_probability', 0.5)*100:.1f}% up / {prediction.get('down_probability', 0.5)*100:.1f}% down
- **Model Accuracy:** {prediction.get('accuracy', 0)*100:.1f}%

### 📰 Sentiment Analysis
- **Overall Sentiment:** {sentiment_summary['overall']}
- **Positive News:** {sentiment_summary['positive']}
- **Negative News:** {sentiment_summary['negative']}
- **Neutral News:** {sentiment_summary['neutral']}
- **Articles Analyzed:** {len(sentiment_df)}

---
*Analysis generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
    """
    
    # 7. Send to Zapier if webhook provided
    zapier_status = ""
    if zapier_webhook and zapier_webhook.strip():
        payload = create_zapier_payload(ticker, sentiment_summary, prediction or {})
        zapier_status = send_to_zapier(zapier_webhook, payload)
    
    # 8. Generate JSON payload for display
    json_payload = json.dumps(
        create_zapier_payload(ticker, sentiment_summary, prediction or {}),
        indent=2
    )
    
    return (
        summary,
        sentiment_df,
        stock_chart,
        sentiment_chart,
        zapier_status,
        json_payload
    )

# ============================================
# GRADIO INTERFACE
# ============================================

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="Stock Sentiment Analyzer with Zapier",
        theme=gr.themes.Soft(),
        css="""
        .main-title {
            text-align: center;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 📈 Stock News Sentiment Analyzer
        ### With Machine Learning Predictions & Zapier Automation
        
        This app demonstrates:
        - **Free Data Sources:** Yahoo Finance (stock data & news), Reddit
        - **Machine Learning:** FinBERT sentiment analysis + Random Forest price prediction
        - **Zapier Integration:** Automated alerts via webhooks
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                ticker_input = gr.Textbox(
                    label="Stock Ticker Symbol",
                    placeholder="e.g., AAPL, GOOGL, MSFT, TSLA",
                    value="AAPL"
                )
                
                with gr.Accordion("🔗 Zapier Integration (Optional)", open=False):
                    gr.Markdown("""
                    ### How to set up Zapier automation:
                    1. Create a free Zapier account at [zapier.com](https://zapier.com)
                    2. Create a new Zap with "Webhooks by Zapier" as trigger
                    3. Choose "Catch Hook" and copy the webhook URL
                    4. Paste it below and connect to any action (Email, Slack, Google Sheets, etc.)
                    """)
                    zapier_webhook = gr.Textbox(
                        label="Zapier Webhook URL",
                        placeholder="https://hooks.zapier.com/hooks/catch/...",
                        type="password"
                    )
                
                analyze_btn = gr.Button("🔍 Analyze Stock", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("""
                ### 💡 Quick Tips
                - Enter any valid stock ticker
                - Analysis includes 3 months of data
                - ML model predicts next-day direction
                - Sentiment from multiple sources
                - Zapier sends real-time alerts
                """)
        
        gr.Markdown("---")
        
        # Results Section
        with gr.Row():
            with gr.Column():
                summary_output = gr.Markdown(label="Analysis Summary")
        
        with gr.Row():
            with gr.Column():
                stock_chart = gr.Plot(label="Stock Price Chart")
            with gr.Column():
                sentiment_chart = gr.Plot(label="Sentiment Distribution")
        
        with gr.Row():
            sentiment_table = gr.Dataframe(
                label="📰 News Sentiment Details",
                headers=["Title", "Sentiment", "Confidence", "Source"],
                wrap=True
            )
        
        with gr.Row():
            with gr.Column():
                zapier_status = gr.Textbox(label="Zapier Status", interactive=False)
            with gr.Column():
                json_output = gr.Code(
                    label="📤 Data Sent to Zapier (JSON)",
                    language="json"
                )
        
        # Wire up the analyze button
        analyze_btn.click(
            fn=analyze_stock,
            inputs=[ticker_input, zapier_webhook],
            outputs=[
                summary_output,
                sentiment_table,
                stock_chart,
                sentiment_chart,
                zapier_status,
                json_output
            ]
        )
        
        gr.Markdown("""
        ---
        ### 📚 About This Project
        
        This is a demonstration project showing how to:
        
        1. **Fetch free data** from Yahoo Finance and Reddit APIs
        2. **Apply Machine Learning** for sentiment analysis (FinBERT) and price prediction (Random Forest)
        3. **Integrate with Zapier** for automated workflows and alerts
        4. **Deploy on Hugging Face Spaces** for easy sharing
        
        **Technologies Used:**
        - 🐍 Python, Gradio, Transformers (Hugging Face)
        - 📊 yfinance, Plotly, scikit-learn
        - 🔗 Zapier Webhooks
        
        **Free Tier Limitations:**
        - Zapier Free: 100 tasks/month, 5 Zaps
        - Yahoo Finance: Unlimited (unofficial API)
        - Hugging Face Spaces: Free hosting
        
        ---
        *Built for demonstrating data automation skills*
        """)
    
    return demo

# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
