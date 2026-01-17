# 📈 Stock News Sentiment Analyzer with ML & Zapier Automation

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-world machine learning project demonstrating:
- **Free Data Sources** (Yahoo Finance, Reddit)
- **Machine Learning** (Sentiment Analysis + Price Prediction)
- **Workflow Automation** (Zapier Integration)
- **Cloud Deployment** (Hugging Face Spaces)

![App Screenshot](screenshot.png)

---

## 🌟 Features

### 📊 Data Collection (100% Free)
- **Yahoo Finance API**: Real-time stock prices, historical data, and news
- **Reddit API**: Community sentiment from r/stocks and r/wallstreetbets
- No API keys required for basic functionality

### 🤖 Machine Learning Components
1. **FinBERT Sentiment Analysis**
   - Pre-trained transformer model for financial text
   - Classifies news as Positive/Negative/Neutral
   - Confidence scoring for each prediction

2. **Random Forest Price Predictor**
   - Predicts next-day price direction (UP/DOWN)
   - Features: SMA, RSI, Volatility, Daily Returns
   - Trained on 3 months of historical data

### 🔗 Zapier Automation
- Webhook integration for real-time alerts
- Send data to 5,000+ apps (Email, Slack, Sheets, etc.)
- Alert levels based on sentiment analysis
- Works with Zapier's free tier (100 tasks/month)

### 📱 User Interface
- Clean, responsive Gradio interface
- Interactive Plotly charts
- Real-time analysis results
- JSON export for debugging

---

## 🚀 Quick Start

### Option 1: Run on Hugging Face Spaces (Easiest)

1. Go to [Hugging Face Spaces](https://huggingface.co/new-space)
2. Create a new Space with **Gradio** SDK
3. Upload the files from this repository
4. Your app is live! 🎉

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-sentiment-analyzer.git
cd stock-sentiment-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will be available at `http://localhost:7860`

### Option 3: Docker

```bash
docker build -t stock-sentiment .
docker run -p 7860:7860 stock-sentiment
```

---

## 📁 Project Structure

```
stock-sentiment-analyzer/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── ZAPIER_SETUP.md       # Detailed Zapier integration guide
├── HUGGINGFACE_DEPLOY.md # Hugging Face deployment instructions
└── examples/
    └── sample_zaps.json  # Example Zapier configurations
```

---

## 🔧 How It Works

### 1. Data Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Yahoo Finance  │────▶│   Data Cleaning │────▶│  Feature Eng.   │
│  (Stock Data)   │     │   & Processing  │     │  (Technical     │
└─────────────────┘     └─────────────────┘     │   Indicators)   │
                                                 └────────┬────────┘
                                                          │
┌─────────────────┐     ┌─────────────────┐              │
│  Yahoo Finance  │────▶│   Sentiment     │              │
│  (News Articles)│     │   Analysis      │              │
└─────────────────┘     │   (FinBERT)     │              │
                        └────────┬────────┘              │
                                 │                       │
┌─────────────────┐              │                       │
│  Reddit API     │──────────────┤                       │
│  (Community)    │              │                       │
└─────────────────┘              │                       │
                                 ▼                       ▼
                        ┌─────────────────────────────────┐
                        │        Analysis Engine          │
                        │  • Sentiment Aggregation        │
                        │  • ML Price Prediction          │
                        │  • Alert Level Calculation      │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────┴────────────────┐
                        ▼                                 ▼
               ┌─────────────────┐               ┌─────────────────┐
               │  Gradio UI      │               │  Zapier Webhook │
               │  (Visualization)│               │  (Automation)   │
               └─────────────────┘               └─────────────────┘
```

### 2. Machine Learning Models

#### Sentiment Analysis (FinBERT)
- **Model**: `ProsusAI/finbert` (Hugging Face)
- **Input**: News headlines and Reddit post titles
- **Output**: Positive/Negative/Neutral + confidence score
- **Why FinBERT?**: Trained specifically on financial text

#### Price Direction Prediction (Random Forest)
- **Features**:
  - SMA (5-day and 20-day Simple Moving Averages)
  - RSI (Relative Strength Index)
  - Daily Returns
  - 5-day Volatility
- **Target**: Binary classification (Price UP or DOWN next day)
- **Accuracy**: Typically 55-65% (better than random guessing)

### 3. Zapier Integration

The app sends structured JSON to your Zapier webhook:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "ticker": "AAPL",
  "sentiment": {
    "overall": "POSITIVE",
    "positive_count": 5,
    "negative_count": 2
  },
  "ml_prediction": {
    "direction": "UP",
    "confidence": 0.68
  },
  "alert_level": "NORMAL"
}
```

This data can trigger:
- 📧 Email alerts
- 💬 Slack/Discord notifications
- 📊 Google Sheets logging
- 📱 SMS messages
- And 5,000+ more apps!

---

## 📚 Free Resources Used

| Resource | What It Provides | Limitations |
|----------|------------------|-------------|
| **Yahoo Finance** | Stock data, news | Unofficial API, rate limited |
| **Reddit API** | Community posts | 100 requests/minute |
| **Hugging Face** | ML models, hosting | Free tier has cold starts |
| **Zapier Free** | 100 tasks/month | 5 active Zaps |
| **FinBERT Model** | Financial sentiment | ~250MB model size |

---

## 🎓 Learning Objectives

This project teaches:

1. **API Integration**
   - Working with REST APIs
   - Handling rate limits and errors
   - Parsing JSON responses

2. **Machine Learning**
   - Using pre-trained NLP models
   - Feature engineering for time series
   - Training classification models

3. **Web Development**
   - Building UIs with Gradio
   - Creating interactive visualizations
   - Deploying to cloud platforms

4. **Automation**
   - Webhook architecture
   - Event-driven workflows
   - No-code automation tools

---

## 🛠️ Customization Ideas

### Add More Data Sources
```python
# Example: Add Alpha Vantage (free tier)
def fetch_alpha_vantage_news(ticker, api_key):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
    response = requests.get(url)
    return response.json()
```

### Add More ML Models
```python
# Example: Add a LSTM for price prediction
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm_model(sequence_length, n_features):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
        LSTM(50),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
```

### Add More Zapier Actions
- Create multi-step workflows
- Add conditional logic
- Integrate with CRM systems

---

## 📈 Performance Notes

- **Initial Load**: 10-30 seconds (loading ML models)
- **Analysis Time**: 5-15 seconds per stock
- **Memory Usage**: ~2GB (mainly for FinBERT model)
- **API Calls**: ~5-10 per analysis

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- [ ] Add more data sources (Twitter, SEC filings)
- [ ] Improve ML models (LSTM, Transformer)
- [ ] Add backtesting functionality
- [ ] Create mobile-friendly UI
- [ ] Add portfolio tracking

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. 

- Not financial advice
- Past performance doesn't guarantee future results
- ML predictions are not reliable for actual trading
- Use at your own risk

---

## 📄 License

MIT License - feel free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for model hosting
- [Yahoo Finance](https://finance.yahoo.com/) for free data
- [Zapier](https://zapier.com/) for automation platform
- [ProsusAI](https://huggingface.co/ProsusAI/finbert) for FinBERT model

---

## 📬 Contact

Questions? Issues? Open a GitHub issue or reach out!

---

*Built with ❤️ for learning automation and ML*
