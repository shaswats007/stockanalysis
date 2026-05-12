# Stock Analyzer

A Streamlit web app for US equity analysis combining technical indicators, fundamental metrics, and AI-powered recommendations via Google Gemini.

## Features

- **Technical Analysis** — RSI, MACD, Moving Averages (20/50/200 SMA), Bollinger Bands, Volume trends
- **Fundamental Analysis** — P/E, Forward P/E, P/B, Revenue & EPS growth, Net Margin, ROE, Debt/Equity, Current Ratio, Analyst Consensus
- **AI Recommendation** — Buy/Hold/Avoid verdict with narrative from Google Gemini
- **Interactive Chart** — Candlestick with overlaid SMAs, Bollinger Bands, RSI, and MACD subplots
- **500+ Tickers** — Searchable dropdown of S&P 500 stocks

## Setup

**1. Clone and create a virtual environment**
```bash
git clone https://github.com///github.com/shaswats007/stockanalysis.git
cd stockanalysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Add your API key**
```bash
cp .env.example .env
# Edit .env and set your Google Gemini API key
```
Get a free key at [aistudio.google.com](https://aistudio.google.com).

**3. Run**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Usage

1. Select a ticker from the dropdown (or type to search)
2. Choose an analysis period
3. Enter your Gemini API key (or set `GEMINI_API_KEY` in `.env`)
4. Select a Gemini model
5. Click **Analyze**

## Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Market Data | yfinance |
| Technical Indicators | pandas-ta |
| Charts | Plotly |
| AI | Google Gemini (google-genai) |

## Disclaimer

This tool is for informational purposes only and does not constitute financial advice. Always do your own research before making investment decisions.
