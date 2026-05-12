import os
import streamlit as st
from dotenv import load_dotenv

from src.data.fetcher import StockFetcher
from src.analysis.technical import analyze_technical
from src.analysis.fundamental import analyze_fundamental
from src.recommendation.engine import get_recommendation
from src.report.charts import candlestick_chart

load_dotenv()

def _get_api_key_default() -> str:
    if key := os.getenv("GEMINI_API_KEY", ""):
        return key
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


_POPULAR_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","BRK-B","JPM","V",
    "UNH","XOM","LLY","JNJ","WMT","MA","AVGO","PG","HD","CVX","MRK","ABBV",
    "COST","PEP","KO","ADBE","CRM","TMO","ACN","MCD","BAC","NFLX","AMD",
    "INTC","QCOM","TXN","NEE","DHR","LIN","PM","RTX","AMGN","ORCL","T",
    "CSCO","INTU","IBM","GE","HON","CAT","SPGI","BLK","AXP","BKNG","UBER",
    "ABNB","SNAP","PINS","SPOT","COIN","PLTR","SOFI","RBLX","HOOD","AFRM",
]


@st.cache_data(show_spinner=False, ttl=86400)
def _load_tickers() -> list[str]:
    try:
        import pandas as pd
        sp500 = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]["Symbol"].tolist()
        return sorted(set(sp500 + _POPULAR_TICKERS))
    except Exception:
        return sorted(_POPULAR_TICKERS)


_WORKING_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
]

st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.verdict-card {
    padding: 1.5rem 2rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1rem;
}
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; }
.signal-bullish { color: #26a69a; font-weight: 600; }
.signal-bearish { color: #ef5350; font-weight: 600; }
.signal-neutral  { color: #ffa726; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

VERDICT_COLORS = {
    "Strong Buy": ("#00c853", "#e8f5e9"),
    "Buy":        ("#64dd17", "#f1f8e9"),
    "Hold":       ("#ffd600", "#fffde7"),
    "Avoid":      ("#ff1744", "#fce4ec"),
}

SIGNAL_ICON = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}


def fmt_large(n: float | None) -> str:
    if n is None:
        return "N/A"
    if n >= 1e12:
        return f"${n/1e12:.2f}T"
    if n >= 1e9:
        return f"${n/1e9:.2f}B"
    if n >= 1e6:
        return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


def score_bar(label: str, normalized: float, summary: str) -> None:
    pct = int((normalized + 100) / 2)  # convert -100..100 -> 0..100
    color = "#26a69a" if normalized > 20 else "#ef5350" if normalized < -20 else "#ffa726"
    st.markdown(f"**{label}**: {summary} ({normalized:+.0f}/100)")
    st.markdown(f"""
    <div style="background:#1e1e2e;border-radius:6px;height:10px;margin-bottom:8px;">
      <div style="width:{pct}%;background:{color};height:10px;border-radius:6px;"></div>
    </div>""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Stock Analyzer")
    st.caption("US Equity Analysis · Technical + Fundamental + AI")
    st.divider()

    _tickers = _load_tickers()
    ticker_input = st.selectbox(
        "Ticker Symbol",
        options=_tickers,
        index=_tickers.index("AAPL") if "AAPL" in _tickers else 0,
        help="Type to search. US equities only.",
    ).upper().strip()
    period = st.selectbox(
        "Analysis Period",
        options=["3mo", "6mo", "1y", "2y", "5y"],
        index=2,
        help="Historical price window for technical analysis",
    )
    _default_key = _get_api_key_default()
    api_key = st.text_input(
        "Gemini API Key",
        value=_default_key,
        type="password",
        placeholder="AIza… (or set GEMINI_API_KEY env var)",
        help="Required for AI-powered recommendation. Set via env var, .env file, or .streamlit/secrets.toml to avoid re-entering.",
    )
    gemini_model = st.selectbox(
        "Gemini Model",
        options=_WORKING_MODELS,
        index=0,
        help="Only models verified to work with this API key are shown.",
    )
    st.divider()
    run_btn = st.button("Analyze", type="primary", use_container_width=True)

    st.caption("Data via Yahoo Finance · AI via Gemini")


# ── Main ───────────────────────────────────────────────────────────────────────
st.title("Stock Analysis Report")

if not run_btn:
    st.info("Enter a ticker symbol in the sidebar and click **Analyze** to generate a report.")
    st.stop()

# 1. Fetch data
with st.spinner(f"Fetching data for **{ticker_input}**…"):
    try:
        fetcher = StockFetcher(ticker_input)
        if not fetcher.is_valid():
            st.error(f"Could not find ticker **{ticker_input}**. Check the symbol and try again.")
            st.stop()

        info = fetcher.info
        df = fetcher.get_price_history(period)
        financials = fetcher.get_income_statement()
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

if df.empty or len(df) < 20:
    st.error("Not enough price history to run analysis. Try a longer period.")
    st.stop()

# 2. Header metrics
name = info.get("shortName") or info.get("longName") or ticker_input
sector = info.get("sector", "N/A")
industry = info.get("industry", "N/A")
current_price = info.get("currentPrice") or info.get("regularMarketPrice") or float(df["Close"].iloc[-1])
prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or float(df["Close"].iloc[-2])
change_abs = current_price - prev_close
change_pct = (change_abs / prev_close * 100) if prev_close else 0
market_cap = info.get("marketCap")
week52_high = info.get("fiftyTwoWeekHigh")
week52_low = info.get("fiftyTwoWeekLow")

st.subheader(f"{name} ({ticker_input})")
st.caption(f"{sector} · {industry}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Price", f"${current_price:.2f}", f"{change_pct:+.2f}% (${change_abs:+.2f})")
col2.metric("Market Cap", fmt_large(market_cap))
col3.metric("52W High", f"${week52_high:.2f}" if week52_high else "N/A")
col4.metric("52W Low", f"${week52_low:.2f}" if week52_low else "N/A")
col5.metric("Period", period.upper())

st.divider()

# 3. Run analysis
with st.spinner("Running technical & fundamental analysis…"):
    technical = analyze_technical(df)
    fundamental = analyze_fundamental(info, financials)

# 4. Tabs
tab_chart, tab_tech, tab_fund, tab_rec = st.tabs([
    "Chart", "Technical Analysis", "Fundamental Analysis", "AI Recommendation"
])

# ── Chart tab ─────────────────────────────────────────────────────────────────
with tab_chart:
    fig = candlestick_chart(df, ticker_input)
    st.plotly_chart(fig, use_container_width=True)

# ── Technical tab ─────────────────────────────────────────────────────────────
with tab_tech:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Overall Signal")
        score_bar("Technical Score", technical.normalized, technical.summary)
        st.metric("Raw Score", f"{technical.score:+d} / {technical.max_score}", help="Sum of individual signal scores")

    with col_right:
        st.subheader("Signal Breakdown")
        rows = []
        for s in technical.signals:
            rows.append({
                "Signal": s.name,
                "": SIGNAL_ICON.get(s.signal, "⚪"),
                "Assessment": s.signal.capitalize(),
                "Score": f"{s.score:+d}",
                "Detail": s.detail,
            })
        if rows:
            import pandas as pd
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "": st.column_config.TextColumn(width="small"),
                    "Score": st.column_config.TextColumn(width="small"),
                },
            )

# ── Fundamental tab ───────────────────────────────────────────────────────────
with tab_fund:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Overall Signal")
        score_bar("Fundamental Score", fundamental.normalized, fundamental.summary)
        st.metric("Raw Score", f"{fundamental.score:+d} / {fundamental.max_score}")

        # Key stats snapshot
        st.divider()
        st.caption("**Key Stats**")
        kv = {
            "EPS (TTM)": info.get("trailingEps"),
            "Revenue (TTM)": fmt_large(info.get("totalRevenue")),
            "Free Cash Flow": fmt_large(info.get("freeCashflow")),
            "Dividend Yield": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get("dividendYield") else "N/A",
            "Beta": f"{info.get('beta', 'N/A'):.2f}" if info.get("beta") else "N/A",
        }
        for k, v in kv.items():
            st.markdown(f"**{k}:** {v or 'N/A'}")

    with col_right:
        st.subheader("Metric Breakdown")
        rows = []
        for m in fundamental.metrics:
            rows.append({
                "Metric": m.name,
                "": SIGNAL_ICON.get(m.signal, "⚪"),
                "Value": m.value,
                "Assessment": m.signal.capitalize(),
                "Score": f"{m.score:+d}",
                "Detail": m.detail,
            })
        if rows:
            import pandas as pd
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "": st.column_config.TextColumn(width="small"),
                    "Score": st.column_config.TextColumn(width="small"),
                    "Value": st.column_config.TextColumn(width="small"),
                },
            )

# ── Recommendation tab ────────────────────────────────────────────────────────
with tab_rec:
    if not api_key:
        st.warning("Enter your Gemini API key in the sidebar to get an AI-generated recommendation.")
        st.info(f"**Score-based verdict:** Combined score {technical.normalized*0.4 + fundamental.normalized*0.6:+.1f}/100")
        st.stop()

    with st.spinner("Generating AI recommendation…"):
        try:
            rec = get_recommendation(ticker_input, name, technical, fundamental, api_key, gemini_model)
        except Exception as e:
            st.error(f"AI recommendation failed: {e}")
            st.stop()

    # Verdict card
    text_color, bg_color = VERDICT_COLORS.get(rec.verdict, ("#ffffff", "#333333"))
    st.markdown(f"""
    <div class="verdict-card" style="background:{bg_color};border:2px solid {text_color};">
      <p style="font-size:0.9rem;margin:0;color:#555;">Combined Score: {rec.combined_score:+.1f} / 100</p>
      <h1 style="color:{text_color};margin:0.3rem 0;font-size:3rem;">{rec.verdict}</h1>
      <p style="font-size:0.85rem;margin:0;color:#777;">Confidence: {rec.confidence}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Score bars
    col_l, col_r = st.columns(2)
    with col_l:
        score_bar("Technical", technical.normalized, technical.summary)
    with col_r:
        score_bar("Fundamental", fundamental.normalized, fundamental.summary)

    st.divider()

    # AI narrative
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Technical Picture**")
        st.write(rec.technical_summary)
        st.markdown("**Fundamental Picture**")
        st.write(rec.fundamental_summary)

    with col_b:
        st.markdown("**Investment Rationale**")
        st.write(rec.rationale)
        st.markdown("**Key Risks**")
        st.warning(rec.risks)

    st.caption("This is not financial advice. Always do your own research before investing.")
