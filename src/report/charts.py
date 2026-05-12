import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta


def candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    bb = ta.bbands(df["Close"], length=20)
    rsi = ta.rsi(df["Close"], length=14)
    macd_df = ta.macd(df["Close"])

    has_macd = macd_df is not None and not macd_df.empty
    rows = 4 if has_macd else 3
    row_heights = [0.55, 0.15, 0.15, 0.15] if has_macd else [0.60, 0.20, 0.20]
    subplot_titles = [f"{ticker} — Price & Indicators", "Volume", "RSI (14)"]
    if has_macd:
        subplot_titles.append("MACD (12,26,9)")

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # SMAs
    for ma, color, label in [
        ("SMA20", "#ffa726", "20 SMA"),
        ("SMA50", "#29b6f6", "50 SMA"),
        ("SMA200", "#ef5350", "200 SMA"),
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[ma], name=label,
            line=dict(color=color, width=1.2),
        ), row=1, col=1)

    # Bollinger Bands
    if bb is not None and not bb.empty:
        bbu_col = "BBU_20_2.0_2.0" if "BBU_20_2.0_2.0" in bb.columns else "BBU_20_2.0"
        bbl_col = "BBL_20_2.0_2.0" if "BBL_20_2.0_2.0" in bb.columns else "BBL_20_2.0"
        fig.add_trace(go.Scatter(
            x=df.index, y=bb[bbu_col], name="BB Upper",
            line=dict(color="rgba(150,150,150,0.6)", width=1, dash="dot"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=bb[bbl_col], name="BB Lower",
            fill="tonexty",
            fillcolor="rgba(150,150,150,0.07)",
            line=dict(color="rgba(150,150,150,0.6)", width=1, dash="dot"),
            showlegend=False,
        ), row=1, col=1)

    # Volume bars
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=vol_colors, showlegend=False,
    ), row=2, col=1)

    # RSI
    if rsi is not None and not rsi.empty:
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi, name="RSI",
            line=dict(color="#ab47bc", width=1.5), showlegend=False,
        ), row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.08)",
                      line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.08)",
                      line_width=0, row=3, col=1)
        fig.add_hline(y=70, line_color="rgba(239,83,80,0.5)",
                      line_dash="dash", row=3, col=1)
        fig.add_hline(y=30, line_color="rgba(38,166,154,0.5)",
                      line_dash="dash", row=3, col=1)

    # MACD
    if has_macd:
        hist_vals = macd_df["MACDh_12_26_9"]
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in hist_vals]
        fig.add_trace(go.Bar(
            x=df.index, y=hist_vals, name="MACD Hist",
            marker_color=hist_colors, showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=macd_df["MACD_12_26_9"], name="MACD",
            line=dict(color="#29b6f6", width=1.2), showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=macd_df["MACDs_12_26_9"], name="Signal",
            line=dict(color="#ffa726", width=1.2), showlegend=False,
        ), row=4, col=1)

    fig.update_layout(
        height=750,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=50, b=20, l=60, r=20),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")

    return fig
