import pandas as pd
import pandas_ta as ta
import numpy as np
from dataclasses import dataclass


@dataclass
class TechnicalSignal:
    name: str
    value: float | None
    signal: str  # "bullish", "bearish", "neutral"
    score: int   # -2 to +2
    detail: str


@dataclass
class TechnicalAnalysis:
    signals: list[TechnicalSignal]
    score: int
    max_score: int
    normalized: float  # -100 to 100
    summary: str


def analyze_technical(df: pd.DataFrame) -> TechnicalAnalysis:
    signals: list[TechnicalSignal] = []
    close = df["Close"]
    current_price = float(close.iloc[-1])

    # RSI
    try:
        rsi_series = ta.rsi(close, length=14)
        if rsi_series is not None and not rsi_series.empty:
            rsi_val = float(rsi_series.iloc[-1])
            if rsi_val < 30:
                sig, score = "bullish", 2
                detail = f"RSI {rsi_val:.1f} — oversold, potential reversal"
            elif rsi_val > 70:
                sig, score = "bearish", -2
                detail = f"RSI {rsi_val:.1f} — overbought, potential pullback"
            elif rsi_val < 45:
                sig, score = "neutral", 0
                detail = f"RSI {rsi_val:.1f} — leaning weak"
            elif rsi_val > 55:
                sig, score = "neutral", 0
                detail = f"RSI {rsi_val:.1f} — leaning strong"
            else:
                sig, score = "neutral", 0
                detail = f"RSI {rsi_val:.1f} — neutral zone"
            signals.append(TechnicalSignal("RSI (14)", rsi_val, sig, score, detail))
    except Exception:
        pass

    # MACD
    try:
        macd_df = ta.macd(close)
        if macd_df is not None and not macd_df.empty:
            macd_line = float(macd_df["MACD_12_26_9"].iloc[-1])
            signal_line = float(macd_df["MACDs_12_26_9"].iloc[-1])
            hist = float(macd_df["MACDh_12_26_9"].iloc[-1])
            prev_hist = float(macd_df["MACDh_12_26_9"].iloc[-2]) if len(macd_df) > 1 else 0.0

            if hist > 0 and prev_hist <= 0:
                sig, score = "bullish", 2
                detail = "MACD bullish crossover — momentum turning positive"
            elif hist < 0 and prev_hist >= 0:
                sig, score = "bearish", -2
                detail = "MACD bearish crossover — momentum turning negative"
            elif macd_line > signal_line:
                sig, score = "bullish", 1
                detail = f"MACD ({macd_line:.3f}) above signal ({signal_line:.3f})"
            else:
                sig, score = "bearish", -1
                detail = f"MACD ({macd_line:.3f}) below signal ({signal_line:.3f})"
            signals.append(TechnicalSignal("MACD (12,26,9)", macd_line, sig, score, detail))
    except Exception:
        pass

    # Moving Averages
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200_series = close.rolling(200).mean()
    sma200 = float(sma200_series.iloc[-1]) if not pd.isna(sma200_series.iloc[-1]) else None

    if not pd.isna(sma50):
        if current_price > sma50:
            sig, score = "bullish", 1
            detail = f"Price ${current_price:.2f} above 50 SMA ${sma50:.2f} (+{((current_price/sma50)-1)*100:.1f}%)"
        else:
            sig, score = "bearish", -1
            detail = f"Price ${current_price:.2f} below 50 SMA ${sma50:.2f} ({((current_price/sma50)-1)*100:.1f}%)"
        signals.append(TechnicalSignal("Price vs 50 SMA", sma50, sig, score, detail))

    if sma200 is not None:
        if current_price > sma200:
            sig, score = "bullish", 1
            detail = f"Price ${current_price:.2f} above 200 SMA ${sma200:.2f} (+{((current_price/sma200)-1)*100:.1f}%)"
        else:
            sig, score = "bearish", -1
            detail = f"Price ${current_price:.2f} below 200 SMA ${sma200:.2f} ({((current_price/sma200)-1)*100:.1f}%)"
        signals.append(TechnicalSignal("Price vs 200 SMA", sma200, sig, score, detail))

        # Golden / Death cross
        if len(close) >= 201:
            prev_sma50 = float(close.rolling(50).mean().iloc[-2])
            prev_sma200 = float(sma200_series.iloc[-2]) if not pd.isna(sma200_series.iloc[-2]) else None
            if prev_sma200 is not None:
                if sma50 > sma200 and prev_sma50 <= prev_sma200:
                    signals.append(TechnicalSignal("Golden Cross", None, "bullish", 2,
                        "50 SMA just crossed above 200 SMA — strong long-term bullish signal"))
                elif sma50 < sma200 and prev_sma50 >= prev_sma200:
                    signals.append(TechnicalSignal("Death Cross", None, "bearish", -2,
                        "50 SMA just crossed below 200 SMA — strong long-term bearish signal"))
                elif sma50 > sma200:
                    signals.append(TechnicalSignal("MA Alignment", None, "bullish", 1,
                        "50 SMA above 200 SMA — sustained uptrend"))
                else:
                    signals.append(TechnicalSignal("MA Alignment", None, "bearish", -1,
                        "50 SMA below 200 SMA — sustained downtrend"))

    # Bollinger Bands
    try:
        bb = ta.bbands(close, length=20)
        if bb is not None and not bb.empty:
            suffix = "_2.0_2.0" if "BBU_20_2.0_2.0" in bb.columns else "_2.0"
            upper = float(bb[f"BBU_20{suffix}"].iloc[-1])
            lower = float(bb[f"BBL_20{suffix}"].iloc[-1])
            pct_b = (current_price - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
            bandwidth = (upper - lower) / float(bb[f"BBM_20{suffix}"].iloc[-1]) * 100

            if pct_b < 0.15:
                sig, score = "bullish", 1
                detail = f"Price near lower Bollinger Band (%B={pct_b:.2f}) — potential bounce"
            elif pct_b > 0.85:
                sig, score = "bearish", -1
                detail = f"Price near upper Bollinger Band (%B={pct_b:.2f}) — potential resistance"
            else:
                sig, score = "neutral", 0
                detail = f"Price within Bollinger Bands (%B={pct_b:.2f}, bandwidth={bandwidth:.1f}%)"
            signals.append(TechnicalSignal("Bollinger Bands", pct_b, sig, score, detail))
    except Exception:
        pass

    # Volume trend
    try:
        avg_vol = float(df["Volume"].rolling(20).mean().iloc[-1])
        last_vol = float(df["Volume"].iloc[-1])
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
        prev_close = float(close.iloc[-2]) if len(close) > 1 else current_price

        if vol_ratio > 1.5 and current_price >= prev_close:
            sig, score = "bullish", 1
            detail = f"High volume ({vol_ratio:.1f}x avg) on up move — conviction buying"
        elif vol_ratio > 1.5 and current_price < prev_close:
            sig, score = "bearish", -1
            detail = f"High volume ({vol_ratio:.1f}x avg) on down move — conviction selling"
        elif vol_ratio < 0.5:
            sig, score = "neutral", 0
            detail = f"Low volume ({vol_ratio:.1f}x avg) — weak conviction"
        else:
            sig, score = "neutral", 0
            detail = f"Volume {vol_ratio:.1f}x 20-day average"
        signals.append(TechnicalSignal("Volume", vol_ratio, sig, score, detail))
    except Exception:
        pass

    total_score = sum(s.score for s in signals)
    max_possible = len(signals) * 2
    normalized = (total_score / max_possible * 100) if max_possible > 0 else 0.0

    if normalized > 50:
        summary = "Bullish"
    elif normalized > 20:
        summary = "Mildly Bullish"
    elif normalized > -20:
        summary = "Neutral"
    elif normalized > -50:
        summary = "Mildly Bearish"
    else:
        summary = "Bearish"

    return TechnicalAnalysis(
        signals=signals,
        score=total_score,
        max_score=max_possible,
        normalized=normalized,
        summary=summary,
    )
