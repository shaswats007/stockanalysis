import pandas as pd
from dataclasses import dataclass, field


@dataclass
class FundamentalMetric:
    name: str
    value: str
    signal: str  # "bullish", "bearish", "neutral"
    score: int   # -2 to +2
    detail: str


@dataclass
class FundamentalAnalysis:
    metrics: list[FundamentalMetric]
    score: int
    max_score: int
    normalized: float  # -100 to 100
    summary: str
    raw: dict = field(default_factory=dict)


def _get(info: dict, *keys, default=None):
    for key in keys:
        val = info.get(key)
        if val is not None:
            return val
    return default


def analyze_fundamental(info: dict, financials: pd.DataFrame) -> FundamentalAnalysis:
    metrics: list[FundamentalMetric] = []

    # Valuation: Trailing P/E
    pe = _get(info, "trailingPE")
    if pe and pe > 0:
        if pe < 15:
            sig, score = "bullish", 2
            detail = f"P/E {pe:.1f}x — potentially undervalued vs market (~20x)"
        elif pe < 25:
            sig, score = "neutral", 0
            detail = f"P/E {pe:.1f}x — fairly valued"
        elif pe < 40:
            sig, score = "bearish", -1
            detail = f"P/E {pe:.1f}x — premium valuation, growth expected"
        else:
            sig, score = "bearish", -2
            detail = f"P/E {pe:.1f}x — very expensive, high growth priced in"
        metrics.append(FundamentalMetric("Trailing P/E", f"{pe:.1f}x", sig, score, detail))

    # Valuation: Forward P/E
    fwd_pe = _get(info, "forwardPE")
    if fwd_pe and fwd_pe > 0:
        if fwd_pe < 15:
            sig, score = "bullish", 1
        elif fwd_pe < 25:
            sig, score = "neutral", 0
        else:
            sig, score = "bearish", -1
        pe_expansion = ""
        if pe and fwd_pe:
            pe_expansion = f" (PE expanding)" if fwd_pe > pe else f" (PE contracting)"
        metrics.append(FundamentalMetric("Forward P/E", f"{fwd_pe:.1f}x", sig, score,
            f"Forward P/E {fwd_pe:.1f}x{pe_expansion}"))

    # Valuation: Price/Book
    pb = _get(info, "priceToBook")
    if pb is not None:
        if pb < 1:
            sig, score = "bullish", 2
            detail = f"P/B {pb:.2f}x — trading below book value"
        elif pb < 3:
            sig, score = "neutral", 0
            detail = f"P/B {pb:.2f}x — reasonable premium to book"
        elif pb < 6:
            sig, score = "bearish", -1
            detail = f"P/B {pb:.2f}x — significant premium to book"
        else:
            sig, score = "bearish", -2
            detail = f"P/B {pb:.2f}x — very high premium, intangible-heavy"
        metrics.append(FundamentalMetric("Price/Book", f"{pb:.2f}x", sig, score, detail))

    # Growth: Revenue YoY
    rev_growth = _get(info, "revenueGrowth")
    if rev_growth is not None:
        pct = rev_growth * 100
        if pct > 20:
            sig, score = "bullish", 2
        elif pct > 10:
            sig, score = "bullish", 1
        elif pct > 0:
            sig, score = "neutral", 0
        elif pct > -10:
            sig, score = "bearish", -1
        else:
            sig, score = "bearish", -2
        metrics.append(FundamentalMetric("Revenue Growth (YoY)", f"{pct:+.1f}%", sig, score,
            f"Revenue growing at {pct:.1f}% year-over-year"))

    # Growth: EPS YoY
    eps_growth = _get(info, "earningsGrowth")
    if eps_growth is not None:
        pct = eps_growth * 100
        if pct > 20:
            sig, score = "bullish", 2
        elif pct > 10:
            sig, score = "bullish", 1
        elif pct > 0:
            sig, score = "neutral", 0
        elif pct > -10:
            sig, score = "bearish", -1
        else:
            sig, score = "bearish", -2
        metrics.append(FundamentalMetric("EPS Growth (YoY)", f"{pct:+.1f}%", sig, score,
            f"Earnings growing at {pct:.1f}% year-over-year"))

    # Profitability: Net Margin
    margin = _get(info, "profitMargins")
    if margin is not None:
        pct = margin * 100
        if pct > 20:
            sig, score = "bullish", 2
            detail = f"Net margin {pct:.1f}% — highly profitable"
        elif pct > 10:
            sig, score = "bullish", 1
            detail = f"Net margin {pct:.1f}% — healthy profitability"
        elif pct > 0:
            sig, score = "neutral", 0
            detail = f"Net margin {pct:.1f}% — slim margins"
        else:
            sig, score = "bearish", -2
            detail = f"Net margin {pct:.1f}% — unprofitable"
        metrics.append(FundamentalMetric("Net Profit Margin", f"{pct:.1f}%", sig, score, detail))

    # Profitability: ROE
    roe = _get(info, "returnOnEquity")
    if roe is not None:
        pct = roe * 100
        if pct > 20:
            sig, score = "bullish", 2
            detail = f"ROE {pct:.1f}% — excellent capital efficiency"
        elif pct > 10:
            sig, score = "bullish", 1
            detail = f"ROE {pct:.1f}% — good capital returns"
        elif pct > 0:
            sig, score = "neutral", 0
            detail = f"ROE {pct:.1f}% — modest returns on equity"
        else:
            sig, score = "bearish", -2
            detail = f"ROE {pct:.1f}% — destroying shareholder value"
        metrics.append(FundamentalMetric("Return on Equity", f"{pct:.1f}%", sig, score, detail))

    # Balance Sheet: Debt/Equity
    de = _get(info, "debtToEquity")
    if de is not None:
        # yfinance returns D/E * 100
        de_ratio = de / 100
        if de_ratio < 0.3:
            sig, score = "bullish", 1
            detail = f"D/E {de_ratio:.2f}x — low leverage, strong balance sheet"
        elif de_ratio < 1.0:
            sig, score = "neutral", 0
            detail = f"D/E {de_ratio:.2f}x — manageable leverage"
        elif de_ratio < 2.0:
            sig, score = "bearish", -1
            detail = f"D/E {de_ratio:.2f}x — elevated leverage"
        else:
            sig, score = "bearish", -2
            detail = f"D/E {de_ratio:.2f}x — high leverage, financial risk"
        metrics.append(FundamentalMetric("Debt/Equity", f"{de_ratio:.2f}x", sig, score, detail))

    # Balance Sheet: Current Ratio
    cr = _get(info, "currentRatio")
    if cr is not None:
        if cr > 2:
            sig, score = "bullish", 1
            detail = f"Current ratio {cr:.2f}x — strong short-term liquidity"
        elif cr > 1:
            sig, score = "neutral", 0
            detail = f"Current ratio {cr:.2f}x — adequate liquidity"
        else:
            sig, score = "bearish", -1
            detail = f"Current ratio {cr:.2f}x — potential liquidity risk"
        metrics.append(FundamentalMetric("Current Ratio", f"{cr:.2f}x", sig, score, detail))

    # Analyst Consensus
    rec_key = _get(info, "recommendationKey", default="")
    if rec_key:
        rec_key = rec_key.lower()
        rec_map = {
            "strong_buy": ("bullish", 2),
            "buy": ("bullish", 1),
            "hold": ("neutral", 0),
            "underperform": ("bearish", -1),
            "sell": ("bearish", -1),
            "strong_sell": ("bearish", -2),
        }
        if rec_key in rec_map:
            sig, score = rec_map[rec_key]
            target = _get(info, "targetMeanPrice")
            current = _get(info, "currentPrice", "regularMarketPrice")
            upside_str = ""
            if target and current:
                upside = ((target - current) / current) * 100
                upside_str = f" | Target ${target:.2f} ({upside:+.1f}% upside)"
            num_analysts = _get(info, "numberOfAnalystOpinions", default=0)
            detail = f"Consensus: {rec_key.replace('_', ' ').title()}{upside_str} ({num_analysts} analysts)"
            metrics.append(FundamentalMetric(
                "Analyst Consensus",
                rec_key.replace("_", " ").title(),
                sig, score, detail,
            ))

    total_score = sum(m.score for m in metrics)
    max_possible = len(metrics) * 2
    normalized = (total_score / max_possible * 100) if max_possible > 0 else 0.0

    if normalized > 50:
        summary = "Strong Fundamentals"
    elif normalized > 20:
        summary = "Good Fundamentals"
    elif normalized > -20:
        summary = "Mixed Fundamentals"
    elif normalized > -50:
        summary = "Weak Fundamentals"
    else:
        summary = "Poor Fundamentals"

    return FundamentalAnalysis(
        metrics=metrics,
        score=total_score,
        max_score=max_possible,
        normalized=normalized,
        summary=summary,
        raw=info,
    )
