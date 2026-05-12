import json
import time
import random
from google import genai
from google.genai import types
from dataclasses import dataclass
from ..analysis.technical import TechnicalAnalysis
from ..analysis.fundamental import FundamentalAnalysis

_RETRYABLE_CODES = {429, 500, 503}
_MAX_RETRIES = 4
_BASE_DELAY = 1.0


def _gemini_generate(client: genai.Client, model: str, contents: str, system: str) -> str:
    delay = _BASE_DELAY
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system),
            )
            return response.text.strip()
        except Exception as e:
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            # Extract HTTP status from error message if not directly available
            if code is None:
                msg = str(e)
                for c in _RETRYABLE_CODES:
                    if str(c) in msg[:10]:
                        code = c
                        break
            if code not in _RETRYABLE_CODES or attempt == _MAX_RETRIES:
                raise
            jitter = random.uniform(0, 0.3 * delay)
            time.sleep(delay + jitter)
            delay *= 2

SYSTEM_PROMPT = """You are a professional equity analyst. Given quantitative technical and fundamental \
analysis data for a US stock, provide a concise investment recommendation. Be specific, reference \
actual data points, and explain your reasoning clearly for a retail investor.

Respond ONLY with valid JSON (no markdown, no code blocks) using exactly these keys:
- "technical_summary": 1-2 sentences on the technical picture
- "fundamental_summary": 1-2 sentences on the fundamental picture
- "rationale": 2-3 sentences explaining the overall recommendation and key reasons
- "risks": 1-2 sentences on the main risks an investor should watch"""


@dataclass
class Recommendation:
    verdict: str       # "Strong Buy", "Buy", "Hold", "Avoid"
    combined_score: float  # -100 to 100
    confidence: int    # 0-100
    technical_summary: str
    fundamental_summary: str
    rationale: str
    risks: str


def _score_to_verdict(score: float) -> str:
    if score >= 55:
        return "Strong Buy"
    if score >= 25:
        return "Buy"
    if score >= -15:
        return "Hold"
    return "Avoid"


def _calc_confidence(
    combined: float,
    technical: TechnicalAnalysis,
    fundamental: FundamentalAnalysis,
) -> int:
    all_scores = [s.score for s in technical.signals] + [m.score for m in fundamental.metrics]
    if not all_scores:
        return min(100, max(0, int(abs(combined))))

    direction = 1 if combined >= 0 else -1
    agreeing = sum(1 for s in all_scores if s * direction > 0)
    agreement_ratio = agreeing / len(all_scores)      # 0..1: fraction of signals pointing same way
    score_component = min(1.0, abs(combined) / 100)   # 0..1: magnitude of combined score

    # 60% weight to signal consensus, 40% to score magnitude
    return min(100, max(0, int((agreement_ratio * 0.6 + score_component * 0.4) * 100)))


def get_recommendation(
    ticker: str,
    company_name: str,
    technical: TechnicalAnalysis,
    fundamental: FundamentalAnalysis,
    api_key: str,
    model: str = "gemini-2.5-flash-lite",
) -> Recommendation:
    # Weighted combined score: fundamentals matter more for long-term buy decisions
    combined = technical.normalized * 0.40 + fundamental.normalized * 0.60
    verdict = _score_to_verdict(combined)
    confidence = _calc_confidence(combined, technical, fundamental)

    info = fundamental.raw
    current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0) or 0

    tech_lines = "\n".join(
        f"  [{s.signal.upper()}] {s.name}: {s.detail} (score: {s.score:+d})"
        for s in technical.signals
    )
    fund_lines = "\n".join(
        f"  [{m.signal.upper()}] {m.name} = {m.value}: {m.detail} (score: {m.score:+d})"
        for m in fundamental.metrics
    )

    context = f"""
STOCK: {ticker} ({company_name})
Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}
Market Cap: ${info.get('marketCap', 0):,.0f}
Current Price: ${current_price:.2f}
52-Week High: ${info.get('fiftyTwoWeekHigh', 0):.2f} | Low: ${info.get('fiftyTwoWeekLow', 0):.2f}

TECHNICAL ANALYSIS (score: {technical.score:+d}/{technical.max_score}, normalized: {technical.normalized:+.1f}/100):
{tech_lines}
Technical Verdict: {technical.summary}

FUNDAMENTAL ANALYSIS (score: {fundamental.score:+d}/{fundamental.max_score}, normalized: {fundamental.normalized:+.1f}/100):
{fund_lines}
Fundamental Verdict: {fundamental.summary}

COMBINED SCORE: {combined:+.1f}/100
PRELIMINARY VERDICT: {verdict}
""".strip()

    client = genai.Client(api_key=api_key)
    raw_text = _gemini_generate(client, model, context, SYSTEM_PROMPT)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        try:
            parsed = json.loads(match.group()) if match else {}
        except Exception:
            parsed = {}

    return Recommendation(
        verdict=verdict,
        combined_score=combined,
        confidence=confidence,
        technical_summary=parsed.get("technical_summary", technical.summary),
        fundamental_summary=parsed.get("fundamental_summary", fundamental.summary),
        rationale=parsed.get("rationale", f"Combined score {combined:+.1f}/100 indicates {verdict}."),
        risks=parsed.get("risks", "Review individual metrics above for specific risk factors."),
    )
