import yfinance as yf
import pandas as pd


class StockFetcher:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self._info = None

    @property
    def info(self) -> dict:
        if self._info is None:
            self._info = self.stock.info
        return self._info

    def get_price_history(self, period: str = "1y") -> pd.DataFrame:
        df = self.stock.history(period=period)
        df.index = pd.to_datetime(df.index)
        return df

    def get_income_statement(self) -> pd.DataFrame:
        try:
            return self.stock.financials
        except Exception:
            return pd.DataFrame()

    def get_balance_sheet(self) -> pd.DataFrame:
        try:
            return self.stock.balance_sheet
        except Exception:
            return pd.DataFrame()

    def get_cash_flow(self) -> pd.DataFrame:
        try:
            return self.stock.cashflow
        except Exception:
            return pd.DataFrame()

    def get_analyst_recommendations(self) -> pd.DataFrame:
        try:
            return self.stock.recommendations
        except Exception:
            return pd.DataFrame()

    def is_valid(self) -> bool:
        try:
            info = self.info
            return bool(
                info
                and (info.get("regularMarketPrice") or info.get("currentPrice"))
            )
        except Exception:
            return False
