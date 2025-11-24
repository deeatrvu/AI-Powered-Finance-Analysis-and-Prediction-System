import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta


class MarketDataLoader: 
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a given symbol.

        Args:
            symbol: Stock symbol to fetch data for

        Returns:
            DataFrame containing stock data or None if error
        """
        try:
            # Get data from Yahoo Finance
            stock = yf.Ticker(symbol)
            data = stock.history(
                period=self.config["market"]["period"],
                interval=self.config["market"]["interval"],
            )

            if data.empty:
                self.logger.error(f"No data found for symbol: {symbol}") # creates a logger that knows what part of your code it's logging from.
                return None

            # Add technical indicators
            data = self._add_technical_indicators(data)

            # Validate data
            if self.config["market"]["data_validation"]:
                data = self._validate_data(data, symbol)

            return data

        except Exception as e:
            self.logger.error(f"Error fetching stock data: {str(e)}")
            return None

    def _validate_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean the stock data.

        Args:
            data: DataFrame containing stock data
            symbol: Stock symbol

        Returns:
            Cleaned DataFrame
        """
        try:
            # Check for missing values
            if data.isna().any().any():
                self.logger.warning(f"Missing values found in {symbol} data. Cleaning...")
                data = data.ffill().bfill()

            # Check for zero or negative values in price columns
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if (data[col] <= 0).any():
                    self.logger.warning(f"Invalid price values found in {symbol} {col}. Cleaning...")
                    data[col] = data[col].replace(0, pd.NA)
                    data[col] = data[col].ffill()

            # Check for zero or negative values in volume
            if (data['Volume'] <= 0).any():
                self.logger.warning(f"Invalid volume values found in {symbol}. Cleaning...")
                data['Volume'] = data['Volume'].replace(0, pd.NA)
                data['Volume'] = data['Volume'].ffill()

            return data

        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            return data

    def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch company information for a given symbol.

        Args:
            symbol: Stock symbol to fetch info for

        Returns:
            Dictionary containing company information or None if error
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            # Extract relevant information
            company_info = {
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
            }

            return company_info

        except Exception as e:
            self.logger.error(f"Error fetching company info: {str(e)}")
            return None

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the stock data.

        Args:
            data: DataFrame containing stock data

        Returns:
            DataFrame with added technical indicators
        """
        try:
            # Calculate moving averages
            data["SMA_20"] = data["Close"].rolling(window=20).mean()
            data["SMA_50"] = data["Close"].rolling(window=50).mean()
            data["SMA_200"] = data["Close"].rolling(window=200).mean()

            # Calculate RSI
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data["RSI"] = 100 - (100 / (1 + rs))

            # Calculate MACD
            exp1 = data["Close"].ewm(span=12, adjust=False).mean()
            exp2 = data["Close"].ewm(span=26, adjust=False).mean()
            data["MACD"] = exp1 - exp2
            data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

            # Calculate Bollinger Bands
            data["BB_Middle"] = data["Close"].rolling(window=20).mean()
            data["BB_Upper"] = data["BB_Middle"] + 2 * data["Close"].rolling(window=20).std()
            data["BB_Lower"] = data["BB_Middle"] - 2 * data["Close"].rolling(window=20).std()

            # Clean any NaN values from technical indicators
            data = data.ffill().bfill()

            return data

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return data

    def get_market_trend(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Analyze market trend for a given symbol.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            Dictionary containing trend analysis or None if error
        """
        try:
            data = self.get_stock_data(symbol)
            if data is None:
                return None

            # Calculate trend indicators
            current_price = data["Close"].iloc[-1]
            sma_20 = data["SMA_20"].iloc[-1]
            sma_50 = data["SMA_50"].iloc[-1]
            sma_200 = data["SMA_200"].iloc[-1]
            rsi = data["RSI"].iloc[-1]

            # Determine trend
            trend = {
                "price": current_price,
                "trend": "Neutral",
                "strength": "Moderate",
                "indicators": {
                    "above_sma20": current_price > sma_20,
                    "above_sma50": current_price > sma_50,
                    "above_sma200": current_price > sma_200,
                    "rsi": rsi,
                },
            }

            # Determine trend direction
            if (
                current_price > sma_20
                and current_price > sma_50
                and current_price > sma_200
            ):
                trend["trend"] = "Bullish"
            elif (
                current_price < sma_20
                and current_price < sma_50
                and current_price < sma_200
            ):
                trend["trend"] = "Bearish"

            # Determine trend strength
            if abs(current_price - sma_20) / sma_20 > 0.05:
                trend["strength"] = "Strong"
            elif abs(current_price - sma_20) / sma_20 < 0.02:
                trend["strength"] = "Weak"

            return trend

        except Exception as e:
            self.logger.error(f"Error analyzing market trend: {str(e)}")
            return None
