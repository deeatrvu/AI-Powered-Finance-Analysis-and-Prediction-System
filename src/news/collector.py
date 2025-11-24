import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging


class NewsCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config["api_keys"]["newsapi"]
        self.base_url = "https://newsapi.org/v2"
        self.sources = config["news"]["sources"]
        self.logger = logging.getLogger(__name__)

    def get_news(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a given stock symbol.

        Args:
            symbol: Stock symbol to fetch news for

        Returns:
            List of news articles with title, content, source, and timestamp
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            # Prepare query
            query = f"{symbol} stock OR {symbol} shares OR {symbol} company"

            # Make API request
            response = requests.get(
                f"{self.base_url}/everything",
                params={
                    "q": query,
                    "from": start_date.strftime("%Y-%m-%d"),
                    "to": end_date.strftime("%Y-%m-%d"),
                    "language": "en",
                    "sortBy": "publishedAt",
                    "apiKey": self.api_key,
                },
            )

            if response.status_code != 200:
                self.logger.error(f"News API request failed: {response.status_code}")
                return []

            data = response.json()

            if data["status"] != "ok":
                self.logger.error(f"News API returned error: {data['message']}")
                return []

            # Process and return articles
            articles = []
            for article in data["articles"]:
                processed_article = {
                    "title": article["title"],
                    "content": article["description"] or article["content"],
                    "source": article["source"]["name"],
                    "url": article["url"],
                    "publishedAt": article["publishedAt"],
                    "sentiment": None,  # Will be filled by sentiment analyzer
                }
                articles.append(processed_article)

            return articles[: self.config["news"]["max_articles"]]

        except Exception as e:
            self.logger.error(f"Error fetching news: {str(e)}")
            return []

    def get_company_news(self, company_name: str) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a company by name.

        Args:
            company_name: Name of the company

        Returns:
            List of news articles
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            # Prepare query
            query = f'"{company_name}" company OR "{company_name}" corporation'

            # Make API request
            response = requests.get(
                f"{self.base_url}/everything",
                params={
                    "q": query,
                    "from": start_date.strftime("%Y-%m-%d"),
                    "to": end_date.strftime("%Y-%m-%d"),
                    "language": "en",
                    "sortBy": "publishedAt",
                    "apiKey": self.api_key,
                },
            )

            if response.status_code != 200:
                self.logger.error(f"News API request failed: {response.status_code}")
                return []

            data = response.json()

            if data["status"] != "ok":
                self.logger.error(f"News API returned error: {data['message']}")
                return []

            # Process and return articles
            articles = []
            for article in data["articles"]:
                processed_article = {
                    "title": article["title"],
                    "content": article["description"] or article["content"],
                    "source": article["source"]["name"],
                    "url": article["url"],
                    "publishedAt": article["publishedAt"],
                    "sentiment": None,
                }
                articles.append(processed_article)

            return articles[: self.config["news"]["max_articles"]]

        except Exception as e:
            self.logger.error(f"Error fetching company news: {str(e)}")
            return []
