import openai
from typing import Dict, Any, List
import logging
import numpy as np


class SentimentAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config["api_keys"]["openai"]
        self.model = config["llm"]["model"]
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"]["max_tokens"]
        self.threshold = config["models"]["sentiment"]["threshold"]
        self.logger = logging.getLogger(__name__)

        # Set OpenAI API key
        openai.api_key = self.api_key

    def analyze_news(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment of a list of news articles.

        Args:
            articles: List of news articles

        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            if not articles:
                return {
                    "overall_sentiment": "Neutral",
                    "confidence": 0.0,
                    "article_sentiments": [],
                }

            # Analyze each article
            article_sentiments = []
            for article in articles:
                sentiment = self._analyze_single_article(article)
                article_sentiments.append(sentiment)

            # Calculate overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(article_sentiments)

            return {
                "overall_sentiment": overall_sentiment["sentiment"],
                "confidence": overall_sentiment["confidence"],
                "article_sentiments": article_sentiments,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {str(e)}")
            return {
                "overall_sentiment": "Error",
                "confidence": 0.0,
                "article_sentiments": [],
            }

    def _analyze_single_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment of a single news article.

        Args:
            article: News article dictionary

        Returns:
            Dictionary containing sentiment analysis for the article
        """
        try:
            # Prepare the prompt
            prompt = f"""Analyze the sentiment of the following financial news article.
            Provide:
            1. Sentiment (Bullish/Bearish/Neutral)
            2. Confidence score (0-1)
            3. Key factors influencing the sentiment
            4. Market impact (Positive/Negative/Neutral)

            Article:
            Title: {article['title']}
            Content: {article['content']}

            Analysis:"""

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial sentiment analyzer. Provide detailed sentiment analysis focusing on market impact.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract and parse the analysis
            analysis = response.choices[0].message.content.strip()

            # Parse the sentiment and confidence
            sentiment_info = self._parse_sentiment_analysis(analysis)

            return {
                "title": article["title"],
                "sentiment": sentiment_info["sentiment"],
                "confidence": sentiment_info["confidence"],
                "market_impact": sentiment_info["market_impact"],
                "analysis": analysis,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing article sentiment: {str(e)}")
            return {
                "title": article["title"],
                "sentiment": "Neutral",
                "confidence": 0.0,
                "market_impact": "Neutral",
                "analysis": "Error analyzing sentiment",
            }

    def _parse_sentiment_analysis(self, analysis: str) -> Dict[str, Any]:
        """
        Parse sentiment analysis text to extract structured information.

        Args:
            analysis: Sentiment analysis text

        Returns:
            Dictionary containing parsed sentiment information
        """
        try:
            # Extract sentiment
            sentiment = "Neutral"
            if "bullish" in analysis.lower():
                sentiment = "Bullish"
            elif "bearish" in analysis.lower():
                sentiment = "Bearish"

            # Extract confidence
            confidence = 0.5  # Default
            if "confidence score" in analysis.lower():
                try:
                    confidence_str = (
                        analysis.lower().split("confidence score")[1].split()[0]
                    )
                    confidence = float(confidence_str)
                except:
                    pass

            # Extract market impact
            market_impact = "Neutral"
            if "positive" in analysis.lower():
                market_impact = "Positive"
            elif "negative" in analysis.lower():
                market_impact = "Negative"

            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "market_impact": market_impact,
            }

        except Exception as e:
            self.logger.error(f"Error parsing sentiment analysis: {str(e)}")
            return {
                "sentiment": "Neutral",
                "confidence": 0.5,
                "market_impact": "Neutral",
            }

    def _calculate_overall_sentiment(
        self, article_sentiments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate overall sentiment from individual article sentiments.

        Args:
            article_sentiments: List of article sentiment dictionaries

        Returns:
            Dictionary containing overall sentiment and confidence
        """
        try:
            if not article_sentiments:
                return {"sentiment": "Neutral", "confidence": 0.0}

            # Calculate weighted sentiment scores
            sentiment_scores = []
            for article in article_sentiments:
                score = 0
                if article["sentiment"] == "Bullish":
                    score = 1
                elif article["sentiment"] == "Bearish":
                    score = -1
                sentiment_scores.append(score * article["confidence"])

            # Calculate overall sentiment
            overall_score = np.mean(sentiment_scores)
            confidence = np.mean(
                [article["confidence"] for article in article_sentiments]
            )

            # Determine sentiment
            if overall_score > self.threshold:
                sentiment = "Bullish"
            elif overall_score < -self.threshold:
                sentiment = "Bearish"
            else:
                sentiment = "Neutral"

            return {"sentiment": sentiment, "confidence": confidence}

        except Exception as e:
            self.logger.error(f"Error calculating overall sentiment: {str(e)}")
            return {"sentiment": "Neutral", "confidence": 0.0}
