import openai
from typing import Dict, Any, List, Optional
import logging
import json


class LLMInterface:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config["api_keys"]["openai"]
        self.model = config["llm"]["model"]
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"]["max_tokens"]
        self.logger = logging.getLogger(__name__)

        # Set OpenAI API key
        openai.api_key = self.api_key

    def analyze_market_conditions(
        self,
        stock_data: Dict[str, Any],
        news_data: List[Dict[str, Any]],
        sentiment_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze overall market conditions using LLM.

        Args:
            stock_data: Dictionary containing stock data
            news_data: List of news articles
            sentiment_data: Dictionary containing sentiment analysis

        Returns:
            Dictionary containing market analysis
        """
        try:
            # Prepare the prompt
            prompt = f"""Analyze the following market data and provide insights:

            Stock Data:
            - Current Price: ${stock_data['current_price']:.2f}
            - 52-Week High: ${stock_data['52_week_high']:.2f}
            - 52-Week Low: ${stock_data['52_week_low']:.2f}
            - Market Cap: ${stock_data['market_cap']:,.2f}
            - P/E Ratio: {stock_data['pe_ratio']:.2f}
            - Beta: {stock_data['beta']:.2f}

            News Sentiment:
            - Overall Sentiment: {sentiment_data['overall_sentiment']}
            - Confidence: {sentiment_data['confidence']:.2%}

            Recent News Headlines:
            {self._format_news_headlines(news_data)}

            Please provide:
            1. Market Analysis
            2. Key Factors Affecting the Stock
            3. Risk Assessment
            4. Investment Recommendation
            5. Confidence Level (0-1)

            Analysis:"""

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial market analyst. Provide detailed market analysis and investment recommendations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract and parse the analysis
            analysis = response.choices[0].message.content.strip()

            # Parse the analysis into structured format
            structured_analysis = self._parse_market_analysis(analysis)

            return structured_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")
            return {
                "analysis": "Error generating market analysis.",
                "confidence": 0.0,
                "recommendation": "Unable to provide recommendation.",
            }

    def generate_investment_report(
        self, symbol: str, market_analysis: Dict[str, Any], prediction: Dict[str, Any]
    ) -> str:
        """
        Generate a comprehensive investment report.

        Args:
            symbol: Stock symbol
            market_analysis: Dictionary containing market analysis
            prediction: Dictionary containing price prediction

        Returns:
            Formatted investment report
        """
        try:
            # Prepare the prompt
            prompt = f"""Generate a comprehensive investment report for {symbol}:

            Market Analysis:
            {json.dumps(market_analysis, indent=2)}

            Price Prediction:
            - Predicted Price: ${prediction['predicted_price']:.2f}
            - Confidence: {prediction['confidence']:.2%}
            - Time Horizon: {prediction['horizon']} days

            Please provide a well-structured report including:
            1. Executive Summary
            2. Market Analysis
            3. Technical Analysis
            4. Fundamental Analysis
            5. Risk Assessment
            6. Investment Recommendation
            7. Price Target and Time Horizon

            Report:"""

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional investment analyst. Generate detailed investment reports.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.error(f"Error generating investment report: {str(e)}")
            return "Error generating investment report."

    def _format_news_headlines(self, news_data: List[Dict[str, Any]]) -> str:
        """
        Format news headlines for the prompt.

        Args:
            news_data: List of news articles

        Returns:
            Formatted string of headlines
        """
        headlines = []
        for article in news_data[:5]:  # Use top 5 headlines
            headlines.append(f"- {article['title']} ({article['source']})")
        return "\n".join(headlines)

    def _parse_market_analysis(self, analysis: str) -> Dict[str, Any]:
        """
        Parse market analysis text into structured format.

        Args:
            analysis: Market analysis text

        Returns:
            Dictionary containing structured analysis
        """
        try:
            # Extract sections
            sections = analysis.split("\n\n")

            # Initialize structured analysis
            structured_analysis = {
                "market_analysis": "",
                "key_factors": [],
                "risk_assessment": "",
                "recommendation": "",
                "confidence": 0.5,  # Default
            }

            # Parse sections
            for section in sections:
                if "Market Analysis" in section:
                    structured_analysis["market_analysis"] = section.split(
                        "Market Analysis:"
                    )[1].strip()
                elif "Key Factors" in section:
                    factors = section.split("Key Factors:")[1].strip().split("\n")
                    structured_analysis["key_factors"] = [
                        f.strip("- ") for f in factors if f.strip()
                    ]
                elif "Risk Assessment" in section:
                    structured_analysis["risk_assessment"] = section.split(
                        "Risk Assessment:"
                    )[1].strip()
                elif "Investment Recommendation" in section:
                    structured_analysis["recommendation"] = section.split(
                        "Investment Recommendation:"
                    )[1].strip()
                elif "Confidence Level" in section:
                    try:
                        confidence_str = section.split("Confidence Level:")[1].strip()
                        structured_analysis["confidence"] = float(confidence_str)
                    except:
                        pass

            return structured_analysis

        except Exception as e:
            self.logger.error(f"Error parsing market analysis: {str(e)}")
            return {
                "market_analysis": analysis,
                "key_factors": [],
                "risk_assessment": "",
                "recommendation": "",
                "confidence": 0.5,
            }
