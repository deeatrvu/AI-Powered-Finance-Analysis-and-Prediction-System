from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Dict, Any
import logging
import torch


class NewsSummarizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"]["max_tokens"]
        self.logger = logging.getLogger(__name__)

        try:
            # Initialize models with PyTorch backend
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/bart-large-cnn"
            )

            # Move model to GPU if available
            self.device = 0 if torch.cuda.is_available() else -1
            if self.device == 0:
                self.model = self.model.cuda()

            self.logger.info("Successfully initialized local AI models")

        except Exception as e:
            self.logger.error(f"Error initializing AI models: {str(e)}")
            self.model = None
            self.tokenizer = None

    def summarize(self, text: str) -> str:
        """
        Summarize a news article using local AI model.

        Args:
            text: The article text to summarize

        Returns:
            Summarized text
        """
        try:
            if self.model is None or self.tokenizer is None:
                return "AI model not available. Please check installation."

            # Truncate text if too long (BART has token limits)
            max_input_length = 1000  # Adjust based on model capacity
            if len(text) > max_input_length:
                text = text[:max_input_length] + "..."
                self.logger.warning("Text truncated due to length limits")

            # Tokenize and generate summary
            inputs = self.tokenizer(
                text, max_length=1024, truncation=True, return_tensors="pt"
            )
            if self.device == 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=min(self.max_tokens, 150),
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary.strip()

        except Exception as e:
            self.logger.error(f"Error summarizing text: {str(e)}")
            return "Error generating summary."

    def analyze_impact(self, text: str) -> Dict[str, Any]:
        """
        Analyze the potential market impact of a news article using sentiment analysis.

        Args:
            text: The article text to analyze

        Returns:
            Dictionary containing impact analysis
        """
        try:
            # Simple sentiment analysis based on keywords
            positive_words = [
                "growth",
                "increase",
                "profit",
                "success",
                "positive",
                "gain",
                "upgrade",
            ]
            negative_words = [
                "loss",
                "decline",
                "decrease",
                "negative",
                "downgrade",
                "risk",
                "concern",
            ]

            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            # Calculate sentiment score
            total_words = positive_count + negative_count
            if total_words == 0:
                sentiment = "neutral"
                confidence = 0.5
            else:
                sentiment_score = (positive_count - negative_count) / total_words
                if sentiment_score > 0.2:
                    sentiment = "positive"
                elif sentiment_score < -0.2:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                confidence = abs(sentiment_score)

            # Map sentiment to risk levels
            risk_mapping = {"positive": "Low", "negative": "High", "neutral": "Medium"}

            impact_mapping = {
                "positive": "Positive market sentiment detected. Potential for stock price increases in short to medium term.",
                "negative": "Negative market sentiment detected. Risk of stock price decline and increased volatility.",
                "neutral": "Neutral market sentiment. Limited immediate impact expected, monitor for developing trends.",
            }

            # Create analysis
            risk_level = risk_mapping.get(sentiment, "Unknown")
            base_analysis = impact_mapping.get(
                sentiment, "Unable to determine market impact."
            )

            detailed_analysis = f"""
            Sentiment Analysis: {sentiment.upper()} (Confidence: {confidence:.2f})
            
            Short-term Impact (1-2 days): {base_analysis}
            
            Medium-term Impact (1-2 weeks): Market reaction will depend on broader economic context and company fundamentals.
            
            Key Factors: Sentiment strength ({confidence:.2f}), market volatility, sector performance.
            
            Risk Level: {risk_level}
            """

            return {
                "analysis": detailed_analysis.strip(),
                "risk_level": risk_level,
                "sentiment": sentiment,
                "confidence": float(confidence),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing impact: {str(e)}")
            return {
                "analysis": "Error generating analysis.",
                "risk_level": "Unknown",
                "sentiment": "Unknown",
                "confidence": 0.0,
            }

    def _extract_risk_level(self, analysis: str) -> str:
        """
        Extract risk level from analysis text.

        Args:
            analysis: The analysis text

        Returns:
            Risk level (Low/Medium/High)
        """
        analysis_lower = analysis.lower()
        if "high risk" in analysis_lower or "high" in analysis_lower:
            return "High"
        elif "medium risk" in analysis_lower or "medium" in analysis_lower:
            return "Medium"
        elif "low risk" in analysis_lower or "low" in analysis_lower:
            return "Low"
        else:
            return "Unknown"
