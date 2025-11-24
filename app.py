import streamlit as st
import yaml
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

from src.news.collector import NewsCollector
from src.news.summarizer import NewsSummarizer
from src.market.data_loader import MarketDataLoader
from src.models.sentiment import SentimentAnalyzer
from src.models.prediction import PricePredictor
from src.utils.visualization import create_candlestick_chart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Error loading config: {str(e)}")
    st.error("Error loading configuration. Please check config.yaml file.")
    st.stop()

# Set page config
st.set_page_config(page_title="Finance Agent", page_icon="📈", layout="wide")


# Initialize components
@st.cache_resource
def init_components():
    try:
        news_collector = NewsCollector(config)
        news_summarizer = NewsSummarizer(config)
        market_loader = MarketDataLoader(config)
        sentiment_analyzer = SentimentAnalyzer(config)
        price_predictor = PricePredictor(config)
        return (
            news_collector,
            news_summarizer,
            market_loader,
            sentiment_analyzer,
            price_predictor,
        )
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        st.error("Error initializing components. Please check the logs.")
        return None, None, None, None, None


# Sidebar
st.sidebar.title("Finance Agent")
user_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL)", value="AAPL").upper().strip()

if not user_symbol:
    st.error("Please enter a valid stock symbol in the sidebar.")
    st.stop()

symbol = user_symbol

# Main content
st.title(f"📈 {symbol} Analysis")

# Market Data Section
st.header("Market Data")
try:
    market_loader = MarketDataLoader(config)
    data = market_loader.get_stock_data(symbol)

    if data is not None:
        # Create candlestick chart
        fig = create_candlestick_chart(data, symbol)
        st.plotly_chart(fig, use_container_width=True)

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
        with col2:
            daily_change = (
                (data["Close"].iloc[-1] - data["Close"].iloc[-2]) / data["Close"].iloc[-2]
            ) * 100
            st.metric("Daily Change", f"{daily_change:.2f}%")
        with col3:
            st.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
        with col4:
            st.metric("52-Week High", f"${data['High'].max():.2f}")
    else:
        st.error("Unable to fetch market data. Please try again later.")
except Exception as e:
    logger.error(f"Error in market data section: {str(e)}")
    st.error("Error displaying market data. Please check the logs.")

# News & Prediction Section
st.header("Latest News & Price Prediction")
news_collector = NewsCollector(config)
news_summarizer = NewsSummarizer(config)

col_news, col_pred = st.columns(2)

# --- Latest News (Left Column) ---
with col_news:
    st.subheader("Latest News")
    try:
        news = news_collector.get_news(symbol)
        if news:
            for article in news[:5]:  # Show top 5 news articles
                with st.expander(article["title"]):
                    st.write(f"Source: {article['source']}")
                    st.write(f"Published: {article['publishedAt']}")
                    summary = news_summarizer.summarize(article["content"])
                    st.write(summary)
                    st.write(f"[Read more]({article['url']})")
        else:
            st.warning("No news articles found for this stock in the last 7 days.")
    except Exception as e:
        logger.error(f"Error fetching or displaying news: {str(e)}")
        st.error("Error fetching or displaying news. Please check your NewsAPI key and internet connection.")

# --- Price Prediction (Right Column) ---
with col_pred:
    st.subheader("Price Prediction")
    try:
        price_predictor = PricePredictor(config)
        prediction = price_predictor.predict(symbol)
        if prediction:
            st.metric(
                "Predicted Price",
                f"${prediction['predicted_price']:.2f}",
                f"{((prediction['predicted_price'] - prediction['current_price']) / prediction['current_price'] * 100):.2f}%"
            )
            st.write(f"Current Price: ${prediction['current_price']:.2f}")
            st.write(f"Prediction Date: {prediction['prediction_date']}")
            st.write(f"Confidence: {prediction['confidence']:.2%}")
            st.write("---")
            st.write("**Model Performance:**")
            metrics = prediction['evaluation_metrics']
            st.write(f"MAE: ${metrics['mae']:.2f}")
            st.write(f"MSE: ${metrics['mse']:.2f}")
            st.write(f"RMSE: ${metrics['rmse']:.2f}")
            st.write(f"R² Score: {metrics['r2_score']:.4f}")
            st.progress(max(0.0, min(1.0, metrics['r2_score'])))
        else:
            st.error("Unable to generate price prediction. Please try again later.")
    except Exception as e:
        logger.error(f"Error in price prediction section: {str(e)}")
        st.error("Error generating price prediction. Please check the logs.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and LLMs")
