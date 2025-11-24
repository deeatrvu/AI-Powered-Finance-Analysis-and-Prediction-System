import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, List, Optional


def create_candlestick_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Create a candlestick chart with technical indicators.

    Args:
        data: DataFrame containing stock data
        symbol: Stock symbol

    Returns:
        Plotly figure object
    """
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA_20"],
            name="SMA 20",
            line=dict(color="blue", width=1),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA_50"],
            name="SMA 50",
            line=dict(color="orange", width=1),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA_200"],
            name="SMA 200",
            line=dict(color="red", width=1),
        ),
        row=1,
        col=1,
    )

    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["BB_Upper"],
            name="BB Upper",
            line=dict(color="gray", width=1, dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["BB_Lower"],
            name="BB Lower",
            line=dict(color="gray", width=1, dash="dash"),
            fill="tonexty",
        ),
        row=1,
        col=1,
    )

    # Add RSI
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["RSI"], name="RSI", line=dict(color="purple", width=1)
        ),
        row=2,
        col=1,
    )

    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Add MACD
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["MACD"], name="MACD", line=dict(color="blue", width=1)
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Signal_Line"],
            name="Signal Line",
            line=dict(color="orange", width=1),
        ),
        row=3,
        col=1,
    )

    # Update layout
    fig.update_layout(
        title=f"{symbol} Stock Price",
        yaxis_title="Price",
        yaxis2_title="RSI",
        yaxis3_title="MACD",
        xaxis_rangeslider_visible=False,
        height=800,
        template="plotly_white",
    )

    return fig


def create_sentiment_chart(sentiment_data: Dict[str, Any]) -> go.Figure:
    """
    Create a sentiment analysis chart.

    Args:
        sentiment_data: Dictionary containing sentiment analysis results

    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()

    # Add sentiment bars
    sentiments = ["Bullish", "Neutral", "Bearish"]
    counts = [
        sum(
            1
            for article in sentiment_data["article_sentiments"]
            if article["sentiment"] == "Bullish"
        ),
        sum(
            1
            for article in sentiment_data["article_sentiments"]
            if article["sentiment"] == "Neutral"
        ),
        sum(
            1
            for article in sentiment_data["article_sentiments"]
            if article["sentiment"] == "Bearish"
        ),
    ]

    colors = ["green", "gray", "red"]

    fig.add_trace(
        go.Bar(
            x=sentiments,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition="auto",
        )
    )

    # Update layout
    fig.update_layout(
        title="News Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Number of Articles",
        template="plotly_white",
    )

    return fig


def create_prediction_chart(
    historical_data: pd.DataFrame, prediction: Dict[str, Any], symbol: str
) -> go.Figure:
    """
    Create a price prediction chart.

    Args:
        historical_data: DataFrame containing historical price data
        prediction: Dictionary containing prediction results
        symbol: Stock symbol

    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()

    # Add historical price
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data["Close"],
            name="Historical Price",
            line=dict(color="blue", width=1),
        )
    )

    # Add prediction
    prediction_date = pd.Timestamp(prediction["prediction_date"])
    fig.add_trace(
        go.Scatter(
            x=[historical_data.index[-1], prediction_date],
            y=[historical_data["Close"].iloc[-1], prediction["predicted_price"]],
            name="Prediction",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    # Add confidence interval
    confidence_range = prediction["predicted_price"] * 0.1  # 10% range
    fig.add_trace(
        go.Scatter(
            x=[prediction_date, prediction_date],
            y=[
                prediction["predicted_price"] - confidence_range,
                prediction["predicted_price"] + confidence_range,
            ],
            fill="toself",
            fillcolor="rgba(255,0,0,0.1)",
            line=dict(color="rgba(255,0,0,0)"),
            name="Confidence Interval",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{symbol} Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        showlegend=True,
    )

    return fig
