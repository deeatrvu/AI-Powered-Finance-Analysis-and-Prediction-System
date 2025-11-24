# Mlops-and-LLM-
" LLM based Finance Agent - An intelligent agent utilizing Large Language Models (LLMs) for automated financial news retrieval and stock price prediction."

# ================================
#        CONFIGURATION TEMPLATE
# ================================

# -------------------------------
# API Keys (Use your own keys)
# -------------------------------
api_keys:
  newsapi: "YOUR_NEWSAPI_KEY_HERE"
  # openai: "YOUR_OPENAI_KEY_HERE"

# -------------------------------
# News Collection Settings
# -------------------------------
news:
  sources:
    - "bloomberg"
    - "reuters"
    - "financial-times"
    - "wall-street-journal"
  update_interval: 3600   # in seconds
  max_articles: 100
  fallback_sources: true  # Use alternative sources if primary fails

# -------------------------------
# Market Data Settings
# -------------------------------
market:
  symbols:
    - "AAPL"
    - "MSFT"
    - "GOOGL"
    - "AMZN"
    - "META"
  interval: "1d"
  period: "1y"
  data_validation: true

# -------------------------------
# LLM Settings
# -------------------------------
llm:
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 500

# -------------------------------
# ML Model Settings
# -------------------------------
models:
  sentiment:
    threshold: 0.6
    batch_size: 32
  
  prediction:
    lookback_days: 30
    prediction_days: 7
    confidence_threshold: 0.8
    train_test_split: 0.8

    evaluation_metrics:
      - "mae"
      - "mse"
      - "rmse"
      - "r2_score"

    data_cleaning:
      handle_nan: "drop"      # Options: drop, fill, interpolate
      fill_method: "ffill"    # Forward fill for missing values

# -------------------------------
# Application Settings
# -------------------------------
app:
  debug: true
  port: 8501
  theme: "light"

  error_handling:
    show_detailed_errors: true
    log_level: "INFO"

