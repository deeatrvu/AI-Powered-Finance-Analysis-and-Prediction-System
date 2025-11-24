import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class PricePredictor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback_days = config["models"]["prediction"]["lookback_days"]
        self.prediction_days = config["models"]["prediction"]["prediction_days"]
        self.confidence_threshold = config["models"]["prediction"][
            "confidence_threshold"
        ]
        self.logger = logging.getLogger(__name__)
        self.model_dir = os.path.join("data", "models")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

        # Verify NumPy installation
        try:
            np.array([1, 2, 3])
        except Exception as e:
            self.logger.error(f"NumPy initialization error: {str(e)}")
            raise RuntimeError(
                "NumPy is not properly installed. Please install numpy==1.24.3"
            )

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        try:
            # Log initial data state
            self.logger.info(f"Initial data shape: {data.shape}")
            self.logger.info(f"Initial NaN counts:\n{data.isna().sum()}")

            # Handle missing values based on config
            handle_nan = self.config["models"]["prediction"]["data_cleaning"][
                "handle_nan"
            ]
            fill_method = self.config["models"]["prediction"]["data_cleaning"][
                "fill_method"
            ]

            # First, check for any completely empty columns
            empty_cols = data.columns[data.isna().all()].tolist()
            if empty_cols:
                self.logger.warning(f"Dropping completely empty columns: {empty_cols}")
                data = data.drop(columns=empty_cols)

            # Handle missing values
            if handle_nan == "drop":
                data = data.dropna()
            elif handle_nan == "fill":
                if fill_method == "ffill":
                    data = data.fillna(method="ffill").fillna(method="bfill")
                elif fill_method == "bfill":
                    data = data.fillna(method="bfill").fillna(method="ffill")
                else:
                    data = data.fillna(data.mean())
            elif handle_nan == "interpolate":
                data = (
                    data.interpolate(method="linear")
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                )

            # Ensure no NaN values remain
            if data.isna().any().any():
                self.logger.warning(
                    "NaN values still present after cleaning. Dropping remaining NaN values."
                )
                data = data.dropna()

            # Validate data after cleaning
            if data.empty:
                raise ValueError("No data remaining after cleaning")

            if len(data) < self.lookback_days + self.prediction_days:
                raise ValueError(
                    f"Insufficient data points after cleaning. Need at least {self.lookback_days + self.prediction_days} points."
                )

            # Log final data state
            self.logger.info(f"Final data shape: {data.shape}")
            self.logger.info(f"Final NaN counts:\n{data.isna().sum()}")

            return data

        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            raise

    def predict(
        self, symbol: str, data: Optional[pd.DataFrame] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            # Load or prepare data
            if data is None:
                from src.market.data_loader import MarketDataLoader

                market_loader = MarketDataLoader(self.config)
                data = market_loader.get_stock_data(symbol)

            if data is None or data.empty:
                self.logger.error(f"No data available for prediction: {symbol}")
                return None

            # Log initial data state
            self.logger.info(f"Initial data shape for {symbol}: {data.shape}")

            # Clean the data
            data = self._clean_data(data)

            if data.empty:
                self.logger.error(
                    f"No valid data remaining after cleaning for {symbol}"
                )
                return None

            # Prepare data for prediction
            X, y = self._prepare_data(data)

            if X is None or y is None:
                self.logger.error(f"Error preparing data for prediction: {symbol}")
                return None

            # Split data into train and test sets
            train_size = int(
                len(X) * self.config["models"]["prediction"]["train_test_split"]
            )
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Load or train model
            model = self._get_model(symbol)
            if model is None:
                model = self._train_model(symbol, X_train, y_train)

            if model is None:
                self.logger.error(f"Failed to get or train model for {symbol}")
                return None

            # Make prediction
            prediction = self._make_prediction(model, X[-1:])
            if prediction is None:
                self.logger.error(f"Failed to make prediction for {symbol}")
                return None

            # Calculate evaluation metrics
            y_pred = self._make_prediction(model, X_test)
            if y_pred is None:
                self.logger.error(f"Failed to make test predictions for {symbol}")
                return None

            y_test_actual = self.scaler.inverse_transform(
                np.concatenate(
                    [y_test.reshape(-1, 1), np.zeros((len(y_test), 3))], axis=1
                )
            )[:, 0]

            mae = mean_absolute_error(y_test_actual, y_pred)
            mse = mean_squared_error(y_test_actual, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_actual, y_pred)

            # Calculate confidence
            confidence = self._calculate_confidence(prediction, data)

            return {
                "predicted_price": float(prediction[0]),
                "confidence": float(confidence),
                "horizon": self.prediction_days,
                "current_price": float(data["Close"].iloc[-1]),
                "prediction_date": (
                    pd.Timestamp.now() + pd.Timedelta(days=self.prediction_days)
                ).strftime("%Y-%m-%d"),
                "evaluation_metrics": {
                    "mae": float(mae),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "r2_score": float(r2),
                },
            }

        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return None

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        try:
            # Select features
            features = ["Close", "Volume", "RSI", "MACD"]
            df = data[features].copy()

            # Log feature statistics and NaN counts
            self.logger.info(f"Feature statistics before scaling:\n{df.describe()}")
            self.logger.info(f"NaN counts before processing:\n{df.isna().sum()}")
            self.logger.info(f"Data types:\n{df.dtypes}")

            # Ensure all features are present
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                self.logger.error(f"Missing features: {missing_features}")
                return None, None

            # Handle infinite values
            df = df.replace([np.inf, -np.inf], np.nan)

            # Fill NaN values with forward fill, then backward fill
            df = df.fillna(method="ffill").fillna(method="bfill")

            # If any NaN values remain, fill with column means
            if df.isna().any().any():
                df = df.fillna(df.mean())

            # Scale data
            try:
                scaled_data = self.scaler.fit_transform(df)
                self.logger.info(f"Scaled data shape: {scaled_data.shape}")
                self.logger.info(f"Scaled data sample:\n{scaled_data[:5]}")
            except Exception as e:
                self.logger.error(f"Error scaling data: {str(e)}")
                return None, None

            # Create sequences
            X, y = [], []
            for i in range(
                len(scaled_data) - self.lookback_days - self.prediction_days + 1
            ):
                X.append(scaled_data[i : (i + self.lookback_days)])
                y.append(
                    scaled_data[i + self.lookback_days + self.prediction_days - 1, 0]
                )

            X = np.array(X)
            y = np.array(y)

            self.logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
            self.logger.info(f"X sample:\n{X[0]}")
            self.logger.info(f"y sample:\n{y[:5]}")

            return X, y

        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            return None, None

    def _get_model(self, symbol: str) -> Optional[LSTMModel]:
        try:
            model_path = os.path.join(self.model_dir, f"{symbol}_model.pth")
            scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = LSTMModel(input_size=4, hidden_size=50, num_layers=2)
                model.load_state_dict(torch.load(model_path))
                model.to(self.device)
                self.scaler = joblib.load(scaler_path)
                return model

            return None

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None

    def _train_model(self, symbol: str, X: np.ndarray, y: np.ndarray) -> LSTMModel:
        try:
            # Convert data to PyTorch tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)

            # Create and train model
            model = LSTMModel(input_size=4, hidden_size=50, num_layers=2)
            model.to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters())

            # Training loop
            model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs.squeeze(), y_tensor)
                loss.backward()
                optimizer.step()

            # Save model and scaler
            model_path = os.path.join(self.model_dir, f"{symbol}_model.pth")
            scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")

            torch.save(model.state_dict(), model_path)
            joblib.dump(self.scaler, scaler_path)

            return model

        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

    def _make_prediction(self, model: LSTMModel, X: np.ndarray) -> np.ndarray:
        try:
            # Convert input to PyTorch tensor
            X_tensor = torch.FloatTensor(X).to(self.device)

            # Make prediction
            model.eval()
            with torch.no_grad():
                prediction = model(X_tensor)

            # Convert prediction to numpy array
            prediction = prediction.cpu().numpy()

            # Inverse transform prediction
            prediction = self.scaler.inverse_transform(
                np.concatenate([prediction, np.zeros((len(prediction), 3))], axis=1)
            )[:, 0]

            return prediction

        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise

    def _calculate_confidence(
        self, prediction: np.ndarray, data: pd.DataFrame
    ) -> float:
        try:
            # Calculate recent volatility
            recent_volatility = data["Close"].pct_change().std()

            # Calculate prediction range
            prediction_range = (
                abs(prediction[0] - data["Close"].iloc[-1]) / data["Close"].iloc[-1]
            )

            # Calculate confidence based on volatility and prediction range
            confidence = 1 - (recent_volatility * prediction_range)

            # Ensure confidence is between 0 and 1
            confidence = max(0, min(1, confidence))

            return float(confidence)

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
