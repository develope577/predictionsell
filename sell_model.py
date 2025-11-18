import pandas as pd
import xgboost as xgb
import logging
import os

logging.basicConfig(level=logging.INFO)

def run(inputs):
    """
    Run Sell model prediction with buy_ and sell_ feature prefixes.

    inputs:
        - 'data': DataFrame prepared by orchestrator (bought + latest features, latest prefixed with 'sell_')
        - 'model_file': path to Sell XGBoost model JSON
    Returns:
        dict with 'status' and 'suggested_trades' DataFrame containing prediction results
    """

    df = inputs.get("data")
    model_file = inputs.get("model_file")

    if df is None or df.empty:
        logging.error("Input data is missing or empty")
        return {"status": "error", "message": "No input data"}

    if not model_file or not os.path.exists(model_file):
        logging.error(f"Sell model file not found: {model_file}")
        return {"status": "error", "message": "model_file missing"}

    # Load the Sell XGBoost model
    try:
        model = xgb.Booster()
        model.load_model(model_file)
        logging.info(f"Loaded Sell XGBoost model from {model_file}")
    except Exception as e:
        logging.error(f"Error loading Sell model: {e}")
        return {"status": "error", "message": "Failed to load model"}

    # Original bought columns
    BOUGHT_COLUMNS = [
        "open", "high", "low", "close", "volume",
        "macd", "macd_signal", "macd_histogram",
        "rsi", "rsi_sma",
        "ema_100", "ema_200",
        "atr", "ema_ratio",
        "macd_histogram_x_atr", "buy_sell_pressure_x_ema_ratio",
        "buy_sell_pressure", "relative_volume",
        "quote_volume_ratio", "rsi_x_relative_volume"
    ]

    # Add 'buy_' prefix to DataFrame columns if not already present
    for col in BOUGHT_COLUMNS:
        if col in df.columns:
            df[f"buy_{col}"] = df[col]

    # Add prefixes for model feature list
    BUY_COLUMNS = [f"buy_{col}" for col in BOUGHT_COLUMNS]   # bought asset features
    SELL_COLUMNS = [f"sell_{col}" for col in BOUGHT_COLUMNS] # latest/current features

    # Combine both for expected features
    EXPECTED_FEATURES = BUY_COLUMNS + SELL_COLUMNS

    # Filter df to only available columns
    available_features = [col for col in EXPECTED_FEATURES if col in df.columns]

    # Log missing features
    missing = set(EXPECTED_FEATURES) - set(df.columns)
    if missing:
        logging.warning(f"Missing expected columns: {missing}")

    # Run predictions
    try:
        dmatrix = xgb.DMatrix(df[available_features])
        df["confidence_score"] = model.predict(dmatrix)
        logging.info(f"Predictions generated for {len(df)} symbols")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"status": "error", "message": "Prediction failed"}

    # Include sell_close from latest features if present
    if "sell_close" in df.columns:
        df["sell_close"] = df["sell_close"]

    # Columns to return
    columns_to_return = ["symbol", "timestamp", "confidence_score", "open", "high", "low", "volume"]
    if "sell_close" in df.columns:
        columns_to_return.append("sell_close")

    # Keep only columns that exist in df to avoid KeyError
    columns_to_return = [col for col in columns_to_return if col in df.columns]

    df_to_return = df[columns_to_return]

    return {"status": "success", "suggested_trades": df_to_return}
