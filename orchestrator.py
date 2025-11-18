import logging
import pandas as pd
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from sell_model import run as sell_model_run  # your Sell model logic
from datetime import datetime
from bson import ObjectId  # for proper buyid handling

logging.basicConfig(level=logging.INFO)

# Define the columns to pass to the model for bought and current row
BOUGHT_COLUMNS = [
    "open", "high", "low", "close", "volume", "quote_asset_volume",
    "number_of_trades", "taker_buy_base", "taker_buy_quote", "macd",
    "macd_signal", "macd_histogram", "rsi", "rsi_sma", "ema_100",
    "ema_200", "atr", "relative_volume", "quote_volume_ratio",
    "buy_sell_pressure", "ema_ratio", "rsi_x_relative_volume",
    "macd_histogram_x_atr", "buy_sell_pressure_x_ema_ratio"
]

SELL_COLUMNS = [f"sell_{col}" for col in BOUGHT_COLUMNS]

# Minimum sell score threshold
MIN_SELL_SCORE = 0.7  # adjust as needed

def orchestrator_stage2_sell():
    # -------------------------- Load environment variables --------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))  # SAME FOLDER AS SCRIPT
    dotenv_path = os.path.join(base_dir, ".env")            # .env RIGHT HERE

    if not os.path.exists(dotenv_path):
        logging.error(f".env file not found at {dotenv_path}")
        return

    load_dotenv(dotenv_path)

    # -------------------------- Inputs from environment --------------------------
    connection_str = os.getenv("MONGO_CONN_STR")
    db_name = os.getenv("MONGO_DB_NAME")
    active_trades_collection_name = os.getenv("ACTIVE_TRADES_TABLE")
    historical_collection_name = os.getenv("MONGO_COLLECTION")
    suggested_trades_collection_name = os.getenv("SUGGESTED_TRADES_COLLECTION")
    model_file_name = os.getenv("SELL_MODEL_FILE")  # file name from .env

    if not connection_str or not db_name or not model_file_name:
        logging.error("Missing required environment variables: MONGO_CONN_STR, MONGO_DB_NAME, or SELL_MODEL_FILE")
        return

    # -------------------------- Ensure model file path --------------------------
    model_file = os.path.join(base_dir, model_file_name)
    if not os.path.exists(model_file):
        logging.error(f"Sell model file not found at {model_file}")
        return

    # -------------------------- Connect to MongoDB --------------------------
    try:
        client = MongoClient(connection_str)
        db = client[db_name]
        active_trades_col = db[active_trades_collection_name]
        historical_col = db[historical_collection_name]
        suggested_trades_col = db[suggested_trades_collection_name]

        # Ensure unique index on symbol + trade_type
        suggested_trades_col.create_index([("symbol", 1), ("trade_type", 1)], unique=True)
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}")
        return

    # -------------------------- Fetch all active trades --------------------------
    active_trades = list(active_trades_col.find({}, {"symbol": 1, "buyid": 1, "_id": 0}))
    if not active_trades:
        logging.info("No active trades found.")
        return

    # -------------------------- Process each trade --------------------------
    for trade in active_trades:
        symbol = trade['symbol']
        buyid = trade.get('buyid')

        if not buyid:
            logging.warning(f"Trade for symbol {symbol} missing buyid. Skipping.")
            continue

        logging.info(f"Processing trade for symbol={symbol}, buyid={buyid}")

        # Convert buyid to ObjectId
        try:
            buyid_obj = ObjectId(buyid)
        except Exception as e:
            logging.warning(f"Invalid buyid format for {symbol}: {buyid}. Skipping trade.")
            continue

        # Fetch the row when we bought it
        df_bought = pd.DataFrame(list(historical_col.find({"_id": buyid_obj})))
        if df_bought.empty:
            logging.warning(f"No bought row found for buyid {buyid}")
            continue

        # Fetch the latest/current row for this symbol
        latest_row_cursor = historical_col.find({"symbol": symbol}).sort("timestamp", -1).limit(1)
        df_latest = pd.DataFrame(list(latest_row_cursor))
        if df_latest.empty:
            logging.warning(f"No latest row found for symbol {symbol}")
            continue

        # --- Extract sellid (latest row _id) ---
        sellid = df_latest["_id"].iloc[0]

        # Prefix latest columns with 'sell_'
        df_latest_prefixed = df_latest.add_prefix('sell_')

        # Filter columns to only expected for model
        df_bought_filtered = df_bought[[col for col in BOUGHT_COLUMNS if col in df_bought.columns]]
        df_latest_filtered = df_latest_prefixed[[col for col in SELL_COLUMNS if col in df_latest_prefixed.columns]]

        # Merge bought row + latest row side by side
        df_combined = pd.concat([df_bought_filtered.reset_index(drop=True),
                                 df_latest_filtered.reset_index(drop=True)], axis=1)

        # -------------------------- Run Sell model --------------------------
        result = sell_model_run({"data": df_combined, "model_file": model_file})
        if result.get("status") != "success":
            logging.error(f"Sell model prediction failed for {symbol}: {result.get('message')}")
            continue

        df_suggested = result.get("suggested_trades")
        if df_suggested.empty:
            logging.info(f"Sell model returned no suggested trades for {symbol}.")
            continue

        # -------------------------- Postprocess predictions --------------------------
        df_suggested["trade_type"] = "SELL"
        df_suggested["buyid"] = buyid_obj  # store as ObjectId
        df_suggested["sellid"] = sellid

        # Ensure confidence score exists
        if "confidence_score" not in df_suggested.columns or df_suggested["confidence_score"].isnull().all():
            logging.warning(f"No confidence_score returned for {symbol}. Skipping save.")
            continue

        # Rename confidence_score → sell_score
        df_suggested.rename(columns={"confidence_score": "sell_score"}, inplace=True)

        # Filter by minimum sell_score
        df_suggested = df_suggested[df_suggested["sell_score"] >= MIN_SELL_SCORE]
        if df_suggested.empty:
            logging.info(f"No sell predictions above threshold for {symbol}. Skipping save.")
            continue

        # Ensure symbol exists
        if "symbol" not in df_suggested.columns:
            df_suggested["symbol"] = symbol

        # Columns to save
        columns_to_save = ["symbol", "trade_type", "sell_score", "buyid", "sellid"]
        df_to_save = df_suggested[[c for c in columns_to_save if c in df_suggested.columns]]

        # -------------------------- Save to suggested_trades collection with created_at --------------------------
        try:
            for record in df_to_save.to_dict("records"):
                now = datetime.utcnow()
                suggested_trades_col.update_one(
                    {"symbol": record["symbol"], "trade_type": record["trade_type"]},
                    {"$set": {**record, "created_at": now}},  # overwrite created_at every run
                    upsert=True
                )
            logging.info(f"Saved {len(df_to_save)} sell predictions for {symbol} (upserted) to {suggested_trades_collection_name}")
        except Exception as e:
            logging.error(f"Error saving suggested trades for {symbol}: {e}")

    logging.info("Stage 2 Sell orchestrator completed successfully.")


if __name__ == "__main__":
    orchestrator_stage2_sell()
