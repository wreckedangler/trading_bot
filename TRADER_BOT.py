import pandas as pd
import ta
import ccxt
import time
import logging
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional

# Setup logging
logging.basicConfig(filename="trading_bot.log", level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

# Connect to Binance API
exchange = ccxt.binance({
    "apiKey": "YOUR_API_KEY",
    "secret": "YOUR_SECRET_KEY",
    "enableRateLimit": True,
})

# Set trading parameters
SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"
TRADE_AMOUNT = 0.001
STOP_LOSS_PERCENTAGE = 0.02  # 2%
TAKE_PROFIT_PERCENTAGE = 0.04  # 4%
RISK_PER_TRADE = 0.01  # 1% of the account balance
MIN_PROFIT_PERCENTAGE = 0.01  # 1% minimum profit target
SMA_PERIOD = 50  # Simple Moving Average period
TRADING_FEE = 0.001  # 0.1% expressed as a decimal

# Performance monitoring
total_trades: int = 0
winning_trades: int = 0
losing_trades: int = 0
total_pnl: float = 0

entry_signal: Optional[str] = None
position_direction: Optional[str] = None
entry_price: Optional[float] = None
position_size: Optional[float] = None
take_profit_price: Optional[float] = None

def get_market_data() -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME)
    return convert_to_dataframe(ohlcv)

def convert_to_dataframe(data: list) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["macd"], df["macd_signal"], df["macd_hist"] = ta.trend.MACD(df["close"]).macd(), ta.trend.MACD(df["close"]).macd_signal(), ta.trend.MACD(df["close"]).macd_diff()
    df["upper_band"], df["middle_band"], df["lower_band"] = ta.volatility.bollinger_hband(df["close"]), ta.volatility.bollinger_mavg(df["close"]), ta.volatility.bollinger_lband(df["close"])
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
    df["sma"] = ta.trend.SimpleMovingAverage(df["close"]).sma(SMA_PERIOD)
    return df

def get_entry_signal(df: pd.DataFrame) -> Optional[str]:
    if df["close"][-1] > df["middle_band"][-1] and df["rsi"][-1] < 30 and df["close"][-1] > df["sma"][-1]:
        if df["macd"][-1] > df["macd_signal"][-1] and df["macd_hist"][-1] > 0:
            return "buy"
    elif df["close"][-1] < df["middle_band"][-1] and df["rsi"][-1] > 70 and df["close"][-1] < df["sma"][-1]:
        if df["macd"][-1] < df["macd_signal"][-1] and df["macd_hist"][-1] < 0:
            return "sell"
    return None

def calculate_position_size(account_balance: float, risk_per_trade: float, stop_loss_percentage: float) -> float:
    risk_amount = account_balance * risk_per_trade
    position_size = risk_amount / (stop_loss_percentage * account_balance)
    return position_size

def chandelier_exit(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, multiplier: float) -> float:
    df = pd.DataFrame({"high": high, "low": low, "close": close})
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], timeperiod=period)
    upper_band = df["close"].rolling(window=period).apply(lambda x: max(x) + multiplier * df["atr"].rolling(window=period).mean(), raw=True)
    return upper_band[-1] if close[-1] > df["close"].rolling(window=period).mean() else low[-1]

def execute_simulation_trade(order_type: str, amount: float):
    global position_direction, entry_price
    if order_type == "buy":
        position_direction = "long"
        entry_price = get_latest_price()
    elif order_type == "sell":
        position_direction = "short"
        entry_price = get_latest_price()

def execute_actual_trade(order_type: str, amount: float):
    if order_type == "buy":
        order = exchange.create_market_buy_order(SYMBOL, amount)
    elif order_type == "sell":
        order = exchange.create_market_sell_order(SYMBOL, amount)
    return order

def get_latest_price() -> float:
    ticker = exchange.fetch_ticker(SYMBOL)
    return ticker["last"]

def monitor_trades(simulation_mode: bool):
    global position_direction, entry_price, position_size, total_trades, winning_trades, losing_trades, total_pnl
    take_profit_price = 0

    while True:
        try:
            df = get_market_data()
            df = calculate_indicators(df)

            if position_direction is None:
                entry_signal = get_entry_signal(df)
                balance = exchange.fetch_balance()
                account_balance = balance["free"][SYMBOL.split("/")[1]]
                position_size = calculate_position_size(account_balance, RISK_PER_TRADE, STOP_LOSS_PERCENTAGE)

                if not simulation_mode:
                    if entry_signal == "buy" and account_balance >= position_size * df["close"][-1]:
                        entry_price = df["close"][-1]
                        take_profit_price = entry_price * (1 + TAKE_PROFIT_PERCENTAGE)
                        execute_actual_trade("buy", position_size)
                        logging.info(f"Bought {position_size} at {entry_price}")
                    elif entry_signal == "sell" and account_balance >= position_size:
                        entry_price = df["close"][-1]
                        take_profit_price = entry_price * (1 - TAKE_PROFIT_PERCENTAGE)
                        execute_actual_trade("sell", position_size)
                        logging.info(f"Sold {position_size} at {entry_price}")
                else:
                    if entry_signal == "buy" and account_balance >= position_size * df["close"][-1]:
                        execute_simulation_trade("buy", position_size)
                        logging.info(f"Simulated Buy {position_size} at {df['close'][-1]}")
                    elif entry_signal == "sell" and account_balance >= position_size:
                        execute_simulation_trade("sell", position_size)
                        logging.info(f"Simulated Sell {position_size} at {df['close'][-1]}")

            else:
                current_price = get_latest_price() if simulation_mode else df["close"][-1]
                stop_loss_price = chandelier_exit(df["high"], df["low"], df["close"], period=20, multiplier=3)

                if (current_price >= take_profit_price) if position_direction == "long" else (current_price <= take_profit_price):
                    if not simulation_mode:
                        if position_direction == "long":
                            execute_actual_trade("sell", position_size)
                            logging.info(f"Sold {position_size} at {current_price} with profit")
                        elif position_direction == "short":
                            execute_actual_trade("buy", position_size)
                            logging.info(f"Bought {position_size} at {current_price} with profit")
                    else:
                        if position_direction == "long":
                            execute_simulation_trade("sell", position_size)
                            logging.info(f"Simulated Sell {position_size} at {current_price} with profit")
                        elif position_direction == "short":
                            execute_simulation_trade("buy", position_size)
                            logging.info(f"Simulated Buy {position_size} at {current_price} with profit")

                    total_trades += 1
                    total_pnl += (current_price / entry_price) - 1 if position_direction == "long" else (entry_price / current_price) - 1

                    if (current_price / entry_price) - 1 > MIN_PROFIT_PERCENTAGE:
                        winning_trades += 1
                    else:
                        losing_trades += 1

                    logging.info(f"PNL: {(current_price / entry_price) - 1:.4f}, Total PNL: {total_pnl:.4f}, Winning Trades: {winning_trades}, Losing Trades: {losing_trades}")
                    position_direction = None

            time.sleep(exchange.rateLimit / 1000)  # Wait for the rate limit to avoid API errors

        except ccxt.RequestTimeout as e:
            logging.error(f"Request timeout: {e}")
            time.sleep(60)
        except ccxt.ExchangeError as e:
            logging.error(f"Exchange error: {e}")
            time.sleep(60)
        except Exception as e:
            logging.error(f"Error: {e}")
            time.sleep(60)
            
def update_chart(frame):
    # Update the chart with the latest price
    current_price = get_latest_price()
    plt.cla()
    plt.plot([], [])  # Clear the axes
    plt.plot(df.index, df["close"], label="Price", color="blue")
    plt.scatter([df.index[-1]], [current_price], color="red", marker="o", label="Current Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Real-Time Price Chart")
    plt.legend()

def create_realtime_chart():
    global df
    df = get_market_data()
    ani = FuncAnimation(plt.gcf(), update_chart, interval=10)  # Update chart every 10 seconds
    plt.show()

if __name__ == "__main__":
    # Get user input for simulation mode
    while True:
        mode_input = input("Choose mode (1 for actual trades, 2 for simulation trades): ")
        if mode_input == "1":
            simulation_mode = False
            break
        elif mode_input == "2":
            simulation_mode = True
            break
        else:
            print("Invalid input. Please choose 1 or 2.")

    # Start monitoring trades in a separate thread
    trade_thread = threading.Thread(target=monitor_trades, args=(simulation_mode,))
    trade_thread.start()

    # Create and display the real-time chart
    create_realtime_chart()