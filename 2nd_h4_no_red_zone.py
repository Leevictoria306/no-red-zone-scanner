import os
import requests
import pandas as pd

from datetime import (
    datetime,
    timezone
)

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed
)

# =========================================================
# CONFIG
# =========================================================

BASE_URL = "https://data-api.binance.vision"

TIMEFRAME = "4h"

MIN_GREEN_CANDLES = 3
TOP_RESULTS = 20

MAX_WORKERS = 25

VERBOSE_SCAN = False

# =========================================================
# EXECUTION WINDOW
# =========================================================

# GitHub runs every 5 mins.
# Actual scan only happens
# near H4 candle close.

EXECUTION_WINDOW_MINUTES = 7

# H4 candle closes
VALID_HOURS = [3, 7, 11, 15, 19, 23]

# =========================================================
# TELEGRAM
# =========================================================

TELEGRAM_ENABLED = True

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID"
)

# =========================================================
# STRATEGY FILTERS
# =========================================================

MIN_BODY_PERCENT = 0.7

MAX_FINAL_BODY_PERCENT = 18

MIN_LAST_EXPANSION = 1.10

# =========================================================
# SHOULD RUN?
# =========================================================

def should_run_scan():

    now = datetime.now(timezone.utc)

    for hour in VALID_HOURS:

        target_minutes = (
            hour * 60 + 50
        )

        current_minutes = (
            now.hour * 60
            + now.minute
        )

        difference = abs(
            current_minutes
            - target_minutes
        )

        if (
            difference
            <= EXECUTION_WINDOW_MINUTES
        ):
            return True

    return False

# =========================================================
# SHOULD USE LATEST CANDLE?
# =========================================================

def should_use_latest_candle(now):

    if (
        now.hour in VALID_HOURS
        and now.minute >= 50
    ):
        return True

    return False

# =========================================================
# GET PAIRS
# =========================================================

def get_usdt_pairs():

    url = f"{BASE_URL}/api/v3/exchangeInfo"

    response = requests.get(
        url,
        timeout=10
    )

    data = response.json()

    pairs = []

    for symbol in data["symbols"]:

        if (
            symbol.get("quoteAsset")
            == "USDT"

            and symbol.get("status")
            == "TRADING"

            and symbol.get(
                "isSpotTradingAllowed"
            )
        ):

            pairs.append(
                symbol["symbol"]
            )

    return pairs

# =========================================================
# GET KLINES
# =========================================================

def get_klines(symbol, limit=20):

    url = f"{BASE_URL}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "limit": limit
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        if response.status_code != 200:
            return None

        data = response.json()

        if not isinstance(data, list):
            return None

        df = pd.DataFrame(data, columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore"
        ])

        numeric_cols = [
            "open",
            "high",
            "low",
            "close"
        ]

        for col in numeric_cols:

            df[col] = df[col].astype(float)

        return df

    except:
        return None

# =========================================================
# STRATEGY
# =========================================================

def qualifies(
    df,
    use_latest_candle,
    candles=MIN_GREEN_CANDLES
):

    # =====================================================
    # CANDLE SELECTION
    # =====================================================

    if use_latest_candle:

        recent = df.tail(candles)

    else:

        recent = (
            df.iloc[:-1]
            .tail(candles)
        )

    if len(recent) < candles:
        return None

    bodies = []

    # =====================================================
    # GREEN CHECK
    # =====================================================

    for _, row in recent.iterrows():

        open_price = row["open"]
        close_price = row["close"]

        if close_price <= open_price:
            return None

        body_pct = (
            (
                close_price
                - open_price
            )
            / open_price
        ) * 100

        # reject tiny candles
        if (
            body_pct
            < MIN_BODY_PERCENT
        ):
            return None

        bodies.append(body_pct)

    # =====================================================
    # STRICT PROGRESSIVE
    # =====================================================

    progressive = all(
        bodies[i]
        < bodies[i + 1]

        for i in range(
            len(bodies) - 1
        )
    )

    if not progressive:
        return None

    # =====================================================
    # REJECT FLAT MOMENTUM
    # =====================================================

    last_expansion = (
        bodies[-1]
        / bodies[-2]
    )

    if (
        last_expansion
        < MIN_LAST_EXPANSION
    ):
        return None

    # =====================================================
    # REJECT EXHAUSTION
    # =====================================================

    final_body = bodies[-1]

    if (
        final_body
        > MAX_FINAL_BODY_PERCENT
    ):
        return None

    # =====================================================
    # MOMENTUM SCORE
    # =====================================================

    momentum_score = sum(bodies)

    # =====================================================
    # ACCELERATION BONUS
    # =====================================================

    acceleration_bonus = (
        bodies[-1]
        * (
            bodies[-1]
            / bodies[-2]
        )
    )

    # =====================================================
    # LAST CANDLE WICK ANALYSIS
    # =====================================================

    last = recent.iloc[-1]

    open_price = last["open"]
    high_price = last["high"]
    low_price = last["low"]
    close_price = last["close"]

    body_size = abs(
        close_price
        - open_price
    )

    if body_size <= 0:
        return None

    upper_wick = (
        high_price
        - close_price
    )

    lower_wick = (
        open_price
        - low_price
    )

    upper_wick_ratio = (
        upper_wick
        / body_size
    )

    lower_wick_ratio = (
        lower_wick
        / body_size
    )

    # =====================================================
    # WICK PENALTY
    # =====================================================

    wick_penalty = (
        upper_wick_ratio * 12
        + lower_wick_ratio * 3
    )

    # =====================================================
    # FINAL SCORE
    # =====================================================

    final_score = (
        momentum_score
        + acceleration_bonus
        - wick_penalty
    )

    return {
        "score": final_score,
        "momentum_score": momentum_score,
        "acceleration_bonus":
            acceleration_bonus,
        "upper_wick_ratio":
            upper_wick_ratio,
        "lower_wick_ratio":
            lower_wick_ratio,
        "bodies": bodies
    }

# =========================================================
# PROCESS SYMBOL
# =========================================================

def process_symbol(
    symbol,
    use_latest_candle
):

    try:

        if VERBOSE_SCAN:

            print(
                f"Checking {symbol}"
            )

        df = get_klines(symbol)

        if df is None:
            return None

        result = qualifies(
            df,
            use_latest_candle
        )

        if result:

            print(
                f"✅ MATCH: {symbol}"
            )

            return {
                "symbol": symbol,
                **result
            }

    except Exception as e:

        print(
            f"{symbol} Error: {e}"
        )

    return None

# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):

    if not TELEGRAM_ENABLED:
        return

    try:

        url = (
            f"https://api.telegram.org/bot"
            f"{TELEGRAM_BOT_TOKEN}"
            f"/sendMessage"
        )

        payload = {
            "chat_id":
                TELEGRAM_CHAT_ID,

            "text":
                message
        }

        requests.post(
            url,
            data=payload,
            timeout=10
        )

        print("📩 Telegram sent")

    except Exception as e:

        print(
            f"Telegram Error: {e}"
        )

# =========================================================
# MAIN
# =========================================================

def main():

    now = datetime.now(
        timezone.utc
    )

    print(
        f"\n[{now}] "
        f"H4 scanner started"
    )

    # =====================================================
    # EXECUTION WINDOW
    # =====================================================

    if not should_run_scan():

        print(
            "Outside execution window."
        )

        return

    print(
        "Inside execution window."
    )

    # =====================================================
    # CANDLE MODE
    # =====================================================

    use_latest_candle = (
        should_use_latest_candle(now)
    )

    if use_latest_candle:

        print(
            "Using latest H4 candle."
        )

    else:

        print(
            "Ignoring forming H4 candle."
        )

    # =====================================================
    # GET PAIRS
    # =====================================================

    pairs = get_usdt_pairs()

    print(
        f"Scanning "
        f"{len(pairs)} pairs..."
    )

    results = []

    # =====================================================
    # MULTITHREADED SCAN
    # =====================================================

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [

            executor.submit(
                process_symbol,
                symbol,
                use_latest_candle
            )

            for symbol in pairs
        ]

        for future in as_completed(
            futures
        ):

            result = future.result()

            if result:
                results.append(result)

    # =====================================================
    # SORT
    # =====================================================

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    top = results[:TOP_RESULTS]

    # =====================================================
    # MESSAGE
    # =====================================================

    message = (
        "🚀 H4 NO RED ZONE 🚀\n\n"

        f"UTC Time: "
        f"{now.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )

    if not top:

        message += (
            "No qualifying coins found."
        )

    else:

        for i, coin in enumerate(
            top,
            start=1
        ):

            body_text = ", ".join(
                f"{x:.2f}%"

                for x in coin["bodies"]
            )

            message += (
                f"{i}. {coin['symbol']}\n"
                f"Score: "
                f"{coin['score']:.2f}\n"
                f"Momentum: "
                f"{coin['momentum_score']:.2f}\n"
                f"Acceleration Bonus: "
                f"{coin['acceleration_bonus']:.2f}\n"
                f"Upper Wick: "
                f"{coin['upper_wick_ratio']:.2f}\n"
                f"Bodies: "
                f"[{body_text}]\n\n"
            )

    print("\n")
    print("=" * 60)
    print(message)
    print("=" * 60)

    send_telegram(message)

# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()
