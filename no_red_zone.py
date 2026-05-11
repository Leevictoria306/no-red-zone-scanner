import os
import math
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

MIN_GREEN_DAYS = 4
TOP_RESULTS = 20

MAX_WORKERS = 25

VERBOSE_SCAN = False

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
# STRATEGY TUNING
# =========================================================

MIN_BODY_PERCENT = 1.0

MAX_FINAL_BODY_PERCENT = 35.0

IDEAL_ACCELERATION_MIN = 1.4
IDEAL_ACCELERATION_MAX = 4.5

MIN_LAST_EXPANSION = 1.15

# =========================================================
# SHOULD USE CURRENT DAILY CANDLE?
# =========================================================

def should_use_latest_candle(now):

    # ONLY use latest candle
    # after UTC 23:50

    if (
        now.hour < 23
        or (
            now.hour == 23
            and now.minute < 50
        )
    ):
        return False

    return True

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
            symbol.get("quoteAsset") == "USDT"
            and symbol.get("status") == "TRADING"
            and symbol.get("isSpotTradingAllowed")
        ):

            pairs.append(symbol["symbol"])

    return pairs

# =========================================================
# GET KLINES
# =========================================================

def get_daily_klines(symbol, limit=10):

    url = f"{BASE_URL}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": "1d",
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
    days=MIN_GREEN_DAYS
):

    if use_latest_candle:

        recent = df.tail(days)

    else:

        recent = df.iloc[:-1].tail(days)

    if len(recent) < days:
        return None

    bodies = []

    for _, row in recent.iterrows():

        open_price = row["open"]
        close_price = row["close"]

        if close_price <= open_price:
            return None

        body_pct = (
            (close_price - open_price)
            / open_price
        ) * 100

        if body_pct < MIN_BODY_PERCENT:
            return None

        bodies.append(body_pct)

    # STRICT PROGRESSIVE
    progressive = all(
        bodies[i] < bodies[i + 1]
        for i in range(len(bodies) - 1)
    )

    if not progressive:
        return None

    # REJECT FLAT EXPANSION
    last_expansion = (
        bodies[-1] / bodies[-2]
    )

    if last_expansion < MIN_LAST_EXPANSION:
        return None

    # REJECT EXHAUSTION
    final_body = bodies[-1]

    if final_body > MAX_FINAL_BODY_PERCENT:
        return None

    # =====================================================
    # ACCELERATION QUALITY
    # =====================================================

    acceleration_scores = []

    for i in range(len(bodies) - 1):

        prev_body = bodies[i]
        next_body = bodies[i + 1]

        ratio = next_body / prev_body

        if (
            IDEAL_ACCELERATION_MIN
            <= ratio
            <= IDEAL_ACCELERATION_MAX
        ):

            score = 1.0

        else:

            distance = abs(
                ratio - (
                    (
                        IDEAL_ACCELERATION_MIN
                        + IDEAL_ACCELERATION_MAX
                    ) / 2
                )
            )

            score = max(
                0,
                1 - (distance / 5)
            )

        acceleration_scores.append(score)

    acceleration_quality = (
        sum(acceleration_scores)
        / len(acceleration_scores)
    )

    # =====================================================
    # WICK ANALYSIS
    # =====================================================

    last = recent.iloc[-1]

    open_price = last["open"]
    high_price = last["high"]
    low_price = last["low"]
    close_price = last["close"]

    body_size = abs(
        close_price - open_price
    )

    if body_size <= 0:
        return None

    upper_wick = (
        high_price - close_price
    )

    lower_wick = (
        open_price - low_price
    )

    upper_wick_ratio = (
        upper_wick / body_size
    )

    lower_wick_ratio = (
        lower_wick / body_size
    )

    # =====================================================
    # FINAL SCORE
    # =====================================================

    momentum_score = sum(bodies)

    curve_bonus = (
        bodies[-1] * acceleration_quality
    )

    wick_penalty = (
        upper_wick_ratio * 14
        + lower_wick_ratio * 4
    )

    exhaustion_penalty = max(
        0,
        final_body - 18
    ) * 0.6

    final_score = (
        momentum_score
        + curve_bonus
        - wick_penalty
        - exhaustion_penalty
    )

    return {
        "score": final_score,
        "momentum_score": momentum_score,
        "curve_bonus": curve_bonus,
        "upper_wick_ratio": upper_wick_ratio,
        "bodies": bodies
    }

# =========================================================
# PROCESS SYMBOL
# =========================================================

def process_symbol(
    symbol,
    use_latest_candle
):

    if VERBOSE_SCAN:
        print(f"Checking {symbol}")

    df = get_daily_klines(symbol)

    if df is None:
        return None

    result = qualifies(
        df,
        use_latest_candle
    )

    if result:

        return {
            "symbol": symbol,
            **result
        }

    return None

# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):

    if not TELEGRAM_ENABLED:
        return

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    requests.post(
        url,
        data=payload,
        timeout=10
    )

# =========================================================
# MAIN
# =========================================================

def main():

    now = datetime.now(timezone.utc)

    print(
        f"\n[{now}] "
        f"RUNNING NO RED ZONE SCAN"
    )

    use_latest_candle = (
        should_use_latest_candle(now)
    )

    pairs = get_usdt_pairs()

    results = []

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

        for future in as_completed(futures):

            result = future.result()

            if result:
                results.append(result)

    # SORT
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
        "🚀 NO RED ZONE STRATEGY 🚀\n"
        f"UTC: "
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
                f"Bodies: "
                f"[{body_text}]\n\n"
            )

    print(message)

    send_telegram(message)

# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()
