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

MIN_GREEN_DAYS = 3
TOP_RESULTS = 20

MAX_WORKERS = 25

VERBOSE_SCAN = False

# =========================================================
# EXECUTION WINDOW
# =========================================================

# GitHub Actions runs every 5 mins.
# Script only executes near 23:50 UTC.

EXECUTION_WINDOW_MINUTES = 7

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

# Reject tiny candles
MIN_BODY_PERCENT = 1.0

# Reject exhaustion candles
MAX_FINAL_BODY_PERCENT = 35.0

# Ideal exponential acceleration
IDEAL_ACCELERATION_MIN = 1.4
IDEAL_ACCELERATION_MAX = 4.5

# Reject flattening
MIN_LAST_EXPANSION = 1.15

# =========================================================
# SHOULD RUN?
# =========================================================

def should_run_scan():

    now = datetime.now(timezone.utc)

    target_hour = 23
    target_minute = 50

    current_minutes = (
        now.hour * 60
        + now.minute
    )

    target_minutes = (
        target_hour * 60
        + target_minute
    )

    difference = abs(
        current_minutes
        - target_minutes
    )

    return (
        difference
        <= EXECUTION_WINDOW_MINUTES
    )

# =========================================================
# SHOULD USE CURRENT DAILY CANDLE?
# =========================================================

def should_use_latest_candle(now):

    # ONLY USE latest candle
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
# GET DAILY KLINES
# =========================================================

def get_daily_klines(
    symbol,
    limit=10
):

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

    # =====================================================
    # SELECT CANDLES
    # =====================================================

    if use_latest_candle:

        recent = df.tail(days)

    else:

        # Ignore current forming candle
        recent = (
            df.iloc[:-1]
            .tail(days)
        )

    if len(recent) < days:
        return None

    bodies = []

    # =====================================================
    # BUILD BODY %
    # =====================================================

    for _, row in recent.iterrows():

        open_price = row["open"]
        close_price = row["close"]

        # MUST BE GREEN
        if close_price <= open_price:
            return None

        body_pct = (
            (
                close_price
                - open_price
            )
            / open_price
        ) * 100

        # Reject weak candles
        if (
            body_pct
            < MIN_BODY_PERCENT
        ):
            return None

        bodies.append(body_pct)

    # =====================================================
    # STRICTLY PROGRESSIVE
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
    #
    # 1,10,11 ❌
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
    # REJECT OVEREXTENSION
    #
    # 1,8,121 ❌
    # =====================================================

    final_body = bodies[-1]

    if (
        final_body
        > MAX_FINAL_BODY_PERCENT
    ):
        return None

    # =====================================================
    # ACCELERATION QUALITY
    # =====================================================

    acceleration_scores = []

    for i in range(
        len(bodies) - 1
    ):

        prev_body = bodies[i]
        next_body = bodies[i + 1]

        ratio = (
            next_body
            / prev_body
        )

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
                1 - (
                    distance / 5
                )
            )

        acceleration_scores.append(
            score
        )

    acceleration_quality = (
        sum(acceleration_scores)
        / len(acceleration_scores)
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
    # SCORING
    # =====================================================

    momentum_score = sum(bodies)

    curve_bonus = (
        bodies[-1]
        * acceleration_quality
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
        "lower_wick_ratio": lower_wick_ratio,
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

        df = get_daily_klines(symbol)

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

    if (
        not TELEGRAM_BOT_TOKEN
        or not TELEGRAM_CHAT_ID
    ):
        print(
            "Telegram secrets missing"
        )

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

        print(
            "📩 Telegram sent"
        )

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
        f"Workflow started"
    )

    # =====================================================
    # ONLY EXECUTE INSIDE WINDOW
    # =====================================================

    if not should_run_scan():

        print(
            "Not inside execution window."
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
            "Using latest candle."
        )

    else:

        print(
            "Ignoring current "
            "forming candle."
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
    # SORT RESULTS
    # =====================================================

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    top = results[:TOP_RESULTS]

    # =====================================================
    # BUILD MESSAGE
    # =====================================================

    message = (
        "🚀 NO RED ZONE STRATEGY 🚀\n\n"

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
                f"Curve Bonus: "
                f"{coin['curve_bonus']:.2f}\n"
                f"Upper Wick: "
                f"{coin['upper_wick_ratio']:.2f}\n"
                f"Lower Wick: "
                f"{coin['lower_wick_ratio']:.2f}\n"
                f"Bodies: "
                f"[{body_text}]\n\n"
            )

    # =====================================================
    # OUTPUT
    # =====================================================

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
