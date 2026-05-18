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

INTERVAL = "1h"

MIN_GREEN_CANDLES = 3

TOP_RESULTS = 20

MAX_WORKERS = 25

VERBOSE_SCAN = False

# =========================================================
# EXECUTION WINDOW
# =========================================================

# GitHub Actions runs every 5 mins
# Scanner executes ONLY near:
#
# 03:57 UTC
# 07:57 UTC
# 11:57 UTC
# 15:57 UTC
# 19:57 UTC
# 23:57 UTC

EXECUTION_WINDOW_MINUTES = 3

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
# STRATEGY TUNING (H4)
# =========================================================

# Reject tiny candles
MIN_BODY_PERCENT = 0.25

MAX_FINAL_BODY_PERCENT = 6.5

IDEAL_ACCELERATION_MIN = 1.2
IDEAL_ACCELERATION_MAX = 3.0

MIN_LAST_EXPANSION = 1.08

MIN_VOLUME_MULTIPLIER = 1.35

# =========================================================
# EXECUTION WINDOW CHECK
# =========================================================

def should_run_scan():

    now = datetime.now(timezone.utc)

    valid_hours = [
        3,
        7,
        11,
        15,
        19,
        23
    ]

    if now.hour not in valid_hours:
        return False

    return (
        57
        <= now.minute
        <= 59
    )

# =========================================================
# USE CURRENT H4 CANDLE?
# =========================================================

def should_use_latest_candle(now):

    # YES
    # We intentionally scan
    # near H4 close

    return True

# =========================================================
# GET ALL USDT PAIRS
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

def get_klines(
    symbol,
    limit=10
):

    url = f"{BASE_URL}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": INTERVAL,
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
            "close",
            "volume"
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
    # SELECT RECENT CANDLES
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

        # Reject weak momentum
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
    # VOLUME EXPANSION
    # =====================================================

    recent_volume = (
        recent["volume"]
        .astype(float)
    )

    latest_volume = (
        recent_volume.iloc[-1]
    )

    previous_volume = (
        recent_volume.iloc[-2]
    )

    if latest_volume < (
        previous_volume
        * MIN_VOLUME_MULTIPLIER
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
    # LAST CANDLE ANALYSIS
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
    # SCORE
    # =====================================================

    momentum_score = sum(bodies)

    curve_bonus = (
        bodies[-1]
        * acceleration_quality
    )

    wick_penalty = (
        upper_wick_ratio * 12
        + lower_wick_ratio * 3
    )

    exhaustion_penalty = max(
        0,
        final_body - 7
    ) * 1.0

    final_score = (
        momentum_score
        + curve_bonus
        - wick_penalty
        - exhaustion_penalty
    )

    return {

        "score":
        final_score,

        "momentum_score":
        momentum_score,

        "curve_bonus":
        curve_bonus,

        "upper_wick_ratio":
        upper_wick_ratio,

        "lower_wick_ratio":
        lower_wick_ratio,

        "latest_volume":
        latest_volume,

        "bodies":
        bodies
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
        f"H4 scanner started"
    )

    # =====================================================
    # EXECUTION WINDOW
    # =====================================================

    if not should_run_scan():

        print(
            "Not inside "
            "execution window."
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

    print(
        "Using latest H4 candle."
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
    # MULTITHREADING
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
    # MESSAGE
    # =====================================================

    message = (
        "H1 no red zone\n\n"

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

                f"{i}. "
                f"{coin['symbol']}\n"

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

                f"Volume: "
                f"{coin['latest_volume']:.0f}\n"

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
