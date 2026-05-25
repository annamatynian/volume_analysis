# coingecko_client.py
# НВ-03 | BTC Dominance — клиент CoinGecko API + parquet-кэш
#
# Диагностика 2026-05-25:
#   /api/v3/global                  → 200 OK; поле data.btc_dominance (float, %)
#   /api/v3/global/market_cap_chart → 401 PRO-only (недоступен)
#   /api/v3/coins/bitcoin/market_chart → 200 OK (не используется для BTC.D)
#
# Архитектура: parquet-кэш в data/coingecko_btc_dominance.parquet
#   Каждый запуск добавляет сегодняшнее значение BTC.D (идемпотентно).
#   get_btc_dominance_with_history() возвращает (current, 30d_ago_or_None).
#
# Примечание: первые 30 дней работы prev=None → classify вернёт NEUTRAL.
#   Это честный fallback при недостатке истории.

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.coingecko.com/api/v3"
_CACHE_COL = "btc_dominance_pct"
_DATE_COL  = "date"

# Путь кэша по умолчанию (переопределяется в тестах через параметр)
_DEFAULT_CACHE_PATH = Path("data") / "coingecko_btc_dominance.parquet"


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

def get_btc_dominance_current() -> float:
    """
    Возвращает текущий % доминирования BTC из CoinGecko /api/v3/global.

    Returns:
        float: btc_dominance в % (например, 55.3 означает 55.3%).

    Raises:
        requests.HTTPError: при HTTP ≠ 2xx (rate limit, сеть и т.д.)
        KeyError: если структура ответа изменилась (data.btc_dominance отсутствует).

    WHY data["btc_dominance"] (не верхний уровень):
        CoinGecko /global оборачивает метрики в ключ "data";
        верхний уровень содержит только "data". Без вложенного доступа → KeyError.
    """
    url = f"{_BASE_URL}/global"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()["data"]
    # CoinGecko free tier: btc_dominance в market_cap_percentage.btc
    # Pro API / будущие версии: поле btc_dominance напрямую
    # Диагностика 2026-05-25: free tier возвращает market_cap_percentage.btc
    if "btc_dominance" in data:
        return float(data["btc_dominance"])
    return float(data["market_cap_percentage"]["btc"])


def get_btc_dominance_with_history(
    lookback_days: int = 30,
    cache_path: Path = _DEFAULT_CACHE_PATH,
) -> tuple[float, float | None]:
    """
    Возвращает (btc_d_current, btc_d_lookback_ago).

    Алгоритм:
      1. Получить текущий BTC.D с API.
      2. Добавить сегодняшнее значение в parquet-кэш (идемпотентно).
      3. Если в кэше >= lookback_days записей → вернуть значение lookback_days дней назад.
         Иначе → вернуть None (недостаточно истории).

    Args:
        lookback_days: Глубина истории для сравнения (по умолчанию 30).
        cache_path:    Путь к parquet-кэшу (переопределяется в тестах).

    Returns:
        tuple: (current_btc_d, prev_btc_d_or_None)
          - current_btc_d: всегда свежий float с API.
          - prev_btc_d_or_None: float если накоплено >= lookback_days дней, иначе None.

    WHY prev=None, а не 0.0 или current:
        0.0 → delta = current - 0 → ложный ROTATION_BTC почти всегда.
        current → delta = 0 → вечный NEUTRAL, сигнал мёртв.
        None → оркестратор явно знает что истории нет → передаёт в missing.
    """
    current = get_btc_dominance_current()
    _update_cache(current, cache_path)
    prev = _get_lookback_value(lookback_days, cache_path)
    return current, prev


# ---------------------------------------------------------------------------
# Внутренние функции кэша
# ---------------------------------------------------------------------------

def _update_cache(btc_d: float, cache_path: Path) -> None:
    """
    Добавляет сегодняшнее значение BTC.D в parquet-кэш.

    Идемпотентно: если запись за сегодня уже есть — не дублирует.
    Создаёт файл и директорию при первом вызове.

    WHY идемпотентность: оркестратор может запускаться несколько раз в день;
        дублирование строк смещает индекс lookup и возвращает неверную дату.
    """
    today = str(date.today())
    cache_path = Path(cache_path)

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        df = pd.DataFrame(columns=[_DATE_COL, _CACHE_COL])
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Не добавлять если сегодня уже есть
    if today in df[_DATE_COL].values:
        return

    new_row = pd.DataFrame([{_DATE_COL: today, _CACHE_COL: float(btc_d)}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_parquet(cache_path, index=False)


def _get_lookback_value(lookback_days: int, cache_path: Path) -> float | None:
    """
    Возвращает значение BTC.D из кэша lookback_days строк назад (с конца).

    Returns None если:
      - кэш-файл не существует
      - записей меньше lookback_days

    WHY по индексу строк, не по дате:
        Оркестратор запускается ежедневно → строки соответствуют дням.
        Поиск по дате требует точного совпадения (выходные, пропуски);
        индексный lookup прост и достаточен для торгового сигнала.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    df = pd.read_parquet(cache_path)
    df = df.sort_values(_DATE_COL).reset_index(drop=True)

    if len(df) < lookback_days:
        return None

    return float(df.iloc[-lookback_days][_CACHE_COL])
