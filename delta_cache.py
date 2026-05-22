"""
delta_cache.py
==============
Этап 8D Шаг 3: ZIP aggTrades → CVD → parquet кэш.

Ответственность: IO-слой между ZIP-файлами и аналитическими функциями.
Аналитика (calculate_cvd_in_zone) остаётся в volume_density.py.

Структура кэша:
    data/delta_cache/BTCUSDT-anchor-YYYY-MM.parquet   <- monthly anchor period
    data/delta_cache/BTCUSDT-reaction-YYYY-MM.parquet <- daily reaction period (8D-5)

Использование:
    from delta_cache import build_cache_path, build_delta_cache
    from download_anchor_data import download_anchor_month

    zip_paths = {(2025, 1): "/path/to/BTCUSDT-aggTrades-2025-01.zip", ...}
    cache = build_delta_cache(zip_paths, poc=95000.0, atr=1500.0, cache_dir="data/delta_cache")
    # cache == {(2025, 1): "data/delta_cache/BTCUSDT-anchor-2025-01.parquet", ...}
"""

import os
import pandas as pd

from volume_density import load_aggtrades_zip, calculate_cvd_in_zone

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(SCRIPT_DIR, "data", "delta_cache")


# ---------------------------------------------------------------------------
# Чистая функция — путь к parquet
# ---------------------------------------------------------------------------

def build_cache_path(year: int, month: int, cache_dir: str,
                     kind: str = 'anchor') -> str:
    """
    Строит путь к parquet-файлу дельта-кэша.

    Чистая функция — не создаёт файл/папку, только возвращает путь.

    Args:
        year:      Год (например 2025).
        month:     Месяц 1-12.
        cache_dir: Папка для хранения parquet-файлов.
        kind:      'anchor' (monthly) или 'reaction' (daily, для 8D-5).

    Returns:
        Абсолютный путь вида: cache_dir/BTCUSDT-{kind}-YYYY-MM.parquet

    WHY отдельная функция: оркестратор проверяет существование кэша
    через этот путь перед загрузкой ZIP.
    """
    filename = f"BTCUSDT-{kind}-{year}-{month:02d}.parquet"
    return os.path.join(cache_dir, filename)


# ---------------------------------------------------------------------------
# Основная функция кэширования
# ---------------------------------------------------------------------------

def build_delta_cache(
    zip_paths: dict,
    poc: float,
    atr: float,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> dict:
    """
    Обрабатывает ZIP aggTrades и сохраняет CVD-профиль в parquet.

    Idempotent: если parquet уже существует — пропускает ZIP (не читает 660 MB).
    При ошибке чтения ZIP — возвращает None для этого месяца, продолжает остальные.

    Args:
        zip_paths: dict[(year, month) -> zip_path] — от download_anchor_month().
        poc:       Point of Control — центр зоны анализа.
        atr:       Average True Range — ширина зоны (+-1.5 ATR).
        cache_dir: Папка для сохранения parquet (создаётся если не существует).

    Returns:
        dict[(year, month) -> parquet_path | None]
            parquet_path — путь к файлу при успехе или если уже существовал.
            None         — при ошибке чтения/обработки ZIP.

    Схема parquet (одна строка на месяц):
        cvd_slope (float) — наклон CVD в зоне POC за этот месяц
        poc       (float) — POC использованный при расчёте
        atr       (float) — ATR использованный при расчёте
        year      (int)   — год месяца
        month     (int)   — номер месяца
    """
    os.makedirs(cache_dir, exist_ok=True)

    result = {}

    for (year, month), zip_path in zip_paths.items():
        parquet_path = build_cache_path(year, month, cache_dir, kind='anchor')

        # Idempotent: parquet уже есть — пропускаем тяжёлый ZIP
        if os.path.exists(parquet_path):
            print(f"[SKIP] {os.path.basename(parquet_path)} уже существует")
            result[(year, month)] = parquet_path
            continue

        try:
            print(f"[PROC] {year}-{month:02d} ← {os.path.basename(zip_path)}")
            df_trades = load_aggtrades_zip(zip_path)
            _, cvd_slope = calculate_cvd_in_zone(df_trades, poc, atr)

            # Сохраняем одну строку с результатом месяца
            cache_df = pd.DataFrame([{
                'cvd_slope': float(cvd_slope),
                'poc':       float(poc),
                'atr':       float(atr),
                'year':      int(year),
                'month':     int(month),
            }])
            cache_df.to_parquet(parquet_path, index=False)
            print(f"[OK]   {os.path.basename(parquet_path)}  cvd_slope={cvd_slope:.4f}")
            result[(year, month)] = parquet_path

        except Exception as e:
            # WHY не поднимаем: один сломанный ZIP не должен остановить весь pipeline
            print(f"[ERR]  {year}-{month:02d}: {e}")
            result[(year, month)] = None

    return result


# ---------------------------------------------------------------------------
# Reaction Period cache — Шаг 2 (из задания): daily ZIP → recent_delta
# ---------------------------------------------------------------------------

import re as _re


def build_reaction_zip_cache_path(zip_filename: str, cache_dir: str) -> str:
    """
    Строит путь к parquet-кэшу для одного daily ZIP.

    Чистая функция — не создаёт файл/папку.

    Args:
        zip_filename: Имя ZIP-файла (basename): BTCUSDT-aggTrades-YYYY-MM-DD.zip
        cache_dir:    Папка кэша.

    Returns:
        cache_dir/BTCUSDT-reaction-YYYY-MM-DD.parquet

    WHY отдельная функция: тесты и оркестратор могут проверить кэш без запуска всей логики.
    """
    # Извлекаем дату YYYY-MM-DD из имени файла BTCUSDT-aggTrades-YYYY-MM-DD.zip
    m = _re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(zip_filename))
    if m:
        date_str = m.group(1)
    else:
        # Фоллбэк: если дата не найдена — используем имя файла целиком
        date_str = os.path.splitext(os.path.basename(zip_filename))[0]
    filename = f"BTCUSDT-reaction-{date_str}.parquet"
    return os.path.join(cache_dir, filename)


def build_reaction_delta(
    zip_paths_daily: list,
    poc: float,
    atr: float,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> float:
    """
    Даёт суммарную дельту (recent_delta) за reaction period (14 дней).

    Для каждого daily ZIP:
      1. Проверяем кэш (BTCUSDT-reaction-YYYY-MM-DD.parquet).
      2. Если кэш есть — читаем recent_cvd из него.
      3. Если нет — load_aggtrades_zip → calculate_cvd_in_zone
         → cvd_series.iloc[-1] → сохраняем в parquet.
      4. Сломанный ZIP → 0.0 для этого дня (pipeline продолжается).

    recent_delta = сумма cvd_series.iloc[-1] для всех дней.

    WHY последнее значение CVD, не slope:
      - slope — направление (anchor period: качество уровня).
      - cvd.iloc[-1] — абсолютная дельта (reaction period: количественное давление).

    Args:
        zip_paths_daily: list[стр] — пути к daily ZIP (до 14 штук).
                         Имя файла должно содержать дату YYYY-MM-DD.
        poc:             Point of Control — центр зоны анализа.
        atr:             Average True Range — ширина зоны (±1.5 ATR).
        cache_dir:       Папка кэша (создаётся если не существует).

    Returns:
        float — суммарная дельта за reaction period.
                > 0: покупатели доминируют (подтверждение магнита).
                < 0: продавцы доминируют (подтверждение ловушки).
                0.0 если нет данных.
    """
    if not zip_paths_daily:
        return 0.0

    os.makedirs(cache_dir, exist_ok=True)

    total_delta = 0.0

    for zip_path in zip_paths_daily:
        zip_basename = os.path.basename(zip_path)
        parquet_path = build_reaction_zip_cache_path(zip_basename, cache_dir)

        # Idempotent: parquet уже есть — читаем сохранённое значение
        if os.path.exists(parquet_path):
            try:
                row = pd.read_parquet(parquet_path)
                total_delta += float(row['recent_cvd'].iloc[0])
                print(f"[SKIP] {os.path.basename(parquet_path)} (кэш)")
                continue
            except Exception:
                pass  # Повреждённый парчайный — пересчитываем

        try:
            print(f"[PROC] {zip_basename}")
            df_trades = load_aggtrades_zip(zip_path)
            cvd_series, _ = calculate_cvd_in_zone(df_trades, poc, atr)

            # WHY iloc[-1]: последнее значение CVD = накопленная дельта за весь день
            if len(cvd_series) > 0:
                day_delta = float(cvd_series.iloc[-1])
            else:
                day_delta = 0.0

            # Сохраняем в parquet для idempotency
            cache_df = pd.DataFrame([{'recent_cvd': day_delta}])
            cache_df.to_parquet(parquet_path, index=False)
            print(f"[OK]   {os.path.basename(parquet_path)}  cvd={day_delta:.4f}")

            total_delta += day_delta

        except Exception as e:
            # WHY 0.0: один сломанный день не стопит pipeline
            print(f"[ERR]  {zip_basename}: {e}")
            total_delta += 0.0

    return float(total_delta)


# ---------------------------------------------------------------------------
# Кэш для Delta Volume Profile (klines 1m) — Этап 9
# ---------------------------------------------------------------------------

from volume_density import load_klines_zip, build_delta_profile
import numpy as _np


def build_klines_cache_path(year: int, month: int, day: int,
                            cache_dir: str) -> str:
    """
    Строит путь к parquet-кэшу для одного дня klines delta-профиля.

    Чистая функция — не создаёт файл/папку, только возвращает путь.

    Args:
        year:      Год (например 2025).
        month:     Месяц 1-12.
        day:       День 1-31.
        cache_dir: Папка для хранения parquet-файлов.

    Returns:
        Абсолютный путь вида: cache_dir/BTCUSDT-klines-delta-YYYY-MM-DD.parquet

    WHY klines-delta в имени: отличает от reaction/anchor чтобы не перепутать
    файлы в одной cache_dir.
    """
    filename = f"BTCUSDT-klines-delta-{year}-{month:02d}-{day:02d}.parquet"
    return os.path.join(cache_dir, filename)


def build_klines_delta_cache(
    zip_paths_daily: dict,
    poc: float,
    atr: float,
    global_bins: _np.ndarray,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> dict:
    """
    Обрабатывает daily klines ZIP и сохраняет poc_bin_delta в parquet.

    Для каждого дня (year, month, day):
      1. Проверяем кэш (BTCUSDT-klines-delta-YYYY-MM-DD.parquet).
      2. Если кэш есть — пропускаем ZIP (экономим ~15 MB IO).
      3. Если нет — load_klines_zip → build_delta_profile
         → poc_bin_delta → сохраняем в parquet.
      4. Сломанный ZIP → None для этого дня (pipeline продолжается).

    Args:
        zip_paths_daily: dict[(year, month, day) -> zip_path]
        poc:             Point of Control — центр бина для poc_bin_delta.
        atr:             Average True Range (не используется в расчёте, но
                         сохраняется в parquet для диагностики).
        global_bins:     Массив бинов (тот же что в build_profile/оркестраторе).
        cache_dir:       Папка кэша (создаётся если не существует).

    Returns:
        dict[(year, month, day) -> parquet_path | None]
            parquet_path — при успехе или если кэш уже существовал.
            None         — при ошибке чтения/обработки ZIP.

    Схема parquet (одна строка на день):
        poc_bin_delta (float) — delta в бине POC за этот день
        poc           (float) — POC использованный при расчёте
        atr           (float) — ATR использованный при расчёте
        year          (int)   — год
        month         (int)   — месяц
        day           (int)   — день
    """
    os.makedirs(cache_dir, exist_ok=True)

    result = {}

    for (year, month, day), zip_path in zip_paths_daily.items():
        parquet_path = build_klines_cache_path(year, month, day, cache_dir)

        # Idempotent: parquet уже есть — пропускаем ZIP
        if os.path.exists(parquet_path):
            print(f"[SKIP] {os.path.basename(parquet_path)} уже существует")
            result[(year, month, day)] = parquet_path
            continue

        try:
            print(f"[PROC] {year}-{month:02d}-{day:02d} <- {os.path.basename(zip_path)}")
            klines_df   = load_klines_zip(zip_path)
            delta_prof  = build_delta_profile(klines_df, global_bins)

            # Находим бин POC
            poc_mask     = (
                (delta_prof['price_low']  <= poc) &
                (delta_prof['price_high'] >= poc)
            )
            poc_bin_delta = float(delta_prof.loc[poc_mask, 'delta'].sum())

            # Сохраняем одну строку
            cache_df = pd.DataFrame([{
                'poc_bin_delta': poc_bin_delta,
                'poc':           float(poc),
                'atr':           float(atr),
                'year':          int(year),
                'month':         int(month),
                'day':           int(day),
            }])
            cache_df.to_parquet(parquet_path, index=False)
            print(f"[OK]   {os.path.basename(parquet_path)}  poc_bin_delta={poc_bin_delta:.2f}")
            result[(year, month, day)] = parquet_path

        except Exception as e:
            # WHY не поднимаем: один сломанный день не должен остановить весь pipeline
            print(f"[ERR]  {year}-{month:02d}-{day:02d}: {e}")
            result[(year, month, day)] = None

    return result
