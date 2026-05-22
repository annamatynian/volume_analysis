"""
download_anchor_data.py
=======================
Этап 8D Шаг 2: скачивает monthly ZIP aggTrades для anchor period.

Использование:
    from download_anchor_data import build_anchor_zip_path, download_anchor_month

    anchor_months = get_anchor_months(df, poc, atr, lookback_days=120)
    for year, month in anchor_months:
        path = download_anchor_month(year, month, OUT_DIR, BASE_URL)

Все файлы сохраняются на D: — никогда на C:.
"""

import os
import time
import urllib.request

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

# WHY __file__: абсолютный путь всегда указывает на D:\DeFi-RAG-Projects\volume_analysis\
# независимо от того, из какой директории запущен python.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_URL = (
    "https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT"
)
OUT_DIR = os.path.join(
    SCRIPT_DIR, "data", "futures", "um", "monthly", "aggTrades", "BTCUSDT"
)


# ---------------------------------------------------------------------------
# Чистые функции — полностью тестируемы без сети
# ---------------------------------------------------------------------------

def build_anchor_zip_path(year: int, month: int, base_dir: str) -> str:
    """
    Строит путь к ZIP-файлу aggTrades для заданного месяца.

    Чистая функция — не создаёт файл/папку, только возвращает путь.

    Args:
        year:     Год (например 2025).
        month:    Месяц 1–12.
        base_dir: Папка для хранения ZIP-файлов.

    Returns:
        Абсолютный путь вида: base_dir/BTCUSDT-aggTrades-YYYY-MM.zip

    WHY отдельная функция: позволяет проверить корректность пути без сети.
    Оркестратор использует её для проверки кэша перед скачиванием.
    """
    filename = f"BTCUSDT-aggTrades-{year}-{month:02d}.zip"
    return os.path.join(base_dir, filename)


# ---------------------------------------------------------------------------
# Вспомогательная функция прогресса
# ---------------------------------------------------------------------------

def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct      = min(downloaded / total_size * 100, 100)
        mb       = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {pct:5.1f}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


# ---------------------------------------------------------------------------
# Основная функция скачивания
# ---------------------------------------------------------------------------

def download_anchor_month(
    year: int,
    month: int,
    out_dir: str,
    base_url: str = BASE_URL,
) -> "str | None":
    """
    Скачивает один monthly ZIP aggTrades из Binance Vision.

    Idempotent: если файл уже существует — пропускает скачивание.
    Текущий незавершённый месяц: пропускает (файла ещё нет на Binance Vision).
    При ошибке сети: удаляет неполный файл, возвращает None (не поднимает).
    """
    from datetime import date as _date
    _today = _date.today()
    # WHY пропуск текущего месяца: Binance Vision публикует monthly ZIP только
    # после окончания месяца. Попытка скачать текущий месяц = гарантированный 404.
    if year == _today.year and month == _today.month:
        print(f"[SKIP] {year}-{month:02d} — текущий месяц ещё не закончен, файл недоступен")
        return None

    os.makedirs(out_dir, exist_ok=True)

    zip_path = build_anchor_zip_path(year, month, out_dir)
    filename = os.path.basename(zip_path)
    url      = f"{base_url}/{filename}"

    # Idempotent: файл уже есть — пропускаем
    if os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / 1_048_576
        print(f"[SKIP] {filename} уже существует ({size_mb:.1f} MB)")
        return zip_path

    print(f"[DOWN] {url}")
    t0 = time.perf_counter()

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
    except Exception as e:
        print(f"\n[ERR]  Ошибка скачивания {filename}: {e}")
        # WHY удаляем: неполный ZIP сломает load_aggtrades_zip() при следующем запуске
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return None

    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(zip_path) / 1_048_576
    print(f"\n[OK]   {filename}  {size_mb:.1f} MB  за {elapsed:.1f} сек")
    return zip_path


# ---------------------------------------------------------------------------
# Daily ZIP — Reaction Period (Шаг 1 из задания)
# ---------------------------------------------------------------------------

DAILY_BASE_URL = (
    "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT"
)
DAILY_OUT_DIR = os.path.join(
    SCRIPT_DIR, "data", "futures", "um", "daily", "aggTrades", "BTCUSDT"
)


def build_reaction_zip_path(year: int, month: int, day: int, base_dir: str) -> str:
    """
    Строит путь к daily ZIP-файлу aggTrades для заданного дня.

    Чистая функция — не создаёт файл/папку, только возвращает путь.

    Args:
        year:     Год (например 2025).
        month:    Месяц 1–12.
        day:      День 1–31.
        base_dir: Папка для хранения daily ZIP-файлов.

    Returns:
        Абсолютный путь вида: base_dir/BTCUSDT-aggTrades-YYYY-MM-DD.zip

    WHY отдельная функция: аналогично build_anchor_zip_path — позволяет
    проверить корректность пути без сети, используется в idempotent-логике.
    """
    filename = f"BTCUSDT-aggTrades-{year}-{month:02d}-{day:02d}.zip"
    return os.path.join(base_dir, filename)


def download_reaction_month(
    year: int,
    month: int,
    day: int,
    out_dir: str,
    base_url: str = DAILY_BASE_URL,
) -> "str | None":
    """
    Скачивает один daily ZIP aggTrades из Binance Vision.

    Idempotent: если файл уже существует — пропускает скачивание.
    При ошибке сети: удаляет неполный файл, возвращает None (не поднимает).

    Args:
        year:     Год дня для скачивания.
        month:    Месяц 1–12.
        day:      День 1–31.
        out_dir:  Локальная папка для сохранения (создаётся если не существует).
        base_url: Базовый URL Binance Vision daily (переопределяется в тестах).

    Returns:
        str  — абсолютный путь к ZIP-файлу при успехе или если уже существует.
        None — при ошибке сети.

    WHY аналогична download_anchor_month: тот же контракт (idempotent, None on error),
    другой URL-шаблон (daily: YYYY-MM-DD вместо monthly: YYYY-MM).
    """
    os.makedirs(out_dir, exist_ok=True)

    zip_path = build_reaction_zip_path(year, month, day, out_dir)
    filename = os.path.basename(zip_path)
    url      = f"{base_url}/{filename}"

    # Idempotent: файл уже есть — пропускаем
    if os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / 1_048_576
        print(f"[SKIP] {filename} уже существует ({size_mb:.1f} MB)")
        return zip_path

    print(f"[DOWN] {url}")
    t0 = time.perf_counter()

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
    except Exception as e:
        print(f"\n[ERR]  Ошибка скачивания {filename}: {e}")
        # WHY удаляем: неполный ZIP сломает load_aggtrades_zip() при следующем запуске
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return None

    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(zip_path) / 1_048_576
    print(f"\n[OK]   {filename}  {size_mb:.1f} MB  за {elapsed:.1f} сек")
    return zip_path


# ---------------------------------------------------------------------------
# Daily klines (1m) — Этап 9: Delta Volume Profile
# ---------------------------------------------------------------------------

KLINES_BASE_URL = (
    "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m"
)
KLINES_OUT_DIR = os.path.join(
    SCRIPT_DIR, "data", "futures", "um", "daily", "klines", "BTCUSDT", "1m"
)


def build_klines_zip_path(year: int, month: int, day: int, base_dir: str) -> str:
    """
    Строит путь к daily klines ZIP-файлу 1m свечей.

    Чистая функция — не создаёт файл/папку, только возвращает путь.

    Args:
        year:     Год (например 2025).
        month:    Месяц 1–12.
        day:      День 1–31.
        base_dir: Папка для хранения ZIP-файлов.

    Returns:
        Абсолютный путь вида: base_dir/BTCUSDT-1m-YYYY-MM-DD.zip

    WHY отдельная функция: позволяет проверить корректность пути без сети.
    Аналогично build_reaction_zip_path, но другой шаблон имени (BTCUSDT-1m-...).
    """
    filename = f"BTCUSDT-1m-{year}-{month:02d}-{day:02d}.zip"
    return os.path.join(base_dir, filename)


def download_klines_day(
    year: int,
    month: int,
    day: int,
    out_dir: str,
    base_url: str = KLINES_BASE_URL,
) -> "str | None":
    """
    Скачивает один daily 1m klines ZIP из Binance Vision.

    Idempotent: если файл уже существует — пропускает скачивание.
    При ошибке сети: удаляет неполный файл, возвращает None (не поднимает).

    Args:
        year:     Год дня для скачивания.
        month:    Месяц 1–12.
        day:      День 1–31.
        out_dir:  Локальная папка (создаётся если не существует).
        base_url: Базовый URL Binance Vision klines (переопределяется в тестах).

    Returns:
        str  — абсолютный путь к ZIP-файлу при успехе или если уже существует.
        None — при ошибке сети.
    """
    os.makedirs(out_dir, exist_ok=True)

    zip_path = build_klines_zip_path(year, month, day, out_dir)
    filename = os.path.basename(zip_path)
    url      = f"{base_url}/{filename}"

    # Idempotent: файл уже есть — пропускаем
    if os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / 1_048_576
        print(f"[SKIP] {filename} уже существует ({size_mb:.1f} MB)")
        return zip_path

    print(f"[DOWN] {url}")
    t0 = time.perf_counter()

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
    except Exception as e:
        print(f"\n[ERR]  Ошибка скачивания {filename}: {e}")
        # WHY удаляем: неполный ZIP сломает load_klines_zip() при следующем запуске
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return None

    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(zip_path) / 1_048_576
    print(f"\n[OK]   {filename}  {size_mb:.1f} MB  за {elapsed:.1f} сек")
    return zip_path


# ---------------------------------------------------------------------------
# CLI — запуск вручную для конкретного диапазона месяцев
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Пример использования: скачать anchor period вручную.
    В production вызывается из liquidity_density_audit() через get_anchor_months().
    """
    # Пример: anchor period январь–апрель 2025
    anchor_months = [(2025, 1), (2025, 2), (2025, 3), (2025, 4)]

    print(f"[PATH] Целевая папка: {os.path.abspath(OUT_DIR)}")
    print(f"[PLAN] Месяцев к скачиванию: {len(anchor_months)}")
    for y, m in anchor_months:
        print(f"       {y}-{m:02d}")

    confirm = input("[?]    Продолжить? (y/n): ").strip().lower()
    if confirm != "y":
        print("[АБОРТ] Отменено.")
        return

    results = []
    for year, month in anchor_months:
        path = download_anchor_month(year, month, OUT_DIR, BASE_URL)
        results.append((year, month, path))

    print("\n[ИТОГ]")
    ok      = [(y, m, p) for y, m, p in results if p is not None]
    failed  = [(y, m)    for y, m, p in results if p is None]
    print(f"  Успешно:  {len(ok)}")
    print(f"  Ошибок:   {len(failed)}")
    if failed:
        for y, m in failed:
            print(f"  [FAIL] {y}-{m:02d}")


if __name__ == "__main__":
    main()
