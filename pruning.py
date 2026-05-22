"""
pruning.py
==========
Этап В: Прунинг старых ZIP-файлов.

Назначение: очищать daily ZIP-файлы (klines ~15 MB/день + aggTrades ~22 MB/день),
которые уже обработаны в parquet-кэш и больше не нужны.

Стратегия:
- Удалять только *.zip файлы с датой YYYY-MM-DD в имени.
- Parquet-файлы никогда не трогать (они маленькие, пересчёт дорогой).
- keep_days=45 покрывает anchor period (30 дней) + запас (15 дней).
- dry_run=True — только показать что будет удалено, ничего не трогать.
- Файлы без распознаваемой даты — оставлять (нельзя определить возраст).
- Вызывается в конце liquidity_density_audit() автоматически.

Использование:
    from pruning import prune_old_zips

    # Проверить что будет удалено:
    result = prune_old_zips("data/futures/um/daily", keep_days=45, dry_run=True)
    print(result)

    # Реально удалить:
    result = prune_old_zips("data/futures/um/daily", keep_days=45, dry_run=False)
"""

import os
import re
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

def prune_old_zips(
    directory: str,
    keep_days: int = 45,
    dry_run: bool = False,
) -> dict:
    """
    Удаляет ZIP-файлы старше keep_days дней из указанной директории.

    Правила:
    - Ищет только *.zip файлы (не рекурсивно).
    - ZIP с датой YYYY-MM-DD в имени: удаляется если (today - date) > keep_days.
    - Граничный случай: ровно keep_days дней назад — ОСТАВЛЯЕТСЯ (строгое >).
    - ZIP без даты в имени — попадает в 'kept', не удаляется.
    - Не-ZIP файлы — полностью игнорируются (не в deleted/kept/errors).
    - Parquet и другие файлы никогда не трогаются.

    Args:
        directory: Директория для сканирования (не рекурсивно).
        keep_days: Сколько дней хранить ZIP (по умолчанию 45).
                   Файлы СТРОГО СТАРШЕ keep_days дней — удаляются.
        dry_run:   True — только репортить, не удалять.
                   False — реально удалять (по умолчанию).

    Returns:
        dict с ключами:
            'deleted': list[str] — пути удалённых (или which would be deleted).
            'kept':    list[str] — пути оставленных файлов.
            'errors':  list[str] — описания ошибок при удалении.

    WHY строгое > для keep_days:
        Предотвращает случайное удаление файла в день его создания.
        ZIP за сегодня минус keep_days дней = граница — файл ещё актуален.
    """
    deleted = []
    kept    = []
    errors  = []

    # WHY replace(hour=0,...): нормализуем до начала дня.
    # _extract_date_from_filename возвращает datetime(Y,M,D) = начало дня.
    # Без нормализации datetime.today() содержит время (15:30:00) и граничный
    # файл (ровно keep_days назад) ошибочно попадёт под удаление (00:00 < 15:30).
    today_start = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff_date = today_start - timedelta(days=keep_days)

    if not os.path.isdir(directory):
        return {'deleted': deleted, 'kept': kept, 'errors': errors}

    for filename in os.listdir(directory):
        # Только ZIP-файлы
        if not filename.lower().endswith('.zip'):
            continue

        filepath = os.path.join(directory, filename)

        # Извлекаем дату из имени файла
        file_date = _extract_date_from_filename(filename)

        if file_date is None:
            # Нет даты — не можем определить возраст → оставляем
            kept.append(filepath)
            continue

        # Строгое сравнение: удаляем только СТРОГО СТАРШЕ keep_days
        if file_date < cutoff_date:
            if dry_run:
                deleted.append(filepath)
            else:
                try:
                    os.remove(filepath)
                    deleted.append(filepath)
                except OSError as e:
                    errors.append(f"{filepath}: {e}")
        else:
            kept.append(filepath)

    return {'deleted': deleted, 'kept': kept, 'errors': errors}


# ---------------------------------------------------------------------------
# Вспомогательные функции (приватные)
# ---------------------------------------------------------------------------

_DATE_PATTERN = re.compile(r'(\d{4})-(\d{2})-(\d{2})')


def _extract_date_from_filename(filename: str):
    """
    Извлекает дату YYYY-MM-DD из имени файла.

    Примеры:
        'BTCUSDT-aggTrades-2025-04-01.zip' → datetime(2025, 4, 1)
        'BTCUSDT-1m-2025-04-01.zip'        → datetime(2025, 4, 1)
        'BTCUSDT-unknown.zip'              → None

    Returns:
        datetime или None если дата не найдена / некорректна.
    """
    m = _DATE_PATTERN.search(filename)
    if not m:
        return None
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return datetime(year, month, day)
    except ValueError:
        # Некорректная дата (например 2025-13-40) → не трогаем файл
        return None
