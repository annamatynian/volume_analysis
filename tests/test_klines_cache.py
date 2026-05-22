"""
tests/test_klines_cache.py
==========================
Unit tests for build_klines_cache_path() and build_klines_delta_cache()
in delta_cache.py — Этап 9 (кэш для build_delta_profile).

Принципы:
- load_klines_zip() мокируется — тест не читает реальные ZIP (~15 MB).
- Тесты проверяют контракт: idempotency, схема parquet, обработка ошибок.
- Все файлы в tmp_path — ничего не пишется в реальный проект.
"""

import os
import pytest
import numpy as np
import pandas as pd
import zipfile
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from delta_cache import build_klines_cache_path, build_klines_delta_cache


# ---------------------------------------------------------------------------
# Вспомогательные фикстуры
# ---------------------------------------------------------------------------

def _make_fake_klines_df(n: int = 100, price_center: float = 50_000.0) -> pd.DataFrame:
    """
    Синтетический DataFrame klines — результат load_klines_zip().
    Колонки: open_time (int64), high (float32), low (float32),
             volume (float32), taker_buy_vol (float32).
    Свечи расположены вокруг price_center.
    """
    rng = np.random.default_rng(42)
    base_time = 1_700_000_000_000
    highs = (price_center + rng.uniform(0, 200, n)).astype('float32')
    lows  = (price_center - rng.uniform(0, 200, n)).astype('float32')
    vols  = rng.uniform(1.0, 10.0, n).astype('float32')
    buy_fractions = rng.uniform(0.3, 0.7, n).astype('float32')
    return pd.DataFrame({
        'open_time':     (base_time + np.arange(n) * 60_000).astype('int64'),
        'high':          highs,
        'low':           lows,
        'volume':        vols,
        'taker_buy_vol': (vols * buy_fractions).astype('float32'),
    })


def _make_fake_zip(path: str) -> None:
    """Создаёт минимальный ZIP — достаточно чтобы файл существовал на диске."""
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr("dummy.csv", "col1,col2\n1,2\n")


# ---------------------------------------------------------------------------
# build_klines_cache_path — чистая функция пути
# ---------------------------------------------------------------------------

class TestBuildKlinesCachePath:
    """
    Контракт build_klines_cache_path(year, month, day, cache_dir) -> str:
    - Имя файла: BTCUSDT-klines-delta-YYYY-MM-DD.parquet
    - Путь вложен в cache_dir.
    - Чистая функция — не создаёт файл/папку.
    """

    def test_filename_format(self, tmp_path):
        """
        Контракт: имя файла BTCUSDT-klines-delta-YYYY-MM-DD.parquet.
        WHY klines-delta: отличается от reaction/anchor — не перепутать в cache_dir.
        """
        path = build_klines_cache_path(2025, 4, 1, str(tmp_path))
        assert os.path.basename(path) == "BTCUSDT-klines-delta-2025-04-01.parquet"

    def test_day_and_month_zero_padded(self, tmp_path):
        """
        Контракт: однозначные day и month дополняются нулём.
        """
        path = build_klines_cache_path(2025, 1, 9, str(tmp_path))
        assert "2025-01-09" in os.path.basename(path)

    def test_path_inside_cache_dir(self, tmp_path):
        """
        Контракт: путь начинается с cache_dir.
        """
        path = build_klines_cache_path(2025, 4, 1, str(tmp_path))
        assert path.startswith(str(tmp_path))

    def test_does_not_create_file(self, tmp_path):
        """
        Контракт: чистая функция — никаких сайд-эффектов.
        """
        build_klines_cache_path(2025, 4, 1, str(tmp_path))
        parquet = os.path.join(str(tmp_path), "BTCUSDT-klines-delta-2025-04-01.parquet")
        assert not os.path.exists(parquet)


# ---------------------------------------------------------------------------
# build_klines_delta_cache — основная функция кэширования
# ---------------------------------------------------------------------------

class TestBuildKlinesDeltaCache:
    """
    Контракт build_klines_delta_cache(zip_paths_daily, poc, atr, global_bins, cache_dir)
        -> dict[(year, month, day) -> parquet_path | None]:
    - Обрабатывает каждый ZIP → load_klines_zip → build_delta_profile → parquet.
    - Idempotent: если parquet существует — ZIP не читается повторно.
    - Сломанный ZIP → None для этого дня, pipeline продолжается.
    - Пустой список → пустой dict.
    - Схема parquet: poc_bin_delta (float), poc (float), atr (float),
                     year (int), month (int), day (int).
    """

    POC   = 50_000.0
    ATR   = 500.0
    BINS  = np.linspace(49_000.0, 51_000.0, 101)  # 100 бинов вокруг POC

    def _zip_path(self, tmp_path, year: int, month: int, day: int) -> str:
        name = f"BTCUSDT-1m-{year}-{month:02d}-{day:02d}.zip"
        path = str(tmp_path / name)
        _make_fake_zip(path)
        return path

    def test_returns_parquet_path_on_success(self, tmp_path):
        """
        Контракт: успешная обработка → возвращает путь к parquet-файлу.
        """
        zip_path = self._zip_path(tmp_path, 2025, 4, 1)
        fake_df  = _make_fake_klines_df()

        with patch("delta_cache.load_klines_zip", return_value=fake_df):
            result = build_klines_delta_cache(
                {(2025, 4, 1): zip_path}, self.POC, self.ATR, self.BINS,
                cache_dir=str(tmp_path)
            )

        assert (2025, 4, 1) in result
        assert result[(2025, 4, 1)] is not None
        assert result[(2025, 4, 1)].endswith(".parquet")

    def test_parquet_file_created_on_disk(self, tmp_path):
        """
        Контракт: parquet-файл реально создаётся на диске.
        """
        zip_path = self._zip_path(tmp_path, 2025, 4, 2)
        fake_df  = _make_fake_klines_df()

        with patch("delta_cache.load_klines_zip", return_value=fake_df):
            result = build_klines_delta_cache(
                {(2025, 4, 2): zip_path}, self.POC, self.ATR, self.BINS,
                cache_dir=str(tmp_path)
            )

        assert os.path.exists(result[(2025, 4, 2)])

    def test_parquet_schema_correct(self, tmp_path):
        """
        Контракт: parquet содержит колонки poc_bin_delta, poc, atr, year, month, day.
        WHY явная схема: оркестратор читает по имени колонки — несовпадение = тихая ошибка.
        """
        zip_path = self._zip_path(tmp_path, 2025, 4, 3)
        fake_df  = _make_fake_klines_df()

        with patch("delta_cache.load_klines_zip", return_value=fake_df):
            result = build_klines_delta_cache(
                {(2025, 4, 3): zip_path}, self.POC, self.ATR, self.BINS,
                cache_dir=str(tmp_path)
            )

        row = pd.read_parquet(result[(2025, 4, 3)])
        for col in ['poc_bin_delta', 'poc', 'atr', 'year', 'month', 'day']:
            assert col in row.columns, f"Missing column in parquet: {col}"

    def test_idempotent_skips_zip_on_second_call(self, tmp_path):
        """
        Контракт: если parquet уже существует — load_klines_zip НЕ вызывается повторно.
        WHY: именно это экономит ~15 MB IO при каждом повторном запуске.
        """
        zip_path = self._zip_path(tmp_path, 2025, 4, 4)
        fake_df  = _make_fake_klines_df()

        with patch("delta_cache.load_klines_zip", return_value=fake_df) as mock_load:
            # Первый вызов — создаёт parquet
            build_klines_delta_cache(
                {(2025, 4, 4): zip_path}, self.POC, self.ATR, self.BINS,
                cache_dir=str(tmp_path)
            )
            first_call_count = mock_load.call_count

            # Второй вызов — должен пропустить ZIP
            build_klines_delta_cache(
                {(2025, 4, 4): zip_path}, self.POC, self.ATR, self.BINS,
                cache_dir=str(tmp_path)
            )
            second_call_count = mock_load.call_count

        assert first_call_count == 1, "ZIP должен читаться при первом вызове"
        assert second_call_count == 1, "ZIP НЕ должен читаться при повторном вызове (idempotent)"

    def test_broken_zip_returns_none_continues(self, tmp_path):
        """
        Контракт: сломанный ZIP → None для этого дня, остальные обрабатываются.
        WHY: один отсутствующий день не должен стопить весь pipeline.
        """
        zip_ok     = self._zip_path(tmp_path, 2025, 4, 5)
        zip_broken = self._zip_path(tmp_path, 2025, 4, 6)
        fake_df    = _make_fake_klines_df()

        def load_side_effect(path):
            if "2025-04-06" in path:
                raise Exception("Corrupted ZIP")
            return fake_df

        with patch("delta_cache.load_klines_zip", side_effect=load_side_effect):
            result = build_klines_delta_cache(
                {
                    (2025, 4, 5): zip_ok,
                    (2025, 4, 6): zip_broken,
                },
                self.POC, self.ATR, self.BINS,
                cache_dir=str(tmp_path)
            )

        assert result[(2025, 4, 5)] is not None, "Корректный ZIP должен обработаться"
        assert result[(2025, 4, 6)] is None,     "Сломанный ZIP должен вернуть None"

    def test_empty_input_returns_empty_dict(self, tmp_path):
        """
        Контракт: пустой dict на входе → пустой dict на выходе.
        """
        result = build_klines_delta_cache(
            {}, self.POC, self.ATR, self.BINS,
            cache_dir=str(tmp_path)
        )
        assert result == {}
