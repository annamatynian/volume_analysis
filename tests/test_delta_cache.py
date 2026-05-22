"""
tests/test_delta_cache.py
=========================
Unit tests for delta_cache.py — Этап 8D Шаг 3.

Тестируем:
  - build_cache_path()   — чистая функция, путь к parquet
  - build_delta_cache()  — ZIP → CVD → parquet (мокируем load_aggtrades_zip)

Принципы:
- load_aggtrades_zip() мокируется — тест не читает реальные ZIP (660 MB).
- Тесты проверяют контракт: что сохраняется в parquet и что возвращается.
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

from delta_cache import build_cache_path, build_delta_cache


# ---------------------------------------------------------------------------
# Вспомогательные фикстуры
# ---------------------------------------------------------------------------

def _make_fake_trades_df(n: int = 200, poc: float = 50_000.0,
                         atr: float = 500.0, seed: int = 42) -> pd.DataFrame:
    """
    Синтетический DataFrame aggTrades — результат load_aggtrades_zip().
    Колонки: price (float32), qty (float32), side (str).
    Половина сделок внутри зоны POC ± 1.5*ATR.
    """
    rng = np.random.default_rng(seed)
    zone_low  = poc - 1.5 * atr
    zone_high = poc + 1.5 * atr

    prices = np.concatenate([
        rng.uniform(zone_low, zone_high, n // 2).astype('float32'),
        rng.uniform(zone_high + 100, zone_high + 5000, n - n // 2).astype('float32'),
    ])
    qtys  = rng.uniform(0.001, 2.0, n).astype('float32')
    sides = np.where(np.arange(n) % 2 == 0, 'sell', 'buy')

    return pd.DataFrame({'price': prices, 'qty': qtys, 'side': sides})


def _make_fake_zip(path: str) -> None:
    """Создаёт минимальный ZIP — достаточно чтобы файл существовал на диске."""
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr("dummy.csv", "col1,col2\n1,2\n")


# ---------------------------------------------------------------------------
# build_cache_path — чистая функция
# ---------------------------------------------------------------------------

class TestBuildCachePath:
    """
    Контракт build_cache_path(year, month, cache_dir, kind='anchor') -> str:
    - Возвращает путь к parquet-файлу.
    - Имя файла: BTCUSDT-{kind}-{YYYY}-{MM:02d}.parquet
    - kind='anchor'   → monthly cache
    - kind='reaction' → daily cache (зарезервировано для Шага 5)
    - Не создаёт файл/папку — только строит путь.
    """

    def test_returns_string(self, tmp_path):
        result = build_cache_path(2025, 1, str(tmp_path))
        assert isinstance(result, str)

    def test_default_kind_is_anchor(self, tmp_path):
        """
        Контракт: kind='anchor' по умолчанию.
        """
        path = build_cache_path(2025, 1, str(tmp_path))
        assert "anchor" in os.path.basename(path)

    def test_filename_format_anchor(self, tmp_path):
        """
        Контракт: BTCUSDT-anchor-YYYY-MM.parquet с ведущим нулём месяца.
        """
        path = build_cache_path(2025, 1, str(tmp_path), kind='anchor')
        assert os.path.basename(path) == "BTCUSDT-anchor-2025-01.parquet"

    def test_filename_format_reaction(self, tmp_path):
        """
        Контракт: kind='reaction' → BTCUSDT-reaction-YYYY-MM.parquet.
        """
        path = build_cache_path(2025, 1, str(tmp_path), kind='reaction')
        assert os.path.basename(path) == "BTCUSDT-reaction-2025-01.parquet"

    def test_month_zero_padded(self, tmp_path):
        path = build_cache_path(2024, 9, str(tmp_path))
        assert "2024-09" in os.path.basename(path)

    def test_path_inside_cache_dir(self, tmp_path):
        path = build_cache_path(2025, 1, str(tmp_path))
        assert path.startswith(str(tmp_path))

    def test_does_not_create_file(self, tmp_path):
        """
        Контракт: чистая функция — никаких сайд-эффектов.
        """
        build_cache_path(2025, 1, str(tmp_path))
        parquet_path = os.path.join(str(tmp_path), "BTCUSDT-anchor-2025-01.parquet")
        assert not os.path.exists(parquet_path)

    def test_different_months_different_paths(self, tmp_path):
        p1 = build_cache_path(2024, 12, str(tmp_path))
        p2 = build_cache_path(2025,  1, str(tmp_path))
        assert p1 != p2


# ---------------------------------------------------------------------------
# build_delta_cache — мокируем load_aggtrades_zip
# ---------------------------------------------------------------------------

class TestBuildDeltaCache:
    """
    Контракт build_delta_cache(zip_paths, poc, atr, cache_dir) -> dict:
    - zip_paths: dict[(year, month) -> zip_path] — результат download_anchor_month
    - Для каждого ZIP: load_aggtrades_zip → calculate_cvd_in_zone → parquet
    - Возвращает dict[(year, month) -> parquet_path | None]
    - Idempotent: parquet уже существует → пропустить загрузку ZIP
    - При ошибке чтения ZIP → None для этого месяца, остальные обрабатываются
    - Создаёт cache_dir если не существует
    """

    POC = 50_000.0
    ATR = 500.0

    def _zip_paths(self, tmp_path: str, months: list) -> dict:
        """Создаёт фиктивные ZIP-файлы и возвращает словарь {(y,m): path}."""
        result = {}
        for year, month in months:
            path = os.path.join(str(tmp_path), f"BTCUSDT-aggTrades-{year}-{month:02d}.zip")
            _make_fake_zip(path)
            result[(year, month)] = path
        return result

    def test_returns_dict(self, tmp_path):
        """
        Контракт: возвращает dict.
        """
        zip_paths = self._zip_paths(str(tmp_path), [(2025, 1)])
        cache_dir = os.path.join(str(tmp_path), "cache")

        with patch("delta_cache.load_aggtrades_zip",
                   return_value=_make_fake_trades_df()):
            result = build_delta_cache(zip_paths, self.POC, self.ATR, cache_dir)

        assert isinstance(result, dict)

    def test_keys_match_input_months(self, tmp_path):
        """
        Контракт: ключи результата == ключи zip_paths.
        """
        months = [(2025, 1), (2025, 2)]
        zip_paths = self._zip_paths(str(tmp_path), months)
        cache_dir = os.path.join(str(tmp_path), "cache")

        with patch("delta_cache.load_aggtrades_zip",
                   return_value=_make_fake_trades_df()):
            result = build_delta_cache(zip_paths, self.POC, self.ATR, cache_dir)

        assert set(result.keys()) == {(2025, 1), (2025, 2)}

    def test_parquet_file_created_on_disk(self, tmp_path):
        """
        Контракт: parquet-файл создаётся на диске после обработки ZIP.
        """
        zip_paths = self._zip_paths(str(tmp_path), [(2025, 1)])
        cache_dir = os.path.join(str(tmp_path), "cache")

        with patch("delta_cache.load_aggtrades_zip",
                   return_value=_make_fake_trades_df()):
            result = build_delta_cache(zip_paths, self.POC, self.ATR, cache_dir)

        parquet_path = result[(2025, 1)]
        assert parquet_path is not None
        assert os.path.exists(parquet_path), "parquet file must exist on disk"

    def test_parquet_contains_required_columns(self, tmp_path):
        """
        Контракт: parquet содержит колонки cvd_slope, poc, atr, year, month.
        WHY: build_delta_cache() — это кэш для calculate_delta_context_score().
        Оркестратор читает cvd_slope из кэша, не пересчитывает из ZIP каждый раз.
        """
        zip_paths = self._zip_paths(str(tmp_path), [(2025, 1)])
        cache_dir = os.path.join(str(tmp_path), "cache")

        with patch("delta_cache.load_aggtrades_zip",
                   return_value=_make_fake_trades_df()):
            result = build_delta_cache(zip_paths, self.POC, self.ATR, cache_dir)

        df = pd.read_parquet(result[(2025, 1)])
        for col in ('cvd_slope', 'poc', 'atr', 'year', 'month'):
            assert col in df.columns, f"Missing column: {col}"

    def test_parquet_values_correct(self, tmp_path):
        """
        Контракт: значения poc, atr, year, month в parquet совпадают с входными.
        """
        zip_paths = self._zip_paths(str(tmp_path), [(2025, 3)])
        cache_dir = os.path.join(str(tmp_path), "cache")

        with patch("delta_cache.load_aggtrades_zip",
                   return_value=_make_fake_trades_df()):
            result = build_delta_cache(zip_paths, self.POC, self.ATR, cache_dir)

        df = pd.read_parquet(result[(2025, 3)])
        assert df['poc'].iloc[0]   == self.POC
        assert df['atr'].iloc[0]   == self.ATR
        assert df['year'].iloc[0]  == 2025
        assert df['month'].iloc[0] == 3

    def test_idempotent_skips_existing_parquet(self, tmp_path):
        """
        Контракт: parquet уже существует → load_aggtrades_zip НЕ вызывается.
        WHY: ZIP весит 660 MB — повторная загрузка недопустима.
        """
        zip_paths = self._zip_paths(str(tmp_path), [(2025, 1)])
        cache_dir = os.path.join(str(tmp_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Создаём parquet заранее
        parquet_path = build_cache_path(2025, 1, cache_dir)
        dummy_df = pd.DataFrame({'cvd_slope': [0.0], 'poc': [self.POC],
                                  'atr': [self.ATR], 'year': [2025], 'month': [1]})
        dummy_df.to_parquet(parquet_path, index=False)

        with patch("delta_cache.load_aggtrades_zip") as mock_load:
            result = build_delta_cache(zip_paths, self.POC, self.ATR, cache_dir)

        mock_load.assert_not_called()
        assert result[(2025, 1)] == parquet_path

    def test_returns_none_on_zip_error(self, tmp_path):
        """
        Контракт: ошибка чтения ZIP → None для этого месяца, не исключение.
        WHY: оркестратор должен продолжить с остальными месяцами.
        """
        zip_paths = self._zip_paths(str(tmp_path), [(2025, 1)])
        cache_dir = os.path.join(str(tmp_path), "cache")

        with patch("delta_cache.load_aggtrades_zip",
                   side_effect=Exception("Corrupted ZIP")):
            result = build_delta_cache(zip_paths, self.POC, self.ATR, cache_dir)

        assert result[(2025, 1)] is None

    def test_error_on_one_month_does_not_affect_others(self, tmp_path):
        """
        Контракт: ошибка одного месяца не прерывает обработку остальных.
        """
        months = [(2025, 1), (2025, 2)]
        zip_paths = self._zip_paths(str(tmp_path), months)
        cache_dir = os.path.join(str(tmp_path), "cache")

        call_count = 0

        def fake_load(path):
            nonlocal call_count
            call_count += 1
            if "2025-01" in path:
                raise Exception("Corrupted")
            return _make_fake_trades_df()

        with patch("delta_cache.load_aggtrades_zip", side_effect=fake_load):
            result = build_delta_cache(zip_paths, self.POC, self.ATR, cache_dir)

        assert result[(2025, 1)] is None
        assert result[(2025, 2)] is not None
        assert os.path.exists(result[(2025, 2)])

    def test_creates_cache_dir_if_not_exists(self, tmp_path):
        """
        Контракт: cache_dir создаётся автоматически.
        """
        zip_paths = self._zip_paths(str(tmp_path), [(2025, 1)])
        cache_dir = os.path.join(str(tmp_path), "deep", "cache")
        assert not os.path.exists(cache_dir)

        with patch("delta_cache.load_aggtrades_zip",
                   return_value=_make_fake_trades_df()):
            build_delta_cache(zip_paths, self.POC, self.ATR, cache_dir)

        assert os.path.isdir(cache_dir)

    def test_empty_zip_paths_returns_empty_dict(self, tmp_path):
        """
        Контракт: пустой zip_paths → пустой dict, не ошибка.
        """
        cache_dir = os.path.join(str(tmp_path), "cache")
        result = build_delta_cache({}, self.POC, self.ATR, cache_dir)
        assert result == {}


# ---------------------------------------------------------------------------
# build_reaction_delta — Шаг 2 Reaction Period
# ---------------------------------------------------------------------------

from delta_cache import build_reaction_delta


class TestBuildReactionDelta:
    """
    Контракт build_reaction_delta(zip_paths_daily, poc, atr, cache_dir) -> float:
    - zip_paths_daily: list[str] — пути к daily ZIP-файлам (14 штук).
    - Для каждого ZIP: load_aggtrades_zip → calculate_cvd_in_zone → берём последнее значение CVD.
    - recent_delta = сумма последних значений CVD по всем дням (float).
    - Кэш: BTCUSDT-reaction-YYYY-MM-DD.parquet (в cache_dir).
    - Idempotent: парчайное уже есть → берёт сохранённое значение, не читает ZIP.
    - Сломанный ZIP → 0.0 для этого дня, остальные обрабатываются.
    - Пустой список → 0.0.

    Логика recent_delta:
      Для каждого daily ZIP: calculate_cvd_in_zone() → (cvd_series, slope).
      recent_delta дня = cvd_series.iloc[-1] (последнее значение CVD).
      WHY последнее CVD, не slope: абсолютная дельта (buy-sell) отражает
      количественное давление; slope нужен для anchor period (качество уровня),
      recent_delta — для reaction period (преобладание через 14 дней).
      recent_delta = sum(cvd_series.iloc[-1] для каждого дня).
    """

    POC = 50_000.0
    ATR = 500.0

    def _make_fake_zip(self, path: str) -> None:
        with zipfile.ZipFile(path, 'w') as zf:
            zf.writestr("dummy.csv", "col1,col2\n1,2\n")

    def _zip_paths_daily(self, tmp_path: str, n_days: int = 3) -> list:
        """
        Создаёт n_days фиктивных daily ZIP и возвращает list[str] путей.
        Имя включает дату — будет использоваться для build_reaction_zip_cache_path.
        """
        paths = []
        for d in range(1, n_days + 1):
            path = os.path.join(
                str(tmp_path),
                f"BTCUSDT-aggTrades-2025-04-{d:02d}.zip"
            )
            self._make_fake_zip(path)
            paths.append(path)
        return paths

    def test_returns_float(self, tmp_path):
        """
        Контракт: возвращает float.
        """
        cache_dir = os.path.join(str(tmp_path), "cache")
        zip_paths = self._zip_paths_daily(str(tmp_path))

        fake_df = _make_fake_trades_df(poc=self.POC, atr=self.ATR)
        with patch("delta_cache.load_aggtrades_zip", return_value=fake_df):
            result = build_reaction_delta(zip_paths, self.POC, self.ATR, cache_dir)

        assert isinstance(result, float)

    def test_empty_zip_paths_returns_zero(self, tmp_path):
        """
        Контракт: пустой список → 0.0.
        WHY: оркестратор должен продолжать работу если нет daily ZIP.
        """
        cache_dir = os.path.join(str(tmp_path), "cache")
        result = build_reaction_delta([], self.POC, self.ATR, cache_dir)
        assert result == 0.0

    def test_positive_delta_when_buys_dominate(self, tmp_path):
        """
        Контракт: покупки доминируют → recent_delta > 0.
        WHY: CVD.iloc[-1] > 0 если все сделки — покупки.
        """
        cache_dir = os.path.join(str(tmp_path), "cache")
        zip_paths = self._zip_paths_daily(str(tmp_path), n_days=2)

        poc, atr = self.POC, self.ATR
        # Все сделки покупки внутри зоны
        df_buys = pd.DataFrame({
            'price': np.linspace(poc - atr, poc + atr, 20),
            'qty':   np.ones(20),
            'side':  ['buy'] * 20,
        })
        with patch("delta_cache.load_aggtrades_zip", return_value=df_buys):
            result = build_reaction_delta(zip_paths, poc, atr, cache_dir)

        assert result > 0, f"Expected positive recent_delta for buy-dominated days, got {result}"

    def test_negative_delta_when_sells_dominate(self, tmp_path):
        """
        Контракт: продажи доминируют → recent_delta < 0.
        """
        cache_dir = os.path.join(str(tmp_path), "cache")
        zip_paths = self._zip_paths_daily(str(tmp_path), n_days=2)

        poc, atr = self.POC, self.ATR
        df_sells = pd.DataFrame({
            'price': np.linspace(poc - atr, poc + atr, 20),
            'qty':   np.ones(20),
            'side':  ['sell'] * 20,
        })
        with patch("delta_cache.load_aggtrades_zip", return_value=df_sells):
            result = build_reaction_delta(zip_paths, poc, atr, cache_dir)

        assert result < 0, f"Expected negative recent_delta for sell-dominated days, got {result}"

    def test_parquet_created_per_day(self, tmp_path):
        """
        Контракт: для каждого daily ZIP создаётся parquet-файл кэша.
        WHY: idempotent pipeline — повторный запуск не читает 22 MB ZIP.
        """
        cache_dir = os.path.join(str(tmp_path), "cache")
        zip_paths = self._zip_paths_daily(str(tmp_path), n_days=2)

        fake_df = _make_fake_trades_df(poc=self.POC, atr=self.ATR)
        with patch("delta_cache.load_aggtrades_zip", return_value=fake_df):
            build_reaction_delta(zip_paths, self.POC, self.ATR, cache_dir)

        # Для дня 2025-04-01 и 2025-04-02 должны быть parquet-файлы reaction-кэша
        from delta_cache import build_reaction_zip_cache_path
        for day in (1, 2):
            p = build_reaction_zip_cache_path(
                f"BTCUSDT-aggTrades-2025-04-{day:02d}.zip", cache_dir
            )
            assert os.path.exists(p), f"Expected parquet cache for day {day} at {p}"

    def test_idempotent_skips_existing_parquet(self, tmp_path):
        """
        Контракт: parquet уже есть → load_aggtrades_zip НЕ вызывается.
        WHY: повторный запуск не читает 22 MB ZIP.
        """
        cache_dir = os.path.join(str(tmp_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        zip_paths = self._zip_paths_daily(str(tmp_path), n_days=1)

        # Создаём parquet заранее
        from delta_cache import build_reaction_zip_cache_path
        p = build_reaction_zip_cache_path(
            os.path.basename(zip_paths[0]), cache_dir
        )
        dummy = pd.DataFrame([{'recent_cvd': 42.0}])
        dummy.to_parquet(p, index=False)

        with patch("delta_cache.load_aggtrades_zip") as mock_load:
            build_reaction_delta(zip_paths, self.POC, self.ATR, cache_dir)

        mock_load.assert_not_called()

    def test_broken_zip_contributes_zero(self, tmp_path):
        """
        Контракт: ошибка чтения ZIP → день вносит 0.0, остальные обрабатываются.
        WHY: аналогично build_delta_cache — один сломанный день не стопит pipeline.
        """
        cache_dir = os.path.join(str(tmp_path), "cache")
        zip_paths = self._zip_paths_daily(str(tmp_path), n_days=2)

        poc, atr = self.POC, self.ATR
        # zip_paths[0] — сломанный, zip_paths[1] — все покупки
        df_buys = pd.DataFrame({
            'price': np.linspace(poc - atr, poc + atr, 10),
            'qty':   np.ones(10),
            'side':  ['buy'] * 10,
        })
        call_idx = [0]

        def fake_load(path):
            call_idx[0] += 1
            if call_idx[0] == 1:
                raise Exception("Corrupted")
            return df_buys

        with patch("delta_cache.load_aggtrades_zip", side_effect=fake_load):
            result = build_reaction_delta(zip_paths, poc, atr, cache_dir)

        # Только второй день дал дельту — итог должен быть > 0
        assert result > 0, (
            f"One broken day + one buy day should give positive delta, got {result}"
        )
