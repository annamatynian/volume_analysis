"""
tests/test_onchain_cache.py
===========================
Unit tests for onchain_cache.py — локальный parquet-кэш для BGeometrics API.

Мотивация: BGeometrics бесплатный тариф = 8 запросов/час.
Решение: кэшируем ответы в parquet, дёргаем API только если кэш старше 23 часов.

Принципы:
- Нет реальных сетевых вызовов — API мокируется через unittest.mock.
- Временные файлы через tmp_path (pytest fixture) — не засоряем проект.
- Тесты проверяют КОНТРАКТЫ функций, не детали реализации.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from onchain_cache import (
    build_cache_path,
    is_cache_fresh,
    load_from_cache,
    save_to_cache,
    CachedBGeometricsClient,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loss_df(n: int = 5) -> pd.DataFrame:
    """Минимальный DataFrame realized_loss_usd для тестов round-trip."""
    dates = [datetime(2024, 3, 1) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        'date': pd.to_datetime(dates),
        'realized_loss_usd': [float(i * 100_000_000) for i in range(1, n + 1)],
    })


def _make_sth_df(n: int = 3) -> pd.DataFrame:
    """Минимальный DataFrame sth_realized_price."""
    dates = [datetime(2024, 3, 1) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        'date': pd.to_datetime(dates),
        'sth_realized_price': [60000.0 + i * 500 for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Группа 1: build_cache_path
# ---------------------------------------------------------------------------

class TestBuildCachePath:

    def test_returns_path_inside_cache_dir(self, tmp_path):
        """build_cache_path возвращает путь внутри указанной директории."""
        result = build_cache_path("sth-realized-price", cache_dir=tmp_path)
        assert str(tmp_path) in str(result)

    def test_different_metrics_give_different_paths(self, tmp_path):
        """Разные метрики → разные пути (нет коллизий)."""
        path_a = build_cache_path("sth-realized-price", cache_dir=tmp_path)
        path_b = build_cache_path("realized-loss-lth-usd", cache_dir=tmp_path)
        assert path_a != path_b

    def test_returns_parquet_extension(self, tmp_path):
        """Кэш-файл имеет расширение .parquet."""
        result = build_cache_path("sopr", cache_dir=tmp_path)
        assert Path(result).suffix == ".parquet"


# ---------------------------------------------------------------------------
# Группа 2: is_cache_fresh
# ---------------------------------------------------------------------------

class TestIsCacheFresh:

    def test_fresh_file_returns_true(self, tmp_path):
        """Файл изменён 5 минут назад → кэш свежий → True."""
        cache_file = tmp_path / "test.parquet"
        cache_file.write_text("dummy")

        # Устанавливаем mtime = сейчас - 5 минут
        fresh_time = (datetime.now() - timedelta(minutes=5)).timestamp()
        os.utime(cache_file, (fresh_time, fresh_time))

        assert is_cache_fresh(cache_file, max_age_hours=23) is True

    def test_old_file_returns_false(self, tmp_path):
        """Файл изменён 25 часов назад → кэш устарел → False."""
        cache_file = tmp_path / "test.parquet"
        cache_file.write_text("dummy")

        old_time = (datetime.now() - timedelta(hours=25)).timestamp()
        os.utime(cache_file, (old_time, old_time))

        assert is_cache_fresh(cache_file, max_age_hours=23) is False

    def test_missing_file_returns_false(self, tmp_path):
        """Файл не существует → False (нет кэша)."""
        missing = tmp_path / "nonexistent.parquet"
        assert is_cache_fresh(missing, max_age_hours=23) is False


# ---------------------------------------------------------------------------
# Группа 3: load_from_cache / save_to_cache
# ---------------------------------------------------------------------------

class TestCacheRoundTrip:

    def test_save_and_load_preserves_data(self, tmp_path):
        """Сохранить DataFrame → загрузить → данные совпадают."""
        df = _make_loss_df(5)
        cache_file = tmp_path / "test.parquet"

        save_to_cache(df, cache_file)
        loaded = load_from_cache(cache_file)

        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)
        pd.testing.assert_frame_equal(
            loaded.reset_index(drop=True),
            df.reset_index(drop=True),
        )

    def test_dates_survive_round_trip(self, tmp_path):
        """Колонка date сохраняется как datetime64 после round-trip."""
        df = _make_loss_df(3)
        cache_file = tmp_path / "test.parquet"

        save_to_cache(df, cache_file)
        loaded = load_from_cache(cache_file)

        assert pd.api.types.is_datetime64_any_dtype(loaded['date'])

    def test_empty_dataframe_round_trip(self, tmp_path):
        """Пустой DataFrame сохраняется и загружается без ошибок."""
        df = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'),
                           'realized_loss_usd': pd.Series(dtype='float64')})
        cache_file = tmp_path / "empty.parquet"

        save_to_cache(df, cache_file)
        loaded = load_from_cache(cache_file)

        assert len(loaded) == 0
        assert 'date' in loaded.columns


# ---------------------------------------------------------------------------
# Группа 4: CachedBGeometricsClient
# ---------------------------------------------------------------------------

class TestCachedBGeometricsClient:

    def _make_client(self, cache_dir: Path) -> CachedBGeometricsClient:
        """Создаём клиент с мок-обёрнутым BGeometricsClient."""
        mock_inner = MagicMock()
        return CachedBGeometricsClient(client=mock_inner, cache_dir=cache_dir)

    def test_fresh_cache_skips_api_call(self, tmp_path):
        """Свежий кэш → внутренний API-клиент НЕ вызывается."""
        df = _make_loss_df(5)
        client = self._make_client(tmp_path)

        # Заранее кладём свежий кэш
        cache_path = build_cache_path("realized-loss-lth-usd", cache_dir=tmp_path)
        save_to_cache(df, cache_path)
        fresh_time = (datetime.now() - timedelta(minutes=10)).timestamp()
        os.utime(cache_path, (fresh_time, fresh_time))

        result = asyncio.run(client.get_realized_loss_lth_usd())

        # API не должен был вызываться
        client._client.get_realized_loss_lth_usd.assert_not_called()
        assert len(result) == len(df)

    def test_missing_cache_calls_api_and_saves(self, tmp_path):
        """Нет кэша → API вызывается → данные сохраняются в parquet."""
        df = _make_loss_df(4)
        mock_inner = MagicMock()
        mock_inner.get_realized_loss_lth_usd = AsyncMock(return_value=df)

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)

        result = asyncio.run(client.get_realized_loss_lth_usd())

        mock_inner.get_realized_loss_lth_usd.assert_called_once()
        assert len(result) == len(df)

        # Кэш-файл должен появиться на диске
        cache_path = build_cache_path("realized-loss-lth-usd", cache_dir=tmp_path)
        assert Path(cache_path).exists()

    def test_stale_cache_calls_api_and_updates(self, tmp_path):
        """Устаревший кэш → API вызывается → кэш перезаписывается."""
        old_df = _make_loss_df(3)
        new_df = _make_loss_df(7)  # Больше строк — отличается от старого

        # Кладём старый кэш
        cache_path = build_cache_path("realized-loss-lth-usd", cache_dir=tmp_path)
        save_to_cache(old_df, cache_path)
        old_time = (datetime.now() - timedelta(hours=25)).timestamp()
        os.utime(cache_path, (old_time, old_time))

        mock_inner = MagicMock()
        mock_inner.get_realized_loss_lth_usd = AsyncMock(return_value=new_df)

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        result = asyncio.run(client.get_realized_loss_lth_usd())

        mock_inner.get_realized_loss_lth_usd.assert_called_once()
        assert len(result) == len(new_df)

    def test_api_error_falls_back_to_stale_cache(self, tmp_path):
        """API бросает исключение (429) → возвращает устаревший кэш как fallback."""
        stale_df = _make_loss_df(5)

        # Кладём старый кэш
        cache_path = build_cache_path("realized-loss-lth-usd", cache_dir=tmp_path)
        save_to_cache(stale_df, cache_path)
        old_time = (datetime.now() - timedelta(hours=25)).timestamp()
        os.utime(cache_path, (old_time, old_time))

        mock_inner = MagicMock()
        mock_inner.get_realized_loss_lth_usd = AsyncMock(
            side_effect=Exception("429 RATE_LIMIT_HOUR_EXCEEDED")
        )

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        result = asyncio.run(client.get_realized_loss_lth_usd())

        # Должен вернуть данные из старого кэша, не бросить исключение
        assert len(result) == len(stale_df)


# ---------------------------------------------------------------------------
# Группа 5: Holder Structure — 5 новых методов cache-слоя
# ---------------------------------------------------------------------------

class TestHolderStructureCacheMethods:
    """
    WHY тесты на metric-ключ: каждый из 5 методов должен писать в свой
    parquet-файл. Если два метода используют один ключ — они затирают
    данные друг друга (silent data corruption).

    WHY тесты на вызов inner-метода: cache-слой — делегат, а не реализация.
    Если CachedBGeometricsClient.get_lth_mvrv() вызовет get_sth_mvrv() —
    данные будут неверными без какого-либо исключения.
    """

    METHODS = [
        # (cache_method_name, inner_method_name, expected_metric_key)
        ('get_lth_mvrv',                  'get_lth_mvrv',                  'lth-mvrv'),
        ('get_sth_mvrv',                  'get_sth_mvrv',                  'sth-mvrv'),
        ('get_lth_sopr',                  'get_lth_sopr',                  'lth-sopr'),
        ('get_lth_net_position_change_30d', 'get_lth_net_position_change_30d', 'lth-net-position-change-30d-btc'),
        ('get_sth_net_position_change_30d', 'get_sth_net_position_change_30d', 'sth-net-position-change-30d-btc'),
    ]

    def _make_df(self, value_col: str) -> pd.DataFrame:
        return pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01']),
            value_col: [1.5],
        })

    @pytest.mark.parametrize('cache_method,inner_method,metric_key', METHODS)
    def test_calls_correct_inner_method(self, tmp_path, cache_method, inner_method, metric_key):
        """
        WHY: cache-метод X должен делегировать inner-клиенту метод X,
        а не соседний. Ошибка делегирования — silent wrong data.
        """
        df = self._make_df('value')
        mock_inner = MagicMock()
        # Настраиваем нужный inner-метод как AsyncMock
        setattr(mock_inner, inner_method, AsyncMock(return_value=df))

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        asyncio.run(getattr(client, cache_method)())

        getattr(mock_inner, inner_method).assert_called_once()

    @pytest.mark.parametrize('cache_method,inner_method,metric_key', METHODS)
    def test_each_method_uses_unique_cache_key(self, tmp_path, cache_method, inner_method, metric_key):
        """
        WHY: уникальный metric_key → уникальный parquet-файл.
        Если два метода используют один ключ — они перезаписывают
        данные друг друга. Тест фиксирует ожидаемый путь к файлу.
        """
        expected_path = build_cache_path(metric_key, cache_dir=tmp_path)
        df = self._make_df('value')
        mock_inner = MagicMock()
        setattr(mock_inner, inner_method, AsyncMock(return_value=df))

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        asyncio.run(getattr(client, cache_method)())

        # Файл должен появиться именно по ожидаемому пути
        assert expected_path.exists(), (
            f'{cache_method} должен писать кэш в {expected_path.name}, '
            f'но файл не создан'
        )


# ---------------------------------------------------------------------------
# Группа 6: М-15 — get_funding_rate_series в CachedBGeometricsClient
# ---------------------------------------------------------------------------

class TestCachedFundingRate:
    """
    WHY этот класс: get_funding_rate_series — нестандартный метод.
    Принимает records: int вместо start_date/end_date.
    API игнорирует параметры дат и возвращает всю историю (~3178 записей).
    Кэш-слой кэширует всю историю и возвращает tail(records).
    Без этого метода в CachedBGeometricsClient — оркестратор получает
    AttributeError и М-15 всегда попадает в 'Н/Д' в Signal Alignment.
    """

    def _make_funding_df(self, n: int = 5) -> pd.DataFrame:
        """
        Нейтральный мок-DataFrame: значения funding_rate не похожи
        на реальные ставки Binance (0.0001), чтобы не создавать
        ложной связи с production-константами.
        """
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='8h'),
            'funding_rate': [float(i) * 0.1 for i in range(1, n + 1)],
        })

    # ------------------------------------------------------------------
    # Тест 1: метод существует и делегирует к inner client
    # ------------------------------------------------------------------

    def test_method_exists_and_delegates(self, tmp_path):
        """
        WHY: отсутствие делегации — прямой источник текущего
        '[WARN] М-15: CachedBGeometricsClient object has no attribute
        get_funding_rate_series' в оркестраторе. М-15 попадает в Н/Д.
        """
        df = self._make_funding_df(5)
        mock_inner = MagicMock()
        mock_inner.get_funding_rate_series = AsyncMock(return_value=df)

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        asyncio.run(client.get_funding_rate_series(records=5))

        # WHY: если метод не делегирует — inner никогда не вызывается,
        # оркестратор никогда не получает данные М-15.
        mock_inner.get_funding_rate_series.assert_called_once()

    # ------------------------------------------------------------------
    # Тест 2: при свежем кэше inner client НЕ вызывается
    # ------------------------------------------------------------------

    def test_fresh_cache_skips_inner_client(self, tmp_path):
        """
        WHY: кэш-слой должен защищать от лимита 10 req/hour.
        Если при свежем кэше всё равно вызывается API — лимит
        исчерпывается за один прогон оркестратора.
        """
        df = self._make_funding_df(5)
        # Сохраняем свежий кэш вручную
        cache_path = build_cache_path('funding-rate', tmp_path)
        save_to_cache(df, cache_path)
        # mtime по умолчанию = сейчас → кэш свежий

        mock_inner = MagicMock()
        mock_inner.get_funding_rate_series = AsyncMock(return_value=df)

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        asyncio.run(client.get_funding_rate_series(records=5))

        # WHY: inner НЕ должен вызываться если кэш свежий —
        # иначе кэш не выполняет своей функции защиты от rate-limit.
        mock_inner.get_funding_rate_series.assert_not_called()

    # ------------------------------------------------------------------
    # Тест 3: при отсутствии кэша вызывается inner, результат кэшируется
    # ------------------------------------------------------------------

    def test_no_cache_calls_inner_and_saves(self, tmp_path):
        """
        WHY: контракт cache-first — при пустом кэше данные должны
        прийти от inner и сохраниться. Если не сохраняются —
        следующий вызов снова пойдёт в API, быстро исчерпав лимит.
        """
        df = self._make_funding_df(5)
        mock_inner = MagicMock()
        mock_inner.get_funding_rate_series = AsyncMock(return_value=df)

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        asyncio.run(client.get_funding_rate_series(records=5))

        expected_cache = build_cache_path('funding-rate', tmp_path)
        # WHY: файл кэша должен появиться после вызова —
        # иначе каждый прогон оркестратора делает новый API-запрос.
        assert expected_cache.exists(), (
            'После вызова API данные должны быть сохранены в кэш '
            f'по пути {expected_cache.name}'
        )

    # ------------------------------------------------------------------
    # Тест 4: возвращает DataFrame с колонками ['date', 'funding_rate']
    # ------------------------------------------------------------------

    def test_returns_dataframe_with_correct_columns(self, tmp_path):
        """
        WHY: оркестратор ожидает колонки ['date', 'funding_rate'].
        Если имена колонок отличаются — KeyError при вычислении MA,
        М-15 падает с необработанным исключением.
        """
        df = self._make_funding_df(5)
        mock_inner = MagicMock()
        mock_inner.get_funding_rate_series = AsyncMock(return_value=df)

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        result = asyncio.run(client.get_funding_rate_series(records=5))

        # WHY: контракт колонок — оркестратор обращается к df['funding_rate']
        # и df['date'] напрямую; отсутствие любой из них = silent wrong calc.
        assert 'date' in result.columns, (
            "Колонка 'date' обязательна — оркестратор сортирует по дате"
        )
        assert 'funding_rate' in result.columns, (
            "Колонка 'funding_rate' обязательна — по ней считается 30d MA"
        )
        assert isinstance(result, pd.DataFrame), (
            'get_funding_rate_series должен возвращать pd.DataFrame, '
            'не список или другой тип'
        )

    # ------------------------------------------------------------------
    # Тест 5 (граница): records > len(кэша) — не падает, возвращает всё
    # ------------------------------------------------------------------

    def test_records_exceeds_cache_length_returns_all(self, tmp_path):
        """
        WHY граничное значение: если в кэше 3 строки, а запросили
        records=100 — tail(100) на DataFrame из 3 строк должен вернуть
        3 строки, не вызвать исключение. Проверяем что cache-слой
        не добавляет дополнительных ограничений сверх pandas-поведения.
        """
        small_df = self._make_funding_df(3)  # 3 строки
        cache_path = build_cache_path('funding-rate', tmp_path)
        save_to_cache(small_df, cache_path)

        mock_inner = MagicMock()
        mock_inner.get_funding_rate_series = AsyncMock()

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        result = asyncio.run(client.get_funding_rate_series(records=100))

        # WHY: pandas tail(N) при N > len возвращает весь DataFrame —
        # cache-слой не должен нарушать это поведение или вызывать IndexError.
        assert len(result) == 3, (
            f'Ожидали 3 строки (весь кэш), получили {len(result)}. '
            'tail(N) при N > len должен возвращать все доступные строки.'
        )

    # ------------------------------------------------------------------
    # Тест 6: уникальный cache key 'funding-rate'
    # ------------------------------------------------------------------

    def test_uses_unique_cache_key(self, tmp_path):
        """
        WHY: уникальный metric_key → уникальный parquet-файл.
        Если funding-rate перепутать с другим ключом —
        данные funding rate перезапишут другую метрику (silent corruption).
        """
        df = self._make_funding_df(5)
        mock_inner = MagicMock()
        mock_inner.get_funding_rate_series = AsyncMock(return_value=df)

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        asyncio.run(client.get_funding_rate_series(records=5))

        expected_path = build_cache_path('funding-rate', tmp_path)
        # WHY: файл должен называться 'funding-rate.parquet' —
        # оркестратор и cache-invalidation опираются на этот ключ.
        assert expected_path.exists(), (
            f'Кэш М-15 должен писаться в funding-rate.parquet, '
            f'но файл не найден в {tmp_path}'
        )


# ===========================================================================
# P1–P5: Expansion — cache behavior (PLAN_ONCHAIN_EXPANSION.md 2026-05-10)
# ===========================================================================

class TestExpansionCacheMethods:
    """
    WHY параметризованный подход: все 6 расширенных методов следуют одному
    cache-first контракту через _get_with_cache(). Параметризация
    избавляет от копирования 4 одинаковых теста × 6 методов.

    METHODS: (cache_method, inner_method, metric_key, value_col)
      - value_col: используется только для построения тестового DataFrame.
      - hodl-waves: 'value' как placeholder — реальные имена
        когортных колонок определяются после диагностики API.
    """

    METHODS = [
        # (cache_method, inner_method, metric_key, value_col)
        ('get_sth_sopr',         'get_sth_sopr',         'sth-sopr',             'sth_sopr'),
        ('get_exchange_netflow', 'get_exchange_netflow', 'exchange-netflow-btc', 'exchange_netflow_btc'),
        ('get_nupl_lth',         'get_nupl_lth',         'nupl-lth',             'nupl_lth'),
        ('get_nupl_sth',         'get_nupl_sth',         'nupl-sth',             'nupl_sth'),
        ('get_etf_flow',         'get_etf_flow',         'etf-flow-btc',         'etf_flow_btc'),
        ('get_hodl_waves',       'get_hodl_waves',       'hodl-waves-supply',    'value'),
    ]

    def _make_df(self, value_col: str, n: int = 3) -> pd.DataFrame:
        """Минимальный тестовый DataFrame: date + одна value-колонка."""
        return pd.DataFrame({
            'date': pd.to_datetime([f'2024-01-0{i+1}' for i in range(n)]),
            value_col: [float(i) for i in range(1, n + 1)],
        })

    @pytest.mark.parametrize('cache_method,inner_method,metric_key,value_col', METHODS)
    def test_calls_correct_inner_method(self, tmp_path, cache_method, inner_method, metric_key, value_col):
        """
        WHY: cache-метод X должен делегировать inner-методу X, а не соседнему.
        Ошибка делегирования — silent wrong data: нет исключения,
        просто неверные цифры в [HOLDER STRUCTURE] / [EXCHANGE FLOWS].
        """
        df = self._make_df(value_col)
        mock_inner = MagicMock()
        setattr(mock_inner, inner_method, AsyncMock(return_value=df))

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        asyncio.run(getattr(client, cache_method)())

        getattr(mock_inner, inner_method).assert_called_once()

    @pytest.mark.parametrize('cache_method,inner_method,metric_key,value_col', METHODS)
    def test_each_method_uses_unique_cache_key(self, tmp_path, cache_method, inner_method, metric_key, value_col):
        """
        WHY: уникальный metric_key → уникальный parquet-файл.
        Коллизия ключей → два метода перезаписывают один файл
        → silent data corruption без каких-либо исключений.
        """
        expected_path = build_cache_path(metric_key, cache_dir=tmp_path)
        df = self._make_df(value_col)
        mock_inner = MagicMock()
        setattr(mock_inner, inner_method, AsyncMock(return_value=df))

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        asyncio.run(getattr(client, cache_method)())

        assert expected_path.exists(), (
            f'{cache_method} должен писать кэш в {expected_path.name}, '
            f'но файл не создан'
        )

    @pytest.mark.parametrize('cache_method,inner_method,metric_key,value_col', METHODS)
    def test_missing_cache_calls_api_and_saves(self, tmp_path, cache_method, inner_method, metric_key, value_col):
        """
        WHY: нет кэша → API вызывается ровно один раз → parquet появляется
        на диске. Без сохранения каждый запуск оркестратора
        тратит 1 из 10 req/hour (лимит BGeometrics free tier).
        """
        df = self._make_df(value_col)
        mock_inner = MagicMock()
        setattr(mock_inner, inner_method, AsyncMock(return_value=df))

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        result = asyncio.run(getattr(client, cache_method)())

        getattr(mock_inner, inner_method).assert_called_once()
        assert len(result) == len(df)
        assert build_cache_path(metric_key, cache_dir=tmp_path).exists()

    @pytest.mark.parametrize('cache_method,inner_method,metric_key,value_col', METHODS)
    def test_fresh_cache_skips_api_call(self, tmp_path, cache_method, inner_method, metric_key, value_col):
        """
        WHY: свежий кэш (< 23ч) → внутренний API-клиент не вызывается.
        Основная цель кэша: повторные запуски оркестратора = 0 req/hour.
        """
        df = self._make_df(value_col)
        cache_path = build_cache_path(metric_key, cache_dir=tmp_path)
        save_to_cache(df, cache_path)
        fresh_time = (datetime.now() - timedelta(minutes=10)).timestamp()
        os.utime(cache_path, (fresh_time, fresh_time))

        mock_inner = MagicMock()
        setattr(mock_inner, inner_method, AsyncMock(return_value=df))

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        asyncio.run(getattr(client, cache_method)())

        getattr(mock_inner, inner_method).assert_not_called()

    @pytest.mark.parametrize('cache_method,inner_method,metric_key,value_col', METHODS)
    def test_api_error_falls_back_to_stale_cache(self, tmp_path, cache_method, inner_method, metric_key, value_col):
        """
        WHY: API бросает 429 + устаревший кэш есть → возвращаем кэш,
        не бросаем исключение. Без fallback весь on-chain слой
        падает при превышении лимита req/hour.
        """
        df = self._make_df(value_col)
        cache_path = build_cache_path(metric_key, cache_dir=tmp_path)
        save_to_cache(df, cache_path)
        old_time = (datetime.now() - timedelta(hours=25)).timestamp()
        os.utime(cache_path, (old_time, old_time))

        mock_inner = MagicMock()
        setattr(mock_inner, inner_method, AsyncMock(
            side_effect=Exception('429 RATE_LIMIT_HOUR_EXCEEDED')
        ))

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        result = asyncio.run(getattr(client, cache_method)())

        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Группа 6: sth_realized_price — cache behavior (регрессия против 429-bug)
# ---------------------------------------------------------------------------

class TestSthRealizedPriceCache:
    """
    WHY эти тесты: get_sth_realized_price используется OnChainValidator.initialize().
    Без кэша каждый запуск оркестратора расходует 1 из 10 req/hour.
    При 429 без fallback-кэша весь on-chain слой падает (_onchain_available=False).
    Тесты фиксируют cache-first контракт для этого эндпоинта.
    """

    def _make_sth_df(self, n: int = 3) -> pd.DataFrame:
        dates = [datetime(2024, 3, 1) + timedelta(days=i) for i in range(n)]
        return pd.DataFrame({
            'date': pd.to_datetime(dates),
            'sth_realized_price': [75000.0 + i * 500 for i in range(n)],
        })

    def test_missing_cache_calls_api_and_saves(self, tmp_path):
        """
        Кэша нет → API вызывается один раз → parquet сохраняется на диск.

        WHY: без parquet каждый запуск оркестратора расходует лимит req/hour.
        Этот тест фиксирует что метод следует cache-first паттерну,
        аналогичному get_realized_loss_lth_usd.
        """
        df = self._make_sth_df(3)
        mock_inner = MagicMock()
        mock_inner.get_sth_realized_price = AsyncMock(return_value=df)

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        result = asyncio.run(client.get_sth_realized_price())

        mock_inner.get_sth_realized_price.assert_called_once()
        assert len(result) == len(df)
        cache_path = build_cache_path("sth-realized-price", cache_dir=tmp_path)
        assert Path(cache_path).exists()

    def test_api_error_falls_back_to_stale_cache(self, tmp_path):
        """
        API возвращает 429 + устаревший кэш есть → возвращает кэш, не бросает.

        WHY: именно этот сценарий убил [HOLDER STRUCTURE] и [ON-CHAIN: LTH REALIZED LOSS]
        в run_holder_2026-05-09.txt: OnChainValidator.initialize() получал raw client,
        429 пробрасывался наверх → _onchain_available=False.
        С CachedBGeometricsClient fallback защищает от этого.
        """
        stale_df = self._make_sth_df(5)
        cache_path = build_cache_path("sth-realized-price", cache_dir=tmp_path)
        save_to_cache(stale_df, cache_path)
        old_time = (datetime.now() - timedelta(hours=25)).timestamp()
        os.utime(cache_path, (old_time, old_time))

        mock_inner = MagicMock()
        mock_inner.get_sth_realized_price = AsyncMock(
            side_effect=Exception("429 Too Many Requests")
        )

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        result = asyncio.run(client.get_sth_realized_price())

        assert len(result) == len(stale_df)


# ---------------------------------------------------------------------------
# Группа 5: Holder Structure — 5 новых методов cache-слоя
# ---------------------------------------------------------------------------

class TestHolderStructureCacheMethods:
    """
    WHY тесты на metric-ключ: каждый из 5 методов должен писать в свой
    parquet-файл. Если два метода используют один ключ — они затирают
    данные друг друга (silent data corruption).

    WHY тесты на вызов inner-метода: cache-слой — делегат, а не реализация.
    Если CachedBGeometricsClient.get_lth_mvrv() вызовет get_sth_mvrv() —
    данные будут неверными без какого-либо исключения.
    """

    METHODS = [
        # (cache_method_name, inner_method_name, expected_metric_key)
        ('get_lth_mvrv',                  'get_lth_mvrv',                  'lth-mvrv'),
        ('get_sth_mvrv',                  'get_sth_mvrv',                  'sth-mvrv'),
        ('get_lth_sopr',                  'get_lth_sopr',                  'lth-sopr'),
        ('get_lth_net_position_change_30d', 'get_lth_net_position_change_30d', 'lth-net-position-change-30d-btc'),
        ('get_sth_net_position_change_30d', 'get_sth_net_position_change_30d', 'sth-net-position-change-30d-btc'),
    ]

    def _make_df(self, value_col: str) -> pd.DataFrame:
        return pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01']),
            value_col: [1.5],
        })

    @pytest.mark.parametrize('cache_method,inner_method,metric_key', METHODS)
    def test_calls_correct_inner_method(self, tmp_path, cache_method, inner_method, metric_key):
        """
        WHY: cache-метод X должен делегировать inner-клиенту метод X,
        а не соседний. Ошибка делегирования — silent wrong data.
        """
        df = self._make_df('value')
        mock_inner = MagicMock()
        # Настраиваем нужный inner-метод как AsyncMock
        setattr(mock_inner, inner_method, AsyncMock(return_value=df))

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        asyncio.run(getattr(client, cache_method)())

        getattr(mock_inner, inner_method).assert_called_once()

    @pytest.mark.parametrize('cache_method,inner_method,metric_key', METHODS)
    def test_each_method_uses_unique_cache_key(self, tmp_path, cache_method, inner_method, metric_key):
        """
        WHY: уникальный metric_key → уникальный parquet-файл.
        Если два метода используют один ключ — они перезаписывают
        данные друг друга. Тест фиксирует ожидаемый путь к файлу.
        """
        expected_path = build_cache_path(metric_key, cache_dir=tmp_path)
        df = self._make_df('value')
        mock_inner = MagicMock()
        setattr(mock_inner, inner_method, AsyncMock(return_value=df))

        client = CachedBGeometricsClient(client=mock_inner, cache_dir=tmp_path)
        asyncio.run(getattr(client, cache_method)())

        # Файл должен появиться именно по ожидаемому пути
        assert expected_path.exists(), (
            f'{cache_method} должен писать кэш в {expected_path.name}, '
            f'но файл не создан'
        )
