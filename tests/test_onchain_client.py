"""
tests/test_onchain_client.py
============================
Unit tests for BGeometricsClient._fetch_timeseries() и 5 holder-методов.

Принципы:
- Нет реальных сетевых вызовов — session.get() мокируется.
- Тесты проверяют КОНТРАКТ функции (колонки, типы, slug), не детали HTTP.
- _fetch_timeseries() — единственное место нетривиальной логики (defensive
  rename, type cast, empty-response guard) — тестируется изолированно.
- 5 методов-обёрток тестируются на правильность slug → убеждаемся, что
  каждый метод дёргает верный эндпоинт, а не соседний.

WHY mock session, а не AsyncMock: BGeometricsClient использует синхронный
requests.Session внутри async-методов — мокируем session.get, не корутину.
"""

import pytest
import asyncio
import json
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from onchain_client import BGeometricsClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(rows: list) -> MagicMock:
    """
    Возвращает мок requests.Response с заданным JSON-телом.
    raise_for_status() — no-op (200 OK).
    """
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = rows
    return mock_resp


def _client_with_mock_session(rows: list):
    """BGeometricsClient с мок-сессией, возвращающей rows."""
    client = BGeometricsClient()
    client.session = MagicMock()
    client.session.get.return_value = _make_response(rows)
    return client


def run(coro):
    """Запустить корутину синхронно. asyncio.run() создаёт новый loop
    на каждый вызов — совместимо с Python 3.10+ где get_event_loop()
    в MainThread без loop вызывает DeprecationWarning/RuntimeError."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _fetch_timeseries — контракт defensive rename и типы
# ---------------------------------------------------------------------------

class TestFetchTimeseries:

    def test_standard_d_column_renamed_to_date(self):
        """
        WHY: BGeometrics возвращает 'd' как дату — контракт rename d→date.
        Если rename сломан, downstream код упадёт с KeyError('date').
        """
        rows = [{'d': '2024-01-01', 'lthMvrv': '2.14'}]
        client = _client_with_mock_session(rows)
        df = run(client._fetch_timeseries('lth-mvrv', 'lth_mvrv'))
        assert 'date' in df.columns
        assert 'd' not in df.columns

    def test_value_column_renamed_to_given_name(self):
        """
        WHY: имя колонки значения стандартизируется в value_col —
        оркестратор обращается по имени, не по позиции.
        """
        rows = [{'d': '2024-01-01', 'lthMvrv': '2.14'}]
        client = _client_with_mock_session(rows)
        df = run(client._fetch_timeseries('lth-mvrv', 'lth_mvrv'))
        assert 'lth_mvrv' in df.columns

    def test_value_column_is_float(self):
        """
        WHY: API возвращает значения как строки — cast к float обязателен
        для арифметики в оркестраторе (сравнения >1.0, форматирование :.2f).
        """
        rows = [{'d': '2024-01-01', 'lthMvrv': '2.14'}]
        client = _client_with_mock_session(rows)
        df = run(client._fetch_timeseries('lth-mvrv', 'lth_mvrv'))
        assert df['lth_mvrv'].dtype == float

    def test_date_column_is_datetime(self):
        """
        WHY: date должна быть datetime64 — оркестратор вызывает .strftime()
        и pd.to_datetime сравнения. Строка вызовет AttributeError.
        """
        rows = [{'d': '2024-01-01', 'lthMvrv': '2.14'}]
        client = _client_with_mock_session(rows)
        df = run(client._fetch_timeseries('lth-mvrv', 'lth_mvrv'))
        assert pd.api.types.is_datetime64_any_dtype(df['date'])

    def test_result_has_exactly_two_columns(self):
        """
        WHY: результат должен содержать ровно [date, value_col] —
        лишние колонки от API не должны утекать в оркестратор.
        """
        rows = [{'d': '2024-01-01', 'lthMvrv': '2.14', 'extra': 'junk'}]
        client = _client_with_mock_session(rows)
        df = run(client._fetch_timeseries('lth-mvrv', 'lth_mvrv'))
        assert list(df.columns) == ['date', 'lth_mvrv']

    def test_empty_api_response_returns_empty_dataframe(self):
        """
        WHY: API может вернуть [] (нет данных за диапазон) — функция не
        должна падать с IndexError или KeyError. Оркестратор проверяет .empty.
        """
        client = _client_with_mock_session([])
        df = run(client._fetch_timeseries('lth-mvrv', 'lth_mvrv'))
        assert df.empty
        assert 'date' in df.columns
        assert 'lth_mvrv' in df.columns

    def test_fallback_when_date_column_not_named_d(self):
        """
        WHY: некоторые эндпоинты BGeometrics могут использовать 'date'
        вместо 'd' — defensive fallback берёт первую колонку как дату.
        Контракт: метод не падает, 'date' присутствует в результате.
        """
        rows = [{'date': '2024-01-01', 'value': '1.05'}]
        client = _client_with_mock_session(rows)
        df = run(client._fetch_timeseries('lth-sopr', 'lth_sopr'))
        assert 'date' in df.columns
        assert not df.empty

    def test_correct_slug_passed_to_session(self):
        """
        WHY: slug определяет какой эндпоинт дёргается — ошибка slug = 404.
        Проверяем что BASE_URL + slug попадает в session.get().
        """
        rows = [{'d': '2024-01-01', 'v': '2.0'}]
        client = _client_with_mock_session(rows)
        run(client._fetch_timeseries('lth-mvrv', 'lth_mvrv'))
        call_url = client.session.get.call_args[0][0]
        assert 'lth-mvrv' in call_url

    def test_unix_ts_field_not_returned_as_value(self):
        """
        WHY: BGeometrics API возвращает 3 поля: d, unixTs, значение.
        _fetch_timeseries должен пропустить unixTs и взять реальное
        значение (lthMvrv=1.586), а не Unix-таймстамп (1652140800).
        """
        rows = [{'d': '2022-05-10', 'unixTs': 1652140800, 'lthMvrv': 1.586}]
        client = _client_with_mock_session(rows)
        df = run(client._fetch_timeseries('lth-mvrv', 'lth_mvrv'))
        assert df['lth_mvrv'].iloc[0] == pytest.approx(1.586), (
            f"Expected 1.586, got {df['lth_mvrv'].iloc[0]} — "
            f"вероятно unixTs был принят за значение"
        )


# ---------------------------------------------------------------------------
# 5 методов-обёрток — правильность slug
# ---------------------------------------------------------------------------

class TestHolderMethodSlugs:
    """
    WHY slug-тесты: каждый метод — тонкая обёртка над _fetch_timeseries.
    Единственный риск — перепутать slug (lth-mvrv vs sth-mvrv и т.д.).
    Тест фиксирует контракт: метод X вызывает эндпоинт Y.
    """

    def _get_slug(self, coro_factory):
        """Запускает метод и возвращает URL из session.get."""
        rows = [{'d': '2024-01-01', 'v': '1.5'}]
        client = _client_with_mock_session(rows)
        run(coro_factory(client))
        return client.session.get.call_args[0][0]

    def test_get_lth_mvrv_uses_correct_slug(self):
        url = self._get_slug(lambda c: c.get_lth_mvrv())
        assert 'lth-mvrv' in url

    def test_get_sth_mvrv_uses_correct_slug(self):
        url = self._get_slug(lambda c: c.get_sth_mvrv())
        assert 'sth-mvrv' in url

    def test_get_lth_sopr_uses_correct_slug(self):
        url = self._get_slug(lambda c: c.get_lth_sopr())
        assert 'lth-sopr' in url

    def test_get_lth_net_position_change_30d_uses_correct_slug(self):
        url = self._get_slug(lambda c: c.get_lth_net_position_change_30d())
        assert 'lth-net-position-change-30d-btc' in url

    def test_get_sth_net_position_change_30d_uses_correct_slug(self):
        url = self._get_slug(lambda c: c.get_sth_net_position_change_30d())
        assert 'sth-net-position-change-30d-btc' in url


# ===========================================================================
# P1–P5: Expansion — slug contracts (PLAN_ONCHAIN_EXPANSION.md 2026-05-10)
# ===========================================================================

class TestExpansionMethodSlugs:
    """
    WHY: 6 новых методов (P1–P5) — обёртки над _fetch_timeseries().
    Риск: перепутать slug → дёргается другой эндпоинт → неверные данные
    (нет исключения, просто неправильные цифры в оркестраторе).
    Каждый тест фиксирует контракт: метод X → URL содержит slug Y.
    """

    def _get_slug(self, coro_factory):
        """Запускает метод и возвращает URL из session.get."""
        rows = [{'d': '2024-01-01', 'v': '1.5'}]
        client = _client_with_mock_session(rows)
        run(coro_factory(client))
        return client.session.get.call_args[0][0]

    def test_get_sth_sopr_uses_correct_slug(self):
        """P1: STH SOPR — slug 'sth-sopr'."""
        url = self._get_slug(lambda c: c.get_sth_sopr())
        assert 'sth-sopr' in url

    def test_get_exchange_netflow_uses_correct_slug(self):
        """P2: Exchange Net Flow — slug 'exchange-netflow-btc'."""
        url = self._get_slug(lambda c: c.get_exchange_netflow())
        assert 'exchange-netflow-btc' in url

    def test_get_nupl_lth_uses_correct_slug(self):
        """P3: LTH NUPL — slug 'nupl-lth'."""
        url = self._get_slug(lambda c: c.get_nupl_lth())
        assert 'nupl-lth' in url

    def test_get_nupl_sth_uses_correct_slug(self):
        """P3: STH NUPL — slug 'nupl-sth'."""
        url = self._get_slug(lambda c: c.get_nupl_sth())
        assert 'nupl-sth' in url

    def test_get_etf_flow_uses_correct_slug(self):
        """P4: ETF Flow — slug 'etf-flow-btc'."""
        url = self._get_slug(lambda c: c.get_etf_flow())
        assert 'etf-flow-btc' in url

    def test_get_hodl_waves_uses_correct_slug(self):
        """P5: HODL Waves — slug 'hodl-waves-supply'."""
        url = self._get_slug(lambda c: c.get_hodl_waves())
        assert 'hodl-waves-supply' in url


class TestExpansionMethodColumns:
    """
    WHY column-контракты: оркестратор обращается к DataFrame по именам колонок.
    Если _fetch_timeseries() вернул camelCase или неверное имя — KeyError
    в print-блоке без каких-либо подсказок о причине.

    P1–P4: полный контракт ['date', value_col] — ровно два столбца.
    P5 (hodl-waves): JSON-структура неизвестна до диагностики API.
    Тест фиксирует минимальный контракт: 'date' присутствует + ≥2 колонки.
    Полный column-контракт дописывается после диагностики эндпоинта.

    WHY mock rows: _fetch_timeseries() выбирает «первую не-date, не-unixTs
    колонку» и переименовывает в value_col — имя в mock-rows не влияет
    на результат. Mock-значения выбраны нейтральными (не фиксируют
    ни одного конкретного числа из API).
    """

    def test_get_sth_sopr_returns_correct_columns(self):
        """
        WHY: оркестратор обращается к df['sth_sopr'] для форматирования.
        Если колонка называется иначе — KeyError в print-блоке.
        Пара lth_sopr + sth_sopr — стандартная аналитическая связка,
        оба имени должны быть предсказуемы.
        """
        rows = [{'d': '2024-01-01', 'sthSopr': '1.05'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_sth_sopr())
        assert list(df.columns) == ['date', 'sth_sopr']

    def test_get_exchange_netflow_returns_correct_columns(self):
        """
        WHY: '_btc' суффикс фиксирует единицу измерения и отличает
        от гипотетического exchange_netflow_usd. Знак значения критичен
        (+ = приток на биржи, − = отток) — правильное имя предотвращает
        перепутку с инвертированными метриками.
        """
        rows = [{'d': '2024-01-01', 'exchangeNetflow': '-500.0'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_exchange_netflow())
        assert list(df.columns) == ['date', 'exchange_netflow_btc']

    def test_get_nupl_lth_returns_correct_columns(self):
        """
        WHY: nupl_lth и nupl_sth — разные метрики с разной интерпретацией
        (LTH NUPL > 0.75 = эйфория LTH; STH NUPL < 0 = капитуляция STH).
        Уникальные имена предотвращают silent data confusion при
        одновременном выводе обоих значений в оркестраторе.
        """
        rows = [{'d': '2024-01-01', 'nuplLth': '0.61'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_nupl_lth())
        assert list(df.columns) == ['date', 'nupl_lth']

    def test_get_nupl_sth_returns_correct_columns(self):
        """
        WHY: см. test_get_nupl_lth_returns_correct_columns.
        STH NUPL < 0 сигнализирует о капитуляции краткосрочников —
        критичная метрика должна иметь предсказуемое имя колонки.
        """
        rows = [{'d': '2024-01-01', 'nuplSth': '0.12'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_nupl_sth())
        assert list(df.columns) == ['date', 'nupl_sth']

    def test_get_etf_flow_returns_correct_columns(self):
        """
        WHY: '_btc' суффикс фиксирует единицу (не USD, не контракты).
        ETF Flow — приток/отток BTC из spot ETF с 2024 г.;
        '_btc' исключает путаницу с гипотетическим etf_flow_usd.
        """
        rows = [{'d': '2024-01-01', 'etfFlow': '1200.5'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_etf_flow())
        assert list(df.columns) == ['date', 'etf_flow_btc']

    def test_get_hodl_waves_contains_date_and_at_least_one_wave_column(self):
        """
        WHY ослабленный контракт: hodl-waves-supply возвращает несколько
        когортных колонок — их точные имена определяются при диагностике
        эндпоинта (PLAN_ONCHAIN_EXPANSION.md, P5).

        Сейчас фиксируем два инварианта:
          1. 'date' присутствует — оркестратор обращается к ней напрямую.
          2. Есть хотя бы одна дополнительная колонка — иначе DataFrame
             бесполезен для блока [HODL WAVES].

        Полный column-контракт (имена когортных колонок 1m-3m, 3m-6m, 6m-12m)
        дописывается отдельным тестом после диагностики JSON структуры.
        """
        rows = [{'d': '2024-01-01', 'unixTs': 1704067200,
                 'wave_1m_3m': '0.15', 'wave_3m_6m': '0.22'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_hodl_waves())
        assert 'date' in df.columns, (
            "get_hodl_waves() должен возвращать колонку 'date'"
        )
        assert len(df.columns) >= 2, (
            f"get_hodl_waves() должен возвращать date + ≥1 когортную колонку, "
            f"получено: {list(df.columns)}"
        )

    def test_get_hodl_waves_real_api_column_names(self):
        """
        WHY: диагностика 2026-05-11 показала реальную структуру API:
        ['d', 'unixTs', 'age_0d_1d', ..., 'age_1m_3m', 'age_3m_6m', 'age_6m_1y', ...]
        Оркестратор обращается к df['age_1m_3m'], df['age_3m_6m'], df['age_6m_1y'] напрямую.
        KeyError если имена несовпадают.
        
        Mock-строки используют реальную API-структуру.
        unixTs в API является строкой ('1704067200'), значения когорт — тоже строки.
        """
        rows = [{'d': '2026-05-10', 'unixTs': '1778371200',
                 'age_1m_3m': '1061452.92361457',
                 'age_3m_6m': '2803837.91143472',
                 'age_6m_1y': '2603563.77490367'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_hodl_waves())
        assert 'age_1m_3m' in df.columns, (
            "age_1m_3m (MTH 1–3 мес) отсутствует — оркестратор упадёт с KeyError"
        )
        assert 'age_3m_6m' in df.columns, (
            "age_3m_6m (MTH 3–6 мес) отсутствует — оркестратор упадёт с KeyError"
        )
        assert 'age_6m_1y' in df.columns, (
            "age_6m_1y (граница LTH) отсутствует — оркестратор упадёт с KeyError"
        )
        assert 'unixTs' not in df.columns, (
            "unixTs должен быть удалён — оркестратор не должен его видеть"
        )

    def test_get_hodl_waves_casts_cohort_columns_to_float(self):
        """
        WHY: API hodl-waves-supply возвращает значения когорт как строки
        ('1427793.16350000') — диагностика 2026-05-11 подтвердила.
        Оркестратор делает df['age_1m_3m'] / 1_000_000 — деление строки бросает
        TypeError. get_hodl_waves() должен кастовать все не-date колонки в float.
        """
        rows = [{'d': '2026-05-10', 'unixTs': '1778371200',
                 'age_1m_3m': '1061452.92361457',
                 'age_3m_6m': '2803837.91143472'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_hodl_waves())
        for col in [c for c in df.columns if c != 'date']:
            assert pd.api.types.is_float_dtype(df[col]), (
                f"'{col}' должна быть float64, получено: {df[col].dtype}. "
                f"API возвращает строки, get_hodl_waves() должен делать cast."
            )


# ===========================================================================
# TestRealizedLossLthUsd — slug + column contracts (диагностика 2026-05-12)
# ===========================================================================

class TestRealizedLossLthUsd:
    """
    WHY: get_realized_loss_lth_usd() исторически использовал slug
    'realized-loss-lth-usd' (дефисы). Диагностика CO-5 (2026-05-20)
    установила: варианты _usd и _btc возвращают 404 на free tier.
    Рабочий slug: 'realized_loss_lth' (без суффикса валюты).
    Реальная структура ответа: {'d': ..., 'unixTs': ..., 'realizedLossLth': ...}
    Выходная колонка lth_realized_loss_usd сохранена (оркестратор и тесты
    ссылаются на это имя). Значения USD-деноминированы без суффикса.
    """

    def _get_slug(self, coro_factory):
        rows = [{'d': '2022-05-12', 'unixTs': 1652313600,
                 'realizedLossLth': -146590000}]
        client = _client_with_mock_session(rows)
        run(coro_factory(client))
        return client.session.get.call_args[0][0]

    def test_uses_correct_slug(self):
        """
        Contract: URL должен содержать 'realized_loss_lth'.

        WHY: CO-5 диагностика (2026-05-20) подтвердила:
          - 'realized-loss-lth-usd' → 404 без date-параметров (нестабильно)
          - 'realized_loss_lth_usd' → 404 на free tier (paywall)
          - 'realized_loss_lth' → 200 OK стабильно
        Этот тест предотвращает регрессию к неверным slug-ам.
        """
        url = self._get_slug(lambda c: c.get_realized_loss_lth_usd())
        assert 'realized_loss_lth' in url, (
            f"Expected 'realized_loss_lth' in URL, got: {url}. "
            f"Slug с _usd суффиксом возвращает 404 на free tier."
        )

    def test_returns_correct_columns(self):
        """
        Contract: результат должен содержать колонки [date, lth_realized_loss_usd].

        WHY: оркестратор обращается к df['lth_realized_loss_usd'] для расчёта
        капитуляции и форматирования [ON-CHAIN: LTH REALIZED LOSS] блока.
        Реальное поле API 'realizedLossLthUsd' (camelCase) содержит 'loss' и 'lth'
        — col_map переименовывает его корректно.
        """
        rows = [{'d': '2022-05-12', 'unixTs': 1652313600,
                 'realizedLossLthUsd': -146590000}]
        client = _client_with_mock_session(rows)
        df = run(client.get_realized_loss_lth_usd())
        assert 'date' in df.columns, "'date' отсутствует"
        assert 'lth_realized_loss_usd' in df.columns, (
            f"'lth_realized_loss_usd' отсутствует. Колонки: {list(df.columns)}"
        )


# ===========================================================================
# CO-6: Блок 2 — slug/column контракты (обнаружено 2026-05-21, сессия L3-1)
# ===========================================================================

class TestBlok2MethodSlugsAndColumns:
    """
    WHY этот класс:
      6 методов добавлены в Блоке 2 без тестов slug/column в test_onchain_client.py.
      Все используют _fetch_timeseries или аналогичную схему — единственный риск
      это неверный slug (молчаливый 404 → пустой DataFrame → сигнал уходит в missing)
      или неверное имя колонки (KeyError в оркестраторе).
      CO-6 закрывает этот долг до интеграции в оркестратор (L3-3).
    """

    def _get_slug(self, coro_factory):
        """Запускает метод и возвращает URL из session.get."""
        rows = [{'d': '2024-01-01', 'v': '1.5', 'unixTs': 1704067200}]
        client = _client_with_mock_session(rows)
        run(coro_factory(client))
        return client.session.get.call_args[0][0]

    # --- МБ-01 | get_realized_price() ---

    def test_get_realized_price_slug(self):
        url = self._get_slug(lambda c: c.get_realized_price())
        assert 'realized-price' in url
        # WHY: МБ-01 Realized Price — «синяя линия» дна цикла. Неверный slug
        # вернёт данные другого эндпоинта (или 404) молча. Realized Price
        # используется в формуле OCA (МБ-08) и как самостоятельный сигнал.

    def test_get_realized_price_column(self):
        rows = [{'d': '2024-01-01', 'unixTs': 1704067200, 'realizedPrice': '53000.0'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_realized_price())
        assert list(df.columns) == ['date', 'realized_price']
        # WHY: оркестратор обращается к df['realized_price'] для вывода
        # и передаёт значение в calculate_one_cycle_average(). KeyError если
        # колонка называется иначе (например 'realizedPrice' camelCase).

    # --- МБ-02 | get_true_market_mean() ---

    def test_get_true_market_mean_slug(self):
        url = self._get_slug(lambda c: c.get_true_market_mean())
        assert 'true-market-mean' in url
        # WHY: МБ-02 True Market Mean — «зелёная линия», рубикон медвежьего рынка.
        # Mozart: пробой вниз = смена глобального тренда. Неверный slug
        # смешает эту метрику с соседними (realized-price, mvrv-zscore).

    def test_get_true_market_mean_column(self):
        rows = [{'d': '2024-01-01', 'unixTs': 1704067200, 'trueMarketMean': '45000.0'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_true_market_mean())
        assert list(df.columns) == ['date', 'true_market_mean']
        # WHY: оркестратор выводит df['true_market_mean'] в блоке [МБ-02].
        # Неверное имя → KeyError без подсказки о причине.

    # --- МБ-04 | get_supply_loss() ---

    def test_get_supply_loss_slug(self):
        url = self._get_slug(lambda c: c.get_supply_loss())
        assert 'supply-loss' in url
        # WHY: МБ-04 Supply in Loss — объём BTC в убытке. Mozart: 5M BTC
        # = триггер структурного тренда. Неверный slug даст неверные числа
        # молча, без исключения.

    def test_get_supply_loss_column(self):
        rows = [{'d': '2024-01-01', 'unixTs': 1704067200, 'supplyLoss': '4200000.0'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_supply_loss())
        assert list(df.columns) == ['date', 'supply_loss']
        # WHY: оркестратор сравнивает df['supply_loss'].iloc[-1] с порогом 5M.
        # Неверное имя → KeyError в блоке вывода.

    # --- МБ-06 | get_nupl() ---

    def test_get_nupl_slug(self):
        url = self._get_slug(lambda c: c.get_nupl())
        assert 'nupl' in url
        # WHY: МБ-06 NUPL — нереализованный P&L всего рынка. Slug 'nupl'
        # короткий и легко перепутать с 'nupl-lth' или 'nupl-sth'.
        # Тест защищает от случайной замены при рефакторинге.

    def test_get_nupl_column(self):
        rows = [{'d': '2024-01-01', 'unixTs': 1704067200, 'nupl': '0.42'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_nupl())
        assert list(df.columns) == ['date', 'nupl']
        # WHY: classify_nupl_regime() принимает float из df['nupl'].iloc[-1].
        # Неверное имя → KeyError до classify-вызова.

    # --- МБ-07 | get_mvrv_zscore() ---

    def test_get_mvrv_zscore_slug(self):
        url = self._get_slug(lambda c: c.get_mvrv_zscore())
        assert 'mvrv-zscore' in url
        # WHY: МБ-07 MVRV Z-Score — макро позиционирование. Slug 'mvrv-zscore'
        # vs 'lth-mvrv' vs 'sth-mvrv' — три похожих эндпоинта, легко спутать.

    def test_get_mvrv_zscore_column(self):
        rows = [{'d': '2024-01-01', 'unixTs': 1704067200, 'mvrvZscore': '0.77'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_mvrv_zscore())
        assert list(df.columns) == ['date', 'mvrv_zscore']
        # WHY: classify_mvrv_zscore_regime() принимает df['mvrv_zscore'].iloc[-1].
        # Неверное имя → KeyError или NaN при расчёте режима.

    # --- МБ-08 | get_realized_cap_hodl_waves() ---

    def test_get_realized_cap_hodl_waves_slug(self):
        rows = [{'d': '2024-01-01', 'unixTs': 1704067200,
                 'age_2y_3y': '0.056', 'age_3y_4y': '0.038'}]
        client = _client_with_mock_session(rows)
        run(client.get_realized_cap_hodl_waves())
        url = client.session.get.call_args[0][0]
        assert 'realized-cap-hodl-waves' in url
        # WHY: МБ-08 RC HODL Waves — RC-доли для формулы OCA. Slug отличается
        # от hodl-waves-supply (get_hodl_waves) — один символ разницы в имени.
        # Путаница → формула OCA получит BTC supply вместо RC-долей.

    def test_get_realized_cap_hodl_waves_has_date_and_age_columns(self):
        rows = [{'d': '2024-01-01', 'unixTs': 1704067200,
                 'age_2y_3y': '0.056', 'age_3y_4y': '0.038'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_realized_cap_hodl_waves())
        assert 'date' in df.columns
        assert 'age_2y_3y' in df.columns
        assert 'age_3y_4y' in df.columns
        # WHY: calculate_one_cycle_average() принимает rc_2y3y и rc_3y4y —
        # оркестратор берёт их из df['age_2y_3y'] и df['age_3y_4y'].
        # Отсутствие колонки → KeyError внутри формулы OCA.

    def test_get_realized_cap_hodl_waves_removes_unix_ts(self):
        rows = [{'d': '2024-01-01', 'unixTs': 1704067200,
                 'age_2y_3y': '0.056', 'age_3y_4y': '0.038'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_realized_cap_hodl_waves())
        assert 'unixTs' not in df.columns
        # WHY: unixTs не является метрикой — оркестратор не должен его видеть.
        # Если утечёт в DataFrame, итерация по колонкам даст лишний float-cast.

    def test_get_realized_cap_hodl_waves_casts_to_float(self):
        rows = [{'d': '2024-01-01', 'unixTs': 1704067200,
                 'age_2y_3y': '0.056', 'age_3y_4y': '0.038'}]
        client = _client_with_mock_session(rows)
        df = run(client.get_realized_cap_hodl_waves())
        for col in [c for c in df.columns if c != 'date']:
            assert pd.api.types.is_float_dtype(df[col]), (
                f"'{col}' должна быть float64. API возвращает строки ('0.056') —"
                f" без каста rc_2y3y / (rc_2y3y + rc_3y4y) бросит TypeError."
            )
        # WHY: RC-доли в API приходят как строки. Формула OCA делит их —
        # деление строк невозможно. Cast обязателен в get_realized_cap_hodl_waves().
