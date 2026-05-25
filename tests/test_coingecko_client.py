# tests/test_coingecko_client.py
# НВ-03 | CoinGecko client — BTC Dominance текущее + parquet-кэш
#
# Публичный API клиента:
#   get_btc_dominance_current() -> float
#   get_btc_dominance_with_history(lookback_days, cache_path) -> tuple[float, float | None]
#
# Диагностика 2026-05-25:
#   /api/v3/global             → 200 OK, поле data.btc_dominance (float, %)
#   /api/v3/global/market_cap_chart → 401 PRO-only (недоступен)
#   /api/v3/coins/bitcoin/market_chart → 200 OK (не используется для BTC.D)
#
# Архитектура: parquet-кэш data/coingecko_btc_dominance.parquet
#   Каждый запуск: fetch current → append в кэш → lookup 30d назад.
#   < lookback_days дней истории → возвращает (current, None).

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from coingecko_client import get_btc_dominance_current, get_btc_dominance_with_history

# ---------------------------------------------------------------------------
# Вспомогательные фикстуры
# ---------------------------------------------------------------------------

# Нейтральный плейсхолдер для btc_dominance — намеренно не API-реалистичен
_MOCK_BTC_D = 42.0

def _make_mock_response(btc_d: float = _MOCK_BTC_D) -> MagicMock:
    """Мок requests.get, возвращающий корректный /global ответ."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {
            "btc_dominance": btc_d,
            "total_market_cap": {"usd": 9_999_999_999_999},  # синтетика
        }
    }
    mock_resp.raise_for_status.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# get_btc_dominance_current() — HTTP-слой
# ---------------------------------------------------------------------------

class TestGetBtcDominanceCurrent:

    def test_calls_global_endpoint(self):
        with patch("coingecko_client.requests.get") as mock_get:
            mock_get.return_value = _make_mock_response()
            get_btc_dominance_current()
            called_url = mock_get.call_args[0][0]
            assert "api.coingecko.com/api/v3/global" in called_url
            # WHY: неверный URL → 404 или чужой endpoint; клиент никогда
            #   не получит btc_dominance. Контракт защищает от опечатки в URL.

    def test_returns_float(self):
        with patch("coingecko_client.requests.get") as mock_get:
            mock_get.return_value = _make_mock_response()
            result = get_btc_dominance_current()
            assert isinstance(result, float)
            # WHY: classify_btc_dominance_trend ожидает float;
            #   str/None → TypeError при вычислении delta.

    def test_parses_data_btc_dominance_field(self):
        with patch("coingecko_client.requests.get") as mock_get:
            mock_get.return_value = _make_mock_response(_MOCK_BTC_D)
            result = get_btc_dominance_current()
            assert result == float(_MOCK_BTC_D)
            # WHY: поле находится в data.btc_dominance, не на верхнем уровне.
            #   Парсинг верхнего уровня → KeyError или None вместо числа.

    def test_raises_on_http_error(self):
        with patch("coingecko_client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = Exception("HTTP 429")
            mock_get.return_value = mock_resp
            with pytest.raises(Exception):
                get_btc_dominance_current()
            # WHY: оркестратор должен поймать ошибку явно (try/except вовне),
            #   а не получить None и молча записать None в кэш.


# ---------------------------------------------------------------------------
# get_btc_dominance_with_history() — кэш-логика
# ---------------------------------------------------------------------------

class TestGetBtcDominanceWithHistory:

    def test_returns_tuple(self, tmp_path):
        """Всегда возвращает tuple из двух элементов."""
        with patch("coingecko_client.requests.get") as mock_get:
            mock_get.return_value = _make_mock_response()
            result = get_btc_dominance_with_history(
                lookback_days=30, cache_path=tmp_path / "btc_d.parquet"
            )
            assert isinstance(result, tuple) and len(result) == 2
            # WHY: оркестратор делает `current, prev = get_btc_dominance_with_history(...)`;
            #   не-tuple → ValueError при распаковке.

    def test_first_element_is_current_float(self, tmp_path):
        """Первый элемент — текущий BTC.D (float)."""
        with patch("coingecko_client.requests.get") as mock_get:
            mock_get.return_value = _make_mock_response(_MOCK_BTC_D)
            current, _ = get_btc_dominance_with_history(
                lookback_days=30, cache_path=tmp_path / "btc_d.parquet"
            )
            assert isinstance(current, float)
            assert current == float(_MOCK_BTC_D)
            # WHY: текущее значение всегда свежее с API; устаревшее значение
            #   из кэша вместо текущего → неверный delta в classify функции.

    def test_second_element_is_none_when_no_cache(self, tmp_path):
        """Нет кэш-файла → второй элемент None (недостаточно истории)."""
        with patch("coingecko_client.requests.get") as mock_get:
            mock_get.return_value = _make_mock_response()
            _, prev = get_btc_dominance_with_history(
                lookback_days=30, cache_path=tmp_path / "btc_d.parquet"
            )
            assert prev is None
            # WHY: при первом запуске кэша нет; classify должен получить NEUTRAL
            #   (orchestrator передаёт None → особый путь), а не KeyError или 0.

    def test_second_element_is_none_when_insufficient_history(self, tmp_path):
        """Кэш есть, но меньше lookback_days записей → None."""
        cache_file = tmp_path / "btc_d.parquet"
        # Создаём кэш с 5 записями (меньше 30)
        df = pd.DataFrame({
            "date": [f"2026-04-{i:02d}" for i in range(1, 6)],
            "btc_dominance_pct": [40.0 + i for i in range(5)],
        })
        df.to_parquet(cache_file, index=False)

        with patch("coingecko_client.requests.get") as mock_get:
            mock_get.return_value = _make_mock_response()
            _, prev = get_btc_dominance_with_history(
                lookback_days=30, cache_path=cache_file
            )
            assert prev is None
            # WHY: 5 дней истории при lookback=30 — значение 30d назад неизвестно;
            #   возврат первой строки кэша → неверный delta и ложный сигнал.

    def test_second_element_is_float_when_sufficient_history(self, tmp_path):
        """Кэш с >= lookback_days записями → второй элемент float."""
        cache_file = tmp_path / "btc_d.parquet"
        # 31 запись = 31 день истории (достаточно для lookback=30)
        df = pd.DataFrame({
            "date": [f"2026-04-{i:02d}" for i in range(1, 32)],
            "btc_dominance_pct": [40.0 + i * 0.1 for i in range(31)],
        })
        df.to_parquet(cache_file, index=False)

        with patch("coingecko_client.requests.get") as mock_get:
            mock_get.return_value = _make_mock_response()
            _, prev = get_btc_dominance_with_history(
                lookback_days=30, cache_path=cache_file
            )
            assert isinstance(prev, float)
            # WHY: с достаточной историей должен вернуть реальное прошлое значение;
            #   None вместо float → classify всегда видит NEUTRAL, сигнал мёртв.

    def test_cache_updated_after_call(self, tmp_path):
        """После вызова кэш-файл существует и содержит сегодняшнюю запись."""
        cache_file = tmp_path / "btc_d.parquet"
        assert not cache_file.exists()

        with patch("coingecko_client.requests.get") as mock_get:
            mock_get.return_value = _make_mock_response(_MOCK_BTC_D)
            get_btc_dominance_with_history(
                lookback_days=30, cache_path=cache_file
            )

        assert cache_file.exists()
        df = pd.read_parquet(cache_file)
        assert len(df) == 1
        assert float(df["btc_dominance_pct"].iloc[0]) == float(_MOCK_BTC_D)
        # WHY: кэш не обновляется → 30d-ago никогда не накапливается;
        #   сигнал НВ-03 остаётся NEUTRAL бесконечно.

    def test_cache_is_idempotent_for_same_day(self, tmp_path):
        """Два вызова в один день → только 1 запись в кэше (нет дублирования)."""
        cache_file = tmp_path / "btc_d.parquet"

        with patch("coingecko_client.requests.get") as mock_get:
            mock_get.return_value = _make_mock_response()
            get_btc_dominance_with_history(lookback_days=30, cache_path=cache_file)
            get_btc_dominance_with_history(lookback_days=30, cache_path=cache_file)

        df = pd.read_parquet(cache_file)
        assert len(df) == 1
        # WHY: дублирование → кэш растёт экспоненциально при частых запусках;
        #   lookup 30d назад смещается и возвращает неверную дату.


# ---------------------------------------------------------------------------
# Реальная структура CoinGecko free tier
# ---------------------------------------------------------------------------

class TestGetBtcDominanceCurrentRealStructure:
    """Диагностика 2026-05-25: btc_dominance находится в market_cap_percentage.btc."""

    def test_parses_market_cap_percentage_btc_as_fallback(self):
        """Фоллбэк: если btc_dominance нет → читаем market_cap_percentage.btc."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "data": {
                "market_cap_percentage": {"btc": _MOCK_BTC_D, "eth": 8.0},
                "total_market_cap": {"usd": 9_999_999_999_999},
            }
        }
        with patch("coingecko_client.requests.get") as mock_get:
            mock_get.return_value = mock_resp
            result = get_btc_dominance_current()
            assert result == float(_MOCK_BTC_D)
            # WHY: CoinGecko free tier возвращает доминанс через
            #   data.market_cap_percentage.btc, а не data.btc_dominance.
            #   Неверное поле → KeyError при каждом запуске оркестратора.

    def test_btc_dominance_field_takes_priority_if_present(self):
        """Если btc_dominance есть в data — используем его (Pro API / будущие версии)."""
        with patch("coingecko_client.requests.get") as mock_get:
            mock_get.return_value = _make_mock_response(_MOCK_BTC_D)
            result = get_btc_dominance_current()
            assert result == float(_MOCK_BTC_D)
            # WHY: обратная совместимость с Pro API и будущими версиями,
            #   где btc_dominance может появиться напрямую.
