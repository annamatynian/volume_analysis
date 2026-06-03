# tests/test_m15_funding_rate_ma.py
# TDD — М-15 | Funding Rate 30d MA
# Mozart-паттерн (пост 11.03.2026):
#   «30-дневная MA funding rate на многолетнем минимуме = предшествует дну цикла»
#
# Контракт функции:
#   classify_funding_rate_ma_regime(ma_value: float) -> str
#
#     ma_value: среднее значение funding rate (в долях, не в %) за 30 дней.
#               API возвращает значения ~0.0001 (базовая ставка Binance = +0.0001).
#               Многолетний минимум: <= floor (MOZART_CONFIG["funding_rate_ma_floor"]).
#
#     Возвращает:
#       'FLOOR_ZONE'  — ma_value <= floor: mozart-паттерн активен
#       'NEUTRAL'     — ma_value > floor:  обычный диапазон
#
# Все пороги из MOZART_CONFIG — не хардкодятся в тестах.

import pytest
from mozart_config import MOZART_CONFIG
from mozart_signals import classify_funding_rate_ma_regime

_FLOOR = MOZART_CONFIG["funding_rate_ma_floor"]   # -0.005 (в долях)


# ---------------------------------------------------------------------------
# Контрактные тесты
# ---------------------------------------------------------------------------

class TestClassifyFundingRateMaRegimeContract:
    """Классификатор возвращает строку из фиксированного набора."""

    def test_returns_string(self):
        # WHY: оркестратор и alignment используют строку напрямую;
        # не-строка даст тихий баг при форматировании.
        result = classify_funding_rate_ma_regime(0.0001)
        assert isinstance(result, str), (
            "classify_funding_rate_ma_regime должен возвращать str"
        )

    def test_known_labels_only(self):
        # WHY: фиксированное множество меток — контракт для signal_polarity().
        # Любая новая метка ломает alignment до обновления полярностей.
        valid = {'FLOOR_ZONE', 'NEUTRAL'}
        for val in [_FLOOR - 0.001, _FLOOR, _FLOOR + 0.001, 0.0, 0.0001, 0.001]:
            result = classify_funding_rate_ma_regime(val)
            assert result in valid, (
                f"Неожиданная метка '{result}' при ma_value={val}"
            )


# ---------------------------------------------------------------------------
# FLOOR_ZONE: ma_value <= floor
# ---------------------------------------------------------------------------

class TestFloorZone:
    """ma_value на или ниже порога многолетнего минимума → FLOOR_ZONE."""

    def test_exactly_at_floor(self):
        # WHY: граничное значение — самое частое место тихих багов.
        # ma_value == floor должен давать FLOOR_ZONE (включительно).
        result = classify_funding_rate_ma_regime(_FLOOR)
        assert result == 'FLOOR_ZONE', (
            f"ma_value == floor ({_FLOOR}) должен давать FLOOR_ZONE, "
            f"got '{result}'"
        )

    def test_below_floor(self):
        # WHY: значение ниже порога — паттерн Mozart активен (исторический минимум).
        val = _FLOOR - abs(_FLOOR) * 0.5  # на 50% ниже порога
        result = classify_funding_rate_ma_regime(val)
        assert result == 'FLOOR_ZONE', (
            f"ma_value={val} (ниже floor={_FLOOR}) должен давать FLOOR_ZONE"
        )

    def test_strongly_negative_ma(self):
        # WHY: экстремально отрицательный фандинг (глубокий медвежий рынок)
        # должен оставаться в FLOOR_ZONE, а не создавать отдельную метку.
        result = classify_funding_rate_ma_regime(-0.05)
        assert result == 'FLOOR_ZONE', (
            "Экстремально отрицательная MA (-0.05) должна давать FLOOR_ZONE"
        )


# ---------------------------------------------------------------------------
# NEUTRAL: ma_value > floor
# ---------------------------------------------------------------------------

class TestNeutral:
    """ma_value выше порога → NEUTRAL."""

    def test_just_above_floor(self):
        # WHY: граничное значение на единицу выше — контракт границы (строгий >).
        val = _FLOOR + 1e-6
        result = classify_funding_rate_ma_regime(val)
        assert result == 'NEUTRAL', (
            f"ma_value={val} (чуть выше floor={_FLOOR}) должен давать NEUTRAL"
        )

    def test_baseline_binance_rate(self):
        # WHY: базовая ставка Binance (+0.0001) — нормальное состояние рынка.
        # При нейтральном фандинге MA должна быть NEUTRAL.
        result = classify_funding_rate_ma_regime(0.0001)
        assert result == 'NEUTRAL', (
            "Базовая ставка Binance (0.0001) должна давать NEUTRAL"
        )

    def test_moderate_positive_ma(self):
        # WHY: умеренно положительный фандинг — не исторический минимум.
        result = classify_funding_rate_ma_regime(0.001)
        assert result == 'NEUTRAL', (
            "Умеренно положительная MA (0.001) должна давать NEUTRAL"
        )

    def test_zero_ma(self):
        # WHY: нулевая 30d MA = баланс лонгов/шортов, не экстремум.
        result = classify_funding_rate_ma_regime(0.0)
        assert result == 'NEUTRAL', (
            "Нулевая MA должна давать NEUTRAL"
        )

    def test_slightly_negative_above_floor(self):
        # WHY: умеренно отрицательная MA, но выше floor — не паттерн Mozart.
        val = _FLOOR + abs(_FLOOR) * 0.5  # на 50% выше floor (ближе к нулю)
        result = classify_funding_rate_ma_regime(val)
        assert result == 'NEUTRAL', (
            f"ma_value={val} (выше floor={_FLOOR}) должен давать NEUTRAL"
        )
