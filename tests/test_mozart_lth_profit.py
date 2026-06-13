"""
tests/test_mozart_lth_profit.py
================================
TDD-тесты для classify_lth_profit_regime() — МБ-05.

Источник паттерна: пост 14.01.2026 (Mr Mozart).
«если давление продаж вновь сильно возрастёт
(выше 1 млрд $ / день в среднем за 7 дней),
то риски будут смещены в сторону медвежки»

Пороги из MOZART_CONFIG:
  lth_profit_7d_ma_warning:  1_000_000_000 USD/день  (Mozart, пост 14.01.2026)
  lth_profit_7d_ma_moderate:   500_000_000 USD/день  (FORMALIZED: половина порога)
"""

import pytest
from mozart_config import MOZART_CONFIG


# ---------------------------------------------------------------------------
# Вспомогательная функция — смещение от порога без хардкода
# ---------------------------------------------------------------------------

def _above(key: str, delta: float = 1.0) -> float:
    """Значение чуть выше порога из конфига."""
    return MOZART_CONFIG[key] + delta


def _below(key: str, delta: float = 1.0) -> float:
    """Значение чуть ниже порога из конфига."""
    return MOZART_CONFIG[key] - delta


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------

class TestClassifyLthProfitRegime:
    """
    WHY этот класс: classify_lth_profit_regime — сигнальная функция МБ-05.
    Без неё оркестратор не может добавить МБ-05 в Signal Alignment.
    """

    # ------------------------------------------------------------------
    # HIGH_PRESSURE: MA строго выше порога warning ($1B)
    # ------------------------------------------------------------------

    def test_high_pressure_above_warning(self):
        """
        WHY: Mozart явно определяет > $1B/день как медвежий риск (пост 14.01.2026).
        Если функция вернёт не HIGH_PRESSURE — alignment не учтёт сигнал.
        """
        from mozart_signals import classify_lth_profit_regime
        result = classify_lth_profit_regime(_above('lth_profit_7d_ma_warning'))
        # WHY: значение строго выше warning → HIGH_PRESSURE по Mozart
        assert result == 'HIGH_PRESSURE', (
            f'MA > warning должно давать HIGH_PRESSURE, получили {result!r}'
        )

    def test_high_pressure_far_above_warning(self):
        """
        WHY: исторический пик ноябрь 2025 ~$2B/день должен классифицироваться
        как HIGH_PRESSURE, а не MODERATE — цифра из поста 14.01.2026.
        """
        from mozart_signals import classify_lth_profit_regime
        result = classify_lth_profit_regime(2_000_000_000)
        # WHY: $2B >> $1B warning → HIGH_PRESSURE (нет верхней границы у Mozart)
        assert result == 'HIGH_PRESSURE', (
            f'$2B/день должно давать HIGH_PRESSURE, получили {result!r}'
        )

    # ------------------------------------------------------------------
    # Граница warning ($1B) — отдельный тест
    # ------------------------------------------------------------------

    def test_boundary_exactly_warning(self):
        """
        WHY граница: MA == $1B — рубикон Mozart. По контракту включительно →
        HIGH_PRESSURE. Ошибка на границе = самый частый silent bug.
        """
        from mozart_signals import classify_lth_profit_regime
        threshold = MOZART_CONFIG['lth_profit_7d_ma_warning']
        result = classify_lth_profit_regime(float(threshold))
        # WHY: равно порогу Mozart → HIGH_PRESSURE (включительная граница)
        assert result == 'HIGH_PRESSURE', (
            f'MA == warning ($1B) должно давать HIGH_PRESSURE, получили {result!r}'
        )

    # ------------------------------------------------------------------
    # MODERATE: warning > MA > moderate ($500M–$1B)
    # ------------------------------------------------------------------

    def test_moderate_between_thresholds(self):
        """
        WHY: диапазон между moderate и warning — умеренное давление.
        Если функция вернёт HIGH_PRESSURE или LOW — alignment будет неверным.
        """
        from mozart_signals import classify_lth_profit_regime
        mid = (
            MOZART_CONFIG['lth_profit_7d_ma_warning'] +
            MOZART_CONFIG['lth_profit_7d_ma_moderate']
        ) / 2
        result = classify_lth_profit_regime(mid)
        # WHY: значение между moderate и warning → MODERATE
        assert result == 'MODERATE', (
            f'MA между $500M и $1B должно давать MODERATE, получили {result!r}'
        )

    def test_moderate_just_above_moderate(self):
        """
        WHY: строго выше moderate ($500M) и ниже warning ($1B) → MODERATE.
        Граничный случай снизу для MODERATE-зоны.
        """
        from mozart_signals import classify_lth_profit_regime
        result = classify_lth_profit_regime(_above('lth_profit_7d_ma_moderate'))
        # WHY: чуть выше moderate и ниже warning → MODERATE
        assert result == 'MODERATE', (
            f'MA чуть выше $500M должно давать MODERATE, получили {result!r}'
        )

    # ------------------------------------------------------------------
    # Граница moderate ($500M) — отдельный тест
    # ------------------------------------------------------------------

    def test_boundary_exactly_moderate(self):
        """
        WHY граница: MA == $500M — граница LOW/MODERATE. По контракту:
        <= moderate → LOW (moderate включается в LOW, не в MODERATE).
        WHY LOW а не MODERATE на границе: умеренное давление начинается
        строго выше $500M; само значение $500M ещё не «умеренное» по Mozart.
        """
        from mozart_signals import classify_lth_profit_regime
        threshold = MOZART_CONFIG['lth_profit_7d_ma_moderate']
        result = classify_lth_profit_regime(float(threshold))
        # WHY: равно moderate → LOW (включительная граница LOW)
        assert result == 'LOW', (
            f'MA == $500M должно давать LOW (включительно), получили {result!r}'
        )

    # ------------------------------------------------------------------
    # LOW: MA <= moderate ($500M)
    # ------------------------------------------------------------------

    def test_low_below_moderate(self):
        """
        WHY: MA ниже $500M — давление LTH слабое, сигнал не активен.
        Если вернётся MODERATE — alignment ошибочно учтёт умеренное давление.
        """
        from mozart_signals import classify_lth_profit_regime
        result = classify_lth_profit_regime(_below('lth_profit_7d_ma_moderate'))
        # WHY: чуть ниже moderate → LOW
        assert result == 'LOW', (
            f'MA ниже $500M должно давать LOW, получили {result!r}'
        )

    def test_low_zero(self):
        """
        WHY: нулевая прибыль LTH (нет реализованного профита) → LOW.
        Нулевое значение — корректный вход, не должен вызывать исключение.
        """
        from mozart_signals import classify_lth_profit_regime
        result = classify_lth_profit_regime(0.0)
        # WHY: 0 << moderate → LOW (LTH не фиксируют прибыль)
        assert result == 'LOW', (
            f'MA == 0 должно давать LOW, получили {result!r}'
        )

    def test_low_small_value(self):
        """
        WHY: небольшое значение $100M/день — типичный «тихий» период,
        когда LTH не распределяют. Должен быть LOW.
        """
        from mozart_signals import classify_lth_profit_regime
        result = classify_lth_profit_regime(100_000_000)
        # WHY: $100M << $500M moderate → LOW (далеко от порога Mozart)
        assert result == 'LOW', (
            f'$100M должно давать LOW, получили {result!r}'
        )

    # ------------------------------------------------------------------
    # Тип возврата
    # ------------------------------------------------------------------

    def test_returns_string(self):
        """
        WHY: оркестратор передаёт результат в build_alignment() как строку.
        Если вернётся не str — TypeError при построении alignment.
        """
        from mozart_signals import classify_lth_profit_regime
        result = classify_lth_profit_regime(_above('lth_profit_7d_ma_warning'))
        # WHY: build_alignment ожидает str, не int или None
        assert isinstance(result, str), (
            f'classify_lth_profit_regime должна возвращать str, получили {type(result)}'
        )
