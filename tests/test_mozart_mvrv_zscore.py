# tests/test_mozart_mvrv_zscore.py
# МБ-07 | MVRV Z-Score — макро позиционирование относительно истории
#
# Endpoint: /v1/mvrv-zscore → поле mvrvZscore (float, десятичная дробь)
# Диагностика 20.05.2026: 200 OK, 1460 записей, диапазон [-0.36, 3.35].
# Текущее значение (19.05.2026): 0.7759 → NEUTRAL.
#
# Зоны (Mozart, пост 25.02.2026 + стандартный on-chain):
#   PEAK    : z >  peak_threshold (7.0)  — исторический топ цикла
#   BULL    : z >= bull_threshold (3.0)  — сильный бычий рынок
#   NEUTRAL : z >= 0.0                   — выше/на уровне Realized Price
#   BEAR    : z >  bottom_threshold      — чуть ниже Realized Price
#   BOTTOM  : z <= bottom_threshold      — Mozart: «зона исторического дна»

import pytest
from mozart_config import MOZART_CONFIG
from mozart_signals import classify_mvrv_zscore_regime


# ---------------------------------------------------------------------------
# Контракт конфига — ключи, типы, порядок
# ---------------------------------------------------------------------------

class TestMvrvZscoreConfig:
    def test_peak_threshold_exists(self):
        assert "mvrv_zscore_peak" in MOZART_CONFIG
        # WHY: classify_mvrv_zscore_regime читает этот ключ;
        #   отсутствие → KeyError при каждом запуске оркестратора.

    def test_bull_threshold_exists(self):
        assert "mvrv_zscore_bull" in MOZART_CONFIG
        # WHY: граница BULL/NEUTRAL; отсутствие → KeyError в production.

    def test_bottom_threshold_exists(self):
        assert "mvrv_zscore_bottom" in MOZART_CONFIG
        # WHY: граница BEAR/BOTTOM; Mozart: «Z<0 = зона дна» (пост 25.02.2026).
        #   Отсутствие → BOTTOM никогда не достигается.

    def test_thresholds_are_numeric(self):
        for key in ("mvrv_zscore_peak", "mvrv_zscore_bull", "mvrv_zscore_bottom"):
            assert isinstance(float(MOZART_CONFIG[key]), float)
            # WHY: classify использует float-сравнение; нечисловое значение →
            #   TypeError или неверный результат сравнения.

    def test_bottom_threshold_is_negative(self):
        assert float(MOZART_CONFIG["mvrv_zscore_bottom"]) < 0
        # WHY: BOTTOM по Mozart = рынок ниже Realized Price = отрицательный Z.
        #   Положительное значение → BOTTOM никогда не классифицируется корректно.

    def test_thresholds_ordered(self):
        peak   = float(MOZART_CONFIG["mvrv_zscore_peak"])
        bull   = float(MOZART_CONFIG["mvrv_zscore_bull"])
        bottom = float(MOZART_CONFIG["mvrv_zscore_bottom"])
        assert bottom < 0.0 < bull < peak
        # WHY: нарушение порядка → зоны инвертируются или перекрываются;
        #   классификация становится непредсказуемой при изменении конфига.


# ---------------------------------------------------------------------------
# Контракт типа возврата
# ---------------------------------------------------------------------------

class TestMvrvZscoreReturnType:
    def test_returns_str(self):
        result = classify_mvrv_zscore_regime(0.0)
        assert isinstance(result, str)
        # WHY: оркестратор встраивает результат в f-строку;
        #   не-str → TypeError при форматировании вывода.


# ---------------------------------------------------------------------------
# Зоны — типичные значения (центры диапазонов)
# ---------------------------------------------------------------------------

class TestClassifyMvrvZscoreRegimeZones:
    def test_peak_zone(self):
        peak = float(MOZART_CONFIG["mvrv_zscore_peak"])
        result = classify_mvrv_zscore_regime(peak + 1.0)
        assert result == "PEAK"
        # WHY: Z > peak = исторический топ цикла; неверный лейбл → пропуск
        #   сигнала распределения в [FINAL VERDICT].

    def test_bull_zone(self):
        peak = float(MOZART_CONFIG["mvrv_zscore_peak"])
        bull = float(MOZART_CONFIG["mvrv_zscore_bull"])
        mid  = (peak + bull) / 2
        result = classify_mvrv_zscore_regime(mid)
        assert result == "BULL"
        # WHY: между bull и peak = сильный бычий рынок; неверный лейбл →
        #   оркестратор не сигнализирует перегрев рынка.

    def test_neutral_zone(self):
        bull = float(MOZART_CONFIG["mvrv_zscore_bull"])
        mid  = bull / 2  # между 0 и bull, например 1.5
        result = classify_mvrv_zscore_regime(mid)
        assert result == "NEUTRAL"
        # WHY: текущее API значение 0.7759 (19.05.2026) попадает в NEUTRAL.
        #   Неверный лейбл → оркестратор неверно оценивает текущую фазу рынка.

    def test_bear_zone(self):
        bottom = float(MOZART_CONFIG["mvrv_zscore_bottom"])
        mid    = bottom / 2  # между bottom и 0, например -0.5
        result = classify_mvrv_zscore_regime(mid)
        assert result == "BEAR"
        # WHY: чуть ниже Realized Price = ранняя медвежья зона;
        #   неверный лейбл → нет различия между «чуть ниже» и «дно цикла».

    def test_bottom_zone(self):
        bottom = float(MOZART_CONFIG["mvrv_zscore_bottom"])
        result = classify_mvrv_zscore_regime(bottom - 1.0)
        assert result == "BOTTOM"
        # WHY: Mozart (пост 25.02.2026): «Z < 0 = зона исторического дна».
        #   Неверный лейбл → оркестратор не сигнализирует исторический уровень дна.


# ---------------------------------------------------------------------------
# Граничные значения
# ---------------------------------------------------------------------------

class TestClassifyMvrvZscoreRegimeBoundaries:
    def test_peak_threshold_exact_is_bull(self):
        """Ровно на peak_threshold → BULL (PEAK строго >)."""
        threshold = float(MOZART_CONFIG["mvrv_zscore_peak"])
        result = classify_mvrv_zscore_regime(threshold)
        assert result == "BULL"
        # WHY: PEAK активируется строгим > (достижение порога = ещё не экстремум).
        #   Если бы z == 7.0 → PEAK: любой достижимый Z на пороге ложно
        #   сигнализировал бы топ цикла; нужно реальное превышение.

    def test_bull_threshold_exact_is_bull(self):
        """Ровно на bull_threshold → BULL (нижняя граница включительно)."""
        threshold = float(MOZART_CONFIG["mvrv_zscore_bull"])
        result = classify_mvrv_zscore_regime(threshold)
        assert result == "BULL"
        # WHY: граница BULL/NEUTRAL включительно снизу.
        #   Строгий > → z == 3.0 стало бы NEUTRAL; потеря сигнала
        #   сильного бычьего рынка при точном попадании на порог.

    def test_zero_is_neutral(self):
        """Z == 0.0 → NEUTRAL (рынок ровно на Realized Price)."""
        result = classify_mvrv_zscore_regime(0.0)
        assert result == "NEUTRAL"
        # WHY: Z = 0 = рынок ровно на средней цене покупки всех BTC.
        #   Это не медвежья территория — Realized Price не пробита вниз.
        #   Если 0.0 → BEAR: ложный медвежий сигнал при нейтральном рынке.

    def test_just_above_zero_is_neutral(self):
        """Минимально выше 0 → NEUTRAL, не BEAR."""
        result = classify_mvrv_zscore_regime(0.001)
        assert result == "NEUTRAL"
        # WHY: любое положительное Z = рынок выше Realized Price = NEUTRAL.
        #   z = 0.001 → BEAR = ложный сигнал при практически нейтральном рынке.

    def test_just_below_zero_is_bear(self):
        """Минимально ниже 0 → BEAR, не NEUTRAL."""
        result = classify_mvrv_zscore_regime(-0.001)
        assert result == "BEAR"
        # WHY: Z < 0 = рынок ниже Realized Price = переход в медвежью территорию.
        #   z = -0.001 → NEUTRAL = скрытый убыток по market cap; пропуск сигнала.

    def test_bottom_threshold_exact_is_bottom(self):
        """Ровно на bottom_threshold → BOTTOM (включительно)."""
        threshold = float(MOZART_CONFIG["mvrv_zscore_bottom"])
        result = classify_mvrv_zscore_regime(threshold)
        assert result == "BOTTOM"
        # WHY: Mozart: «Z<0 = зона исторического дна».
        #   Достижение bottom_threshold = активация сигнала дна.
        #   Строгий < → z ровно на пороге осталось бы в BEAR;
        #   оркестратор не сигнализировал бы исторический уровень.

    def test_extreme_negative_is_bottom(self):
        """Глубоко отрицательный Z → BOTTOM (устойчивость к экстремумам)."""
        result = classify_mvrv_zscore_regime(-10.0)
        assert result == "BOTTOM"
        # WHY: функция должна быть устойчива к любым float-значениям;
        #   экстремальный Z при гипотетическом крахе не должен давать
        #   неожиданный лейбл.
