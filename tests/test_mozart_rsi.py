"""
tests/test_mozart_rsi.py
========================
TDD — П1: RSI(14) guardrail (паттерн Н-01, PLAN_MOZART_PATTERNS.md ЧАСТЬ 3).

Два контракта:
  calculate_rsi(closes, period=14) -> float
      Формула Wilder. Не тестируем формулу — тестируем направление и границы.

  classify_rsi_regime(rsi: float) -> str
      Возвращает 'EXTREME_OVERSOLD' / 'OVERSOLD' / 'NEUTRAL'.
      Пороги берутся из MOZART_CONFIG — числа в assertions не хардкодятся.

Правила:
  - Числовые пороги только через MOZART_CONFIG, не магические числа.
  - Формула RSI в тестах не воспроизводится — только направление.
  - Синтетические данные явно искусственные (100.0, 1.0, не BTC-цены).
  - WHY-комментарий к каждому assert: что сломается в production.
  - Граничные значения (RSI == порогу) — отдельные тесты.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mozart_signals import calculate_rsi, classify_rsi_regime   # RED: файла нет
from mozart_config import MOZART_CONFIG


# ---------------------------------------------------------------------------
# Синтетические данные — явно искусственные, не BTC-реалистичные
# ---------------------------------------------------------------------------

def _closes_all_up(n: int = 20) -> list:
    """n закрытий строго вверх: 100, 101, 102 ... — гарантированно высокий RSI."""
    return [100.0 + i for i in range(n)]


def _closes_all_down(n: int = 20) -> list:
    """n закрытий строго вниз: 100, 99, 98 ... — гарантированно низкий RSI."""
    return [100.0 - i for i in range(n)]


# ---------------------------------------------------------------------------
# TestCalculateRsi
# ---------------------------------------------------------------------------

class TestCalculateRsi:
    """
    Контракт calculate_rsi(closes, period=14) -> float:
      - Принимает list или np.ndarray длиной > period.
      - Возвращает float в [0.0, 100.0].
      - Строго убывающий ряд → очень низкий RSI (ниже rsi_oversold из конфига).
      - Строго возрастающий ряд → RSI > 50.
      - Менее (period + 1) точек → ValueError.
    """

    def test_returns_float(self):
        # WHY: оркестратор сравнивает результат с числовыми порогами из MOZART_CONFIG;
        # не-float сломает сравнение без явной ошибки (например bool < float даёт сюрпризы).
        result = calculate_rsi(_closes_all_up())
        assert isinstance(result, float)

    def test_result_bounded_0_to_100(self):
        # WHY: RSI по определению [0, 100]; выход за диапазон означает
        # ошибку в формуле Wilder и сделает classify_rsi_regime бессмысленным.
        result = calculate_rsi(_closes_all_up())
        assert 0.0 <= result <= 100.0

    def test_all_up_gives_rsi_above_50(self):
        # WHY: при всех положительных изменениях avg_loss → 0, RSI → 100.
        # Значение > 50 — минимальная разумная граница для возрастающего ряда.
        # Тест проверяет направление, не точное значение (формула не воспроизводится).
        result = calculate_rsi(_closes_all_up())
        assert result > 50.0

    def test_all_down_gives_rsi_below_oversold_threshold(self):
        # WHY: строго убывающий ряд = сценарий паттерна Н-01 (пост 05.02.2026, RSI ~15).
        # Если реализация возвращает RSI >= rsi_oversold на таком ряду — формула сломана
        # и production-сигнал никогда не сработает.
        result = calculate_rsi(_closes_all_down())
        threshold = MOZART_CONFIG["rsi_oversold"]
        assert result < threshold

    def test_raises_value_error_on_insufficient_data(self):
        # WHY: с ровно period точками нельзя вычислить ни одного изменения для периода 14.
        # Тихий возврат NaN скроет ошибку в оркестраторе; явный ValueError — нет.
        with pytest.raises(ValueError):
            calculate_rsi(_closes_all_up(n=14), period=14)  # нужно минимум 15

    def test_accepts_numpy_array(self):
        # WHY: после pandas-обработки OHLCV оркестратор может передать np.ndarray;
        # оба типа должны давать корректный результат без дополнительного приведения.
        closes = np.array(_closes_all_up(), dtype=float)
        result = calculate_rsi(closes)
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0


# ---------------------------------------------------------------------------
# TestClassifyRsiRegime
# ---------------------------------------------------------------------------

class TestClassifyRsiRegime:
    """
    Контракт classify_rsi_regime(rsi: float) -> str:
      Зоны (строгое <):
        rsi < rsi_extreme_oversold             → 'EXTREME_OVERSOLD'
        rsi_extreme_oversold <= rsi < rsi_oversold → 'OVERSOLD'
        rsi >= rsi_oversold                    → 'NEUTRAL'

    Тестовые RSI вычисляются из MOZART_CONFIG — числа не хардкодятся.
    """

    # Тестовые значения — явно в середине каждой зоны
    @staticmethod
    def _rsi_extreme() -> float:
        """Середина зоны EXTREME_OVERSOLD: половина от порога."""
        return MOZART_CONFIG["rsi_extreme_oversold"] / 2.0

    @staticmethod
    def _rsi_oversold() -> float:
        """Середина зоны OVERSOLD: среднее между двумя порогами."""
        lo = MOZART_CONFIG["rsi_extreme_oversold"]
        hi = MOZART_CONFIG["rsi_oversold"]
        return (lo + hi) / 2.0

    @staticmethod
    def _rsi_neutral() -> float:
        """Явно в зоне NEUTRAL: порог + фиксированный отступ."""
        return MOZART_CONFIG["rsi_oversold"] + 10.0

    def test_returns_string(self):
        # WHY: оркестратор вставляет метку в строковый блок вывода;
        # не-str вызовет TypeError при форматировании без явной ошибки.
        result = classify_rsi_regime(self._rsi_neutral())
        assert isinstance(result, str)

    def test_extreme_oversold_zone(self):
        # WHY: паттерн Н-01 — RSI ~15 = исторически редкий экстремум (коронадамп 2020).
        # Неверная классификация лишит оркестратор самого сильного сигнала.
        result = classify_rsi_regime(self._rsi_extreme())
        assert result == "EXTREME_OVERSOLD"

    def test_oversold_zone(self):
        # WHY: зона OVERSOLD — повышенная вероятность отскока, но не экстремум.
        # Смешение с EXTREME_OVERSOLD даст ложно-сильный сигнал.
        result = classify_rsi_regime(self._rsi_oversold())
        assert result == "OVERSOLD"

    def test_neutral_zone(self):
        # WHY: большинство торговых дней — нейтральные.
        # Ошибочный OVERSOLD при нейтральном RSI создаст постоянный ложный сигнал.
        result = classify_rsi_regime(self._rsi_neutral())
        assert result == "NEUTRAL"

    def test_boundary_at_extreme_oversold_falls_into_oversold(self):
        # WHY: фиксирует правило строгого "<" на нижней границе.
        # RSI == rsi_extreme_oversold → OVERSOLD, не EXTREME_OVERSOLD.
        # Без этого теста ошибка "<=" vs "<" остаётся незамеченной.
        rsi = float(MOZART_CONFIG["rsi_extreme_oversold"])
        result = classify_rsi_regime(rsi)
        assert result == "OVERSOLD"

    def test_boundary_at_oversold_falls_into_neutral(self):
        # WHY: то же правило строгого "<" на верхней границе зоны OVERSOLD.
        # RSI == rsi_oversold → NEUTRAL, не OVERSOLD.
        rsi = float(MOZART_CONFIG["rsi_oversold"])
        result = classify_rsi_regime(rsi)
        assert result == "NEUTRAL"

    def test_only_valid_labels_returned(self):
        # WHY: опечатка в метке ('OVERSOLD ' с пробелом) — тихий баг в оркестраторе.
        # Перебираем все три зоны и проверяем что метка входит в допустимое множество.
        valid = {"EXTREME_OVERSOLD", "OVERSOLD", "NEUTRAL"}
        for rsi in [self._rsi_extreme(), self._rsi_oversold(), self._rsi_neutral()]:
            assert classify_rsi_regime(rsi) in valid
