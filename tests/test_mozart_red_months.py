"""
tests/test_mozart_red_months.py
================================
TDD — П2: счётчик красных месячных свечей (паттерн Н-02,
PLAN_MOZART_PATTERNS.md ЧАСТЬ 3).

Два контракта:
  count_consecutive_red_months(ohlcv: list) -> int
      ohlcv — список свечей в формате ccxt: [ts, open, high, low, close, volume].
      Красная свеча: close < open.
      Счётчик считается с конца (последние N подряд красных).
      Сбрасывается при первой не-красной свече (close >= open).

  classify_red_months_regime(count: int) -> str
      Возвращает 'EXTREME' / 'RARE' / 'NORMAL'.
      Пороги только из MOZART_CONFIG — числа не хардкодятся.

Правила:
  - Синтетические свечи: open/close = простые целые числа (10, 9, ...), не BTC-цены.
  - Пороги берутся из MOZART_CONFIG, не хардкодятся в assertions.
  - Формат ccxt воспроизводится минимально: [ts, open, high, low, close, volume].
  - WHY-комментарий к каждому assert.
  - Граничные значения (count == порогу) — отдельные тесты.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mozart_signals import (       # RED: функций ещё нет
    count_consecutive_red_months,
    classify_red_months_regime,
)
from mozart_config import MOZART_CONFIG


# ---------------------------------------------------------------------------
# Вспомогательные фабрики свечей
# ---------------------------------------------------------------------------

def _red(ts: int = 0) -> list:
    """Красная свеча: close < open. Числа явно искусственные."""
    return [ts, 10, 11, 8, 9, 100]   # open=10, close=9


def _green(ts: int = 0) -> list:
    """Зелёная свеча: close > open."""
    return [ts, 9, 12, 8, 10, 100]   # open=9, close=10


def _doji(ts: int = 0) -> list:
    """Дожи: close == open — не красная (сброс счётчика)."""
    return [ts, 10, 12, 8, 10, 100]  # open=10, close=10


# ---------------------------------------------------------------------------
# TestCountConsecutiveRedMonths
# ---------------------------------------------------------------------------

class TestCountConsecutiveRedMonths:
    """
    Контракт count_consecutive_red_months(ohlcv: list) -> int:
      - ohlcv: список свечей ccxt [ts, open, high, low, close, volume].
      - Возвращает число последних подряд идущих красных свечей (с конца списка).
      - Красная: close (индекс 4) < open (индекс 1).
      - Не-красная (close >= open): сбрасывает счётчик.
      - Пустой список → 0.
      - Одна красная → 1, одна зелёная → 0.
    """

    def test_returns_int(self):
        # WHY: classify_red_months_regime сравнивает результат с int-порогами
        # из MOZART_CONFIG; не-int может дать неожиданный результат при сравнении.
        result = count_consecutive_red_months([_red(), _red()])
        assert isinstance(result, int)

    def test_empty_ohlcv_returns_zero(self):
        # WHY: оркестратор вызывает функцию до получения данных ccxt (timeout/ошибка);
        # исключение здесь остановит весь pipeline.
        result = count_consecutive_red_months([])
        assert result == 0

    def test_single_red_candle(self):
        # WHY: минимальный ненулевой случай — одна красная свеча в конце.
        result = count_consecutive_red_months([_red()])
        assert result == 1

    def test_single_green_candle(self):
        # WHY: одна зелёная свеча — счётчик должен быть 0, не 1.
        result = count_consecutive_red_months([_green()])
        assert result == 0

    def test_doji_is_not_red(self):
        # WHY: close == open — граничный случай определения «красной» свечи.
        # Паттерн Н-02 использует строгое «close < open»; дожи = сброс счётчика.
        result = count_consecutive_red_months([_doji()])
        assert result == 0

    def test_counts_from_the_end(self):
        # WHY: счётчик считает последовательность С КОНЦА (текущее состояние рынка).
        # Красные в середине истории не считаются — важна только текущая серия.
        ohlcv = [_red(), _red(), _green(), _red(), _red(), _red()]
        result = count_consecutive_red_months(ohlcv)
        assert result == 3

    def test_green_in_middle_resets_counter(self):
        # WHY: одна зелёная свеча прерывает серию; предыдущие красные не считаются.
        # Это core-контракт паттерна Н-02 (пост 11.02.2026).
        ohlcv = [_red(), _red(), _green(), _red()]
        result = count_consecutive_red_months(ohlcv)
        assert result == 1

    def test_all_red_returns_full_count(self):
        # WHY: если все свечи красные — счётчик равен длине списка.
        n = 6
        ohlcv = [_red(i) for i in range(n)]
        result = count_consecutive_red_months(ohlcv)
        assert result == n

    def test_all_green_returns_zero(self):
        # WHY: нет красных свечей → счётчик 0; функция не должна считать зелёные.
        ohlcv = [_green(i) for i in range(5)]
        result = count_consecutive_red_months(ohlcv)
        assert result == 0

    def test_last_candle_green_returns_zero(self):
        # WHY: серия красных, но последняя зелёная — текущий счётчик = 0.
        # Оркестратор получает актуальное состояние, не историческое.
        ohlcv = [_red(), _red(), _red(), _green()]
        result = count_consecutive_red_months(ohlcv)
        assert result == 0


# ---------------------------------------------------------------------------
# TestClassifyRedMonthsRegime
# ---------------------------------------------------------------------------

class TestClassifyRedMonthsRegime:
    """
    Контракт classify_red_months_regime(count: int) -> str:
      Зоны (строгое >=):
        count >= red_months_extreme → 'EXTREME'
        count >= red_months_rare    → 'RARE'
        иначе                       → 'NORMAL'

    Пороги из MOZART_CONFIG — числа не хардкодятся.
    Важная оговорка Н-02: 'EXTREME' ≠ конец медвежки,
    указывает на экстремальную перепроданность по месячному ТФ.
    """

    @staticmethod
    def _count_extreme() -> int:
        """Явно в зоне EXTREME: порог + 1."""
        return MOZART_CONFIG["red_months_extreme"] + 1

    @staticmethod
    def _count_rare() -> int:
        """Середина зоны RARE: между rare и extreme."""
        lo = MOZART_CONFIG["red_months_rare"]
        hi = MOZART_CONFIG["red_months_extreme"]
        return (lo + hi) // 2

    @staticmethod
    def _count_normal() -> int:
        """Явно в зоне NORMAL: ниже rare."""
        return max(0, MOZART_CONFIG["red_months_rare"] - 1)

    def test_returns_string(self):
        # WHY: метка вставляется в строковый блок оркестратора;
        # не-str вызовет TypeError при форматировании.
        result = classify_red_months_regime(self._count_normal())
        assert isinstance(result, str)

    def test_extreme_zone(self):
        # WHY: N >= red_months_extreme = единственный исторический прецедент BTC
        # (авг–дек 2018); неверная классификация скроет редчайший рыночный сигнал.
        result = classify_red_months_regime(self._count_extreme())
        assert result == "EXTREME"

    def test_rare_zone(self):
        # WHY: N >= red_months_rare — редкость, повышенная вероятность отскока.
        # Смешение с EXTREME даст ложно-сильный сигнал.
        result = classify_red_months_regime(self._count_rare())
        assert result == "RARE"

    def test_normal_zone(self):
        # WHY: большинство периодов — нормальные.
        # Ошибочный RARE при нормальном счётчике создаст постоянный ложный сигнал.
        result = classify_red_months_regime(self._count_normal())
        assert result == "NORMAL"

    def test_zero_count_is_normal(self):
        # WHY: нет красных свечей = нет сигнала; минимальный граничный случай.
        result = classify_red_months_regime(0)
        assert result == "NORMAL"

    def test_boundary_at_rare_is_rare(self):
        # WHY: фиксирует правило >= на нижней границе зоны RARE.
        # count == red_months_rare → 'RARE', не 'NORMAL'.
        count = MOZART_CONFIG["red_months_rare"]
        result = classify_red_months_regime(count)
        assert result == "RARE"

    def test_boundary_at_extreme_is_extreme(self):
        # WHY: фиксирует правило >= на нижней границе зоны EXTREME.
        # count == red_months_extreme → 'EXTREME', не 'RARE'.
        count = MOZART_CONFIG["red_months_extreme"]
        result = classify_red_months_regime(count)
        assert result == "EXTREME"

    def test_only_valid_labels_returned(self):
        # WHY: опечатка в метке — тихий баг в оркестраторе.
        valid = {"EXTREME", "RARE", "NORMAL"}
        for count in [self._count_normal(), self._count_rare(), self._count_extreme()]:
            assert classify_red_months_regime(count) in valid
