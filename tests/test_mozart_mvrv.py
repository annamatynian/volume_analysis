"""
tests/test_mozart_mvrv.py
==========================
TDD — ВЕТКА 6: М-03 LTH MVRV + М-04 STH MVRV (оба классификатора, один файл).

Два контракта:

  classify_lth_mvrv_regime(mvrv: float) -> str
      Бинарный рубикон 1.0 (пост 25.02.2026):
        mvrv >= rubicon -> 'BULL'  — LTH в нереализованной прибыли
        mvrv <  rubicon -> 'BEAR'  — LTH в убытке, вынужденные продажи снижаются
      Граница включительно: mvrv == 1.0 -> BULL.
      Нет eps: LTH MVRV — агрегированная когортная метрика, дневной шум
      менее значим чем у STH.

  classify_sth_mvrv_regime(mvrv: float) -> str
      Рубикон 1.0 с eps-буфером 0.02 (+-2%, пост 16.04.2026):
        mvrv >  rubicon + eps -> 'BULL'    — STH продают с прибылью, давление продаж
        mvrv >= rubicon - eps -> 'NEUTRAL' — STH у безубытка, нейтральная зона
        mvrv <  rubicon - eps -> 'BEAR'    — STH капитулируют
      Верхняя граница NEUTRAL включительно: mvrv == rubicon + eps -> NEUTRAL (не BULL).
      Нижняя граница NEUTRAL включительно: mvrv == rubicon - eps -> NEUTRAL (не BEAR).

Отличия:
  LTH MVRV — бинарный (2 зоны, нет eps, нет NEUTRAL).
  STH MVRV — 3 зоны с eps=0.02 (аналог STH SOPR и LTH NUPL по структуре теста).

Правила:
  - Числовые пороги только через MOZART_CONFIG, не хардкодятся в assertions.
  - Тестовые значения вычисляются из порогов.
  - _STEP = 0.001 — шаг «just outside» зоны (по аналогии с sth_sopr).
  - WHY-комментарий к каждому assert: что сломается в production.
  - Все зоны, обе границы — отдельные тесты.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# RED: функций ещё нет — ImportError подтверждает RED
from mozart_signals import classify_lth_mvrv_regime, classify_sth_mvrv_regime
from mozart_config import MOZART_CONFIG


# ---------------------------------------------------------------------------
# _STEP: шаг позиционирования «just outside» зоны.
# 0.001 = 0.1% — ниже eps=0.02 (2%); достаточен для однозначного перехода
# через границу без захода в другую зону.
# По аналогии с sth_sopr (_STEP = 0.001) и lth_nupl (_STEP = 0.001).
# ---------------------------------------------------------------------------
_STEP = 0.001


# ---------------------------------------------------------------------------
# Генераторы тестовых значений для LTH MVRV (бинарный)
# ---------------------------------------------------------------------------

def _lth_bull_center() -> float:
    """Центр BULL-зоны: rubicon + 0.5 (явно выше рубикона, не API-значение)."""
    return float(MOZART_CONFIG["lth_mvrv_rubicon"]) + 0.5


def _lth_bear_center() -> float:
    """Центр BEAR-зоны: rubicon - 0.5 (явно ниже рубикона, не API-значение)."""
    return float(MOZART_CONFIG["lth_mvrv_rubicon"]) - 0.5


# ---------------------------------------------------------------------------
# Генераторы тестовых значений для STH MVRV (3 зоны с eps)
# ---------------------------------------------------------------------------

def _sth_bull_center() -> float:
    """Центр BULL-зоны: rubicon + eps + 0.1 (явно выше верхней границы NEUTRAL)."""
    r = float(MOZART_CONFIG["sth_mvrv_rubicon"])
    e = float(MOZART_CONFIG["sth_mvrv_rubicon_eps"])
    return r + e + 0.1


def _sth_neutral_center() -> float:
    """Центр NEUTRAL-зоны: rubicon (точная середина +-eps буфера)."""
    return float(MOZART_CONFIG["sth_mvrv_rubicon"])


def _sth_bear_center() -> float:
    """Центр BEAR-зоны: rubicon - eps - 0.1 (явно ниже нижней границы NEUTRAL)."""
    r = float(MOZART_CONFIG["sth_mvrv_rubicon"])
    e = float(MOZART_CONFIG["sth_mvrv_rubicon_eps"])
    return r - e - 0.1


# ===========================================================================
# TestClassifyLthMvrvRegime — М-03 | LTH MVRV (бинарный рубикон)
# ===========================================================================

class TestClassifyLthMvrvRegime:
    """
    Контракт classify_lth_mvrv_regime(mvrv: float) -> str:

    Паттерн М-03 (пост 25.02.2026):
      BULL : mvrv >= rubicon (1.0)
             LTH в нереализованной прибыли; среднестатистический LTH выше себестоимости.
             Mozart: LTH не капитулируют принудительно — структурная поддержка.

      BEAR : mvrv < rubicon (1.0)
             LTH в убытке когортно; вынужденные продажи снижаются при уходе под 1.0.
             Mozart: рубикон 1.0 означает смену структуры рынка (пост 25.02.2026).

    Граница: mvrv == 1.0 -> BULL (рубикон не пробит; включительно сверху).
    Нет eps: LTH MVRV агрегирован по всей когорте — дневные флуктуации
    менее значимы, чем у STH (краткосрочных держателей с высокой ротацией).
    """

    # -- Зоны: центры диапазонов ---------------------------------------------

    def test_lth_returns_string(self):
        # WHY: оркестратор вставляет метку в строковый блок вывода;
        # не-str вызовет TypeError при f-строке без явной ошибки в runtime.
        result = classify_lth_mvrv_regime(_lth_bull_center())
        assert isinstance(result, str)

    def test_lth_bull_zone(self):
        # WHY: mvrv явно выше рубикона -> LTH в прибыли структурно.
        # Ошибочный BEAR скроет от оркестратора бычий сигнал структурной поддержки
        # и ложно сигнализирует о вынужденных продажах LTH (пост 25.02.2026).
        assert classify_lth_mvrv_regime(_lth_bull_center()) == "BULL"

    def test_lth_bear_zone(self):
        # WHY: mvrv явно ниже рубикона -> LTH в убытке когортно.
        # Ошибочный BULL скроет медвежий контекст: Mozart использует MVRV < 1.0
        # как признак «вынужденные продажи снижаются» только в BEAR-режиме;
        # BULL при убытке — логическое противоречие в отчёте.
        assert classify_lth_mvrv_regime(_lth_bear_center()) == "BEAR"

    # -- Граничные значения рубикона (самые частые тихие баги) ---------------

    def test_lth_boundary_exact_rubicon_is_bull(self):
        # WHY: mvrv == 1.0 -> BULL (граница включительно; рубикон не пробит вниз).
        # Ошибка строгого > вместо >= для BULL: mvrv == 1.0 -> BEAR —
        # ложный медвежий сигнал при безубытке LTH когортно.
        # Самая частая ошибка при бинарном рубиконе: off-by-one на границе.
        rubicon = float(MOZART_CONFIG["lth_mvrv_rubicon"])
        assert classify_lth_mvrv_regime(rubicon) == "BULL"

    def test_lth_boundary_just_below_rubicon_is_bear(self):
        # WHY: mvrv == rubicon - _STEP -> BEAR.
        # Первое значение строго ниже рубикона = LTH вошли в убыток когортно.
        # Фиксирует что BEAR начинается сразу под рубиконом, без мёртвой зоны.
        # Без этого теста ошибка (rubicon - _STEP -> BULL) остаётся незамеченной.
        rubicon = float(MOZART_CONFIG["lth_mvrv_rubicon"])
        assert classify_lth_mvrv_regime(rubicon - _STEP) == "BEAR"

    # -- Корректность меток --------------------------------------------------

    def test_lth_only_valid_labels_returned(self):
        # WHY: опечатка в метке ('Bull', 'BULL ' с пробелом) — тихий баг:
        # оркестратор не упадёт, но str == 'BULL' вернёт False без исключения.
        # Проверяем обе зоны — ловим опечатку в любой ветке return.
        valid = {"BULL", "BEAR"}
        for mvrv in [_lth_bull_center(), _lth_bear_center()]:
            assert classify_lth_mvrv_regime(mvrv) in valid


# ===========================================================================
# TestClassifySthMvrvRegime — М-04 | STH MVRV (3 зоны с eps-буфером)
# ===========================================================================

class TestClassifySthMvrvRegime:
    """
    Контракт classify_sth_mvrv_regime(mvrv: float) -> str:

    Паттерн М-04 (пост 16.04.2026):
      BULL    : mvrv > rubicon + eps
                STH продают с прибылью -> давление продаж активно.
                Mozart: STH MVRV выше 1.0 = реализация прибыли STH.

      NEUTRAL : rubicon - eps <= mvrv <= rubicon + eps
                STH у безубытка +-eps: нейтральная зона давления/поддержки.
                Mozart: «сильная поддержка / сильное сопротивление»
                (аналогия с STH SOPR == 1.0, пост 16.04.2026).
                Верхняя граница включительно: mvrv == rubicon + eps -> NEUTRAL.
                Нижняя граница включительно: mvrv == rubicon - eps -> NEUTRAL.

      BEAR    : mvrv < rubicon - eps
                STH когортно в убытке -> капитуляция.

    eps = MOZART_CONFIG["sth_mvrv_rubicon_eps"] = 0.02
    """

    # -- Зоны: центры диапазонов ---------------------------------------------

    def test_sth_returns_string(self):
        # WHY: оркестратор вставляет метку в строковый блок вывода;
        # не-str вызовет TypeError при f-строке без явной ошибки в runtime.
        result = classify_sth_mvrv_regime(_sth_neutral_center())
        assert isinstance(result, str)

    def test_sth_bull_zone(self):
        # WHY: mvrv явно выше rubicon + eps -> STH продают с прибылью.
        # Ошибочный NEUTRAL скроет давление продаж от STH; оркестратор
        # не выдаст предупреждение о распределении когорты (пост 16.04.2026).
        assert classify_sth_mvrv_regime(_sth_bull_center()) == "BULL"

    def test_sth_neutral_zone_at_center(self):
        # WHY: mvrv == rubicon (центр) -> NEUTRAL (STH ровно на безубытке).
        # Ошибочный BULL или BEAR изменит интерпретацию нейтрального состояния;
        # Mozart явно выделяет «около 1.0» как зону давления/поддержки.
        assert classify_sth_mvrv_regime(_sth_neutral_center()) == "NEUTRAL"

    def test_sth_bear_zone(self):
        # WHY: mvrv явно ниже rubicon - eps -> STH когортно в убытке.
        # Ошибочный NEUTRAL скроет сигнал капитуляции STH; оркестратор
        # не зафиксирует потенциальный отскок от уровня вынужденных продаж.
        assert classify_sth_mvrv_regime(_sth_bear_center()) == "BEAR"

    # -- Верхняя граница NEUTRAL-зоны ----------------------------------------

    def test_sth_boundary_upper_exact_is_neutral(self):
        # WHY: mvrv == rubicon + eps -> NEUTRAL (не BULL; верхний край включительно).
        # Контракт: значение ровно на верхней границе — не «прибыльная продажа»,
        # а граница нейтральной зоны. BULL начинается строго выше.
        # Ошибка >= вместо > для BULL: mvrv == rubicon + eps -> BULL —
        # ложный сигнал давления продаж при нейтральном STH MVRV.
        rubicon = float(MOZART_CONFIG["sth_mvrv_rubicon"])
        eps = float(MOZART_CONFIG["sth_mvrv_rubicon_eps"])
        assert classify_sth_mvrv_regime(rubicon + eps) == "NEUTRAL"

    def test_sth_boundary_just_above_upper_is_bull(self):
        # WHY: mvrv == rubicon + eps + _STEP -> BULL.
        # Первое значение строго выше верхней границы = реальное давление продаж STH.
        # Фиксирует что BULL начинается ПОСЛЕ rubicon + eps, а не на нём.
        # Без этого теста ошибка знака верхней границы остаётся незамеченной.
        rubicon = float(MOZART_CONFIG["sth_mvrv_rubicon"])
        eps = float(MOZART_CONFIG["sth_mvrv_rubicon_eps"])
        assert classify_sth_mvrv_regime(rubicon + eps + _STEP) == "BULL"

    # -- Нижняя граница NEUTRAL-зоны -----------------------------------------

    def test_sth_boundary_lower_exact_is_neutral(self):
        # WHY: mvrv == rubicon - eps -> NEUTRAL (не BEAR; нижний край включительно).
        # Контракт: значение ровно на нижней границе — не «капитуляция»,
        # а граница нейтральной зоны. BEAR начинается строго ниже.
        # Ошибка > вместо >= для NEUTRAL: mvrv == rubicon - eps -> BEAR —
        # ложный сигнал капитуляции STH при нейтральном MVRV на нижней границе.
        rubicon = float(MOZART_CONFIG["sth_mvrv_rubicon"])
        eps = float(MOZART_CONFIG["sth_mvrv_rubicon_eps"])
        assert classify_sth_mvrv_regime(rubicon - eps) == "NEUTRAL"

    def test_sth_boundary_just_below_lower_is_bear(self):
        # WHY: mvrv == rubicon - eps - _STEP -> BEAR.
        # Первое значение строго ниже нижней границы = реальная капитуляция STH.
        # Фиксирует что BEAR начинается сразу под rubicon - eps.
        # Без этого теста ошибка нижней границы остаётся незамеченной.
        rubicon = float(MOZART_CONFIG["sth_mvrv_rubicon"])
        eps = float(MOZART_CONFIG["sth_mvrv_rubicon_eps"])
        assert classify_sth_mvrv_regime(rubicon - eps - _STEP) == "BEAR"

    # -- Корректность меток --------------------------------------------------

    def test_sth_only_valid_labels_returned(self):
        # WHY: опечатка в метке ('Neutral', 'BULL ' с пробелом) — тихий баг:
        # оркестратор не упадёт, но условная логика перестанет работать.
        # Проверяем все три зоны — ловим опечатку в любой ветке return.
        valid = {"BULL", "NEUTRAL", "BEAR"}
        for mvrv in [_sth_bull_center(), _sth_neutral_center(), _sth_bear_center()]:
            assert classify_sth_mvrv_regime(mvrv) in valid
