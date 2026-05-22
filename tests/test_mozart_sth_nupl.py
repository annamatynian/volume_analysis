"""tests/test_mozart_sth_nupl.py
==========================
TDD — ВЕТКА 7: М-06 STH NUPL (один классификатор, 3 зоны с асимметричным рубиконом).

Контракт:

  classify_sth_nupl_regime(nupl: float) -> str

      3 зоны (пост 16.04.2026):

        POSITIVE     : nupl >  rubicon (0.0)
                       STH когорта в нереализованной прибыли; Mozart: «запас на
                       рост по-прежнему имеет место».

        RUBICON      : rubicon - eps <= nupl <= rubicon
                       STH на нуле или чуть ниже; зона максимального давления.
                       Верхняя граница включительно: nupl == 0.0 → RUBICON (не POSITIVE).
                       Нижняя граница включительно: nupl == -eps → RUBICON (не CAPITULATION).

        CAPITULATION : nupl <  rubicon - eps  (-0.005)
                       STH когортно в убытке; смена режима давления.

Асимметрия рубикона (отличие от LTH NUPL):
  LTH NUPL: симметричный eps-буфер вокруг 0.0 (и сверху, и снизу).
  STH NUPL: асимметричный — eps только снизу.
    nupl == 0.0 → RUBICON (не POSITIVE; нет буфера сверху).
    Нет зоны EUPHORIA (3 зоны вместо 4 у LTH NUPL).

Правила:
  - Числовые пороги только через MOZART_CONFIG, не хардкодятся в assertions.
  - Тестовые значения вычисляются из порогов.
  - _STEP = 0.001 — шаг «just outside» зоны (ниже eps=0.005; достаточен).
  - WHY-комментарий к каждому assert: что сломается в production.
  - Все 3 зоны, обе границы — отдельные тесты.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# RED: функции ещё нет — ImportError подтверждает RED
from mozart_signals import classify_sth_nupl_regime
from mozart_config import MOZART_CONFIG


# ---------------------------------------------------------------------------
# _STEP: шаг позиционирования «just outside» зоны.
# 0.001 = 0.1% — ниже eps=0.005 (0.5%); достаточен для однозначного перехода
# через границу без захода в другую зону.
# По аналогии с sth_mvrv (_STEP=0.001) и lth_nupl (_STEP=0.001).
# ---------------------------------------------------------------------------
_STEP = 0.001


# ---------------------------------------------------------------------------
# Генераторы тестовых значений
# ---------------------------------------------------------------------------

def _positive_center() -> float:
    """Центр POSITIVE-зоны: rubicon + 0.1 (явно выше рубикона, не API-значение)."""
    return float(MOZART_CONFIG["sth_nupl_rubicon"]) + 0.1


def _rubicon_center() -> float:
    """Центр RUBICON-зоны: rubicon - eps/2 (середина буфера снизу от рубикона)."""
    r = float(MOZART_CONFIG["sth_nupl_rubicon"])
    e = float(MOZART_CONFIG["sth_nupl_rubicon_eps"])
    return r - e / 2.0


def _capitulation_center() -> float:
    """Центр CAPITULATION-зоны: rubicon - eps - 0.1 (явно ниже нижней границы)."""
    r = float(MOZART_CONFIG["sth_nupl_rubicon"])
    e = float(MOZART_CONFIG["sth_nupl_rubicon_eps"])
    return r - e - 0.1


# ===========================================================================
# TestClassifySthNuplRegime — М-06 | STH NUPL
# ===========================================================================

class TestClassifySthNuplRegime:
    """
    Контракт classify_sth_nupl_regime(nupl: float) -> str:

    Паттерн М-06 (пост 16.04.2026):
      POSITIVE     : nupl >  rubicon (0.0)
                     STH когорта в нереализованной прибыли; Mozart: «запас на
                     рост по-прежнему имеет место» (пост 16.04.2026).

      RUBICON      : rubicon - eps <= nupl <= rubicon
                     STH на нуле или чуть ниже; зона максимального давления.
                     Верхняя граница включительно: nupl == 0.0 → RUBICON (не POSITIVE).
                     Нижняя граница включительно: nupl == -eps → RUBICON (не CAPITULATION).

      CAPITULATION : nupl <  rubicon - eps (-0.005)
                     STH когортно в убытке; смена режима давления.

    Асимметрия: eps защищает только нижнюю границу RUBICON-зоны.
    nupl == 0.0 → RUBICON (STH ровно на безубытке = давление, не прибыль).
    """

    # -- Тип возвращаемого значения ------------------------------------------

    def test_returns_string(self):
        # WHY: оркестратор вставляет метку в строковый блок вывода;
        # не-str вызовет TypeError при f-строке без явной ошибки в runtime.
        result = classify_sth_nupl_regime(_positive_center())
        assert isinstance(result, str)

    # -- Центры зон ----------------------------------------------------------

    def test_positive_zone(self):
        # WHY: nupl явно выше рубикона → STH когортно в прибыли.
        # Ошибочный RUBICON скроет сигнал запаса на рост; оркестратор
        # не зафиксирует Mozart-паттерн «NUPL > 0 = есть запас» (пост 16.04.2026).
        assert classify_sth_nupl_regime(_positive_center()) == "POSITIVE"

    def test_rubicon_zone_at_center(self):
        # WHY: nupl в центре RUBICON-буфера (rubicon - eps/2) → RUBICON.
        # Ошибочный CAPITULATION ложно сигнализирует капитуляцию при
        # дневном шуме у нуля; Mozart трактует переход через 0 как событие,
        # eps защищает от ложных срабатываний (пост 16.04.2026).
        assert classify_sth_nupl_regime(_rubicon_center()) == "RUBICON"

    def test_capitulation_zone(self):
        # WHY: nupl явно ниже rubicon - eps → STH когортно в убытке.
        # Ошибочный RUBICON скроет сигнал капитуляции STH; оркестратор
        # не выдаст предупреждение о смене режима давления в отчёте.
        assert classify_sth_nupl_regime(_capitulation_center()) == "CAPITULATION"

    # -- Верхняя граница RUBICON (асимметрия: нет eps сверху) ---------------

    def test_boundary_rubicon_exact_is_rubicon(self):
        # WHY: nupl == 0.0 (rubicon) → RUBICON (не POSITIVE).
        # Ключевой контракт асимметрии: POSITIVE начинается строго > 0.
        # STH ровно на безубытке = зона давления (не «запас на рост»).
        # Ошибка >= вместо > для POSITIVE: nupl == 0.0 → POSITIVE —
        # ложный сигнал прибыли при максимальном давлении (пост 16.04.2026).
        rubicon = float(MOZART_CONFIG["sth_nupl_rubicon"])
        assert classify_sth_nupl_regime(rubicon) == "RUBICON"

    def test_boundary_just_above_rubicon_is_positive(self):
        # WHY: nupl == rubicon + _STEP (0.001) → POSITIVE.
        # Первое значение строго выше рубикона = STH вошли в прибыль.
        # Фиксирует что POSITIVE начинается сразу после рубикона, без мёртвой зоны.
        # Без этого теста ошибка знака верхней границы остаётся незамеченной.
        rubicon = float(MOZART_CONFIG["sth_nupl_rubicon"])
        assert classify_sth_nupl_regime(rubicon + _STEP) == "POSITIVE"

    # -- Нижняя граница RUBICON (eps-защита перед CAPITULATION) -------------

    def test_boundary_lower_exact_eps_is_rubicon(self):
        # WHY: nupl == rubicon - eps (-0.005) → RUBICON (не CAPITULATION).
        # Нижняя граница включительно: значение ровно на eps — дневной шум,
        # не реальная капитуляция когорты.
        # Ошибка > вместо >= для RUBICON: nupl == -eps → CAPITULATION —
        # ложный сигнал капитуляции при погранично-шумовом значении.
        rubicon = float(MOZART_CONFIG["sth_nupl_rubicon"])
        eps = float(MOZART_CONFIG["sth_nupl_rubicon_eps"])
        assert classify_sth_nupl_regime(rubicon - eps) == "RUBICON"

    def test_boundary_just_below_eps_is_capitulation(self):
        # WHY: nupl == rubicon - eps - _STEP (-0.006) → CAPITULATION.
        # Первое значение строго ниже нижней границы = когорта STH в убытке.
        # Фиксирует что CAPITULATION начинается сразу под rubicon - eps.
        # Без этого теста ошибка нижней границы остаётся незамеченной.
        rubicon = float(MOZART_CONFIG["sth_nupl_rubicon"])
        eps = float(MOZART_CONFIG["sth_nupl_rubicon_eps"])
        assert classify_sth_nupl_regime(rubicon - eps - _STEP) == "CAPITULATION"

    # -- Корректность меток -------------------------------------------------

    def test_only_valid_labels_returned(self):
        # WHY: опечатка в метке ('Positive', 'CAPITULATION ' с пробелом)
        # — тихий баг: оркестратор не упадёт, но условная логика сломается.
        # Проверяем все 3 зоны — ловим опечатку в любой ветке return.
        valid = {"POSITIVE", "RUBICON", "CAPITULATION"}
        for nupl in [_positive_center(), _rubicon_center(), _capitulation_center()]:
            assert classify_sth_nupl_regime(nupl) in valid
