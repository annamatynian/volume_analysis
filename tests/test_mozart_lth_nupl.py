"""
tests/test_mozart_lth_nupl.py
==============================
TDD — ВЕТКА 3: М-05 LTH NUPL рубикон + эйфория
(паттерн М-05, PLAN_MOZART_PATTERNS.md ЧАСТЬ 1, посты 05.04.2026 и 15.05.2026).

Один контракт:

  classify_lth_nupl_regime(nupl: float) -> str
      Классифицирует LTH NUPL по четырём зонам паттерна М-05.
      Зоны (приоритет сверху вниз):
        nupl >= euphoria                        → 'EUPHORIA' — риск распределения LTH
        nupl >  rubicon + eps                   → 'POSITIVE' — LTH в нереализованной прибыли
        rubicon - eps <= nupl <= rubicon + eps  → 'RUBICON'  — граница смены рынка
        nupl <  rubicon - eps                   → 'BEAR'     — LTH в убытке
      Граница euphoria: nupl == euphoria → 'EUPHORIA' (включительно).
      Границы RUBICON-зоны:
        nupl == rubicon + eps → 'RUBICON' (не POSITIVE — верхний край включительно).
        nupl == rubicon - eps → 'RUBICON' (не BEAR — нижний край включительно).
      Пороги euphoria и rubicon строго из MOZART_CONFIG.
      eps = lth_nupl_rubicon_eps из MOZART_CONFIG (не хардкодится).

Примечание по источнику порога euphoria:
  Mozart явно не называет 0.75 для LTH NUPL в постах 05.04.2026 / 15.05.2026.
  Значение взято из реализации [LTH PAIN PROXY]. При уточнении Mozart —
  обновить только mozart_config.py; тесты не изменятся (берут из конфига).

Правила:
  - Числовые пороги только через MOZART_CONFIG, не хардкодятся в assertions.
  - Тестовые значения вычисляются из порогов (середина зоны, смещение от границы).
  - _STEP = 0.001: шаг позиционирования «just outside» зоны, не рыночный порог.
  - WHY-комментарий к каждому assert: что сломается в production.
  - Все 4 зоны, граница euphoria, обе границы RUBICON-зоны — отдельные тесты.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# RED: функции ещё нет — ImportError подтверждает RED
from mozart_signals import classify_lth_nupl_regime
from mozart_config import MOZART_CONFIG


# ---------------------------------------------------------------------------
# _STEP: шаг для позиционирования тестовых значений «just outside» зоны.
# 0.001 = 1/5 от rubicon_eps (0.005) — достаточно мал, чтобы чётко
# определять «just inside» vs «just outside», но не является рыночным порогом.
# ---------------------------------------------------------------------------
_STEP = 0.001


# ---------------------------------------------------------------------------
# Генераторы тестовых значений — вычисляются из конфига, не хардкодятся
# ---------------------------------------------------------------------------

def _nupl_euphoria_center() -> float:
    """
    Центр зоны EUPHORIA: euphoria + 0.10.
    0.10 — шаг позиционирования, явно выше порога, не API-значение.
    """
    return MOZART_CONFIG["lth_nupl_euphoria"] + 0.10


def _nupl_positive_center() -> float:
    """
    Центр зоны POSITIVE: среднее между (rubicon + eps) и euphoria.
    Завёдомо в зоне POSITIVE независимо от текущих значений конфига.
    """
    r   = MOZART_CONFIG["lth_nupl_rubicon"]
    e   = MOZART_CONFIG["lth_nupl_euphoria"]
    eps = MOZART_CONFIG["lth_nupl_rubicon_eps"]
    return (r + eps + e) / 2.0


def _nupl_rubicon_center() -> float:
    """
    Центр RUBICON-зоны: точно на rubicon (нуль нереализованной прибыли LTH).
    """
    return float(MOZART_CONFIG["lth_nupl_rubicon"])


def _nupl_bear_center() -> float:
    """
    Центр зоны BEAR: rubicon - 0.20.
    0.20 — шаг позиционирования, явно ниже нижней границы RUBICON-зоны.
    """
    return MOZART_CONFIG["lth_nupl_rubicon"] - 0.20


# ---------------------------------------------------------------------------
# TestClassifyLthNuplRegime
# ---------------------------------------------------------------------------

class TestClassifyLthNuplRegime:
    """
    Контракт classify_lth_nupl_regime(nupl: float) -> str:

    Зоны М-05 (посты 05.04.2026, 15.05.2026; пороги из MOZART_CONFIG):

      EUPHORIA : nupl >= euphoria (0.75)
                 Риск распределения LTH; высокая нереализованная прибыль.
                 ⚠️ Порог 0.75 взят из [LTH PAIN PROXY], не прямая цитата Mozart.

      POSITIVE : rubicon + eps < nupl < euphoria
                 LTH в нереализованной прибыли; рынок здоров.

      RUBICON  : rubicon - eps <= nupl <= rubicon + eps
                 Граница прибыль↔убыток (±eps = шум дневных данных).
                 Mozart: «Рубиконом является безубыток» (пост 05.04.2026,
                 аналогия с SOPR). Пересечение нуля — событие, не просто порог.

      BEAR     : nupl < rubicon - eps
                 LTH в убытке; повышенный риск вынужденных продаж.
    """

    def test_returns_string(self):
        # WHY: оркестратор вставляет метку в строковый блок вывода;
        # не-str вызовет TypeError при f-строке без явной ошибки.
        result = classify_lth_nupl_regime(_nupl_positive_center())
        assert isinstance(result, str)

    # ── Зоны — центры диапазонов ────────────────────────────────────────────

    def test_euphoria_zone(self):
        # WHY: NUPL выше euphoria = риск распределения LTH; оркестратор
        # должен поднять флаг. Ошибочный POSITIVE пропустит главный
        # медвежий риск поздней фазы бычьего рынка.
        assert classify_lth_nupl_regime(_nupl_euphoria_center()) == "EUPHORIA"

    def test_positive_zone(self):
        # WHY: NUPL в диапазоне (eps, euphoria) = LTH в прибыли, рынок здоров.
        # Ошибочный RUBICON даст ложный сигнал границы при нормальной бычьей фазе;
        # ошибочный EUPHORIA поднимет ложную тревогу распределения.
        assert classify_lth_nupl_regime(_nupl_positive_center()) == "POSITIVE"

    def test_rubicon_zone_at_zero(self):
        # WHY: NUPL == rubicon (0.0) = центр RUBICON-зоны; Mozart трактует
        # пересечение нуля как событие смены рынка (пост 05.04.2026).
        # Ошибочный POSITIVE или BEAR пропустит этот переходный сигнал.
        assert classify_lth_nupl_regime(_nupl_rubicon_center()) == "RUBICON"

    def test_bear_zone(self):
        # WHY: NUPL ниже -eps = LTH в убытке; повышенный риск вынужденных продаж.
        # Ошибочный RUBICON или POSITIVE скроет медвежью фазу от оркестратора.
        assert classify_lth_nupl_regime(_nupl_bear_center()) == "BEAR"

    # ── Граница euphoria — самое частое место тихих багов ──────────────────

    def test_boundary_euphoria_exact_is_euphoria(self):
        # WHY: nupl == euphoria → EUPHORIA (включительно сверху).
        # Ошибка > вместо >=: nupl == euphoria попал бы в POSITIVE —
        # пропущен сам порог риска распределения LTH; тихий баг на границе.
        nupl = float(MOZART_CONFIG["lth_nupl_euphoria"])
        assert classify_lth_nupl_regime(nupl) == "EUPHORIA"

    def test_boundary_just_below_euphoria_is_positive(self):
        # WHY: nupl == euphoria - _STEP → POSITIVE, не EUPHORIA.
        # Фиксирует что граница строгая снизу для EUPHORIA.
        # Ошибка >= вместо >: значение чуть ниже euphoria попало бы в EUPHORIA —
        # ложный сигнал распределения при здоровой бычьей фазе.
        nupl = MOZART_CONFIG["lth_nupl_euphoria"] - _STEP
        assert classify_lth_nupl_regime(nupl) == "POSITIVE"

    # ── Верхняя граница RUBICON-зоны ────────────────────────────────────────

    def test_rubicon_zone_upper_edge_is_rubicon(self):
        # WHY: nupl == rubicon + eps → RUBICON (не POSITIVE).
        # Верхний край RUBICON-зоны включительно; значение на границе eps
        # ещё остаётся в переходной зоне, не переходит в POSITIVE.
        # Ошибка >= вместо >: nupl == eps попал бы в POSITIVE —
        # потерян ежедневный RUBICON-сигнал из-за шума данных.
        eps  = MOZART_CONFIG["lth_nupl_rubicon_eps"]
        nupl = MOZART_CONFIG["lth_nupl_rubicon"] + eps
        assert classify_lth_nupl_regime(nupl) == "RUBICON"

    def test_boundary_just_above_upper_eps_is_positive(self):
        # WHY: nupl == rubicon + eps + _STEP → POSITIVE.
        # Первое значение выше RUBICON-зоны = чистая зона POSITIVE.
        # Фиксирует строгую границу: значения выше eps не захватываются RUBICON.
        eps  = MOZART_CONFIG["lth_nupl_rubicon_eps"]
        nupl = MOZART_CONFIG["lth_nupl_rubicon"] + eps + _STEP
        assert classify_lth_nupl_regime(nupl) == "POSITIVE"

    # ── Внутренние точки RUBICON-зоны ───────────────────────────────────────

    def test_rubicon_zone_slightly_positive(self):
        # WHY: nupl = rubicon + eps/2 → RUBICON (середина верхней полузоны).
        # Реальные LTH NUPL часто колеблются в [0, eps] из-за сглаживания API;
        # без eps-буфера такие значения давали бы POSITIVE вместо RUBICON.
        eps  = MOZART_CONFIG["lth_nupl_rubicon_eps"]
        nupl = MOZART_CONFIG["lth_nupl_rubicon"] + eps / 2.0
        assert classify_lth_nupl_regime(nupl) == "RUBICON"

    def test_rubicon_zone_slightly_negative(self):
        # WHY: nupl = rubicon - eps/2 → RUBICON (середина нижней полузоны).
        # Без eps-буфера значение чуть ниже нуля давало бы BEAR —
        # ложный медвежий сигнал при фактически нейтральном NUPL.
        eps  = MOZART_CONFIG["lth_nupl_rubicon_eps"]
        nupl = MOZART_CONFIG["lth_nupl_rubicon"] - eps / 2.0
        assert classify_lth_nupl_regime(nupl) == "RUBICON"

    # ── Нижняя граница RUBICON-зоны ─────────────────────────────────────────

    def test_rubicon_zone_lower_edge_is_rubicon(self):
        # WHY: nupl == rubicon - eps → RUBICON (не BEAR).
        # Нижний край RUBICON-зоны включительно; значение на границе -eps
        # ещё в переходной зоне, не переходит в BEAR.
        # Ошибка > вместо >=: nupl == rubicon - eps попал бы в BEAR —
        # потерян eps-буфер с нижней стороны, ложный медвежий сигнал.
        eps  = MOZART_CONFIG["lth_nupl_rubicon_eps"]
        nupl = MOZART_CONFIG["lth_nupl_rubicon"] - eps
        assert classify_lth_nupl_regime(nupl) == "RUBICON"

    def test_boundary_just_below_lower_eps_is_bear(self):
        # WHY: nupl == rubicon - eps - _STEP → BEAR.
        # Первое значение ниже RUBICON-зоны = переход в медвежью зону.
        # Фиксирует строгую нижнюю границу: значения ниже -eps не захватываются RUBICON.
        eps  = MOZART_CONFIG["lth_nupl_rubicon_eps"]
        nupl = MOZART_CONFIG["lth_nupl_rubicon"] - eps - _STEP
        assert classify_lth_nupl_regime(nupl) == "BEAR"

    # ── Корректность меток ───────────────────────────────────────────────────

    def test_only_valid_labels_returned(self):
        # WHY: опечатка в метке ('EUPHORIA ' с пробелом, 'Rubicon' с регистром) —
        # тихий баг: оркестратор не упадёт, но условная логика перестанет работать.
        valid = {"EUPHORIA", "POSITIVE", "RUBICON", "BEAR"}
        for nupl in [_nupl_euphoria_center(), _nupl_positive_center(),
                     _nupl_rubicon_center(), _nupl_bear_center()]:
            assert classify_lth_nupl_regime(nupl) in valid
