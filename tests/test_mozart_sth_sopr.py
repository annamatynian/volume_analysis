"""
tests/test_mozart_sth_sopr.py
==============================
TDD — ВЕТКА 4: М-02 STH SOPR рубикон безубытка
(паттерн М-02, PLAN_MOZART_PATTERNS.md ЧАСТЬ 1, пост 16.04.2026).

Один контракт:

  classify_sth_sopr_regime(sopr: float) -> str
      Классифицирует STH SOPR по трём зонам паттерна М-02.
      Зоны (приоритет сверху вниз):
        sopr >  rubicon + eps                        → 'BULL'
        rubicon - eps <= sopr <= rubicon + eps        → 'RUBICON'
        sopr <  rubicon - eps                        → 'BEAR'

      Граница rubicon (1.0) с ±eps (0.005):
        sopr == rubicon + eps → 'RUBICON' (не BULL; верхний край включительно)
        sopr == rubicon - eps → 'RUBICON' (не BEAR; нижний край включительно)

      Mozart (пост 16.04.2026):
        «нейтральная зона... сильная поддержка на бычьих,
         и сильное сопротивление на медвежьих рынках».

      Пороги rubicon и rubicon_eps строго из MOZART_CONFIG.

Отличие от classify_lth_nupl_regime:
  STH SOPR — 3 зоны (без эйфории), рубикон = 1.0 (безубыток стоимости, не NUPL).
  BULL вместо двух зон EUPHORIA+POSITIVE.

Отличие от classify_lth_sopr_regime:
  LTH SOPR — 4 зоны с историческими якорями 0.50/0.80, без eps-буфера.
  STH SOPR — 3 зоны, один рубикон 1.0, с eps-буфером шума.

Правила:
  - Числовые пороги только через MOZART_CONFIG, не хардкодятся в assertions.
  - Тестовые значения вычисляются из порогов (середина зоны, смещение от границы).
  - _STEP = 0.001: шаг «just outside» зоны, не рыночный порог.
  - WHY-комментарий к каждому assert: что сломается в production.
  - Все 3 зоны, обе границы RUBICON-зоны, внутренние точки — отдельные тесты.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# RED: функции ещё нет — ImportError подтверждает RED
from mozart_signals import classify_sth_sopr_regime
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

def _sopr_bull_center() -> float:
    """
    Центр зоны BULL: rubicon + eps + 0.10.
    0.10 — шаг позиционирования, явно выше верхней границы RUBICON-зоны,
    не API-значение.
    """
    rubicon = MOZART_CONFIG["sth_sopr_rubicon"]
    eps     = MOZART_CONFIG["sth_sopr_rubicon_eps"]
    return rubicon + eps + 0.10


def _sopr_rubicon_center() -> float:
    """
    Центр RUBICON-зоны: точно на rubicon (1.0 = безубыток STH).
    """
    return float(MOZART_CONFIG["sth_sopr_rubicon"])


def _sopr_bear_center() -> float:
    """
    Центр зоны BEAR: rubicon - eps - 0.10.
    0.10 — шаг позиционирования, явно ниже нижней границы RUBICON-зоны.
    """
    rubicon = MOZART_CONFIG["sth_sopr_rubicon"]
    eps     = MOZART_CONFIG["sth_sopr_rubicon_eps"]
    return rubicon - eps - 0.10


# ---------------------------------------------------------------------------
# TestClassifySthSoprRegime
# ---------------------------------------------------------------------------

class TestClassifySthSoprRegime:
    """
    Контракт classify_sth_sopr_regime(sopr: float) -> str:

    Зоны М-02 (пост 16.04.2026; пороги из MOZART_CONFIG):

      BULL    : sopr > rubicon + eps
                STH продают выше себестоимости; бычий фон.
                Mozart: рынок выше «нейтральной зоны» = здоровый бычий.

      RUBICON : rubicon - eps <= sopr <= rubicon + eps
                STH на безубытке; Mozart (пост 16.04.2026):
                «сильная поддержка на бычьих рынках,
                 сильное сопротивление на медвежьих рынках».

      BEAR    : sopr < rubicon - eps
                STH фиксируют убыток; капитуляция коротких держателей.
    """

    def test_returns_string(self):
        # WHY: оркестратор вставляет метку в строковый блок вывода;
        # не-str вызовет TypeError при f-строке без явной ошибки.
        result = classify_sth_sopr_regime(_sopr_bull_center())
        assert isinstance(result, str)

    # ── Зоны — центры диапазонов ────────────────────────────────────────────

    def test_bull_zone(self):
        # WHY: sopr выше RUBICON-зоны = STH продают в прибыль.
        # Ошибочный RUBICON скроет бычий сигнал; оркестратор не получит
        # подтверждение здорового бычьего рынка (пост 16.04.2026).
        assert classify_sth_sopr_regime(_sopr_bull_center()) == "BULL"

    def test_rubicon_zone_at_rubicon(self):
        # WHY: sopr == rubicon (1.0) = центр RUBICON-зоны; безубыток STH.
        # Mozart: «уровень 1.0 — граница, которая действует как двойной
        # агент» (пост 16.04.2026). Ошибочный BULL или BEAR пропустит
        # двойственный сигнал поддержки/сопротивления.
        assert classify_sth_sopr_regime(_sopr_rubicon_center()) == "RUBICON"

    def test_bear_zone(self):
        # WHY: sopr ниже RUBICON-зоны = STH фиксируют убыток, капитуляция.
        # Ошибочный RUBICON скроет медвежью фазу; ошибочный BULL — критическая
        # инверсия сигнала, оркестратор отобразит противоположное направление.
        assert classify_sth_sopr_regime(_sopr_bear_center()) == "BEAR"

    # ── Верхняя граница RUBICON-зоны — самое частое место тихих багов ───────

    def test_boundary_rubicon_upper_edge_is_rubicon(self):
        # WHY: sopr == rubicon + eps → RUBICON (не BULL).
        # Верхний край RUBICON-зоны включительно; значение ровно на границе eps
        # ещё остаётся в переходной зоне, не переходит в BULL.
        # Ошибка > вместо >= в условии (sopr > rubicon + eps → BULL):
        # при > граничное значение правильно остаётся в RUBICON.
        # Но ошибка >= (sopr >= rubicon + eps → BULL) уберёт верхний край из RUBICON —
        # потерян RUBICON-сигнал из-за шума данных на верхней границе.
        eps  = MOZART_CONFIG["sth_sopr_rubicon_eps"]
        sopr = MOZART_CONFIG["sth_sopr_rubicon"] + eps
        assert classify_sth_sopr_regime(sopr) == "RUBICON"

    def test_boundary_just_above_upper_eps_is_bull(self):
        # WHY: sopr == rubicon + eps + _STEP → BULL.
        # Первое значение строго выше RUBICON-зоны = переход в зону продаж в прибыль.
        # Фиксирует строгую верхнюю границу: значения выше rubicon+eps не захватываются
        # RUBICON. Без этого теста ошибка в знаке граничного условия остаётся незамеченной.
        eps  = MOZART_CONFIG["sth_sopr_rubicon_eps"]
        sopr = MOZART_CONFIG["sth_sopr_rubicon"] + eps + _STEP
        assert classify_sth_sopr_regime(sopr) == "BULL"

    # ── Внутренние точки RUBICON-зоны ───────────────────────────────────────

    def test_rubicon_zone_slightly_above_rubicon(self):
        # WHY: sopr = rubicon + eps/2 → RUBICON (середина верхней полузоны).
        # Реальные STH SOPR часто колеблются в [1.0, 1.0 + eps] из-за сглаживания API;
        # без eps-буфера такие значения давали бы BULL вместо RUBICON —
        # ложный бычий сигнал при фактически нейтральном STH SOPR.
        eps  = MOZART_CONFIG["sth_sopr_rubicon_eps"]
        sopr = MOZART_CONFIG["sth_sopr_rubicon"] + eps / 2.0
        assert classify_sth_sopr_regime(sopr) == "RUBICON"

    def test_rubicon_zone_slightly_below_rubicon(self):
        # WHY: sopr = rubicon - eps/2 → RUBICON (середина нижней полузоны).
        # Без eps-буфера значение чуть ниже 1.0 давало бы BEAR —
        # ложный медвежий сигнал при фактически нейтральном STH SOPR.
        # Mozart: дневные данные STH SOPR имеют шум ±0.5% от безубытка.
        eps  = MOZART_CONFIG["sth_sopr_rubicon_eps"]
        sopr = MOZART_CONFIG["sth_sopr_rubicon"] - eps / 2.0
        assert classify_sth_sopr_regime(sopr) == "RUBICON"

    # ── Нижняя граница RUBICON-зоны ─────────────────────────────────────────

    def test_boundary_rubicon_lower_edge_is_rubicon(self):
        # WHY: sopr == rubicon - eps → RUBICON (не BEAR).
        # Нижний край RUBICON-зоны включительно; значение ровно на -eps
        # ещё в переходной зоне, не переходит в BEAR.
        # Ошибка < вместо <= (sopr < rubicon - eps → BEAR, иначе RUBICON):
        # при >= нижний край включительно — правильно.
        # Если ошибочно >, то sopr == rubicon - eps → BEAR, потерян eps-буфер снизу,
        # ложный медвежий сигнал капитуляции при нейтральном STH SOPR.
        eps  = MOZART_CONFIG["sth_sopr_rubicon_eps"]
        sopr = MOZART_CONFIG["sth_sopr_rubicon"] - eps
        assert classify_sth_sopr_regime(sopr) == "RUBICON"

    def test_boundary_just_below_lower_eps_is_bear(self):
        # WHY: sopr == rubicon - eps - _STEP → BEAR.
        # Первое значение строго ниже RUBICON-зоны = переход в зону убытка STH.
        # Фиксирует строгую нижнюю границу: значения ниже rubicon-eps не захватываются
        # RUBICON. Без этого теста ошибка знака нижней границы остаётся незамеченной.
        eps  = MOZART_CONFIG["sth_sopr_rubicon_eps"]
        sopr = MOZART_CONFIG["sth_sopr_rubicon"] - eps - _STEP
        assert classify_sth_sopr_regime(sopr) == "BEAR"

    # ── Корректность меток ───────────────────────────────────────────────────

    def test_only_valid_labels_returned(self):
        # WHY: опечатка в метке ('BULL ' с пробелом, 'Rubicon' с регистром) —
        # тихий баг: оркестратор не упадёт, но условная логика перестанет работать.
        # Проверяем все три зоны, чтобы выловить опечатку в любой ветке return.
        valid = {"BULL", "RUBICON", "BEAR"}
        for sopr in [_sopr_bull_center(), _sopr_rubicon_center(), _sopr_bear_center()]:
            assert classify_sth_sopr_regime(sopr) in valid
