"""
tests/test_mozart_lth_sopr.py
==============================
TDD — ВЕТКА 1: М-01 LTH SOPR рубикон + светофор фаз + детектор разворота
(паттерн М-01, PLAN_MOZART_PATTERNS.md ЧАСТЬ 1, PLAN_MOZART_TDD_LEVEL2.md).

Два контракта:

  classify_lth_sopr_regime(sopr: float) -> str
      Классифицирует текущий LTH SOPR по четырём зонам паттерна М-01.
      Зоны (строгое >=, приоритет сверху вниз):
        sopr >= rubicon    → 'BULL'
        sopr >= early_bear → 'EARLY_BEAR'
        sopr >= deep_bear  → 'MID_BEAR'
        sopr <  deep_bear  → 'CAPITULATION'
      Граница rubicon == 1.0: sopr == 1.0 → 'BULL' (рубикон не пробит).
      Граница deep_bear == 0.50: sopr == 0.50 → 'MID_BEAR' (не CAPITULATION).
      Пороги строго из MOZART_CONFIG — в тестах не хардкодятся.

  detect_lth_sopr_turning(history: list[float], window: int = 5) -> bool
      Паттерн В (пост 05.04.2026): SOPR перестаёт падать и начинает расти.
      True если в последних window значениях min достигнут ДО последней
      точки И последняя точка > min (началось восстановление).
      Недостаточно данных (len(history) < window) → False.

Правила:
  - Числа только через MOZART_CONFIG, не хардкодятся в assertions.
  - Тестовые значения вычисляются из порогов (середина зоны, доля).
  - Синтетические данные явно искусственные (не BTC-реалистичные SOPR).
  - WHY-комментарий к каждому assert: что сломается в production.
  - Все 4 зоны и все 3 границы — отдельные тесты.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# RED: функций ещё нет — ImportError подтверждает RED
from mozart_signals import classify_lth_sopr_regime, detect_lth_sopr_turning
from mozart_config import MOZART_CONFIG


# ---------------------------------------------------------------------------
# Генераторы тестовых значений для classify_lth_sopr_regime
# — вычисляются из конфига, не хардкодятся
# ---------------------------------------------------------------------------

def _sopr_bull() -> float:
    """Середина зоны BULL: rubicon + фиксированный отступ выше."""
    return MOZART_CONFIG["lth_sopr_rubicon"] + 0.20


def _sopr_early_bear() -> float:
    """Середина зоны EARLY_BEAR: среднее между rubicon и early_bear."""
    return (MOZART_CONFIG["lth_sopr_rubicon"] +
            MOZART_CONFIG["lth_sopr_early_bear"]) / 2.0


def _sopr_mid_bear() -> float:
    """Середина зоны MID_BEAR: среднее между early_bear и deep_bear."""
    return (MOZART_CONFIG["lth_sopr_early_bear"] +
            MOZART_CONFIG["lth_sopr_deep_bear"]) / 2.0


def _sopr_capitulation() -> float:
    """Середина зоны CAPITULATION: половина от deep_bear."""
    return MOZART_CONFIG["lth_sopr_deep_bear"] / 2.0


# ---------------------------------------------------------------------------
# Генераторы истории для detect_lth_sopr_turning
# — явно искусственные числа (арифметические последовательности)
# ---------------------------------------------------------------------------

def _history_v_shape(window: int = 5) -> list:
    """
    V-shape в последних window значениях: убывает затем растёт.
    prefix — два значения за пределами окна (проверяем что игнорируются).
    Последние 5: 1.0 → 0.8 → 0.6 → 0.7 → 0.9 (min=0.6 в позиции 2, last=0.9 > 0.6).
    """
    prefix = [1.5, 1.4]
    valley = [1.0, 0.8, 0.6, 0.7, 0.9]   # ровно window=5 значений
    return prefix + valley


def _history_declining(n: int = 7) -> list:
    """
    Монотонно убывающий ряд: SOPR ещё не развернулся.
    1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4 — явно искусственные шаги по 0.1.
    """
    return [1.0 - 0.1 * i for i in range(n)]


def _history_flat(n: int = 7) -> list:
    """
    Плоская история: нет смены направления, last == min.
    0.7 — нейтральное искусственное значение, не из конфига.
    """
    return [0.7] * n


def _history_bottom_flat(n: int = 7) -> list:
    """
    Убывает, затем плоское дно: last == min → паттерн В не выполнен.
    [1.0, 0.9, 0.8, 0.7, 0.7, 0.7, 0.7] — последние 5: [0.8, 0.7, 0.7, 0.7, 0.7].
    """
    return [1.0, 0.9, 0.8, 0.7, 0.7, 0.7, 0.7]


def _history_old_v_new_decline(window: int = 5) -> list:
    """
    V-shape ЗА пределами окна, убывание ВНУТРИ окна.
    Проверяет что старые данные не влияют на результат.
    old_data: [1.0, 0.6, 1.0] — V-shape вне окна.
    recent:   [1.0, 0.9, 0.8, 0.7, 0.6] — ровно window=5, монотонно убывает.
    """
    old_data = [1.0, 0.6, 1.0]
    recent   = [1.0, 0.9, 0.8, 0.7, 0.6]   # min=0.6 в last position → False
    return old_data + recent


# ---------------------------------------------------------------------------
# TestClassifyLthSoprRegime
# ---------------------------------------------------------------------------

class TestClassifyLthSoprRegime:
    """
    Контракт classify_lth_sopr_regime(sopr: float) -> str:

    Зоны М-01 (пост 05.04.2026, пороги из MOZART_CONFIG):
      BULL        : sopr >= rubicon (1.0)
                    LTH продают в прибыль; рубикон не пробит.
      EARLY_BEAR  : early_bear (0.80) <= sopr < rubicon
                    0–20% убыток; ранняя медвежка после пробоя рубикона.
      MID_BEAR    : deep_bear (0.50) <= sopr < early_bear
                    20–50% убыток; разгар медвежки.
      CAPITULATION: sopr < deep_bear
                    >50% убыток; кульминация (скорр. цикл: 40–50%).

    Граница rubicon: sopr == 1.0 → BULL (включительно сверху; рубикон не пробит).
    Граница deep_bear: sopr == 0.50 → MID_BEAR (не CAPITULATION).
    """

    def test_returns_string(self):
        # WHY: оркестратор вставляет метку в строковый блок вывода;
        # не-str вызовет TypeError при f-строке без явной ошибки.
        result = classify_lth_sopr_regime(_sopr_bull())
        assert isinstance(result, str)

    def test_bull_zone(self):
        # WHY: выше рубикона (1.0) LTH продают в прибыль — самый сильный
        # бычий сигнал. Ошибочный EARLY_BEAR занизит оценку рынка оркестратором.
        assert classify_lth_sopr_regime(_sopr_bull()) == "BULL"

    def test_early_bear_zone(self):
        # WHY: зона 0–20% убытка LTH = ранняя медвежка после пробоя рубикона.
        # Смешение с BULL пропустит начало медвежьей фазы;
        # смешение с MID_BEAR преувеличит серьёзность убытков.
        assert classify_lth_sopr_regime(_sopr_early_bear()) == "EARLY_BEAR"

    def test_mid_bear_zone(self):
        # WHY: зона 20–50% убытка LTH = разгар медвежки.
        # Смешение с EARLY_BEAR занизит тяжесть ситуации;
        # смешение с CAPITULATION создаст ложный сигнал дна.
        assert classify_lth_sopr_regime(_sopr_mid_bear()) == "MID_BEAR"

    def test_capitulation_zone(self):
        # WHY: ниже deep_bear = кульминация убытков LTH.
        # Паттерн В (detect_lth_sopr_turning) актуален именно в этой зоне;
        # ошибочный MID_BEAR не даст оркестратору включить детектор разворота.
        assert classify_lth_sopr_regime(_sopr_capitulation()) == "CAPITULATION"

    # ── Граничные значения — самое частое место тихих багов ────────────────

    def test_boundary_rubicon_is_bull(self):
        # WHY: sopr == 1.0 → BULL, не EARLY_BEAR.
        # Рубикон включительно сверху: Mozart: «безубыток = граница перехода»
        # (пост 05.04.2026), поэтому точное значение 1.0 ещё не означает пробой.
        # Ошибка <=/<: sopr == 1.0 попал бы в EARLY_BEAR → пропуск сигнала.
        sopr = float(MOZART_CONFIG["lth_sopr_rubicon"])
        assert classify_lth_sopr_regime(sopr) == "BULL"

    def test_boundary_early_bear_is_early_bear(self):
        # WHY: sopr == 0.80 → EARLY_BEAR, не BULL.
        # Граница early_bear включительно снизу: значение на пороге 0.80
        # уже перешло в раннюю медвежку, не остаётся в BULL.
        # Ошибка >/>=: sopr == 0.80 попал бы в BULL → пропуск начала медвежки.
        sopr = float(MOZART_CONFIG["lth_sopr_early_bear"])
        assert classify_lth_sopr_regime(sopr) == "EARLY_BEAR"

    def test_boundary_deep_bear_is_mid_bear(self):
        # WHY: sopr == 0.50 → MID_BEAR, не CAPITULATION.
        # Граница deep_bear включительно снизу: Mozart определяет кульминацию
        # как ниже 0.50 (пост 05.04.2026: «диапазон от 40 до 50%»);
        # само значение 0.50 = ещё MID_BEAR.
        # Ошибка >/>=: sopr == 0.50 попал бы в CAPITULATION → ложный сигнал дна.
        sopr = float(MOZART_CONFIG["lth_sopr_deep_bear"])
        assert classify_lth_sopr_regime(sopr) == "MID_BEAR"

    def test_just_below_deep_bear_is_capitulation(self):
        # WHY: первое значение ниже deep_bear → CAPITULATION.
        # Фиксирует что граница < строгая, не <=;
        # при ошибке <= соседние значения давали бы MID_BEAR вместо CAPITULATION.
        sopr = MOZART_CONFIG["lth_sopr_deep_bear"] - 0.001
        assert classify_lth_sopr_regime(sopr) == "CAPITULATION"

    def test_only_valid_labels_returned(self):
        # WHY: опечатка в метке ('BULL ' с пробелом, 'Capitulation' с регистром) —
        # тихий баг: оркестратор не упадёт, но условная логика перестанет работать.
        valid = {"BULL", "EARLY_BEAR", "MID_BEAR", "CAPITULATION"}
        for sopr in [_sopr_bull(), _sopr_early_bear(),
                     _sopr_mid_bear(), _sopr_capitulation()]:
            assert classify_lth_sopr_regime(sopr) in valid


# ---------------------------------------------------------------------------
# TestDetectLthSoprTurning
# ---------------------------------------------------------------------------

class TestDetectLthSoprTurning:
    """
    Контракт detect_lth_sopr_turning(history: list[float], window: int = 5) -> bool:

    Паттерн В (пост 05.04.2026):
      «Дно формируется не тогда когда LTH продают в убыток, а тогда
       когда у них заканчиваются монеты которые они готовы продавать в убыток.
       Это момент когда SOPR перестаёт падать и начинает расти.»

    True  если: min(history[-window:]) достигнут ДО последней позиции
                И history[-1] > min(history[-window:]).
    False если: SOPR продолжает падать (min на последней позиции),
                или застыл на дне (last == min),
                или недостаточно данных (len < window).

    WHY только последние window значений: функция детектирует недавний
    разворот, не исторический минимум. Старые V-shape не релевантны.
    """

    def test_returns_bool(self):
        # WHY: оркестратор использует результат в условном выражении if/else;
        # не-bool (например int 0/1) может дать неожиданный результат
        # при отрицании (not 0 == True, но not False == True тоже).
        result = detect_lth_sopr_turning(_history_v_shape())
        assert isinstance(result, bool)

    def test_v_shape_in_window_returns_true(self):
        # WHY: классический паттерн В — SOPR упал и начал расти внутри окна.
        # Ложный False здесь = потерянный сигнал разворота, ради которого
        # функция и существует (пост 05.04.2026).
        assert detect_lth_sopr_turning(_history_v_shape()) is True

    def test_minimal_recovery_returns_true(self):
        # WHY: сигнал должен срабатывать как только SOPR начал расти,
        # даже если рост минимален (первое закрытие выше дна).
        # Слишком строгое условие (e.g., нужно N баров роста) = запоздалый сигнал.
        # [1.0, 0.9, 0.8, 0.7, 0.6, 0.7]: last 5 = [0.9, 0.8, 0.7, 0.6, 0.7]
        # min=0.6 на позиции 3, last=0.7 > 0.6 → разворот зафиксирован.
        history = [1.0, 0.9, 0.8, 0.7, 0.6, 0.7]
        assert detect_lth_sopr_turning(history) is True

    def test_monotonically_declining_returns_false(self):
        # WHY: SOPR ещё не развернулся — сигнал давать нельзя.
        # Ложный True = преждевременный сигнал дна при продолжающейся капитуляции.
        assert detect_lth_sopr_turning(_history_declining()) is False

    def test_flat_history_returns_false(self):
        # WHY: плоский SOPR (last == min) = нет смены направления.
        # Паттерн В требует явного начала роста, боковик не считается.
        assert detect_lth_sopr_turning(_history_flat()) is False

    def test_bottom_flat_returns_false(self):
        # WHY: SOPR упал и застрял на дне (last == min последних window).
        # Разворот не подтверждён — нет роста. Ложный True = сигнал без подтверждения.
        assert detect_lth_sopr_turning(_history_bottom_flat()) is False

    def test_insufficient_history_returns_false(self):
        # WHY: при коротком кэше или первом запуске история может быть короче window.
        # Выброс исключения остановит pipeline оркестратора; False — безопасный fallback.
        short = [0.9, 0.8, 0.9]   # 3 значения < default window=5
        assert detect_lth_sopr_turning(short) is False

    def test_custom_window_parameter_respected(self):
        # WHY: параметр window должен реально использоваться, не игнорироваться.
        # Если window проигнорирован → всегда 5; тогда [0.9, 0.6, 0.9] (len=3 < 5)
        # → всегда False, даже при window=3 где V-shape очевиден.
        history = [0.9, 0.6, 0.9]   # V-shape: min=0.6 в середине, last=0.9 > min
        # С window=3: last 3 = all, min=0.6, last=0.9 > 0.6 → True
        assert detect_lth_sopr_turning(history, window=3) is True
        # С дефолтным window=5: len=3 < 5 → недостаточно данных → False
        assert detect_lth_sopr_turning(history) is False

    def test_old_v_shape_ignored_recent_decline_returns_false(self):
        # WHY: функция смотрит только на последние window значений.
        # Если бы использовалась вся история — старая V-shape давала бы
        # False Positive при продолжающемся снижении. Это критичная ошибка
        # в боковой медвежке с временными отскоками.
        # История: V-shape в старых данных, убывание в последних window.
        assert detect_lth_sopr_turning(_history_old_v_new_decline()) is False
