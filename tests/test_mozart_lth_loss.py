"""
tests/test_mozart_lth_loss.py
==============================
TDD — ВЕТКА 2: М-10 LTH Realized Loss — зонный классификатор + % от пика
(паттерн М-10, PLAN_MOZART_PATTERNS.md ЧАСТЬ 1, пост 02.04.2026).

Два контракта:

  classify_lth_realized_loss(loss_usd: float) -> str
      ⚠️  Принимает raw значение из API (отрицательное — убыток).
      Вызывает abs(loss_usd) перед сравнением с якорями.
      Зоны (строгое >=, приоритет сверху вниз по abs):
        abs < anchor_2018                              → 'BELOW_2018'
        anchor_2018  <= abs < anchor_2022_w1           → 'EARLY_2018_RANGE'
        anchor_2022_w1 <= abs < anchor_2022_ftx        → 'MID_2022_RANGE'
        anchor_2022_ftx <= abs < anchor_cycle_target   → 'PEAK_FTX_RANGE'
        abs >= anchor_cycle_target                     → 'EXTREME'
      Граница anchor_2018 (140M): abs == 140M → EARLY_2018_RANGE (включительно).
      Граница anchor_2022_w1 (300M): abs == 300M → MID_2022_RANGE (включительно).
      Граница anchor_2022_ftx (480M): abs == 480M → PEAK_FTX_RANGE (включительно).
      Граница anchor_cycle_target (500M): abs == 500M → EXTREME (включительно).
      Пороги строго из MOZART_CONFIG — в тестах не хардкодятся.

  lth_loss_pct_of_historical_peak(loss_usd: float) -> float
      % текущего убытка от исторического пика (anchor_2022_ftx = $480M).
      Формула: abs(loss_usd) / anchor_2022_ftx * 100.0
      Может превышать 100% (текущий цикл может установить новый рекорд).
      Принимает оба знака входного значения.
      Пороги строго из MOZART_CONFIG — в assertions не хардкодятся.

Правила:
  - Числа только через MOZART_CONFIG, не хардкодятся в assertions.
  - Тестовые значения вычисляются из порогов (середина зоны, смещение ±1).
  - WHY-комментарий к каждому assert: что сломается в production.
  - Все 5 зон и все 4 границы — отдельные тесты.
  - Оба знака входных данных проверяются (API = отрицательный, abs = положительный).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# RED: функций ещё нет — ImportError подтверждает RED
from mozart_signals import classify_lth_realized_loss, lth_loss_pct_of_historical_peak
from mozart_config import MOZART_CONFIG


# ---------------------------------------------------------------------------
# Генераторы тестовых значений для classify_lth_realized_loss
# — вычисляются из конфига, не хардкодятся
# — знак ОТРИЦАТЕЛЬНЫЙ (как API BGeometrics): abs() вызывает функция, не тест
# ---------------------------------------------------------------------------

def _loss_below_2018() -> float:
    """
    Середина зоны BELOW_2018: половина от anchor_2018, отрицательная (API sign).
    abs = anchor_2018 / 2.0 → заведомо ниже якоря 2018.
    """
    return -(MOZART_CONFIG["lth_loss_anchor_2018"] / 2.0)


def _loss_early_2018_range() -> float:
    """
    Середина зоны EARLY_2018_RANGE: среднее между anchor_2018 и anchor_2022_w1, отрицательная.
    abs строго между 140M и 300M.
    """
    a = MOZART_CONFIG["lth_loss_anchor_2018"]
    b = MOZART_CONFIG["lth_loss_anchor_2022_w1"]
    return -((a + b) / 2.0)


def _loss_mid_2022_range() -> float:
    """
    Середина зоны MID_2022_RANGE: среднее между anchor_2022_w1 и anchor_2022_ftx, отрицательная.
    abs строго между 300M и 480M.
    """
    a = MOZART_CONFIG["lth_loss_anchor_2022_w1"]
    b = MOZART_CONFIG["lth_loss_anchor_2022_ftx"]
    return -((a + b) / 2.0)


def _loss_peak_ftx_range() -> float:
    """
    Середина зоны PEAK_FTX_RANGE: среднее между anchor_2022_ftx и anchor_cycle_target, отрицательная.
    abs строго между 480M и 500M.
    """
    a = MOZART_CONFIG["lth_loss_anchor_2022_ftx"]
    b = MOZART_CONFIG["lth_loss_anchor_cycle_target"]
    return -((a + b) / 2.0)


def _loss_extreme() -> float:
    """
    Зона EXTREME: anchor_cycle_target + 50_000_000, отрицательная.
    50M — явно искусственный отступ выше максимального якоря.
    abs > 500M → EXTREME.
    """
    return -(MOZART_CONFIG["lth_loss_anchor_cycle_target"] + 50_000_000)


# ---------------------------------------------------------------------------
# TestClassifyLthRealizedLoss
# ---------------------------------------------------------------------------

class TestClassifyLthRealizedLoss:
    """
    Контракт classify_lth_realized_loss(loss_usd: float) -> str:

    Зоны М-10 (пост 02.04.2026, якоря из MOZART_CONFIG):
      BELOW_2018      : abs < 140M   — ниже уровня завершения медвежки 2018
      EARLY_2018_RANGE: 140M <= abs < 300M — в диапазоне завершения 2018
      MID_2022_RANGE  : 300M <= abs < 480M — диапазон начала капитуляции 2022
      PEAK_FTX_RANGE  : 480M <= abs < 500M — диапазон пика FTX-краша
      EXTREME         : abs >= 500M  — превышение ожидаемого пика текущего цикла

    ⚠️  API BGeometrics возвращает ОТРИЦАТЕЛЬНЫЕ значения убытков LTH.
    Функция обязана вызывать abs() перед сравнением с якорями.
    Тесты проверяют оба знака входного значения.
    """

    def test_returns_string(self):
        # WHY: оркестратор вставляет метку в строковый блок вывода;
        # не-str вызовет TypeError при f-строке без явной ошибки.
        result = classify_lth_realized_loss(_loss_below_2018())
        assert isinstance(result, str)

    # ── Зоны — центры диапазонов (отрицательный знак = API-знак) ───────────

    def test_below_2018_zone_negative_input(self):
        # WHY: малый убыток LTH (ниже уровня дна 2018) — слабый сигнал.
        # Ошибочный EARLY_2018_RANGE создаст ложную тревогу;
        # оркестратор переоценит серьёзность давления LTH.
        assert classify_lth_realized_loss(_loss_below_2018()) == "BELOW_2018"

    def test_early_2018_range_zone_negative_input(self):
        # WHY: убыток в диапазоне завершения медвежки 2018 ($140–300M) —
        # умеренный исторический уровень. Путаница с BELOW_2018 занизит оценку;
        # путаница с MID_2022_RANGE завысит тревогу.
        assert classify_lth_realized_loss(_loss_early_2018_range()) == "EARLY_2018_RANGE"

    def test_mid_2022_range_zone_negative_input(self):
        # WHY: диапазон начала капитуляции 2022 ($300–480M) — высокое давление.
        # Путаница с EARLY_2018_RANGE пропустит начало экстремальных продаж.
        assert classify_lth_realized_loss(_loss_mid_2022_range()) == "MID_2022_RANGE"

    def test_peak_ftx_range_zone_negative_input(self):
        # WHY: диапазон пика FTX-краша ($480–500M) — исторически экстремальный.
        # Путаница с MID_2022_RANGE пропустит самый тяжёлый исторический уровень;
        # путаница с EXTREME создаст преждевременный сигнал максимальной тревоги.
        assert classify_lth_realized_loss(_loss_peak_ftx_range()) == "PEAK_FTX_RANGE"

    def test_extreme_zone_negative_input(self):
        # WHY: превышение якоря текущего цикла (> $500M) — территория рекордных убытков.
        # Ошибочный PEAK_FTX_RANGE занизит сигнал именно тогда,
        # когда рынок входит в беспрецедентный режим.
        assert classify_lth_realized_loss(_loss_extreme()) == "EXTREME"

    # ── Проверка знака: API даёт отрицательные — abs() обязателен ───────────

    def test_positive_input_gives_same_zone_as_negative(self):
        # WHY: abs() обязан вызываться внутри функции.
        # Если abs() не вызван, положительное значение (например, после
        # коррекции данных) даст неверную зону; тихий баг без исключения.
        # NEXT_SESSION: +250M → EARLY_2018_RANGE (между 140M и 300M).
        loss_positive = float(MOZART_CONFIG["lth_loss_anchor_2018"] +
                              (MOZART_CONFIG["lth_loss_anchor_2022_w1"] -
                               MOZART_CONFIG["lth_loss_anchor_2018"]) / 2.0)
        loss_negative = -loss_positive
        assert classify_lth_realized_loss(loss_positive) == \
               classify_lth_realized_loss(loss_negative)

    def test_mandatory_positive_250m_is_early_2018_range(self):
        # WHY: NEXT_SESSION явно требует проверить +250M → EARLY_2018_RANGE.
        # Подтверждает что функция не различает знак — только abs().
        # Если функция вернёт BELOW_2018 — abs() не вызван, знак учтён неверно.
        loss_250m_positive = float(MOZART_CONFIG["lth_loss_anchor_2018"] +
                                   (MOZART_CONFIG["lth_loss_anchor_2022_w1"] -
                                    MOZART_CONFIG["lth_loss_anchor_2018"]) / 2.0)
        # Проверяем что это значение действительно в диапазоне [140M, 300M)
        assert MOZART_CONFIG["lth_loss_anchor_2018"] <= loss_250m_positive < \
               MOZART_CONFIG["lth_loss_anchor_2022_w1"]
        assert classify_lth_realized_loss(loss_250m_positive) == "EARLY_2018_RANGE"

    # ── Граничные значения — самое частое место тихих багов ─────────────────

    def test_boundary_anchor_2018_exact_is_early_2018_range(self):
        # WHY: abs == anchor_2018 (140M) → EARLY_2018_RANGE, не BELOW_2018.
        # Граница anchor_2018 включительно: Mozart: $140M = завершение медвежки 2018.
        # Ошибка </<=: abs == 140M попал бы в BELOW_2018 → пропуск исторического уровня.
        loss = -float(MOZART_CONFIG["lth_loss_anchor_2018"])
        assert classify_lth_realized_loss(loss) == "EARLY_2018_RANGE"

    def test_just_above_anchor_2018_is_early_2018_range(self):
        # WHY: NEXT_SESSION явно требует: loss = -140_000_001 → EARLY_2018_RANGE.
        # Тест фиксирует что граница < строгая снизу (не <=).
        # 1 — минимально возможный целочисленный шаг выше якоря.
        loss = -(MOZART_CONFIG["lth_loss_anchor_2018"] + 1)
        assert classify_lth_realized_loss(loss) == "EARLY_2018_RANGE"

    def test_just_below_anchor_2018_is_below_2018(self):
        # WHY: abs == anchor_2018 - 1 → BELOW_2018.
        # Фиксирует что граница < строгая; значение вплотную ниже якоря = BELOW.
        # При ошибке >=/<: попал бы в EARLY_2018_RANGE → ложный исторический сигнал.
        loss = -(MOZART_CONFIG["lth_loss_anchor_2018"] - 1)
        assert classify_lth_realized_loss(loss) == "BELOW_2018"

    def test_boundary_anchor_2022_w1_exact_is_mid_2022_range(self):
        # WHY: NEXT_SESSION явно требует: loss = -300_000_000 → MID_2022_RANGE.
        # abs == anchor_2022_w1 (300M) → MID_2022_RANGE (включительно).
        # Ошибка </<= : abs == 300M попал бы в EARLY_2018_RANGE → занижение уровня.
        loss = -float(MOZART_CONFIG["lth_loss_anchor_2022_w1"])
        assert classify_lth_realized_loss(loss) == "MID_2022_RANGE"

    def test_just_below_anchor_2022_w1_is_early_2018_range(self):
        # WHY: abs == 300M - 1 → EARLY_2018_RANGE.
        # Фиксирует строгую границу; вплотную ниже w1 = всё ещё ранний диапазон.
        loss = -(MOZART_CONFIG["lth_loss_anchor_2022_w1"] - 1)
        assert classify_lth_realized_loss(loss) == "EARLY_2018_RANGE"

    def test_boundary_anchor_2022_ftx_exact_is_peak_ftx_range(self):
        # WHY: abs == anchor_2022_ftx (480M) → PEAK_FTX_RANGE (включительно).
        # Это исторический максимум: не должен попадать в MID_2022_RANGE.
        # Ошибка </<= : abs == 480M попал бы в MID → потеря самого важного уровня.
        loss = -float(MOZART_CONFIG["lth_loss_anchor_2022_ftx"])
        assert classify_lth_realized_loss(loss) == "PEAK_FTX_RANGE"

    def test_boundary_anchor_cycle_target_exact_is_extreme(self):
        # WHY: abs == anchor_cycle_target (500M) → EXTREME (включительно).
        # Превышение ожидаемого пика цикла — сигнал беспрецедентных убытков.
        # Ошибка </<= : abs == 500M попал бы в PEAK_FTX_RANGE → потеря максимального сигнала.
        loss = -float(MOZART_CONFIG["lth_loss_anchor_cycle_target"])
        assert classify_lth_realized_loss(loss) == "EXTREME"

    def test_just_below_anchor_cycle_target_is_peak_ftx_range(self):
        # WHY: abs == 500M - 1 → PEAK_FTX_RANGE, не EXTREME.
        # Фиксирует строгую верхнюю границу PEAK_FTX_RANGE;
        # ошибка >=: попал бы в EXTREME преждевременно.
        loss = -(MOZART_CONFIG["lth_loss_anchor_cycle_target"] - 1)
        assert classify_lth_realized_loss(loss) == "PEAK_FTX_RANGE"

    def test_only_valid_labels_returned(self):
        # WHY: опечатка в метке ('EXTREME ' с пробелом, 'extreme' в нижнем регистре) —
        # тихий баг: оркестратор не упадёт, но условная логика перестанет работать.
        valid = {"BELOW_2018", "EARLY_2018_RANGE", "MID_2022_RANGE",
                 "PEAK_FTX_RANGE", "EXTREME"}
        for loss in [_loss_below_2018(), _loss_early_2018_range(),
                     _loss_mid_2022_range(), _loss_peak_ftx_range(),
                     _loss_extreme()]:
            assert classify_lth_realized_loss(loss) in valid


# ---------------------------------------------------------------------------
# TestLthLossPctOfHistoricalPeak
# ---------------------------------------------------------------------------

class TestLthLossPctOfHistoricalPeak:
    """
    Контракт lth_loss_pct_of_historical_peak(loss_usd: float) -> float:

    Процент текущего убытка LTH от исторического пика ($480M, крах FTX).
    Формула: abs(loss_usd) / anchor_2022_ftx * 100.0
    Может превышать 100% если текущий цикл установит новый рекорд.
    Принимает оба знака входного значения.
    Пороги строго из MOZART_CONFIG.

    NEXT_SESSION явные требования:
      pct_of_peak(-480_000_000) → ровно 100.0%
      pct_of_peak(-960_000_000) → 200.0%
    """

    def test_returns_float(self):
        # WHY: оркестратор форматирует результат через f-строку (f"{pct:.1f}%");
        # не-float даст тихий баг при арифметическом сравнении с порогом.
        result = lth_loss_pct_of_historical_peak(_loss_early_2018_range())
        assert isinstance(result, float)

    def test_historical_peak_gives_100_pct(self):
        # WHY: NEXT_SESSION явно требует: pct_of_peak(-480_000_000) → 100.0%.
        # anchor_2022_ftx — знаменатель формулы; при ошибке деления на другой якорь
        # результат будет неверным, интерпретация исторического пика исказится.
        peak = float(MOZART_CONFIG["lth_loss_anchor_2022_ftx"])
        result = lth_loss_pct_of_historical_peak(-peak)
        assert result == pytest.approx(100.0, abs=1e-6)

    def test_double_historical_peak_gives_200_pct(self):
        # WHY: NEXT_SESSION явно требует: pct_of_peak(-960_000_000) → 200.0%.
        # Проверяет линейность и что функция не ограничивает результат в 100%.
        # cap(100) был бы ошибкой: текущий цикл может установить рекорд.
        peak = float(MOZART_CONFIG["lth_loss_anchor_2022_ftx"])
        result = lth_loss_pct_of_historical_peak(-(peak * 2.0))
        assert result == pytest.approx(200.0, abs=1e-6)

    def test_positive_and_negative_input_give_same_result(self):
        # WHY: abs() обязан вызываться внутри функции.
        # API даёт отрицательные значения; если abs() не вызван,
        # отрицательный loss даст отрицательный % — тихий баг форматирования.
        loss_val = float(MOZART_CONFIG["lth_loss_anchor_2022_w1"])
        pct_neg = lth_loss_pct_of_historical_peak(-loss_val)
        pct_pos = lth_loss_pct_of_historical_peak(loss_val)
        assert pct_neg == pytest.approx(pct_pos, abs=1e-9)

    def test_pct_is_positive(self):
        # WHY: отрицательный % убытка невозможен физически;
        # оркестратор печатает «{pct:.1f}%» — отрицательное значение
        # даст «-X.X%», что введёт пользователя в заблуждение.
        result = lth_loss_pct_of_historical_peak(_loss_mid_2022_range())
        assert result > 0.0

    def test_below_peak_gives_less_than_100(self):
        # WHY: убыток ниже исторического пика должен давать < 100%.
        # Проверяет направление: функция не инвертирует формулу.
        below_peak = float(MOZART_CONFIG["lth_loss_anchor_2022_w1"])
        result = lth_loss_pct_of_historical_peak(-below_peak)
        assert result < 100.0

    def test_above_peak_gives_more_than_100(self):
        # WHY: убыток выше исторического пика должен давать > 100%.
        # Подтверждает что функция не зажимает результат в cap(100).
        above_peak = float(MOZART_CONFIG["lth_loss_anchor_cycle_target"] + 100_000_000)
        result = lth_loss_pct_of_historical_peak(-above_peak)
        assert result > 100.0

    def test_zero_loss_gives_zero_pct(self):
        # WHY: нулевой убыток → 0.0%; деление на ноль не происходит
        # (делим на anchor_2022_ftx, не на loss_usd).
        # Граничный случай на старте цикла или при отсутствии данных.
        result = lth_loss_pct_of_historical_peak(0.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_uses_ftx_anchor_as_denominator(self):
        # WHY: знаменатель = anchor_2022_ftx, не anchor_cycle_target.
        # Если перепутать якоря ($500M вместо $480M) — результат будет неверным.
        # Тест проверяет это косвенно: loss == anchor_2022_ftx → ровно 100%.
        # Если знаменатель = anchor_cycle_target (500M) → результат = 96%, не 100%.
        peak = float(MOZART_CONFIG["lth_loss_anchor_2022_ftx"])
        result = lth_loss_pct_of_historical_peak(-peak)
        assert result == pytest.approx(100.0, abs=1e-6)