"""tests/test_mozart_hodl_waves.py
==================================
TDD — ВЕТКА 9: М-12 | HODL Waves направление когорт (delta).

Контракт:

  classify_hodl_wave_regime(
      age_1m_3m_current: float,
      age_1m_3m_prev: float,
      age_3m_6m_current: float,
      age_3m_6m_prev: float,
  ) -> str

Логика (источник: пост 13.05.2026, паттерн М-12):

  Монеты стареют (накопление):
    age_1m_3m_current < age_1m_3m_prev  (когорта 1–3м уменьшается —
                                          монеты переходят в 3–6м)
    И
    age_3m_6m_current > age_3m_6m_prev  (когорта 3–6м растёт —
                                          монеты пришли из 1–3м)
    → 'AGING'

  Монеты молодеют (распределение):
    age_1m_3m_current > age_1m_3m_prev  (когорта 1–3м растёт —
                                          монеты пришли из свежих транзакций)
    И
    age_3m_6m_current < age_3m_6m_prev  (когорта 3–6м уменьшается)
    → 'REJUVENATING'

  Иначе (нет направленного сигнала) → 'MIXED'

Соглашение о нуле (граничные тесты):
  Дельта == 0 не удовлетворяет строгому условию «падает» или «растёт».
  Любой нулевой дельта → MIXED:
    — оба нуля               → MIXED
    — 1m3m падает, 3m6m flat → MIXED  (не AGING — нет роста 3m6m)
    — 1m3m flat, 3m6m растёт → MIXED  (не AGING — нет падения 1m3m)

Правила:
  - Числовых порогов нет → MOZART_CONFIG не импортируется.
  - Тестовые значения выбираются нейтральными (не API-реалистичными числами).
  - WHY-комментарий к каждому assert: что сломается в production.
  - Все режимы + граничные случаи нуля — отдельные тесты.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# RED: функции ещё нет — ImportError подтверждает RED
from mozart_signals import classify_hodl_wave_regime


# ---------------------------------------------------------------------------
# Синтетические тестовые значения — НЕ API-реалистичные числа.
#
# Интерпретация пары (current, prev):
#   _PREV  = 10.0  — нейтральное предыдущее значение (база)
#   _UP    = 11.0  — current > prev: когорта выросла (delta > 0)
#   _DOWN  = 9.0   — current < prev: когорта уменьшилась (delta < 0)
#   _FLAT  = 10.0  — current == prev: нет изменения (delta == 0)
#
# Для AGING:        1m3m=(DOWN, PREV)  + 3m6m=(UP, PREV)
# Для REJUVENATING: 1m3m=(UP, PREV)   + 3m6m=(DOWN, PREV)
# Для MIXED:        любое другое сочетание
# ---------------------------------------------------------------------------

_PREV = 10.0   # предыдущее значение когорты (база)
_UP   = 11.0   # current > prev → когорта растёт
_DOWN = 9.0    # current < prev → когорта падает
_FLAT = 10.0   # current == prev → нет изменения (delta == 0)


# ===========================================================================
# TestClassifyHodlWaveRegime — М-12 | HODL Waves
# ===========================================================================

class TestClassifyHodlWaveRegime:
    """
    Контракт classify_hodl_wave_regime(...) -> str.

    Паттерн М-12 (пост 13.05.2026):
      Mozart описывает внутрикогортный сдвиг качественно — только знак
      изменения, числовых порогов нет. Строгое неравенство (< / >) для
      обоих условий; нулевая дельта → MIXED.
    """

    # -- Тип возвращаемого значения ------------------------------------------

    def test_returns_string(self):
        # WHY: оркестратор вставляет метку в строковый блок вывода;
        # не-str вызовет TypeError при f-строке без явного исключения.
        result = classify_hodl_wave_regime(_DOWN, _PREV, _UP, _PREV)
        assert isinstance(result, str)

    # -- Три основных режима -------------------------------------------------

    def test_aging_1m3m_falls_and_3m6m_rises(self):
        # WHY: age_1m_3m_current < prev И age_3m_6m_current > prev
        # → монеты переходят из 1–3м в 3–6м = накопление = AGING.
        # Ошибка: перепутать направления дельт — оркестратор выдаст AGING
        # при сигнале распределения (инвертированная интерпретация Mozart).
        assert classify_hodl_wave_regime(
            age_1m_3m_current=_DOWN, age_1m_3m_prev=_PREV,
            age_3m_6m_current=_UP,   age_3m_6m_prev=_PREV,
        ) == "AGING"

    def test_rejuvenating_1m3m_rises_and_3m6m_falls(self):
        # WHY: age_1m_3m_current > prev И age_3m_6m_current < prev
        # → монеты перемещаются из 3–6м в свежую 1–3м когорту = распределение = REJUVENATING.
        # Ошибка: вернуть MIXED вместо REJUVENATING — оркестратор потеряет
        # сигнал распределения в период продаж LTH/STH.
        assert classify_hodl_wave_regime(
            age_1m_3m_current=_UP,   age_1m_3m_prev=_PREV,
            age_3m_6m_current=_DOWN, age_3m_6m_prev=_PREV,
        ) == "REJUVENATING"

    def test_mixed_both_cohorts_rising(self):
        # WHY: оба сегмента растут — нет направленного перетока между когортами.
        # Контракт: не AGING (3m6m растёт, но 1m3m тоже растёт, не падает).
        # Ошибка: вернуть AGING при обоих растущих → ложный сигнал накопления.
        assert classify_hodl_wave_regime(
            age_1m_3m_current=_UP, age_1m_3m_prev=_PREV,
            age_3m_6m_current=_UP, age_3m_6m_prev=_PREV,
        ) == "MIXED"

    def test_mixed_both_cohorts_falling(self):
        # WHY: оба сегмента падают — нет направленного перетока между когортами.
        # Контракт: не REJUVENATING (1m3m падает, но 3m6m тоже падает, не растёт).
        # Ошибка: вернуть REJUVENATING при обоих падающих → ложный сигнал распределения.
        assert classify_hodl_wave_regime(
            age_1m_3m_current=_DOWN, age_1m_3m_prev=_PREV,
            age_3m_6m_current=_DOWN, age_3m_6m_prev=_PREV,
        ) == "MIXED"

    # -- Граничные случаи: нулевая дельта ------------------------------------

    def test_boundary_both_zero_delta_is_mixed(self):
        # WHY: дельта == 0 для обеих когорт → нет направленного сигнала.
        # Контракт: MIXED, не AGING и не REJUVENATING.
        # Ошибка: treat flat как «не падает» → пропустить нулевое изменение
        # через условие 'AGING' и вернуть ложный режим накопления.
        # (пост 13.05.2026: Mozart требует именно знак изменения, не «≤ prev»).
        assert classify_hodl_wave_regime(
            age_1m_3m_current=_FLAT, age_1m_3m_prev=_PREV,
            age_3m_6m_current=_FLAT, age_3m_6m_prev=_PREV,
        ) == "MIXED"

    def test_boundary_1m3m_falls_3m6m_flat_is_mixed(self):
        # WHY: age_1m_3m падает (< prev), но age_3m_6m стоит (== prev).
        # Для AGING необходимо СТРОГОЕ условие 3m6m > prev.
        # Нулевая дельта 3m6m = монеты никуда не перешли → MIXED.
        # Ошибка: использовать «>=» вместо «>» для 3m6m — AGING выдастся ложно.
        assert classify_hodl_wave_regime(
            age_1m_3m_current=_DOWN, age_1m_3m_prev=_PREV,
            age_3m_6m_current=_FLAT, age_3m_6m_prev=_PREV,
        ) == "MIXED"

    def test_boundary_1m3m_flat_3m6m_rises_is_mixed(self):
        # WHY: age_3m_6m растёт (> prev), но age_1m_3m стоит (== prev).
        # Для AGING необходимо СТРОГОЕ условие 1m3m < prev.
        # Нулевая дельта 1m3m = источник пополнения 3m6m неизвестен → MIXED.
        # Ошибка: использовать «<=» вместо «<» для 1m3m — AGING выдастся ложно.
        assert classify_hodl_wave_regime(
            age_1m_3m_current=_FLAT, age_1m_3m_prev=_PREV,
            age_3m_6m_current=_UP,   age_3m_6m_prev=_PREV,
        ) == "MIXED"

    # -- Корректность меток --------------------------------------------------

    def test_only_valid_labels_returned(self):
        # WHY: опечатка в метке ('Aging', 'MIXED ' с пробелом, 'REJUVENATION' вместо
        # 'REJUVENATING') — тихий баг: оркестратор не упадёт, но условная логика сломается.
        # Проверяем все режимы + граничные случаи — ловим опечатку в любой ветке return.
        valid = {"AGING", "REJUVENATING", "MIXED"}
        test_cases = [
            # (1m3m_curr, 1m3m_prev, 3m6m_curr, 3m6m_prev)
            (_DOWN, _PREV, _UP,   _PREV),   # AGING
            (_UP,   _PREV, _DOWN, _PREV),   # REJUVENATING
            (_UP,   _PREV, _UP,   _PREV),   # MIXED: оба растут
            (_DOWN, _PREV, _DOWN, _PREV),   # MIXED: оба падают
            (_FLAT, _PREV, _FLAT, _PREV),   # MIXED: оба нуля
            (_DOWN, _PREV, _FLAT, _PREV),   # MIXED: 1m3m падает, 3m6m flat
            (_FLAT, _PREV, _UP,   _PREV),   # MIXED: 1m3m flat, 3m6m растёт
        ]
        for c1, p1, c3, p3 in test_cases:
            result = classify_hodl_wave_regime(c1, p1, c3, p3)
            assert result in valid, (
                f"classify_hodl_wave_regime({c1}, {p1}, {c3}, {p3}) "
                f"вернул '{result}' — не из допустимого множества {valid}"
            )
