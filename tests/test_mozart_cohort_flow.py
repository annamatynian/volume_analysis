"""tests/test_mozart_cohort_flow.py
==================================
TDD — ВЕТКА 8: М-07 + М-08 | Cohort Flow (совместный переток LTH<->STH).

Контракт:

  classify_cohort_flow(lth_net_pos: float, sth_net_pos: float) -> str

      4 квадранта по знаку (источник: пост 14.01.2026):

        lth >= 0 и sth <  0  → 'ACCUMULATION'
                                STH продают монеты → LTH; бычий фон.
                                Нормальная фаза накопления зрелого цикла.

        lth <  0 и sth >= 0  → 'DISTRIBUTION'
                                LTH продают монеты → STH; медвежий фон.
                                Классическое распределение вершины.

        lth >= 0 и sth >= 0  → 'BOTH_BUYING'
                                Обе когорты накапливают одновременно.
                                Редко; возможно после крупной коррекции.

        lth <  0 и sth <  0  → 'BOTH_SELLING'
                                Обе когорты продают одновременно.
                                Стресс-режим; возможна капитуляция.

Соглашение о нуле:
  Знак нуля = положительный: lth == 0.0 или sth == 0.0 считается >= 0.
  Обоснование (пост 14.01.2026): нейтральная позиция когорты не является
  давлением продаж — Mozart анализирует только ОТРИЦАТЕЛЬНОЕ чистое изменение
  как сигнал распределения/стресса.

Правила:
  - Числовых порогов нет → MOZART_CONFIG не импортируется.
  - Тестовые значения выбираются нейтральными (не API-реалистичными числами).
  - WHY-комментарий к каждому assert: что сломается в production.
  - Все 4 квадранта + 3 граничных случая нуля — отдельные тесты.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# RED: функции ещё нет — ImportError подтверждает RED
from mozart_signals import classify_cohort_flow


# ---------------------------------------------------------------------------
# Синтетические тестовые значения — НЕ API-реалистичные числа.
# Используем ±1.0 как «явно положительное / явно отрицательное» значение.
# ---------------------------------------------------------------------------

_POS = 1.0    # явно положительный net_pos (любая когорта накапливает)
_NEG = -1.0   # явно отрицательный net_pos (любая когорта продаёт)
_ZERO = 0.0   # граничный случай: нейтральная позиция


# ===========================================================================
# TestClassifyCohortFlow — М-07 + М-08 | Cohort Flow
# ===========================================================================

class TestClassifyCohortFlow:
    """
    Контракт classify_cohort_flow(lth_net_pos: float, sth_net_pos: float) -> str:

    Паттерн М-07 + М-08 (пост 14.01.2026):
      Mozart: «LTH Net Pos + STH Net Pos должны анализироваться вместе —
      разнонаправленность подтверждает переток монет между когортами».

    4 квадранта и 3 граничных теста для знака нуля.
    """

    # -- Тип возвращаемого значения ------------------------------------------

    def test_returns_string(self):
        # WHY: оркестратор вставляет метку в строковый блок вывода;
        # не-str вызовет TypeError при f-строке без явной ошибки в runtime.
        result = classify_cohort_flow(_POS, _NEG)
        assert isinstance(result, str)

    # -- Четыре квадранта ----------------------------------------------------

    def test_accumulation_lth_positive_sth_negative(self):
        # WHY: lth > 0 и sth < 0 → STH продают монеты → LTH.
        # Оркестратор должен зафиксировать ACCUMULATION как бычий фон.
        # Ошибка: перепутать квадранты (DISTRIBUTION вместо ACCUMULATION) —
        # оркестратор выдаст инвертированный сигнал на правильных данных.
        assert classify_cohort_flow(_POS, _NEG) == "ACCUMULATION"

    def test_distribution_lth_negative_sth_positive(self):
        # WHY: lth < 0 и sth > 0 → LTH продают монеты → STH.
        # Классический сигнал распределения вершины цикла.
        # Ошибка: не заметить DISTRIBUTION → пропущен медвежий сигнал в отчёте.
        assert classify_cohort_flow(_NEG, _POS) == "DISTRIBUTION"

    def test_both_buying_lth_positive_sth_positive(self):
        # WHY: lth > 0 и sth > 0 → оба накапливают.
        # Редкий режим: если смешать с ACCUMULATION, сигнал потеряет
        # информацию о характере движения (одностороннее vs двустороннее).
        assert classify_cohort_flow(_POS, _POS) == "BOTH_BUYING"

    def test_both_selling_lth_negative_sth_negative(self):
        # WHY: lth < 0 и sth < 0 → оба продают.
        # Стресс-режим: если не детектировать — оркестратор не выдаст
        # предупреждение о синхронной капитуляции обеих когорт.
        assert classify_cohort_flow(_NEG, _NEG) == "BOTH_SELLING"

    # -- Граничные случаи: знак нуля -----------------------------------------

    def test_boundary_lth_zero_sth_negative_is_accumulation(self):
        # WHY: lth == 0.0 считается >= 0 (нейтральная позиция LTH = не продаёт).
        # Контракт: lth == 0 и sth < 0 → ACCUMULATION (не BOTH_SELLING).
        # Ошибка: trear zero как отрицательное → lth=0 + sth<0 → BOTH_SELLING,
        # что ложно сигнализирует стресс при нейтральной позиции LTH.
        # (пост 14.01.2026: Mozart анализирует только отрицательное изменение
        # как давление продаж).
        assert classify_cohort_flow(_ZERO, _NEG) == "ACCUMULATION"

    def test_boundary_lth_positive_sth_zero_is_both_buying(self):
        # WHY: sth == 0.0 считается >= 0 (нейтральная позиция STH = не продаёт).
        # Контракт: lth > 0 и sth == 0 → BOTH_BUYING (не ACCUMULATION).
        # Ошибка: treat sth=0 как отрицательное → lth>0 + sth=0 → ACCUMULATION,
        # что ложно показывает переток STH→LTH при отсутствии движения STH.
        assert classify_cohort_flow(_POS, _ZERO) == "BOTH_BUYING"

    def test_boundary_both_zero_is_both_buying(self):
        # WHY: оба нуля = обе когорты нейтральны = >= 0 для обеих.
        # Контракт: lth == 0 и sth == 0 → BOTH_BUYING (не BOTH_SELLING, не ACCUMULATION).
        # Нейтральность обеих когорт — не стресс и не направленный переток.
        # Ошибка любого другого результата: ложный сигнал при отсутствии движения.
        assert classify_cohort_flow(_ZERO, _ZERO) == "BOTH_BUYING"

    # -- Корректность меток --------------------------------------------------

    def test_only_valid_labels_returned(self):
        # WHY: опечатка в метке ('Accumulation', 'BOTH_SELLING ' с пробелом)
        # — тихий баг: оркестратор не упадёт, но условная логика сломается.
        # Проверяем все 4 квадранта + 3 граничных случая — ловим опечатку в любой ветке return.
        valid = {"ACCUMULATION", "DISTRIBUTION", "BOTH_BUYING", "BOTH_SELLING"}
        test_cases = [
            (_POS, _NEG),   # ACCUMULATION
            (_NEG, _POS),   # DISTRIBUTION
            (_POS, _POS),   # BOTH_BUYING
            (_NEG, _NEG),   # BOTH_SELLING
            (_ZERO, _NEG),  # ACCUMULATION (граница lth)
            (_POS, _ZERO),  # BOTH_BUYING  (граница sth)
            (_ZERO, _ZERO), # BOTH_BUYING  (оба нуля)
        ]
        for lth, sth in test_cases:
            result = classify_cohort_flow(lth, sth)
            assert result in valid, (
                f"classify_cohort_flow({lth}, {sth}) вернул '{result}' — "
                f"не из допустимого множества {valid}"
            )
