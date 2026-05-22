# tests/test_mozart_sth_realized_price.py
# ВЕТКА 10 | М-09 | STH Realized Price — паттерн В (Z-score turning)
# Источник паттерна: PLAN_MOZART_TDD_LEVEL2.md ВЕТКА 10 (пост 05.04.2026).
#
# Функция: detect_sth_rp_zscore_turning(zscore_history: list[float], window: int = 5) -> bool
# Аналог:  detect_lth_sopr_turning (ВЕТКА 1) — контракт идентичен.
#
# Контракт:
#   True  ⟺  len(zscore_history) >= window
#             AND zscore_history[-1] > min(zscore_history[-window:])
#   False ⟺  иначе (недостаточно данных, монотонное убывание, плоское дно)
#
# TDD: тест-файл создан ДО реализации (RED должен быть подтверждён pytest).

import pytest

from mozart_signals import detect_sth_rp_zscore_turning


# ---------------------------------------------------------------------------
# Основные сценарии — Паттерн В: разворот Z-score вверх
# ---------------------------------------------------------------------------

class TestDetectSthRpZscoreTurningBasic:
    """Основные режимы: есть разворот / нет разворота."""

    def test_falling_then_rising_returns_true(self):
        """
        Z-score убывал, затем начал расти → паттерн В подтверждён.

        WHY: оркестратор использует True для вывода сигнала «Z-score разворачивается».
        Если функция вернёт False при реальном развороте — сигнал будет пропущен.
        """
        # Последние 5: [3, 1, -1, -2, 0] — min=-2 (позиция 3), last=0 > -2
        history = [10.0, 9.0, 8.0, 3.0, 1.0, -1.0, -2.0, 0.0]
        result = detect_sth_rp_zscore_turning(history, window=5)
        # WHY: last(0.0) > min_of_window(-2.0) — разворот состоялся
        assert result is True

    def test_monotonically_falling_returns_false(self):
        """
        Z-score монотонно убывает → разворота нет → False.

        WHY: при монотонном убывании min == last; оркестратор не должен
        выдавать сигнал разворота на продолжающемся нисходящем тренде.
        """
        # [5, 4, 3, 2, 1] — min=1 == last=1
        history = [10.0, 9.0, 8.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        result = detect_sth_rp_zscore_turning(history, window=5)
        # WHY: last(1.0) == min_of_window(1.0) — разворота нет
        assert result is False

    def test_flat_bottom_returns_false(self):
        """
        Z-score упал и застыл (последняя точка совпадает с минимумом) → False.

        WHY: плоское дно — не начало восстановления.
        Оркестратор ждёт первого движения вверх, а не просто стабилизации.
        Тихий баг: без этой проверки функция могла бы вернуть True при last==min,
        если argmin ищется неверно.
        """
        # [3, 2, 1, 1, 1] — min=1, last=1; last > min? → False
        history = [7.0, 6.0, 3.0, 2.0, 1.0, 1.0, 1.0]
        result = detect_sth_rp_zscore_turning(history, window=5)
        # WHY: last(1.0) == min_of_window(1.0) — стоячее дно, не разворот
        assert result is False


# ---------------------------------------------------------------------------
# Граничные тесты — длина истории
# ---------------------------------------------------------------------------

class TestDetectSthRpZscoreTurningLengthBoundary:
    """Граничные случаи по длине zscore_history относительно window."""

    def test_history_shorter_than_window_returns_false(self):
        """
        len(history) < window → недостаточно данных → False.

        WHY: оркестратор не должен делать выводы при неполном окне наблюдений.
        Возврат True при len < window — тихий баг: сравниваются не все нужные точки.
        """
        history = [-1.0, 0.5]   # len=2, window=5
        result = detect_sth_rp_zscore_turning(history, window=5)
        # WHY: 2 < 5 — окно неполное, сигнал невозможен
        assert result is False

    def test_history_exactly_window_length_falling_then_rising(self):
        """
        len(history) == window — ровно одно окно данных с разворотом → True.

        WHY: граница «достаточности» должна быть включительной (>=, не >).
        Если реализация использует >, при len==window функция вернёт False,
        хотя данных достаточно для полного расчёта.
        """
        # len=5, window=5: [2.0, 1.0, -1.0, -2.0, 0.5]
        # min=-2.0 (pos=3), last=0.5 > -2.0 → True
        history = [2.0, 1.0, -1.0, -2.0, 0.5]
        result = detect_sth_rp_zscore_turning(history, window=5)
        # WHY: len(5)==window(5) — данных ровно достаточно; разворот должен быть обнаружен
        assert result is True

    def test_history_exactly_window_length_monotone(self):
        """
        len(history) == window, монотонное убывание → False.

        WHY: полное окно без разворота не должно давать True.
        Дополняет test_history_exactly_window_length_falling_then_rising —
        проверяет, что граничная длина сама по себе не меняет логику.
        """
        # [5.0, 4.0, 3.0, 2.0, 1.0] — min=1.0==last=1.0
        history = [5.0, 4.0, 3.0, 2.0, 1.0]
        result = detect_sth_rp_zscore_turning(history, window=5)
        # WHY: min==last при len==window — разворота нет, как и при длинной истории
        assert result is False

    def test_history_length_one_less_than_window_returns_false(self):
        """
        len(history) == window - 1 → False (граница «меньше» включительно).

        WHY: один элемент до порога — недостаточно; важно, что граница len < window
        правильно исключает window-1.
        """
        # window=5, history len=4
        history = [3.0, 1.0, -1.0, 0.5]
        result = detect_sth_rp_zscore_turning(history, window=5)
        # WHY: 4 < 5 — окно неполное, даже при наличии разворота в данных
        assert result is False


# ---------------------------------------------------------------------------
# Граничные тесты — window=1
# ---------------------------------------------------------------------------

class TestDetectSthRpZscoreTurningWindowOne:
    """window=1: технически валидный вызов, но всегда False."""

    def test_window_one_always_false(self):
        """
        window=1 → w=[last], min(w)==last → last > last → False.

        WHY: при window=1 нет «истории» для сравнения; функция смотрит
        только на последнюю точку против себя же.
        Если реализация вернёт True при window=1 — значит она не проверяет
        last > min(window), а использует другую (неверную) логику.
        """
        history = [5.0, 4.0, 3.0, 2.0, 1.0, 2.0]  # есть разворот в данных
        result = detect_sth_rp_zscore_turning(history, window=1)
        # WHY: w=[2.0], min=2.0==last=2.0 → разворот относительно window невозможен
        assert result is False

    def test_window_one_single_element_history(self):
        """
        Одна точка в истории, window=1 → len==window, но разворота нет.

        WHY: крайний вырожденный случай — один элемент и окно 1.
        Проверяет, что функция не падает и возвращает False, не True.
        """
        history = [0.7]
        result = detect_sth_rp_zscore_turning(history, window=1)
        # WHY: w=[0.7], min=0.7==last=0.7 → False, не ошибка
        assert result is False


# ---------------------------------------------------------------------------
# Тест типа возвращаемого значения
# ---------------------------------------------------------------------------

class TestDetectSthRpZscoreTurningReturnType:
    """Тип результата: bool, не int, не float."""

    def test_return_type_is_bool_on_true(self):
        """
        Функция возвращает именно bool True, не truthy int/float.

        WHY: оркестратор использует `if signal:` — int(1) пройдёт,
        но при логировании, сериализации или сравнении `is True`
        вернёт False для не-bool. Явный bool — контракт.
        """
        history = [3.0, 1.0, -2.0, -3.0, -1.0]  # min=-3, last=-1 > -3
        result = detect_sth_rp_zscore_turning(history, window=5)
        # WHY: is True, не просто truthy — тип строго bool
        assert result is True
        assert type(result) is bool  # WHY: не int, не np.bool_

    def test_return_type_is_bool_on_false(self):
        """
        Функция возвращает именно bool False, не falsy 0 или None.

        WHY: `result is False` отличает False от None/0 при диагностике.
        """
        history = [5.0, 4.0, 3.0, 2.0, 1.0]
        result = detect_sth_rp_zscore_turning(history, window=5)
        # WHY: is False — тип строго bool, не None и не 0
        assert result is False
        assert type(result) is bool  # WHY: не int, не np.bool_


# ---------------------------------------------------------------------------
# Дополнительный сценарий: min в середине окна (не на последней и не первой позиции)
# ---------------------------------------------------------------------------

class TestDetectSthRpZscoreTurningMinInMiddle:
    """Случай, когда минимум находится строго внутри окна."""

    def test_min_in_middle_of_window_returns_true(self):
        """
        Минимум достигнут в середине окна, последняя точка выше → True.

        WHY: защита от реализации, которая ищет min только на крайних позициях.
        Реальный Z-score может достичь дна в любой точке окна, не обязательно
        у ближайшего к последней позиции.
        """
        # window=5: [0.0, -3.0, -1.0, 1.0, 2.0]
        # min=-3.0 (pos=1), last=2.0 > -3.0 → True
        history = [5.0, 0.0, -3.0, -1.0, 1.0, 2.0]
        result = detect_sth_rp_zscore_turning(history, window=5)
        # WHY: min в середине окна — последняя точка выше, разворот подтверждён
        assert result is True
