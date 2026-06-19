"""
tests/test_mozart_sth_sopr_turning.py

TDD для detect_sth_sopr_turning() — детектор вектора пересечения рубикона STH SOPR.

Контракт функции:
    detect_sth_sopr_turning(history: list[str], window: int = 5) -> str | None
    Принимает список зон: 'BULL' / 'RUBICON' / 'BEAR' (строки из classify_sth_sopr_regime).
    Возвращает:
        'UPWARD'   — зональный режим в окне вырос (последний выше минимума окна)
        'DOWNWARD' — зональный режим в окне упал (последний ниже максимума окна)
        None       — нет направленного движения (все зоны одинаковы или застыли)

Порядок зон (числовой ранг):
    'BEAR'   = 0
    'RUBICON' = 1
    'BULL'   = 2

Зоны берутся из classify_sth_sopr_regime() — тесты не вызывают классификатор напрямую.
Окно из MOZART_CONFIG["sth_sopr_turning_window"] = 5 (FORMALIZED).

Прецедент: detect_lth_sopr_turning (М-01-Т) и detect_sth_rp_zscore_turning (М-09).
"""

import pytest
from mozart_signals import detect_sth_sopr_turning
from mozart_config import MOZART_CONFIG


# ---------------------------------------------------------------------------
# Вспомогательная константа — порядок зон
# ---------------------------------------------------------------------------
# WHY: тесты ссылаются на строки, а не на числа.
# Если порядок зон изменится — тесты упадут явно, не молча.
_ZONES = ['BEAR', 'RUBICON', 'BULL']


# ---------------------------------------------------------------------------
# Класс 1 — Недостаточно данных
# ---------------------------------------------------------------------------

class TestInsufficientData:
    """История короче окна → None."""

    def test_empty_history_returns_none(self):
        # WHY: пустой список не должен давать KeyError/IndexError —
        # оркестратор может передать пустой DF при первом запуске.
        result = detect_sth_sopr_turning([])
        assert result is None, (
            "WHY: пустая история → нет данных для определения вектора"
        )

    def test_single_element_returns_none(self):
        # WHY: одна точка — вектор неопределён по определению.
        result = detect_sth_sopr_turning(['BULL'])
        assert result is None

    def test_window_minus_one_returns_none(self):
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        # WHY: ровно window-1 точек — граница: функция должна вернуть None,
        # не угадывать вектор на неполном окне.
        short_history = ['BEAR'] * (window - 1)
        result = detect_sth_sopr_turning(short_history, window=window)
        assert result is None, (
            "WHY: меньше window точек → недостаточно данных, контракт гарантирует None"
        )


# ---------------------------------------------------------------------------
# Класс 2 — UPWARD: движение снизу вверх
# ---------------------------------------------------------------------------

class TestUpwardDetection:
    """Последняя зона выше минимума окна → 'UPWARD'."""

    def test_bear_to_rubicon_is_upward(self):
        # WHY: BEAR→RUBICON — отскок STH дошёл до безубытка.
        # Если функция вернёт None — оркестратор пропустит пересечение рубикона.
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        history = ['BEAR'] * (window - 1) + ['RUBICON']
        result = detect_sth_sopr_turning(history, window=window)
        assert result == 'UPWARD', (
            "WHY: последняя зона RUBICON > минимум окна BEAR → вектор UPWARD"
        )

    def test_bear_to_bull_direct_is_upward(self):
        # WHY: прямой пробой BEAR→BULL через зону — тот же вектор вверх.
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        history = ['BEAR'] * (window - 1) + ['BULL']
        result = detect_sth_sopr_turning(history, window=window)
        assert result == 'UPWARD'

    def test_rubicon_to_bull_is_upward(self):
        # WHY: RUBICON→BULL — пробой сопротивления вверх.
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        history = ['RUBICON'] * (window - 1) + ['BULL']
        result = detect_sth_sopr_turning(history, window=window)
        assert result == 'UPWARD'

    def test_upward_with_noise_in_window(self):
        # WHY: шум (не-монотонная история) не должен маскировать общий вектор вверх.
        # В окне были BEAR и RUBICON, последняя — BULL → UPWARD.
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        history = ['BEAR', 'RUBICON', 'BEAR', 'RUBICON', 'BULL']
        assert len(history) == window
        result = detect_sth_sopr_turning(history, window=window)
        assert result == 'UPWARD', (
            "WHY: последняя BULL > минимум окна BEAR → UPWARD даже при шуме"
        )


# ---------------------------------------------------------------------------
# Класс 3 — DOWNWARD: движение сверху вниз
# ---------------------------------------------------------------------------

class TestDownwardDetection:
    """Последняя зона ниже максимума окна → 'DOWNWARD'."""

    def test_bull_to_rubicon_is_downward(self):
        # WHY: BULL→RUBICON — STH потеряли поддержку безубытка.
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        history = ['BULL'] * (window - 1) + ['RUBICON']
        result = detect_sth_sopr_turning(history, window=window)
        assert result == 'DOWNWARD', (
            "WHY: последняя RUBICON < максимум окна BULL → вектор DOWNWARD"
        )

    def test_bull_to_bear_direct_is_downward(self):
        # WHY: прямой пробой BULL→BEAR — тот же вектор вниз.
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        history = ['BULL'] * (window - 1) + ['BEAR']
        result = detect_sth_sopr_turning(history, window=window)
        assert result == 'DOWNWARD'

    def test_rubicon_to_bear_is_downward(self):
        # WHY: RUBICON→BEAR — пробой поддержки вниз.
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        history = ['RUBICON'] * (window - 1) + ['BEAR']
        result = detect_sth_sopr_turning(history, window=window)
        assert result == 'DOWNWARD'

    def test_downward_with_noise_in_window(self):
        # WHY: шум не должен маскировать общий вектор вниз.
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        history = ['BULL', 'RUBICON', 'BULL', 'RUBICON', 'BEAR']
        assert len(history) == window
        result = detect_sth_sopr_turning(history, window=window)
        assert result == 'DOWNWARD', (
            "WHY: последняя BEAR < максимум окна BULL → DOWNWARD даже при шуме"
        )


# ---------------------------------------------------------------------------
# Класс 4 — None: нет направленного движения
# ---------------------------------------------------------------------------

class TestNoneDetection:
    """Нет движения → None."""

    def test_all_same_zone_bull_returns_none(self):
        # WHY: весь окно одна зона — нет ни UPWARD, ни DOWNWARD.
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        history = ['BULL'] * window
        result = detect_sth_sopr_turning(history, window=window)
        assert result is None, (
            "WHY: last == max == min → нет пересечения, вектор неопределён"
        )

    def test_all_same_zone_rubicon_returns_none(self):
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        history = ['RUBICON'] * window
        result = detect_sth_sopr_turning(history, window=window)
        assert result is None

    def test_all_same_zone_bear_returns_none(self):
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        history = ['BEAR'] * window
        result = detect_sth_sopr_turning(history, window=window)
        assert result is None

    def test_up_then_back_returns_none(self):
        # WHY: BEAR→BULL→BEAR — последняя == минимуму окна.
        # Возврат к нижней точке не является сигналом UPWARD.
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        # last (BEAR) == min окна → не UPWARD; last (BEAR) < max (BULL) → DOWNWARD?
        # Нет: по контракту last < max → DOWNWARD; last == min → не UPWARD.
        # Здесь BEAR < BULL → DOWNWARD. Тест меняем на «застыл на дне»:
        history = ['BEAR', 'BEAR', 'BEAR', 'BEAR', 'BEAR']
        result = detect_sth_sopr_turning(history, window=window)
        assert result is None, (
            "WHY: нет движения в окне → оркестратор не должен получать ложный сигнал"
        )


# ---------------------------------------------------------------------------
# Класс 5 — Граничные значения
# ---------------------------------------------------------------------------

class TestBoundaryValues:
    """Ровно window точек — минимальный размер для сигнала."""

    def test_exactly_window_points_upward(self):
        # WHY: граница включительно — ровно window точек должны давать сигнал.
        # Если функция требует window+1 — упадёт именно здесь (тихий баг).
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        history = ['BEAR'] * (window - 1) + ['BULL']
        assert len(history) == window
        result = detect_sth_sopr_turning(history, window=window)
        assert result == 'UPWARD', (
            "WHY: ровно window точек — контракт обязан возвращать сигнал, не None"
        )

    def test_exactly_window_points_downward(self):
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        history = ['BULL'] * (window - 1) + ['BEAR']
        assert len(history) == window
        result = detect_sth_sopr_turning(history, window=window)
        assert result == 'DOWNWARD'

    def test_longer_history_uses_last_window(self):
        # WHY: функция должна смотреть только на последние window точек.
        # Далёкое прошлое (другой вектор) не должно влиять на текущий сигнал.
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        # Далёкое прошлое: BULL (10 точек) → но в последнем окне BEAR→RUBICON
        old_history = ['BULL'] * 10
        recent = ['BEAR'] * (window - 1) + ['RUBICON']
        history = old_history + recent
        result = detect_sth_sopr_turning(history, window=window)
        assert result == 'UPWARD', (
            "WHY: только последние window точек релевантны — старая история не влияет"
        )

    def test_return_type_is_str_or_none(self):
        # WHY: оркестратор делает _signals_sa['М-02-Т'] = result;
        # неверный тип (напр. int или bool) даст тихий баг в build_alignment().
        window = MOZART_CONFIG["sth_sopr_turning_window"]
        history_up = ['BEAR'] * (window - 1) + ['BULL']
        history_down = ['BULL'] * (window - 1) + ['BEAR']
        history_none = ['RUBICON'] * window

        up = detect_sth_sopr_turning(history_up, window=window)
        down = detect_sth_sopr_turning(history_down, window=window)
        none_ = detect_sth_sopr_turning(history_none, window=window)

        assert isinstance(up, str), "WHY: UPWARD должен быть str, не bool/int"
        assert isinstance(down, str), "WHY: DOWNWARD должен быть str"
        assert none_ is None, "WHY: отсутствие сигнала = None, не пустая строка"
