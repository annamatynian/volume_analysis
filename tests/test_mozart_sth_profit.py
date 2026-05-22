"""
tests/test_mozart_sth_profit.py
================================
TDD — П3: % STH в прибыли + скорость изменения (паттерн МБ-03,
PLAN_MOZART_PATTERNS.md ЧАСТЬ 2).

Два контракта:

  classify_sth_profit_zone(profit_pct: float) -> str
      Классифицирует текущий % STH в прибыли по зонам паттерна МБ-03.
      Шесть зон (пост 09.09.2025): BEAR / NEUTRAL / NEUTRAL_BROKEN /
      HEATED / EUPHORIA_APPROACH / EUPHORIA.
      Пороги строго из MOZART_CONFIG.

  build_sth_profit_signal(history: list[float]) -> dict
      Принимает хронологический список значений profit_pct (старые → новые).
      Возвращает:
        {
          "zone":               str,   # зона текущего значения
          "profit_pct":         float, # текущее значение (history[-1])
          "delta_7d":           float, # изменение за последние window_days
          "is_dropping_sharply":bool,  # delta < -drop_threshold_7d
        }
      Если истории недостаточно для расчёта дельты — delta_7d=0.0,
      is_dropping_sharply=False.

Правила:
  - Числа только через MOZART_CONFIG, не хардкодятся.
  - Тестовые значения вычисляются из порогов конфига (середина зоны).
  - Синтетические данные явно искусственные (не BTC-реалистичные).
  - WHY-комментарий к каждому assert.
  - Все 6 зон и все граничные значения покрыты отдельными тестами.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mozart_signals import (            # RED: функций ещё нет
    classify_sth_profit_zone,
    build_sth_profit_signal,
)
from mozart_config import MOZART_CONFIG


# ---------------------------------------------------------------------------
# Вспомогательные генераторы тестовых значений — из конфига, не хардкод
# ---------------------------------------------------------------------------

def _pct_bear() -> float:
    """Явно в зоне BEAR: середина диапазона [0, bear_max)."""
    return MOZART_CONFIG["sth_profit_bear_max"] / 2.0


def _pct_neutral() -> float:
    """Середина зоны NEUTRAL (синяя линия): среднее [neutral_min, neutral_max)."""
    return (MOZART_CONFIG["sth_profit_neutral_min"] +
            MOZART_CONFIG["sth_profit_neutral_max"]) / 2.0


def _pct_neutral_broken() -> float:
    """Середина зоны NEUTRAL_BROKEN (между синей и жёлтой линиями)."""
    return (MOZART_CONFIG["sth_profit_neutral_max"] +
            MOZART_CONFIG["sth_profit_heated_min"]) / 2.0


def _pct_heated() -> float:
    """Середина зоны HEATED (жёлтая линия)."""
    return (MOZART_CONFIG["sth_profit_heated_min"] +
            MOZART_CONFIG["sth_profit_heated_max"]) / 2.0


def _pct_euphoria_approach() -> float:
    """Середина зоны EUPHORIA_APPROACH (между жёлтой и красной линиями)."""
    return (MOZART_CONFIG["sth_profit_heated_max"] +
            MOZART_CONFIG["sth_profit_euphoria_min"]) / 2.0


def _pct_euphoria() -> float:
    """Середина зоны EUPHORIA (красная линия, Over-heated)."""
    return (MOZART_CONFIG["sth_profit_euphoria_min"] +
            MOZART_CONFIG["sth_profit_euphoria_max"]) / 2.0


def _history_flat(value: float, n: int = 10) -> list:
    """История без изменений — delta_7d будет ~0."""
    return [value] * n


def _history_sharp_drop(end: float, drop: float, window: int = None) -> list:
    """
    История с резким падением за последние window_days:
    достаточно длинная, последние window_days — падение на drop п.п.
    """
    w = window or MOZART_CONFIG["sth_profit_speed_window_days"]
    # Сначала стабильная база, потом резкое падение
    start = end + drop
    base = [start] * 5                          # стабильная часть
    recent = [start - drop * i / w for i in range(w + 1)]  # линейное падение
    return base + recent


def _history_slow_rise(end: float, rise: float, window: int = None) -> list:
    """История с медленным ростом — не должна давать is_dropping_sharply."""
    w = window or MOZART_CONFIG["sth_profit_speed_window_days"]
    start = end - rise
    return [start + rise * i / (w + 5) for i in range(w + 6)]


# ---------------------------------------------------------------------------
# TestClassifySthProfitZone
# ---------------------------------------------------------------------------

class TestClassifySthProfitZone:
    """
    Контракт classify_sth_profit_zone(profit_pct: float) -> str:

    Зоны МБ-03 (пост 09.09.2025, пороги из MOZART_CONFIG):
      BEAR             : < sth_profit_bear_max (51%)
                         Большинство STH в убытке; давление продаж слабое.
      NEUTRAL          : [neutral_min, neutral_max) (51–59%)
                         Синяя линия; сопротивление в медвежьем, поддержка в бычьем.
      NEUTRAL_BROKEN   : [neutral_max, heated_min) (59–69%)
                         Нейтральная пробита вверх; отскок продолжается.
      HEATED           : [heated_min, heated_max) (69–76%)
                         Жёлтая линия; редко в медвежьем рынке.
      EUPHORIA_APPROACH: [heated_max, euphoria_min) (76–85%)
                         Переход к эйфории; высокий аппетит к риску.
      EUPHORIA         : >= euphoria_min (85%+)
                         Красная линия (Over-heated); подтверждение новой бычки.
    """

    def test_returns_string(self):
        # WHY: метка вставляется в строковый блок оркестратора;
        # не-str вызовет TypeError при форматировании.
        result = classify_sth_profit_zone(_pct_neutral())
        assert isinstance(result, str)

    def test_bear_zone(self):
        # WHY: < 51% — большинство STH в убытке; ошибочная классификация
        # лишит оркестратор сигнала о медвежьей структуре.
        assert classify_sth_profit_zone(_pct_bear()) == "BEAR"

    def test_neutral_zone(self):
        # WHY: синяя линия (51–59%) — ключевой рубикон МБ-03.
        # Смешение с BEAR или NEUTRAL_BROKEN искажает интерпретацию сопротивления.
        assert classify_sth_profit_zone(_pct_neutral()) == "NEUTRAL"

    def test_neutral_broken_zone(self):
        # WHY: зона между синей и жёлтой линиями (59–69%) —
        # нейтральная пробита вверх, отскок продолжается.
        # Путаница с NEUTRAL занизит оценку силы отскока.
        assert classify_sth_profit_zone(_pct_neutral_broken()) == "NEUTRAL_BROKEN"

    def test_heated_zone(self):
        # WHY: жёлтая линия (69–76%) — редкий сигнал в медвежьем рынке.
        # Неверная классификация скроет признак перегрева отскока.
        assert classify_sth_profit_zone(_pct_heated()) == "HEATED"

    def test_euphoria_approach_zone(self):
        # WHY: зона перехода к эйфории (76–85%) — сигнал к осторожности.
        # Смешение с HEATED занижает риск, с EUPHORIA — преувеличивает.
        assert classify_sth_profit_zone(_pct_euphoria_approach()) == "EUPHORIA_APPROACH"

    def test_euphoria_zone(self):
        # WHY: красная линия (85%+) — Over-heated, подтверждение новой бычки
        # по паттерну МБ-03. Самый сильный бычий сигнал; ошибка здесь критична.
        assert classify_sth_profit_zone(_pct_euphoria()) == "EUPHORIA"

    # ── Граничные значения ──────────────────────────────────────────────────

    def test_boundary_bear_to_neutral(self):
        # WHY: profit_pct == neutral_min (51%) → NEUTRAL, не BEAR.
        # Фиксирует поведение на нижней границе синей линии.
        pct = float(MOZART_CONFIG["sth_profit_neutral_min"])
        assert classify_sth_profit_zone(pct) == "NEUTRAL"

    def test_boundary_neutral_to_neutral_broken(self):
        # WHY: profit_pct == neutral_max (59%) → NEUTRAL_BROKEN, не NEUTRAL.
        pct = float(MOZART_CONFIG["sth_profit_neutral_max"])
        assert classify_sth_profit_zone(pct) == "NEUTRAL_BROKEN"

    def test_boundary_neutral_broken_to_heated(self):
        # WHY: profit_pct == heated_min (69%) → HEATED, не NEUTRAL_BROKEN.
        pct = float(MOZART_CONFIG["sth_profit_heated_min"])
        assert classify_sth_profit_zone(pct) == "HEATED"

    def test_boundary_heated_to_euphoria_approach(self):
        # WHY: profit_pct == heated_max (76%) → EUPHORIA_APPROACH, не HEATED.
        pct = float(MOZART_CONFIG["sth_profit_heated_max"])
        assert classify_sth_profit_zone(pct) == "EUPHORIA_APPROACH"

    def test_boundary_euphoria_approach_to_euphoria(self):
        # WHY: profit_pct == euphoria_min (85%) → EUPHORIA, не EUPHORIA_APPROACH.
        pct = float(MOZART_CONFIG["sth_profit_euphoria_min"])
        assert classify_sth_profit_zone(pct) == "EUPHORIA"

    def test_only_valid_labels_returned(self):
        # WHY: опечатка в метке — тихий баг в оркестраторе.
        valid = {"BEAR", "NEUTRAL", "NEUTRAL_BROKEN", "HEATED",
                 "EUPHORIA_APPROACH", "EUPHORIA"}
        test_values = [_pct_bear(), _pct_neutral(), _pct_neutral_broken(),
                       _pct_heated(), _pct_euphoria_approach(), _pct_euphoria()]
        for pct in test_values:
            assert classify_sth_profit_zone(pct) in valid


# ---------------------------------------------------------------------------
# TestBuildSthProfitSignal
# ---------------------------------------------------------------------------

class TestBuildSthProfitSignal:
    """
    Контракт build_sth_profit_signal(history: list[float]) -> dict:

    history — хронологический список значений profit_pct (старые → новые).
    Возвращает словарь с четырьмя ключами:
      "zone"               : str  — зона текущего значения (history[-1])
      "profit_pct"         : float — текущее значение (history[-1])
      "delta_7d"           : float — изменение за последние window_days п.п.
      "is_dropping_sharply": bool  — delta_7d < -drop_threshold_7d

    Семантика delta_7d:
      delta_7d = history[-1] - history[-(window_days+1)]
      Отрицательное значение = STH profit снижается.
      is_dropping_sharply = True только при резком падении (>= drop_threshold_7d п.п.),
      что по паттерну МБ-03 Паттерн Б означает смену структуры.

    При недостаточной истории (len < window_days + 1):
      delta_7d = 0.0, is_dropping_sharply = False.
    """

    def test_returns_dict(self):
        # WHY: оркестратор обращается к полю по ключу;
        # не-dict вызовет TypeError без явной ошибки.
        result = build_sth_profit_signal(_history_flat(_pct_neutral()))
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        # WHY: оркестратор читает все четыре ключа; отсутствие любого — KeyError.
        result = build_sth_profit_signal(_history_flat(_pct_neutral()))
        for key in ("zone", "profit_pct", "delta_7d", "is_dropping_sharply"):
            assert key in result, f"Missing required key: '{key}'"

    def test_profit_pct_is_float(self):
        # WHY: оркестратор форматирует profit_pct как число;
        # не-float вызовет ошибку при f-строке с форматом :.1f.
        result = build_sth_profit_signal(_history_flat(_pct_neutral()))
        assert isinstance(result["profit_pct"], float)

    def test_delta_7d_is_float(self):
        # WHY: delta_7d участвует в числовых сравнениях в оркестраторе;
        # int вместо float может дать неожиданный результат при малых значениях.
        result = build_sth_profit_signal(_history_flat(_pct_neutral()))
        assert isinstance(result["delta_7d"], float)

    def test_is_dropping_sharply_is_bool(self):
        # WHY: оркестратор использует флаг в условном выражении if/else;
        # не-bool (например int 0/1) может дать неожиданный результат.
        result = build_sth_profit_signal(_history_flat(_pct_neutral()))
        assert isinstance(result["is_dropping_sharply"], bool)

    def test_profit_pct_equals_last_history_value(self):
        # WHY: profit_pct должен отражать текущее состояние рынка (history[-1]);
        # использование другого элемента истории даст устаревшее значение.
        history = _history_flat(42.0)
        history[-1] = 55.0      # явно меняем последнее значение
        result = build_sth_profit_signal(history)
        assert result["profit_pct"] == 55.0

    def test_zone_matches_current_value(self):
        # WHY: zone должна соответствовать именно history[-1], а не среднему.
        # Несоответствие zone и profit_pct — логическая ошибка в выводе.
        current = _pct_heated()
        result = build_sth_profit_signal(_history_flat(current))
        assert result["zone"] == classify_sth_profit_zone(current)

    def test_sharp_drop_sets_flag_true(self):
        # WHY: резкое падение сквозь синюю линию (паттерн МБ-03 Паттерн Б)
        # = смена структуры. Пропущенный флаг = потеря критического сигнала.
        # drop_threshold_7d + 2 гарантирует превышение порога.
        drop = MOZART_CONFIG["sth_profit_drop_threshold_7d"] + 2.0
        history = _history_sharp_drop(end=_pct_bear(), drop=drop)
        result = build_sth_profit_signal(history)
        assert result["is_dropping_sharply"] is True

    def test_flat_history_does_not_set_flag(self):
        # WHY: стабильное колебание вокруг синей линии — не смена структуры.
        # Ложный флаг создаст постоянный сигнал тревоги в спокойном рынке.
        result = build_sth_profit_signal(_history_flat(_pct_neutral()))
        assert result["is_dropping_sharply"] is False

    def test_rising_history_does_not_set_flag(self):
        # WHY: рост profit_pct = улучшение позиции STH;
        # is_dropping_sharply должен быть False при положительной дельте.
        history = _history_slow_rise(end=_pct_neutral(), rise=3.0)
        result = build_sth_profit_signal(history)
        assert result["is_dropping_sharply"] is False

    def test_insufficient_history_gives_zero_delta(self):
        # WHY: при первом запуске или коротком кэше история может быть короче окна.
        # Выброс исключения здесь остановит весь pipeline оркестратора.
        short_history = [_pct_neutral()] * 3    # < window_days + 1
        result = build_sth_profit_signal(short_history)
        assert result["delta_7d"] == 0.0
        assert result["is_dropping_sharply"] is False

    def test_delta_7d_sign_reflects_direction(self):
        # WHY: отрицательная дельта = снижение; оркестратор может отображать
        # дельту напрямую. Перепутанный знак инвертирует интерпретацию.
        w = MOZART_CONFIG["sth_profit_speed_window_days"]
        # История: сначала высокое значение, затем низкое
        high = _pct_neutral() + 10.0
        low  = _pct_neutral()
        history = [high] * (w + 2) + [low]
        result = build_sth_profit_signal(history)
        assert result["delta_7d"] < 0.0
