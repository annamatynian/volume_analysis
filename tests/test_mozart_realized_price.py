"""
tests/test_mozart_realized_price.py
=====================================
TDD — МБ-01: Realized Price — «Синяя линия» дна цикла.

Контракт:

  classify_realized_price_regime(price: float, realized_price: float) -> str
      ±20% буфер вокруг Realized Price (пост 25.02.2026):

        price > realized_price * (1 + buf)                        → 'ABOVE'
        realized_price * (1 - buf) <= price
            <= realized_price * (1 + buf)                         → 'AT'
        price < realized_price * (1 - buf)                        → 'BELOW'

      buf = MOZART_CONFIG["realized_price_buffer_pct"] = 0.20

      Граница AT включительно с обеих сторон:
        price == realized_price * (1 + buf) → 'AT'  (не 'ABOVE')
        price == realized_price * (1 - buf) → 'AT'  (не 'BELOW')
      ABOVE начинается строго выше верхней границы.
      BELOW начинается строго ниже нижней границы.

Источник паттерна: пост 25.02.2026 (МБ-01, PLAN_MOZART_PATTERNS.md ЧАСТЬ 5)
  Mozart: «Синяя линия Realized Price ~$54,500» — «самый сильный уровень
  из трёхлинейной модели», исторически совпадает с дном цикла.
  ±20% буфер формализован нами для защиты от ложных пробоев.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# RED: classify_realized_price_regime ещё не существует → ImportError подтверждает RED
from mozart_signals import classify_realized_price_regime
from mozart_config import MOZART_CONFIG


# ---------------------------------------------------------------------------
# Вспомогательные константы тестов
# ---------------------------------------------------------------------------

# _RP — нейтральный плейсхолдер. Круглое число, явно не реалистичная цена BTC.
# WHY не API-реалистичное: тест не должен зависеть от текущей рыночной цены;
# изменение Realized Price не ломает тест, только конфиг.
_RP = 10.0  # базовый realized_price для всех вычислений теста

# _STEP — шаг для граничных тестов «just outside».
# WHY 0.001: меньше buf=0.20 на 3 порядка → однозначно переходит границу
# без захода в другую зону. По аналогии с _STEP в test_mozart_mvrv.py.
_STEP = 0.001


# ---------------------------------------------------------------------------
# Генераторы тестовых значений (price при фиксированном _RP)
# ---------------------------------------------------------------------------

def _above_center() -> float:
    """Центр ABOVE-зоны: RP * (1 + buf) + 2.0 (явно выше верхней границы)."""
    buf = float(MOZART_CONFIG["realized_price_buffer_pct"])
    return _RP * (1.0 + buf) + 2.0


def _at_center() -> float:
    """Центр AT-зоны: RP (точно на Realized Price, середина буфера)."""
    return _RP


def _below_center() -> float:
    """Центр BELOW-зоны: RP * (1 - buf) - 2.0 (явно ниже нижней границы)."""
    buf = float(MOZART_CONFIG["realized_price_buffer_pct"])
    return _RP * (1.0 - buf) - 2.0


# ---------------------------------------------------------------------------
# TestClassifyRealizedPriceRegime
# ---------------------------------------------------------------------------

class TestClassifyRealizedPriceRegime:
    """
    Контракт classify_realized_price_regime(price: float, realized_price: float) -> str

    Паттерн МБ-01 (пост 25.02.2026):
      ABOVE  : price > RP * (1 + buf)
               Цена значительно выше Realized Price — не в зоне дна цикла.
               Mozart: «синяя линия» «самый сильный уровень» при тестировании снизу.
               Нахождение выше RP (с запасом) = рынок ещё не достиг исторического дна.

      AT     : RP * (1 - buf) <= price <= RP * (1 + buf)
               Цена в ±20% зоне вокруг Realized Price = зона исторического дна цикла.
               Mozart (пост 25.02.2026): RP = дно предыдущих циклов 2015, 2018, 2020.
               ±20% — формализованный буфер для защиты от ложного пробоя.

      BELOW  : price < RP * (1 - buf)
               Цена значительно ниже Realized Price = подтверждённый пробой уровня дна.
               Исторически: BTC бывал ниже RP при FTX-краше (кратко), крахе 2018.
    """

    # -----------------------------------------------------------------------
    # Тип возвращаемого значения
    # -----------------------------------------------------------------------

    def test_returns_string(self):
        # WHY: оркестратор вставляет метку в строковый блок вывода;
        # не-str вызовет TypeError при f-строке без исключения в runtime.
        result = classify_realized_price_regime(_at_center(), _RP)
        assert isinstance(result, str)

    # -----------------------------------------------------------------------
    # Три зоны: центры диапазонов
    # -----------------------------------------------------------------------

    def test_above_zone(self):
        # WHY: price явно выше RP * (1 + buf) → ABOVE.
        # Ошибочный AT скроет что цена уже покинула зону дна цикла;
        # оркестратор ложно сигнализирует «цена у синей линии» при цене на 25%+ выше RP.
        assert classify_realized_price_regime(_above_center(), _RP) == "ABOVE"

    def test_at_zone_center(self):
        # WHY: price == RP (точно на Realized Price) → AT.
        # Нахождение на самом RP = ядро «синей линии»; ABOVE или BELOW здесь —
        # грубая ошибка, которая инвертирует сигнал в самой важной точке.
        assert classify_realized_price_regime(_at_center(), _RP) == "AT"

    def test_below_zone(self):
        # WHY: price явно ниже RP * (1 - buf) → BELOW.
        # Ошибочный AT скроет подтверждённый пробой уровня дна;
        # Mozart: BTC ниже синей линии = зона капитуляции последних продавцов.
        assert classify_realized_price_regime(_below_center(), _RP) == "BELOW"

    # -----------------------------------------------------------------------
    # Граничные значения — верхняя граница AT / ABOVE
    # -----------------------------------------------------------------------

    def test_boundary_upper_exact_is_at(self):
        # WHY: price == RP * (1 + buf) → AT (верхняя граница включительно).
        # Контракт: ровно на верхней границе — ещё в зоне «синей линии», не ABOVE.
        # ABOVE начинается строго выше: ошибка >= вместо > для ABOVE
        # даёт ложный «цена покинула зону» при price ровно на границе.
        buf   = float(MOZART_CONFIG["realized_price_buffer_pct"])
        upper = _RP * (1.0 + buf)
        assert classify_realized_price_regime(upper, _RP) == "AT"

    def test_boundary_just_above_upper_is_above(self):
        # WHY: price == RP * (1 + buf) + _STEP → ABOVE.
        # Первое значение строго выше верхней границы = цена покинула зону.
        # Фиксирует что ABOVE начинается ПОСЛЕ границы, а не на ней.
        # Без этого теста ошибка (>= вместо >) остаётся незамеченной.
        buf   = float(MOZART_CONFIG["realized_price_buffer_pct"])
        upper = _RP * (1.0 + buf)
        assert classify_realized_price_regime(upper + _STEP, _RP) == "ABOVE"

    # -----------------------------------------------------------------------
    # Граничные значения — нижняя граница AT / BELOW
    # -----------------------------------------------------------------------

    def test_boundary_lower_exact_is_at(self):
        # WHY: price == RP * (1 - buf) → AT (нижняя граница включительно).
        # Контракт: ровно на нижней границе — ещё в зоне «синей линии», не BELOW.
        # BELOW начинается строго ниже: ошибка < вместо <= для AT
        # даёт ложный «пробой» при price ровно на нижней границе буфера.
        buf   = float(MOZART_CONFIG["realized_price_buffer_pct"])
        lower = _RP * (1.0 - buf)
        assert classify_realized_price_regime(lower, _RP) == "AT"

    def test_boundary_just_below_lower_is_below(self):
        # WHY: price == RP * (1 - buf) - _STEP → BELOW.
        # Первое значение строго ниже нижней границы = подтверждённый пробой.
        # Фиксирует что BELOW начинается сразу под границей, без мёртвой зоны.
        # Mozart: «ложный пробой» защищён буфером — без него каждый тик ниже RP
        # тригерил бы сигнал дна. Тест защищает от сжатия буфера.
        buf   = float(MOZART_CONFIG["realized_price_buffer_pct"])
        lower = _RP * (1.0 - buf)
        assert classify_realized_price_regime(lower - _STEP, _RP) == "BELOW"

    # -----------------------------------------------------------------------
    # AT-зона: верхняя и нижняя внутренние точки
    # -----------------------------------------------------------------------

    def test_at_zone_near_upper(self):
        # WHY: price чуть ниже верхней границы → AT.
        # Проверяет что AT-зона включает весь диапазон [lower, upper],
        # не только centre. Ошибка «урезанного» буфера даст ABOVE раньше границы.
        buf   = float(MOZART_CONFIG["realized_price_buffer_pct"])
        upper = _RP * (1.0 + buf)
        assert classify_realized_price_regime(upper - _STEP, _RP) == "AT"

    def test_at_zone_near_lower(self):
        # WHY: price чуть выше нижней границы → AT.
        # Зеркало предыдущего теста для нижней части буфера.
        # Без этого теста «урезание» буфера снизу остаётся незамеченным.
        buf   = float(MOZART_CONFIG["realized_price_buffer_pct"])
        lower = _RP * (1.0 - buf)
        assert classify_realized_price_regime(lower + _STEP, _RP) == "AT"

    # -----------------------------------------------------------------------
    # Корректность меток
    # -----------------------------------------------------------------------

    def test_only_valid_labels_returned(self):
        # WHY: опечатка в метке ('above', 'AT ' с пробелом) — тихий баг:
        # оркестратор не упадёт, но строковое сравнение вернёт False.
        # Проверяем все три зоны — ловим опечатку в любой ветке return.
        valid = {"ABOVE", "AT", "BELOW"}
        for price in [_above_center(), _at_center(), _below_center()]:
            assert classify_realized_price_regime(price, _RP) in valid
