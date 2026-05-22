"""
tests/test_mozart_true_market_mean.py
======================================
TDD — МБ-02: True Market Mean — «Зелёная линия», рубикон медвежьего рынка.

Контракт:

  classify_true_market_mean_regime(price: float, tmm: float) -> str
      Бинарный рубикон (Mozart, пост 25.02.2026):

        price >= tmm  →  'ABOVE'   (рубикон не пробит)
        price <  tmm  →  'BELOW'   (медвежий рынок подтверждён)

      Граница ВКЛЮЧИТЕЛЬНО:
        price == tmm  →  'ABOVE'   (цена на самом рубиконе — не пробой вниз)

      Нет буфера (в отличие от МБ-01):
        TMM — это «твёрдый рубикон», Mozart: «обозначило смену глобального тренда»
        только при реальном пробое вниз. Нахождение ровно на линии — не пробой.

      Только два допустимых значения: 'ABOVE', 'BELOW'.
      AT-зоны нет.

Источник паттерна: пост 25.02.2026 (МБ-02, PLAN_MOZART_PATTERNS.md ЧАСТЬ 5)
  Mozart: «🟢 Зелёная линия истинной рыночной цены (True Market Mean Price)...
  в конце января она была пробита вниз на огромных объемах и ликвидациях,
  что и обозначило смену глобального тренда»
  Endpoint: /v1/true-market-mean, поле: trueMarketMean (float)
  Диагностика 20.05.2026: 200 OK, 1458 записей.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# RED: classify_true_market_mean_regime ещё не существует → ImportError подтверждает RED
from mozart_signals import classify_true_market_mean_regime


# ---------------------------------------------------------------------------
# Вспомогательные константы тестов
# ---------------------------------------------------------------------------

# _TMM — нейтральный плейсхолдер. Круглое число, явно не реалистичная цена BTC.
# WHY не API-реалистичное: тест не должен зависеть от текущего значения TMM;
# изменение рыночной метрики не ломает тест.
_TMM = 10.0  # базовый true_market_mean для всех вычислений теста

# _STEP — шаг для граничных тестов «just above» / «just below».
# WHY 0.001: меньше _TMM на 4 порядка → однозначно пересекает рубикон
# без случайного захода в несуществующую AT-зону.
# По аналогии с _STEP в test_mozart_realized_price.py.
_STEP = 0.001


# ---------------------------------------------------------------------------
# TestClassifyTrueMarketMeanRegime
# ---------------------------------------------------------------------------

class TestClassifyTrueMarketMeanRegime:
    """
    Контракт classify_true_market_mean_regime(price: float, tmm: float) -> str

    Паттерн МБ-02 (пост 25.02.2026):
      ABOVE  : price >= tmm
               Цена на уровне или выше True Market Mean.
               Mozart: рубикон не пробит вниз.
               Восстановление / бычий рынок.

      BELOW  : price < tmm
               Цена ниже True Market Mean.
               Mozart: «пробой вниз... обозначил смену глобального тренда».
               Медвежий рынок подтверждён.

    Нет AT-зоны, нет буфера — чистый бинарный рубикон.
    """

    # -----------------------------------------------------------------------
    # Тип возвращаемого значения
    # -----------------------------------------------------------------------

    def test_returns_string(self):
        # WHY: оркестратор вставляет метку в строковый блок вывода;
        # не-str вызовет TypeError при f-строке без явного исключения в runtime.
        result = classify_true_market_mean_regime(_TMM + 1.0, _TMM)
        assert isinstance(result, str)

    # -----------------------------------------------------------------------
    # Два основных случая: центры диапазонов
    # -----------------------------------------------------------------------

    def test_above_zone(self):
        # WHY: price явно выше TMM → ABOVE.
        # Ошибочный BELOW инвертирует сигнал: при цене выше зелёной линии
        # оркестратор ложно сообщал бы о медвежьем рынке.
        assert classify_true_market_mean_regime(_TMM + 5.0, _TMM) == "ABOVE"

    def test_below_zone(self):
        # WHY: price явно ниже TMM → BELOW.
        # Ошибочный ABOVE скроет подтверждённый медвежий рынок;
        # Mozart: пробой TMM = «смена глобального тренда» — нельзя упустить.
        assert classify_true_market_mean_regime(_TMM - 5.0, _TMM) == "BELOW"

    # -----------------------------------------------------------------------
    # Граничное значение: price == tmm → ABOVE
    # -----------------------------------------------------------------------

    def test_boundary_exact_tmm_is_above(self):
        # WHY: price == tmm → ABOVE (рубикон не пробит вниз).
        # Контракт: нахождение ровно на TMM — это цена держится на линии,
        # а не пробой под неё. Mozart описывает «пробой вниз» как условие смены тренда.
        # Ошибка: strict > вместо >= → при price==tmm выдаст BELOW,
        # ложно сигнализируя медвежий рынок в момент теста зелёной линии.
        assert classify_true_market_mean_regime(_TMM, _TMM) == "ABOVE"

    def test_boundary_just_below_tmm_is_below(self):
        # WHY: price == tmm - _STEP → BELOW.
        # Первый тик строго ниже рубикона = медвежий рынок.
        # Фиксирует что BELOW начинается строго ниже TMM, а не на ней.
        # Без этого теста ошибка >= вместо > в условии ABOVE остаётся незамеченной
        # (при >= tmm - _step выдал бы ABOVE, но рынок уже «пробил» рубикон).
        assert classify_true_market_mean_regime(_TMM - _STEP, _TMM) == "BELOW"

    def test_boundary_just_above_tmm_is_above(self):
        # WHY: price == tmm + _STEP → ABOVE.
        # Первый тик строго выше TMM = однозначно ABOVE.
        # Дополняет тест boundary_exact и обеспечивает непрерывность ABOVE-зоны.
        # Без этого теста урезание ABOVE-зоны выше границы остаётся незамеченным.
        assert classify_true_market_mean_regime(_TMM + _STEP, _TMM) == "ABOVE"

    # -----------------------------------------------------------------------
    # Нет AT-зоны: только ABOVE и BELOW
    # -----------------------------------------------------------------------

    def test_only_valid_labels_returned(self):
        # WHY: опечатка в метке ('above', 'BELOW ' с пробелом, 'AT') — тихий баг:
        # оркестратор не упадёт, но строковое сравнение вернёт False.
        # Проверяем оба случая + границу — ловим опечатку в любой ветке return.
        # WHY нет 'AT' в valid: МБ-02 — чистый бинарный рубикон, AT-зоны нет.
        valid = {"ABOVE", "BELOW"}
        for price in [_TMM + 5.0, _TMM, _TMM - 5.0]:
            result = classify_true_market_mean_regime(price, _TMM)
            assert result in valid, (
                f"classify_true_market_mean_regime({price}, {_TMM}) вернул {result!r}, "
                f"ожидалось одно из {valid}"
            )
