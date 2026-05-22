"""
tests/test_mozart_lth_realized_profit.py
=========================================
TDD — МБ-05: LTH Realized Profit USD — порог медвежьего давления.
(паттерн МБ-05, PLAN_MOZART_PATTERNS.md ЧАСТЬ 2, пост 14.01.2026)

Контракт:

  classify_lth_realized_profit_regime(profit_7d_ma_usd: float) -> str

  Вход:  7-дневная скользящая средняя дневного LTH realized profit (USD).
         Вычисляется оркестратором из временного ряда get_lth_realized_profit_usd().

  Зоны (порог из MOZART_CONFIG["lth_profit_7d_ma_warning"] = 1_000_000_000):
    profit_7d_ma_usd > warning  →  'HIGH'    (медвежий риск распределения)
    profit_7d_ma_usd <= warning →  'NORMAL'  (нет аномального давления)

  Источник паттерна: пост 14.01.2026 (МБ-05, PLAN_MOZART_PATTERNS.md):
    «если давление продаж вновь сильно возрастёт (выше 1 млрд $ / день
     в среднем за 7 дней), то риски будут смещены в сторону медвежки»

  Граница:
    profit_7d_ma_usd == warning → 'NORMAL' (строгий >, ровно 1B = ещё не HIGH)

  Диагностика 2026-05-21: slug 'realized-profit-lth-usd' → 200 OK,
    1440 записей, поле realizedProfitLthUsd (float, USD, всегда >= 0).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# RED: classify_lth_realized_profit_regime ещё не существует → ImportError
from mozart_signals import classify_lth_realized_profit_regime
from mozart_config import MOZART_CONFIG

_WARNING = float(MOZART_CONFIG["lth_profit_7d_ma_warning"])  # 1_000_000_000


class TestClassifyLthRealizedProfitRegime:
    """
    Контракт classify_lth_realized_profit_regime(profit_7d_ma_usd) -> str

    Паттерн МБ-05 (пост 14.01.2026):
      HIGH   : 7d MA > $1B/день — сигнал медвежьего риска от LTH-распределения
      NORMAL : 7d MA <= $1B/день — давление в норме, нет аномалии
    """

    # -----------------------------------------------------------------------
    # Тип возвращаемого значения
    # -----------------------------------------------------------------------

    def test_returns_string(self):
        result = classify_lth_realized_profit_regime(_WARNING / 2)
        assert isinstance(result, str)
        # WHY: оркестратор вставляет метку в строковый блок вывода.
        # Не-str → TypeError при форматировании без явного исключения.

    # -----------------------------------------------------------------------
    # Две зоны: центры диапазонов
    # -----------------------------------------------------------------------

    def test_high_zone(self):
        result = classify_lth_realized_profit_regime(_WARNING * 2)
        assert result == "HIGH"
        # WHY: значение явно выше порога $1B → медвежий риск.
        # Ошибочный NORMAL: движок не сигнализирует об аномальном давлении
        # LTH-распределения, которое Mozart явно называет триггером медвежки.

    def test_normal_zone(self):
        result = classify_lth_realized_profit_regime(_WARNING / 2)
        assert result == "NORMAL"
        # WHY: значение явно ниже порога → нет аномалии.
        # Ошибочный HIGH: ложная тревога при нормальном давлении продаж LTH.

    # -----------------------------------------------------------------------
    # Граничные значения — порог $1B
    # -----------------------------------------------------------------------

    def test_boundary_exact_warning_is_normal(self):
        result = classify_lth_realized_profit_regime(_WARNING)
        assert result == "NORMAL"
        # WHY: Mozart: «выше 1 млрд» (строгий >). Ровно $1B = ещё не HIGH.
        # Ошибка >= вместо > для HIGH: ровно на границе даёт ложный сигнал
        # медвежьего риска в момент, когда его ещё нет по правилу Mozart.

    def test_boundary_just_above_warning_is_high(self):
        result = classify_lth_realized_profit_regime(_WARNING + 1)
        assert result == "HIGH"
        # WHY: первый цент выше $1B — порог пересечён, HIGH активируется.
        # Без этого теста смещение границы с > на >= остаётся незамеченным.

    def test_boundary_just_below_warning_is_normal(self):
        result = classify_lth_realized_profit_regime(_WARNING - 1)
        assert result == "NORMAL"
        # WHY: первый цент ниже $1B — ещё NORMAL. Фиксирует что HIGH
        # не начинается раньше порога.

    # -----------------------------------------------------------------------
    # Граничные случаи значений
    # -----------------------------------------------------------------------

    def test_zero_is_normal(self):
        result = classify_lth_realized_profit_regime(0.0)
        assert result == "NORMAL"
        # WHY: нулевой profit (LTH не фиксируют прибыль) — нет давления.
        # Функция не должна падать на 0.0.

    def test_large_value_is_high(self):
        result = classify_lth_realized_profit_regime(_WARNING * 10)
        assert result == "HIGH"
        # WHY: исторический пик ноябрь 2024 ~$2B/день. Функция должна
        # корректно обрабатывать любые реально наблюдаемые значения.

    # -----------------------------------------------------------------------
    # Корректность меток
    # -----------------------------------------------------------------------

    def test_only_valid_labels_returned(self):
        valid = {"HIGH", "NORMAL"}
        for val in [0.0, _WARNING / 2, _WARNING, _WARNING + 1, _WARNING * 2]:
            result = classify_lth_realized_profit_regime(val)
            assert result in valid, (
                f"classify_lth_realized_profit_regime({val}) вернул {result!r}, "
                f"ожидалось одно из {valid}"
            )
        # WHY: опечатка в метке ('high', 'High', 'NORMAL ') — тихий баг:
        # оркестратор не упадёт, но строковое сравнение вернёт False.
