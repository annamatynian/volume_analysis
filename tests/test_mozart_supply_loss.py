"""
tests/test_mozart_supply_loss.py
=================================
TDD — МБ-04: Supply in Loss — счётчик монет в убытке.

Контракт:

  classify_supply_loss_regime(supply_loss_btc: float) -> str

  Зоны (пороги из MOZART_CONFIG):
    supply_loss_btc >= structural_trigger  (5 000 000)  →  'EXTREME'
    supply_loss_btc >= intermediate_trigger (3 500 000)  →  'ELEVATED'
    supply_loss_btc >  0  (ниже 3.5M)                   →  'INTERMEDIATE'
    supply_loss_btc <= 0                                 →  'LOW'

  Границы включительно:
    supply_loss_btc == structural_trigger   →  'EXTREME'   (не 'ELEVATED')
    supply_loss_btc == intermediate_trigger →  'ELEVATED'  (не 'INTERMEDIATE')

Источник паттерна: посты 02.04.2026 и 08.04.2026 (МБ-04, PLAN_MOZART_PATTERNS.md ЧАСТЬ 2)
  Mozart: «5M монет = смена структурного тренда»,
          «3–3.5M убыточных монет над ценой = активное сопротивление».
  Диагностика 20.05.2026: 200 OK, 1461 запись, поле supplyLoss (float, BTC).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# RED: classify_supply_loss_regime ещё не существует → ImportError подтверждает RED
from mozart_signals import classify_supply_loss_regime
from mozart_config import MOZART_CONFIG


# ---------------------------------------------------------------------------
# Вспомогательные константы тестов
# ---------------------------------------------------------------------------

# Пороги из конфига — тест никогда не хардкодит числа вида 5_000_000.
# WHY: если Mozart изменит порог — меняется только конфиг, тест не трогается.
_STRUCTURAL   = float(MOZART_CONFIG["supply_loss_structural_trigger"])    # 5_000_000
_INTERMEDIATE = float(MOZART_CONFIG["supply_loss_intermediate_trigger"])  # 3_500_000

# _STEP — шаг для граничных тестов «just above / just below».
# WHY 1.0 BTC: меньше любого порога на много порядков → однозначно пересекает
# границу зоны, но не захватывает соседнюю зону (зоны шириной ~1.5M BTC).
_STEP = 1.0


# ---------------------------------------------------------------------------
# TestClassifySupplyLossRegime
# ---------------------------------------------------------------------------

class TestClassifySupplyLossRegime:
    """
    Контракт classify_supply_loss_regime(supply_loss_btc: float) -> str

    Паттерн МБ-04 (посты 02.04.2026, 08.04.2026):

      EXTREME     : supply_loss >= structural_trigger (5M BTC)
                    Исторически: пики 2019 (10.05M) и 2022 (9.7M).
                    «Коррекция к 5M = смена структурного тренда» (02.04.2026).

      ELEVATED    : intermediate_trigger <= supply_loss < structural_trigger
                    «3–3.5M убыточных монет над ценой = активное сопротивление» (08.04.2026).

      INTERMEDIATE: 0 < supply_loss < intermediate_trigger
                    Supply in Loss ниже ключевых Mozart-уровней —
                    давление есть, но структурных сигналов нет.

      LOW         : supply_loss <= 0
                    Теоретическая зона (пик бычьего рынка, почти все в прибыли).
                    В реальных данных BTC не наблюдается, но функция обязана
                    корректно обрабатывать любой float-вход.
    """

    # -----------------------------------------------------------------------
    # Тип возвращаемого значения
    # -----------------------------------------------------------------------

    def test_returns_string(self):
        # WHY: оркестратор вставляет метку в строковый блок вывода;
        # не-str вызовет TypeError при форматировании без явного исключения.
        result = classify_supply_loss_regime(_STRUCTURAL + _STEP)
        assert isinstance(result, str)

    # -----------------------------------------------------------------------
    # Четыре зоны: центры диапазонов
    # -----------------------------------------------------------------------

    def test_extreme_zone(self):
        # WHY: значение явно выше structural_trigger → EXTREME.
        # Ошибочный ELEVATED означал бы что движок не сигнализирует
        # о структурном давлении при исторически пиковых объёмах убыточных монет.
        assert classify_supply_loss_regime(_STRUCTURAL + 1_000_000) == "EXTREME"

    def test_elevated_zone(self):
        # WHY: значение между structural и intermediate → ELEVATED.
        # Центр зоны ELEVATED: intermediate + (structural - intermediate) / 2.
        # Ошибочный EXTREME завысил бы сигнал; ошибочный INTERMEDIATE — занизил бы.
        center = (_STRUCTURAL + _INTERMEDIATE) / 2
        assert classify_supply_loss_regime(center) == "ELEVATED"

    def test_intermediate_zone(self):
        # WHY: значение явно ниже intermediate_trigger, но > 0 → INTERMEDIATE.
        # Ошибочный ELEVATED: движок ложно сигнализирует «активное сопротивление»
        # при объёме ниже ключевого уровня Mozart (3–3.5M).
        assert classify_supply_loss_regime(_INTERMEDIATE / 2) == "INTERMEDIATE"

    def test_low_zone_zero(self):
        # WHY: supply_loss == 0 → LOW.
        # Граничный случай: ровно ноль монет в убытке — теоретически чистый
        # бычий рынок. Функция должна обрабатывать 0.0 без исключений.
        assert classify_supply_loss_regime(0.0) == "LOW"

    def test_low_zone_negative(self):
        # WHY: supply_loss < 0 → LOW.
        # Отрицательные значения физически не встречаются в данных BGeometrics,
        # но функция обязана быть устойчивой к любому float-вводу.
        # Ошибочный INTERMEDIATE при < 0 — тихий баг при некорректных данных API.
        assert classify_supply_loss_regime(-1.0) == "LOW"

    # -----------------------------------------------------------------------
    # Граничные значения — structural_trigger (5M)
    # -----------------------------------------------------------------------

    def test_boundary_structural_trigger_exact_is_extreme(self):
        # WHY: supply_loss == structural_trigger → EXTREME.
        # Контракт: достижение 5M = структурный сигнал выполнен (включительно).
        # Ошибка >= вместо > для ELEVATED: ровно 5M даёт ELEVATED →
        # пропускает момент достижения исторического рубикона.
        assert classify_supply_loss_regime(_STRUCTURAL) == "EXTREME"

    def test_boundary_just_below_structural_is_elevated(self):
        # WHY: supply_loss == structural_trigger - _STEP → ELEVATED.
        # Первый тик ниже 5M = ещё не структурный, но активное давление.
        # Фиксирует что EXTREME начинается строго на boundary, не ниже.
        assert classify_supply_loss_regime(_STRUCTURAL - _STEP) == "ELEVATED"

    def test_boundary_just_above_structural_is_extreme(self):
        # WHY: supply_loss == structural_trigger + _STEP → EXTREME.
        # Первый тик выше 5M = структурный сигнал активирован.
        # Без этого теста «урезание» EXTREME зоны снизу остаётся незамеченным.
        assert classify_supply_loss_regime(_STRUCTURAL + _STEP) == "EXTREME"

    # -----------------------------------------------------------------------
    # Граничные значения — intermediate_trigger (3.5M)
    # -----------------------------------------------------------------------

    def test_boundary_intermediate_trigger_exact_is_elevated(self):
        # WHY: supply_loss == intermediate_trigger → ELEVATED.
        # Контракт: достижение 3.5M = нижняя граница активного давления (включительно).
        # Mozart: «3–3.5M убыточных монет над ценой» (пост 08.04.2026).
        # Ошибка >= вместо > для INTERMEDIATE: ровно 3.5M даёт INTERMEDIATE →
        # движок недооценивает давление на нижней границе Mozart-зоны.
        assert classify_supply_loss_regime(_INTERMEDIATE) == "ELEVATED"

    def test_boundary_just_below_intermediate_is_intermediate(self):
        # WHY: supply_loss == intermediate_trigger - _STEP → INTERMEDIATE.
        # Первый тик ниже 3.5M = ниже ключевого Mozart-уровня, не ELEVATED.
        # Фиксирует что ELEVATED начинается строго на boundary, не ниже.
        assert classify_supply_loss_regime(_INTERMEDIATE - _STEP) == "INTERMEDIATE"

    def test_boundary_just_above_intermediate_is_elevated(self):
        # WHY: supply_loss == intermediate_trigger + _STEP → ELEVATED.
        # Первый тик выше 3.5M = активное давление по Mozart.
        # Без этого теста «урезание» ELEVATED зоны снизу остаётся незамеченным.
        assert classify_supply_loss_regime(_INTERMEDIATE + _STEP) == "ELEVATED"

    # -----------------------------------------------------------------------
    # Корректность меток
    # -----------------------------------------------------------------------

    def test_only_valid_labels_returned(self):
        # WHY: опечатка в метке ('extreme', 'ELEVATED ' с пробелом) — тихий баг:
        # оркестратор не упадёт, но строковое сравнение вернёт False.
        # Проверяем все четыре зоны — ловим опечатку в любой ветке return.
        valid = {"EXTREME", "ELEVATED", "INTERMEDIATE", "LOW"}
        test_values = [
            _STRUCTURAL + _STEP,    # EXTREME
            (_STRUCTURAL + _INTERMEDIATE) / 2,  # ELEVATED
            _INTERMEDIATE / 2,      # INTERMEDIATE
            0.0,                    # LOW
        ]
        for val in test_values:
            result = classify_supply_loss_regime(val)
            assert result in valid, (
                f"classify_supply_loss_regime({val}) вернул {result!r}, "
                f"ожидалось одно из {valid}"
            )
