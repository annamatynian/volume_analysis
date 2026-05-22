"""
tests/test_mozart_etf_flow.py
==============================
TDD — ВЕТКА 5: М-11 ETF Flow светофор спроса
(паттерн М-11, PLAN_MOZART_PATTERNS.md ЧАСТЬ 1, пост 08.04.2026).

Один контракт:

  classify_etf_flow_regime(flow_btc: float) -> str
      Классифицирует суточный поток BTC в/из спотовых ETF по трём зонам.
      Зоны (приоритет сверху вниз):
        flow_btc >  significant  → 'INFLOW'   — институциональный спрос активен
        flow_btc >= -significant → 'NEUTRAL'  — нет направленного сигнала
        flow_btc <  -significant → 'OUTFLOW'  — нет нового покупателя

      Контракт границы (FORMALIZED):
        flow_btc == significant  → 'NEUTRAL' (строгий >; не пересекает порог INFLOW)
        flow_btc == -significant → 'NEUTRAL' (включительно; нижний край NEUTRAL-зоны)

      Mozart (пост 08.04.2026) использует бинарно: «есть притоки / нет».
      Нейтральная зона и числовой порог — FORMALIZED.
      Порог строго из MOZART_CONFIG["etf_flow_significant_btc"].

Отличие от classify_sth_sopr_regime:
  ETF Flow — 3 зоны без eps-буфера. Граница «жёсткая» (нет шума ±0.5%):
  суточный ETF поток — агрегированная публичная отчётность, не on-chain SOPR.

Отличие от classify_lth_nupl_regime:
  NUPL — 4 зоны с рубиконом нуля и эйфорией.
  ETF Flow — 3 зоны симметричные ±significant; без зоны эйфории.

Правила:
  - Числовые пороги только через MOZART_CONFIG, не хардкодятся в assertions.
  - Тестовые значения вычисляются из порогов (шаг от границы, центр зоны).
  - _STEP = 1: шаг «just outside» зоны (1 BTC), не рыночный порог.
  - WHY-комментарий к каждому assert: что сломается в production.
  - Все 3 зоны, обе границы — отдельные тесты.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# RED: функции ещё нет — ImportError подтверждает RED
from mozart_signals import classify_etf_flow_regime
from mozart_config import MOZART_CONFIG


# ---------------------------------------------------------------------------
# _STEP: шаг для позиционирования тестовых значений «just outside» зоны.
# 1 BTC = минимальный шаг для целочисленного порога 500 BTC.
# Не является рыночным значением — используется только для чёткой позиции
# «строго за границей» в тестах граничных условий.
# ---------------------------------------------------------------------------
_STEP = 1


# ---------------------------------------------------------------------------
# Генераторы тестовых значений — вычисляются из конфига, не хардкодятся
# ---------------------------------------------------------------------------

def _flow_inflow_center() -> float:
    """
    Центр зоны INFLOW: significant + 100.
    100 BTC — шаг позиционирования, явно выше порога, не API-значение.
    """
    return float(MOZART_CONFIG["etf_flow_significant_btc"]) + 100.0


def _flow_neutral_center() -> float:
    """
    Центр NEUTRAL-зоны: 0 BTC.
    Точная середина симметричной NEUTRAL-зоны [-significant, +significant].
    """
    return 0.0


def _flow_outflow_center() -> float:
    """
    Центр зоны OUTFLOW: -significant - 100.
    100 BTC — шаг позиционирования, явно ниже порога, не API-значение.
    """
    return -(float(MOZART_CONFIG["etf_flow_significant_btc"]) + 100.0)


# ---------------------------------------------------------------------------
# TestClassifyEtfFlowRegime
# ---------------------------------------------------------------------------

class TestClassifyEtfFlowRegime:
    """
    Контракт classify_etf_flow_regime(flow_btc: float) -> str:

    Зоны М-11 (пост 08.04.2026; порог из MOZART_CONFIG):

      INFLOW  : flow_btc > significant
                Институциональный спрос активен; ETF net-buyer.
                Mozart: подтверждает покупателя при тестировании POC.

      NEUTRAL : -significant <= flow_btc <= significant
                Нет направленного сигнала; ETF-потоки в пределах шума.
                Верхняя граница включительно (flow == significant → NEUTRAL).
                Нижняя граница включительно (flow == -significant → NEUTRAL).

      OUTFLOW : flow_btc < -significant
                Нет нового институционального покупателя; ETF net-seller.
                Mozart: «нет нового покупателя» = отсутствие поддержки.
    """

    def test_returns_string(self):
        # WHY: оркестратор вставляет метку в строковый блок вывода;
        # не-str вызовет TypeError при f-строке без явной ошибки в runtime.
        result = classify_etf_flow_regime(_flow_neutral_center())
        assert isinstance(result, str)

    # ── Зоны — центры диапазонов ────────────────────────────────────────────

    def test_inflow_zone(self):
        # WHY: поток выше significant = институциональный спрос подтверждён.
        # Ошибочный NEUTRAL скроет от оркестратора бычий сигнал ETF;
        # Mozart использует INFLOW как подтверждение уровня POC (пост 08.04.2026).
        assert classify_etf_flow_regime(_flow_inflow_center()) == "INFLOW"

    def test_neutral_zone_at_zero(self):
        # WHY: flow == 0 = нет направленного сигнала; Mozart трактует как
        # «покупателя нет, но и продавца нет» — не медвежий сигнал по умолчанию.
        # Ошибочный INFLOW или OUTFLOW добавит ложный сигнал в нейтральной фазе.
        assert classify_etf_flow_regime(_flow_neutral_center()) == "NEUTRAL"

    def test_outflow_zone(self):
        # WHY: поток ниже -significant = нет институционального покупателя.
        # Mozart (пост 08.04.2026): при OUTFLOW уровень POC не получает
        # поддержки со стороны ETF — ослабляет бычий тезис.
        # Ошибочный NEUTRAL скроет медвежий сигнал.
        assert classify_etf_flow_regime(_flow_outflow_center()) == "OUTFLOW"

    # ── Верхняя граница NEUTRAL-зоны ─────────────────────────────────────────

    def test_boundary_upper_exact_is_neutral(self):
        # WHY: flow == significant → NEUTRAL (не INFLOW; строгий > для INFLOW).
        # Контракт FORMALIZED: Mozart не называет числовой порог — 500 BTC
        # отсекает шум, но само значение «на пороге» — ещё не значимый приток.
        # Ошибка >= вместо > для INFLOW: flow == significant → INFLOW —
        # ложный сигнал спроса при нейтральном потоке ровно на границе.
        significant = float(MOZART_CONFIG["etf_flow_significant_btc"])
        assert classify_etf_flow_regime(significant) == "NEUTRAL"

    def test_boundary_just_above_upper_is_inflow(self):
        # WHY: flow == significant + _STEP → INFLOW.
        # Первое значение строго выше порога = реальный сигнал спроса.
        # Фиксирует что зона INFLOW начинается ПОСЛЕ significant, не на нём.
        # Без этого теста ошибка знака верхней границы остаётся незамеченной.
        significant = float(MOZART_CONFIG["etf_flow_significant_btc"])
        assert classify_etf_flow_regime(significant + _STEP) == "INFLOW"

    # ── Нижняя граница NEUTRAL-зоны ──────────────────────────────────────────

    def test_boundary_lower_exact_is_neutral(self):
        # WHY: flow == -significant → NEUTRAL (не OUTFLOW; >= -significant).
        # Контракт FORMALIZED: значение ровно на нижней границе — ещё шум,
        # не сигнальный отток. OUTFLOW начинается строго ниже.
        # Ошибка > вместо >= (flow > -significant → NEUTRAL, else OUTFLOW):
        # flow == -significant → OUTFLOW — ложный сигнал «нет покупателя»
        # при нейтральном потоке на нижней границе.
        significant = float(MOZART_CONFIG["etf_flow_significant_btc"])
        assert classify_etf_flow_regime(-significant) == "NEUTRAL"

    def test_boundary_just_below_lower_is_outflow(self):
        # WHY: flow == -significant - _STEP → OUTFLOW.
        # Первое значение строго ниже нижней границы = значимый отток.
        # Фиксирует строгую нижнюю границу: значения ниже -significant
        # не захватываются NEUTRAL-зоной.
        # Без этого теста ошибка знака нижней границы остаётся незамеченной.
        significant = float(MOZART_CONFIG["etf_flow_significant_btc"])
        assert classify_etf_flow_regime(-significant - _STEP) == "OUTFLOW"

    # ── Корректность меток ───────────────────────────────────────────────────

    def test_only_valid_labels_returned(self):
        # WHY: опечатка в метке ('INFLOW ' с пробелом, 'Outflow' с регистром) —
        # тихий баг: оркестратор не упадёт, но условная логика перестанет
        # работать (str == 'OUTFLOW' → False для 'Outflow').
        # Проверяем все три зоны, чтобы выловить опечатку в любой ветке return.
        valid = {"INFLOW", "NEUTRAL", "OUTFLOW"}
        for flow in [_flow_inflow_center(), _flow_neutral_center(), _flow_outflow_center()]:
            assert classify_etf_flow_regime(flow) in valid
