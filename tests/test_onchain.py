"""
tests/test_onchain.py
=====================
Unit tests for OnChainValidator.check_capitulation_signal().

Design principles:
- Tests verify CONTRACTS, not implementation logic.
- No network calls — check_capitulation_signal() is a pure function:
  it receives a ready DataFrame, no API involved.
- Edge cases: empty df, one day below threshold, all days above.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from onchain_validator import OnChainValidator
from onchain_client import BGeometricsClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loss_df(values: list) -> pd.DataFrame:
    """
    Build a minimal realized_loss DataFrame from a list of USD values.
    Matches the contract: columns [date, lth_realized_loss_usd].
    """
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(values))]
    return pd.DataFrame({
        'date':                  dates,
        'lth_realized_loss_usd': values,
    })


def _make_validator() -> OnChainValidator:
    """OnChainValidator without network — check_capitulation_signal is pure."""
    # WHY: BGeometricsClient не вызывается в check_capitulation_signal,
    # поэтому передаём None — тест проверяет только логику агрегации.
    return OnChainValidator(client=None)


# ---------------------------------------------------------------------------
# check_capitulation_signal
# ---------------------------------------------------------------------------

class TestCheckCapitulationSignal:

    THRESHOLD = 300_000_000   # $300M — дефолт из ТЗ

    def test_capitulation_true_when_all_days_above_threshold(self):
        """
        Последние 3 дня > $300M → сигнал капитуляции True.
        Контракт: (recent['realized_loss_usd'] > threshold).all() == True.
        """
        df = _make_loss_df([
            100_000_000,   # ранние дни — ниже порога
            150_000_000,
            400_000_000,   # последние 3 — все выше $300M
            500_000_000,
            600_000_000,
        ])
        validator = _make_validator()
        assert validator.check_capitulation_signal(df, threshold_usd=self.THRESHOLD) is True

    def test_capitulation_false_when_one_day_below(self):
        """
        Один из последних 3 дней < $300M → сигнал False.
        Контракт: .all() требует ВСЕ три дня выше порога.
        """
        df = _make_loss_df([
            500_000_000,
            600_000_000,
            100_000_000,   # ниже порога — ломает условие
        ])
        validator = _make_validator()
        assert validator.check_capitulation_signal(df, threshold_usd=self.THRESHOLD) is False

    def test_capitulation_false_when_empty_df(self):
        """
        Пустой DataFrame → сигнал False (нет данных = нет капитуляции).
        Контракт: граничный случай, функция не падает с исключением.
        """
        df = _make_loss_df([])
        validator = _make_validator()
        assert validator.check_capitulation_signal(df, threshold_usd=self.THRESHOLD) is False

    def test_accepts_lth_realized_loss_usd_column_name(self):
        """
        Contract: функция должна работать с колонкой 'lth_realized_loss_usd' —
        именно такое имя возвращает get_realized_loss_lth_usd().

        WHY: оркестратор передаёт DataFrame из get_realized_loss_lth_usd()
        напрямую в check_capitulation_signal(). Обе функции должны
        согласовать по имени колонки. Несоответствие было
        подтверждено WARN в оркестраторе 2026-05-12:
        KeyError: 'realized_loss_usd' при получении lth_realized_loss_usd.
        """
        df = pd.DataFrame({
            'date': [datetime(2026, 5, 9), datetime(2026, 5, 10), datetime(2026, 5, 11)],
            'lth_realized_loss_usd': [400_000_000, 500_000_000, 600_000_000],
        })
        validator = _make_validator()
        # все 3 дня > $300M — ожидаем True
        result = validator.check_capitulation_signal(df, threshold_usd=300_000_000)
        assert result is True, (
            "check_capitulation_signal должна работать с колонкой 'lth_realized_loss_usd', "
            "которую возвращает get_realized_loss_lth_usd()"
        )


class TestCheckCapitulationSignalNegativeValues:
    """
    API realized_loss_lth возвращает убытки как ОТРИЦАТЕЛЬНЫЕ числа.
    Диагностика 2026-05-23: last value = -258_141_195.69 USD.

    Контракт: функция обязана работать с abs() —
    убыток -350M по модулю превышает порог $300M → капитуляция True.

    WHY отдельный класс: существующие тесты используют положительные
    mock-значения и не проверяют реальный знак API-данных.
    """

    THRESHOLD = 300_000_000

    def test_capitulation_true_when_all_negative_losses_exceed_threshold_by_abs(self):
        """
        Убытки отрицательные, но |value| > порога → капитуляция True.

        WHY: реальный API возвращает -258M, -350M (знак минус = убыток).
        abs(-350M) = 350M > 300M — должно быть True.

        WHY RED сейчас: текущая реализация проверяет value > threshold
        (без abs), поэтому -350M > 300M → False — баг в production.
        """
        df = _make_loss_df([
            -100_000_000,   # ранние дни — ниже порога по модулю
            -150_000_000,
            -350_000_000,   # последние 3 — все > $300M по модулю
            -420_000_000,
            -510_000_000,
        ])
        validator = _make_validator()
        # WHY: abs(-350M) = 350M > 300M → сигнал капитуляции должен сработать
        assert validator.check_capitulation_signal(df, threshold_usd=self.THRESHOLD) is True

    def test_capitulation_false_when_negative_losses_below_threshold_by_abs(self):
        """
        Убытки отрицательные, но |value| < порога → капитуляция False.

        WHY: гарантирует что abs() не инвертирует логику для малых убытков.
        abs(-100M) = 100M < 300M → False корректно.
        """
        df = _make_loss_df([
            -50_000_000,
            -80_000_000,
            -100_000_000,
        ])
        validator = _make_validator()
        # WHY: abs(-100M) = 100M < 300M → капитуляции нет
        assert validator.check_capitulation_signal(df, threshold_usd=self.THRESHOLD) is False
