# tests/test_macro_client.py
# TDD — НВ-02 | PPI → CPI макро-фильтр
# Mozart-паттерн: PPI (PPIACO) опережает CPI на 1–3 месяца.
#
# Правила проекта:
#   1. RED подтверждён pytest до написания production-кода
#   2. Пороги берутся из macro_client, не хардкодятся независимо
#   3. Тесты проверяют контракт и поведение, не воспроизводят логику
#   4. WHY-комментарий к каждому assert
#   5. Синтетические данные — нейтральные плейсхолдеры, не API-реалистичные
#   6. Граничные значения — отдельные тесты

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from macro_client import (
    MacroClient,
    classify_ppi_regime,
    PPI_FLAT_THRESHOLD,
    PPI_MONTHLY_SIGNAL_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Вспомогательные функции для синтетических данных
# ---------------------------------------------------------------------------

def _make_series(values: list) -> pd.Series:
    """Создаёт pd.Series с DatetimeIndex — нейтральные плейсхолдеры."""
    idx = pd.date_range('2025-01-01', periods=len(values), freq='MS')
    return pd.Series(values, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# MacroClient — инициализация и базовые методы
# ---------------------------------------------------------------------------

class TestMacroClientInit:

    def test_macro_client_no_key_raises(self, monkeypatch):
        """WHY: без API-ключа клиент должен упасть немедленно с ясным сообщением;
        иначе ошибка всплывёт только при первом сетевом запросе — трудно диагностировать."""
        monkeypatch.delenv('FRED_API_KEY', raising=False)
        with pytest.raises(ValueError, match='FRED_API_KEY'):
            MacroClient(api_key=None)

    def test_macro_client_accepts_explicit_key(self):
        """WHY: оркестратор может передать ключ явно (CI/CD без env);
        конструктор не должен падать при наличии api_key=."""
        with patch('macro_client.Fred') as mock_fred:
            client = MacroClient(api_key='test-key-placeholder')
            mock_fred.assert_called_once_with(api_key='test-key-placeholder')


class TestMacroClientSeries:

    def setup_method(self):
        """Мок Fred — не делаем реальных сетевых запросов."""
        patcher = patch('macro_client.Fred')
        self.mock_fred_cls = patcher.start()
        self.mock_fred_instance = MagicMock()
        self.mock_fred_cls.return_value = self.mock_fred_instance
        self.client = MacroClient(api_key='test-key-placeholder')
        self.addCleanup = patcher.stop

    def test_get_ppi_series_returns_series(self):
        """WHY: оркестратор передаёт Series в classify_ppi_regime;
        если get_ppi_series вернёт не Series — classifier упадёт с непонятной ошибкой."""
        full_series = _make_series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        self.mock_fred_instance.get_series.return_value = full_series
        result = self.client.get_ppi_series(months=6)
        assert isinstance(result, pd.Series), (
            # WHY: контракт метода — pd.Series; любой другой тип сломает downstream
            f'Ожидался pd.Series, получен {type(result)}'
        )

    def test_get_ppi_series_respects_months_param(self):
        """WHY: оркестратор запрашивает конкретное окно (6 мес для режима, 12 для тренда);
        если tail(months) не применяется — получим весь ряд и цикл упадёт по памяти."""
        full_series = _make_series(list(range(1, 13)))  # 12 значений
        self.mock_fred_instance.get_series.return_value = full_series
        result = self.client.get_ppi_series(months=4)
        assert len(result) == 4, (
            # WHY: months=4 должен давать ровно 4 точки
            f'Ожидалось 4 значения, получено {len(result)}'
        )

    def test_get_cpi_series_returns_series(self):
        """WHY: симметричный контракт с get_ppi_series;
        оркестратор выводит CPI в информационный блок — нужен pd.Series."""
        full_series = _make_series([300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0])
        self.mock_fred_instance.get_series.return_value = full_series
        result = self.client.get_cpi_series(months=6)
        assert isinstance(result, pd.Series), (
            f'Ожидался pd.Series, получен {type(result)}'
        )

    def test_get_cpi_series_respects_months_param(self):
        """WHY: то же окно, что и PPI — контракт симметричный."""
        full_series = _make_series(list(range(100, 113)))  # 13 значений
        self.mock_fred_instance.get_series.return_value = full_series
        result = self.client.get_cpi_series(months=3)
        assert len(result) == 3, (
            f'Ожидалось 3 значения, получено {len(result)}'
        )


# ---------------------------------------------------------------------------
# classify_ppi_regime — режимы
# ---------------------------------------------------------------------------

class TestClassifyPpiRegime:

    def test_classify_ppi_regime_rising(self):
        """WHY: Mozart-правило — 3 мес роста PPI = сигнал роста CPI через 1–3 мес;
        если RISING не детектируется — оркестратор не выводит макро-предупреждение."""
        # Синтетические данные: каждый следующий выше предыдущего
        series = _make_series([10.0, 11.0, 12.0, 13.0])
        result = classify_ppi_regime(series)
        assert result == 'RISING', (
            # WHY: все три изменения положительны → контракт должен вернуть RISING
            f'Ожидался RISING для монотонного роста, получен {result!r}'
        )

    def test_classify_ppi_regime_falling(self):
        """WHY: три мес снижения = предшественник дезинфляции;
        неверная классификация скроет от оркестратора улучшение макро-условий."""
        series = _make_series([13.0, 12.0, 11.0, 10.0])
        result = classify_ppi_regime(series)
        assert result == 'FALLING', (
            f'Ожидался FALLING для монотонного снижения, получен {result!r}'
        )

    def test_classify_ppi_regime_flat(self):
        """WHY: изменения в пределах PPI_FLAT_THRESHOLD → нет тренда = FLAT;
        неверная классификация даст ложный RISING/FALLING — тихий баг в оркестраторе."""
        # Суммарное изменение = 0.3, что меньше PPI_FLAT_THRESHOLD (0.5)
        tiny_delta = PPI_FLAT_THRESHOLD / 4  # заведомо меньше порога
        series = _make_series([10.0, 10.0 + tiny_delta, 10.0, 10.0 + tiny_delta])
        result = classify_ppi_regime(series)
        assert result == 'FLAT', (
            # WHY: суммарное изменение ниже порога → должен быть FLAT
            f'Ожидался FLAT при незначительных изменениях, получен {result!r}'
        )

    def test_classify_ppi_regime_mixed(self):
        """WHY: разнонаправленное движение = нет тренда, Mozart не применяет фильтр;
        если MIXED классифицируется как RISING — ложный медвежий сигнал."""
        # +2, -1, +2 → рост не устойчивый
        series = _make_series([10.0, 12.0, 11.0, 13.0])
        result = classify_ppi_regime(series)
        assert result == 'MIXED', (
            f'Ожидался MIXED для разнонаправленного движения, получен {result!r}'
        )

    def test_classify_ppi_regime_too_few_values_raises(self):
        """WHY: менее 4 точек → нельзя посчитать 3 периода изменений;
        тихий возврат MIXED скроет ошибку данных — должно быть явное исключение."""
        series = _make_series([10.0, 11.0, 12.0])  # только 3 точки
        with pytest.raises(ValueError, match='минимум 4'):
            classify_ppi_regime(series)

    # --- Граничные значения ---

    def test_classify_ppi_regime_flat_boundary_exactly_at_threshold(self):
        """WHY: суммарное изменение == PPI_FLAT_THRESHOLD — граница FLAT/не-FLAT;
        ошибка «строго <» vs «≤» здесь превратит FLAT в MIXED — классический off-by-one."""
        # Строим серию, где |c1+c2+c3| == PPI_FLAT_THRESHOLD ровно
        # Три одинаковых малых изменения, сумма = threshold
        delta = PPI_FLAT_THRESHOLD / 3
        series = _make_series([10.0, 10.0 + delta, 10.0 + 2 * delta, 10.0 + PPI_FLAT_THRESHOLD])
        result = classify_ppi_regime(series)
        assert result == 'FLAT', (
            # WHY: на границе (==) должен быть FLAT — пороговое значение включительно
            f'На границе PPI_FLAT_THRESHOLD ожидался FLAT, получен {result!r}'
        )

    def test_classify_ppi_regime_rising_uses_exactly_last_four(self):
        """WHY: если classifier берёт не последние 4 точки — начало ряда может
        «испортить» тренд; оркестратор всегда даёт актуальный сигнал."""
        # Первые 4 — падение, последние 4 — рост
        series = _make_series([20.0, 19.0, 18.0, 17.0, 10.0, 11.0, 12.0, 13.0])
        result = classify_ppi_regime(series)
        assert result == 'RISING', (
            # WHY: classifier должен смотреть только на хвост ряда
            f'Ожидался RISING по последним 4 точкам, получен {result!r}'
        )
