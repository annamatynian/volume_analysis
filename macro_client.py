# macro_client.py
# Макро-данные из FRED для блока НВ-02 (PPI → CPI паттерн Mozart 13.05.2026)
#
# Mozart-паттерн:
#   PPI (PPIACO) — опережающий индикатор CPI с лагом 1–3 месяца.
#   Месячный PPI > 1% = сигнал устойчивого роста CPI.
#   Рост CPI → ФРС удерживает ставку → ликвидность ограничена.
#   (Mozart, пост 13.05.2026)

import os
import pandas as pd
from fredapi import Fred

# --- Конфигурация ---
FRED_SERIES = {
    'ppi': 'PPIACO',    # Producer Price Index — опережающий к CPI
    'cpi': 'CPIAUCSL',  # Consumer Price Index — запаздывающий к PPI
}

# Порог месячного изменения PPI (Mozart: > 1% = сигнал устойчивой инфляции)
PPI_MONTHLY_SIGNAL_THRESHOLD = 1.0  # процентных пункта

# Порог «плоского» тренда — изменение меньше этого значения за 3 мес = FLAT
PPI_FLAT_THRESHOLD = 0.5  # процентных пункта суммарно за 3 периода


class MacroClient:
    """Клиент FRED API для макро-индикаторов (НВ-02).

    Предоставляет временные ряды PPI и CPI для анализа
    опережающего макро-фильтра в Mozart.
    """

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get('FRED_API_KEY')
        if not key:
            raise ValueError(
                'FRED_API_KEY не установлен. '
                'Передайте api_key= или установите переменную окружения FRED_API_KEY.'
            )
        self.fred = Fred(api_key=key)

    def get_ppi_series(self, months: int = 6) -> pd.Series:
        """PPIACO — последние N месяцев (Producer Price Index).

        Returns:
            pd.Series с DatetimeIndex, значения — уровень индекса.
        """
        return self.fred.get_series(FRED_SERIES['ppi']).tail(months)

    def get_cpi_series(self, months: int = 6) -> pd.Series:
        """CPIAUCSL — последние N месяцев (Consumer Price Index).

        Returns:
            pd.Series с DatetimeIndex, значения — уровень индекса.
        """
        return self.fred.get_series(FRED_SERIES['cpi']).tail(months)


def classify_ppi_regime(ppi_series: pd.Series) -> str:
    """Классифицирует режим PPI по 3-месячному тренду последних значений.

    Требует минимум 4 значения в серии (для 3 периодов изменений).

    Args:
        ppi_series: pd.Series с хронологически упорядоченными значениями PPI.
                    Должна содержать не менее 4 точек.

    Returns:
        'RISING'  — PPI растёт 3 периода подряд (Mozart: → рост CPI через 1–3 мес)
        'FALLING' — PPI падает 3 периода подряд
        'FLAT'    — суммарное изменение за 3 периода ≤ PPI_FLAT_THRESHOLD
        'MIXED'   — нет устойчивого однонаправленного движения

    Raises:
        ValueError: если серия содержит менее 4 значений.
    """
    if len(ppi_series) < 4:
        raise ValueError(
            f'classify_ppi_regime требует минимум 4 значения, получено: {len(ppi_series)}'
        )

    # Берём последние 4 точки → 3 изменения
    last_four = ppi_series.iloc[-4:]
    changes = last_four.diff().dropna()  # 3 значения

    c1, c2, c3 = changes.iloc[0], changes.iloc[1], changes.iloc[2]
    total_change = abs(c1 + c2 + c3)

    # FLAT: суммарное движение незначительно
    if total_change <= PPI_FLAT_THRESHOLD:
        return 'FLAT'

    # RISING: все три периода — рост
    if c1 > 0 and c2 > 0 and c3 > 0:
        return 'RISING'

    # FALLING: все три периода — падение
    if c1 < 0 and c2 < 0 and c3 < 0:
        return 'FALLING'

    return 'MIXED'
