# tests/test_mozart_one_cycle.py
# МБ-08 | One-Cycle Average — средняя когорты 2-4 года (пост 13.05.2026)
#
# Диагностика 20.05.2026:
#   /v1/realized-cap-hodl-waves → age_2y_3y=0.056 (RC-доля), age_3y_4y=0.051
#   /v1/hodl-waves-supply       → age_2y_3y=1,121,764 BTC, age_3y_4y=1,021,607 BTC
#
# Формула OCA:
#   OCA = (rc_2y3y + rc_3y4y) × realized_price × total_supply_btc
#         ─────────────────────────────────────────────────────────
#                   supply_2y3y_btc + supply_3y4y_btc
#
# Зоны classify_one_cycle_regime:
#   ABOVE          : price >= one_cycle_avg
#   TECHNICAL_BEAR : price < one_cycle_avg, days_below <= confirmed_days
#   CONFIRMED_BEAR : price < one_cycle_avg, days_below > confirmed_days

import pytest
from mozart_config import MOZART_CONFIG
from mozart_signals import calculate_one_cycle_average, classify_one_cycle_regime


# ---------------------------------------------------------------------------
# Контракт конфига
# ---------------------------------------------------------------------------

class TestOneCycleConfig:
    def test_confirmed_days_exists(self):
        assert "one_cycle_bear_confirmed_days" in MOZART_CONFIG
        # WHY: classify_one_cycle_regime читает этот ключ для CONFIRMED_BEAR;
        #   отсутствие → KeyError при каждой классификации в оркестраторе.

    def test_confirmed_days_is_numeric(self):
        assert isinstance(float(MOZART_CONFIG["one_cycle_bear_confirmed_days"]), float)
        # WHY: days_below сравнивается с int(config); нечисловое значение →
        #   TypeError при int() конвертации.

    def test_confirmed_days_positive(self):
        assert float(MOZART_CONFIG["one_cycle_bear_confirmed_days"]) > 0
        # WHY: ноль или отрицательное → CONFIRMED_BEAR активируется немедленно
        #   при любом снижении цены; смысл «более 2 месяцев» теряется.


# ---------------------------------------------------------------------------
# Контракт calculate_one_cycle_average — математические инварианты
# ---------------------------------------------------------------------------

class TestCalculateOneCycleAverage:
    def test_returns_float(self):
        result = calculate_one_cycle_average(
            age_2y_3y_rc_frac=0.06,
            age_3y_4y_rc_frac=0.05,
            realized_price=50_000.0,
            age_2y_3y_supply_btc=1_100_000.0,
            age_3y_4y_supply_btc=1_000_000.0,
            total_supply_btc=19_000_000.0,
        )
        assert isinstance(result, float)
        # WHY: OCA передаётся в classify_one_cycle_regime как float;
        #   не-float → TypeError при сравнении с price.

    def test_oca_positive(self):
        """OCA > 0 при корректных положительных входах."""
        result = calculate_one_cycle_average(
            age_2y_3y_rc_frac=0.06,
            age_3y_4y_rc_frac=0.05,
            realized_price=50_000.0,
            age_2y_3y_supply_btc=1_100_000.0,
            age_3y_4y_supply_btc=1_000_000.0,
            total_supply_btc=19_000_000.0,
        )
        assert result > 0
        # WHY: OCA = средняя цена покупки когорты в USD; отрицательное значение →
        #   classify всегда вернёт ABOVE, сигнал медвежьего рынка полностью ломается.

    def test_oca_equals_realized_price_when_fractions_match(self):
        """OCA == realized_price когда RC-доля == supply-доле (симметрия)."""
        realized_price = 55_000.0
        total_supply   = 20_000_000.0
        # Когорта держит 10% и RC, и supply — среднее совпадает с рынком
        cohort_supply  = total_supply * 0.10
        result = calculate_one_cycle_average(
            age_2y_3y_rc_frac=0.055,
            age_3y_4y_rc_frac=0.045,
            realized_price=realized_price,
            age_2y_3y_supply_btc=cohort_supply * 0.55,
            age_3y_4y_supply_btc=cohort_supply * 0.45,
            total_supply_btc=total_supply,
        )
        assert abs(result - realized_price) < 1.0
        # WHY: когда RC-доля == supply-доле, OCA тождественно равна realized_price.
        #   Это фундаментальный инвариант формулы; нарушение → ошибка реализации.

    def test_oca_above_realized_when_rc_fraction_exceeds_supply_fraction(self):
        """OCA > realized_price когда когорта держит больше RC чем supply."""
        realized_price = 50_000.0
        total_supply   = 20_000_000.0
        # RC-доля = 15%, supply-доля = 10% → купили дороже среднего
        result = calculate_one_cycle_average(
            age_2y_3y_rc_frac=0.09,
            age_3y_4y_rc_frac=0.06,
            realized_price=realized_price,
            age_2y_3y_supply_btc=1_000_000.0,
            age_3y_4y_supply_btc=1_000_000.0,
            total_supply_btc=total_supply,
        )
        assert result > realized_price
        # WHY: когорта держит непропорционально большую долю RC →
        #   заплатила выше средней. Если OCA <= realized_price — формула инвертирована.

    def test_oca_below_realized_when_supply_fraction_exceeds_rc_fraction(self):
        """OCA < realized_price когда когорта держит больше supply чем RC."""
        realized_price = 50_000.0
        total_supply   = 20_000_000.0
        # RC-доля = 10%, supply-доля = 15% → купили дешевле среднего
        result = calculate_one_cycle_average(
            age_2y_3y_rc_frac=0.06,
            age_3y_4y_rc_frac=0.04,
            realized_price=realized_price,
            age_2y_3y_supply_btc=1_500_000.0,
            age_3y_4y_supply_btc=1_500_000.0,
            total_supply_btc=total_supply,
        )
        assert result < realized_price
        # WHY: когорта держит непропорционально большую долю supply →
        #   заплатила ниже средней. Если OCA >= realized_price — формула инвертирована.

    def test_oca_scales_with_realized_price(self):
        """OCA пропорциональна realized_price — удвоение цены удваивает OCA."""
        kwargs = dict(
            age_2y_3y_rc_frac=0.06,
            age_3y_4y_rc_frac=0.05,
            age_2y_3y_supply_btc=1_100_000.0,
            age_3y_4y_supply_btc=1_000_000.0,
            total_supply_btc=19_000_000.0,
        )
        oca1 = calculate_one_cycle_average(realized_price=40_000.0, **kwargs)
        oca2 = calculate_one_cycle_average(realized_price=80_000.0, **kwargs)
        assert abs(oca2 / oca1 - 2.0) < 0.001
        # WHY: OCA линейна по realized_price (формула: rc_frac × rp × total / supply).
        #   Нелинейность → ошибка реализации (лишнее возведение в степень и т.п.).


# ---------------------------------------------------------------------------
# Контракт classify_one_cycle_regime — зоны и границы
# ---------------------------------------------------------------------------

class TestClassifyOneCycleRegimeReturnType:
    def test_returns_str(self):
        result = classify_one_cycle_regime(
            price=80_000.0, one_cycle_avg=89_000.0, days_below=0
        )
        assert isinstance(result, str)
        # WHY: оркестратор встраивает результат в f-строку; не-str → TypeError.


class TestClassifyOneCycleRegimeZones:
    def test_above_when_price_above_oca(self):
        oca = 89_000.0
        result = classify_one_cycle_regime(
            price=oca + 5_000.0, one_cycle_avg=oca, days_below=0
        )
        assert result == "ABOVE"
        # WHY: цена выше OCA = технической медвежки нет; Mozart (пост 13.05.2026):
        #   «как только цена опускается ниже» — не наш случай.

    def test_technical_bear_when_below_short_duration(self):
        confirmed_days = int(MOZART_CONFIG["one_cycle_bear_confirmed_days"])
        oca = 89_000.0
        result = classify_one_cycle_regime(
            price=oca - 5_000.0,
            one_cycle_avg=oca,
            days_below=confirmed_days // 2,
        )
        assert result == "TECHNICAL_BEAR"
        # WHY: Mozart (пост 13.05.2026): «техническая медвежка» = менее 2 месяцев
        #   ниже OCA. CONFIRMED_BEAR требует sustained duration; ранняя классификация
        #   как CONFIRMED → ложный медвежий сигнал при краткосрочной коррекции.

    def test_confirmed_bear_when_below_long_duration(self):
        confirmed_days = int(MOZART_CONFIG["one_cycle_bear_confirmed_days"])
        oca = 89_000.0
        result = classify_one_cycle_regime(
            price=oca - 5_000.0,
            one_cycle_avg=oca,
            days_below=confirmed_days + 10,
        )
        assert result == "CONFIRMED_BEAR"
        # WHY: Mozart (пост 13.05.2026): «настоящий медвежий рынок» = цена ниже OCA
        #   более 2 месяцев. TECHNICAL_BEAR при долгом нахождении ниже → потеря
        #   структурного сигнала, который оркестратор выводит в [FINAL VERDICT].


class TestClassifyOneCycleRegimeBoundaries:
    def test_price_at_oca_is_above(self):
        """price == one_cycle_avg → ABOVE (рубикон не пробит вниз)."""
        oca = 89_000.0
        result = classify_one_cycle_regime(
            price=oca, one_cycle_avg=oca, days_below=0
        )
        assert result == "ABOVE"
        # WHY: Mozart (пост 13.05.2026): «как только цена опускается НИЖЕ» —
        #   строго <; ровно на OCA = рубикон не пробит.
        #   Если oca == price → TECHNICAL_BEAR: ложный сигнал при нейтральном рынке.

    def test_days_at_threshold_is_technical_bear(self):
        """days_below == confirmed_days → TECHNICAL_BEAR (ещё не превышен)."""
        confirmed_days = int(MOZART_CONFIG["one_cycle_bear_confirmed_days"])
        oca = 89_000.0
        result = classify_one_cycle_regime(
            price=oca - 1.0,
            one_cycle_avg=oca,
            days_below=confirmed_days,
        )
        assert result == "TECHNICAL_BEAR"
        # WHY: Mozart «более 2 месяцев» = строго > порога; ровно на пороге = не подтверждено.
        #   Если == confirmed_days → CONFIRMED_BEAR: преждевременная активация,
        #   недопустимая потеря одного дня точности.

    def test_days_one_above_threshold_is_confirmed_bear(self):
        """days_below == confirmed_days + 1 → CONFIRMED_BEAR."""
        confirmed_days = int(MOZART_CONFIG["one_cycle_bear_confirmed_days"])
        oca = 89_000.0
        result = classify_one_cycle_regime(
            price=oca - 1.0,
            one_cycle_avg=oca,
            days_below=confirmed_days + 1,
        )
        assert result == "CONFIRMED_BEAR"
        # WHY: минимальное превышение порога = переход в CONFIRMED_BEAR.
        #   Если days+1 → TECHNICAL_BEAR: граница смещена на один день,
        #   пропуск перехода в структурный медвежий режим.

    def test_above_overrides_days_below(self):
        """Цена выше OCA → ABOVE независимо от days_below."""
        confirmed_days = int(MOZART_CONFIG["one_cycle_bear_confirmed_days"])
        oca = 89_000.0
        result = classify_one_cycle_regime(
            price=oca + 1.0,
            one_cycle_avg=oca,
            days_below=confirmed_days + 100,
        )
        assert result == "ABOVE"
        # WHY: цена вернулась выше OCA — медвежий режим снят.
        #   Сохранение CONFIRMED_BEAR при восстановлении → навсегда залипший сигнал,
        #   оркестратор не отразит смену фазы.
