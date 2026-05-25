# tests/test_mozart_btc_dominance.py
# НВ-03 | BTC Dominance — ротация ликвидности (пост 10.05.2026)
#
# Источник: PLAN_MOZART_PATTERNS.md ЧАСТЬ 4 НВ-03
# API:      CoinGecko /api/v3/global → поле btc_dominance (float, %)
# Сигнал:   снижение BTC.D > 2% за скользящий месяц = начало rotation в алты
#
# Функция:  classify_btc_dominance_trend(btc_d_current, btc_d_30d_ago) -> str
# Зоны:
#   delta = btc_d_current - btc_d_30d_ago
#   delta < -threshold  → ROTATION_ALTCOIN  (алты опережают BTC)
#   delta > +threshold  → ROTATION_BTC       (BTC доминирует над рынком)
#   иначе               → NEUTRAL

import pytest
from mozart_config import MOZART_CONFIG
from mozart_signals import classify_btc_dominance_trend

# Нейтральный плейсхолдер для базового BTC.D (не API-реалистичное число)
_BASE_PCT = 50.0


# ---------------------------------------------------------------------------
# Контракт конфига — ключи и типы
# ---------------------------------------------------------------------------

class TestBtcDominanceConfig:

    def test_threshold_key_exists(self):
        assert "btc_dominance_rotation_threshold_pct" in MOZART_CONFIG
        # WHY: classify_btc_dominance_trend читает этот ключ;
        #   отсутствие → KeyError при каждом вызове оркестратора.

    def test_threshold_is_numeric(self):
        val = MOZART_CONFIG["btc_dominance_rotation_threshold_pct"]
        assert isinstance(float(val), float)
        # WHY: функция делает float-сравнение delta с порогом;
        #   str-значение → TypeError или молчаливое неверное сравнение.

    def test_threshold_is_positive(self):
        val = float(MOZART_CONFIG["btc_dominance_rotation_threshold_pct"])
        assert val > 0
        # WHY: порог <= 0 означает ЛЮБОЕ изменение BTC.D = ROTATION,
        #   NEUTRAL становится недостижимым → каждый тик сигнализирует ложно.


# ---------------------------------------------------------------------------
# Контракт типа возврата
# ---------------------------------------------------------------------------

class TestBtcDominanceReturnType:

    def test_returns_str(self):
        result = classify_btc_dominance_trend(_BASE_PCT, _BASE_PCT)
        assert isinstance(result, str)
        # WHY: signal_polarity() и build_alignment() сравнивают строку с таблицей;
        #   не-str → TypeError или отсутствие метки в таблице полярности.


# ---------------------------------------------------------------------------
# Зоны — типичные значения (центры диапазонов)
# ---------------------------------------------------------------------------

class TestClassifyBtcDominanceTrendZones:

    def test_rotation_altcoin_when_btcd_drops_significantly(self):
        threshold = float(MOZART_CONFIG["btc_dominance_rotation_threshold_pct"])
        # delta = -(threshold + 1.0) < -threshold → ROTATION_ALTCOIN
        current = _BASE_PCT
        prev_30d = _BASE_PCT + threshold + 1.0
        result = classify_btc_dominance_trend(current, prev_30d)
        assert result == "ROTATION_ALTCOIN"
        # WHY: Mozart (пост 10.05.2026): снижение BTC.D = возврат риск-аппетита;
        #   неверный лейбл → alignment не видит BULLISH-сигнал ротации.

    def test_rotation_btc_when_btcd_rises_significantly(self):
        threshold = float(MOZART_CONFIG["btc_dominance_rotation_threshold_pct"])
        # delta = +(threshold + 1.0) > +threshold → ROTATION_BTC
        prev_30d = _BASE_PCT
        current = _BASE_PCT + threshold + 1.0
        result = classify_btc_dominance_trend(current, prev_30d)
        assert result == "ROTATION_BTC"
        # WHY: рост BTC.D = ликвидность уходит из альтов в BTC;
        #   неверный лейбл → alignment пропускает медвежий сигнал для альт-сезона.

    def test_neutral_when_change_is_small(self):
        threshold = float(MOZART_CONFIG["btc_dominance_rotation_threshold_pct"])
        # delta = -(threshold - 0.5): по модулю меньше threshold → NEUTRAL
        current = _BASE_PCT
        prev_30d = _BASE_PCT + (threshold - 0.5)
        result = classify_btc_dominance_trend(current, prev_30d)
        assert result == "NEUTRAL"
        # WHY: шум внутри порога не должен триггерить rotation-сигнал;
        #   иначе alignment получает ложные сигналы при незначимых колебаниях BTC.D.

    def test_neutral_when_no_change(self):
        result = classify_btc_dominance_trend(_BASE_PCT, _BASE_PCT)
        assert result == "NEUTRAL"
        # WHY: нулевое изменение BTC.D = нет ротации; NEUTRAL ≠ ROTATION_ALTCOIN.


# ---------------------------------------------------------------------------
# Граничные значения — отдельные тесты
# ---------------------------------------------------------------------------

class TestClassifyBtcDominanceTrendBoundaries:

    def test_boundary_drop_exactly_threshold_is_neutral(self):
        threshold = float(MOZART_CONFIG["btc_dominance_rotation_threshold_pct"])
        # delta ровно == -threshold → NEUTRAL (Mozart: «снижение > 2%» — строгий >)
        current = _BASE_PCT
        prev_30d = _BASE_PCT + threshold        # delta = -threshold точно
        result = classify_btc_dominance_trend(current, prev_30d)
        assert result == "NEUTRAL"
        # WHY: Mozart формулирует «> 2%» — строгое неравенство.
        #   Ровно на пороге = сигнал НЕ активирован; ошибка → ложный ROTATION_ALTCOIN
        #   при каждом движении BTC.D ровно на 2 п.п. (частое рыночное значение).

    def test_boundary_rise_exactly_threshold_is_neutral(self):
        threshold = float(MOZART_CONFIG["btc_dominance_rotation_threshold_pct"])
        # delta ровно == +threshold → NEUTRAL (аналогичная логика строгого >)
        prev_30d = _BASE_PCT
        current = _BASE_PCT + threshold         # delta = +threshold точно
        result = classify_btc_dominance_trend(current, prev_30d)
        assert result == "NEUTRAL"
        # WHY: симметричная граница для ROTATION_BTC.
        #   Ошибка → ложный ROTATION_BTC при движении ровно +2 п.п.

    def test_boundary_just_below_threshold_drop_is_neutral(self):
        threshold = float(MOZART_CONFIG["btc_dominance_rotation_threshold_pct"])
        epsilon = 0.001
        current = _BASE_PCT
        prev_30d = _BASE_PCT + threshold - epsilon  # delta = -(threshold - eps) → NEUTRAL
        result = classify_btc_dominance_trend(current, prev_30d)
        assert result == "NEUTRAL"
        # WHY: delta на eps меньше порога — внутри нейтральной зоны;
        #   любое значение ниже строгого порога = NEUTRAL.

    def test_boundary_just_above_threshold_drop_is_rotation_altcoin(self):
        threshold = float(MOZART_CONFIG["btc_dominance_rotation_threshold_pct"])
        epsilon = 0.001
        current = _BASE_PCT
        prev_30d = _BASE_PCT + threshold + epsilon  # delta = -(threshold + eps) → ROTATION_ALTCOIN
        result = classify_btc_dominance_trend(current, prev_30d)
        assert result == "ROTATION_ALTCOIN"
        # WHY: delta на eps больше порога — сигнал должен активироваться;
        #   граница — самое частое место тихих багов ≤ vs <.
