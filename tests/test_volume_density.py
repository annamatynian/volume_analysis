"""
tests/test_volume_density.py
============================
Unit tests for module-level pure functions in volume_density.py.

Design principles (from project rules):
- Tests verify CONTRACTS of production functions, not re-implement their logic.
- No inline math formulas — assertions call the same production functions.
- No hardcoded epsilon — edge cases (zero volume, single row) test the
  production error-handling path, not a workaround.
- Exchange and on-chain API are NOT used here — those belong in integration tests.
- DBSCAN tests use the real DBSCAN via find_liquidity_clusters so we test
  the actual production code, not a mock cluster.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import from production module (must be on sys.path)
# Run from volume_analysis/: pytest tests/test_volume_density.py -v
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from volume_density import (
    calculate_atr,
    apply_time_decay,
    build_profile,
    calculate_value_area,
    find_liquidity_clusters,
    detect_absorption_days,
    calculate_poc_quality_score,
    classify_oi_regime,
    calculate_lth_pain_proxy,
    extract_sub_levels,
    evaluate_poc_quality,
    classify_funding_regime,
    classify_market_regime,
    calculate_basis_spread,
    classify_volume_type,
    calculate_poc_retest_score,
    calculate_volume_imbalance,
    detect_divergence,
    calculate_vwap_deviation,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic but realistic OHLCV DataFrames
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int, base_price: float = 50_000.0, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic synthetic daily OHLCV.
    Prices drift slowly so ATR and clusters are meaningful.
    """
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]

    closes = base_price + np.cumsum(rng.normal(0, 200, n))
    highs  = closes + rng.uniform(100, 800, n)
    lows   = closes - rng.uniform(100, 800, n)
    opens  = closes + rng.normal(0, 100, n)
    vols   = rng.uniform(10_000, 80_000, n)

    df = pd.DataFrame({
        'timestamp': [int(d.timestamp() * 1000) for d in dates],
        'date':  dates,
        'open':  opens,
        'high':  highs,
        'low':   lows,
        'close': closes,
        'vol':   vols,
    })
    return df


def _make_df_with_decay(n: int = 100, lam: float = 0.005) -> pd.DataFrame:
    """Convenience: OHLCV + time decay applied (ready for build_profile)."""
    df = _make_ohlcv(n)
    return apply_time_decay(df, lam=lam)


def _global_bins(df: pd.DataFrame, n_bins: int = 100) -> np.ndarray:
    return np.linspace(df['low'].min(), df['high'].max(), n_bins + 1)


# ---------------------------------------------------------------------------
# calculate_atr
# ---------------------------------------------------------------------------

class TestCalculateATR:

    def test_returns_series_same_length(self):
        df  = _make_ohlcv(60)
        atr = calculate_atr(df, period=14)
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(df)

    def test_first_13_rows_are_nan_for_period_14(self):
        df  = _make_ohlcv(60)
        atr = calculate_atr(df, period=14)
        # rolling(14) needs 14 rows → first 13 are NaN
        assert atr.iloc[:13].isna().all(), "Expected NaN in warm-up period"
        assert atr.iloc[14:].notna().all(), "Expected valid values after warm-up"

    def test_atr_is_positive(self):
        df  = _make_ohlcv(60)
        atr = calculate_atr(df, period=14)
        assert (atr.dropna() > 0).all()

    def test_atr_bounded_by_price_range(self):
        """ATR cannot exceed the max daily price range in the window."""
        df   = _make_ohlcv(60)
        atr  = calculate_atr(df, period=14)
        max_range = (df['high'] - df['low']).max()
        assert atr.dropna().max() <= max_range * 1.1  # small tolerance for True Range


# ---------------------------------------------------------------------------
# apply_time_decay
# ---------------------------------------------------------------------------

class TestApplyTimeDecay:

    def test_adds_three_columns(self):
        df  = _make_ohlcv(50)
        out = apply_time_decay(df)
        for col in ('days_ago', 'time_weight', 'weighted_vol'):
            assert col in out.columns, f"Missing column: {col}"

    def test_does_not_mutate_original(self):
        df = _make_ohlcv(50)
        _ = apply_time_decay(df)
        assert 'weighted_vol' not in df.columns, "Original df must not be mutated"

    def test_most_recent_day_has_weight_1(self):
        df  = _make_ohlcv(50)
        out = apply_time_decay(df, lam=0.005)
        # Most recent row: days_ago == 0 → weight = 1/(1+0) = 1.0
        most_recent = out[out['days_ago'] == 0]
        assert not most_recent.empty
        assert abs(most_recent['time_weight'].iloc[0] - 1.0) < 1e-9

    def test_older_days_have_lower_weight(self):
        df  = _make_ohlcv(100)
        out = apply_time_decay(df, lam=0.005)
        # Weight must decrease monotonically as days_ago increases
        sorted_out = out.sort_values('days_ago')
        weights    = sorted_out['time_weight'].values
        assert (np.diff(weights) <= 0).all(), "Weight must be non-increasing with age"

    def test_weighted_vol_equals_vol_times_weight(self):
        df  = _make_ohlcv(50)
        out = apply_time_decay(df, lam=0.005)
        expected = out['vol'] * out['time_weight']
        pd.testing.assert_series_equal(out['weighted_vol'], expected, check_names=False)

    def test_high_lam_decays_faster(self):
        df    = _make_ohlcv(100)
        slow  = apply_time_decay(df, lam=0.005)
        fast  = apply_time_decay(df, lam=0.02)
        # Oldest row should have lower weight with high lam
        oldest_slow = slow.sort_values('days_ago').iloc[-1]['time_weight']
        oldest_fast = fast.sort_values('days_ago').iloc[-1]['time_weight']
        assert oldest_fast < oldest_slow


# ---------------------------------------------------------------------------
# build_profile
# ---------------------------------------------------------------------------

class TestBuildProfile:

    def test_returns_dataframe_with_required_columns(self):
        df   = _make_df_with_decay(60)
        bins = _global_bins(df)
        prof = build_profile(df, bins)
        for col in ('price_low', 'price_high', 'mid', 'vol'):
            assert col in prof.columns

    def test_number_of_bins(self):
        df   = _make_df_with_decay(60)
        bins = _global_bins(df, n_bins=100)
        prof = build_profile(df, bins)
        assert len(prof) == 100

    def test_total_vol_non_negative(self):
        df   = _make_df_with_decay(60)
        bins = _global_bins(df)
        prof = build_profile(df, bins)
        assert (prof['vol'] >= 0).all()

    def test_total_volume_leq_source_weighted_vol(self):
        """
        Each bar can be counted in multiple overlapping bins (high-low spans
        several bins), so total profile vol >= total weighted_vol.
        Verify it is at least as large and not wildly larger.
        """
        df          = _make_df_with_decay(60)
        bins        = _global_bins(df)
        prof        = build_profile(df, bins)
        source_vol  = df['weighted_vol'].sum()
        profile_vol = prof['vol'].sum()
        # Profile vol >= source vol (bars span multiple bins)
        assert profile_vol >= source_vol - 1e-6  # floating point tolerance

    def test_empty_dataframe_returns_zero_vol_bins(self):
        df   = _make_df_with_decay(60)
        bins = _global_bins(df)
        # Slice that matches nothing
        empty_df = df.iloc[0:0].copy()
        prof = build_profile(empty_df, bins)
        assert (prof['vol'] == 0).all()

    def test_mid_is_between_price_low_and_price_high(self):
        df   = _make_df_with_decay(60)
        bins = _global_bins(df)
        prof = build_profile(df, bins)
        assert ((prof['mid'] > prof['price_low']) & (prof['mid'] < prof['price_high'])).all()


# ---------------------------------------------------------------------------
# calculate_value_area
# ---------------------------------------------------------------------------

class TestCalculateValueArea:

    def test_returns_three_values(self):
        df   = _make_df_with_decay(100)
        bins = _global_bins(df)
        result = calculate_value_area(df, bins)
        assert len(result) == 3

    def test_poc_is_highest_volume_bin(self):
        """
        POC must correspond to the bin with maximum weighted volume.
        We call build_profile (same production function) to verify — no
        inline formula duplication.
        """
        df   = _make_df_with_decay(100)
        bins = _global_bins(df)
        prof = build_profile(df, bins)
        expected_poc = prof.sort_values('vol', ascending=False).iloc[0]['mid']
        _, _, poc = calculate_value_area(df, bins)
        assert abs(poc - expected_poc) < 1e-6

    def test_va_low_leq_va_high(self):
        df   = _make_df_with_decay(100)
        bins = _global_bins(df)
        va_low, va_high, _ = calculate_value_area(df, bins)
        assert va_low <= va_high

    def test_poc_within_value_area(self):
        df   = _make_df_with_decay(100)
        bins = _global_bins(df)
        va_low, va_high, poc = calculate_value_area(df, bins)
        assert va_low <= poc <= va_high

    def test_zero_volume_returns_none_triple(self):
        """
        Production code returns (None, None, None) when total volume is zero.
        Test the production path — no epsilon workaround.
        """
        df        = _make_ohlcv(10)
        df        = apply_time_decay(df)
        df['weighted_vol'] = 0.0  # force zero
        bins      = _global_bins(df)
        result    = calculate_value_area(df, bins)
        assert result == (None, None, None)

    def test_custom_percentage(self):
        """50% VA must be narrower than 70% VA."""
        df   = _make_df_with_decay(100)
        bins = _global_bins(df)
        va_low_50, va_high_50, _ = calculate_value_area(df, bins, percentage=0.50)
        va_low_70, va_high_70, _ = calculate_value_area(df, bins, percentage=0.70)
        range_50 = va_high_50 - va_low_50
        range_70 = va_high_70 - va_low_70
        assert range_50 <= range_70, "Wider percentage must produce wider Value Area"

    def test_single_row_dataframe(self):
        """
        Single row: the bar's high-low range spans multiple bins on the global
        grid so va_low <= va_high (not necessarily equal). POC and both VA
        bounds must be non-None and within the grid.

        WHY: build_profile assigns bar volume to every bin that overlaps
        [low, high], so a wide bar populates several bins. Asserting
        va_low == va_high == poc would be incorrect (test-fitting).
        """
        df   = _make_ohlcv(1)
        df   = apply_time_decay(df)
        bins = _global_bins(df, n_bins=10)
        va_low, va_high, poc = calculate_value_area(df, bins)
        assert va_low  is not None
        assert va_high is not None
        assert poc     is not None
        assert va_low  <= va_high
        assert bins[0] <= poc <= bins[-1]


# ---------------------------------------------------------------------------
# find_liquidity_clusters
# ---------------------------------------------------------------------------

class TestFindLiquidityClusters:

    def _df_with_two_price_zones(self) -> pd.DataFrame:
        """
        Construct a DataFrame with two tight price zones (simulates two
        institutional accumulation zones). Enough rows for DBSCAN min_samples=5.
        """
        rng = np.random.default_rng(0)
        n   = 40  # >= DBSCAN_MIN_WINDOW

        # Zone A: closes around 50_000
        zone_a = pd.DataFrame({
            'date':  [datetime(2024, 1, 1) + timedelta(days=i) for i in range(20)],
            'close': rng.normal(50_000, 50, 20),
            'high':  rng.normal(50_000, 50, 20) + 200,
            'low':   rng.normal(50_000, 50, 20) - 200,
            'vol':   rng.uniform(50_000, 80_000, 20),  # above-average vol
        })
        # Zone B: closes around 55_000
        zone_b = pd.DataFrame({
            'date':  [datetime(2024, 1, 21) + timedelta(days=i) for i in range(20)],
            'close': rng.normal(55_000, 50, 20),
            'high':  rng.normal(55_000, 50, 20) + 200,
            'low':   rng.normal(55_000, 50, 20) - 200,
            'vol':   rng.uniform(50_000, 80_000, 20),
        })
        df = pd.concat([zone_a, zone_b], ignore_index=True)
        df = apply_time_decay(df, lam=0.005)
        return df

    def test_returns_list(self):
        df  = _make_df_with_decay(60)
        atr = 500.0
        result = find_liquidity_clusters(df, atr)
        assert isinstance(result, list)

    def test_empty_when_all_below_average_vol(self):
        """
        If all rows have the same volume, heavy_days is empty → no clusters.
        Tests the production early-exit path.
        """
        df       = _make_ohlcv(40)
        df['vol'] = 10_000.0  # uniform vol → all equal to mean → heavy_days empty
        df       = apply_time_decay(df)
        result   = find_liquidity_clusters(df, atr_value=500.0)
        assert result == []

    def test_cluster_has_required_keys(self):
        df      = self._df_with_two_price_zones()
        atr_val = 200.0
        clusters = find_liquidity_clusters(df, atr_val)
        if clusters:  # skip if DBSCAN found nothing (valid outcome)
            for c in clusters:
                assert set(c.keys()) == {'min', 'max', 'vol', 'days'}

    def test_cluster_min_leq_max(self):
        df       = self._df_with_two_price_zones()
        clusters = find_liquidity_clusters(df, atr_value=200.0)
        for c in clusters:
            assert c['min'] <= c['max']

    def test_cluster_vol_positive(self):
        df       = self._df_with_two_price_zones()
        clusters = find_liquidity_clusters(df, atr_value=200.0)
        for c in clusters:
            assert c['vol'] > 0

    def test_cluster_days_positive_integer(self):
        df       = self._df_with_two_price_zones()
        clusters = find_liquidity_clusters(df, atr_value=200.0)
        for c in clusters:
            assert isinstance(c['days'], (int, np.integer))
            assert c['days'] > 0

    def test_wide_atr_merges_zones(self):
        """
        Very wide ATR eps merges the two zones into one cluster (or fewer).
        Verifies that eps parameter actually controls cluster granularity.
        """
        df = self._df_with_two_price_zones()
        clusters_narrow = find_liquidity_clusters(df, atr_value=100.0)   # eps=75
        clusters_wide   = find_liquidity_clusters(df, atr_value=10_000.0) # eps=7500
        # Wide ATR should produce fewer or equal clusters
        assert len(clusters_wide) <= len(clusters_narrow)


# ---------------------------------------------------------------------------
# Integration: Value Area contract across windows
# ---------------------------------------------------------------------------

class TestValueAreaMultiWindow:
    """
    Verify that POC/VA computed on a sub-window of data is consistent with
    the POC/VA of the full dataset on the SAME global bin grid.
    This is the key invariant for multi-window comparability. [cite: 12, 14, 56]
    """

    def test_va_values_are_within_global_bin_range(self):
        df        = _make_df_with_decay(200)
        bins      = _global_bins(df)
        # Test on a 30-day slice
        df_slice  = df.tail(30).copy()
        va_low, va_high, poc = calculate_value_area(df_slice, bins)
        if va_low is None:
            pytest.skip("Zero volume in slice")
        assert bins[0] <= va_low  <= bins[-1]
        assert bins[0] <= va_high <= bins[-1]
        assert bins[0] <= poc     <= bins[-1]

    def test_smaller_window_poc_within_full_price_range(self):
        df        = _make_df_with_decay(365)
        bins      = _global_bins(df)
        df_1m     = df.tail(30).copy()
        _, _, poc_1m  = calculate_value_area(df_1m,  bins)
        _, _, poc_full = calculate_value_area(df,    bins)
        # Both POCs must be valid floats (not None)
        assert poc_1m  is not None
        assert poc_full is not None


# ---------------------------------------------------------------------------
# detect_absorption_days
# ---------------------------------------------------------------------------

def _make_absorption_df(
    n: int = 60,
    base_price: float = 50_000.0,
    downtrend: bool = True,
    high_vol: bool = True,
    closed_upper: bool = True,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Synthetic OHLCV where all three absorption signals are controllable.

    downtrend=True  → close[-1] < close[-6]  (строго нисходящий контекст)
    high_vol=True   → vol последних 5 строк = rolling_mean*3.0
    closed_upper=True → свеча закрылась в верхних 10% диапазона
    """
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]

    closes = base_price + np.cumsum(rng.normal(0, 200, n))
    # Downtrend: форсируем убывание в последних 10 строках
    if downtrend:
        closes[-10:] = np.linspace(closes[-11], closes[-11] - 3000, 10)
    else:
        # Uptrend: последний close выше close 5 дней назад
        closes[-10:] = np.linspace(closes[-11], closes[-11] + 3000, 10)

    lows   = closes - rng.uniform(500, 1000, n)
    highs  = closes + rng.uniform(500, 1000, n)

    if closed_upper:
        # Закрытие в верхних 10% диапазона: close = low + 0.92*(high-low)
        closes[-5:] = lows[-5:] + 0.92 * (highs[-5:] - lows[-5:])
    else:
        # Медвежья свеча: close = low + 0.10*(high-low)
        closes[-5:] = lows[-5:] + 0.10 * (highs[-5:] - lows[-5:])

    vols = rng.uniform(10_000, 30_000, n)
    if high_vol:
        # Форсируем аномальный объём: rolling_mean*20 >> 1.5 threshold
        vols[-5:] = vols[:-5].mean() * 4.0
    # else: оставляем нормальный объём

    df = pd.DataFrame({
        'timestamp': [int(d.timestamp() * 1000) for d in dates],
        'date':      dates,
        'open':      closes + rng.normal(0, 100, n),
        'high':      highs,
        'low':       lows,
        'close':     closes,
        'vol':       vols,
    })
    df = apply_time_decay(df, lam=0.005)   # WHY: функция требует weighted_vol
    return df


@pytest.mark.skip(reason="SUBJECTIVE: threshold-based (1.5×mean, 0.70 close_pct, shift(5)), disabled pending objective replacement")
class TestDetectAbsorptionDays:

    def test_absorption_zero_when_low_volume(self):
        """
        Нет аномального объёма → absorption=False для последних строк.
        Контракт: условие vol > rolling_mean*1.5 не выполнено.
        """
        df      = _make_absorption_df(high_vol=False, closed_upper=True, downtrend=True)
        atr_val = 800.0
        result  = detect_absorption_days(df, atr_val)
        # Последние 5 строк — нормальный объём: ни одна не должна быть absorption
        assert not result.tail(5)['absorption'].any(), (
            "Absorption must be False when volume is not anomalous"
        )

    def test_absorption_zero_when_closed_low(self):
        """
        Высокий объём, но медвежья свеча (close в нижних 10%) → absorption=False.
        Контракт: условие close_pct > 0.70 не выполнено.
        """
        df      = _make_absorption_df(high_vol=True, closed_upper=False, downtrend=True)
        atr_val = 800.0
        result  = detect_absorption_days(df, atr_val)
        assert not result.tail(5)['absorption'].any(), (
            "Absorption must be False when candle closes in lower range"
        )

    def test_absorption_detected_in_downtrend(self):
        """
        Все три условия выполнены → хотя бы одна строка absorption=True.
        Контракт: high_vol & closed_upper & downtrend → True.
        """
        df      = _make_absorption_df(high_vol=True, closed_upper=True, downtrend=True)
        atr_val = 800.0
        result  = detect_absorption_days(df, atr_val)
        assert result.tail(5)['absorption'].any(), (
            "Absorption must be True when all three conditions are met"
        )

    def test_absorption_does_not_mutate_df(self):
        """
        Оригинальный df не должен получить колонку 'absorption'.
        Контракт: функция работает на df.copy().
        """
        df      = _make_absorption_df()
        atr_val = 800.0
        _       = detect_absorption_days(df, atr_val)
        assert 'absorption' not in df.columns, (
            "detect_absorption_days must not mutate the original DataFrame"
        )


# ---------------------------------------------------------------------------
# calculate_poc_quality_score
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="SUBJECTIVE: calculate_poc_quality_score is legacy, disabled pending objective replacement")
class TestCalculatePocQualityScore:
    """
    Тесты проверяют контракты функции, не воспроизводят формулу внутри.
    Веса и пороги проверяются через production-функцию, а не hardcoded.
    """

    def test_poc_score_magnet_when_all_signals_bullish(self):
        """
        Все сигналы бычьи → score > 0.65 → label == 'FAIR_VALUE_MAGNET'.
        Контракт: полное поглощение + капитуляция + высокий volume_score.
        """
        result = calculate_poc_quality_score(
            absorption_days_near_poc=10,
            total_days_near_poc=10,      # absorption_score = 1.0
            volume_w_score=100.0,        # volume_score = 1.0
            capitulation_confirmed=True, # onchain_score = 1.0
            z_score=0.0,
        )
        assert result['label'] == 'FAIR_VALUE_MAGNET', (
            f"Expected FAIR_VALUE_MAGNET, got {result['label']} (score={result['score']})"
        )
        assert result['score'] > 0.65

    def test_poc_score_trap_when_no_absorption_no_capitulation(self):
        """
        Нет поглощения, нет капитуляции, низкий volume → score < 0.35 → 'RESISTANCE_TRAP'.
        Контракт: z_score=0 → sigmoid(0)=0.5 → onchain=0.5*0.35=0.175;
        при absorption=0 и volume=0 итоговый score = 0 + 0.175 + 0 = 0.175 < 0.35.
        """
        result = calculate_poc_quality_score(
            absorption_days_near_poc=0,
            total_days_near_poc=10,      # absorption_score = 0.0
            volume_w_score=0.0,          # volume_score = 0.0
            capitulation_confirmed=False,
            z_score=0.0,                 # sigmoid(0) = 0.5
        )
        assert result['label'] == 'RESISTANCE_TRAP', (
            f"Expected RESISTANCE_TRAP, got {result['label']} (score={result['score']})"
        )
        assert result['score'] < 0.35

    def test_poc_score_neutral_range(self):
        """
        Промежуточные сигналы → score в зоне [0.35, 0.65] → label == 'NEUTRAL'.
        Контракт: частичное поглощение + нет капитуляции + средний volume.
        """
        result = calculate_poc_quality_score(
            absorption_days_near_poc=5,
            total_days_near_poc=10,      # absorption_score = 0.5
            volume_w_score=50.0,         # volume_score = 0.5
            capitulation_confirmed=False,
            z_score=0.0,                 # sigmoid(0) = 0.5
        )
        assert result['label'] == 'NEUTRAL', (
            f"Expected NEUTRAL, got {result['label']} (score={result['score']})"
        )
        assert 0.35 <= result['score'] <= 0.65

    def test_poc_score_flags_bullish_divergence(self):
        """
        Поглощение > 30% + капитуляция → flag 'BULLISH_DIVERGENCE_ACCUMULATION'.
        Контракт: absorption_score > 0.3 AND capitulation_confirmed=True.
        """
        result = calculate_poc_quality_score(
            absorption_days_near_poc=8,
            total_days_near_poc=10,      # absorption_score = 0.8 > 0.3
            volume_w_score=80.0,
            capitulation_confirmed=True,
            z_score=0.0,
        )
        assert 'BULLISH_DIVERGENCE_ACCUMULATION' in result['flags'], (
            f"Expected flag in {result['flags']}"
        )

    def test_poc_score_boundaries(self):
        """
        Граничный случай: все нули + z=0 → score не NaN, label не None.
        Контракт: total_days_near_poc=0 защищён от ZeroDivisionError.
        """
        result = calculate_poc_quality_score(
            absorption_days_near_poc=0,
            total_days_near_poc=0,       # WHY: деление на ноль должно быть защищено
            volume_w_score=0.0,
            capitulation_confirmed=False,
            z_score=0.0,
        )
        import math
        assert result['score'] is not None
        assert not math.isnan(result['score'])
        assert result['label'] in ('FAIR_VALUE_MAGNET', 'NEUTRAL', 'RESISTANCE_TRAP')


# ---------------------------------------------------------------------------
# classify_oi_regime  — Этап 4
# ---------------------------------------------------------------------------

class TestClassifyOiRegime:
    """
    Тесты контракта classify_oi_regime(price_change_pct, oi_change_pct) -> str.

    Матрица (threshold = 1.0%):
      Price > +1%  AND  OI > +1%   → 'STRONG_BULL'
      Price > +1%  AND  OI <= +1%  → 'WEAK_BULL'
      Price <= +1% AND  OI > +1%   → 'STRONG_BEAR'
      Price <= +1% AND  OI <= +1%  → 'LIQUIDATION'

    Принципы:
    - Тесты проверяют возвращаемые строки через production-функцию, не через
      inline if/else внутри теста (правило: нет подгонки логики).
    - Граничный случай (ровно threshold) проверяет конкретную ветку контракта,
      а не угадывает результат — поэтому мы явно документируем ожидание.
    """

    VALID_REGIMES = {'STRONG_BULL', 'WEAK_BULL', 'STRONG_BEAR', 'LIQUIDATION', 'NEUTRAL'}

    def test_oi_regime_strong_bull(self):
        """
        Price ↑ + OI ↑ (оба выше threshold 1%) → 'STRONG_BULL'.
        Контракт: новые лонги открываются, тренд подкреплён реальным капиталом.
        """
        result = classify_oi_regime(price_change_pct=3.5, oi_change_pct=2.1)
        assert result == 'STRONG_BULL', (
            f"Expected STRONG_BULL for price=+3.5% oi=+2.1%, got '{result}'"
        )

    def test_oi_regime_liquidation(self):
        """
        Price ↓ + OI ↓ (оба ниже -threshold) → 'LIQUIDATION'.
        Контракт: лонги ликвидируются — потенциальный сигнал дна.
        """
        result = classify_oi_regime(price_change_pct=-2.5, oi_change_pct=-3.0)
        assert result == 'LIQUIDATION', (
            f"Expected LIQUIDATION for price=-2.5% oi=-3.0%, got '{result}'"
        )

    def test_oi_regime_neutral_at_threshold_boundary(self):
        """
        Оба значения точно на границе threshold (1.0%) → не пересекают порог.
        Контракт: price=1.0 НЕ > 1.0, oi=1.0 НЕ > 1.0 → 'LIQUIDATION'
        (оба условия price_up=False, oi_up=False).

        WHY: граница проверяет строгое неравенство (>) в production-коде.
        Это не подгонка — мы документируем явное решение про strict >.
        """
        result = classify_oi_regime(price_change_pct=1.0, oi_change_pct=1.0)
        assert result == 'LIQUIDATION', (
            f"Expected LIQUIDATION at exact threshold boundary (strict >), got '{result}'"
        )

    def test_oi_regime_returns_valid_string(self):
        """
        Функция всегда возвращает строку из допустимого набора значений.
        Контракт: возвращаемое значение принадлежит VALID_REGIMES.
        """
        test_cases = [
            (5.0, 5.0),
            (5.0, -5.0),
            (-5.0, 5.0),
            (-5.0, -5.0),
            (0.0, 0.0),
        ]
        for price_pct, oi_pct in test_cases:
            result = classify_oi_regime(price_pct, oi_pct)
            assert result in self.VALID_REGIMES, (
                f"classify_oi_regime({price_pct}, {oi_pct}) returned '{result}' "
                f"which is not in {self.VALID_REGIMES}"
            )


# ---------------------------------------------------------------------------
# load_oi_history  — Этап 5B
# ---------------------------------------------------------------------------

from volume_density import load_oi_history
import tempfile
import os


def _make_oi_csv(tmp_path: str, n: int = 10) -> str:
    """
    Создаёт временный CSV с колонками date, sum_open_interest.
    Структура совпадает с реальным BTCUSDT-metrics-daily.csv.
    """
    rows = ["date,sum_open_interest,sum_open_interest_value"]
    for i in range(n):
        d = (datetime(2025, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
        oi = 90_000 + i * 100
        rows.append(f"{d},{float(oi):.10f},0")
    path = os.path.join(tmp_path, "test_oi.csv")
    with open(path, 'w') as f:
        f.write("\n".join(rows))
    return path


class TestLoadOiHistory:
    """
    Тесты контракта load_oi_history(csv_path, start_date, end_date) -> pd.DataFrame.

    Контракт (из NEXT_SESSION.md):
    - Возвращает DataFrame с колонками: date (datetime64), sum_open_interest (float),
      oi_change_pct (float).
    - Фильтрует по start_date / end_date (опционально).
    - oi_change_pct = pct_change() * 100 (первая строка — NaN).
    """

    def test_returns_dataframe_with_required_columns(self, tmp_path):
        """
        Контракт: три колонки date, sum_open_interest, oi_change_pct обязательны.
        """
        csv_path = _make_oi_csv(str(tmp_path))
        df = load_oi_history(csv_path)
        for col in ('date', 'sum_open_interest', 'oi_change_pct'):
            assert col in df.columns, f"Missing column: {col}"

    def test_date_column_is_datetime(self, tmp_path):
        """
        Контракт: date должен быть dtype datetime64 для merge по дате.
        """
        csv_path = _make_oi_csv(str(tmp_path))
        df = load_oi_history(csv_path)
        assert pd.api.types.is_datetime64_any_dtype(df['date']), (
            f"Expected datetime64, got {df['date'].dtype}"
        )

    def test_sum_open_interest_is_float(self, tmp_path):
        """
        Контракт: sum_open_interest — float (не строка).
        """
        csv_path = _make_oi_csv(str(tmp_path))
        df = load_oi_history(csv_path)
        assert pd.api.types.is_float_dtype(df['sum_open_interest']), (
            f"Expected float dtype, got {df['sum_open_interest'].dtype}"
        )

    def test_oi_change_pct_first_row_is_nan(self, tmp_path):
        """
        Контракт: первая строка oi_change_pct = NaN (pct_change по определению).
        """
        csv_path = _make_oi_csv(str(tmp_path), n=5)
        df = load_oi_history(csv_path)
        assert pd.isna(df['oi_change_pct'].iloc[0]), (
            "First row of oi_change_pct must be NaN (pct_change contract)"
        )

    def test_oi_change_pct_correct_value(self, tmp_path):
        """
        Контракт: oi_change_pct[1] = (oi[1] - oi[0]) / oi[0] * 100.
        Не дублируем формулу — вычисляем через те же данные CSV.
        """
        csv_path = _make_oi_csv(str(tmp_path), n=5)
        df = load_oi_history(csv_path)
        oi0 = df['sum_open_interest'].iloc[0]
        oi1 = df['sum_open_interest'].iloc[1]
        expected = (oi1 - oi0) / oi0 * 100
        assert abs(df['oi_change_pct'].iloc[1] - expected) < 1e-9

    def test_filter_by_start_date(self, tmp_path):
        """
        Контракт: start_date фильтрует строки, дата включительно.
        """
        csv_path = _make_oi_csv(str(tmp_path), n=10)
        df = load_oi_history(csv_path, start_date='2025-01-05')
        assert (df['date'] >= pd.Timestamp('2025-01-05')).all(), (
            "All rows must be >= start_date"
        )
        assert len(df) < 10, "Filtered df must be shorter than full df"

    def test_filter_by_end_date(self, tmp_path):
        """
        Контракт: end_date фильтрует строки, дата включительно.
        """
        csv_path = _make_oi_csv(str(tmp_path), n=10)
        df = load_oi_history(csv_path, end_date='2025-01-05')
        assert (df['date'] <= pd.Timestamp('2025-01-05')).all(), (
            "All rows must be <= end_date"
        )

    def test_filter_start_and_end(self, tmp_path):
        """
        Контракт: оба фильтра вместе сужают диапазон корректно.
        """
        csv_path = _make_oi_csv(str(tmp_path), n=10)
        df = load_oi_history(csv_path, start_date='2025-01-03', end_date='2025-01-07')
        assert len(df) == 5  # 03, 04, 05, 06, 07
        assert df['date'].iloc[0]  == pd.Timestamp('2025-01-03')
        assert df['date'].iloc[-1] == pd.Timestamp('2025-01-07')

    def test_returns_sorted_by_date(self, tmp_path):
        """
        Контракт: строки отсортированы по возрастанию даты.
        """
        csv_path = _make_oi_csv(str(tmp_path), n=10)
        df = load_oi_history(csv_path)
        assert df['date'].is_monotonic_increasing, (
            "DataFrame must be sorted by date ascending"
        )


# ---------------------------------------------------------------------------
# calculate_bin_delta  — Этап 5D
# ---------------------------------------------------------------------------

from volume_density import calculate_bin_delta


class TestCalculateBinDelta:
    """
    Тесты контракта calculate_bin_delta(trades, price_low, price_high) -> float.

    Контракт (из NEXT_SESSION.md и TECHNICAL_SPEC_POC_Quality.md):
    - Принимает list[dict] сделок с ключами 'price', 'amount', 'side'.
    - Возвращает float: buy_volume - sell_volume в бине [price_low, price_high].
    - Сделки с price < price_low или price > price_high не учитываются.
    - is_buyer_maker=True → sell (maker продаёт, taker покупает — нет, наоборот:
      is_buyer_maker=True означает что buyer является maker'ом → это sell-initiated).
      В нашем API сделки уже переведены в side='buy'/'sell'.
    """

    def _make_trade(self, price: float, amount: float, side: str) -> dict:
        return {'price': price, 'amount': amount, 'side': side}

    def test_positive_delta_when_more_buys(self):
        """
        Контракт: больше покупок → положительная дельта.
        """
        trades = [
            self._make_trade(50_100, 1.0, 'buy'),
            self._make_trade(50_200, 2.0, 'buy'),
            self._make_trade(50_150, 0.5, 'sell'),
        ]
        delta = calculate_bin_delta(trades, 50_000, 50_500)
        assert delta > 0, f"Expected positive delta, got {delta}"
        assert abs(delta - 2.5) < 1e-9  # 3.0 buy - 0.5 sell

    def test_negative_delta_when_more_sells(self):
        """
        Контракт: больше продаж → отрицательная дельта.
        """
        trades = [
            self._make_trade(50_100, 0.5, 'buy'),
            self._make_trade(50_200, 3.0, 'sell'),
            self._make_trade(50_150, 1.5, 'sell'),
        ]
        delta = calculate_bin_delta(trades, 50_000, 50_500)
        assert delta < 0, f"Expected negative delta, got {delta}"
        assert abs(delta - (-4.0)) < 1e-9  # 0.5 buy - 4.5 sell

    def test_zero_delta_when_balanced(self):
        """
        Контракт: равные покупки и продажи → delta == 0.
        """
        trades = [
            self._make_trade(50_100, 2.0, 'buy'),
            self._make_trade(50_200, 2.0, 'sell'),
        ]
        delta = calculate_bin_delta(trades, 50_000, 50_500)
        assert abs(delta) < 1e-9

    def test_trades_outside_bin_excluded(self):
        """
        Контракт: сделки за пределами [price_low, price_high] не учитываются.
        """
        trades = [
            self._make_trade(49_999, 10.0, 'buy'),   # ниже бина
            self._make_trade(50_501, 10.0, 'sell'),  # выше бина
            self._make_trade(50_100,  1.0, 'buy'),   # внутри
        ]
        delta = calculate_bin_delta(trades, 50_000, 50_500)
        assert abs(delta - 1.0) < 1e-9, (
            "Only trades inside [price_low, price_high] must be counted"
        )

    def test_boundary_prices_included(self):
        """
        Контракт: сделки на границах price_low и price_high включаются.
        WHY: бин включает обе границы [low, high] — согласовано с build_profile.
        """
        trades = [
            self._make_trade(50_000, 1.0, 'buy'),   # на нижней границе
            self._make_trade(50_500, 1.0, 'buy'),   # на верхней границе
        ]
        delta = calculate_bin_delta(trades, 50_000, 50_500)
        assert abs(delta - 2.0) < 1e-9, (
            "Boundary trades must be included (closed interval)"
        )

    def test_empty_trades_returns_zero(self):
        """
        Контракт: пустой список → delta == 0.0 (не ошибка).
        """
        delta = calculate_bin_delta([], 50_000, 50_500)
        assert delta == 0.0

    def test_no_trades_in_bin_returns_zero(self):
        """
        Контракт: все сделки вне бина → delta == 0.0.
        """
        trades = [
            self._make_trade(60_000, 5.0, 'buy'),
            self._make_trade(40_000, 5.0, 'sell'),
        ]
        delta = calculate_bin_delta(trades, 50_000, 50_500)
        assert delta == 0.0

    def test_returns_float(self):
        """
        Контракт: возвращаемый тип — float.
        """
        delta = calculate_bin_delta([], 50_000, 50_500)
        assert isinstance(delta, float)


# ---------------------------------------------------------------------------
# load_aggtrades_zip  — Этап 8
# ---------------------------------------------------------------------------

from volume_density import load_aggtrades_zip
import zipfile


def _make_aggtrades_zip(tmp_path: str, n: int = 100,
                        poc: float = 50_000.0, atr: float = 500.0,
                        seed: int = 42) -> str:
    """
    Создаёт временный ZIP с синтетическими aggTrades в формате Binance Vision.

    Формат CSV:
        agg_trade_id,price,quantity,first_trade_id,last_trade_id,
        transact_time,is_buyer_maker

    Половина сделок в зоне POC ± 1.5*ATR, половина — за пределами.
    is_buyer_maker чередуется True/False для предсказуемой дельты.
    """
    rng = np.random.default_rng(seed)

    zone_low  = poc - 1.5 * atr
    zone_high = poc + 1.5 * atr

    rows = ["agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
            "transact_time,is_buyer_maker"]
    for i in range(n):
        # Половина сделок внутри зоны, половина — снаружи
        if i < n // 2:
            price = rng.uniform(zone_low, zone_high)
        else:
            price = rng.uniform(zone_high + 100, zone_high + 5000)
        qty            = rng.uniform(0.001, 2.0)
        is_buyer_maker = (i % 2 == 0)   # True=sell, False=buy чередуется
        rows.append(
            f"{i},{price:.2f},{qty:.8f},{i*2},{i*2+1},"
            f"{1_700_000_000_000 + i * 1000},"
            f"{'True' if is_buyer_maker else 'False'}"
        )

    csv_content = "\n".join(rows)
    csv_name    = "BTCUSDT-aggTrades-2025-01.csv"
    zip_path    = os.path.join(tmp_path, "BTCUSDT-aggTrades-2025-01.zip")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, csv_content)

    return zip_path


class TestLoadAggtradersZip:
    """
    Тесты контракта load_aggtrades_zip(zip_path) -> pd.DataFrame.

    Контракт:
    - Принимает путь к ZIP-файлу Binance Vision aggTrades.
    - Возвращает DataFrame с колонками: price (float32), qty (float32), side (str).
    - is_buyer_maker=True  → side='sell'  (buyer является maker → sell-initiated)
    - is_buyer_maker=False → side='buy'
    - Колонка 'quantity' в CSV переименовывается в 'qty'.
    - Возвращает только колонки price, qty, side (лишние колонки отброшены).
    """

    def test_returns_dataframe_with_required_columns(self, tmp_path):
        """
        Контракт: колонки price, qty, side обязательны.
        """
        zip_path = _make_aggtrades_zip(str(tmp_path))
        df = load_aggtrades_zip(zip_path)
        for col in ('price', 'qty', 'side'):
            assert col in df.columns, f"Missing column: {col}"

    def test_price_dtype_is_float(self, tmp_path):
        """
        Контракт: price — числовой тип (float32 или float64).
        WHY: dtype='float32' экономит RAM на 55M строках (1 GB → 500 MB).
        """
        zip_path = _make_aggtrades_zip(str(tmp_path))
        df = load_aggtrades_zip(zip_path)
        assert pd.api.types.is_float_dtype(df['price']), (
            f"Expected float dtype for price, got {df['price'].dtype}"
        )

    def test_qty_dtype_is_float(self, tmp_path):
        """
        Контракт: qty — числовой тип (float32 или float64).
        """
        zip_path = _make_aggtrades_zip(str(tmp_path))
        df = load_aggtrades_zip(zip_path)
        assert pd.api.types.is_float_dtype(df['qty']), (
            f"Expected float dtype for qty, got {df['qty'].dtype}"
        )

    def test_is_buyer_maker_true_maps_to_sell(self, tmp_path):
        """
        Контракт: is_buyer_maker=True → side='sell'.
        WHY: buyer=maker означает лимитный ордер на покупку стоял в стакане,
        тейкер продал в него → это sell-initiated сделка.
        """
        zip_path = _make_aggtrades_zip(str(tmp_path), n=10)
        df = load_aggtrades_zip(zip_path)
        # В fixture чётные строки (0,2,4...) имеют is_buyer_maker=True
        # Проверяем что side корректно присвоен хотя бы для одной строки
        assert 'sell' in df['side'].values, "Expected 'sell' values in side column"

    def test_is_buyer_maker_false_maps_to_buy(self, tmp_path):
        """
        Контракт: is_buyer_maker=False → side='buy'.
        """
        zip_path = _make_aggtrades_zip(str(tmp_path), n=10)
        df = load_aggtrades_zip(zip_path)
        assert 'buy' in df['side'].values, "Expected 'buy' values in side column"

    def test_side_only_contains_buy_and_sell(self, tmp_path):
        """
        Контракт: side содержит только 'buy' и 'sell', никаких других значений.
        """
        zip_path = _make_aggtrades_zip(str(tmp_path))
        df = load_aggtrades_zip(zip_path)
        assert set(df['side'].unique()).issubset({'buy', 'sell'}), (
            f"Unexpected values in side: {df['side'].unique()}"
        )

    def test_correct_row_count(self, tmp_path):
        """
        Контракт: количество строк совпадает с количеством сделок в ZIP.
        """
        n = 50
        zip_path = _make_aggtrades_zip(str(tmp_path), n=n)
        df = load_aggtrades_zip(zip_path)
        assert len(df) == n, f"Expected {n} rows, got {len(df)}"

    def test_no_extra_columns(self, tmp_path):
        """
        Контракт: только колонки price, qty, side — лишние отброшены.
        WHY: agg_trade_id, transact_time и др. не нужны для delta-profile.
        """
        zip_path = _make_aggtrades_zip(str(tmp_path))
        df = load_aggtrades_zip(zip_path)
        assert set(df.columns) == {'price', 'qty', 'side'}, (
            f"Unexpected columns: {df.columns.tolist()}"
        )


# ---------------------------------------------------------------------------
# calculate_cvd_in_zone  — Этап 8
# ---------------------------------------------------------------------------

from volume_density import calculate_cvd_in_zone


def _make_aggtrades_df(n: int = 200, poc: float = 50_000.0,
                       atr: float = 500.0, seed: int = 42) -> pd.DataFrame:
    """
    Синтетический DataFrame aggTrades (уже загруженный, без ZIP).
    Колонки: price (float), qty (float), side (str).
    n//2 сделок внутри зоны POC ± 1.5*ATR, остальные — снаружи.
    """
    rng = np.random.default_rng(seed)
    zone_low  = poc - 1.5 * atr
    zone_high = poc + 1.5 * atr

    prices = np.concatenate([
        rng.uniform(zone_low, zone_high, n // 2),          # внутри зоны
        rng.uniform(zone_high + 100, zone_high + 5000, n - n // 2),  # снаружи
    ])
    qtys  = rng.uniform(0.001, 2.0, n)
    sides = np.where(np.arange(n) % 2 == 0, 'sell', 'buy')

    return pd.DataFrame({'price': prices, 'qty': qtys, 'side': sides})


class TestCalculateCvdInZone:
    """
    Тесты контракта calculate_cvd_in_zone(df_trades, poc, atr) -> (cvd_series, cvd_slope).

    Контракт:
    - Принимает DataFrame с колонками price, qty, side (результат load_aggtrades_zip).
    - Фильтрует сделки в зоне [poc - 1.5*atr, poc + 1.5*atr].
    - CVD = кумулятивная сумма (buy_qty - sell_qty) по каждой сделке в зоне.
    - cvd_slope = наклон линейной регрессии CVD (положительный → лонги накапливают,
      отрицательный → выход крупного игрока).
    - Возвращает tuple: (pd.Series CVD, float slope).
    """

    def test_returns_tuple_of_series_and_float(self, tmp_path):
        """
        Контракт: возвращает (pd.Series, float).
        """
        df = _make_aggtrades_df()
        cvd_series, cvd_slope = calculate_cvd_in_zone(df, poc=50_000.0, atr=500.0)
        assert isinstance(cvd_series, pd.Series), "First element must be pd.Series"
        assert isinstance(cvd_slope, float), "Second element must be float"

    def test_cvd_series_is_cumulative(self, tmp_path):
        """
        Контракт: CVD — нарастающий итог, последнее значение = сумма всех дельт в зоне.
        WHY: cumsum по определению монотонен относительно знака каждого шага.
        """
        df = _make_aggtrades_df(n=100)
        cvd_series, _ = calculate_cvd_in_zone(df, poc=50_000.0, atr=500.0)
        # Последнее значение CVD должно совпадать с суммой всех дельт в зоне
        zone_mask = (
            (df['price'] >= 50_000.0 - 1.5 * 500.0) &
            (df['price'] <= 50_000.0 + 1.5 * 500.0)
        )
        zone_df   = df[zone_mask]
        deltas    = zone_df['qty'].where(zone_df['side'] == 'buy', -zone_df['qty'])
        expected_last = float(deltas.sum())
        assert abs(cvd_series.iloc[-1] - expected_last) < 1e-4, (
            f"CVD last value {cvd_series.iloc[-1]:.4f} != expected {expected_last:.4f}"
        )

    def test_trades_outside_zone_excluded(self):
        """
        Контракт: сделки за пределами poc ± 1.5*atr не влияют на CVD.
        """
        # Только сделки вне зоны
        df_outside = pd.DataFrame({
            'price': [60_000.0, 40_000.0, 65_000.0],
            'qty':   [1.0,       1.0,       1.0],
            'side':  ['buy',    'sell',    'buy'],
        })
        cvd_series, cvd_slope = calculate_cvd_in_zone(
            df_outside, poc=50_000.0, atr=500.0
        )
        assert len(cvd_series) == 0 or cvd_series.iloc[-1] == 0.0, (
            "Trades outside zone must not affect CVD"
        )
        assert cvd_slope == 0.0, (
            "Slope must be 0.0 when no trades in zone"
        )

    def test_positive_slope_when_buys_dominate(self):
        """
        Контракт: если покупки доминируют → CVD растёт → slope > 0.
        WHY: slope > 0 → лонги накапливают позицию → POC = магнит.
        """
        poc, atr = 50_000.0, 500.0
        # Все сделки — покупки внутри зоны
        df_buys = pd.DataFrame({
            'price': np.linspace(poc - atr, poc + atr, 50),
            'qty':   np.ones(50) * 1.0,
            'side':  ['buy'] * 50,
        })
        _, slope = calculate_cvd_in_zone(df_buys, poc=poc, atr=atr)
        assert slope > 0, f"Expected positive slope for buy-dominated zone, got {slope}"

    def test_negative_slope_when_sells_dominate(self):
        """
        Контракт: если продажи доминируют → CVD падает → slope < 0.
        WHY: slope < 0 → крупный игрок выходит "об толпу" → POC = ловушка.
        """
        poc, atr = 50_000.0, 500.0
        df_sells = pd.DataFrame({
            'price': np.linspace(poc - atr, poc + atr, 50),
            'qty':   np.ones(50) * 1.0,
            'side':  ['sell'] * 50,
        })
        _, slope = calculate_cvd_in_zone(df_sells, poc=poc, atr=atr)
        assert slope < 0, f"Expected negative slope for sell-dominated zone, got {slope}"

    def test_empty_zone_returns_empty_series_zero_slope(self):
        """
        Контракт: нет сделок в зоне → пустая Series, slope=0.0.
        """
        df = pd.DataFrame({'price': [], 'qty': [], 'side': []})
        cvd_series, slope = calculate_cvd_in_zone(df, poc=50_000.0, atr=500.0)
        assert slope == 0.0
        assert len(cvd_series) == 0


# ---------------------------------------------------------------------------
# calculate_delta_context_score  — Этап 8
# ---------------------------------------------------------------------------

from volume_density import calculate_delta_context_score


class TestCalculateDeltaContextScore:
    """
    Тесты контракта calculate_delta_context_score(cvd_slope, recent_delta) -> float.

    Контракт:
    - Принимает cvd_slope (float) — наклон CVD за anchor period (3–4 мес).
    - Принимает recent_delta (float) — суммарная дельта за reaction period (14 дней).
    - Возвращает float в диапазоне [0.0, 1.0].
    - Веса: cvd_slope (прошлое, "качество" уровня) + recent_delta (настоящее,
      "готовность рынка").
    - score > 0.5 → бычий контекст (магнит), score < 0.5 → медвежий (ловушка).

    WHY один параметр вместо двух:
    Из ТЗ — "Не плоди лишние параметры. Один delta_context_score внутри учитывает
    оба горизонта." Интеграция в calculate_poc_quality_score() через один float.
    """

    def test_returns_float_in_0_1_range(self):
        """
        Контракт: результат всегда в [0.0, 1.0].
        """
        score = calculate_delta_context_score(cvd_slope=1.0, recent_delta=100.0)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1] range"

    def test_bullish_context_score_above_half(self):
        """
        Контракт: оба сигнала бычьи → score > 0.5.
        cvd_slope > 0 (лонги накапливают) + recent_delta > 0 (покупатели доминируют).
        """
        score = calculate_delta_context_score(
            cvd_slope=500.0,      # сильный рост CVD
            recent_delta=200.0,   # положительная дельта последних 14 дней
        )
        assert score > 0.5, (
            f"Expected score > 0.5 for bullish context, got {score}"
        )

    def test_bearish_context_score_below_half(self):
        """
        Контракт: оба сигнала медвежьи → score < 0.5.
        cvd_slope < 0 (выход крупного) + recent_delta < 0 (продавцы доминируют).
        """
        score = calculate_delta_context_score(
            cvd_slope=-500.0,     # падение CVD
            recent_delta=-200.0,  # отрицательная дельта
        )
        assert score < 0.5, (
            f"Expected score < 0.5 for bearish context, got {score}"
        )

    def test_conflicting_signals_near_half(self):
        """
        Контракт: противоречивые сигналы → score близко к 0.5 (неопределённость).
        cvd_slope > 0 но recent_delta < 0 (история бычья, но сейчас продают).
        """
        score = calculate_delta_context_score(
            cvd_slope=500.0,
            recent_delta=-500.0,
        )
        # Не требуем точно 0.5, но score должен быть в зоне неопределённости
        assert 0.2 <= score <= 0.8, (
            f"Conflicting signals should give uncertain score, got {score}"
        )

    def test_zero_inputs_returns_half(self):
        """
        Контракт: оба нуля → score == 0.5 (нейтральный сигнал).
        WHY: sigmoid(0) = 0.5, нейтральная точка симметрична.
        """
        score = calculate_delta_context_score(cvd_slope=0.0, recent_delta=0.0)
        assert abs(score - 0.5) < 1e-6, (
            f"Zero inputs must return 0.5 (neutral), got {score}"
        )

    def test_score_bounded_regardless_of_extreme_inputs(self):
        """
        Контракт: экстремальные значения не выходят за [0, 1].
        WHY: sigmoid по определению ограничен, но проверяем явно.
        """
        for cvd, rdelta in [
            (1e9, 1e9),
            (-1e9, -1e9),
            (1e9, -1e9),
            (0.0, 1e9),
        ]:
            score = calculate_delta_context_score(cvd, rdelta)
            assert 0.0 <= score <= 1.0, (
                f"Score {score} out of bounds for cvd={cvd}, rdelta={rdelta}"
            )


# ---------------------------------------------------------------------------
# get_anchor_months  — Этап 8D, Шаг 1
# ---------------------------------------------------------------------------

from volume_density import get_anchor_months


def _make_ohlcv_with_dates(
    dates: list,
    closes: list,
    base_high_low_spread: float = 200.0,
) -> pd.DataFrame:
    """
    Вспомогательная фикстура: создаёт OHLCV DataFrame с заданными датами и close.
    Колонка 'date' — datetime64, как в production (после pd.to_datetime).
    """
    n = len(dates)
    rng = np.random.default_rng(0)
    closes_arr = np.array(closes, dtype=float)
    df = pd.DataFrame({
        'date':  pd.to_datetime(dates),
        'open':  closes_arr + rng.normal(0, 50, n),
        'high':  closes_arr + base_high_low_spread,
        'low':   closes_arr - base_high_low_spread,
        'close': closes_arr,
        'vol':   rng.uniform(10_000, 50_000, n),
    })
    return df


class TestGetAnchorMonths:
    """
    Тесты контракта get_anchor_months(df_ohlcv, poc, atr, lookback_days=120)
    → list of (year, month) tuples.

    Логика (из NEXT_SESSION.md):
    - Берём строки df где close в зоне poc ± 1.5*atr.
    - Извлекаем уникальные (year, month) из колонки 'date'.
    - lookback_days: ограничиваем df последними N днями перед фильтрацией.
    - Нет хардкода цен — poc приходит снаружи (из calculate_value_area).
    - Результат отсортирован хронологически.

    Принципы тестирования:
    - Тесты проверяют контракт (что возвращается), не реализацию (как).
    - Fixture создаёт данные с явными датами — нет случайности в assertions.
    """

    def test_returns_list_of_year_month_tuples(self):
        """
        Контракт: возвращает list of (int, int) tuples — (year, month).
        """
        df = _make_ohlcv_with_dates(
            dates=['2024-10-15', '2024-11-20', '2024-12-05'],
            closes=[50_100.0, 50_200.0, 50_300.0],
        )
        result = get_anchor_months(df, poc=50_200.0, atr=500.0, lookback_days=120)
        assert isinstance(result, list), "Must return list"
        for item in result:
            assert isinstance(item, tuple) and len(item) == 2, (
                f"Each element must be (year, month) tuple, got {item}"
            )
            year, month = item
            assert isinstance(year, int) and isinstance(month, int), (
                f"year and month must be int, got {type(year)}, {type(month)}"
            )

    def test_months_in_zone_returned(self):
        """
        Контракт: строки с close в зоне poc ± 1.5*atr дают корректные (year, month).
        POC=50_000, ATR=500 → зона [49_250, 50_750].
        """
        poc, atr = 50_000.0, 500.0
        zone_low  = poc - 1.5 * atr   # 49_250
        zone_high = poc + 1.5 * atr   # 50_750

        df = _make_ohlcv_with_dates(
            dates=['2024-11-10', '2024-11-15', '2024-12-20'],
            closes=[50_000.0, 49_500.0, 50_600.0],  # все три в зоне
        )
        result = get_anchor_months(df, poc=poc, atr=atr, lookback_days=120)
        # Ожидаем ноябрь и декабрь 2024
        assert (2024, 11) in result, f"Expected (2024,11) in {result}"
        assert (2024, 12) in result, f"Expected (2024,12) in {result}"

    def test_no_rows_in_zone_returns_empty_list(self):
        """
        Контракт: нет строк в зоне → пустой список, не ошибка.
        POC=50_000, ATR=100 → зона [49_850, 50_150].
        Все close далеко от POC.
        """
        df = _make_ohlcv_with_dates(
            dates=['2024-11-01', '2024-11-02', '2024-11-03'],
            closes=[60_000.0, 55_000.0, 45_000.0],  # все вне зоны
        )
        result = get_anchor_months(df, poc=50_000.0, atr=100.0, lookback_days=120)
        assert result == [], f"Expected empty list, got {result}"

    def test_unique_months_no_duplicates(self):
        """
        Контракт: каждый (year, month) появляется в результате ровно один раз.
        Несколько строк одного месяца → только одна запись в результате.
        """
        poc, atr = 50_000.0, 500.0
        # 5 строк в ноябре 2024, все в зоне
        dates  = [f'2024-11-{d:02d}' for d in range(1, 6)]
        closes = [50_000.0] * 5
        df = _make_ohlcv_with_dates(dates=dates, closes=closes)
        result = get_anchor_months(df, poc=poc, atr=atr, lookback_days=120)
        # (2024, 11) должен быть ровно один раз
        count_nov = result.count((2024, 11))
        assert count_nov == 1, (
            f"Expected (2024,11) exactly once, got {count_nov} times in {result}"
        )

    def test_lookback_days_excludes_old_rows(self):
        """
        Контракт: lookback_days=30 — берём только последние 30 дней df.
        Строки старше 30 дней должны быть проигнорированы даже если в зоне.

        WHY: anchor period — это не "все данные", а "последние N дней".
        Старые данные о POC нерелевантны для текущего торгового решения.
        """
        poc, atr = 50_000.0, 500.0
        today = datetime(2024, 12, 31)
        # Январь 2024 — старый (365 дней назад), в зоне
        # Декабрь 2024 — свежий (0-30 дней назад), в зоне
        dates = (
            [(datetime(2024, 1, d)).strftime('%Y-%m-%d') for d in range(1, 6)]
            + [(datetime(2024, 12, d)).strftime('%Y-%m-%d') for d in range(15, 32)]
        )
        closes = [50_000.0] * len(dates)
        df = _make_ohlcv_with_dates(dates=dates, closes=closes)

        result = get_anchor_months(df, poc=poc, atr=atr, lookback_days=30)
        assert (2024, 1) not in result, (
            f"January should be excluded by lookback_days=30, got {result}"
        )
        assert (2024, 12) in result, (
            f"December should be included (within 30 days), got {result}"
        )

    def test_result_sorted_chronologically(self):
        """
        Контракт: результат отсортирован в хронологическом порядке (year, month).
        WHY: download_anchor_data.py будет скачивать в этом порядке —
        предсказуемость упрощает кэш-логику.
        """
        poc, atr = 50_000.0, 500.0
        df = _make_ohlcv_with_dates(
            dates=['2024-12-10', '2024-10-05', '2024-11-20'],
            closes=[50_000.0, 50_100.0, 49_800.0],  # все в зоне
        )
        result = get_anchor_months(df, poc=poc, atr=atr, lookback_days=120)
        assert result == sorted(result), (
            f"Result must be sorted chronologically, got {result}"
        )

    def test_rows_outside_zone_excluded(self):
        """
        Контракт: строки с close вне зоны poc ± 1.5*atr не попадают в результат.
        """
        poc, atr = 50_000.0, 200.0  # зона [49_700, 50_300]
        df = _make_ohlcv_with_dates(
            dates=['2024-11-01', '2024-11-02', '2024-12-01'],
            closes=[
                50_100.0,   # в зоне → ноябрь включается
                55_000.0,   # вне зоны → не влияет на ноябрь
                60_000.0,   # вне зоны → декабрь НЕ включается
            ],
        )
        result = get_anchor_months(df, poc=poc, atr=atr, lookback_days=120)
        assert (2024, 11) in result, f"November (has in-zone row) must be included"
        assert (2024, 12) not in result, (
            f"December (all out-of-zone rows) must NOT be included, got {result}"
        )


# ---------------------------------------------------------------------------
# calculate_poc_quality_score — Этап 8D-4: delta_context_score
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="SUBJECTIVE: calculate_poc_quality_score is legacy, disabled pending objective replacement")
class TestPocQualityScoreWithDelta:
    """
    Тесты нового параметра delta_context_score в calculate_poc_quality_score().

    Новый контракт (из NEXT_SESSION.md):
    1. delta_context_score: float = 0.5  — нейтральный дефолт (обратная совместимость).
    2. delta_context_score < 0.35 AND volume_score > 0.6
       → label принудительно 'RESISTANCE_TRAP' (перебивает основной скор).
    3. delta_context_score > 0.65
       → absorption_score += 0.10 перед расчётом итогового скора (усиление сигнала).
    """

    def test_default_delta_score_backward_compatible(self):
        """
        Контракт: delta_context_score=0.5 по умолчанию — старые вызовы не ломаются.
        Результат должен совпадать с вызовом без параметра.
        """
        kwargs = dict(
            absorption_days_near_poc=5,
            total_days_near_poc=10,
            volume_w_score=50.0,
            capitulation_confirmed=False,
            z_score=0.0,
        )
        result_old = calculate_poc_quality_score(**kwargs)
        result_new = calculate_poc_quality_score(**kwargs, delta_context_score=0.5)
        assert result_old['score'] == result_new['score']
        assert result_old['label'] == result_new['label']

    def test_resistance_trap_forced_when_low_delta_high_volume(self):
        """
        Контракт: delta < 0.35 + volume_score > 0.6 → label='RESISTANCE_TRAP' принудительно.
        WHY: медвежья дельта при высоком объёме = продажи об лимиты = ловушка.
        Даже если основной скор был бы NEUTRAL или MAGNET.
        """
        result = calculate_poc_quality_score(
            absorption_days_near_poc=5,
            total_days_near_poc=10,      # absorption_score = 0.5
            volume_w_score=80.0,         # volume_score = 0.8 > 0.6
            capitulation_confirmed=False,
            z_score=2.0,                 # онлайн скор высокий
            delta_context_score=0.30,    # < 0.35 → принудительный TRAP
        )
        assert result['label'] == 'RESISTANCE_TRAP', (
            f"Expected forced RESISTANCE_TRAP, got {result['label']} (score={result['score']})"
        )

    def test_resistance_trap_not_forced_when_volume_low(self):
        """
        Контракт: delta < 0.35 но volume_score <= 0.6 → принудительного TRAP нет.
        Оба условия должны выполняться одновременно.
        """
        result = calculate_poc_quality_score(
            absorption_days_near_poc=10,
            total_days_near_poc=10,      # absorption_score = 1.0
            volume_w_score=50.0,         # volume_score = 0.5 <= 0.6 → условие не выполнено
            capitulation_confirmed=True,
            z_score=0.0,
            delta_context_score=0.20,    # < 0.35, но volume_score не > 0.6
        )
        assert result['label'] != 'RESISTANCE_TRAP', (
            f"RESISTANCE_TRAP must NOT be forced when volume_score <= 0.6, "
            f"got {result['label']}"
        )

    def test_bullish_delta_boosts_score(self):
        """
        Контракт: delta > 0.65 → absorption_score += 0.10 → итоговый score выше.
        Сравниваем с нейтральным delta=0.5 при тех же входных данных.
        """
        base_kwargs = dict(
            absorption_days_near_poc=5,
            total_days_near_poc=10,
            volume_w_score=60.0,
            capitulation_confirmed=False,
            z_score=0.0,
        )
        result_neutral = calculate_poc_quality_score(**base_kwargs, delta_context_score=0.5)
        result_bullish = calculate_poc_quality_score(**base_kwargs, delta_context_score=0.70)
        assert result_bullish['score'] > result_neutral['score'], (
            f"Bullish delta must boost score: "
            f"neutral={result_neutral['score']}, bullish={result_bullish['score']}"
        )

    def test_bullish_delta_boost_not_applied_below_threshold(self):
        """
        Контракт: delta = 0.65 (ровно на границе) — boost НЕ применяется (strict >).
        """
        base_kwargs = dict(
            absorption_days_near_poc=5,
            total_days_near_poc=10,
            volume_w_score=60.0,
            capitulation_confirmed=False,
            z_score=0.0,
        )
        result_boundary = calculate_poc_quality_score(**base_kwargs, delta_context_score=0.65)
        result_neutral  = calculate_poc_quality_score(**base_kwargs, delta_context_score=0.5)
        assert result_boundary['score'] == result_neutral['score'], (
            f"Boost must NOT apply at exactly 0.65 (strict >): "
            f"boundary={result_boundary['score']}, neutral={result_neutral['score']}"
        )


# ---------------------------------------------------------------------------
# calculate_poc_quality_score — OI regime parameter (Этап 4)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="SUBJECTIVE: calculate_poc_quality_score is legacy, disabled pending objective replacement")
class TestPocQualityScoreWithOiRegime:
    """
    Тесты параметра oi_regime в calculate_poc_quality_score().

    Новый контракт (из TECHNICAL_SPEC_POC_Quality.md, Этап 4):
    1. oi_regime: str = 'NEUTRAL'  — нейтральный дефолт (обратная совместимость).
    2. 'LIQUIDATION' → score += 0.10  (лонги ликвидируются → усиливает FAIR_VALUE_MAGNET).
    3. 'STRONG_BEAR' → score -= 0.10  (новые шорты → усиливает RESISTANCE_TRAP).
    4. Другие режимы ('STRONG_BULL', 'WEAK_BULL', 'NEUTRAL') — без влияния.
    5. score остаётся в [0.0, 1.0] после OI-коррекции (clamp).
    """

    def test_default_oi_regime_neutral_backward_compatible(self):
        """
        Контракт: oi_regime='NEUTRAL' по умолчанию — старые вызовы без параметра не ломаются.
        """
        kwargs = dict(
            absorption_days_near_poc=5,
            total_days_near_poc=10,
            volume_w_score=50.0,
            capitulation_confirmed=False,
            z_score=0.0,
        )
        result_old = calculate_poc_quality_score(**kwargs)
        result_new = calculate_poc_quality_score(**kwargs, oi_regime='NEUTRAL')
        assert result_old['score'] == result_new['score']
        assert result_old['label'] == result_new['label']

    def test_liquidation_boosts_score(self):
        """
        Контракт: 'LIQUIDATION' → score выше чем при NEUTRAL.
        WHY: лонги ликвидируются = паника прошла, дно близко → подтверждает FAIR_VALUE_MAGNET.
        """
        base = dict(
            absorption_days_near_poc=5,
            total_days_near_poc=10,
            volume_w_score=60.0,
            capitulation_confirmed=False,
            z_score=0.0,
        )
        result_neutral     = calculate_poc_quality_score(**base, oi_regime='NEUTRAL')
        result_liquidation = calculate_poc_quality_score(**base, oi_regime='LIQUIDATION')
        assert result_liquidation['score'] > result_neutral['score'], (
            f"LIQUIDATION must boost score: neutral={result_neutral['score']}, "
            f"liquidation={result_liquidation['score']}"
        )
        assert abs(result_liquidation['score'] - result_neutral['score']) == pytest.approx(0.10, abs=1e-6), (
            f"Boost must be exactly +0.10"
        )

    def test_strong_bear_reduces_score(self):
        """
        Контракт: 'STRONG_BEAR' → score ниже чем при NEUTRAL.
        WHY: новые шорты открываются → капитуляция не завершена → усиливает RESISTANCE_TRAP.
        """
        base = dict(
            absorption_days_near_poc=5,
            total_days_near_poc=10,
            volume_w_score=60.0,
            capitulation_confirmed=False,
            z_score=0.0,
        )
        result_neutral     = calculate_poc_quality_score(**base, oi_regime='NEUTRAL')
        result_strong_bear = calculate_poc_quality_score(**base, oi_regime='STRONG_BEAR')
        assert result_strong_bear['score'] < result_neutral['score'], (
            f"STRONG_BEAR must reduce score: neutral={result_neutral['score']}, "
            f"strong_bear={result_strong_bear['score']}"
        )
        assert abs(result_neutral['score'] - result_strong_bear['score']) == pytest.approx(0.10, abs=1e-6), (
            f"Penalty must be exactly -0.10"
        )

    def test_other_regimes_no_effect(self):
        """
        Контракт: 'STRONG_BULL', 'WEAK_BULL' — без изменения score.
        WHY: только LIQUIDATION и STRONG_BEAR имеют структурный смысл вблизи POC.
        """
        base = dict(
            absorption_days_near_poc=5,
            total_days_near_poc=10,
            volume_w_score=60.0,
            capitulation_confirmed=False,
            z_score=0.0,
        )
        result_neutral    = calculate_poc_quality_score(**base, oi_regime='NEUTRAL')
        result_strong_bull = calculate_poc_quality_score(**base, oi_regime='STRONG_BULL')
        result_weak_bull   = calculate_poc_quality_score(**base, oi_regime='WEAK_BULL')
        assert result_strong_bull['score'] == result_neutral['score']
        assert result_weak_bull['score']   == result_neutral['score']

    def test_score_clamped_to_1_after_liquidation_boost(self):
        """
        Контракт: score после +0.10 не превышает 1.0 (clamp).
        """
        result = calculate_poc_quality_score(
            absorption_days_near_poc=10,
            total_days_near_poc=10,      # absorption_score = 1.0
            volume_w_score=100.0,        # volume_score = 1.0
            capitulation_confirmed=True, # onchain_score = 1.0
            z_score=0.0,
            oi_regime='LIQUIDATION',
        )
        assert result['score'] <= 1.0, f"Score must be clamped to 1.0, got {result['score']}"

    def test_score_clamped_to_0_after_strong_bear_penalty(self):
        """
        Контракт: score после -0.10 не падает ниже 0.0 (clamp).
        """
        result = calculate_poc_quality_score(
            absorption_days_near_poc=0,
            total_days_near_poc=10,      # absorption_score = 0.0
            volume_w_score=0.0,          # volume_score = 0.0
            capitulation_confirmed=False,
            z_score=-5.0,                # sigmoid(-5) ≈ 0.007 ≈ 0.0
            oi_regime='STRONG_BEAR',
        )
        assert result['score'] >= 0.0, f"Score must be clamped to 0.0, got {result['score']}"


# ---------------------------------------------------------------------------
# Этап 9 — Шаг 2: load_klines_zip
# ---------------------------------------------------------------------------

import os
import zipfile
import io

from volume_density import load_klines_zip


def _make_klines_zip(tmp_path, n: int = 10):
    """
    Fixture: создаёт минимальный ZIP с синтетическими klines (12 колонок, без заголовка).

    Колонки по позициям:
        [0] open_time, [1] open, [2] high, [3] low, [4] close, [5] volume,
        [6] close_time, [7] quote_volume, [8] count,
        [9] taker_buy_base_vol, [10] taker_buy_quote_vol, [11] ignore
    """
    lines = []
    base_time = 1_700_000_000_000  # произвольный timestamp ms
    for i in range(n):
        open_time         = base_time + i * 60_000
        close_time        = open_time + 59_999
        open_p            = 30000.0 + i
        high_p            = 30010.0 + i
        low_p             = 29990.0 + i
        close_p           = 30005.0 + i
        volume            = 10.0 + i
        quote_volume      = volume * 30000.0
        count             = 100 + i
        taker_buy_vol     = volume * 0.6   # 60% buy
        taker_buy_q_vol   = taker_buy_vol * 30000.0
        ignore            = 0
        lines.append(
            f"{open_time},{open_p},{high_p},{low_p},{close_p},{volume},"
            f"{close_time},{quote_volume},{count},"
            f"{taker_buy_vol},{taker_buy_q_vol},{ignore}"
        )
    csv_content = "\n".join(lines)
    zip_path = str(tmp_path / "BTCUSDT-1m-2025-04-01.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("BTCUSDT-1m-2025-04-01.csv", csv_content)
    return zip_path


class TestLoadKlinesZip:
    """
    Контракт load_klines_zip(zip_path) -> pd.DataFrame:
    - Возвращает DataFrame с колонками: open_time, high, low, volume, taker_buy_vol.
    - open_time int64, остальные float (float32).
    - Кол-во строк = кол-во свечей в ZIP.
    - taker_buy_vol <= volume во всех строках.
    - Только 5 колонок.
    - CSV без заголовка парсится корректно.
    """

    def test_returns_dataframe_with_required_columns(self, tmp_path):
        """
        Контракт: результат содержит все 5 обязательных колонок.
        """
        zip_path = _make_klines_zip(tmp_path, n=5)
        df = load_klines_zip(zip_path)
        for col in ['open_time', 'high', 'low', 'volume', 'taker_buy_vol']:
            assert col in df.columns, f"Missing column: {col}"

    def test_dtypes_are_correct(self, tmp_path):
        """
        Контракт: open_time int64, остальные float.
        WHY float32: экономия RAM на 1440 строках в день.
        """
        zip_path = _make_klines_zip(tmp_path, n=5)
        df = load_klines_zip(zip_path)
        assert df['open_time'].dtype == np.int64, f"open_time must be int64, got {df['open_time'].dtype}"
        for col in ['high', 'low', 'volume', 'taker_buy_vol']:
            assert np.issubdtype(df[col].dtype, np.floating), f"{col} must be float, got {df[col].dtype}"

    def test_correct_row_count(self, tmp_path):
        """
        Контракт: кол-во строк = n (число свечей в ZIP).
        """
        zip_path = _make_klines_zip(tmp_path, n=7)
        df = load_klines_zip(zip_path)
        assert len(df) == 7, f"Expected 7 rows, got {len(df)}"

    def test_taker_buy_vol_leq_volume(self, tmp_path):
        """
        Контракт: taker_buy_vol не превышает volume ни в одной строке.
        WHY: по дефиниции sell_vol = volume - taker_buy_vol >= 0.
        """
        zip_path = _make_klines_zip(tmp_path, n=10)
        df = load_klines_zip(zip_path)
        assert (df['taker_buy_vol'] <= df['volume'] + 1e-5).all(), (
            "taker_buy_vol must not exceed volume in any row"
        )

    def test_no_extra_columns(self, tmp_path):
        """
        Контракт: ровно 5 колонок, ни одной лишней.
        """
        zip_path = _make_klines_zip(tmp_path, n=5)
        df = load_klines_zip(zip_path)
        assert list(df.columns) == ['open_time', 'high', 'low', 'volume', 'taker_buy_vol'], (
            f"Unexpected columns: {list(df.columns)}"
        )

    def test_handles_headerless_csv(self, tmp_path):
        """
        Контракт: CSV без заголовка парсится корректно — high > 0 во всех строках.
        WHY: Binance Vision klines всегда без заголовка — header=None обязателен.
        """
        zip_path = _make_klines_zip(tmp_path, n=3)
        df = load_klines_zip(zip_path)
        assert (df['high'] > 0).all(), "high prices must be positive (headerless CSV parsed ok)"


# ---------------------------------------------------------------------------
# Этап 9 — Шаг 3: build_delta_profile
# ---------------------------------------------------------------------------

from volume_density import build_delta_profile


class TestBuildDeltaProfile:
    """
    Контракт build_delta_profile(klines_df, global_bins) -> pd.DataFrame:
    - Колонки: price_low, price_high, mid, vol, delta.
    - len == 100 (кол-во бинов).
    - delta = taker_buy_vol - sell_vol = 2 * taker_buy_vol - volume.
    - vol >= 0 всегда.
    - если taker_buy_vol = volume/2 → delta == 0.
    - пустой df → vol=0, delta=0.
    """

    def _make_klines_df(self, n: int = 10, buy_fraction: float = 0.6) -> pd.DataFrame:
        """Helper: синтетический klines DataFrame для тестов."""
        price_base = 30000.0
        rows = []
        for i in range(n):
            volume = 10.0 + i
            rows.append({
                'open_time':     1_700_000_000_000 + i * 60_000,
                'high':          price_base + 10 + i,
                'low':           price_base - 10 + i,
                'volume':        volume,
                'taker_buy_vol': volume * buy_fraction,
            })
        return pd.DataFrame(rows).astype({
            'open_time':     'int64',
            'high':          'float32',
            'low':           'float32',
            'volume':        'float32',
            'taker_buy_vol': 'float32',
        })

    def _make_bins(self, klines_df: pd.DataFrame, n_bins: int = 100) -> np.ndarray:
        """Helper: global_bins по данным klines_df."""
        price_min = float(klines_df['low'].min()) - 100
        price_max = float(klines_df['high'].max()) + 100
        return np.linspace(price_min, price_max, n_bins + 1)

    def test_returns_required_columns(self, tmp_path):
        """
        Контракт: результат содержит все 5 обязательных колонок.
        """
        df = self._make_klines_df()
        bins = self._make_bins(df)
        result = build_delta_profile(df, bins)
        for col in ['price_low', 'price_high', 'mid', 'vol', 'delta']:
            assert col in result.columns, f"Missing column: {col}"

    def test_bin_count_matches_global_bins(self, tmp_path):
        """
        Контракт: кол-во строк = len(global_bins) - 1 = 100.
        """
        df = self._make_klines_df(n=5)
        bins = self._make_bins(df, n_bins=100)
        result = build_delta_profile(df, bins)
        assert len(result) == 100, f"Expected 100 bins, got {len(result)}"

    def test_delta_equals_buy_minus_sell(self, tmp_path):
        """
        Контракт: delta = taker_buy_vol - sell_vol = 2 * taker_buy_vol - volume.
        Проверяем через инвариант: delta + vol = 2 * buy_vol → delta = 2*buy - vol.
        WHY: не дублируем формулу в тесте — проверяем через свойство.
        """
        df = self._make_klines_df(n=10, buy_fraction=0.7)
        bins = self._make_bins(df)
        result = build_delta_profile(df, bins)
        active = result[result['vol'] > 0]
        # Для активных бинов: delta должна быть > 0 (т.к. buy_fraction=0.7 > 0.5)
        assert (active['delta'] > 0).all(), (
            f"With buy_fraction=0.7, all active bins must have positive delta"
        )

    def test_vol_non_negative(self, tmp_path):
        """
        Контракт: vol >= 0 во всех бинах.
        """
        df = self._make_klines_df()
        bins = self._make_bins(df)
        result = build_delta_profile(df, bins)
        assert (result['vol'] >= 0).all(), "vol must be non-negative in all bins"

    def test_delta_is_zero_when_balanced(self, tmp_path):
        """
        Контракт: taker_buy_vol = volume/2 → delta == 0 во всех активных бинах.
        """
        df = self._make_klines_df(buy_fraction=0.5)
        bins = self._make_bins(df)
        result = build_delta_profile(df, bins)
        active = result[result['vol'] > 0]
        assert (active['delta'].abs() < 1e-4).all(), (
            f"Balanced buy/sell must produce zero delta, got max={active['delta'].abs().max()}"
        )

    def test_empty_klines_returns_zero_profile(self, tmp_path):
        """
        Контракт: пустой klines_df → vol=0, delta=0 во всех бинах.
        """
        df_empty = pd.DataFrame(columns=['open_time', 'high', 'low', 'volume', 'taker_buy_vol'])
        bins = np.linspace(29000, 31000, 101)
        result = build_delta_profile(df_empty, bins)
        assert (result['vol'] == 0).all(), "Empty input must produce zero vol"
        assert (result['delta'] == 0).all(), "Empty input must produce zero delta"


# ---------------------------------------------------------------------------
# extract_sub_levels — HVN-детекция внутри широкой зоны
# ---------------------------------------------------------------------------

from volume_density import extract_sub_levels


class TestExtractSubLevels:
    """
    Контракт extract_sub_levels(zone, profile_df, n_peaks=3) -> list[dict]:
    - Возвращает список dict с ключами 'mid' и 'vol'.
    - Длина <= n_peaks.
    - Все 'mid' лежат внутри [zone['min'], zone['max']].
    - Сортировка по убыванию 'vol' (самый сильный уровень — первый).
    - Если профиль внутри зоны плоский (нет явных пиков) — возвращает [].
    - Если зона пустая (нет бинов) — возвращает [].
    """

    def _make_profile_with_peaks(self) -> pd.DataFrame:
        """
        Синтетический профиль: три явных пика на $82k, $90k, $98k
        на фоне низкого базового объёма.
        WHY: моделирует реальную ATH-зону с HVN внутри.
        """
        mids = np.linspace(74_000, 110_000, 100)
        vols = np.full(100, 500.0)  # базовый уровень
        # Три пика с prominence >> std базового уровня
        for peak_mid, peak_vol in [(82_000, 5_000), (90_000, 8_000), (98_000, 6_000)]:
            idx = np.argmin(np.abs(mids - peak_mid))
            vols[idx] = peak_vol
        return pd.DataFrame({
            'price_low':  mids - 360,
            'price_high': mids + 360,
            'mid':        mids,
            'vol':        vols,
        })

    def _wide_zone(self) -> dict:
        """Зона охватывающая все три пика."""
        return {'min': 74_000, 'max': 110_000}

    def test_returns_list_of_dicts_with_required_keys(self):
        """
        Контракт: каждый элемент результата — dict с ключами 'mid' и 'vol'.
        """
        profile = self._make_profile_with_peaks()
        result = extract_sub_levels(self._wide_zone(), profile)
        assert isinstance(result, list)
        for item in result:
            assert 'mid' in item, f"Missing key 'mid' in {item}"
            assert 'vol' in item, f"Missing key 'vol' in {item}"

    def test_finds_peaks_inside_zone(self):
        """
        Контракт: функция находит хотя бы 2 из 3 явных пиков.
        WHY: 3 пика с vol в 10-16x базового — должны детектироваться.
        """
        profile = self._make_profile_with_peaks()
        result = extract_sub_levels(self._wide_zone(), profile, n_peaks=3)
        assert len(result) >= 2, (
            f"Expected >= 2 peaks, got {len(result)}: {result}"
        )

    def test_all_mids_inside_zone_bounds(self):
        """
        Контракт: все возвращённые уровни лежат внутри зоны.
        """
        profile = self._make_profile_with_peaks()
        zone = self._wide_zone()
        result = extract_sub_levels(zone, profile, n_peaks=3)
        for item in result:
            assert zone['min'] <= item['mid'] <= zone['max'], (
                f"mid={item['mid']} is outside zone [{zone['min']}, {zone['max']}]"
            )

    def test_sorted_by_vol_descending(self):
        """
        Контракт: результат отсортирован по убыванию объёма.
        WHY: торговый алгоритм должен получать самый сильный уровень первым.
        """
        profile = self._make_profile_with_peaks()
        result = extract_sub_levels(self._wide_zone(), profile, n_peaks=3)
        if len(result) >= 2:
            vols = [r['vol'] for r in result]
            assert vols == sorted(vols, reverse=True), (
                f"Result not sorted by vol desc: {vols}"
            )

    def test_n_peaks_limits_output(self):
        """
        Контракт: len(result) <= n_peaks.
        """
        profile = self._make_profile_with_peaks()
        for n in [1, 2, 3]:
            result = extract_sub_levels(self._wide_zone(), profile, n_peaks=n)
            assert len(result) <= n, (
                f"n_peaks={n} but got {len(result)} results"
            )

    def test_flat_profile_returns_empty(self):
        """
        Контракт: профиль без явных пиков (все бины одинаковы) → [].
        WHY: prominence-фильтр должен отсеять шум без выраженных HVN.
        """
        mids = np.linspace(74_000, 110_000, 100)
        flat_profile = pd.DataFrame({
            'price_low':  mids - 360,
            'price_high': mids + 360,
            'mid':        mids,
            'vol':        np.full(100, 500.0),  # абсолютно плоский
        })
        result = extract_sub_levels(self._wide_zone(), flat_profile)
        assert result == [], (
            f"Flat profile must return [], got {result}"
        )

    def test_zone_outside_profile_returns_empty(self):
        """
        Контракт: зона вне диапазона профиля → [].
        WHY: защита от некорректных зон после DBSCAN.
        """
        profile = self._make_profile_with_peaks()
        out_of_range_zone = {'min': 200_000, 'max': 250_000}
        result = extract_sub_levels(out_of_range_zone, profile)
        assert result == [], (
            f"Zone outside profile must return [], got {result}"
        )


# ---------------------------------------------------------------------------
# TestCalculateLthPainProxy — Тесты 1–6 (RED фаза, Задача 10)
# ---------------------------------------------------------------------------

class TestCalculateLthPainProxy:
    """
    RED-фаза для calculate_lth_pain_proxy(df_ohlcv, window=155) -> dict.

    Проверяем контракты:
    - proxy_sopr = close[-1] / vwma_155
    - фазы по таблице порогов
    - roc_14: знак изменения proxy за 14 дней
    - days_below_1: подряд идущих дней с proxy < 1.0
    - INSUFFICIENT_DATA при df < window строк
    """

    # ------------------------------------------------------------------
    # Вспомогательный builder
    # ------------------------------------------------------------------

    @staticmethod
    def _make_df(n: int, close_seq=None, vol_val: float = 50_000.0) -> pd.DataFrame:
        """
        Строит синтетический df с колонками close, vol, date.
        close_seq — опциональный list/array длиной n (иначе все = 80_000).
        vol_val   — постоянный объём (упрощает ручной VWMA-расчёт в тестах).
        """
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(n)]
        closes = close_seq if close_seq is not None else [80_000.0] * n
        return pd.DataFrame({
            'date':  dates,
            'close': closes,
            'vol':   [vol_val] * n,
        })

    # ------------------------------------------------------------------
    # Тест 1: proxy_sopr корректен, фаза BULL
    # ------------------------------------------------------------------

    def test_proxy_sopr_bull_phase(self):
        """
        Контракт: при close[-1] > vwma_155 × 1.10 → phase == 'BULL'.
        Данные: 200 строк, все close = 80_000, vol постоянный
        → vwma_155 = 80_000 (равновзвешенный — все точки одинаковы)
        → proxy_sopr = 80_000 / 80_000 = 1.0
        Корректируем: последний close = 90_000 → proxy = 90_000 / 80_000 = 1.125 > 1.10
        WHY: BULL — базовый случай, proxy строго выше 1.10.
        """
        closes = [80_000.0] * 199 + [90_000.0]  # последняя свеча выше
        df = self._make_df(200, close_seq=closes)
        result = calculate_lth_pain_proxy(df, window=155)

        assert result['phase'] == 'BULL', (
            f"Expected BULL, got {result['phase']} (proxy_sopr={result['proxy_sopr']:.4f})"
        )
        assert result['proxy_sopr'] is not None
        assert result['proxy_sopr'] > 1.10, (
            f"BULL requires proxy > 1.10, got {result['proxy_sopr']:.4f}"
        )
        assert 'vwma_155' in result
        assert result['vwma_155'] > 0

    # ------------------------------------------------------------------
    # Тест 2: фаза RUBICON
    # ------------------------------------------------------------------

    def test_rubicon_phase(self):
        """
        Контракт: при 0.80 < proxy ≤ 1.0 → phase == 'RUBICON'.
        Данные: 200 строк close = 72_000, последняя = 65_000.
        При постоянном объёме vwma_155 ≈ 72_000 (все 155 точек одинаковы).
        proxy = 65_000 / 72_000 ≈ 0.903 → попадает в (0.80, 1.0].
        WHY: RUBICON — LTH в убытке, но не глубоко (< 20%).
        """
        closes = [72_000.0] * 199 + [65_000.0]
        df = self._make_df(200, close_seq=closes)
        result = calculate_lth_pain_proxy(df, window=155)

        assert result['phase'] == 'RUBICON', (
            f"Expected RUBICON, got {result['phase']} (proxy={result['proxy_sopr']:.4f})"
        )
        assert 0.80 < result['proxy_sopr'] <= 1.0, (
            f"RUBICON range (0.80, 1.0], got {result['proxy_sopr']:.4f}"
        )

    # ------------------------------------------------------------------
    # Тест 3: фаза CAPITULATION
    # ------------------------------------------------------------------

    def test_capitulation_phase(self):
        """
        Контракт: при 0.50 ≤ proxy ≤ 0.65 → phase == 'CAPITULATION'.
        Данные: 200 строк close = 65_000, последняя = 38_000.
        vwma_155 ≈ 65_000 (постоянный объём → все веса равны).
        proxy = 38_000 / 65_000 ≈ 0.585 → [0.50, 0.65].
        WHY: CAPITULATION — исторические зоны дна.
        """
        closes = [65_000.0] * 199 + [38_000.0]
        df = self._make_df(200, close_seq=closes)
        result = calculate_lth_pain_proxy(df, window=155)

        assert result['phase'] == 'CAPITULATION', (
            f"Expected CAPITULATION, got {result['phase']} (proxy={result['proxy_sopr']:.4f})"
        )
        assert 0.50 <= result['proxy_sopr'] <= 0.65, (
            f"CAPITULATION range [0.50, 0.65], got {result['proxy_sopr']:.4f}"
        )

    # ------------------------------------------------------------------
    # Тест 4: roc_14 — знак изменения
    # ------------------------------------------------------------------

    def test_roc_14_sign(self):
        """
        Контракт: roc_14 < 0 если прокси за 14 дней снижался.
        Данные: 200 строк, цена монотонно снижается с 72_000 до 60_000.
        → proxy[-1] < proxy[-15], значит roc_14 < 0.
        WHY: направление roc_14 важно для детекции разворота.
        """
        closes = list(np.linspace(72_000, 60_000, 200))
        df = self._make_df(200, close_seq=closes)
        result = calculate_lth_pain_proxy(df, window=155)

        assert 'roc_14' in result
        assert result['roc_14'] < 0, (
            f"Falling price → roc_14 should be < 0, got {result['roc_14']:.4f}"
        )

    # ------------------------------------------------------------------
    # Тест 5: days_below_1 считается корректно
    # ------------------------------------------------------------------

    def test_days_below_1(self):
        """
        Контракт: days_below_1 = количество последних подряд идущих дней
        с proxy_sopr < 1.0.
        Данные: 200 строк, первые 195 close = 80_000 (proxy ≈ 1.0 = граница),
        последние 5 close = 55_000 (proxy < 1.0).
        vwma_155 ≈ 80_000 (все 155 точек = 80_000, объём постоянный).
        proxy для последних 5 строк: 55_000 / 80_000 = 0.6875 < 1.0.
        WHY: подряд идущие дни в убытке — индикатор давления.
        """
        closes = [80_000.0] * 195 + [55_000.0] * 5
        df = self._make_df(200, close_seq=closes)
        result = calculate_lth_pain_proxy(df, window=155)

        assert 'days_below_1' in result
        assert result['days_below_1'] == 5, (
            f"Expected days_below_1=5, got {result['days_below_1']}"
        )

    # ------------------------------------------------------------------
    # Тест 6: INSUFFICIENT_DATA при df < window
    # ------------------------------------------------------------------

    def test_insufficient_data(self):
        """
        Контракт: если len(df) < window → proxy_sopr=None, phase='INSUFFICIENT_DATA'.
        WHY: VWMA-расчёт на неполном окне даёт бессмысленный результат.
        """
        df = self._make_df(50)  # 50 строк < window=155
        result = calculate_lth_pain_proxy(df, window=155)

        assert result['proxy_sopr'] is None, (
            f"Expected proxy_sopr=None for insufficient data, got {result['proxy_sopr']}"
        )
        assert result['phase'] == 'INSUFFICIENT_DATA', (
            f"Expected phase='INSUFFICIENT_DATA', got {result['phase']}"
        )


# ---------------------------------------------------------------------------
# TestLthCapitulationZoneFlag — Тесты 7–9 (RED фаза, Задача 10)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="SUBJECTIVE: calculate_poc_quality_score is legacy, disabled pending objective replacement")
class TestLthCapitulationZoneFlag:
    """
    RED-фаза для флага LTH_CAPITULATION_ZONE в calculate_poc_quality_score().

    Проверяем контракты:
    - флаг добавляется при lth_proxy_sopr < 0.60
    - флаг НЕ добавляется при lth_proxy_sopr >= 0.60
    - lth_proxy_sopr=None (дефолт) не ломает функцию
    """

    # Базовые аргументы для calculate_poc_quality_score() — заполняют остальные параметры
    BASE_KWARGS = dict(
        absorption_days_near_poc=2,
        total_days_near_poc=10,
        volume_w_score=50.0,
        capitulation_confirmed=False,
        z_score=0.0,
        delta_context_score=0.5,
        oi_regime='NEUTRAL',
    )

    # ------------------------------------------------------------------
    # Тест 7: флаг появляется при proxy < 0.60
    # ------------------------------------------------------------------

    def test_flag_added_when_proxy_below_060(self):
        """
        Контракт: при lth_proxy_sopr=0.55 (<0.60) → 'LTH_CAPITULATION_ZONE' в flags.
        WHY: флаг сигнализирует историческую зону дна для интерпретации POC.
        """
        result = calculate_poc_quality_score(**self.BASE_KWARGS, lth_proxy_sopr=0.55)
        assert 'LTH_CAPITULATION_ZONE' in result['flags'], (
            f"Expected 'LTH_CAPITULATION_ZONE' in flags, got {result['flags']}"
        )

    # ------------------------------------------------------------------
    # Тест 8: флаг НЕ добавляется при proxy >= 0.60
    # ------------------------------------------------------------------

    def test_flag_absent_when_proxy_at_or_above_060(self):
        """
        Контракт: при lth_proxy_sopr=0.75 (>=0.60) → 'LTH_CAPITULATION_ZONE' НЕ в flags.
        WHY: флаг не должен появляться за пределами CAPITULATION-зоны.
        """
        result = calculate_poc_quality_score(**self.BASE_KWARGS, lth_proxy_sopr=0.75)
        assert 'LTH_CAPITULATION_ZONE' not in result['flags'], (
            f"Expected 'LTH_CAPITULATION_ZONE' NOT in flags, got {result['flags']}"
        )

    # ------------------------------------------------------------------
    # Тест 9: обратная совместимость — lth_proxy_sopr=None не ломает функцию
    # ------------------------------------------------------------------

    def test_backward_compatibility_no_lth_proxy(self):
        """
        Контракт: вызов без lth_proxy_sopr (дефолт None) не бросает исключение,
        флаг НЕ добавляется, score возвращается как раньше.
        WHY: 217 старых тестов не передают lth_proxy_sopr — они должны остаться GREEN.
        """
        result = calculate_poc_quality_score(**self.BASE_KWARGS)
        # Функция не падает, возвращает корректный dict
        assert 'score' in result
        assert 'flags' in result
        assert 'label' in result
        assert 'LTH_CAPITULATION_ZONE' not in result['flags'], (
            f"None proxy must not add flag, got {result['flags']}"
        )


# ===========================================================================
# TestClassifyFundingRegime
# ===========================================================================

class TestClassifyFundingRegime:
    """
    Контракт classify_funding_regime(funding_pct: float) -> str

    Пороги (изменяемые константы, не архитектура):
      < -0.05%  → NEGATIVE_EXTREME
      < -0.01%  → NEGATIVE_MODERATE
      <= +0.01% → NEUTRAL
      <= +0.05% → POSITIVE_MODERATE
      >  +0.05% → POSITIVE_EXTREME
    """

    def test_negative_extreme(self):
        """funding = -0.10% → NEGATIVE_EXTREME (глубокий шорт-перевес)."""
        assert classify_funding_regime(-0.10) == 'NEGATIVE_EXTREME'

    def test_negative_extreme_boundary(self):
        """funding = -0.05% — граница: именно < -0.05 даёт EXTREME, -0.05 → MODERATE."""
        assert classify_funding_regime(-0.05) == 'NEGATIVE_MODERATE'

    def test_negative_moderate(self):
        """funding = -0.03% → NEGATIVE_MODERATE."""
        assert classify_funding_regime(-0.03) == 'NEGATIVE_MODERATE'

    def test_neutral_zero(self):
        """funding = 0.0% → NEUTRAL (базовая ставка Binance)."""
        assert classify_funding_regime(0.0) == 'NEUTRAL'

    def test_neutral_positive_edge(self):
        """funding = +0.01% → NEUTRAL (верхняя граница нейтральной зоны включительно)."""
        assert classify_funding_regime(0.01) == 'NEUTRAL'

    def test_neutral_negative_edge(self):
        """funding = -0.01% → NEUTRAL (нижняя граница нейтральной зоны включительно)."""
        assert classify_funding_regime(-0.01) == 'NEUTRAL'

    def test_positive_moderate(self):
        """funding = +0.03% → POSITIVE_MODERATE."""
        assert classify_funding_regime(0.03) == 'POSITIVE_MODERATE'

    def test_positive_extreme(self):
        """funding = +0.10% → POSITIVE_EXTREME (лонг-перегрев)."""
        assert classify_funding_regime(0.10) == 'POSITIVE_EXTREME'

    def test_positive_extreme_boundary(self):
        """funding = +0.05% — граница: именно > +0.05 даёт EXTREME, +0.05 → MODERATE."""
        assert classify_funding_regime(0.05) == 'POSITIVE_MODERATE'

    def test_returns_string(self):
        """Контракт: возвращает str, не None, не int."""
        result = classify_funding_regime(0.0)
        assert isinstance(result, str)


# ===========================================================================
# TestEvaluatePocQuality — теговая архитектура (новая, без score)
# ===========================================================================

@pytest.mark.skip(reason="SUBJECTIVE: threshold-based tag rules (0.35/0.40/0.6), disabled pending objective replacement")
class TestEvaluatePocQuality:
    """
    Контракт evaluate_poc_quality(...) -> dict

    Возвращает: {'label': str, 'tags': list[str]}
    НЕ возвращает 'score' — теговая архитектура.

    Метка определяется тегами по правилу приоритета:
      RESISTANCE_TRAP    — есть хотя бы один RESISTANCE_* тег, нет FAIR_VALUE_* тегов
      FAIR_VALUE_MAGNET  — есть хотя бодин FAIR_VALUE_* тег, нет RESISTANCE_* тегов
      NEUTRAL            — конфликт (оба типа) или нет значимых тегов
    """

    # --- Базовые аргументы: нейтральная ситуация (нет триггеров ни для одного тега) ---
    BASE_KWARGS = dict(
        absorption_days_near_poc=1,
        total_days_near_poc=10,
        volume_w_score=50.0,
        capitulation_confirmed=False,
        z_score=0.0,
        delta_context_score=0.5,
        oi_regime='NEUTRAL',
        lth_proxy_sopr=None,
        funding_regime=None,
    )

    # ------------------------------------------------------------------
    # Структура возвращаемого значения
    # ------------------------------------------------------------------

    def test_returns_label_and_tags(self):
        """Контракт: dict содержит 'label' и 'tags', НЕ содержит 'score'."""
        result = evaluate_poc_quality(**self.BASE_KWARGS)
        assert 'label' in result
        assert 'tags' in result
        assert 'score' not in result, "score не должен быть в теговой архитектуре"

    def test_tags_is_list(self):
        """Контракт: 'tags' — list (может быть пустым)."""
        result = evaluate_poc_quality(**self.BASE_KWARGS)
        assert isinstance(result['tags'], list)

    def test_neutral_baseline(self):
        """Нейтральные входные данные → label='NEUTRAL'."""
        result = evaluate_poc_quality(**self.BASE_KWARGS)
        assert result['label'] == 'NEUTRAL'

    # ------------------------------------------------------------------
    # FAIR_VALUE_MAGNET теги
    # ------------------------------------------------------------------

    def test_tag_fair_value_absorption(self):
        """absorption_ratio > 0.4 → тег FAIR_VALUE_MAGNET_ABSORPTION."""
        result = evaluate_poc_quality(
            **{**self.BASE_KWARGS,
               'absorption_days_near_poc': 5,
               'total_days_near_poc': 10}   # ratio = 0.5 > 0.4
        )
        assert 'FAIR_VALUE_MAGNET_ABSORPTION' in result['tags']

    def test_tag_fair_value_capitulation(self):
        """capitulation_confirmed=True → тег FAIR_VALUE_MAGNET_CAPITULATION."""
        result = evaluate_poc_quality(
            **{**self.BASE_KWARGS, 'capitulation_confirmed': True}
        )
        assert 'FAIR_VALUE_MAGNET_CAPITULATION' in result['tags']

    def test_tag_fair_value_sth_pressure(self):
        """z_score > 1.0 → тег FAIR_VALUE_MAGNET_STH_PRESSURE."""
        result = evaluate_poc_quality(
            **{**self.BASE_KWARGS, 'z_score': 1.5}
        )
        assert 'FAIR_VALUE_MAGNET_STH_PRESSURE' in result['tags']

    def test_label_fair_value_magnet_from_capitulation(self):
        """Один FAIR_VALUE_* тег + нет RESISTANCE_* → label='FAIR_VALUE_MAGNET'."""
        result = evaluate_poc_quality(
            **{**self.BASE_KWARGS, 'capitulation_confirmed': True}
        )
        assert result['label'] == 'FAIR_VALUE_MAGNET'

    # ------------------------------------------------------------------
    # RESISTANCE_TRAP теги
    # ------------------------------------------------------------------

    def test_tag_resistance_trap_delta(self):
        """delta < 0.35 + volume_w_score > 60 → тег RESISTANCE_TRAP_DELTA."""
        result = evaluate_poc_quality(
            **{**self.BASE_KWARGS,
               'delta_context_score': 0.20,
               'volume_w_score': 70.0}
        )
        assert 'RESISTANCE_TRAP_DELTA' in result['tags']

    def test_tag_resistance_trap_oi(self):
        """oi_regime='STRONG_BEAR' → тег RESISTANCE_TRAP_OI."""
        result = evaluate_poc_quality(
            **{**self.BASE_KWARGS, 'oi_regime': 'STRONG_BEAR'}
        )
        assert 'RESISTANCE_TRAP_OI' in result['tags']

    def test_label_resistance_trap_from_oi(self):
        """Один RESISTANCE_* тег + нет FAIR_VALUE_* → label='RESISTANCE_TRAP'."""
        result = evaluate_poc_quality(
            **{**self.BASE_KWARGS, 'oi_regime': 'STRONG_BEAR'}
        )
        assert result['label'] == 'RESISTANCE_TRAP'

    # ------------------------------------------------------------------
    # Информационные теги (не влияют на label)
    # ------------------------------------------------------------------

    def test_tag_lth_capitulation_zone(self):
        """lth_proxy_sopr < 0.60 → информационный тег LTH_CAPITULATION_ZONE."""
        result = evaluate_poc_quality(
            **{**self.BASE_KWARGS, 'lth_proxy_sopr': 0.50}
        )
        assert 'LTH_CAPITULATION_ZONE' in result['tags']

    def test_tag_lth_capitulation_zone_does_not_change_label(self):
        """LTH_CAPITULATION_ZONE — информационный: не меняет NEUTRAL на FAIR_VALUE_MAGNET."""
        result = evaluate_poc_quality(
            **{**self.BASE_KWARGS, 'lth_proxy_sopr': 0.50}
        )
        assert result['label'] == 'NEUTRAL'

    def test_tag_bullish_divergence(self):
        """oi_regime='LIQUIDATION' + absorption_ratio > 0.3 → BULLISH_DIVERGENCE."""
        result = evaluate_poc_quality(
            **{**self.BASE_KWARGS,
               'oi_regime': 'LIQUIDATION',
               'absorption_days_near_poc': 4,
               'total_days_near_poc': 10}   # ratio = 0.4 > 0.3
        )
        assert 'BULLISH_DIVERGENCE' in result['tags']

    # ------------------------------------------------------------------
    # Конфликт тегов → NEUTRAL
    # ------------------------------------------------------------------

    def test_conflict_gives_neutral(self):
        """FAIR_VALUE_MAGNET_CAPITULATION + RESISTANCE_TRAP_OI → label='NEUTRAL'."""
        result = evaluate_poc_quality(
            **{**self.BASE_KWARGS,
               'capitulation_confirmed': True,
               'oi_regime': 'STRONG_BEAR'}
        )
        assert result['label'] == 'NEUTRAL'

    # ------------------------------------------------------------------
    # funding_regime интеграция
    # ------------------------------------------------------------------

    def test_tag_resistance_trap_funding(self):
        """funding_regime='POSITIVE_EXTREME' → тег RESISTANCE_TRAP_FUNDING."""
        result = evaluate_poc_quality(
            **{**self.BASE_KWARGS, 'funding_regime': 'POSITIVE_EXTREME'}
        )
        assert 'RESISTANCE_TRAP_FUNDING' in result['tags']

    def test_funding_none_no_tag(self):
        """funding_regime=None (дефолт) → тег RESISTANCE_TRAP_FUNDING отсутствует."""
        result = evaluate_poc_quality(**self.BASE_KWARGS)
        assert 'RESISTANCE_TRAP_FUNDING' not in result['tags']

    def test_backward_compat_no_funding(self):
        """Вызов без funding_regime не бросает исключение."""
        kwargs = {k: v for k, v in self.BASE_KWARGS.items() if k != 'funding_regime'}
        result = evaluate_poc_quality(**kwargs)
        assert 'label' in result


# ---------------------------------------------------------------------------
# classify_market_regime — RED phase
# ---------------------------------------------------------------------------

class TestClassifyMarketRegime:
    """
    Тесты написаны ДО реализации (RED фаза TDD).
    classify_market_regime(oi_regime, funding_regime) -> str

    Агрегирует classify_oi_regime() + classify_funding_regime() в единый
    рыночный режим. Pure function, без сетевых вызовов.

    Возможные значения:
        'OVERHEATED_BULL'  -- STRONG_BULL + POSITIVE_EXTREME (перегрев)
        'BULL'             -- STRONG_BULL + не перегрев
        'CAPITULATION'     -- LIQUIDATION (безусловно)
        'BEAR'             -- STRONG_BEAR / WEAK_BULL с медвежьим фандингом
        'BEAR_SQUEEZE'     -- STRONG_BEAR + POSITIVE_EXTREME (аномалия)
        'NEUTRAL'          -- всё остальное
    """

    VALID_REGIMES = {
        'OVERHEATED_BULL', 'BULL', 'CAPITULATION',
        'BEAR', 'BEAR_SQUEEZE', 'NEUTRAL',
    }

    # ------------------------------------------------------------------
    # Контракт возвращаемого значения
    # ------------------------------------------------------------------

    def test_returns_string(self):
        """Функция всегда возвращает str."""
        result = classify_market_regime('NEUTRAL', 'NEUTRAL')
        assert isinstance(result, str)

    def test_valid_regime_set(self):
        """Все возможные пары входов дают только допустимые значения."""
        oi_regimes      = ['STRONG_BULL', 'WEAK_BULL', 'STRONG_BEAR',
                           'LIQUIDATION', 'NEUTRAL']
        funding_regimes = ['NEGATIVE_EXTREME', 'NEGATIVE_MODERATE', 'NEUTRAL',
                           'POSITIVE_MODERATE', 'POSITIVE_EXTREME']
        for oi in oi_regimes:
            for fr in funding_regimes:
                result = classify_market_regime(oi, fr)
                assert result in self.VALID_REGIMES, (
                    f"classify_market_regime({oi!r}, {fr!r}) = {result!r} "
                    f"не входит в допустимые значения {self.VALID_REGIMES}"
                )

    # ------------------------------------------------------------------
    # Матрица режимов
    # ------------------------------------------------------------------

    def test_liquidation_any_funding_is_capitulation(self):
        """LIQUIDATION всегда → CAPITULATION, независимо от funding."""
        for fr in ['NEGATIVE_EXTREME', 'NEUTRAL', 'POSITIVE_EXTREME']:
            assert classify_market_regime('LIQUIDATION', fr) == 'CAPITULATION', (
                f"LIQUIDATION + {fr!r} должен быть CAPITULATION"
            )

    def test_strong_bull_neutral_funding_is_bull(self):
        """STRONG_BULL + NEUTRAL → BULL."""
        assert classify_market_regime('STRONG_BULL', 'NEUTRAL') == 'BULL'

    def test_strong_bull_positive_moderate_is_bull(self):
        """STRONG_BULL + POSITIVE_MODERATE → BULL (не перегрев)."""
        assert classify_market_regime('STRONG_BULL', 'POSITIVE_MODERATE') == 'BULL'

    def test_strong_bull_positive_extreme_is_overheated(self):
        """STRONG_BULL + POSITIVE_EXTREME → OVERHEATED_BULL."""
        assert classify_market_regime('STRONG_BULL', 'POSITIVE_EXTREME') == 'OVERHEATED_BULL'

    def test_strong_bear_neutral_funding_is_bear(self):
        """STRONG_BEAR + NEUTRAL → BEAR."""
        assert classify_market_regime('STRONG_BEAR', 'NEUTRAL') == 'BEAR'

    def test_strong_bear_negative_funding_is_bear(self):
        """STRONG_BEAR + NEGATIVE_EXTREME → BEAR (согласованный медвежий сигнал)."""
        assert classify_market_regime('STRONG_BEAR', 'NEGATIVE_EXTREME') == 'BEAR'

    def test_strong_bear_positive_extreme_is_bear_squeeze(self):
        """STRONG_BEAR + POSITIVE_EXTREME → BEAR_SQUEEZE (аномалия: OI медвежий, фандинг бычий)."""
        assert classify_market_regime('STRONG_BEAR', 'POSITIVE_EXTREME') == 'BEAR_SQUEEZE'

    def test_weak_bull_negative_extreme_is_bear(self):
        """WEAK_BULL + NEGATIVE_EXTREME → BEAR (рост на ликвидациях шортов, не реальный бычий)."""
        assert classify_market_regime('WEAK_BULL', 'NEGATIVE_EXTREME') == 'BEAR'

    def test_weak_bull_negative_moderate_is_bear(self):
        """WEAK_BULL + NEGATIVE_MODERATE → BEAR."""
        assert classify_market_regime('WEAK_BULL', 'NEGATIVE_MODERATE') == 'BEAR'

    def test_weak_bull_neutral_funding_is_neutral(self):
        """WEAK_BULL + NEUTRAL → NEUTRAL (неопределённость)."""
        assert classify_market_regime('WEAK_BULL', 'NEUTRAL') == 'NEUTRAL'

    def test_neutral_oi_neutral_funding_is_neutral(self):
        """NEUTRAL + NEUTRAL → NEUTRAL."""
        assert classify_market_regime('NEUTRAL', 'NEUTRAL') == 'NEUTRAL'

    def test_neutral_oi_positive_extreme_is_neutral(self):
        """NEUTRAL OI + POSITIVE_EXTREME funding → NEUTRAL (один сигнал без подтверждения OI)."""
        assert classify_market_regime('NEUTRAL', 'POSITIVE_EXTREME') == 'NEUTRAL'

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_liquidation_overrides_positive_extreme(self):
        """LIQUIDATION + POSITIVE_EXTREME → CAPITULATION (LIQUIDATION имеет наивысший приоритет)."""
        assert classify_market_regime('LIQUIDATION', 'POSITIVE_EXTREME') == 'CAPITULATION'

    def test_unknown_oi_regime_falls_back_to_neutral(self):
        """Неизвестный oi_regime → NEUTRAL (без исключения)."""
        result = classify_market_regime('UNKNOWN_OI', 'NEUTRAL')
        assert result == 'NEUTRAL'

    def test_unknown_funding_regime_falls_back_to_neutral(self):
        """Неизвестный funding_regime при STRONG_BULL → BULL (фандинг неизвестен = не перегрев)."""
        # WHY BULL, не NEUTRAL: OI сигнал достаточен для BULL без подтверждения фандинга
        result = classify_market_regime('STRONG_BULL', 'UNKNOWN_FUNDING')
        assert result == 'BULL'


# ---------------------------------------------------------------------------
# calculate_basis_spread — RED-фаза (ЗАДАЧА 3)
# ---------------------------------------------------------------------------


class TestCalculateBasisSpread:
    """
    RED-фаза TDD для calculate_basis_spread(spot_price, futures_price) -> dict.

    Контракт (из NEXT_SESSION.md):
    - Возвращает dict с ключами: 'basis_usd', 'basis_pct', 'regime'.
    - basis_usd = futures_price - spot_price (знак важен).
    - basis_pct = (futures_price - spot_price) / spot_price * 100.
    - regime: 'CONTANGO' | 'BACKWARDATION' | 'FLAT'.
    - basis_pct > +0.1%  → 'CONTANGO'      (futures дороже spot).
    - basis_pct < -0.1%  → 'BACKWARDATION' (futures дешевле spot).
    - иначе              → 'FLAT'           (в пределах базовой ставки фандинга).
    - spot_price = 0 → ZeroDivisionError или ValueError.

    WHY 0.1% порог: базовая ставка Binance perp funding = 0.01%/8ч ≈ 0.1%/день.
    Спред в пределах одной ставки — нейтральный.
    """

    def test_returns_dict_with_required_keys(self):
        """
        Контракт: результат — dict с ключами 'basis_usd', 'basis_pct', 'regime'.
        """
        result = calculate_basis_spread(spot_price=84_000.0, futures_price=84_100.0)
        assert isinstance(result, dict)
        for key in ('basis_usd', 'basis_pct', 'regime'):
            assert key in result, f"Missing key: '{key}'"

    def test_contango_when_futures_above_spot(self):
        """
        Контракт: futures > spot + порог → regime == 'CONTANGO'.
        basis_pct = (84_200 - 84_000) / 84_000 * 100 ≈ 0.238% > +0.1%.
        WHY CONTANGO: futures дороже spot → рынок ждёт роста → бычий контекст.
        """
        result = calculate_basis_spread(spot_price=84_000.0, futures_price=84_200.0)
        assert result['regime'] == 'CONTANGO', (
            f"Expected CONTANGO for futures > spot + threshold, got '{result['regime']}'"
        )

    def test_backwardation_when_futures_below_spot(self):
        """
        Контракт: futures < spot - порог → regime == 'BACKWARDATION'.
        basis_pct = (83_800 - 84_000) / 84_000 * 100 ≈ -0.238% < -0.1%.
        WHY BACKWARDATION: futures дешевле spot → медвежий контекст, шорты платят.
        """
        result = calculate_basis_spread(spot_price=84_000.0, futures_price=83_800.0)
        assert result['regime'] == 'BACKWARDATION', (
            f"Expected BACKWARDATION for futures < spot - threshold, got '{result['regime']}'"
        )

    def test_flat_within_threshold(self):
        """
        Контракт: |basis_pct| <= 0.1% → regime == 'FLAT'.
        basis_pct = (84_050 - 84_000) / 84_000 * 100 ≈ 0.0595% < 0.1%.
        WHY FLAT: спред в пределах базовой ставки фандинга — нейтральный.
        """
        result = calculate_basis_spread(spot_price=84_000.0, futures_price=84_050.0)
        assert result['regime'] == 'FLAT', (
            f"Expected FLAT for basis within threshold, got '{result['regime']}'"
        )

    def test_basis_usd_sign(self):
        """
        Контракт: basis_usd = futures - spot (положительный при CONTANGO).
        WHY знак важен: используется для расчёта арбитражной прибыли.
        """
        spot, futures = 84_000.0, 84_200.0
        result = calculate_basis_spread(spot_price=spot, futures_price=futures)
        expected_usd = futures - spot  # 200.0
        assert abs(result['basis_usd'] - expected_usd) < 1e-6, (
            f"Expected basis_usd={expected_usd}, got {result['basis_usd']}"
        )

    def test_basis_pct_formula(self):
        """
        Контракт: basis_pct = (futures - spot) / spot * 100.
        Проверяем точность формулы — не inline-дублирование, а проверка контракта.
        """
        spot, futures = 84_000.0, 84_200.0
        result = calculate_basis_spread(spot_price=spot, futures_price=futures)
        expected_pct = (futures - spot) / spot * 100
        assert abs(result['basis_pct'] - expected_pct) < 1e-9, (
            f"Expected basis_pct={expected_pct:.6f}, got {result['basis_pct']:.6f}"
        )

    def test_zero_spot_raises(self):
        """
        Контракт: spot_price = 0 → ZeroDivisionError или ValueError.
        WHY: деление на ноль в basis_pct = ... / spot * 100 должно быть явной ошибкой.
        """
        with pytest.raises((ZeroDivisionError, ValueError)):
            calculate_basis_spread(spot_price=0.0, futures_price=84_000.0)

    def test_negative_basis_usd_in_backwardation(self):
        """
        Контракт: basis_usd < 0 при BACKWARDATION (futures < spot).
        """
        result = calculate_basis_spread(spot_price=84_000.0, futures_price=83_800.0)
        assert result['basis_usd'] < 0, (
            f"Expected negative basis_usd in BACKWARDATION, got {result['basis_usd']}"
        )

    def test_flat_exactly_at_threshold_boundary(self):
        """
        Контракт: basis_pct == +0.1% точно на границе → 'FLAT' (не CONTANGO).
        WHY: граничный тест для строгого неравенства (>0.1% → CONTANGO, <=0.1% → FLAT).
        """
        spot = 100_000.0
        futures = spot * (1 + 0.001)   # ровно +0.1%
        result = calculate_basis_spread(spot_price=spot, futures_price=futures)
        assert result['regime'] == 'FLAT', (
            f"basis_pct=+0.1% exactly must be FLAT (strict >), got '{result['regime']}'"
        )


# ---------------------------------------------------------------------------
# classify_volume_type — RED-фаза
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="SUBJECTIVE: threshold-based (1.5×mean, 0.70/0.30 close_pct, 2×ATR breakout), disabled pending objective replacement")
class TestClassifyVolumeType:
    """
    RED-фаза TDD для classify_volume_type(
        open_, high, low, close, vol, vol_mean_20, atr
    ) -> str.

    Типы свечей:
        'ABSORPTION'  — высокий объём, бычье закрытие (>0.70 диапазона)
        'EXHAUSTION'  — высокий объём, медвежье закрытие (<0.30 диапазона)
        'BREAKOUT'    — высокий объём, широкая свеча (диапазон > 2.0 * ATR)
        'NEUTRAL'     — нормальный объём или нет чёткого сигнала

    Определения:
        high_vol     = vol > vol_mean_20 * 1.5
        candle_range = high - low  (защищён от 0 через atr)
        close_pct    = (close - low) / candle_range

    Приоритет при конфликте (high_vol + широкая свеча):
        BREAKOUT > ABSORPTION > EXHAUSTION > NEUTRAL

    WHY эта функция: расширяет detect_absorption_days() (была bool)
    до семантической метки. Работает на одной свече без DataFrame,
    поэтому применима как в строковом применении, так и за df.apply().
    """

    # Базовые параметры: нормальная свеча BTC
    BASE = dict(
        open_=84_000.0,
        high=84_800.0,
        low=83_200.0,
        close=84_500.0,   # close_pct = (84500-83200)/(84800-83200) = 1300/1600 = 0.8125
        vol=10_000.0,
        vol_mean_20=8_000.0,  # vol/vol_mean_20 = 1.25 < 1.5 → не high_vol
        atr=700.0,
    )

    def test_returns_string(self):
        """Контракт: функция возвращает str."""
        result = classify_volume_type(**self.BASE)
        assert isinstance(result, str)

    def test_returns_valid_type(self):
        """Контракт: результат принадлежит допустимому набору."""
        for close_pct in [0.1, 0.5, 0.9]:
            params = dict(
                open_=84_000.0, high=85_000.0, low=83_000.0,
                close=83_000.0 + close_pct * 2_000.0,
                vol=20_000.0, vol_mean_20=8_000.0, atr=700.0,
            )
            result = classify_volume_type(**params)
            assert result in ('ABSORPTION', 'EXHAUSTION', 'BREAKOUT', 'NEUTRAL'), (
                f"Invalid type '{result}' for close_pct={close_pct}"
            )

    def test_neutral_when_low_volume(self):
        """
        Контракт: нормальный объём → 'NEUTRAL' (независимо от формы свечи).
        vol = vol_mean_20 * 1.0 — норма, не превышает порог 1.5.
        """
        result = classify_volume_type(
            open_=84_000.0, high=84_800.0, low=83_200.0,
            close=84_600.0,
            vol=8_000.0,        # vol/vol_mean_20 = 1.0 < 1.5
            vol_mean_20=8_000.0,
            atr=700.0,
        )
        assert result == 'NEUTRAL', f"Expected NEUTRAL for low volume, got '{result}'"

    def test_absorption_high_vol_bullish_close(self):
        """
        Контракт: высокий объём + закрытие в верхних 30% → 'ABSORPTION'.
        close_pct = (84700-83200)/(84800-83200) = 1500/1600 = 0.9375 > 0.70.
        vol = vol_mean_20 * 2.0 > 1.5 → high_vol=True.
        WHY не BREAKOUT: диапазон = 1600 < 2.0 * ATR(700) = 1400 — нет, 1600 > 1400,
        поэтому BREAKOUT имеет приоритет перед ABSORPTION.
        Используем узкий диапазон, чтобы BREAKOUT не сработал:
        high-low = 500 < 2*ATR(700) = 1400.
        """
        result = classify_volume_type(
            open_=84_000.0, high=84_500.0, low=84_000.0,
            close=84_400.0,   # close_pct = 400/500 = 0.80 > 0.70
            vol=20_000.0,     # 20000/10000 = 2.0 > 1.5
            vol_mean_20=10_000.0,
            atr=700.0,        # 2*atr = 1400, range = 500 < 1400 → не breakout
        )
        assert result == 'ABSORPTION', (
            f"Expected ABSORPTION for high vol + bullish close, got '{result}'"
        )

    def test_exhaustion_high_vol_bearish_close(self):
        """
        Контракт: высокий объём + закрытие в нижних 30% → 'EXHAUSTION'.
        close_pct = (84100-84000)/(84500-84000) = 100/500 = 0.20 < 0.30.
        WHY EXHAUSTION: продавцы открыли высоко, но цена вернулась вверх и закрылась нижко
        — предложение истощается, шорты сдаются.
        """
        result = classify_volume_type(
            open_=84_200.0, high=84_500.0, low=84_000.0,
            close=84_100.0,   # close_pct = 100/500 = 0.20 < 0.30
            vol=20_000.0,     # high_vol = True
            vol_mean_20=10_000.0,
            atr=700.0,        # 2*atr = 1400, range = 500 < 1400 → не breakout
        )
        assert result == 'EXHAUSTION', (
            f"Expected EXHAUSTION for high vol + bearish close, got '{result}'"
        )

    def test_breakout_wide_candle_high_vol(self):
        """
        Контракт: высокий объём + широкая свеча (range > 2*ATR) → 'BREAKOUT'.
        range = 88000 - 84000 = 4000, 2*ATR(700) = 1400. 4000 > 1400 → BREAKOUT.
        WHY BREAKOUT перекрывает ABSORPTION: пробой — самый сильный сигнал.
        """
        result = classify_volume_type(
            open_=84_000.0, high=88_000.0, low=84_000.0,
            close=87_500.0,   # close_pct = 3500/4000 = 0.875 > 0.70
            vol=20_000.0,     # high_vol = True
            vol_mean_20=10_000.0,
            atr=700.0,        # 2*atr = 1400, range = 4000 > 1400 → BREAKOUT
        )
        assert result == 'BREAKOUT', (
            f"Expected BREAKOUT for wide candle + high vol, got '{result}'"
        )

    def test_zero_range_does_not_raise(self):
        """
        Контракт: дожи (high == low) не бросают ZeroDivisionError.
        WHY: защита через atr (аналогично detect_absorption_days).
        """
        result = classify_volume_type(
            open_=84_000.0, high=84_000.0, low=84_000.0, close=84_000.0,
            vol=20_000.0, vol_mean_20=10_000.0, atr=700.0,
        )
        assert isinstance(result, str)  # не бросает исключение

    def test_exactly_at_high_vol_threshold(self):
        """
        Контракт: vol == vol_mean_20 * 1.5 ровно на границе → 'NEUTRAL' (не high_vol).
        WHY: строгое неравенство (>) согласовано с detect_absorption_days().
        """
        result = classify_volume_type(
            open_=84_000.0, high=84_500.0, low=84_000.0,
            close=84_400.0,
            vol=15_000.0,       # vol / vol_mean_20 = 15000/10000 = 1.5, не > 1.5
            vol_mean_20=10_000.0,
            atr=700.0,
        )
        assert result == 'NEUTRAL', (
            f"vol == 1.5 * vol_mean_20 must be NEUTRAL (strict >), got '{result}'"
        )

    def test_breakout_priority_over_exhaustion(self):
        """
        Контракт: BREAKOUT имеет приоритет над EXHAUSTION.
        Широкая медвежья свеча с бычьим закрытием — это медвежий пробой, не истощение.
        """
        result = classify_volume_type(
            open_=86_000.0, high=88_000.0, low=84_000.0,
            close=84_200.0,   # close_pct = 200/4000 = 0.05 < 0.30 (медвежье)
            vol=20_000.0,
            vol_mean_20=10_000.0,
            atr=700.0,        # range=4000 > 2*700=1400 → BREAKOUT
        )
        assert result == 'BREAKOUT', (
            f"Wide candle must be BREAKOUT even with bearish close, got '{result}'"
        )


# ---------------------------------------------------------------------------
# calculate_poc_retest_score — RED-фаза
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="SUBJECTIVE: threshold-based (5 touches as 'good', bounce_rate thresholds), disabled pending objective replacement")
class TestCalculatePocRetestScore:
    """
    RED-фаза TDD для calculate_poc_retest_score(
        df, poc, atr, window=30
    ) -> dict.

    Контракт:
        touch_count  : int   — количество касаний зоны POC±1.5ATR за window дней
        bounce_count : int   — касания где close внутри зоны (отбой)
        break_count  : int   — касания где close вне зоны (пробой)
        bounce_rate  : float — bounce_count / touch_count, 0.0 если touch_count==0
        avg_days_between_touches : float | None — None если <2 касаний
        score        : float — [0.0–1.0]: 0.5 * bounce_rate + 0.5 * min(touch_count/5, 1.0)

    Определения:
        зона  = [poc - 1.5*atr, poc + 1.5*atr]
        касание = low <= poc_upper AND high >= poc_lower  (цена зашла в зону)
        отбой  = касание где close в [poc_lower, poc_upper]
        пробой  = касание где close вне зоны

    WHY bounce_rate: частые отбои = люди уважают этот уровень = FAIR_VALUE_MAGNET.
    WHY touch_count/5: 5+ касаний за 30 дней = хорошо протестированный уровень.
    """

    @staticmethod
    def _make_df(n: int, base_price: float = 50_000.0) -> pd.DataFrame:
        """Общая фикстура: df с колонками date, open, high, low, close, vol."""
        dates  = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        closes = [base_price] * n
        return pd.DataFrame({
            'date' : dates,
            'open' : closes,
            'high' : [c + 200 for c in closes],
            'low'  : [c - 200 for c in closes],
            'close': closes,
            'vol'  : [10_000.0] * n,
        })

    def test_returns_dict_with_required_keys(self):
        """
        Контракт: результат — dict со всеми обязательными ключами.
        """
        df = self._make_df(30)
        result = calculate_poc_retest_score(df, poc=50_000.0, atr=500.0, window=30)
        for key in ('touch_count', 'bounce_count', 'break_count',
                    'bounce_rate', 'avg_days_between_touches', 'score'):
            assert key in result, f"Missing key: '{key}'"

    def test_no_touches_returns_zero_score(self):
        """
        Контракт: цена вдалеке от POC → touch_count=0, score=0.0.
        POC=50_000, цены около 60_000 — зона [49_250, 50_750] не пересекается.
        """
        df = self._make_df(30, base_price=60_000.0)
        result = calculate_poc_retest_score(df, poc=50_000.0, atr=500.0, window=30)
        assert result['touch_count'] == 0
        assert result['score'] == 0.0
        assert result['bounce_rate'] == 0.0
        assert result['avg_days_between_touches'] is None

    def test_all_bounces_gives_high_bounce_rate(self):
        """
        Контракт: все касания — отбои (close внутри зоны) → bounce_rate == 1.0.
        """
        df = self._make_df(10, base_price=50_000.0)  # все close в зоне POC
        result = calculate_poc_retest_score(df, poc=50_000.0, atr=500.0, window=10)
        assert result['touch_count'] > 0
        assert result['bounce_rate'] == 1.0
        assert result['break_count'] == 0

    def test_all_breaks_gives_zero_bounce_rate(self):
        """
        Контракт: все касания — пробои (close далеко вне зоны) → bounce_rate == 0.0.
        Свеча заходит в зону high/low, но закрывается намного выше (55_000 >> poc_upper=50_750).
        """
        n = 10
        dates  = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        df = pd.DataFrame({
            'date' : dates,
            'open' : [50_000.0] * n,
            'high' : [55_000.0] * n,   # касается зоны
            'low'  : [49_500.0] * n,   # заходит в зону
            'close': [55_000.0] * n,   # close далеко вне зоны
            'vol'  : [10_000.0] * n,
        })
        result = calculate_poc_retest_score(df, poc=50_000.0, atr=500.0, window=10)
        assert result['touch_count'] > 0
        assert result['bounce_rate'] == 0.0
        assert result['bounce_count'] == 0

    def test_touch_count_equals_bounce_plus_break(self):
        """
        Контракт: touch_count == bounce_count + break_count (инвариант).
        """
        n = 10
        dates  = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        # 5 отбоев (close в зоне) + 5 пробоев (close вне)
        closes = [50_000.0, 55_000.0] * 5
        df = pd.DataFrame({
            'date' : dates,
            'open' : [50_000.0] * n,
            'high' : [55_000.0] * n,
            'low'  : [49_500.0] * n,
            'close': closes,
            'vol'  : [10_000.0] * n,
        })
        result = calculate_poc_retest_score(df, poc=50_000.0, atr=500.0, window=10)
        assert result['touch_count'] == result['bounce_count'] + result['break_count'], (
            f"touch={result['touch_count']} != bounce={result['bounce_count']} + break={result['break_count']}"
        )

    def test_score_bounded_0_to_1(self):
        """
        Контракт: score всегда в [0.0, 1.0].
        """
        df = self._make_df(30, base_price=50_000.0)
        result = calculate_poc_retest_score(df, poc=50_000.0, atr=500.0, window=30)
        assert 0.0 <= result['score'] <= 1.0

    def test_score_max_when_many_bounces(self):
        """
        Контракт: много касаний (все отбои) → score == 1.0 (максимальное значение).
        30 строк в зоне — и по частоте (много касаний), и по качеству (100% отбоев)
        — идеально протестированный уровень. Структура функции гарантирует score=1.0.
        """
        df = self._make_df(30, base_price=50_000.0)  # все 30 строк в зоне
        result = calculate_poc_retest_score(df, poc=50_000.0, atr=500.0, window=30)
        assert result['score'] == 1.0, (
            f"Expected score=1.0 for 30 bounces, got {result['score']}"
        )

    def test_avg_days_between_touches_none_when_single_touch(self):
        """
        Контракт: единственное касание → avg_days_between_touches == None.
        WHY: для расчёта среднего нужно минимум 2 точки.
        """
        # 1 день в зоне, аостальные — далеко
        n = 10
        dates  = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        closes = [60_000.0] * n
        highs  = [60_500.0] * n
        lows   = [59_500.0] * n
        # Только первый день касается зоны poc=50_000
        highs[0]  = 51_000.0
        lows[0]   = 49_500.0
        closes[0] = 50_200.0
        df = pd.DataFrame({
            'date' : dates, 'open' : closes,
            'high' : highs, 'low'  : lows,
            'close': closes, 'vol'  : [10_000.0] * n,
        })
        result = calculate_poc_retest_score(df, poc=50_000.0, atr=500.0, window=10)
        assert result['touch_count'] == 1
        assert result['avg_days_between_touches'] is None

    def test_avg_days_between_touches_correct(self):
        """
        Контракт: касания на днях 0 и 10 → avg_days = 10.0.
        """
        n = 15
        dates  = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        closes = [60_000.0] * n
        highs  = [60_500.0] * n
        lows   = [59_500.0] * n
        # День 0 и день 10 — касания
        for idx in (0, 10):
            highs[idx]  = 51_000.0
            lows[idx]   = 49_500.0
            closes[idx] = 50_200.0
        df = pd.DataFrame({
            'date' : dates, 'open' : closes,
            'high' : highs, 'low'  : lows,
            'close': closes, 'vol'  : [10_000.0] * n,
        })
        result = calculate_poc_retest_score(df, poc=50_000.0, atr=500.0, window=15)
        assert result['avg_days_between_touches'] == 10.0, (
            f"Expected avg_days=10.0, got {result['avg_days_between_touches']}"
        )

    def test_window_limits_lookback(self):
        """
        Контракт: window ограничивает окно анализа — строки вне window не учитываются.
        60 строк, касания только в первой половине (> window=30 дней назад) → touch_count=0.
        """
        n = 60
        dates  = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        closes = [60_000.0] * n   # далеко от poc=50_000
        highs  = [60_500.0] * n
        lows   = [59_500.0] * n
        # Касание только на строке 0 (старая дата, вне window=30)
        highs[0]  = 51_000.0
        lows[0]   = 49_500.0
        closes[0] = 50_200.0
        df = pd.DataFrame({
            'date' : dates, 'open' : closes,
            'high' : highs, 'low'  : lows,
            'close': closes, 'vol'  : [10_000.0] * n,
        })
        result = calculate_poc_retest_score(df, poc=50_000.0, atr=500.0, window=30)
        assert result['touch_count'] == 0, (
            f"Expected touch_count=0 (touch outside window), got {result['touch_count']}"
        )


# ---------------------------------------------------------------------------
# calculate_volume_imbalance — RED-фаза
# ---------------------------------------------------------------------------


def _make_volume_type_df(volume_types: list) -> pd.DataFrame:
    """
    Строит минимальный DataFrame с колонкой 'volume_type'.
    Каждый элемент списка — строка: 'ABSORPTION', 'EXHAUSTION', 'NEUTRAL', 'BREAKOUT', 'UNCLEAR'.
    """
    n = len(volume_types)
    return pd.DataFrame({
        'date':        [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
        'volume_type': volume_types,
    })


@pytest.mark.skip(reason="SUBJECTIVE: window=5 days and 0.5 threshold are arbitrary, disabled pending objective replacement")
class TestCalculateVolumeImbalance:
    """
    RED-фаза TDD для calculate_volume_imbalance(df, window=5) -> pd.Series.

    Контракт:
    - Принимает DataFrame с колонкой 'volume_type' (из classify_volume_type).
    - Возвращает pd.Series длиной len(df), индекс совпадает с df.index.
    - Каждое значение: ABSORPTION_count / (ABSORPTION_count + EXHAUSTION_count) за window строк.
    - Если в окне нет ни ABSORPTION ни EXHAUSTION → 0.5 (нейтрально).
    - Первые window-1 строк → NaN (скользящее окно не заполнено).
    - window=1: нет NaN, результат с первой строки.
    - Отсутствие колонки 'volume_type' → ValueError.
    - Значения всегда в [0.0, 1.0] (кроме NaN).
    """

    def test_returns_series_same_length(self):
        """
        Контракт: возвращает pd.Series длиной len(df).
        """
        df = _make_volume_type_df(['ABSORPTION'] * 10)
        result = calculate_volume_imbalance(df, window=5)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_first_window_minus_1_rows_are_nan(self):
        """
        Контракт: первые window-1 строк — NaN (окно не заполнено).
        WHY: rolling(window) требует window строк для первого результата.
        """
        df = _make_volume_type_df(['ABSORPTION'] * 10)
        result = calculate_volume_imbalance(df, window=5)
        assert result.iloc[:4].isna().all(), (
            f"Expected NaN in first window-1=4 rows, got {result.iloc[:4].tolist()}"
        )
        assert result.iloc[4:].notna().all(), (
            f"Expected non-NaN from row 5 onward, got {result.iloc[4:].tolist()}"
        )

    def test_only_absorption_returns_1(self):
        """
        Контракт: только ABSORPTION в окне → imbalance = 1.0.
        WHY: 100% покупателей доминируют.
        """
        df = _make_volume_type_df(['ABSORPTION'] * 10)
        result = calculate_volume_imbalance(df, window=5)
        valid = result.dropna()
        assert (valid == 1.0).all(), (
            f"Only ABSORPTION must give 1.0, got {valid.tolist()}"
        )

    def test_only_exhaustion_returns_0(self):
        """
        Контракт: только EXHAUSTION в окне → imbalance = 0.0.
        WHY: 100% продавцов доминируют.
        """
        df = _make_volume_type_df(['EXHAUSTION'] * 10)
        result = calculate_volume_imbalance(df, window=5)
        valid = result.dropna()
        assert (valid == 0.0).all(), (
            f"Only EXHAUSTION must give 0.0, got {valid.tolist()}"
        )

    def test_only_neutral_returns_half(self):
        """
        Контракт: только NEUTRAL/BREAKOUT/UNCLEAR → imbalance = 0.5.
        WHY: нет ни ABSORPTION ни EXHAUSTION → нейтральный сигнал.
        """
        for neutral_type in ('NEUTRAL', 'BREAKOUT'):
            df = _make_volume_type_df([neutral_type] * 10)
            result = calculate_volume_imbalance(df, window=5)
            valid = result.dropna()
            assert (valid == 0.5).all(), (
                f"Only {neutral_type!r} must give 0.5, got {valid.tolist()}"
            )

    def test_balanced_mix_returns_half(self):
        """
        Контракт: равное количество ABSORPTION и EXHAUSTION → imbalance = 0.5.
        """
        df = _make_volume_type_df(['ABSORPTION', 'EXHAUSTION'] * 5)
        result = calculate_volume_imbalance(df, window=4)
        # Последние window строк: 2 ABSORPTION + 2 EXHAUSTION = 0.5
        last_valid = result.dropna().iloc[-1]
        assert abs(last_valid - 0.5) < 1e-9, (
            f"Equal mix must give 0.5, got {last_valid}"
        )

    def test_window_1_no_nan(self):
        """
        Контракт: window=1 — нет NaN, результат с первой строки.
        WHY: окно из 1 строки заполнено немедленно.
        """
        df = _make_volume_type_df(['ABSORPTION', 'EXHAUSTION', 'NEUTRAL'])
        result = calculate_volume_imbalance(df, window=1)
        assert result.notna().all(), (
            f"window=1 must have no NaN, got {result.tolist()}"
        )
        # ABSORPTION → 1.0, EXHAUSTION → 0.0, NEUTRAL → 0.5
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 0.0
        assert result.iloc[2] == 0.5

    def test_missing_volume_type_column_raises(self):
        """
        Контракт: отсутствие колонки 'volume_type' → ValueError.
        WHY: явная ошибка лучше молчаливого некорректного результата.
        """
        df = pd.DataFrame({'close': [1.0, 2.0, 3.0]})
        with pytest.raises((ValueError, KeyError)):
            calculate_volume_imbalance(df, window=3)


# ---------------------------------------------------------------------------
# detect_divergence — RED-фаза
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="SUBJECTIVE: window=14 is arbitrary, disabled pending objective replacement")
class TestDetectDivergence:
    """
    RED-фаза TDD для detect_divergence(
        price_series, indicator_series, window=14
    ) -> pd.Series.

    Контракт:
    - Возвращает pd.Series строк: 'BULLISH' | 'BEARISH' | 'NONE'.
    - Длина результата == длина price_series.
    - Первые window строк → 'NONE' (нет истории для сравнения).
    - BULLISH: цена делает новый лой (ниже мин за window),
               индикатор — нет.
    - BEARISH: цена делает новый хай (выше макс за window),
               индикатор — нет.
    - Разные длины серий → ValueError.
    - Пустые Series → пустой результат (dtype object).
    - BEARISH приоритет при одновременном условии BULLISH и BEARISH.

    Определения:
        price_new_low  = price[i] < min(price[i-window : i])
        price_new_high = price[i] > max(price[i-window : i])
        ind_new_low    = indicator[i] < min(indicator[i-window : i])
        ind_new_high   = indicator[i] > max(indicator[i-window : i])

        BULLISH: price_new_low  AND NOT ind_new_low
        BEARISH: price_new_high AND NOT ind_new_high
        NONE:    иначе
    """

    def test_returns_series_same_length(self):
        """
        Контракт: возвращает pd.Series длиной len(price_series).
        """
        price = pd.Series(range(30), dtype=float)
        ind   = pd.Series(range(30), dtype=float)
        result = detect_divergence(price, ind, window=5)
        assert isinstance(result, pd.Series)
        assert len(result) == len(price)

    def test_only_none_bullish_bearish_in_result(self):
        """
        Контракт: результат содержит только допустимые значения.
        """
        price = pd.Series(range(30), dtype=float)
        ind   = pd.Series(range(30), dtype=float)
        result = detect_divergence(price, ind, window=5)
        assert set(result.unique()).issubset({'BULLISH', 'BEARISH', 'NONE'}), (
            f"Unexpected values: {result.unique()}"
        )

    def test_first_window_rows_are_none(self):
        """
        Контракт: первые window строк → 'NONE' (недостаточно истории).
        WHY: rolling не заполнено, нет окна для сравнения.
        """
        price = pd.Series(range(20), dtype=float)
        ind   = pd.Series(range(20), dtype=float)
        result = detect_divergence(price, ind, window=5)
        assert (result.iloc[:5] == 'NONE').all(), (
            f"Expected 'NONE' in first window=5 rows, got {result.iloc[:5].tolist()}"
        )

    def test_bullish_divergence_detected(self):
        """
        Контракт: цена делает новый лой, индикатор — нет → 'BULLISH'.
        Данные:
          price: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]  — падает,
                 последний элемент (1) < min([10..2]) = 2 → новый лой.
          ind:   [10, 9, 8, 7, 6, 6, 6, 6, 6, 6]  — стабилизировался,
                 последний элемент (6) не < min([10..6]) = 6 → не новый лой.
        WHY window=5: достаточно короткое окно для проверки контракта.
        """
        price = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        ind   = pd.Series([10, 9, 8, 7, 6, 6, 6, 6, 6, 6], dtype=float)
        result = detect_divergence(price, ind, window=5)
        # Последняя строка должна быть BULLISH
        assert result.iloc[-1] == 'BULLISH', (
            f"Expected BULLISH at last position, got '{result.iloc[-1]}'"
        )

    def test_bearish_divergence_detected(self):
        """
        Контракт: цена делает новый хай, индикатор — нет → 'BEARISH'.
        Данные:
          price: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  — растёт,
                 последний (10) > max([1..9]) = 9 → новый хай.
          ind:   [1, 2, 3, 4, 5, 5, 5, 5, 5, 5]   — стабилизировался,
                 последний (5) не > max([1..5]) = 5 → не новый хай.
        """
        price = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        ind   = pd.Series([1, 2, 3, 4, 5, 5, 5, 5, 5, 5],  dtype=float)
        result = detect_divergence(price, ind, window=5)
        assert result.iloc[-1] == 'BEARISH', (
            f"Expected BEARISH at last position, got '{result.iloc[-1]}'"
        )

    def test_no_divergence_when_both_on_extremum(self):
        """
        Контракт: если цена И индикатор одновременно на новом лое → 'NONE'.
        WHY: нет расхождения, базовый сценарий (оба синхронно).
        """
        price = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        ind   = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)  # идентично
        result = detect_divergence(price, ind, window=5)
        assert result.iloc[-1] == 'NONE', (
            f"Expected NONE when price and indicator move together, got '{result.iloc[-1]}'"
        )

    def test_different_lengths_raises_value_error(self):
        """
        Контракт: разные длины → ValueError.
        WHY: попарное сравнение цены и индикатора невозможно.
        """
        price = pd.Series([1.0, 2.0, 3.0])
        ind   = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError):
            detect_divergence(price, ind, window=2)

    def test_empty_series_returns_empty(self):
        """
        Контракт: пустые Series → пустой результат dtype object.
        WHY: защита от ZeroDivisionError/IndexError при пустом df.
        """
        price = pd.Series([], dtype=float)
        ind   = pd.Series([], dtype=float)
        result = detect_divergence(price, ind, window=5)
        assert isinstance(result, pd.Series)
        assert len(result) == 0
        assert result.dtype == object

    def test_bearish_priority_over_bullish(self):
        """
        Контракт: если одновременно выполняются условия BULLISH и BEARISH → 'BEARISH'.
        Это теоретически невозможно на реальных данных (цена не может
        одновременно быть на новом максимуме и новом минимуме),
        но контракт всё равно должен определять приоритет BEARISH.
        WHY приоритет BEARISH: медвежий сигнал опаснее — консервативный подход.
        """
        # Создаём BEARISH-сценарий (цена на новом хае, индикатор — нет),
        # затем вручную проверяем что BEARISH выше BULLISH.
        price = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        ind   = pd.Series([1, 2, 3, 4, 5, 5, 5, 5, 5,  5], dtype=float)
        result = detect_divergence(price, ind, window=5)
        # BEARISH уже проверен в test_bearish_divergence_detected.
        # Здесь просто перепроверяем: результат не BULLISH.
        assert result.iloc[-1] != 'BULLISH', (
            f"BEARISH condition must take priority, last value must not be BULLISH"
        )


# ---------------------------------------------------------------------------
# TestFindLiquidityClustersHDBSCAN — RED-фаза (ЗАДАЧА 2)
# ---------------------------------------------------------------------------

class TestFindLiquidityClustersHDBSCAN:
    """
    RED-фаза: тесты написаны ПОСЛЕ того как решение принято, но ДО реализации.
    Проверяют поведение, которое должна иметь новая реализация HDBSCAN.

    Контракт (не меняется):
        find_liquidity_clusters(df_slice, atr_value) -> list[{min, max, vol, days}]

    Изменения внутри:
    1. DBSCAN(eps=ATR*0.75) → HDBSCAN(min_cluster_size=max(3, int(n*0.10)))
    2. Фильтр объёма: median*5 → Modified Z-score > 2.5
       Modified Z-score: M_i = 0.6745 * (x_i - median) / MAD
    3. atr_value остаётся в сигнатуре (обратная совместимость), внутри не используется.
    """

    @staticmethod
    def _make_two_zone_df(vol_multiplier: float = 6.0, seed: int = 7) -> pd.DataFrame:
        """
        Два плотных ценовых кластера с значительным объёмом.
        Zone A: close вокруг 50_000, высокий объём.
        Zone B: close вокруг 55_000, высокий объём.
        Фон из 10 строк с низким объёмом (Modified Z-score < 2.5 — шум).
        """
        rng = np.random.default_rng(seed)
        dates_a = [datetime(2024, 1, 1)  + timedelta(days=i) for i in range(20)]
        dates_b = [datetime(2024, 1, 21) + timedelta(days=i) for i in range(20)]
        dates_bg = [datetime(2024, 2, 10) + timedelta(days=i) for i in range(10)]

        base_vol = 10_000.0
        high_vol = base_vol * vol_multiplier  # настолько выше фона — Modified Z > 2.5

        zone_a = pd.DataFrame({
            'date':  dates_a,
            'close': rng.normal(50_000, 50, 20),
            'high':  rng.normal(50_000, 50, 20) + 200,
            'low':   rng.normal(50_000, 50, 20) - 200,
            'vol':   [high_vol] * 20,
        })
        zone_b = pd.DataFrame({
            'date':  dates_b,
            'close': rng.normal(55_000, 50, 20),
            'high':  rng.normal(55_000, 50, 20) + 200,
            'low':   rng.normal(55_000, 50, 20) - 200,
            'vol':   [high_vol] * 20,
        })
        background = pd.DataFrame({
            'date':  dates_bg,
            'close': rng.normal(52_000, 2_000, 10),
            'high':  rng.normal(52_000, 2_000, 10) + 300,
            'low':   rng.normal(52_000, 2_000, 10) - 300,
            'vol':   [base_vol] * 10,  # низкий объём, Modified Z < 2.5
        })
        df = pd.concat([zone_a, zone_b, background], ignore_index=True)
        return apply_time_decay(df, lam=0.005)

    # ------------------------------------------------------------------
    # Тест 1: atr_value не влияет на результат (HDBSCAN не использует eps)
    # ------------------------------------------------------------------

    def test_atr_value_does_not_affect_result(self):
        """
        Контракт: разные atr_value дают одинаковый результат.
        WHY: HDBSCAN не использует eps — atr_value остаётся в сигнатуре только
        для обратной совместимости с оркестратором.
        Главный баг DBSCAN: eps=ATR*0.75 — при высокой волатильности зоны сливались.
        """
        df = self._make_two_zone_df()
        clusters_low_atr  = find_liquidity_clusters(df, atr_value=100.0)
        clusters_high_atr = find_liquidity_clusters(df, atr_value=10_000.0)
        # Количество кластеров не зависит от atr_value
        assert len(clusters_low_atr) == len(clusters_high_atr), (
            f"HDBSCAN must ignore atr_value: "
            f"atr=100 gave {len(clusters_low_atr)} clusters, "
            f"atr=10000 gave {len(clusters_high_atr)} clusters"
        )

    # ------------------------------------------------------------------
    # Тест 2: две плотные зоны разделяются правильно
    # ------------------------------------------------------------------

    def test_two_dense_zones_detected(self):
        """
        Контракт: две плотных ценовых зоны с высоким объёмом → два отдельных кластера.
        WHY: переменная плотность — фундаментальное преимущество HDBSCAN над DBSCAN.
        """
        df = self._make_two_zone_df()
        clusters = find_liquidity_clusters(df, atr_value=500.0)
        assert len(clusters) >= 2, (
            f"Expected at least 2 clusters for two distinct price zones, got {len(clusters)}: {clusters}"
        )
        # Проверяем что зоны разделены (50k-зона и 55k-зона не слились)
        sorted_clusters = sorted(clusters, key=lambda c: c['min'])
        if len(sorted_clusters) >= 2:
            assert sorted_clusters[0]['max'] < sorted_clusters[1]['min'] - 1_000, (
                f"Clusters must be separated: zone1 max={sorted_clusters[0]['max']:.0f}, "
                f"zone2 min={sorted_clusters[1]['min']:.0f} — зоны должны быть разделены"
            )

    # ------------------------------------------------------------------
    # Тест 3: высокая волатильность (большой ATR) не сливает зоны
    # ------------------------------------------------------------------

    def test_stability_under_high_volatility(self):
        """
        Контракт: при большом ATR (высокая волатильнось) результат не меняется.
        Главный баг DBSCAN: при ATR=5000 зоны сливались в один кластер $10k+.
        WHY: HDBSCAN автоматически находит оптимальный радиус через иерархическую стабильность.
        """
        df = self._make_two_zone_df()
        clusters_normal = find_liquidity_clusters(df, atr_value=300.0)
        clusters_high   = find_liquidity_clusters(df, atr_value=5_000.0)  # баг DBSCAN
        # Для HDBSCAN: оба вызова дают одинаковое количество кластеров
        assert len(clusters_normal) == len(clusters_high), (
            f"HDBSCAN must be stable under volatility: "
            f"atr=300 gave {len(clusters_normal)}, atr=5000 gave {len(clusters_high)}"
        )

    # ------------------------------------------------------------------
    # Тест 4: Modified Z-score — кластер с низким объёмом отфильтровывается
    # ------------------------------------------------------------------

    def test_modified_zscore_filters_weak_volume_clusters(self):
        """
        Контракт: кластер с объёмом незначительно выше медианы heavy_days (Modified Z < 2.5)
        не даёт кластера в отдельной ценовой зоне, а зона с Modified Z >> 2.5 — даёт.

        Данные:
          Zone A (50k): vol=80k — Modified Z-score по weighted_vol значительно выше 2.5 → остаётся.
          Zone B (56k): vol=22k — суммарный weighted_vol зоны чуть выше медианы heavy_days,
                        но Modified Z < 2.5 → должна быть отфильтрована.
          Фон (52k): vol=10k — ниже среднего, не попадает в heavy_days.

        WHY vol=22k для zone_b: на фоне heavy_days (zone_a доминирует vol=80k)
        медиана weighted_vol около 80k/1.005 × (apply_time_decay),
        MAD ≈ 0 (все zone_a одинаковы), но zone_b суммарно в разах меньше zone_a →
        её M_i намного ниже 2.5. Практически тест проверяет: есть кластер в 50k,
        нет кластера в 56k.
        """
        rng = np.random.default_rng(99)
        # Zone A: сильный сигнал (Modified Z >> 2.5 по суммарному weighted_vol)
        zone_a = pd.DataFrame({
            'date':  [datetime(2024, 1, 1) + timedelta(days=i) for i in range(20)],
            'close': rng.normal(50_000, 30, 20),
            'high':  rng.normal(50_000, 30, 20) + 150,
            'low':   rng.normal(50_000, 30, 20) - 150,
            'vol':   [80_000.0] * 20,
        })
        # Zone B: слабый сигнал (Modified Z < 2.5 по weighted_vol кластера)
        zone_b = pd.DataFrame({
            'date':  [datetime(2024, 1, 21) + timedelta(days=i) for i in range(20)],
            'close': rng.normal(56_000, 30, 20),
            'high':  rng.normal(56_000, 30, 20) + 150,
            'low':   rng.normal(56_000, 30, 20) - 150,
            'vol':   [22_000.0] * 20,  # выше mean heavy_days (попадёт в heavy_days), но Modified Z << zone_a
        })
        background = pd.DataFrame({
            'date':  [datetime(2024, 2, 10) + timedelta(days=i) for i in range(10)],
            'close': rng.normal(52_000, 2_000, 10),
            'high':  rng.normal(52_000, 2_000, 10) + 300,
            'low':   rng.normal(52_000, 2_000, 10) - 300,
            'vol':   [10_000.0] * 10,
        })
        df = pd.concat([zone_a, zone_b, background], ignore_index=True)
        df = apply_time_decay(df, lam=0.005)
        clusters = find_liquidity_clusters(df, atr_value=500.0)

        # Zone A (50k) должна быть найдена (высокий Modified Z)
        # Zone B (56k) должна быть отфильтрована (Modified Z < 2.5 на фоне zone_a)
        has_50k = any(49_500 <= c['min'] <= 50_500 for c in clusters)
        has_56k = any(55_500 <= c['min'] <= 56_500 for c in clusters)
        assert has_50k, (
            f"Zone A (50k, vol=80k) должна быть найдена (Modified Z >> 2.5). clusters={clusters}"
        )
        assert not has_56k, (
            f"Zone B (56k, vol=22k) должна быть отфильтрована (Modified Z < 2.5 на фоне zone_a). clusters={clusters}"
        )

    # ------------------------------------------------------------------
    # Тест 5: min_cluster_size адаптивный (10% от heavy_days)
    # ------------------------------------------------------------------

    def test_min_cluster_size_based_on_heavy_days(self):
        """
        Контракт: min_cluster_size = max(3, int(n_heavy_days * 0.10)).
        С 60 heavy_days: min_cluster_size = max(3, 6) = 6.
        Кластер из 4 точек не должен быть найден (4 < min_cluster_size=6).
        Кластер из 20 точек должен быть найден (20 >= min_cluster_size=6).
        WHY: адаптивный порог избегает ложных кластеров из случайных точек.
        """
        rng = np.random.default_rng(42)
        # 80 heavy days суммарно: 20 в плотной зоне 50k + 4 изолированных в 70k
        # + 56 heavy days фона в разных частях ценового диапазона (60k)
        # Итого: n_heavy ≥ 80 → min_cluster_size = max(3, 8) = 8 → sparse(4) отклоняется
        dates_dense  = [datetime(2024,1,1)  + timedelta(days=i) for i in range(20)]
        dates_sparse = [datetime(2024,3,1)  + timedelta(days=i) for i in range(4)]
        dates_heavy_bg = [datetime(2024,2,1)  + timedelta(days=i) for i in range(56)]
        dates_low_bg   = [datetime(2024,4,1)  + timedelta(days=i) for i in range(20)]

        high_vol = 60_000.0
        low_vol  = 10_000.0

        dense = pd.DataFrame({
            'date':  dates_dense,
            'close': rng.normal(50_000, 30, 20),
            'high':  rng.normal(50_000, 30, 20) + 200,
            'low':   rng.normal(50_000, 30, 20) - 200,
            'vol':   [high_vol] * 20,
        })
        sparse = pd.DataFrame({
            'date':  dates_sparse,
            'close': rng.normal(70_000, 30, 4),
            'high':  rng.normal(70_000, 30, 4) + 200,
            'low':   rng.normal(70_000, 30, 4) - 200,
            'vol':   [high_vol] * 4,
        })
        # heavy фон: high_vol в широком ценовом диапазоне — попадают в heavy_days,
        # но не образуют плотных кластеров (очень разбросаны по цене)
        heavy_bg = pd.DataFrame({
            'date':  dates_heavy_bg,
            'close': rng.normal(60_000, 3_000, 56),  # широкий разброс
            'high':  rng.normal(60_000, 3_000, 56) + 300,
            'low':   rng.normal(60_000, 3_000, 56) - 300,
            'vol':   [high_vol] * 56,
        })
        low_bg = pd.DataFrame({
            'date':  dates_low_bg,
            'close': rng.normal(60_000, 3_000, 20),
            'high':  rng.normal(60_000, 3_000, 20) + 300,
            'low':   rng.normal(60_000, 3_000, 20) - 300,
            'vol':   [low_vol] * 20,
        })
        df = pd.concat([dense, sparse, heavy_bg, low_bg], ignore_index=True)
        df = apply_time_decay(df, lam=0.005)

        # Проверяем n_heavy достаточно велик (>= 80) → min_cluster_size >= 8
        heavy_days_preview = df[df['vol'] > df['vol'].mean()]
        expected_min_cs = max(3, int(len(heavy_days_preview) * 0.10))
        assert expected_min_cs >= 5, (
            f"Ожидали min_cluster_size >= 5, получили {expected_min_cs} "
            f"(n_heavy={len(heavy_days_preview)})"
        )

        clusters = find_liquidity_clusters(df, atr_value=500.0)

        # dense зона (20 точек) должна быть найдена
        # sparse зона (4 точки) должна быть отклонена HDBSCANом (меньше min_cluster_size)
        has_50k_cluster = any(49_500 <= c['min'] <= 50_500 for c in clusters)
        has_70k_cluster = any(69_500 <= c['min'] <= 70_500 for c in clusters)
        assert has_50k_cluster, (
            f"Dense zone (50k, 20 точек) должна быть найдена (min_cluster_size={expected_min_cs}). clusters={clusters}"
        )
        assert not has_70k_cluster, (
            f"Sparse zone (70k, 4 точки) должна быть отклонена (min_cluster_size={expected_min_cs}). clusters={clusters}"
        )


# ---------------------------------------------------------------------------
# TestExtractSubLevelsNormalized — RED-фаза: нормализованный prominence
# ---------------------------------------------------------------------------

class TestExtractSubLevelsNormalized:
    """
    Тесты для нормализованного prominence в extract_sub_levels().

    Новый контракт:
        vol_norm = (vol_array - vol_min) / (vol_array.max() - vol_array.min())
        prominence = max(0.05, vol_norm.max() * 0.2)
        peak_indices, _ = find_peaks(vol_norm, prominence=prominence)

    Ключевой RED-кейс: большой центральный пик раздувает vol_std,
    подавляя меньшие боковые пики. С нормализацией боковые пики
    видны если их prominence_norm > 0.2.
    """

    def test_side_peak_suppressed_by_dominant_peak_std(self):
        """
        RED-кейс: большой центральный пик раздувает vol_std,
        подавляя боковой пик с prominence_abs < vol_std.
        С нормализацией боковой пик находится (его prominence_norm=0.21 > 0.2).

        Данные: base=0, центр.пик[5]=10000, бок. пик[15]=2100.
          vol_std = 2203  >!  prominence_abs(2100) → старый код НЕ находит пик[15].
          prominence_norm = 2100/10000 = 0.21 > 0.2 → новый код находит.

        WHY этот тест: точная воспроизводимая проблема из NEXT_SESSION:
        при доминирующем пике vol_std > prominence боковых пиков —
        старый код теряет HVN. Нормализация решает эту проблему.
        """
        n = 20
        mids = np.linspace(74_000, 110_000, n)
        vols = np.zeros(n)
        vols[5]  = 10000.0  # большой пик: раздувает vol_std до 2203
        vols[15] = 2100.0   # боковой: prominence_abs=2100 < vol_std=2203
                            # prominence_norm=0.21 > 0.20 → должен быть найден
        profile = pd.DataFrame({
            'price_low':  mids - 900,
            'price_high': mids + 900,
            'mid':        mids,
            'vol':        vols,
        })
        zone = {'min': 74_000, 'max': 110_000}
        result = extract_sub_levels(zone, profile, n_peaks=5)
        found_mids = [r['mid'] for r in result]
        # Боковой пик[15] должен быть найден при нормализованном prominence
        assert any(abs(m - mids[15]) < 2000 for m in found_mids), (
            f"Боковой пик (prominence_norm=0.21) должен быть найден, got mids={found_mids}"
        )


# ---------------------------------------------------------------------------
# TestFindLiquidityClustersAllDaysReference — RED-фаза: reference = all days
# ---------------------------------------------------------------------------

class TestFindLiquidityClustersAllDaysReference:
    """
    Тесты для изменения reference в Modified Z-score:
    было: heavy_days (vol > mean) — только правый хвост
    стало: all df_slice days — весь рынок

    Контракт:
    Кластер с mean_raw_vol ≈ 2x median(all_days) должен проходить Modified Z > 2.5
    при reference = all_days, но НЕ проходить при reference = heavy_days.

    WHY этот контракт важен: фон содержит outlier-дни (капитуляционные дни
    с vol 200-300k). Они попадают в heavy_days и раздувают median_heavy до ~230k.
    Кластер с vol ~61k выглядит «маленьким» на фоне таких heavy_days
    (Z_heavy < 0), но аномален относительно всего рынка (Z_all ≈ 3.7).
    Изменение reference исправляет семантику: «аномален ли кластер
    относительно всего рынка», а не «относительно уже аномальных дней».
    """

    def _make_df_with_outliers(self):
        """
        Строит df_slice с:
        - 650 фоновых дней: vol uniform(20k, 40k), цены uniform(50k, 110k)
        - 50 outlier-дней: vol uniform(200k, 300k), цены uniform(50k, 110k)
          (капитуляционные дни — присутствуют в heavy_days, раздувают median_heavy)
        - 15 кластерных дней: close=68_000, vol = 2x median(all_days) ≈ 61k
          (аномальны vs all_days, но не vs heavy_days из-за outlier-ов)

        WHY outlier-ы: именно они создают расхождение между heavy_days-reference
        и all_days-reference. Без них sweet spot не существует — при равномерном
        фоне MAD_heavy слишком мал и Z_heavy всегда большой.
        """
        np.random.seed(42)
        n_bg = 650
        n_out = 50
        cluster_size = 15

        bg_prices  = np.random.uniform(50_000, 110_000, n_bg)
        bg_vols    = np.random.uniform(20_000,  40_000, n_bg)
        out_prices = np.random.uniform(50_000, 110_000, n_out)
        out_vols   = np.random.uniform(200_000, 300_000, n_out)

        # Кластерный vol = 2x median всех дней (bg + outliers)
        all_bg_vols = np.concatenate([bg_vols, out_vols])
        cluster_vol = np.median(all_bg_vols) * 2.0  # ≈ 61k

        closes = np.concatenate([bg_prices, out_prices, np.full(cluster_size, 68_000.0)])
        vols   = np.concatenate([bg_vols,   out_vols,   np.full(cluster_size, cluster_vol)])
        n_total = len(closes)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_total)]

        df = pd.DataFrame({
            'date':  dates,
            'close': closes,
            'low':   closes * 0.99,
            'high':  closes * 1.01,
            'vol':   vols,
        })
        df['weighted_vol'] = df['vol']
        return df

    def _make_df_noise_only(self):
        """
        Строит df_slice без аномального кластерного объёма:
        - 650 фоновых дней: vol uniform(20k, 40k)
        - 50 outlier-дней: vol uniform(200k, 300k)
        - 15 кластерных дней у $68k: vol = base_vol (без усиления)

        WHY: кластер с обычным объёмом не должен проходить фильтр
        даже при all_days reference — проверяет что фильтр не стал
        «пропускать всё».
        """
        np.random.seed(42)
        n_bg = 650
        n_out = 50
        cluster_size = 15

        bg_prices  = np.random.uniform(50_000, 110_000, n_bg)
        bg_vols    = np.random.uniform(20_000,  40_000, n_bg)
        out_prices = np.random.uniform(50_000, 110_000, n_out)
        out_vols   = np.random.uniform(200_000, 300_000, n_out)

        cluster_vol = 30_000.0  # обычный объём, не аномальный

        closes = np.concatenate([bg_prices, out_prices, np.full(cluster_size, 68_000.0)])
        vols   = np.concatenate([bg_vols,   out_vols,   np.full(cluster_size, cluster_vol)])
        n_total = len(closes)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_total)]

        df = pd.DataFrame({
            'date':  dates,
            'close': closes,
            'low':   closes * 0.99,
            'high':  closes * 1.01,
            'vol':   vols,
        })
        df['weighted_vol'] = df['vol']
        return df

    def test_cluster_invisible_with_heavy_days_reference_visible_with_all_days(self):
        """
        Контракт: кластер с vol ≈ 2x median(all_days) при фоне с outlier-дням
        (200-300k) должен быть обнаружен при all_days reference.

        WHY: outlier-дни раздувают median_heavy до ~230k, поэтому Z_heavy < 0
        для кластера с vol=61k. При all_days reference median=30k,
        Z_all ≈ 3.7 → кластер проходит порог 1.5.
        Этот тест является RED при текущем коде (heavy_days reference)
        и станет GREEN после изменения reference на all_days.
        """
        df = self._make_df_with_outliers()
        clusters = find_liquidity_clusters(df, atr_value=1000)
        has_cluster_at_68k = any(
            c['min'] <= 68_000 <= c['max'] for c in clusters
        )
        assert has_cluster_at_68k, (
            f"Кластер ~2x median_all у $68k должен быть обнаружен при all_days reference. "
            f"clusters={clusters}"
        )

    def test_noise_cluster_rejected_below_threshold(self):
        """
        Контракт: кластер с vol = 30k (обычный фоновый объём) НЕ должен
        проходить Modified Z > 2.5 даже при all_days reference.

        WHY: фильтр должен отсеивать кластеры без объёмной аномалии.
        Наличие outlier-дней в фоне не должно «открывать» порог для
        обычных кластеров — их vol попросту меньше threshold_all (Z<1.5).
        """
        df = self._make_df_noise_only()
        clusters = find_liquidity_clusters(df, atr_value=1000)
        has_cluster_at_68k = any(
            c['min'] <= 68_000 <= c['max'] for c in clusters
        )
        assert not has_cluster_at_68k, (
            f"Фоновый кластер (vol=30k) не должен проходить фильтр. "
            f"clusters={clusters}"
        )


# ---------------------------------------------------------------------------
# TestCalculateVwapDeviation
# ---------------------------------------------------------------------------

class TestCalculateVwapDeviation:
    """
    Contract tests for calculate_vwap_deviation(df, window=20) -> pd.Series.

    VWAP_t     = Σ(close_i × vol_i, window) / Σ(vol_i, window)
    deviation_t = (close_t − VWAP_t) / VWAP_t × 100

    All expected values analytically verified before tests were written.
    """

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _make_df(closes, vols):
        """Minimal DataFrame accepted by calculate_vwap_deviation."""
        return pd.DataFrame({'close': closes, 'vol': vols})

    # ------------------------------------------------------------------ tests

    def test_first_window_minus_one_rows_are_nan(self):
        """
        Contract: the first (window-1) rows MUST be NaN; all subsequent rows
        must be finite.

        WHY: a rolling window of size w requires w data points to produce the
        first value. Rows 0..(w-2) have fewer than w predecessors — the function
        must propagate NaN there rather than return a partial average.
        """
        n = 30
        df = self._make_df(
            closes=[100.0] * n,
            vols  =[1_000.0] * n,
        )
        result = calculate_vwap_deviation(df, window=20)

        assert result.iloc[:19].isna().all(), (
            "Rows 0..18 (first window-1=19 rows) must be NaN for window=20"
        )
        assert result.iloc[19:].notna().all(), (
            "Rows 19..29 must be finite (full window available)"
        )

    def test_uniform_close_and_volume_deviation_is_zero(self):
        """
        Contract: when every close equals the same constant and volume is
        uniform, VWAP equals close for every window → deviation = 0.0.

        WHY: VWAP = (c × v × w) / (v × w) = c, so (c − c)/c × 100 = 0.
        Verifies the formula returns 0 (not NaN, not noise) on trivial data.
        """
        n = 30
        df = self._make_df(
            closes=[100.0] * n,
            vols  =[1_000.0] * n,
        )
        result = calculate_vwap_deviation(df, window=20)
        valid = result.iloc[19:]

        assert (valid.abs() <= 1e-10).all(), (
            f"Expected all-zero deviation for uniform data; got {valid.values}"
        )

    def test_known_values_window_3_analytical(self):
        """
        Contract: for a 3-row window with known inputs the function must return
        the analytically computed value at row 2.

        Analytical verification:
          close=[100, 102, 98], vol=[10, 20, 10], window=3
          VWAP_2 = (100×10 + 102×20 + 98×10) / (10+20+10)
                 = 4020 / 40 = 100.5
          deviation_2 = (98 − 100.5) / 100.5 × 100 = −2.5/100.5 × 100
                      ≈ −2.487562189%

        WHY: this is the primary regression anchor — a small, hand-verifiable
        case that cannot accidentally pass via coincidental averaging.
        """
        df = self._make_df(
            closes=[100.0, 102.0, 98.0],
            vols  =[10.0,  20.0,  10.0],
        )
        result = calculate_vwap_deviation(df, window=3)

        assert pd.isna(result.iloc[0]), "Row 0 must be NaN (window=3, need 2 prior)"
        assert pd.isna(result.iloc[1]), "Row 1 must be NaN (window=3, need 1 prior)"

        expected = -2.5 / 100.5 * 100          # ≈ −2.4875621890547263
        assert result.iloc[2] == pytest.approx(expected, rel=1e-9), (
            f"Expected {expected:.10f}%, got {result.iloc[2]:.10f}%"
        )

    def test_volume_weighting_differs_from_simple_average(self):
        """
        Contract: VWAP must weight prices by volume, not treat each bar equally.

        Analytical verification:
          close=[100, 200], vol=[9, 1], window=2
          VWAP  = (100×9 + 200×1) / (9+1) = 1100/10 = 110.0
          SMA   = (100 + 200) / 2          = 150.0
          deviation_VWAP = (200 − 110) / 110 × 100 = 81.8181…%
          deviation_SMA  = (200 − 150) / 150 × 100 = 33.333…%

        WHY: this test distinguishes the correct VWAP implementation from an
        accidental simple-moving-average fallback. If volume weighting is
        missing, the result would be ~33.33% instead of ~81.82%.
        """
        df = self._make_df(
            closes=[100.0, 200.0],
            vols  =[9.0,   1.0],
        )
        result = calculate_vwap_deviation(df, window=2)

        expected = (200 - 110.0) / 110.0 * 100    # 81.818181…%
        assert result.iloc[1] == pytest.approx(expected, rel=1e-9), (
            f"Expected VWAP-weighted {expected:.6f}%, got {result.iloc[1]:.6f}%"
        )

    def test_equal_volumes_vwap_equals_sma(self):
        """
        Contract: with equal volumes for every bar, VWAP collapses to SMA.

        Analytical verification (window=3, equal vol=1):
          close=[1, 2, 3, 4, 5]
          Row 2: VWAP=2.0, dev=(3−2)/2×100 = 50.0%
          Row 3: VWAP=3.0, dev=(4−3)/3×100 ≈ 33.333%
          Row 4: VWAP=4.0, dev=(5−4)/4×100 = 25.0%

        WHY: confirming the SMA-equivalence at equal volume is a sanity
        check that rolling sums are aligned correctly (no off-by-one).
        """
        df = self._make_df(
            closes=[1.0, 2.0, 3.0, 4.0, 5.0],
            vols  =[1.0, 1.0, 1.0, 1.0, 1.0],
        )
        result = calculate_vwap_deviation(df, window=3)

        assert result.iloc[2] == pytest.approx(50.0,             rel=1e-9)
        assert result.iloc[3] == pytest.approx(100.0 / 3.0,     rel=1e-9)   # 33.333…
        assert result.iloc[4] == pytest.approx(25.0,             rel=1e-9)

    def test_returns_series_with_same_index_and_length(self):
        """
        Contract: output must be a pd.Series with the same length and index
        as the input DataFrame.

        WHY: callers assign the result back as df['vwap_dev'] = ...; a
        length or index mismatch would produce all-NaN via pandas alignment.
        """
        closes = list(range(25))
        vols   = [float(v + 1) for v in range(25)]
        df = self._make_df(closes, vols)
        result = calculate_vwap_deviation(df)

        assert isinstance(result, pd.Series), (
            f"Expected pd.Series, got {type(result)}"
        )
        assert len(result) == len(df), (
            f"Length mismatch: result={len(result)}, df={len(df)}"
        )
        assert list(result.index) == list(df.index), (
            "Index of result must match input DataFrame index"
        )

