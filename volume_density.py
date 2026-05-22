"""
volume_density.py
=================
Institutional Volume Profile analysis for BTC/USDT.

Module-level pure functions (build_profile, calculate_value_area,
find_liquidity_clusters, apply_time_decay, calculate_atr) are extracted
from liquidity_density_audit() so they can be unit-tested without any
exchange or network calls.

Exchange-dependent helpers (get_precise_volume, audit_level_details) stay
inside the orchestrator because they close over `exchange` and `symbol` —
testing them requires mocking the exchange, which belongs in integration tests.
"""

import asyncio
import os
import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime
from sklearn.cluster import HDBSCAN

from onchain_client import BGeometricsClient
from onchain_validator import OnChainValidator



# ---------------------------------------------------------------------------
# Pure module-level functions — fully testable without network
# ---------------------------------------------------------------------------

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR) for dynamic zone tolerance. [cite: 31, 92]

    Args:
        df: DataFrame with columns high, low, close.
        period: Rolling window (default 14).

    Returns:
        Series of ATR values aligned to df index.
    """
    high_low = df['high'] - df['low']
    high_cp = np.abs(df['high'] - df['close'].shift())
    low_cp  = np.abs(df['low']  - df['close'].shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_vwap_deviation(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Rolling VWAP deviation of close price (in percent).

    VWAP_t      = Σ(close_i × vol_i, window) / Σ(vol_i, window)
    deviation_t = (close_t − VWAP_t) / VWAP_t × 100

    Returns NaN for the first (window-1) rows where a full window
    is not yet available. pandas rolling().sum() propagates NaN
    automatically for incomplete windows.

    Args:
        df:     DataFrame with columns 'close' and 'vol'.
        window: Rolling window size (default 20 days).

    Returns:
        pd.Series of percent deviation values, same index as df.
    """
    rolling_pv  = (df['close'] * df['vol']).rolling(window).sum()
    rolling_vol = df['vol'].rolling(window).sum()
    vwap = rolling_pv / rolling_vol
    return (df['close'] - vwap) / vwap * 100


def apply_time_decay(df: pd.DataFrame, lam: float = 0.005) -> pd.DataFrame:
    """
    Hyperbolic time-decay: older volume loses significance. [cite: 53, 54]

    Adds columns days_ago, time_weight, weighted_vol to a COPY of df.
    Does NOT mutate the original DataFrame.

    WHY hyperbolic (1/(1+lam*t)) over exponential: smoother decay curve,
    avoids near-zero weights for data that is only 1-2 years old.

    Args:
        df:  DataFrame with a 'date' (datetime) and 'vol' column.
        lam: Decay coefficient. 0.005 for base levels, 0.02 for impulse zones.

    Returns:
        New DataFrame with added decay columns.
    """
    out = df.copy()
    out['days_ago']     = (out['date'].max() - out['date']).dt.days
    out['time_weight']  = 1 / (1 + lam * out['days_ago'])
    out['weighted_vol'] = out['vol'] * out['time_weight']
    return out


def build_profile(df_slice: pd.DataFrame, global_bins: np.ndarray) -> pd.DataFrame:
    """
    Build a volume profile on a fixed global bin grid. [cite: 12, 14, 56]

    All time windows use the same grid so profiles are directly comparable.

    WHY 100 bins for 2y range (~55 k wide): step ≈ $550 — fine enough to
    resolve institutional levels without noise from sparse bins.

    Args:
        df_slice:    Slice of the main DataFrame (already time-filtered).
                     Must contain columns: low, high, weighted_vol.
        global_bins: 1-D array of bin edges (e.g. np.linspace(min, max, 101)
                     gives 100 bins).

    Returns:
        DataFrame with columns: price_low, price_high, mid, vol.
    """
    profile = []
    for i in range(len(global_bins) - 1):
        mask = (
            (df_slice['low']  <= global_bins[i + 1]) &
            (df_slice['high'] >= global_bins[i])
        )
        bin_vol = df_slice.loc[mask, 'weighted_vol'].sum()
        profile.append({
            'price_low':  global_bins[i],
            'price_high': global_bins[i + 1],
            'mid':        (global_bins[i] + global_bins[i + 1]) / 2,
            'vol':        bin_vol,
        })
    return pd.DataFrame(profile)


def calculate_value_area(
    df_slice: pd.DataFrame,
    global_bins: np.ndarray,
    percentage: float = 0.70,
) -> tuple:
    """
    Point of Control (POC) and Value Area from the volume profile. [cite: 14, 15, 34]

    Args:
        df_slice:   Time-filtered DataFrame (see build_profile).
        global_bins: Bin edges array.
        percentage:  Value Area target (default 70 %).

    Returns:
        (va_low, va_high, poc_price) — all as float mid-prices, or
        (None, None, None) if total volume is zero.
    """
    prof_df   = build_profile(df_slice, global_bins)
    total_vol = prof_df['vol'].sum()

    if total_vol == 0:
        return None, None, None

    sorted_prof = prof_df.sort_values('vol', ascending=False).copy()
    sorted_prof['cum_vol'] = sorted_prof['vol'].cumsum()

    # Value Area: bins that together contain `percentage` of total volume
    va_bins   = sorted_prof[sorted_prof['cum_vol'] <= total_vol * percentage]
    va_low    = va_bins['mid'].min()
    va_high   = va_bins['mid'].max()
    poc_price = sorted_prof.iloc[0]['mid']

    return va_low, va_high, poc_price


def find_liquidity_clusters(
    df_slice: pd.DataFrame,
    atr_value: float,
) -> list:
    """
    Two-stage liquidity cluster detection. [cite: 27, 37]

    Stage 1 — HDBSCAN: price-proximity clustering on above-average-volume days.
               min_cluster_size = max(3, int(n_heavy_days * 0.10)) — adaptive,
               scales with data density. atr_value retained in signature for
               backward compatibility with orchestrator but NOT used internally.
    Stage 2 — Volume post-filter: Modified Z-score > 1.5 on cluster weighted_vol.
               M_i = 0.6745 * (x_i - median) / MAD
               Threshold 1.5 (~93rd percentile) identifies above-normal institutional volume.
               Calibrated on BTC 2024-2026 data: max observed Z=2.19, threshold 2.5 yields 0 clusters.

    WHY HDBSCAN over DBSCAN: DBSCAN(eps=ATR*0.75) is unstable — in trending
    markets ATR grows and zones merge into $10k+ ranges; in sideways ATR
    shrinks and zones fragment. HDBSCAN finds optimal radius automatically
    via hierarchical cluster stability, requiring only min_cluster_size.

    WHY Modified Z-score over median*5: robust to extreme outliers (uses MAD
    not std), scale-invariant, standard statistical anomaly threshold.

    WHY 10% of heavy_days for min_cluster_size: adapts to data density —
    avoids false clusters from isolated spikes in low-data windows.

    Args:
        df_slice:  Time-filtered DataFrame with vol, close, weighted_vol.
        atr_value: Retained for backward compatibility — not used internally.

    Returns:
        List of dicts: {min, max, vol, days}. Empty list if no clusters found.
    """
    heavy_days = df_slice[df_slice['vol'] > df_slice['vol'].mean()].copy()
    if heavy_days.empty:
        return []

    prices = heavy_days['close'].values.reshape(-1, 1)

    # Stage 1: HDBSCAN — adaptive spatial density clustering
    # WHY max(3, ...): minimum viable cluster even for small windows (<30 days)
    min_cluster_size = max(3, int(len(heavy_days) * 0.10))
    model = HDBSCAN(min_cluster_size=min_cluster_size).fit(prices)
    # WHY reset_index: heavy_days is a slice with non-contiguous index;
    # assigning model.labels_ (0..N array) directly causes pandas index
    # misalignment and fills with NaN. reset_index ensures correct alignment.
    heavy_days = heavy_days.reset_index(drop=True)
    heavy_days['cluster'] = model.labels_

    # Stage 2: Modified Z-score volume filter on raw (unweighted) daily vol
    # WHY raw vol, not weighted_vol: time_decay compresses all values into a
    # narrow range (e.g. 50k–55k) when data spans only weeks — MAD becomes
    # too small to discriminate. Raw vol preserves the original anomaly signal
    # (institutional day = 3–10x normal vol) regardless of decay window.
    # A cluster passes if its mean raw daily vol has Modified Z > 2.5 relative
    # to ALL df_slice days (not just heavy_days).
    # WHY all days, not heavy_days: heavy_days are defined as vol > mean(vol),
    # i.e. the right tail. Their median is shifted right, so a cluster with
    # vol ≈ 2x market median (a normal active day) gets Modified Z < 1.0
    # against heavy_days reference. Using all_days gives correct semantics:
    # "is this cluster anomalous relative to the whole market?"
    cluster_labels = [c for c in set(model.labels_) if c != -1]
    if not cluster_labels:
        return []

    cluster_vols = np.array([
        heavy_days[heavy_days['cluster'] == c]['weighted_vol'].sum()
        for c in cluster_labels
    ])
    cluster_sizes = np.array([
        len(heavy_days[heavy_days['cluster'] == c])
        for c in cluster_labels
    ])
    # Mean raw (unweighted) vol per day for each cluster
    mean_raw_vols = np.array([
        heavy_days[heavy_days['cluster'] == c]['vol'].mean()
        for c in cluster_labels
    ])

    # Reference: raw vol distribution of ALL df_slice days
    all_raw_vols = df_slice['vol'].values
    median_raw   = np.median(all_raw_vols)
    mad_raw      = np.median(np.abs(all_raw_vols - median_raw))

    clusters = []
    for c, cluster_vol, mean_raw in zip(cluster_labels, cluster_vols, mean_raw_vols):
        # Modified Z-score: M_i = 0.6745 * (x_i - median) / MAD
        # WHY 0.6745: scaling factor so MAD-based Z is consistent with std-based Z
        # WHY MAD==0 guard: all heavy_days have identical raw vol — uniform market,
        # no discrimination possible — keep all clusters
        if mad_raw > 0:
            modified_z = 0.6745 * (mean_raw - median_raw) / mad_raw
            if modified_z <= 1.5:
                continue
        # If MAD==0 all vols are equal — keep all
        c_data = heavy_days[heavy_days['cluster'] == c]
        clusters.append({
            'min':  c_data['close'].min(),
            'max':  c_data['close'].max(),
            'vol':  cluster_vol,
            'days': len(c_data),
        })
    return clusters


def detect_absorption_days(df: pd.DataFrame, atr_value: float) -> pd.DataFrame:
    """
    Детектирует дни пассивного поглощения на дневных свечах.
    Адаптация AccumulationDetector._check_passive_absorption() для OHLCV.

    Args:
        df:        DataFrame с колонками open, high, low, close, vol, date.
                   Должен содержать weighted_vol (после apply_time_decay).
        atr_value: Текущий ATR для фильтрации шума (защита от нулевого диапазона).

    Returns:
        df с новой колонкой 'absorption' (bool).

    Признаки дня поглощения (все три обязательны):
    - vol > 20-дневного rolling_mean * 1.5  (аномальный объём)
    - (close - low) / (high - low) > 0.70   (закрытие в верхних 30% = бычье)
    - close < close.shift(5)                 (нисходящий контекст)
    """
    vol_mean_20 = df['vol'].rolling(20).mean()
    high_vol    = df['vol'] > vol_mean_20 * 1.5

    candle_range = df['high'] - df['low']
    # WHY replace(0, atr_value): защита от дожи с нулевым диапазоном
    safe_range   = candle_range.replace(0, atr_value)
    close_pct    = (df['close'] - df['low']) / safe_range
    closed_high  = close_pct > 0.70   # верхние 30% диапазона = бычье поглощение

    downtrend    = df['close'] < df['close'].shift(5)

    out = df.copy()   # WHY copy: не мутируем оригинал
    out['absorption'] = high_vol & closed_high & downtrend
    return out


def extract_sub_levels(
    zone: dict,
    profile_df: pd.DataFrame,
    n_peaks: int = 3,
) -> list:
    """
    Находит High Volume Nodes (HVN) внутри широкой DBSCAN-зоны.

    Использует scipy.signal.find_peaks с prominence-фильтром чтобы выделить
    локальные максимумы объёма внутри зоны. Reuses уже вычисленный profile_df
    из build_profile() — без дополнительных вычислений.

    WHY prominence, не просто peaks: prominence гарантирует что пик реально
    выделяется над соседями, а не просто чуть выше среднего (шум).
    WHY не sub-DBSCAN: один параметр (prominence=std) vs два (eps + min_samples).

    Args:
        zone:       dict с ключами 'min' и 'max' (выход find_liquidity_clusters).
        profile_df: DataFrame из build_profile() — колонки mid, vol.
        n_peaks:    Максимальное кол-во sub-levels для возврата (default 3).

    Returns:
        list[dict]: [{mid: float, vol: float}, ...] отсортированный по vol desc.
        Пустой список если пиков не найдено или зона вне профиля.
    """
    from scipy.signal import find_peaks

    mask = (profile_df['mid'] >= zone['min']) & (profile_df['mid'] <= zone['max'])
    zone_profile = profile_df[mask].copy()

    if zone_profile.empty:
        return []

    vol_array = zone_profile['vol'].values
    vol_min   = vol_array.min()
    vol_max   = vol_array.max()

    # WHY нормализация [0,1]: dominantный пик с огромным abs-объёмом раздувал
    # vol_std, подавляя боковые пики с prominence_abs < vol_std. Нормализация
    # делает порог относительным: пик должен выступать на >20% диапазона зоны.
    # WHY max(0.05, ...): защита от нулевого диапазона при крайне плоских зонах.
    if vol_max == vol_min:
        return []

    vol_norm   = (vol_array - vol_min) / (vol_max - vol_min)
    prominence = max(0.05, vol_norm.max() * 0.2)
    peak_indices, _ = find_peaks(vol_norm, prominence=prominence)

    if len(peak_indices) == 0:
        return []

    peaks_df = zone_profile.iloc[peak_indices][['mid', 'vol']]
    result   = (
        peaks_df
        .nlargest(n_peaks, 'vol')
        .to_dict('records')
    )
    return result


def calculate_poc_quality_score(
    absorption_days_near_poc: int,
    total_days_near_poc: int,
    volume_w_score: float,
    capitulation_confirmed: bool,
    z_score: float,
    delta_context_score: float = 0.5,
    oi_regime: str = 'NEUTRAL',
    lth_proxy_sopr: float = None,
) -> dict:
    """
    Агрегирующая метрика качества POC.

    Args:
        absorption_days_near_poc: Дней с поглощением в зоне POC ± 1.5 ATR.
        total_days_near_poc:      Всего дней в зоне POC.
        volume_w_score:           W-Score из audit_level_details [0–100].
        capitulation_confirmed:   True если LTH убытки > $300M/день × 3 дня.
        z_score:                  Z-Score от STH realized price.
        delta_context_score:      Дельта-скор [0.0–1.0] из calculate_delta_context_score.
        oi_regime:                Режим OI из classify_oi_regime() (дефолт 'NEUTRAL').

    Returns:
        dict: {score: float, label: str, flags: list}

    Классификация:
    score > 0.65  → 'FAIR_VALUE_MAGNET'
    score < 0.35  → 'RESISTANCE_TRAP'
    иначе   → 'NEUTRAL'

    Коррекции delta_context_score:
        delta_context_score < 0.35 AND volume_score > 0.6
            → label принудительно 'RESISTANCE_TRAP'
        delta_context_score > 0.65
            → absorption_score += 0.10 перед расчётом итога

    Коррекции oi_regime (применяются после взвешенного score, до clamp):
        'LIQUIDATION' → score += 0.10 (лонги ликвидируются = дно близко)
        'STRONG_BEAR' → score -= 0.10 (новые шорты = капитуляция не завершена)
        Другие режимы — без влияния.
        score после OI-коррекции clamp в [0.0, 1.0].
    """
    import math

    # absorption_score: доля absorption-дней вблизи POC
    if total_days_near_poc > 0:
        absorption_score = absorption_days_near_poc / total_days_near_poc
    else:
        absorption_score = 0.0   # WHY: защита от ZeroDivisionError

    # WHY до расчёта score: boost влияет на взвешенный итог, не на label напрямую
    if delta_context_score > 0.65:
        absorption_score += 0.10

    # onchain_score: 1.0 при капитуляции, иначе sigmoid(z_score)
    if capitulation_confirmed:
        onchain_score = 1.0
    else:
        onchain_score = 1.0 / (1.0 + math.exp(-z_score))  # sigmoid [-∞,+∞] → [0,1]

    # volume_score: нормализация W-Score в [0, 1]
    volume_score = max(0.0, min(1.0, volume_w_score / 100.0))

    # Итоговый взвешенный скор
    score = (
        0.40 * absorption_score
        + 0.35 * onchain_score
        + 0.25 * volume_score
    )

    # --- Этап 4: OI-коррекция score (применяется до clamp и классификации) ---
    # WHY до clamp: нужно сначала изменить score, потом ограничить, потом round()
    if oi_regime == 'LIQUIDATION':
        score += 0.10   # лонги ликвидируются → дно близко → бычьий сигнал
    elif oi_regime == 'STRONG_BEAR':
        score -= 0.10   # новые шорты → капитуляция не завершена → медвежьий сигнал
    # WHY clamp: OI-коррекция не должна выводить score за [0, 1]
    score = max(0.0, min(1.0, score))

    # Классификация (после clamp)
    if score > 0.65:
        label = 'FAIR_VALUE_MAGNET'
    elif score < 0.35:
        label = 'RESISTANCE_TRAP'
    else:
        label = 'NEUTRAL'

    # Принудительный override: медвежья дельта при высоком объёме = ловушка
    # WHY оба условия: только дельта без объёма — слабый сигнал
    if delta_context_score < 0.35 and volume_score > 0.6:
        label = 'RESISTANCE_TRAP'

    flags = []
    if capitulation_confirmed and absorption_score > 0.3:
        flags.append('BULLISH_DIVERGENCE_ACCUMULATION')

    # WHY None-guard: lth_proxy_sopr опциональный — старые вызовы без него не ломаются
    if lth_proxy_sopr is not None and lth_proxy_sopr < 0.60:
        flags.append('LTH_CAPITULATION_ZONE')

    return {'score': round(score, 3), 'label': label, 'flags': flags}


def classify_funding_regime(funding_pct: float) -> str:
    """
    Классифицирует режим фандинга по величине ставки (в процентах).

    Args:
        funding_pct: Текущая ставка фандинга в %, напр. -0.012 = -0.012%.
                     Базовая ставка Binance USDM = +0.01%.

    Returns:
        str: Один из пяти режимов:
            'NEGATIVE_EXTREME'   -- funding < -0.05% (глубокий шорт-перевес)
            'NEGATIVE_MODERATE'  -- -0.05% <= funding < -0.01%
            'NEUTRAL'            -- -0.01% <= funding <= +0.01%
            'POSITIVE_MODERATE'  -- +0.01% < funding <= +0.05%
            'POSITIVE_EXTREME'   -- funding > +0.05% (лонг-перегрев)

    WHY пять зон: базовая ставка Binance +0.01% входит в NEUTRAL.
    WHY порог +-0.05%: эмпирически на BTC +-0.05% -- редкие события,
    отражающие сильный дисбаланс позиций. Пороги изменяются как константы,
    не как архитектура.
    """
    if funding_pct < -0.05:
        return 'NEGATIVE_EXTREME'
    elif funding_pct < -0.01:
        return 'NEGATIVE_MODERATE'
    elif funding_pct <= 0.01:
        return 'NEUTRAL'
    elif funding_pct <= 0.05:
        return 'POSITIVE_MODERATE'
    else:
        return 'POSITIVE_EXTREME'


def classify_market_regime(
    oi_regime: str,
    funding_regime: str,
) -> str:
    """
    Агрегирует OI-режим + режим фандинга в единый рыночный режим.

    Pure function. Не делает сетевых вызовов, полностью тестируемая без mock.
    Используется после classify_oi_regime() и classify_funding_regime()
    для финального синтеза в [FINAL VERDICT].

    Args:
        oi_regime:      Режим OI из classify_oi_regime().
                        Ожидаемые значения: 'STRONG_BULL', 'WEAK_BULL',
                        'STRONG_BEAR', 'LIQUIDATION', 'NEUTRAL'.
        funding_regime: Режим фандинга из classify_funding_regime().
                        Ожидаемые значения: 'NEGATIVE_EXTREME',
                        'NEGATIVE_MODERATE', 'NEUTRAL',
                        'POSITIVE_MODERATE', 'POSITIVE_EXTREME'.

    Returns:
        str: Один из шести режимов:
            'OVERHEATED_BULL' -- STRONG_BULL + POSITIVE_EXTREME
                                 (лонг-перегрев, риск разворота)
            'BULL'            -- STRONG_BULL без перегрева
            'CAPITULATION'    -- LIQUIDATION (принудительные ликвидации лонгов,
                                 независимо от funding)
            'BEAR'            -- STRONG_BEAR без POSITIVE_EXTREME,
                                 или WEAK_BULL с медвежьим funding
            'BEAR_SQUEEZE'    -- STRONG_BEAR + POSITIVE_EXTREME
                                 (аномалия: OI медвежий, фандинг бычий)
            'NEUTRAL'         -- всё остальное (включая неизвестные входы)

    WHY LIQUIDATION → CAPITULATION безусловно: принудительные ликвидации
    лонгов -- наиболее прямой сигнал дна цикла. Фандинг здесь
    неважен: даже POSITIVE_EXTREME funding во время liquidation
    означает flash-панику, не перегрев лонгов.
    WHY BEAR_SQUEEZE: STRONG_BEAR + POSITIVE_EXTREME -- аномалия, где
    OI растёт (новые шорты) но лонги платят шортам -- squeeze возможен.
    WHY WEAK_BULL + negative funding → BEAR: рост цены при падении OI --
    шорты ликвидируются, не покупатели открываются -- переход в BEAR идёт.
    """
    # Приоритет 1: LIQUIDATION безусловно (самый сильный сигнал дна)
    if oi_regime == 'LIQUIDATION':
        return 'CAPITULATION'

    # Приоритет 2: STRONG_BULL
    if oi_regime == 'STRONG_BULL':
        # WHY OVERHEATED_BULL только при POSITIVE_EXTREME:
        # POSITIVE_MODERATE = повышенный фандинг, но не перегрев.
        if funding_regime == 'POSITIVE_EXTREME':
            return 'OVERHEATED_BULL'
        return 'BULL'

    # Приоритет 3: STRONG_BEAR
    if oi_regime == 'STRONG_BEAR':
        # WHY BEAR_SQUEEZE: новые шорты + лонговый фандинг = squeeze-опасность
        if funding_regime == 'POSITIVE_EXTREME':
            return 'BEAR_SQUEEZE'
        return 'BEAR'

    # Приоритет 4: WEAK_BULL с медвежьим фандингом
    if oi_regime == 'WEAK_BULL' and funding_regime in (
        'NEGATIVE_EXTREME', 'NEGATIVE_MODERATE'
    ):
        # WHY BEAR: когда OI падает на росте цены и фандинг отрицательный,
        # рост вызван ликвидациями шортов, а не реальным спросом.
        return 'BEAR'

    # Всё остальное (NEUTRAL OI, WEAK_BULL + нейтральный фандинг, неизвестные режимы)
    return 'NEUTRAL'


def evaluate_poc_quality(
    absorption_days_near_poc: int,
    total_days_near_poc: int,
    volume_w_score: float,
    capitulation_confirmed: bool,
    z_score: float,
    delta_context_score: float = 0.5,
    oi_regime: str = 'NEUTRAL',
    lth_proxy_sopr: float = None,
    funding_regime: str = None,
) -> dict:
    """
    Теговая оценка качества POC. Заменяет calculate_poc_quality_score().

    Вместо взвешенного score использует явные теги с приоритетом:
      FAIR_VALUE_MAGNET  -- есть FAIR_VALUE_* тег, нет RESISTANCE_* тегов
      RESISTANCE_TRAP    -- есть RESISTANCE_* тег, нет FAIR_VALUE_* тегов
      NEUTRAL            -- конфликт (оба типа) или нет значимых тегов

    Args:
        absorption_days_near_poc: Дней с поглощением в зоне POC +/- 1.5 ATR.
        total_days_near_poc:      Всего дней в зоне POC.
        volume_w_score:           W-Score [0-100].
        capitulation_confirmed:   True если LTH убытки > $300M/день x 3 дня.
        z_score:                  Z-Score от STH realized price.
        delta_context_score:      Дельта-скор [0.0-1.0].
        oi_regime:                Режим OI из classify_oi_regime().
        lth_proxy_sopr:           Прокси SOPR LTH из calculate_lth_pain_proxy().
        funding_regime:           Режим фандинга из classify_funding_regime().

    Returns:
        dict: {'label': str, 'tags': list[str]}
        НЕ содержит 'score' -- теговая архитектура.
    """
    tags = []

    # --- Подготовка производных значений ---
    absorption_ratio = (
        absorption_days_near_poc / total_days_near_poc
        if total_days_near_poc > 0 else 0.0
    )
    volume_score = max(0.0, min(1.0, volume_w_score / 100.0))

    # --- FAIR_VALUE_MAGNET теги ---
    # WHY > 0.4: больше 40% дней с поглощением = значимый институциональный фотслед.
    if absorption_ratio > 0.4:
        tags.append('FAIR_VALUE_MAGNET_ABSORPTION')

    # WHY capitulation: прямой сигнал дна цикла для BTC.
    if capitulation_confirmed:
        tags.append('FAIR_VALUE_MAGNET_CAPITULATION')

    # WHY z > 1.0: STH в убытках = паника рынка, типичное дно.
    if z_score > 1.0:
        tags.append('FAIR_VALUE_MAGNET_STH_PRESSURE')

    # --- RESISTANCE_TRAP теги ---
    # WHY оба условия: медвежья дельта без объёма = шум; с объёмом = подтвержденное давление.
    if delta_context_score < 0.35 and volume_score > 0.6:
        tags.append('RESISTANCE_TRAP_DELTA')

    # WHY STRONG_BEAR: новые шорты открываются = капитуляция не завершена.
    if oi_regime == 'STRONG_BEAR':
        tags.append('RESISTANCE_TRAP_OI')

    # WHY POSITIVE_EXTREME: лонг-перегрев = рынок платит шортам, риск разворота.
    if funding_regime == 'POSITIVE_EXTREME':
        tags.append('RESISTANCE_TRAP_FUNDING')

    # --- Информационные теги (не влияют на label) ---
    if lth_proxy_sopr is not None and lth_proxy_sopr < 0.60:
        tags.append('LTH_CAPITULATION_ZONE')

    if oi_regime == 'LIQUIDATION' and absorption_ratio > 0.3:
        tags.append('BULLISH_DIVERGENCE')

    # --- Агрегация: метка из тегов по правилу приоритета ---
    # WHY приоритет: детерминированный результат, аналитик видит причину из тегов.
    has_fair   = any(t.startswith('FAIR_VALUE_') for t in tags)
    has_resist = any(t.startswith('RESISTANCE_') for t in tags)

    if has_fair and not has_resist:
        label = 'FAIR_VALUE_MAGNET'
    elif has_resist and not has_fair:
        label = 'RESISTANCE_TRAP'
    else:
        label = 'NEUTRAL'

    return {'label': label, 'tags': tags}


def load_oi_history(
    csv_path: str,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Загружает исторический OI из локального CSV (Binance Vision format).

    Args:
        csv_path:   Путь к CSV-файлу (например data/metrics/BTCUSDT-metrics-daily.csv).
        start_date: Опционально, формат 'YYYY-MM-DD', включительно.
        end_date:   Опционально, формат 'YYYY-MM-DD', включительно.

    Returns:
        DataFrame с колонками:
            date (datetime64), sum_open_interest (float), oi_change_pct (float).
        oi_change_pct = pct_change() * 100 (первая строка — NaN по определению).
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['sum_open_interest'] = df['sum_open_interest'].astype(float)

    # WHY sort_values: CSV от Binance Vision обычно уже сортирован, но гарантируем merge-совместимость
    df = df.sort_values('date').reset_index(drop=True)

    if start_date is not None:
        df = df[df['date'] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df['date'] <= pd.Timestamp(end_date)]

    df = df.reset_index(drop=True)
    # WHY * 100: сохраняем единицы измерения с price_change_pct в classify_oi_regime()
    df['oi_change_pct'] = df['sum_open_interest'].pct_change() * 100

    return df[['date', 'sum_open_interest', 'oi_change_pct']]


def calculate_bin_delta(
    trades: list,
    price_low: float,
    price_high: float,
) -> float:
    """
    Delta = buy_volume - sell_volume в ценовом бине [price_low, price_high].

    Адаптация Этапа 5 (из TECHNICAL_SPEC_POC_Quality.md).

    Args:
        trades:     list[dict] со ключами 'price' (float), 'amount' (float), 'side' (str).
                    side='buy' | 'sell'.
                    Источник: exchange.fetch_trades() или aggTrades Binance.
        price_low:  Нижняя граница бина (включительно).
        price_high: Верхняя граница бина (включительно).

    Returns:
        float: положительное = давление покупателей (ИНСТИТУЦИОНАЛЬНЫЙ магнит),
                отрицательное = давление продавцов (RESISTANCE_TRAP-сигнал).

    WHY закрытый интервал [low, high]: согласовано с build_profile(), где
    бин (включает обе границы) совпадает с проверкой low <= price_high AND high >= price_low.
    """
    buy_vol = sum(
        t['amount'] for t in trades
        if price_low <= t['price'] <= price_high and t['side'] == 'buy'
    )
    sell_vol = sum(
        t['amount'] for t in trades
        if price_low <= t['price'] <= price_high and t['side'] == 'sell'
    )
    return float(buy_vol - sell_vol)


def classify_oi_regime(
    price_change_pct: float,
    oi_change_pct: float,
) -> str:
    """
    Классификация режима Open Interest по матрице Price x OI.
    Адаптация SmartCandle.get_trend_fuel() для дневных данных.

    Args:
        price_change_pct: Изменение цены за период, в процентах.
        oi_change_pct:    Изменение Open Interest за период, в процентах.

    Returns:
        'STRONG_BULL'  — Price > threshold AND OI > threshold (новые лонги)
        'WEAK_BULL'    — Price > threshold AND OI <= threshold (short covering)
        'STRONG_BEAR'  — Price <= threshold AND OI > threshold (новые шорты)
        'LIQUIDATION'  — Price <= threshold AND OI <= threshold (лонги ликвидируются)
        'NEUTRAL'      — зарезервировано для будущих расширений

    WHY strict > для threshold: значение ровно на границе (1.0%) не является
    значимым сигналом — требуется явное пересечение порога.
    """
    threshold = 1.0   # WHY 1%: фильтр шума; движения < 1% — внутридневной шум
    price_up  = price_change_pct > threshold
    oi_up     = oi_change_pct    > threshold

    if price_up and oi_up:
        return 'STRONG_BULL'
    if price_up and not oi_up:
        return 'WEAK_BULL'
    if not price_up and oi_up:
        return 'STRONG_BEAR'
    return 'LIQUIDATION'


def calculate_basis_spread(
    spot_price: float,
    futures_price: float,
) -> dict:
    """
    Спред между spot и futures ценой BTC.

    Args:
        spot_price:    Цена BTC на спот-рынке (Binance BTCUSDT).
        futures_price: Цена BTC на фьючерсном рынке (Binance BTCUSDT perp).

    Returns:
        dict: {
            'basis_usd':  float,   # futures - spot в USD
            'basis_pct':  float,   # (futures - spot) / spot * 100
            'regime':     str,     # 'CONTANGO' | 'BACKWARDATION' | 'FLAT'
        }

    Классификация:
        basis_pct > +0.1%  → 'CONTANGO'      (futures дороже spot = бычий)
        basis_pct < -0.1%  → 'BACKWARDATION' (futures дешевле spot = медвежий)
        иначе              → 'FLAT'

    WHY 0.1% порог: базовая ставка Binance perp funding = 0.01%/8ч = ~0.1%/день.
    Спред в пределах одной ставки фандинга — нейтральный.
    """
    # WHY нет guard: spot=0 бросает ZeroDivisionError по контракту (см. тест).
    # На продакшне spot никогда не 0 — явная ошибка лучше silent NaN.
    basis_usd = futures_price - spot_price
    basis_pct = basis_usd / spot_price * 100

    if basis_pct > 0.1:
        regime = 'CONTANGO'
    elif basis_pct < -0.1:
        regime = 'BACKWARDATION'
    else:
        regime = 'FLAT'

    return {
        'basis_usd': float(basis_usd),
        'basis_pct': float(basis_pct),
        'regime':    regime,
    }


def classify_volume_type(
    open_: float,
    high: float,
    low: float,
    close: float,
    vol: float,
    vol_mean_20: float,
    atr: float,
) -> str:
    """
    Классифицирует дневную свечу по типу объёма + формы.

    Расширяет detect_absorption_days() (давала только bool) до семантической метки.
    Работает на одной свече (не DataFrame) — применима через df.apply() или построчно.

    Args:
        open_:       Цена открытия.
        high:        Максимум.
        low:         Минимум.
        close:       Цена закрытия.
        vol:         Объём свечи.
        vol_mean_20: 20-дневная скользящая средняя (должна быть подсчитана вне).
        atr:         Текущий ATR — защита от дожи + порог BREAKOUT.

    Returns:
        str: 'ABSORPTION' | 'EXHAUSTION' | 'BREAKOUT' | 'NEUTRAL'

    Классификация (high_vol = vol > vol_mean_20 * 1.5):

        1. Нет high_vol         → NEUTRAL
        2. high_vol + range > 2*ATR → BREAKOUT  (приоритет над остальными)
        3. high_vol + close_pct > 0.70 → ABSORPTION
        4. high_vol + close_pct < 0.30 → EXHAUSTION
        5. high_vol + 0.30 <= close_pct <= 0.70 → NEUTRAL

    WHY BREAKOUT первый: широкая свеча — самый сильный сигнал; закрытие
    верхнее/нижнее неважно при пробое.
    WHY 2*ATR для BREAKOUT: ATR = средний дневной диапазон; > 2x = выход за норму.
    WHY 0.70/0.30 для ABSORPTION/EXHAUSTION: согласовано с detect_absorption_days().
    """
    high_vol = vol > vol_mean_20 * 1.5

    if not high_vol:
        return 'NEUTRAL'

    candle_range = high - low
    # WHY atr вместо 0: дожи (range=0) — защищаем от ZeroDivisionError,
    # аналогично detect_absorption_days()
    safe_range  = candle_range if candle_range > 0 else atr
    close_pct   = (close - low) / safe_range

    # BREAKOUT: широкая свеча — высший приоритет независимо от закрытия
    if candle_range > 2.0 * atr:
        return 'BREAKOUT'

    if close_pct > 0.70:
        return 'ABSORPTION'

    if close_pct < 0.30:
        return 'EXHAUSTION'

    # high_vol есть, но закрытие в средине диапазона — нет чёткого сигнала
    return 'NEUTRAL'


def calculate_poc_retest_score(
    df: pd.DataFrame,
    poc: float,
    atr: float,
    window: int = 30,
) -> dict:
    """
    Оценивает надёжность POC по истории ретестов цены.

    Args:
        df:     DataFrame с колонками date, high, low, close.
        poc:    Point of Control — центр зоны.
        atr:    Average True Range — ширина зоны (±1.5 ATR).
        window: Количество последних дней для анализа (default 30).

    Returns:
        dict:
          touch_count  (int)          — кол-во касаний зоны POC ±1.5ATR
          bounce_count (int)          — отбои (close внутри зоны)
          break_count  (int)          — пробои (close вне зоны)
          bounce_rate  (float)        — bounce_count / touch_count, 0.0 если 0 касаний
          avg_days_between_touches (float | None) — None если <2 касаний
          score (float)               — [0.0, 1.0]

    Определения:
        порог  = poc ± 1.5 * atr
        касание = low <= poc_upper AND high >= poc_lower
        отбой  = касание где close в [poc_lower, poc_upper]
        пробой  = касание где close вне зоны

    Формула score:
        score = 0.5 * bounce_rate + 0.5 * min(touch_count / 5.0, 1.0)
        WHY 0.5/0.5: оба фактора равноважны — частота без уважения ненадёжна,
        уважение без частоты недостаточно.
        WHY 5: эмпирически 5 касаний за 30 дней = хорошо протестированный уровень.

    WHY ±1.5ATR: согласовано с detect_absorption_days() и audit_level_details().
    """
    poc_lower = poc - 1.5 * atr
    poc_upper = poc + 1.5 * atr

    # Ограничиваем окно анализа последними window строками
    df_window = df.tail(window).copy()

    # Касания: свеча зашла в зону (low <= poc_upper AND high >= poc_lower)
    touch_mask = (
        (df_window['low']  <= poc_upper) &
        (df_window['high'] >= poc_lower)
    )
    touches = df_window[touch_mask].copy()

    touch_count = len(touches)

    if touch_count == 0:
        return {
            'touch_count':              0,
            'bounce_count':             0,
            'break_count':              0,
            'bounce_rate':              0.0,
            'avg_days_between_touches': None,
            'score':                    0.0,
        }

    # Отбои: close внутри зоны
    bounce_mask  = touches['close'].between(poc_lower, poc_upper)
    bounce_count = int(bounce_mask.sum())
    break_count  = touch_count - bounce_count

    bounce_rate = bounce_count / touch_count

    # Среднее кол-во дней между касаниями
    # WHY через индексы df: '.date' может отсутствовать — используем позиционные индексы
    avg_days_between_touches: float | None = None
    if touch_count >= 2:
        # Позиции касаний в df_window (0..window-1)
        touch_positions = [i for i, idx in enumerate(df_window.index) if idx in touches.index]
        gaps = [touch_positions[i+1] - touch_positions[i]
                for i in range(len(touch_positions) - 1)]
        avg_days_between_touches = float(sum(gaps) / len(gaps))

    score = 0.5 * bounce_rate + 0.5 * min(touch_count / 5.0, 1.0)
    score = round(min(max(score, 0.0), 1.0), 4)

    return {
        'touch_count':              touch_count,
        'bounce_count':             bounce_count,
        'break_count':              break_count,
        'bounce_rate':              round(bounce_rate, 4),
        'avg_days_between_touches': avg_days_between_touches,
        'score':                    score,
    }


def calculate_volume_imbalance(
    df: pd.DataFrame,
    window: int = 5,
) -> pd.Series:
    """
    Скользящий дисбаланс объёма: доля ABSORPTION относительно (ABSORPTION + EXHAUSTION)
    за последние `window` дней.

    Использует колонку 'volume_type' из classify_volume_type().

    Args:
        df:     DataFrame с колонкой 'volume_type'.
                Ожидаемые значения: 'ABSORPTION', 'EXHAUSTION', 'BREAKOUT', 'NEUTRAL'.
        window: Размер скользящего окна (default 5).

    Returns:
        pd.Series длиной len(df), индекс совпадает с df.index.
        Каждое значение в [0.0, 1.0]:
            1.0  → только ABSORPTION в окне (покупатели доминируют)
            0.0  → только EXHAUSTION в окне (продавцы доминируют)
            0.5  → нейтрально (нет ни ABSORPTION ни EXHAUSTION в окне,
                   или равное их количество)
            NaN  → первые window-1 строк (окно не заполнено)

    WHY 0.5 при отсутствии сигналов: NEUTRAL/BREAKOUT не говорят о перевесе
    покупателей или продавцов — честный нейтральный дефолт.
    WHY rolling(window): сглаживает однодневные выбросы, даёт trend-сигнал.
    """
    if 'volume_type' not in df.columns:
        raise ValueError(
            "DataFrame must contain 'volume_type' column. "
            "Call classify_volume_type() first."
        )

    is_abs = (df['volume_type'] == 'ABSORPTION').astype(float)
    is_exh = (df['volume_type'] == 'EXHAUSTION').astype(float)

    abs_sum = is_abs.rolling(window).sum()
    exh_sum = is_exh.rolling(window).sum()
    total   = abs_sum + exh_sum

    # WHY два шага:
    # 1. total.isna() → rolling-окно не заполнено → оставляем NaN
    # 2. total == 0   → окно заполнено, но нет ни ABSORPTION ни EXHAUSTION
    #                   → нейтральное значение 0.5
    result = abs_sum / total                         # NaN где total==0 или rolling не заполнен
    result = result.where(total.isna() | (total > 0), other=0.5)  # 0.5 только где total==0
    return result


def detect_divergence(
    price_series: pd.Series,
    indicator_series: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Классическое дивергентное расхождение цены и индикатора.

    Args:
        price_series:     Серия цен (close, high или любой price).
        indicator_series: Серия индикатора (RSI, CVD, OI и др.).
                          Должна быть одинаковой длины с price_series.
        window:           Размер скользящего окна (default 14).

    Returns:
        pd.Series строк: 'BULLISH' | 'BEARISH' | 'NONE'.
        - Первые window строк → 'NONE' (нет истории для сравнения).

    Определения:
        BULLISH: price на новом лое (< rolling_min за window),
                 индикатор — нет (сигнал разворота вверх).
        BEARISH: price на новом хае (> rolling_max за window),
                 индикатор — нет (сигнал разворота вниз).
        BEARISH имеет приоритет при одновременном условии.

    WHY rolling соседей предшествующих (не включая текущую): расхождение
    определяется по тому, что было ДО текущей свечи, не включая её.
    """
    if len(price_series) != len(indicator_series):
        raise ValueError(
            f"price_series and indicator_series must have the same length, "
            f"got {len(price_series)} and {len(indicator_series)}"
        )

    if len(price_series) == 0:
        return pd.Series([], dtype=object)

    # rolling min/max по window предыдущим строкам (не включая текущую).
    # WHY shift(1): текущая свеча не входит в окно сравнения —
    # расхождение определяется относительно предыдущей истории.
    price_roll_min = price_series.shift(1).rolling(window).min()
    price_roll_max = price_series.shift(1).rolling(window).max()
    ind_roll_min   = indicator_series.shift(1).rolling(window).min()
    ind_roll_max   = indicator_series.shift(1).rolling(window).max()

    price_new_low  = price_series < price_roll_min
    price_new_high = price_series > price_roll_max
    ind_new_low    = indicator_series < ind_roll_min
    ind_new_high   = indicator_series > ind_roll_max

    # WHY сначала BEARISH: медвежий сигнал опаснее — консервативный подход.
    result = pd.Series('NONE', index=price_series.index, dtype=object)
    result = result.where(~(price_new_low  & ~ind_new_low),  other='BULLISH')
    result = result.where(~(price_new_high & ~ind_new_high), other='BEARISH')

    # Первые window строк: rolling не заполнено → 'NONE'
    # WHY window, а не window-1: shift(1) + rolling(window) требует window+1 строк
    result.iloc[:window] = 'NONE'

    return result


# ---------------------------------------------------------------------------
# aggTrades functions — Этап 8
# ---------------------------------------------------------------------------

def load_aggtrades_zip(zip_path: str) -> pd.DataFrame:
    """
    Загружает один ZIP-файл aggTrades из Binance Vision.

    Формат CSV внутри ZIP:
        agg_trade_id, price, quantity, first_trade_id, last_trade_id,
        transact_time, is_buyer_maker

    Args:
        zip_path: Путь к ZIP-файлу (например BTCUSDT-aggTrades-2025-01.zip).

    Returns:
        DataFrame с колонками:
            price  (float32) — цена сделки
            qty    (float32) — объём сделки в базовой валюте
            side   (str)     — 'buy' или 'sell'

    WHY float32: экономит ~500 MB RAM на 55M строках (vs float64).
    WHY is_buyer_maker=True → 'sell': buyer является maker-ом (лимитный ордер
        на покупку стоял в стакане), taker продал в него → sell-initiated сделка.
    """
    import zipfile

    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            first_line = f.readline().decode('utf-8').strip()

    # Определяем наличие заголовка: если первое поле нечисловое — заголовок есть
    has_header = not first_line.split(',')[0].lstrip('-').isdigit()

    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            if has_header:
                df = pd.read_csv(
                    f,
                    header=0,
                    usecols=['price', 'quantity', 'is_buyer_maker'],
                    dtype={
                        'price':          'float32',
                        'quantity':       'float32',
                        'is_buyer_maker': 'bool',
                    },
                )
                df = df.rename(columns={'quantity': 'qty'})
            else:
                # Без заголовка: колонки по позиции [1]=price, [2]=qty, [6]=is_buyer_maker
                df = pd.read_csv(
                    f,
                    header=None,
                    usecols=[1, 2, 6],
                    dtype={1: 'float32', 2: 'float32', 6: 'bool'},
                )
                df.columns = ['price', 'qty', 'is_buyer_maker']

    # WHY np.where: векторизованное присвоение side без Python loop
    df['side'] = np.where(df['is_buyer_maker'], 'sell', 'buy')
    df = df.drop(columns='is_buyer_maker')

    return df[['price', 'qty', 'side']]


def calculate_cvd_in_zone(
    df_trades: pd.DataFrame,
    poc: float,
    atr: float,
) -> tuple:
    """
    Cumulative Volume Delta (CVD) в зоне POC ± 1.5 * ATR.

    Используется для Anchor Period (3–4 месяца): определяет характер уровня.
    CVD росла с ценой → там реальные лонги (POC = магнит).
    CVD падала при росте цены → крупный выходил "об толпу" (POC = ловушка).

    Args:
        df_trades: DataFrame с колонками price (float), qty (float), side (str).
                   Результат load_aggtrades_zip() или аналогичной загрузки.
        poc:       Point of Control — центр зоны анализа.
        atr:       Average True Range — определяет ширину зоны (±1.5 ATR).

    Returns:
        (cvd_series, cvd_slope):
            cvd_series  — pd.Series CVD (нарастающий итог buy_qty - sell_qty).
                          Пустая Series если нет сделок в зоне.
            cvd_slope   — float, наклон линейной регрессии CVD.
                          > 0: покупатели накапливают (бычий сигнал).
                          < 0: продавцы доминируют (медвежий сигнал).
                          0.0 если нет сделок в зоне.

    WHY ±1.5 ATR: согласовано с зоной POC в detect_absorption_days()
    и audit_level_details() — единый стандарт "зоны POC" в проекте.
    """
    zone_low  = poc - 1.5 * atr
    zone_high = poc + 1.5 * atr

    mask     = (df_trades['price'] >= zone_low) & (df_trades['price'] <= zone_high)
    zone_df  = df_trades[mask].copy()

    if zone_df.empty:
        return pd.Series(dtype=float), 0.0

    # delta: +qty для buy, -qty для sell
    zone_df['delta'] = np.where(
        zone_df['side'] == 'buy', zone_df['qty'], -zone_df['qty']
    )
    cvd_series = zone_df['delta'].cumsum().reset_index(drop=True)

    # Наклон линейной регрессии CVD — направление давления за период
    x = np.arange(len(cvd_series), dtype=float)
    # WHY np.polyfit degree=1: простейший линейный тренд, устойчив к выбросам
    coeffs    = np.polyfit(x, cvd_series.values, 1)
    cvd_slope = float(coeffs[0])   # slope = первый коэффициент

    return cvd_series, cvd_slope


def calculate_delta_context_score(
    cvd_slope: float,
    recent_delta: float,
) -> float:
    """
    Агрегирует два дельта-сигнала в единый скор [0.0, 1.0].

    Гибридный горизонт (из ТЗ):
    - cvd_slope    (Anchor Period, 3–4 мес): "качество" уровня — кто там застрял.
    - recent_delta (Reaction Period, 14 дней): "готовность рынка" — кто доминирует сейчас.

    Args:
        cvd_slope:    Наклон CVD за anchor period (результат calculate_cvd_in_zone).
                      > 0 → лонги накапливали → бычий сигнал.
                      < 0 → крупный выходил → медвежий сигнал.
        recent_delta: Суммарная дельта за последние 14 дней в зоне POC.
                      > 0 → покупатели доминируют.
                      < 0 → продавцы доминируют.

    Returns:
        float в [0.0, 1.0]:
            > 0.5 → бычий контекст (POC как магнит подтверждён).
            < 0.5 → медвежий контекст (POC как ловушка).
            = 0.5 → нейтрально (сигналы противоречивы или нулевые).

    Формула: sigmoid(W_ANCHOR * tanh(cvd_slope/1000) + W_REACTION * tanh(recent_delta/500))
    WHY tanh нормализация: сжимает любое число в (-1,+1), устойчива к выбросам.
    WHY веса 0.6/0.4: структурная память (anchor) важнее краткосрочного сигнала.
    WHY sigmoid на выходе: ограничивает результат в [0,1], симметричен вокруг 0.5.
    """
    import math

    W_ANCHOR   = 0.6   # WHY 0.6: структурная память важнее краткосрочного сигнала
    W_REACTION = 0.4

    # WHY масштаб 1000/500: типичные величины CVD/дельты в BTC (эмпирически из benchmark)
    norm_slope  = math.tanh(cvd_slope   / 1_000.0)
    norm_recent = math.tanh(recent_delta / 500.0)

    combined = W_ANCHOR * norm_slope + W_REACTION * norm_recent

    # sigmoid: (-∞, +∞) → (0, 1), симметрична вокруг 0 → 0.5
    score = 1.0 / (1.0 + math.exp(-combined))

    return float(score)


def get_anchor_months(
    df_ohlcv: pd.DataFrame,
    poc: float,
    atr: float,
    lookback_days: int = 120,
) -> list:
    """
    Определяет месяцы для скачивания aggTrades (анкерный период).

    Фильтрует df_ohlcv по последним lookback_days дням, затем извлекает
    уникальные (year, month) для строк, где close находился в зоне POC ± 1.5 * ATR.

    Args:
        df_ohlcv:     DataFrame с колонками 'date' (datetime64) и 'close' (float).
        poc:          Point of Control — центр зоны (из calculate_value_area).
        atr:          Average True Range — ширина зоны (±1.5 ATR).
        lookback_days: Количество последних дней df для анализа (default 120).

    Returns:
        list of (year: int, month: int) — сортирован хронологически.
        Пустой список если цена не попадала в зону в указанный период.

    WHY ±1.5 ATR: согласовано с detect_absorption_days() и audit_level_details()
    — единый стандарт "зоны POC" в проекте.
    WHY lookback_days=120: ~4 месяца — anchor period из архитектуры гибридного горизонта.
    """
    # Ограничиваем lookback window
    cutoff_date = df_ohlcv['date'].max() - pd.Timedelta(days=lookback_days)
    df_window   = df_ohlcv[df_ohlcv['date'] >= cutoff_date]

    # Фильтр: close в зоне POC ± 1.5 * ATR
    zone_low  = poc - 1.5 * atr
    zone_high = poc + 1.5 * atr
    zone_mask = df_window['close'].between(zone_low, zone_high)
    anchor_df = df_window[zone_mask]

    if anchor_df.empty:
        return []

    # Уникальные (year, month), хронологически отсортированные
    periods = anchor_df['date'].dt.to_period('M').unique()
    periods  = sorted(periods)  # Period сопоставим
    return [(int(p.year), int(p.month)) for p in periods]


# ---------------------------------------------------------------------------
# Orchestrator — requires exchange + on-chain network access
# ---------------------------------------------------------------------------

PROFILE_WINDOWS   = {"1w": 7, "2w": 14, "1m": 30, "3m": 90, "6m": 180, "1y": 365, "2y": 730}
DBSCAN_MIN_WINDOW = 30   # WHY: see find_liquidity_clusters docstring


def load_klines_zip(zip_path: str) -> pd.DataFrame:
    """Загружает daily 1m klines ZIP из Binance Vision.

    WHY детекция заголовка: Binance Vision добавил заголовок в новые файлы
    (с ~2026 года), старые файлы без заголовка. Паттерн аналогичен load_aggtrades_zip.
    Колонки по позиции: [0]=open_time, [2]=high, [3]=low, [5]=volume, [9]=taker_buy_vol.
    """
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            first_line = f.readline().decode('utf-8').strip()

    # Детектируем заголовок: если первое поле нечисловое — заголовок есть
    has_header = not first_line.split(',')[0].lstrip('-').isdigit()

    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            if has_header:
                df = pd.read_csv(
                    f,
                    header=0,
                    usecols=['open_time', 'high', 'low', 'volume', 'taker_buy_volume'],
                    dtype={
                        'open_time':        'int64',
                        'high':             'float32',
                        'low':              'float32',
                        'volume':           'float32',
                        'taker_buy_volume': 'float32',
                    },
                )
                df = df.rename(columns={'taker_buy_volume': 'taker_buy_vol'})
            else:
                df = pd.read_csv(
                    f,
                    header=None,
                    usecols=[0, 2, 3, 5, 9],
                    dtype={0: 'int64', 2: 'float32', 3: 'float32', 5: 'float32', 9: 'float32'},
                )
                df.columns = ['open_time', 'high', 'low', 'volume', 'taker_buy_vol']

    return df


def build_delta_profile(klines_df: pd.DataFrame, global_bins: np.ndarray) -> pd.DataFrame:
    """Строит Volume Profile с колонкой delta (buy_vol - sell_vol) по каждому бину."""
    profile = []
    for i in range(len(global_bins) - 1):
        if klines_df.empty:
            mask_vol = 0.0
            mask_buy = 0.0
        else:
            mask = (
                (klines_df['low']  <= global_bins[i + 1]) &
                (klines_df['high'] >= global_bins[i])
            )
            mask_vol = float(klines_df.loc[mask, 'volume'].sum())
            mask_buy = float(klines_df.loc[mask, 'taker_buy_vol'].sum())
        sell     = mask_vol - mask_buy
        delta    = mask_buy - sell
        profile.append({
            'price_low':  global_bins[i],
            'price_high': global_bins[i + 1],
            'mid':        (global_bins[i] + global_bins[i + 1]) / 2,
            'vol':        mask_vol,
            'delta':      delta,
        })
    return pd.DataFrame(profile)


def calculate_lth_pain_proxy(
    df_ohlcv: pd.DataFrame,
    window: int = 155,
) -> dict:
    """Volume-Weighted MA proxy аналог SOPR для LTH-когорты.

    Почему VWMA: SMA даёт одинаковый вес тихому боковику и панической распродаже с
    объёмом ×3. VWMA взвешивает цену по объёму — цена с высокой проторговкой
    влияет сильнее. Это ближе к реальному среднему входу LTH-когорты.

    Args:
        df_ohlcv: DataFrame с колонками 'close', 'vol', 'date'.
                  Тот же df что в оркестраторе, после apply_time_decay.
        window:   Период VWMA в днях (155 = стандартное определение LTH:
                  монеты не двигавшиеся > 155 дней).

    Returns:
        dict с ключами:
          proxy_sopr   (float | None) — close[-1] / vwma_155.
                                         None если len(df) < window.
          vwma_155     (float | None) — VWMA за window дней.
          phase        (str)          — Текущая фаза из таблицы:
                                         BULL / EARLY_BEAR / RUBICON /
                                         BEAR_PRESSURE / CAPITULATION /
                                         EXTREME / INSUFFICIENT_DATA.
          phase_comment (str)         — Человекочитаемое описание фазы.
          roc_14        (float | None) — Изменение proxy_sopr за 14 дней в %.
                                         + = proxy растёт (разворот).
          days_below_1  (int)          — Дней подряд с proxy_sopr < 1.0
                                         (считаются с конца).
    """
    # --- Таблица фаз ---
    _PHASES = [
        (1.10, float('inf'),  'BULL',          'LTH в прибыли >10%. Бычий рынок или ранняя коррекция.'),
        (1.00, 1.10,          'EARLY_BEAR',    'LTH у безубытка. Рубикон ещё не пройдён.'),
        (0.80, 1.00,          'RUBICON',       'LTH перешли в убыток. Начало фазы давления продаж.'),
        (0.65, 0.80,          'BEAR_PRESSURE', 'Убыток LTH 20–35%. Середина медвежьего цикла.'),
        (0.50, 0.65,          'CAPITULATION',  'Убыток LTH 35–50%. Исторически — зона дна (2015, 2018, 2022).'),
        (0.00, 0.50,          'EXTREME',       'Убыток LTH >50%. Экстремальная капитуляция (2011, 2015).'),
    ]

    _insufficient = {
        'proxy_sopr':    None,
        'vwma_155':      None,
        'phase':         'INSUFFICIENT_DATA',
        'phase_comment': f'Меньше {window} строк — VWMA-расчёт на неполном окне бессмыслен.',
        'roc_14':        None,
        'days_below_1':  0,
    }

    if len(df_ohlcv) < window:
        return _insufficient

    closes = df_ohlcv['close'].values.astype(float)
    vols   = df_ohlcv['vol'].values.astype(float)

    # VWMA за последние `window` строк
    w_closes = closes[-window:]
    w_vols   = vols[-window:]
    total_vol = w_vols.sum()
    if total_vol == 0:
        return _insufficient
    vwma = float(np.dot(w_closes, w_vols) / total_vol)

    current_close = float(closes[-1])
    proxy = current_close / vwma

    # Определяем фазу
    phase, phase_comment = 'EXTREME', 'Убыток LTH >50%. Экстремальная капитуляция (2011, 2015).'
    for low, high, p_name, p_comment in _PHASES:
        if low < proxy:   # proxy > low
            phase, phase_comment = p_name, p_comment
            break

    # roc_14: изменение proxy за 14 дней в %
    # WHY -15: proxy[-15] = 14 дней назад относительно proxy[-1]
    roc_14: float | None = None
    if len(df_ohlcv) >= window + 14:
        # Для VWMA 14 дней назад: используем окно [-window-14 : -14]
        w14_closes = closes[-window - 14 : -14]
        w14_vols   = vols[-window - 14 : -14]
        total_vol_14 = w14_vols.sum()
        if total_vol_14 > 0:
            vwma_14ago = float(np.dot(w14_closes, w14_vols) / total_vol_14)
            proxy_14ago = float(closes[-15]) / vwma_14ago
            if proxy_14ago != 0:
                roc_14 = (proxy - proxy_14ago) / abs(proxy_14ago) * 100.0

    # days_below_1: подряд идущие дни с proxy < 1.0 с конца
    # WHY здесь построчный proxy не вычисляется для всех строк — только
    # rolling VWMA последнего дня достаточно для контроля серии.
    # Для days_below_1 считаем построчный proxy через rolling VWMA.
    days_below_1 = 0
    n = len(df_ohlcv)
    if n >= window:
        # Считаем подряд идущие proxy < 1.0 с последнего дня назад
        for i in range(n - 1, n - 1 - (n - window + 1), -1):
            if i < window - 1:
                break
            c_i = closes[i]
            v_w = vols[i - window + 1 : i + 1]
            c_w = closes[i - window + 1 : i + 1]
            tv  = v_w.sum()
            if tv == 0:
                break
            p_i = c_i / (float(np.dot(c_w, v_w) / tv))
            if p_i < 1.0:
                days_below_1 += 1
            else:
                break

    return {
        'proxy_sopr':    round(proxy, 6),
        'vwma_155':      round(vwma, 2),
        'phase':         phase,
        'phase_comment': phase_comment,
        'roc_14':        round(roc_14, 4) if roc_14 is not None else None,
        'days_below_1':  days_below_1,
    }


async def liquidity_density_audit():
    # 1. Data mining
    exchange = ccxt.binance()
    symbol   = 'BTC/USDT'

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=730)  # [cite: 4, 110]
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol'])
    df[['open', 'high', 'low', 'close', 'vol']] = df[['open', 'high', 'low', 'close', 'vol']].apply(pd.to_numeric)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 2. ATR + time decay (use module-level pure functions)
    df['atr'] = calculate_atr(df)
    current_atr = df['atr'].iloc[-1]

    df = apply_time_decay(df, lam=0.005)  # adds days_ago, time_weight, weighted_vol
    # SUBJECTIVE — temporarily disabled (пороговые суждения, см. NEXT_SESSION)
    # df = detect_absorption_days(df, current_atr)  # adds 'absorption' column
    df['absorption'] = False  # placeholder while detect_absorption_days is disabled

    # --- Volume Type Classification ---
    # SUBJECTIVE — temporarily disabled (пороговые суждения, см. NEXT_SESSION)
    # WHY placeholder: df['volume_type'] используется ниже в оркестраторе
    # _vol_mean_20 = df['vol'].rolling(20).mean().fillna(df['vol'])
    # df['volume_type'] = [
    #     classify_volume_type(
    #         open_=row['open'], high=row['high'], low=row['low'], close=row['close'],
    #         vol=row['vol'], vol_mean_20=_vol_mean_20.iloc[i], atr=current_atr,
    #     )
    #     for i, (_, row) in enumerate(df.iterrows())
    # ]
    df['volume_type'] = 'NEUTRAL'  # placeholder while classify_volume_type is disabled

    # --- Этап 5C: интеграция исторического OI (Binance Vision) ---
    # WHY os.path.dirname(__file__): абсолютный путь — не зависит от cwd при запуске.
    # Гарантирует чтение с D:, даже если скрипт запущен из другой директории.
    _oi_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data', 'metrics', 'BTCUSDT-metrics-daily.csv')
    try:
        oi_df = load_oi_history(_oi_csv)
        # Нормализуем date до даты (без времени) для корректного merge
        df['date_only']    = df['date'].dt.normalize()
        oi_df['date_only'] = oi_df['date'].dt.normalize()
        df = df.merge(
            oi_df[['date_only', 'oi_change_pct']],
            on='date_only', how='left'
        ).drop(columns='date_only')
        df['price_change_pct'] = df['close'].pct_change() * 100
        df['oi_regime'] = df.apply(
            lambda r: classify_oi_regime(r['price_change_pct'], r['oi_change_pct'])
            if pd.notna(r['oi_change_pct']) and pd.notna(r['price_change_pct'])
            else 'NEUTRAL',
            axis=1,
        )
    except FileNotFoundError:
        print(f"[WARN] OI CSV not found at {_oi_csv} — oi_regime set to NEUTRAL")
        df['oi_change_pct']    = float('nan')
        df['price_change_pct'] = df['close'].pct_change() * 100
        df['oi_regime']        = 'NEUTRAL'

    # 3. Global bin grid — fixed across all windows so profiles are comparable
    GLOBAL_BINS = np.linspace(df['low'].min(), df['high'].max(), 101)  # 100 bins

    # ------------------------------------------------------------------
    # Exchange-dependent helpers (close over exchange/symbol/df/current_atr)
    # ------------------------------------------------------------------

    def get_precise_volume(target_price: float, days_timestamps: list) -> float:
        """
        Deep Drill: fetch 1m bars for specific days and sum volume inside
        the ATR zone. Eliminates High-Low uniform distribution error. [cite: 20, 23]
        """
        time.sleep(0.1)
        precise_vol   = 0.0
        atr_tolerance = 1.5 * current_atr  # [cite: 31, 33]
        lower_b       = target_price - atr_tolerance
        upper_b       = target_price + atr_tolerance

        for ts in days_timestamps:
            try:
                m1_data = exchange.fetch_ohlcv(symbol, '1m', since=ts, limit=1440)
                m_df    = pd.DataFrame(m1_data, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                mask    = (m_df['l'] <= upper_b) & (m_df['h'] >= lower_b)
                precise_vol += m_df.loc[mask, 'v'].sum()
            except Exception as e:
                print(f"Error drilling into timestamp {ts}: {e}")
                time.sleep(2)
        return precise_vol

    def audit_level_details(name: str, target_price: float, lam: float = 0.005) -> dict:
        """
        Per-level audit: ATR boundaries + 1m Deep Drill + local time decay. [cite: 20, 23, 53, 79]
        lam=0.005 for base levels (POC), lam=0.02 for impulse zones (VAH/VAL). [cite: 53, 29]
        """
        atr_tolerance = 1.5 * current_atr
        lower_bound   = target_price - atr_tolerance
        upper_bound   = target_price + atr_tolerance

        # Local decay — does NOT mutate df [cite: 53, 56]
        local_weight  = 1 / (1 + lam * df['days_ago'])

        mask          = (df['low'] <= upper_bound) & (df['high'] >= lower_bound)
        relevant_days = df[mask].copy()
        relevant_days['local_weight'] = local_weight[mask].values
        days_count    = len(relevant_days)

        if not relevant_days.empty:
            raw_precise_vol  = get_precise_volume(target_price, relevant_days['timestamp'].tolist())
            avg_local_weight = relevant_days['local_weight'].mean()
            range_vol        = raw_precise_vol * avg_local_weight
            first_seen       = relevant_days['date'].min().strftime('%Y-%m-%d')
            last_seen        = relevant_days['date'].max().strftime('%Y-%m-%d')
            avg_daily_vol    = relevant_days['vol'].mean()
        else:
            range_vol                     = 0.0
            first_seen, last_seen         = "N/A", "N/A"
            avg_daily_vol                 = 0.0

        return {
            "name":       name,
            "price":      target_price,
            "vol":        range_vol,
            "days":       days_count,
            "first_date": first_seen,
            "last_date":  last_seen,
            "avg_vol":    avg_daily_vol,
        }

    # ------------------------------------------------------------------
    # On-chain layer
    # ------------------------------------------------------------------
    current_price    = df['close'].iloc[-1]
    # WHY CachedBGeometricsClient: защищает все эндпоинты от повторных API-запросов.
    # Особенно критично для sth-realized-price: OnChainValidator.initialize() вызывает
    # его без кэша при каждом запуске. При 429 без кэша весь on-chain блок
    # падает (_onchain_available=False). С кэшем: fallback на устаревший parquet.
    from onchain_cache import CachedBGeometricsClient as _CachedBGClient
    onchain_client   = _CachedBGClient(BGeometricsClient())
    onchain_validator = OnChainValidator(onchain_client)
    # WHY try/except: API может вернуть 429 (лимит 8/час) или быть недоступен.
    # Ончайн-слой необязателен для работы основного пиплайна — деградируем грейсфулльно.
    _onchain_available = True
    try:
        await onchain_validator.initialize(lookback_days=365)
    except Exception as _e:
        print(f"[WARN] on-chain API недоступен: {_e}")
        print("[WARN] z_score=0.0, capitulation=False — продолжаем без on-chain данных.")
        _onchain_available = False

    # ------------------------------------------------------------------
    # 1. Main 2y profile
    # ------------------------------------------------------------------
    va_low, va_high, poc = calculate_value_area(df, GLOBAL_BINS)
    clusters             = find_liquidity_clusters(df, current_atr)

    # ------------------------------------------------------------------
    # 2. Dynamic levels for deep-drill audit
    # ------------------------------------------------------------------
    dynamic_levels = {
        "Point of Control": {"price": poc,     "lam": 0.005},
        "Value Area High":  {"price": va_high, "lam": 0.02},
        "Value Area Low":   {"price": va_low,  "lam": 0.02},
    }
    if clusters:
        top_cluster = sorted(clusters, key=lambda x: x['vol'], reverse=True)[0]
        dynamic_levels["Major Cluster Base"] = {"price": top_cluster['min'], "lam": 0.005}

    # ------------------------------------------------------------------
    # 3. Deep Drill audit
    # ------------------------------------------------------------------
    print(f"\n{'[RUNNING DYNAMIC INSTITUTIONAL AUDIT]':^95}")
    raw_results = []
    for name, level in dynamic_levels.items():
        time.sleep(0.2)
        raw_results.append(audit_level_details(name, level['price'], level['lam']))

    max_vol = max(r['vol'] for r in raw_results) if raw_results else 0
    print("-" * 110)
    print(f"{'LEVEL':<25} | {'W-SCORE':<8} | {'DAYS':<5} | {'DATE RANGE':<25} | {'ADV (BTC)'}")
    print("-" * 110)
    for r in raw_results:
        score  = (r['vol'] / max_vol * 100) if max_vol > 0 else 0
        icon   = "💎" if score > 70 else ("🧱" if score > 40 else "💨")
        d_range = f"{r['first_date']} -> {r['last_date']}"
        print(f"{icon} {r['name'] + ' (' + str(r['price']) + ')':<22} | "
              f"{score:>5.1f}% | {r['days']:<5} | {d_range:<25} | {r['avg_vol']:>10.1f}")

    print("-" * 110)
    print(f"VALUE AREA (70%): VAH: ${va_high:,.0f} | POC: ${poc:,.0f} | VAL: ${va_low:,.0f}")
    print(f"VOLATILITY: ATR: ${current_atr:,.2f} | Tolerance (1.5x): ${1.5 * current_atr:,.2f}")

    # ------------------------------------------------------------------
    # 4. Multi-window profiles
    # ------------------------------------------------------------------
    print(f"\n{'[MULTI-WINDOW VOLUME PROFILE]':^95}")
    print("=" * 110)
    print(f"{'Window':<8} | {'POC':>10} | {'VA Low':>10} | {'VA High':>10} | {'Active Bins':<12} | {'Clusters'}")
    print("-" * 110)

    _window_clusters = {}  # w_name -> list[dict] — кластеры по каждому окну
    for w_name, w_days in PROFILE_WINDOWS.items():
        df_slice            = df.tail(w_days).copy()
        w_va_low, w_va_high, w_poc = calculate_value_area(df_slice, GLOBAL_BINS)
        _w_clusters = []
        if w_days >= DBSCAN_MIN_WINDOW:
            _w_clusters = find_liquidity_clusters(df_slice, current_atr)
        _window_clusters[w_name] = _w_clusters

        prof        = build_profile(df_slice, GLOBAL_BINS)
        active_bins = (prof['vol'] > 0).sum()

        if w_days < DBSCAN_MIN_WINDOW:
            _cl_str = 'skip'
        elif not _w_clusters:
            _cl_str = '0'
        else:
            _top = sorted(_w_clusters, key=lambda x: x['vol'], reverse=True)[0]
            _cl_str = f"{len(_w_clusters)}  [${_top['min']:,.0f}–${_top['max']:,.0f}]"

        print(f"{w_name:<8} | "
              f"{'$' + f'{w_poc:,.0f}' if w_poc else 'N/A':>10} | "
              f"{'$' + f'{w_va_low:,.0f}' if w_va_low else 'N/A':>10} | "
              f"{'$' + f'{w_va_high:,.0f}' if w_va_high else 'N/A':>10} | "
              f"{active_bins:<12} | "
              f"{_cl_str}")

    # --- Зона POC (нужна для тегового блока POC QUALITY) ---
    poc_lower = poc - 1.5 * current_atr
    poc_upper = poc + 1.5 * current_atr
    absorption_near_poc = df[
        df['absorption'] &
        df['close'].between(poc_lower, poc_upper)
    ]
    total_near_poc = df[df['close'].between(poc_lower, poc_upper)]

    # ------------------------------------------------------------------
    # 5b. Volume Type Analysis
    # ------------------------------------------------------------------
    _WINDOW_VT = 60   # WHY 60 дней: 2 месяца — достаточно для статистики, не перегружает вывод
    df_vt60      = df.tail(_WINDOW_VT)
    vt_counts    = df_vt60['volume_type'].value_counts()
    vt_total     = len(df_vt60)
    vt_last_type = df['volume_type'].iloc[-1]

    _vt_icons = {'ABSORPTION': '⬆️', 'EXHAUSTION': '⬇️', 'BREAKOUT': '⚡', 'NEUTRAL': '▪️'}
    _vt_descs = {
        'ABSORPTION': 'высокий объём + бычье закрытие (>70% диапазона) — покупки поглощают продажи.',
        'EXHAUSTION' : 'высокий объём + медвежье закрытие (<30% диапазона) — предложение истощается.',
        'BREAKOUT'   : 'высокий объём + широкая свеча (>2×ATR) — пробой уровня.',
        'NEUTRAL'    : 'нормальный объём или средина диапазона.',
    }

    _vt_rows = ''
    for _vt in ('ABSORPTION', 'EXHAUSTION', 'BREAKOUT', 'NEUTRAL'):
        _n   = vt_counts.get(_vt, 0)
        _pct = _n / vt_total * 100 if vt_total > 0 else 0.0
        _bar = '█' * int(_pct / 5) + '░' * (20 - int(_pct / 5))
        _vt_rows += f"  {_vt_icons[_vt]} {_vt:<11}: {_n:>3} дней ({_pct:4.1f}%)  [{_bar}]  {_vt_descs[_vt]}\n"

    # Последние 5 дней ABSORPTION и BREAKOUT
    _last_absorb = df[df['volume_type'] == 'ABSORPTION']['date'].tail(5)
    _last_break  = df[df['volume_type'] == 'BREAKOUT']['date'].tail(5)
    _absorb_str  = ', '.join(_last_absorb.dt.strftime('%Y-%m-%d').tolist()) if not _last_absorb.empty else 'нет'
    _break_str   = ', '.join(_last_break.dt.strftime('%Y-%m-%d').tolist()) if not _last_break.empty else 'нет'

    # SUBJECTIVE — temporarily disabled (окно 5 дней, порог 0.5 — произвольны, см. NEXT_SESSION)
    # _imbalance_s   = calculate_volume_imbalance(df, window=5)
    # _imbalance_now = _imbalance_s.iloc[-1]
    _imbalance_s   = pd.Series([float('nan')] * len(df))  # placeholder
    _imbalance_now = float('nan')  # placeholder
    _imbalance_str = f'{_imbalance_now:.2f}' if not pd.isna(_imbalance_now) else 'н/д'
    _imbalance_dir = (
        'покупатели доминируют' if (not pd.isna(_imbalance_now) and _imbalance_now > 0.5)
        else ('продавцы доминируют' if (not pd.isna(_imbalance_now) and _imbalance_now < 0.5)
              else 'нейтрально')
    )

    if False: print(f"""
[VOLUME TYPE ANALYSIS — последние {_WINDOW_VT} дней]
{'-'*66}
{_vt_rows}
Текущая свеча  : {_vt_icons.get(vt_last_type, '')} {vt_last_type}  —  {_vt_descs.get(vt_last_type, '')}
Порог высокого объёма : vol > 20-дн. среднее × 1.5  |  Порог BREAKOUT: range > 2 × ATR

Последние ABSORPTION: {_absorb_str}
Последние BREAKOUT  : {_break_str}

Volume Imbalance (окно 5 дней): {_imbalance_str}  ({_imbalance_dir})
  Что измеряет: ABSORPTION / (ABSORPTION + EXHAUSTION) за 5 дней.
  > 0.5 = покупатели доминируют | < 0.5 = продавцы | = 0.5 = нейтрально или нет сигналов."""    )

    # SUBJECTIVE — temporarily disabled (окно 14 дней — произвольно, см. NEXT_SESSION)
    # _div_series = detect_divergence(
    #     df['close'].reset_index(drop=True),
    #     _imbalance_s.reset_index(drop=True),
    #     window=14,
    # )
    _div_series = pd.Series(['NONE'] * len(df))  # placeholder
    _div_last    = 'NONE'
    _div_30d     = _div_series.tail(30)
    _div_30_bull = 0
    _div_30_bear = 0
    _div_icons   = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NONE': '▪️'}
    _div_descs   = {
        'BULLISH': 'цена на новом лое, volume_imbalance — нет → покупательский дисбаланс не подтверждает слабость.',
        'BEARISH': 'цена на новом хае, volume_imbalance — нет → покупательский дисбаланс не подтверждает силу.',
        'NONE'   : 'нет расхождения между ценой и volume_imbalance.',
    }
    if False: print(f"""
[DIVERGENCE SIGNAL — close vs volume_imbalance]
{'-'*66}
Что измеряет: расхождение между направлением цены и volume_imbalance.
  volume_imbalance = ABSORPTION / (ABSORPTION + EXHAUSTION) за 5 дней.
  BULLISH: цена на новом лое (< min за 14 дней), imbalance — нет → скрытое накопление.
  BEARISH: цена на новом хае (> max за 14 дней), imbalance — нет → скрытое распределение.
  Окно сравнения: 14 дней.

Текущий сигнал  : {_div_icons.get(_div_last, '')} {_div_last}
  {_div_descs.get(_div_last, '')}

За последние 30 дней:
  BULLISH: {_div_30_bull} дней
  BEARISH: {_div_30_bear} дней
  NONE   : {30 - _div_30_bull - _div_30_bear} дней""")

    # ------------------------------------------------------------------
    # 5c. POC Retest History
    # ------------------------------------------------------------------
    _RETEST_WINDOW = 90   # WHY 90: 3 месяца — достаточно для статистики ретестов
    # SUBJECTIVE — temporarily disabled (5 касаний как "хорошо", bounce_rate пороги — произвольны, см. NEXT_SESSION)
    # _retest = calculate_poc_retest_score(df, poc=poc, atr=current_atr, window=_RETEST_WINDOW)
    _retest = {
        'touch_count': 0, 'bounce_count': 0, 'break_count': 0,
        'bounce_rate': 0.0, 'avg_days_between_touches': None, 'score': 0.0,
    }  # placeholder while calculate_poc_retest_score is disabled

    _rt_bounce_pct = _retest['bounce_rate'] * 100
    _rt_break_pct  = (1.0 - _retest['bounce_rate']) * 100 if _retest['touch_count'] > 0 else 0.0
    _rt_avg_str    = (f"{_retest['avg_days_between_touches']:.1f} дней"
                     if _retest['avg_days_between_touches'] is not None else 'недостаточно касаний')

    if _retest['touch_count'] == 0:
        _rt_summary = 'Цена не подходила к зоне POC за указанный период.'
    elif _retest['bounce_rate'] >= 0.7:
        _rt_summary = 'Уровень уважается — цена отбивается чаще чем пробивает (бычий сигнал).'
    elif _retest['bounce_rate'] <= 0.3:
        _rt_summary = 'Уровень чаще пробивается — слабое подтверждение (медвежий сигнал).'
    else:
        _rt_summary = 'Смешанный сигнал — отбои и пробои примерно равны.'

    if False: print(f"""
[POC RETEST HISTORY — последние {_RETEST_WINDOW} дней]
{'-'*66}
Касаний зоны POC (±1.5ATR): {_retest['touch_count']}
  Отбоев (close в зоне) : {_retest['bounce_count']}  ({_rt_bounce_pct:.1f}%)
  Пробоев (close вне)   : {_retest['break_count']}  ({_rt_break_pct:.1f}%)
Среднее дней между касаниями: {_rt_avg_str}

Retest Score: {_retest['score']:.3f}  —  {_rt_summary}
  Что означает: 0.0 = уровень не тестировался или пробивается,
               1.0 = часто тестируется и всегда отбивается."""    )

    # ------------------------------------------------------------------
    # 6. DBSCAN clusters report
    # ------------------------------------------------------------------
    # WHY _profile_2y: extract_sub_levels reuses already-computed profile
    # (build_profile is idempotent, no extra network calls needed)
    _profile_2y = build_profile(df, GLOBAL_BINS)
    # WHY per-window: аналитик видит не просто счётчик, а какие зоны, на каком горизонте и HVN внутри.
    # WHY INSIDE: цена может находиться внутри зоны — это ключевой сигнал для возможного отбоя.
    print(f"\n[LIQUIDITY CLUSTERS — по временным окнам]")
    print("=" * 95)
    print(f"Цена сейчас: ${current_price:,.0f}  |  Алгоритм: HDBSCAN + Modified Z-score > 1.5")
    print(f"  RESISTANCE: цена ниже минимума зоны | SUPPORT: цена выше максимума зоны | INSIDE: цена внутри зоны")
    print(f"  Сходимость: несколько окон указывают на одну зону = более сильный сигнал ликвидности.")
    print("-" * 95)
    print(f"{'\u041eкно':<6} | {'\u0417она (min\u2013max)':<26} | {'\u0422ип':<12} | {'\u0414ней':<5} | HVN внутри зоны")
    print("-" * 95)
    for w_name, _wc in _window_clusters.items():
        if not _wc:
            print(f"{w_name:<6} | {'\u2014':<26} | {'\u2014':<12} | {'\u2014':<5} | —")
        else:
            for j, clus in enumerate(sorted(_wc, key=lambda x: x['vol'], reverse=True)):
                _zone_rel = (
                    'INSIDE'     if clus['min'] <= current_price <= clus['max'] else
                    'SUPPORT'    if current_price > clus['max'] else
                    'RESISTANCE'
                )
                sub = extract_sub_levels(clus, _profile_2y, n_peaks=3)
                hvn_str = '  /  '.join(f"${s['mid']:,.0f}" for s in sub) if sub else 'нет пиков'
                zone_str = f"${clus['min']:,.0f} – ${clus['max']:,.0f}"
                prefix = w_name if j == 0 else ''
                print(f"{prefix:<6} | {zone_str:<26} | {_zone_rel:<12} | {clus['days']:<5} | {hvn_str}")
    print("=" * 95)

    # ------------------------------------------------------------------
    # 6b. OI Regime
    # ------------------------------------------------------------------
    _oi_regime_last    = str(df['oi_regime'].iloc[-1]) if 'oi_regime' in df.columns else 'NEUTRAL'
    _price_chg_last    = df['price_change_pct'].iloc[-1] if 'price_change_pct' in df.columns else float('nan')
    _oi_chg_last       = df['oi_change_pct'].iloc[-1]    if 'oi_change_pct'    in df.columns else float('nan')
    _price_chg_str     = f"{_price_chg_last:+.1f}%" if not pd.isna(_price_chg_last) else "н/д"
    _oi_chg_str        = f"{_oi_chg_last:+.1f}%"   if not pd.isna(_oi_chg_last)    else "н/д"

    _oi_regime_descriptions = {
        'STRONG_BULL': 'цена растёт (>+1%), открытый интерес растёт (>+1%).',
        'WEAK_BULL'  : 'цена растёт (>+1%), открытый интерес падает (≤+1%).',
        'STRONG_BEAR': 'цена падает (<−1%), открытый интерес растёт (>+1%).',
        'LIQUIDATION': 'цена падает (<−1%), открытый интерес падает (≤+1%).',
        'NEUTRAL'    : 'изменения цены и/или OI в пределах ±1% — сигнал ниже порога.',
    }
    _oi_current_desc = _oi_regime_descriptions.get(_oi_regime_last, 'нет описания')

    print(f"""
[OI REGIME — последний день]
{'-'*66}
Режим: {_oi_regime_last}
Данные: цена {_price_chg_str} | OI {_oi_chg_str}

Что измеряет режим: соотношение направления цены и изменения
открытого интереса (суммарный объём незакрытых позиций) за период.

Матрица режимов (порог ±1%):
  STRONG_BULL : цена↑ >+1% И OI↑ >+1%  — цена растёт, OI растёт
  WEAK_BULL   : цена↑ >+1% И OI↓ ≤1% — цена растёт, OI падает
  STRONG_BEAR : цена↓ <−1% И OI↑ >+1%  — цена падает, OI растёт
  LIQUIDATION : цена↓ <−1% И OI↓ ≤1% — цена падает, OI падает
  NEUTRAL     : изменения в пределах ±1% по любому из параметров

► Текущий режим: {_oi_regime_last} — {_oi_current_desc}
  OI-коррекция скора: LIQUIDATION → +0.10 | STRONG_BEAR → −0.10 | остальные → без изменений."""
    )

    # ------------------------------------------------------------------
    # 6c. Funding Rate Regime
    # ------------------------------------------------------------------
    _funding_regime = 'NEUTRAL'   # WHY default: если API недоступен — нейтральный фандинг
    _funding_pct_raw = None
    try:
        _fr_history = exchange.fetch_funding_rate_history(symbol, limit=1)
        if _fr_history:
            # WHY *100: ccxt возвращает долю (0.0001), нам нужны %
            _funding_pct_raw = float(_fr_history[-1]['fundingRate']) * 100
            _funding_regime  = classify_funding_regime(_funding_pct_raw)
    except Exception as _fe:
        print(f"[WARN] funding rate недоступен: {_fe}")

    # --- Basis Spread: spot vs futures ---
    # WHY spot exchange: current_price уже есть (futures), нужна spot-цена для сравнения.
    # WHY ccxt.binance() (spot), а не binanceusdm: futures уже на binanceusdm (exchange),
    # spot BTC/USDT на Binance spot — это другой endpoint.
    _basis = None
    _spot_price_for_basis = None
    try:
        _spot_exchange        = ccxt.binance()   # spot, не futures
        _spot_ticker          = _spot_exchange.fetch_ticker('BTC/USDT')
        _spot_price_for_basis = float(_spot_ticker['last'])
        _basis                = calculate_basis_spread(_spot_price_for_basis, current_price)
    except Exception as _be:
        print(f"[WARN] basis spread недоступен: {_be}")

    _funding_regime_descriptions = {
        'NEGATIVE_EXTREME'  : 'фандинг < −0.05% — шорт-позиции доминируют, рынок перекуплен шортами.',
        'NEGATIVE_MODERATE' : 'фандинг −0.05% ... −0.01% — умеренный перевес шортов.',
        'NEUTRAL'           : 'фандинг −0.01% ... +0.01% — баланс лонгов и шортов.',
        'POSITIVE_MODERATE' : 'фандинг +0.01% ... +0.05% — умеренный перевес лонгов.',
        'POSITIVE_EXTREME'  : 'фандинг > +0.05% — лонг-позиции перегреты, лонги платят шортам.',
    }
    _funding_current_desc = _funding_regime_descriptions.get(_funding_regime, 'нет описания')
    _funding_val_str      = f'{_funding_pct_raw:+.4f}%' if _funding_pct_raw is not None else 'н/д'

    print(f"""
[FUNDING RATE REGIME]
{'-'*66}
Текущий режим: {_funding_regime}
Ставка фандинга: {_funding_val_str}  (базовая ставка Binance USDM = +0.01%)

Что измеряет ставка: процент, который лонги платят шортам (>0) или шорты
лонгам (<0) каждые 8 часов. Отражает дисбаланс открытых позиций.

Матрица режимов:
  NEGATIVE_EXTREME   : фандинг < −0.05%
  NEGATIVE_MODERATE  : −0.05% <= фандинг < −0.01%
  NEUTRAL            : −0.01% <= фандинг <= +0.01%
  POSITIVE_MODERATE  : +0.01% < фандинг <= +0.05%
  POSITIVE_EXTREME   : фандинг > +0.05%

► Текущий режим: {_funding_regime} — {_funding_current_desc}
  Влияние на оценку POC: POSITIVE_EXTREME → тег RESISTANCE_TRAP_FUNDING.

Basis Spread (spot vs futures): {(_basis['regime'] + f" ({_basis['basis_usd']:+,.0f} USD / {_basis['basis_pct']:+.3f}%)") if _basis else 'н/д'}
  CONTANGO      : futures > spot — рынок ждёт роста (бычий сигнал).
  BACKWARDATION : futures < spot — давление продавцов (медвежий сигнал).
  FLAT          : спред в пределах базовой ставки (±0.1%) — нейтрально."""
    )

    # ------------------------------------------------------------------
    # 7. On-chain stress test [cite: 32, 33, 34]
    # ------------------------------------------------------------------
    async def run_stress_test_onchain(price, poc_, va_high_, validator: OnChainValidator):
        if pd.isna(poc_) or poc_ == 0:
            return None
        z_score          = validator.calculate_z_score(price)
        sth_rp           = validator.sth_historical['sth_realized_price'].iloc[-1]
        deviation_from_va = ((price - va_high_) / va_high_) * 100
        prob_return      = 0
        if abs(deviation_from_va) > 20 and abs(z_score) > 2:
            prob_return = 85
        elif abs(z_score) > 1:
            prob_return = 45
        return {"z_score": z_score, "prob": prob_return, "sth_rp": sth_rp,
                "deviation_from_va": deviation_from_va}

    stress = await run_stress_test_onchain(current_price, poc, va_high, onchain_validator) if _onchain_available else None
    if stress:
        print(f"\n{'[PROGNOSTIC STRESS-TEST]':^95}")
        print("-" * 110)
        print(f"STH Realized Price: ${stress['sth_rp']:,.0f} | "
              f"Z-Score: {stress['z_score']:.2f} | "
              f"Dev from VA: {stress['deviation_from_va']:+.1f}%")
        if stress['prob'] > 80:
            print(f"⚠️  ALERT: High Mean Reversion Probability ({stress['prob']}%) to POC ${poc:,.0f}")
        else:
            print(f"STATUS: Market within statistical distribution. P(Return) = {stress['prob']}%")
        if abs(current_price - 67200) < (1.5 * current_atr):
            print("✅ VERIFIED: Price at $67,200 — Institutional Accumulation Zone (Glassnode CBD)")

    # ------------------------------------------------------------------
    # 8. POC Quality Score
    # ------------------------------------------------------------------
    z_score_val   = stress['z_score'] if stress else 0.0

    # --- Этап 2: подключаем реальный capitulation_signal ---
    # WHY try/except + _onchain_available: API может быть недоступен (429, сеть) —
    # fallback к False не ломает весь pipeline
    capitulation = False
    if _onchain_available:
        try:
            from datetime import timedelta as _td
            _loss_end   = datetime.now()
            _loss_start = _loss_end - _td(days=7)
            _loss_df    = await onchain_client.get_realized_loss_lth_usd(
                start_date=_loss_start, end_date=_loss_end
            )
            capitulation = onchain_validator.check_capitulation_signal(_loss_df)
        except Exception as _e:
            print(f"[WARN] capitulation signal unavailable: {_e}")
            capitulation = False

    # ------------------------------------------------------------------
    # 8b. On-chain: LTH Realized Loss — print-блок
    # ------------------------------------------------------------------
    # WHY здесь: _loss_df уже определён выше в try/except и capitulation уже вычислен.
    # Здесь только форматируем готовые данные в читаемый вид.
    _THRESHOLD_M = 300.0   # порог в млн USD — тот же что в check_capitulation_signal()

    if _onchain_available and 'capitulation' in dir() and '_loss_df' in dir():
        _loss_display = _loss_df.tail(3).copy() if not _loss_df.empty else _loss_df
        _sth_rp_val   = stress['sth_rp']    if stress else None
        _z_val        = stress['z_score']   if stress else None

        # М-10 | 7-дневная скользящая средняя убытка LTH
        # WHY rolling(7): Mozart оперирует «среднесуточным убытком» (пост 02.04.2026) —
        # подразумевает сглаженное значение, не raw. Диагностика 2026-05-18 показала
        # разброс ×20 за неделю ($16M–$339M) — raw-значение даёт шумный сигнал.
        # WHY 7 дней: МБ-05 явно использует «в среднем за 7 дней» (пост 14.01.2026).
        # Для М-10 Mozart окно не называет — 7 как согласованный стандарт. FORMALIZED.
        from mozart_signals import classify_lth_realized_loss, lth_loss_pct_of_historical_peak
        _loss_ma7_series = (
            _loss_df['lth_realized_loss_usd'].tail(10).rolling(7).mean()
            if not _loss_df.empty else pd.Series([], dtype=float)
        )
        _loss_ma7 = (
            float(_loss_ma7_series.iloc[-1])
            if (not _loss_ma7_series.empty and not pd.isna(_loss_ma7_series.iloc[-1]))
            else float('nan')
        )
        if not pd.isna(_loss_ma7):
            _loss_zone    = classify_lth_realized_loss(_loss_ma7)
            _loss_pct_ftx = lth_loss_pct_of_historical_peak(_loss_ma7)
            _loss_ma7_str = f'${abs(_loss_ma7) / 1_000_000:.1f} млн/день'
            _loss_pct_str = f'{_loss_pct_ftx:.1f}%'
        else:
            _loss_zone = _loss_ma7_str = _loss_pct_str = 'н/д'

        _loss_lines = []
        for _, _row in _loss_display.iterrows():
            _date_str = (_row['date'].strftime('%d.%m.%Y')
                         if hasattr(_row['date'], 'strftime')
                         else str(_row['date'])[:10])
            _val_m    = float(_row['lth_realized_loss_usd']) / 1_000_000
            _bar_len  = min(10, int(_val_m / _THRESHOLD_M * 10))
            _bar      = '█' * _bar_len + '░' * (10 - _bar_len)
            _flag     = 'выше порога' if _val_m >= _THRESHOLD_M else 'ниже порога'
            _loss_lines.append(f"  {_date_str}: ${_val_m:>6.1f} млн  [{_bar}]  [{_flag}]")
        _loss_table = '\n'.join(_loss_lines) if _loss_lines else '  данные недоступны'

        _cap_str = 'ДА' if capitulation else 'НЕТ'

        if _z_val is not None:
            _z_sign = '+' if _z_val >= 0 else ''
            _z_dir  = ('выше среднего STH realized price за год'
                       if _z_val >= 0 else
                       'ниже среднего STH realized price за год')
        else:
            _z_sign, _z_dir = '', 'нет данных'

        _sth_rp_str = f'${_sth_rp_val:,.0f}' if _sth_rp_val is not None else 'н/д'
        _z_str      = f'{_z_sign}{_z_val:.2f}' if _z_val is not None else 'н/д'

        print(f"""
[ON-CHAIN: LTH REALIZED LOSS]
{'-'*66}
Реализованный убыток LTH за последние 3 дня (порог ${_THRESHOLD_M:.0f}M):
{_loss_table}

Сигнал капитуляции (все 3 дня > ${_THRESHOLD_M:.0f}M): {_cap_str}

7-дн. MA убытка : {_loss_ma7_str}  (сглаживает дневные выбросы, диагностика: разброс ×20/нед)
Зона М-10       : {_loss_zone}  (<$140M / $140–300M / $300–480M / $480–500M / >$500M)
% от FTX пика   : {_loss_pct_str}  (якорь $480M — пик FTX-краша, пост 02.04.2026)

Что измеряет метрика: объём USD, реализованный LTH-когортой
(монеты >155 дней) в убыток за день. Порог ${_THRESHOLD_M:.0f}M — параметр
функции check_capitulation_signal(), не абсолютный стандарт.

STH Realized Price: {_sth_rp_str}
Z-score текущей цены относительно STH Realized Price: {_z_str}
  Формула: (current_price − mean(STH_RP)) / std(STH_RP) за 365 дней.
  Текущее значение {_z_str}: цена {_z_dir}.
  Типичный диапазон шкалы: −3 ... 0 ... +3."""
        )
    else:
        print(f"\n[ON-CHAIN: LTH REALIZED LOSS]\n{'-'*66}\nДанные недоступны (on-chain API не инициализирован).")

    # ------------------------------------------------------------------
    # 8c. On-chain: Holder Structure (LTH/STH MVRV, SOPR, Net Position)
    # ------------------------------------------------------------------
    # WHY CachedBGeometricsClient: holder-метрики дневные — 23ч кэш
    # предотвращает повторные API-вызовы при нескольких запусках в день.
    # WHY lazy import: onchain_cache не нужен если _onchain_available=False.
    # WHY 7 дней диапазон: берём последнее значение (.iloc[-1]) — нужен
    # хотя бы 1 день данных; 7 дней — буфер на случай задержки API.
    if _onchain_available:
        try:
            from datetime import timedelta as _tdh
            _hc          = onchain_client  # уже CachedBGeometricsClient, двойная обёртка не нужна
            _h_end       = datetime.now()
            _h_start     = _h_end - _tdh(days=7)

            _lth_mvrv_df  = await _hc.get_lth_mvrv(start_date=_h_start, end_date=_h_end)
            _sth_mvrv_df  = await _hc.get_sth_mvrv(start_date=_h_start, end_date=_h_end)
            _lth_sopr_df  = await _hc.get_lth_sopr(start_date=_h_start, end_date=_h_end)
            _lth_np30_df  = await _hc.get_lth_net_position_change_30d(start_date=_h_start, end_date=_h_end)
            _sth_np30_df  = await _hc.get_sth_net_position_change_30d(start_date=_h_start, end_date=_h_end)
            # P1: STH SOPR — замыкает SOPR-пару с LTH SOPR
            _sth_sopr_df  = await _hc.get_sth_sopr(start_date=_h_start, end_date=_h_end)
            # P3: NUPL по когортам
            _nupl_lth_df  = await _hc.get_nupl_lth(start_date=_h_start, end_date=_h_end)
            _nupl_sth_df  = await _hc.get_nupl_sth(start_date=_h_start, end_date=_h_end)
            # P4: ETF Flow (BTC)
            _etf_df       = await _hc.get_etf_flow(start_date=_h_start, end_date=_h_end)
            # P5: HODL Waves (MTH когорта)
            _hodl_df      = await _hc.get_hodl_waves(start_date=_h_start, end_date=_h_end)
            # МБ-01: Realized Price — «Синяя линия» дна цикла
            _rp_mb01_df   = await _hc.get_realized_price(start_date=_h_start, end_date=_h_end)
            # МБ-02: True Market Mean — «Зелёная линия», рубикон медвежьего рынка
            _tmm_mb02_df  = await _hc.get_true_market_mean(start_date=_h_start, end_date=_h_end)
            # МБ-04: Supply in Loss — счётчик монет в убытке
            _sl_mb04_df   = await _hc.get_supply_loss(start_date=_h_start, end_date=_h_end)

            def _hlast(df, col):
                # WHY: берём последнее ненулевое значение;
                # если df пустой — возвращаем nan, оркестратор выведет 'н/д'
                return float(df[col].iloc[-1]) if (not df.empty and col in df.columns) else float('nan')

            _lth_mvrv_v  = _hlast(_lth_mvrv_df,  'lth_mvrv')
            _sth_mvrv_v  = _hlast(_sth_mvrv_df,  'sth_mvrv')
            _lth_sopr_v  = _hlast(_lth_sopr_df,  'lth_sopr')
            _lth_np30_v  = _hlast(_lth_np30_df,  'lth_net_position_30d')
            _sth_np30_v  = _hlast(_sth_np30_df,  'sth_net_position_30d')
            _sth_sopr_v  = _hlast(_sth_sopr_df,  'sth_sopr')
            _nupl_lth_v  = _hlast(_nupl_lth_df,  'nupl_lth')
            _nupl_sth_v  = _hlast(_nupl_sth_df,  'nupl_sth')
            _etf_v       = _hlast(_etf_df,        'etf_flow_btc')

            def _hfmt(v, fmt='.2f'):
                return f'{v:{fmt}}' if not pd.isna(v) else 'н/д'
            def _hfmt_btc(v):
                return f'{v:+,.0f} BTC' if not pd.isna(v) else 'н/д'

            print(f"""
[HOLDER STRUCTURE]
{'-'*66}
LTH MVRV        : {_hfmt(_lth_mvrv_v)}  (>1.0 = LTH в прибыли; <1.0 = в убытке)
STH MVRV        : {_hfmt(_sth_mvrv_v)}  (>1.0 = STH в прибыли; <1.0 = давление продаж)
LTH SOPR        : {_hfmt(_lth_sopr_v)}  (>1.0 = продают в прибыль; <1.0 = капитуляция LTH)
STH SOPR        : {_hfmt(_sth_sopr_v)}  (>1.0 = STH продают в прибыль; <1.0 = капитуляция STH)
LTH NUPL        : {_hfmt(_nupl_lth_v)}  (> 0 = прибыль; < 0 = убыток; > 0.75 = эйфория LTH)
STH NUPL        : {_hfmt(_nupl_sth_v)}  (> 0 = прибыль; < 0 = капитуляция STH)
LTH Net Pos 30d : {_hfmt_btc(_lth_np30_v)}  (+= накопление, -= распродажа)
STH Net Pos 30d : {_hfmt_btc(_sth_np30_v)}  (+= накопление, -= распродажа)

Что измеряет блок: поведение когорт LTH (монеты >155 дней) и STH (<155 дней).
MVRV = market value / realized value — нереализованная прибыль/убыток.
SOPR = spent output profit ratio — продают ли выше/ниже себестоимости.
NUPL = net unrealized profit/loss — точнее MVRV, учитывает реальные позиции.
Net Position 30d = приток минус отток BTC в когорту за скользящие 30 дней."""
            )

            # --- [М-09 | STH Realized Price — паттерн В] ---
            # WHY здесь: данные STH RP уже есть в _hc; BTC close есть в df.
            # Фильтр цены: rolling 7d std доходностей BTC < sth_rp_btc_vol_7d_pct_max.
            # Фильтр Z-score: zscore_current < sth_rp_zscore_gate.
            # WHY оркестратор, не функция: detect_sth_rp_zscore_turning() — чистая;
            # два гейта требуют внешних данных (df, onchain) — остаются в оркестраторе.
            try:
                from mozart_signals import detect_sth_rp_zscore_turning
                from mozart_config import MOZART_CONFIG as _MC

                _sth_rp_gate      = _MC["sth_rp_zscore_gate"]          # -1.0
                _sth_rp_vol_max   = _MC["sth_rp_btc_vol_7d_pct_max"]  # 2.0
                _sth_rp_tw        = _MC["sth_rp_zscore_turning_window"] # 5
                _ZSCORE_NORM      = 90  # согласовано со smoke-тестом

                # Данные STH RP: запрашиваем достаточно для Z-score (norm + window)
                from datetime import timedelta as _tdrp
                _rp_end   = datetime.now()
                _rp_start = _rp_end - _tdrp(days=_ZSCORE_NORM + _sth_rp_tw + 10)
                _rp_df    = await _hc.get_sth_realized_price(
                    start_date=_rp_start, end_date=_rp_end
                )

                if _rp_df.empty or len(_rp_df) < _ZSCORE_NORM + _sth_rp_tw:
                    raise ValueError(f"STH RP: недостаточно данных ({len(_rp_df)} строк, нужно >={_ZSCORE_NORM + _sth_rp_tw})")

                _rp_df = _rp_df.sort_values('date').reset_index(drop=True)
                _rp_roll = _rp_df['sth_realized_price'].rolling(
                    _ZSCORE_NORM, min_periods=_ZSCORE_NORM
                )
                _rp_df['_rm'] = _rp_roll.mean()
                _rp_df['_rs'] = _rp_roll.std()
                _rp_df['_z']  = (
                    (_rp_df['sth_realized_price'] - _rp_df['_rm']) / _rp_df['_rs']
                )
                _rp_valid = _rp_df.dropna(subset=['_z']).copy()
                _rp_valid = _rp_valid[_rp_valid['_rs'] > 0].reset_index(drop=True)

                _z_current = float(_rp_valid['_z'].iloc[-1])
                _sth_rp_current = float(_rp_valid['sth_realized_price'].iloc[-1])

                # Гейт 1: Z-score ниже порога
                _gate1_ok = _z_current < _sth_rp_gate

                # Гейт 2: BTC 7d rolling vol < порога (цена в боковике)
                # WHY log-returns: стандартный способ измерения волатильности цены.
                # WHY 7 дней: Mozart «стоячая цена» — неделя без направления.
                _log_ret    = np.log(df['close'] / df['close'].shift(1)).dropna()
                _vol_7d_pct = float(_log_ret.tail(7).std() * 100)
                _gate2_ok   = _vol_7d_pct < _sth_rp_vol_max

                # Детектор разворота
                _zscore_history  = _rp_valid['_z'].tolist()
                _turning         = detect_sth_rp_zscore_turning(
                    _zscore_history, window=_sth_rp_tw
                )

                # Полный сигнал: все три условия
                _sth_rp_signal   = _turning and _gate1_ok and _gate2_ok

                # Форматирование для принта
                _sig_str   = 'ДА' if _sth_rp_signal else 'НЕТ'
                _turn_str  = 'да' if _turning   else 'нет'
                _g1_str    = 'OK' if _gate1_ok else f'нет (Z={_z_current:+.2f} ≥ {_sth_rp_gate})'  
                _g2_str    = 'OK' if _gate2_ok else f'нет (vol={_vol_7d_pct:.1f}% ≥ {_sth_rp_vol_max}%)'
                _sth_rp_err = None

            except Exception as _rp_e:
                _sth_rp_signal = False
                _sth_rp_current = float('nan')
                _z_current = float('nan')
                _vol_7d_pct = float('nan')
                _gate1_ok = _gate2_ok = _turning = False
                _sig_str = _turn_str = _g1_str = _g2_str = 'н/д'
                _sth_rp_err = str(_rp_e)

            _sth_rp_val_str = f'${_sth_rp_current:,.0f}' if not pd.isna(_sth_rp_current) else 'н/д'
            _z_str_m09      = f'{_z_current:+.3f}'        if not pd.isna(_z_current)      else 'н/д'
            _vol_str_m09    = f'{_vol_7d_pct:.2f}%'       if not pd.isna(_vol_7d_pct)     else 'н/д'

            print(f"""
[М-09 | STH RP — паттерн В: Z-score turning]
{'-'*66}
Что измеряет: Z-score STH Realized Price (скользящая 90д норма) разворачивается
вверх при одновременной стабилизации цены BTC.

STH RP сейчас : {_sth_rp_val_str}
Z-score сейчас : {_z_str_m09}  (norm=90d rolling; < {_sth_rp_gate} = под давлением)
BTC vol 7d  : {_vol_str_m09}  (стд log-доходностей; < {_sth_rp_vol_max}% = цена стоит)

Условия сигнала:
  Разворот Z-score (window={_sth_rp_tw}d) : {_turn_str}
  Гейт 1 — Z < {_sth_rp_gate}              : {_g1_str}
  Гейт 2 — BTC vol < {_sth_rp_vol_max}%        : {_g2_str}

Сигнал паттерна В : {_sig_str}{' [все 3 условия выполнены]' if _sth_rp_signal else ''}
{('[WARN] ' + _sth_rp_err) if _sth_rp_err else ''}"""
            )

            # --- [EXCHANGE FLOWS] ---
            # WHY отдельный блок: потоки ETF и биржи — отдельный аналитический контекст.
            # exchange-netflow-btc: 403 на free tier (диагностика 2026-05-11).
            _etf_str = f'{_etf_v:+,.1f} BTC' if not pd.isna(_etf_v) else 'н/д'
            print(f"""
[EXCHANGE FLOWS]
{'-'*66}
ETF Flow (день)   : {_etf_str}  (+ = приток в ETF, − = отток из ETF)
Exchange Netflow : н/д  (exchange-netflow-btc: 403 на free tier)

Что измеряет блок: потоки BTC между рынком и ETF-фондами.
ETF Flow: суточный приток/отток BTC из spot Bitcoin ETF (данные с 2024-01-11)."""
            )

            # --- [HODL WAVES — MTH когорта] ---
            # WHY MTH: medium-term holders (1–6 мес) — единственный источник из BGeometrics.
            # Реальные имена колонок (API, диагностика 2026-05-11):
            # age_1m_3m, age_3m_6m, age_6m_1y — в абсолютных BTC.
            def _hwlast(col):
                return float(_hodl_df[col].iloc[-1]) if (
                    not _hodl_df.empty and col in _hodl_df.columns
                ) else float('nan')
            def _hfmt_mbtc(v):
                return f'{v / 1_000_000:.3f} M BTC' if not pd.isna(v) else 'н/д'

            _hw_1m3m = _hwlast('age_1m_3m')
            _hw_3m6m = _hwlast('age_3m_6m')
            _hw_6m1y = _hwlast('age_6m_1y')

            # М-12 Signal Alignment: prev-значения для classify_hodl_wave_regime()
            # WHY iloc[-2]: функция требует current и prev для определения направления сдвига когорт.
            # WHY отдельный helper: _hwlast берёт только iloc[-1]; для prev нужна граничная проверка len>=2.
            def _hwlast_prev(col):
                return float(_hodl_df[col].iloc[-2]) if (
                    not _hodl_df.empty and col in _hodl_df.columns and len(_hodl_df) >= 2
                ) else float('nan')

            _hw_1m3m_prev = _hwlast_prev('age_1m_3m')
            _hw_3m6m_prev = _hwlast_prev('age_3m_6m')

            print(f"""
[HODL WAVES — MTH когорта]
{'-'*66}
age_1m_3m (1–3 мес)  : {_hfmt_mbtc(_hw_1m3m)}
age_3m_6m (3–6 мес)  : {_hfmt_mbtc(_hw_3m6m)}
age_6m_1y (6–12 мес) : {_hfmt_mbtc(_hw_6m1y)}

Что измеряет блок: объём BTC по возрасту UTXO (время с последнего движения).
MTH = Medium-Term Holder (1–6 мес.). Значения в абсолютных BTC.
Рост age_3m_6m при падении age_1m_3m = монеты созревают (текущие покупки стареют)."""
            )

            # --- [МБ-01 | REALIZED PRICE — «Синяя линия» дна цикла] ---
            try:
                from mozart_signals import classify_realized_price_regime as _cls_rp
                from mozart_config import MOZART_CONFIG
                _rp_v = _hlast(_rp_mb01_df, 'realized_price')
                if not pd.isna(_rp_v):
                    _rp_zone = _cls_rp(float(current_price), float(_rp_v))
                    _rp_zone_desc = {
                        'ABOVE': 'цена значительно выше Realized Price (за пределами зоны дна цикла).',
                        'AT'   : 'цена в ±20% зоне Realized Price — историческая зона дна цикла.',
                        'BELOW': 'цена значительно ниже Realized Price (подтверждённый пробой уровня).',
                    }[_rp_zone]
                    _buf_pct = int(MOZART_CONFIG['realized_price_buffer_pct'] * 100)
                    _rp_upper = _rp_v * (1 + MOZART_CONFIG['realized_price_buffer_pct'])
                    _rp_lower = _rp_v * (1 - MOZART_CONFIG['realized_price_buffer_pct'])
                    _rp_str   = f'${_rp_v:,.0f}'
                    _rp_upper_str = f'${_rp_upper:,.0f}'
                    _rp_lower_str = f'${_rp_lower:,.0f}'
                else:
                    _rp_zone = _rp_zone_desc = 'н/д'
                    _rp_str = _rp_upper_str = _rp_lower_str = 'н/д'
                    _buf_pct = int(MOZART_CONFIG['realized_price_buffer_pct'] * 100)

                print(f"""
[МБ-01 | REALIZED PRICE — «Синяя линия»]
{'-'*66}
Что измеряет: средняя цена покупки всех BTC в обращении (on-chain).
Исторически совпадает с дном циклов 2015, 2018, 2020 (Mozart, пост 25.02.2026).

Realized Price : {_rp_str}
Зона AT ±{_buf_pct}%   : {_rp_lower_str} — {_rp_upper_str}
Текущая цена    : ${current_price:,.0f}

Зона : {_rp_zone}  ({_rp_zone_desc})

Зоны:
  ABOVE : цена значительно выше уровня дна цикла.
  AT    : цена в ±{_buf_pct}% зоне Realized Price.
  BELOW : цена ниже уровня — исторически редкая зона (кратковременная при FTX-краше 2022)."""
                )
            except Exception as _rp_e:
                print(f"\n[МБ-01 | REALIZED PRICE]\n{'-'*66}\nДанные недоступны: {_rp_e}")

            # --- [МБ-02 | TRUE MARKET MEAN — «Зелёная линия»] ---
            try:
                from mozart_signals import classify_true_market_mean_regime as _cls_tmm
                _tmm_v = _hlast(_tmm_mb02_df, 'true_market_mean')
                if not pd.isna(_tmm_v):
                    _tmm_zone = _cls_tmm(float(current_price), float(_tmm_v))
                    _tmm_zone_desc = {
                        'ABOVE': 'цена выше True Market Mean — рубикон не пробит вниз.',
                        'BELOW': 'цена ниже True Market Mean — медвежий рынок подтверждён.',
                    }[_tmm_zone]
                    _tmm_str = f'${_tmm_v:,.0f}'
                else:
                    _tmm_zone = _tmm_zone_desc = 'н/д'
                    _tmm_str = 'н/д'

                print(f"""
[МБ-02 | TRUE MARKET MEAN — «Зелёная линия»]
{'-'*66}
Что измеряет: средняя цена покупки активных (не потерянных) BTC в обращении.
Рубикон медвежьего рынка Mozart (пост 25.02.2026).

True Market Mean : {_tmm_str}
Текущая цена     : ${current_price:,.0f}

Зона : {_tmm_zone}  ({_tmm_zone_desc})

Зоны:
  ABOVE : цена на уровне или выше TMM — рубикон не пробит вниз.
  BELOW : цена ниже TMM — медвежий рынок подтверждён (Mozart: «смена глобального тренда»)."""
                )
            except Exception as _tmm_e:
                print(f"\n[МБ-02 | TRUE MARKET MEAN]\n{'-'*66}\nДанные недоступны: {_tmm_e}")

            # --- [МБ-04 | SUPPLY IN LOSS] ---
            try:
                from mozart_signals import classify_supply_loss_regime as _cls_sl
                from mozart_config import MOZART_CONFIG
                _sl_v = _hlast(_sl_mb04_df, 'supply_loss')
                if not pd.isna(_sl_v):
                    _sl_zone = _cls_sl(float(_sl_v))
                    _sl_zone_desc = {
                        'EXTREME'     : 'объём в убытке выше 5M BTC — исторический триггер смены структурного тренда.',
                        'ELEVATED'    : 'объём в убытке 3.5–5M BTC — активное сопротивление для роста.',
                        'INTERMEDIATE': 'объём в убытке ниже 3.5M BTC — давление есть, ключевые уровни не достигнуты.',
                        'LOW'         : 'меньшинство монет в убытке — большинство рынка в прибыли.',
                    }[_sl_zone]
                    _sl_str = f'{_sl_v / 1_000_000:.2f}M BTC'
                else:
                    _sl_zone = _sl_zone_desc = 'н/д'
                    _sl_str = 'н/д'

                _structural   = MOZART_CONFIG['supply_loss_structural_trigger']
                _intermediate = MOZART_CONFIG['supply_loss_intermediate_trigger']

                print(f"""
[МБ-04 | SUPPLY IN LOSS — монеты в убытке]
{'-'*66}
Что измеряет: количество BTC, чья средняя цена покупки выше текущей цены.
Рубежи Mozart (посты 02.04.2026, 08.04.2026).

Supply in Loss : {_sl_str}

Зона : {_sl_zone}  ({_sl_zone_desc})

Зоны:
  EXTREME      : >= {_structural / 1_000_000:.0f}M BTC — исторический триггер смены структурного тренда (Mozart).
  ELEVATED     : {_intermediate / 1_000_000:.1f}–{_structural / 1_000_000:.0f}M BTC — активное сопротивление для роста.
  INTERMEDIATE : < {_intermediate / 1_000_000:.1f}M BTC — ниже ключевых уровней, давление есть.
  LOW          : <= 0 BTC — теоретически чистый бычий рынок."""
                )
            except Exception as _sl_e:
                print(f"\n[МБ-04 | SUPPLY IN LOSS]\n{'-'*66}\nДанные недоступны: {_sl_e}")

        except Exception as _he:
            print(f"\n[HOLDER STRUCTURE]\n{'-'*66}\nДанные недоступны: {_he}")
    else:
        print(f"\n[HOLDER STRUCTURE]\n{'-'*66}\nДанные недоступны (on-chain API не инициализирован).")

    volume_w_score_poc = 0.0
    if raw_results and max_vol > 0:
        poc_result = next((r for r in raw_results if r['name'] == 'Point of Control'), None)
        if poc_result:
            volume_w_score_poc = (poc_result['vol'] / max_vol) * 100

    # --- Этап 8D: delta context pipeline ---
    # WHY lazy import: избегаем circular import
    # (delta_cache импортирует из volume_density, поэтому top-level импорт невозможен)
    from download_anchor_data import download_anchor_month
    from download_anchor_data import OUT_DIR as ANCHOR_ZIP_DIR
    from download_anchor_data import (
        download_reaction_month,
        build_reaction_zip_path,
        DAILY_OUT_DIR as REACTION_ZIP_DIR,
        DAILY_BASE_URL as REACTION_BASE_URL,
    )
    from delta_cache import build_delta_cache, build_reaction_delta, DEFAULT_CACHE_DIR

    # Шаг 1: определяем какие месяцы нужны (anchor period)
    anchor_months = get_anchor_months(df, poc, current_atr, lookback_days=120)

    # Шаг 2: скачиваем ZIP для каждого месяца (idempotent — пропускает уже скачанные)
    zip_paths = {}
    for year, month in anchor_months:
        path = download_anchor_month(year, month, ANCHOR_ZIP_DIR)
        if path is not None:
            zip_paths[(year, month)] = path

    # Шаг 3: строим delta-кэш (ZIP → CVD → parquet, idempotent)
    cache = build_delta_cache(zip_paths, poc, current_atr, DEFAULT_CACHE_DIR)

    # Шаг 4: агрегируем cvd_slope по всем месяцам → средний для calculate_delta_context_score
    cvd_slopes = []
    for key, parquet_path in cache.items():
        if parquet_path is not None:
            try:
                row = pd.read_parquet(parquet_path)
                cvd_slopes.append(float(row['cvd_slope'].iloc[0]))
            except Exception:
                pass

    # WHY среднее: несколько anchor-месяцев дают усреднённый структурный сигнал
    avg_cvd_slope = float(np.mean(cvd_slopes)) if cvd_slopes else 0.0

    # --- Шаг 8D-5: Reaction Period (aggTrades daily) ---
    # WHY 14 дней: краткосрочное давление за 2 недели (reaction period из ТЗ)
    from datetime import date as _date, timedelta as _timedelta
    _today = _date.today()
    reaction_zip_paths = []
    for _offset in range(14, 0, -1):   # от -14 до -1 дня включительно
        _d = _today - _timedelta(days=_offset)
        _p = download_reaction_month(
            _d.year, _d.month, _d.day,
            REACTION_ZIP_DIR,
            REACTION_BASE_URL,
        )
        if _p is not None:
            reaction_zip_paths.append(_p)

    # WHY build_reaction_delta: суммирует CVD.iloc[-1] по каждому дню → recent_delta
    recent_delta = build_reaction_delta(
        reaction_zip_paths, poc, current_atr, DEFAULT_CACHE_DIR
    )

    # --- Этап 9: Delta Volume Profile (klines 1m) — с кэшем parquet ---
    # WHY кэш: повторные запуски не читают ~15 MB ZIP — берут готовый poc_bin_delta из parquet.
    # WHY 30 дней: статистически достаточно для профиля (~15 MB vs ~22 MB/день у aggTrades).
    from download_anchor_data import download_klines_day, KLINES_OUT_DIR
    from delta_cache import build_klines_delta_cache
    klines_zip_map = {}   # (year, month, day) -> zip_path
    for _offset in range(30, 0, -1):
        _d = _today - _timedelta(days=_offset)
        _p = download_klines_day(_d.year, _d.month, _d.day, KLINES_OUT_DIR)
        if _p is not None:
            klines_zip_map[(_d.year, _d.month, _d.day)] = _p

    klines_cache = build_klines_delta_cache(
        klines_zip_map, poc, current_atr, GLOBAL_BINS, DEFAULT_CACHE_DIR
    )

    # Агрегируем poc_bin_delta по всем дням из кэша
    poc_bin_deltas = []
    for key, parquet_path in klines_cache.items():
        if parquet_path is not None:
            try:
                row = pd.read_parquet(parquet_path)
                poc_bin_deltas.append(float(row['poc_bin_delta'].iloc[0]))
            except Exception:
                pass

    # WHY сумма: накопленное давление на POC за 30 дней — чем больше, тем сильнее сигнал
    poc_bin_delta = float(sum(poc_bin_deltas)) if poc_bin_deltas else 0.0
    print(f"\n[DELTA PROFILE] poc_bin_delta={poc_bin_delta:.2f} (из {len(poc_bin_deltas)} дней)")

    delta_ctx_score = calculate_delta_context_score(avg_cvd_slope, poc_bin_delta)

    _slope_sign  = "+" if avg_cvd_slope >= 0 else ""
    _delta_sign  = "+" if poc_bin_delta >= 0 else ""
    _slope_dir   = "buy_vol превышал sell_vol" if avg_cvd_slope >= 0 else "sell_vol превышал buy_vol"
    _delta_dir   = "taker_buy_vol суммарно больше taker_sell_vol" if poc_bin_delta >= 0 else "taker_sell_vol суммарно больше taker_buy_vol"
    _months_str  = ", ".join(
        f"{['янв','фев','мар','апр','май','июн','июл','авг','сен','окт','ноя','дек'][m-1]} {y}"
        for y, m in anchor_months
    ) if anchor_months else "нет данных"
    if delta_ctx_score > 0.5:
        _score_interp = "> 0.5: оба или один из сигналов положительны, взвешенная сумма > 0."
    elif delta_ctx_score < 0.5:
        _score_interp = "< 0.5: оба или один из сигналов отрицательны, взвешенная сумма < 0."
    else:
        _score_interp = "= 0.5: сигналы компенсируют друг друга или оба равны нулю."

    print(f"""
[DELTA CONTEXT]
{'-'*66}
Anchor period (~4 мес, aggTrades): avg_cvd_slope = {_slope_sign}{avg_cvd_slope:.1f}
  Месяцы анализа: {_months_str}.
  CVD slope — наклон кумулятивной дельты (buy_vol − sell_vol) за период.
  {'Положительный' if avg_cvd_slope >= 0 else 'Отрицательный'}: суммарно {_slope_dir} на уровне POC.

Reaction period (14 дней, klines 1m): poc_bin_delta = {_delta_sign}{poc_bin_delta:.1f} BTC
  Суммарная дельта (taker_buy − taker_sell) в бине POC за 14 дней.
  {'Положительная' if poc_bin_delta >= 0 else 'Отрицательная'}: {_delta_dir}.

Итоговый delta_context_score = {delta_ctx_score:.3f}
  Формула: sigmoid(0.6 × tanh(slope/1000) + 0.4 × tanh(delta/500))
  Диапазон [0.0–1.0], симметричен вокруг 0.5.
  {_score_interp}
  Влияние на оценку POC (теговая архитектура evaluate_poc_quality):
    delta_context_score < 0.35 при volume_w_score > 60% → тег RESISTANCE_TRAP_DELTA."""
    )

    # ------------------------------------------------------------------
    # 8b. LTH Pain Proxy
    # ------------------------------------------------------------------
    lth_proxy = calculate_lth_pain_proxy(df)
    _lth_proxy_sopr = lth_proxy['proxy_sopr']   # None если INSUFFICIENT_DATA

    _lth_phase_icons = {
        'BULL'          : '🟢',
        'EARLY_BEAR'    : '🟡',
        'RUBICON'       : '🟠',
        'BEAR_PRESSURE' : '🔶',
        'CAPITULATION'  : '🔴',
        'EXTREME'       : '☠️',
        'INSUFFICIENT_DATA': '❓',
    }
    _lth_icon = _lth_phase_icons.get(lth_proxy['phase'], '❓')

    if _lth_proxy_sopr is not None:
        _proxy_str  = f"{_lth_proxy_sopr:.4f}"
        _vwma_str   = f"${lth_proxy['vwma_155']:,.0f}"
        _roc_str    = (f"{lth_proxy['roc_14']:+.2f}%" if lth_proxy['roc_14'] is not None else 'н/д')
        _roc_dir    = ('растёт' if (lth_proxy['roc_14'] or 0) > 0
                       else ('падает' if (lth_proxy['roc_14'] or 0) < 0 else 'без изменений'))
        _days_b1    = lth_proxy['days_below_1']
        _below_str  = (f"{_days_b1} дней подряд" if _days_b1 > 0 else 'не в убытке')
    else:
        _proxy_str = _vwma_str = _roc_str = _roc_dir = _below_str = 'н/д'
        _days_b1   = 0

    print(f"""
[LTH PAIN PROXY]
{'-'*66}
{_lth_icon}  Фаза: {lth_proxy['phase']}
    {lth_proxy['phase_comment']}

Что измеряет proxy_sopr: VWMA(цена, объём, 155 дней) — взвешенная средняя цена входа
приблизительная LTH-когорты (монеты > 155 дней). proxy_sopr = close / VWMA_155.
ρ > 1.0: LTH в прибыли. ρ < 1.0: LTH в убытке.

PROXY_SOPR : {_proxy_str}  (close / VWMA_155)
VWMA_155   : {_vwma_str}  (цена, взвешенная по объёму за 155 дней)
ROC_14     : {_roc_str}  (изменение proxy_sopr за 14 дней, {_roc_dir})
DAYS<1.0   : {_below_str}  (proxy_sopr < 1.0 подряд с последнего дня)

Таблица фаз:
  🟢  BULL           proxy > 1.10   LTH в прибыли >10%
  🟡  EARLY_BEAR     1.00 < proxy <= 1.10   у безубытка
  🟠  RUBICON        0.80 < proxy <= 1.00   LTH перешли в убыток
  🔶  BEAR_PRESSURE  0.65 < proxy <= 0.80   убыток 20–35%
  🔴  CAPITULATION   0.50 < proxy <= 0.65   убыток 35–50% (исторически дно)
  ☠️  EXTREME        proxy <= 0.50          убыток >50%

  Тег LTH_CAPITULATION_ZONE активируется: proxy_sopr < 0.60."""
    )

    # SUBJECTIVE — temporarily disabled (теговые пороги 0.35/0.40/0.6 — произвольны, см. NEXT_SESSION)
    # poc_quality = evaluate_poc_quality(
    #     absorption_days_near_poc=len(absorption_near_poc),
    #     total_days_near_poc=len(total_near_poc),
    #     volume_w_score=volume_w_score_poc,
    #     capitulation_confirmed=capitulation,
    #     z_score=z_score_val,
    #     delta_context_score=delta_ctx_score,
    #     oi_regime=str(df['oi_regime'].iloc[-1]) if 'oi_regime' in df.columns else 'NEUTRAL',
    #     funding_regime=_funding_regime,
    #     lth_proxy_sopr=_lth_proxy_sopr,
    # )
    poc_quality = {'label': 'NEUTRAL', 'tags': []}  # placeholder while evaluate_poc_quality is disabled
    label_icon = ("💯" if poc_quality['label'] == 'FAIR_VALUE_MAGNET'
                  else ("🚨" if poc_quality['label'] == 'RESISTANCE_TRAP' else "⚠️"))

    # --- Вспомогательные переменные для print-блока ---
    _abs_days   = len(absorption_near_poc)
    _total_days = len(total_near_poc)
    _abs_ratio  = _abs_days / _total_days if _total_days > 0 else 0.0
    _oi_regime_display = str(df['oi_regime'].iloc[-1]) if 'oi_regime' in df.columns else 'NEUTRAL'

    # --- Теговый вывод ---
    _label_desc = {
        'FAIR_VALUE_MAGNET': 'есть FAIR_VALUE_* тег(и), нет RESISTANCE_* тегов.',
        'NEUTRAL':           'конфликт тегов (FAIR_VALUE_* и RESISTANCE_* одновременно) или нет значимых тегов.',
        'RESISTANCE_TRAP':   'есть RESISTANCE_* тег(и), нет FAIR_VALUE_* тегов.',
    }

    # WHY группировка тегов: аналитику важно видеть бычьи и медвежьи сигналы отдельно
    _fair_tags    = [t for t in poc_quality['tags'] if t.startswith('FAIR_VALUE_')]
    _resist_tags  = [t for t in poc_quality['tags'] if t.startswith('RESISTANCE_')]
    _info_tags    = [t for t in poc_quality['tags']
                     if not t.startswith('FAIR_VALUE_') and not t.startswith('RESISTANCE_')]

    _tag_descs = {
        'FAIR_VALUE_MAGNET_ABSORPTION'  : f'доля поглощения {_abs_ratio:.2f} > 0.4 — >40% дней у POC = бычье поглощение.',
        'FAIR_VALUE_MAGNET_CAPITULATION': 'капитуляция LTH подтверждена — прямой сигнал дна цикла.',
        'FAIR_VALUE_MAGNET_STH_PRESSURE': f'z_score = {z_score_val:.2f} > 1.0 — STH в убытках, паника рынка.',
        'RESISTANCE_TRAP_DELTA'         : f'delta_context_score = {delta_ctx_score:.3f} < 0.35 при volume_score > 0.6 — медвежья дельта с высоким объёмом.',
        'RESISTANCE_TRAP_OI'            : f'oi_regime = {_oi_regime_display} — новые шорты открываются, капитуляция не завершена.',
        'RESISTANCE_TRAP_FUNDING'       : f'funding_regime = {_funding_regime} — лонг-перегрев, рынок платит шортам.',
        'LTH_CAPITULATION_ZONE'         : 'proxy_sopr < 0.60 — информационный тег, на метку не влияет.',
        'BULLISH_DIVERGENCE'            : f'oi_regime = LIQUIDATION + доля поглощения {_abs_ratio:.2f} > 0.3 — информационный тег.',
    }

    def _fmt_tags(tag_list: list) -> str:
        if not tag_list:
            return '      (нет)'
        return '\n'.join(f'      + {t}:\n          {_tag_descs.get(t, "")}' for t in tag_list)

    print(f"""
[POC QUALITY — ТЕГОВАЯ ОЦЕНКА]
{'='*66}
{label_icon}  Метка: {poc_quality['label']}
    {_label_desc.get(poc_quality['label'], '')}

    Что означает метка:
      FAIR_VALUE_MAGNET : {_label_desc['FAIR_VALUE_MAGNET']}
      NEUTRAL           : {_label_desc['NEUTRAL']}
      RESISTANCE_TRAP   : {_label_desc['RESISTANCE_TRAP']}

    Бычьи теги (FAIR_VALUE_*):
{_fmt_tags(_fair_tags)}

    Медвежьи теги (RESISTANCE_*):
{_fmt_tags(_resist_tags)}

    Информационные теги (не влиют на метку):
{_fmt_tags(_info_tags)}
{'='*66}"""    )

    # ------------------------------------------------------------------
    # L3-3 | Signal Alignment — агрегация 14 сигналов Mozart
    # ------------------------------------------------------------------
    # WHY здесь: все on-chain данные уже получены выше; df и exchange в scope.
    # WHY отдельный try: ошибка в alignment не должна ломать уже напечатанные блоки.
    # WHY 'varname' in dir(): on-chain переменные определены внутри try-блока;
    #   если fetch упал — их нет; dir() корректно проверяет локальный scope.
    # МБ-03: None → missing, т.к. get_sth_profit() в onchain_client не реализован.
    # ------------------------------------------------------------------
    _sa_verdict = 'н/д'
    _sa_score   = None
    try:
        from mozart_alignment import build_alignment
        from mozart_signals import (
            classify_lth_sopr_regime     as _cls_m01,
            classify_sth_sopr_regime     as _cls_m02,
            classify_lth_mvrv_regime     as _cls_m03,
            classify_sth_mvrv_regime     as _cls_m04,
            classify_lth_nupl_regime     as _cls_m05,
            classify_sth_nupl_regime     as _cls_m06,
            classify_cohort_flow         as _cls_m0708,
            classify_etf_flow_regime     as _cls_m11,
            classify_hodl_wave_regime    as _cls_m12,
            classify_rsi_regime          as _cls_h01,
            count_consecutive_red_months as _cnt_h02,
            classify_red_months_regime   as _cls_h02,
            calculate_rsi                as _calc_rsi,
        )

        # --- Н-01 RSI (df['close'] всегда в scope) ---
        _sa_rsi_label = _cls_h01(_calc_rsi(df['close'].tolist()))

        # --- Н-02 Red Months (exchange в scope) ---
        try:
            _sa_monthly = exchange.fetch_ohlcv(symbol, '1M', limit=8)
            _sa_h02     = _cls_h02(_cnt_h02(_sa_monthly))
        except Exception as _sa_h02_e:
            print(f'[WARN] Н-02 Red Months: {_sa_h02_e}')
            _sa_h02 = None

        # --- М-12 HODL Waves: prev-значения добавлены в HODL Waves-секции ---
        # WHY dir(): _hw_1m3m_prev определяется внутри if _onchain_available: try:
        # Если fetch упал — переменная не существует; dir() — безопасная проверка.
        _sa_m12 = (
            _cls_m12(_hw_1m3m, _hw_1m3m_prev, _hw_3m6m, _hw_3m6m_prev)
            if ('_hw_1m3m' in dir() and '_hw_1m3m_prev' in dir()
                and not pd.isna(_hw_1m3m) and not pd.isna(_hw_1m3m_prev)
                and not pd.isna(_hw_3m6m) and not pd.isna(_hw_3m6m_prev))
            else None
        )

        # --- Сборка signals dict ---
        # None → missing; строка-метка → signal_polarity() внутри build_alignment()
        _signals_sa = {
            'М-01'    : (_cls_m01(_lth_sopr_v)
                         if '_lth_sopr_v' in dir() and not pd.isna(_lth_sopr_v)
                         else None),
            'М-02'    : (_cls_m02(_sth_sopr_v)
                         if '_sth_sopr_v' in dir() and not pd.isna(_sth_sopr_v)
                         else None),
            'М-03'    : (_cls_m03(_lth_mvrv_v)
                         if '_lth_mvrv_v' in dir() and not pd.isna(_lth_mvrv_v)
                         else None),
            'М-04'    : (_cls_m04(_sth_mvrv_v)
                         if '_sth_mvrv_v' in dir() and not pd.isna(_sth_mvrv_v)
                         else None),
            'М-05'    : (_cls_m05(_nupl_lth_v)
                         if '_nupl_lth_v' in dir() and not pd.isna(_nupl_lth_v)
                         else None),
            'М-06'    : (_cls_m06(_nupl_sth_v)
                         if '_nupl_sth_v' in dir() and not pd.isna(_nupl_sth_v)
                         else None),
            'М-07+08' : (_cls_m0708(_lth_np30_v, _sth_np30_v)
                         if ('_lth_np30_v' in dir() and '_sth_np30_v' in dir()
                             and not pd.isna(_lth_np30_v) and not pd.isna(_sth_np30_v))
                         else None),
            'М-09'    : (str(_sth_rp_signal)
                         if '_sth_rp_signal' in dir()
                         else None),
            'М-10'    : (_loss_zone
                         if ('_loss_zone' in dir()
                             and _loss_zone not in ('н/д', 'н/д (403)', 'н/д (404)'))
                         else None),
            'М-11'    : (_cls_m11(_etf_v)
                         if '_etf_v' in dir() and not pd.isna(_etf_v)
                         else None),
            'М-12'    : _sa_m12,
            'МБ-03'   : None,   # get_sth_profit() не реализован → missing
            'Н-01'    : _sa_rsi_label,
            'Н-02'    : _sa_h02,
        }

        _alignment = build_alignment(_signals_sa)

        # --- Форматирование вывода ---
        _sa_contr = set(_alignment.contrarian_flags)

        def _sa_fmt(id_list, mark_contr=False):
            if not id_list:
                return '(нет)'
            if mark_contr:
                return ' '.join(f'{s}*' if s in _sa_contr else s for s in id_list)
            return ' '.join(id_list)

        _sa_icon = {
            'BULLISH': chr(0x1f7e2),
            'BEARISH': chr(0x1f534),
            'MIXED'  : chr(0x1f7e1),
            'NEUTRAL': chr(0x26aa),
        }.get(_alignment.verdict, chr(0x26aa))

        print(f"""
[SIGNAL ALIGNMENT]
{'='*66}
  Бычьих   : {len(_alignment.bullish):>2}  {_sa_fmt(_alignment.bullish, mark_contr=True)}
  Нейтр.   : {len(_alignment.neutral):>2}  {_sa_fmt(_alignment.neutral)}
  Медвежьих: {len(_alignment.bearish):>2}  {_sa_fmt(_alignment.bearish, mark_contr=True)}
  Н/Д      : {len(_alignment.missing):>2}  {_sa_fmt(_alignment.missing)}
  Счёт     : {_alignment.score:+d}
  Вердикт  : {_sa_icon}  {_alignment.verdict}
  * — контрарианский сигнал (полярность инвертирована по паттерну Mozart)
{'='*66}"""
        )

        _sa_verdict = _alignment.verdict
        _sa_score   = _alignment.score

    except Exception as _sa_e:
        print(f"\n[SIGNAL ALIGNMENT]\n{'='*66}\nНедоступен: {_sa_e}")

    # --- Этап В: Прунинг старых ZIP-файлов ---
    # WHY в конце оркестратора: все ZIP уже обработаны в parquet-кэш к этому моменту.
    # WHY keep_days=45: anchor period 30 дней + 15 дней запас для повторных запусков.
    # WHY dry_run=False: автоматическая очистка без подтверждения.
    # WHY два каталога: reaction ZIP (~22 MB/день) + klines ZIP (~15 MB/день).
    from pruning import prune_old_zips as _prune_old_zips
    from download_anchor_data import KLINES_OUT_DIR as _KLINES_OUT_DIR
    _REACTION_ZIP_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "futures", "um", "daily", "aggTrades", "BTCUSDT"
    )
    _prune_dirs = [_REACTION_ZIP_DIR, _KLINES_OUT_DIR]
    _total_deleted = 0
    for _prune_dir in _prune_dirs:
        if os.path.isdir(_prune_dir):
            _pr = _prune_old_zips(_prune_dir, keep_days=45, dry_run=False)
            _total_deleted += len(_pr['deleted'])
    if _total_deleted > 0:
        print(f"\n[PRUNE] Удалено {_total_deleted} старых ZIP-файлов (keep_days=45)")

    deviation      = ((current_price - poc) / poc) * 100
    mode           = "DISCOVERY" if current_price > va_high else ("VALUE" if current_price > va_low else "DISCOUNT")
    _val_from_val  = ((current_price - va_low) / va_low * 100) if va_low else float('nan')

    _mode_desc = {
        'DISCOVERY': 'цена выше VAH — выше верхней границы зоны стоимости.',
        'VALUE'    : 'цена между VAL и VAH — внутри зоны стоимости.',
        'DISCOUNT' : 'цена ниже VAL — ниже нижней границы зоны стоимости.',
    }
    _oi_short_desc = {
        'STRONG_BULL': 'цена↑, OI↑',
        'WEAK_BULL'  : 'цена↑, OI↓',
        'STRONG_BEAR': 'цена↓, OI↑',
        'LIQUIDATION': 'цена↓, OI↓',
        'NEUTRAL'    : 'изменения в пределах ±1%',
    }

    # --- Сводная строка по капитуляции ---
    if _onchain_available and '_loss_df' in dir() and not _loss_df.empty:
        _avg_loss_m = _loss_df['lth_realized_loss_usd'].tail(3).mean() / 1_000_000
        _cap_summary = f"{'DA' if capitulation else 'NET'}  (${_avg_loss_m:.0f} млн среднее за 3 дня)"
    else:
        _cap_summary = 'данные недоступны'

    # --- VWAP deviation (20-day rolling) ---
    # WHY last value: оркестратор работает с актуальным состоянием рынка.
    # Положительное значение: текущая цена выше VWAP окна (перекупленность относительно окна).
    # Отрицательное значение: текущая цена ниже VWAP окна (перепроданность относительно окна).
    _vwap_dev_series = calculate_vwap_deviation(df)
    _vwap_dev        = _vwap_dev_series.iloc[-1] if _vwap_dev_series.notna().any() else float('nan')
    _vwap_dev_str    = f'{_vwap_dev:+.2f}%' if not pd.isna(_vwap_dev) else 'н/д'

    # --- Строка delta_context_score для сводки ---
    _delta_summary_dir = ('buy_vol > sell_vol на обоих горизонтах'
                          if delta_ctx_score > 0.5 else
                          'sell_vol > buy_vol на одном или обоих горизонтах')

    _vah_str     = f'${va_high:,.0f}' if va_high else 'н/д'
    _val_str     = f'${va_low:,.0f}'  if va_low  else 'н/д'
    _val_dev_str = f'{_val_from_val:+.1f}%' if not pd.isna(_val_from_val) else 'н/д'
    _oi_short       = _oi_short_desc.get(_oi_regime_last, _oi_regime_last)
    _market_regime  = classify_market_regime(_oi_regime_last, _funding_regime)

    print(f"""
[FINAL VERDICT]
{'='*66}
Текущая цена: ${current_price:,.0f}  |  Отклонение от POC: {deviation:+.1f}%
Рыночный режим: {mode}

Что означает режим (граница — Value Area, 70% объёма):
  DISCOVERY : цена выше VAH — выше верхней границы зоны стоимости.
  VALUE     : цена между VAL и VAH — внутри зоны стоимости.
  DISCOUNT  : цена ниже VAL — ниже нижней границы зоны стоимости.
  ► Текущий: {mode} — {_mode_desc[mode]}
    VAH = {_vah_str} | VAL = {_val_str} | Отклонение от VAL: {_val_dev_str}

Сводка всех сигналов:
  POC Quality       : {poc_quality['label']}  ({len(poc_quality['tags'])} тег(а))
  Delta Context     : {delta_ctx_score:.3f}  ({_delta_summary_dir})
  OI Режим          : {_oi_regime_last}  ({_oi_short})
  Funding Режим      : {_funding_regime}  ({_funding_val_str})
  Basis Spread      : {(_basis['regime'] + f" {_basis['basis_pct']:+.3f}%") if _basis else 'н/д'}
  VWAP Deviation    : {_vwap_dev_str}  (отклонение текущей цены от 20-дн. VWAP; + = выше, − = ниже)
  Рыночный режим    : {_market_regime}
  Капитуляция LTH   : {_cap_summary}
  Realized Price    : {_rp_str if '_rp_str' in dir() else 'н/д'}  (зона: {_rp_zone if '_rp_zone' in dir() else 'н/д'})
  True Market Mean  : {_tmm_str if '_tmm_str' in dir() else 'н/д'}  (зона: {_tmm_zone if '_tmm_zone' in dir() else 'н/д'})
  Supply in Loss    : {_sl_str if '_sl_str' in dir() else 'н/д'}  (зона: {_sl_zone if '_sl_zone' in dir() else 'н/д'})
  Signal Alignment  : {_sa_verdict}  (счёт: {f'{_sa_score:+d}' if _sa_score is not None else 'н/д'}, * = контрарианский)
{'='*66}"""    )


if __name__ == "__main__":
    import sys
    # WHY reconfigure: cmd.exe по умолчанию использует cp1251 на Windows,
    # что не поддерживает emoji. reconfigure(encoding='utf-8') решает это
    # без изменения самих print-строк и без PYTHONIOENCODING в среде.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    asyncio.run(liquidity_density_audit())
