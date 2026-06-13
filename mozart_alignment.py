# mozart_alignment.py
# Уровень 3 — агрегация и выравнивание сигналов Mozart.
# Архитектура: docs/PLAN_MOZART_LEVEL3_SIGNAL_ALIGNMENT.md
#
# Этот модуль — агрегатор. Он зависит от mozart_signals.py,
# но mozart_signals.py не зависит от него.
#
# Изменение интерпретации Mozart → менять только _POLARITY_TABLE.
# Изменение порогов verdict → менять только build_alignment().

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Таблица полярности — единственный источник истины для L3
# ---------------------------------------------------------------------------
#
# Структура: { signal_id: { label: 'BULLISH'|'NEUTRAL'|'BEARISH' } }
#
# Контрарианские метки помечены комментарием # CONTRARIAN.
# BOTH_BUYING → NEUTRAL (** сноска в плане: нет перетока когорт).
# МБ-03 EUPHORIA → BULLISH (упрощение; флаг euphoria_convergence в L3-2).
# М-05 EUPHORIA → BEARISH (риск дистрибуции LTH на вершине цикла).
# М-09 принимает строки 'True'/'False' (str(detect_sth_rp_zscore_turning())).

_POLARITY_TABLE: dict[str, dict[str, str]] = {
    # --- М-01 | LTH SOPR ---
    "М-01": {
        "BULL":         "BULLISH",
        "EARLY_BEAR":   "NEUTRAL",
        "MID_BEAR":     "BEARISH",
        "CAPITULATION": "BEARISH",
    },
    # --- М-02 | STH SOPR ---
    "М-02": {
        "BULL":    "BULLISH",
        "RUBICON": "NEUTRAL",
        "BEAR":    "BEARISH",
    },
    # --- М-03 | LTH MVRV (бинарный) ---
    "М-03": {
        "BULL": "BULLISH",
        "BEAR": "BEARISH",
    },
    # --- М-04 | STH MVRV ---
    "М-04": {
        "BULL":    "BULLISH",
        "NEUTRAL": "NEUTRAL",
        "BEAR":    "BEARISH",
    },
    # --- М-05 | LTH NUPL ---
    "М-05": {
        "POSITIVE": "BULLISH",
        "RUBICON":  "NEUTRAL",
        "BEAR":     "BEARISH",
        "EUPHORIA": "BEARISH",   # CONTRARIAN: вершина цикла = риск дистрибуции
    },
    # --- М-06 | STH NUPL ---
    "М-06": {
        "POSITIVE":    "BULLISH",
        "RUBICON":     "NEUTRAL",
        "CAPITULATION":"BEARISH",
    },
    # --- М-07+08 | Cohort Flow ---
    "М-07+08": {
        "ACCUMULATION": "BULLISH",
        "BOTH_BUYING":  "NEUTRAL",   # нет перетока когорт → не бычий
        "DISTRIBUTION": "BEARISH",
        "BOTH_SELLING": "BEARISH",
    },
    # --- М-09 | STH RP Z-score turning (bool → str) ---
    "М-09": {
        "True":  "BULLISH",
        "False": "NEUTRAL",
    },
    # --- М-10 | LTH Realized Loss ---
    "М-10": {
        "BELOW_2018":       "NEUTRAL",
        "EARLY_2018_RANGE": "BEARISH",
        "MID_2022_RANGE":   "BEARISH",
        "PEAK_FTX_RANGE":   "BEARISH",
        "EXTREME":          "BULLISH",  # CONTRARIAN: исчерпание продавцов = дно
    },
    # --- М-11 | ETF Flow ---
    "М-11": {
        "INFLOW":  "BULLISH",
        "NEUTRAL": "NEUTRAL",
        "OUTFLOW": "BEARISH",
    },
    # --- М-12 | HODL Waves ---
    "М-12": {
        "AGING":        "BULLISH",
        "MIXED":        "NEUTRAL",
        "REJUVENATING": "BEARISH",
    },
    # --- МБ-03 | STH Profit Zone ---
    "МБ-03": {
        "BEAR":              "BEARISH",
        "NEUTRAL":           "NEUTRAL",
        "NEUTRAL_BROKEN":    "BULLISH",
        "HEATED":            "BULLISH",
        "EUPHORIA_APPROACH": "BULLISH",
        "EUPHORIA":          "BULLISH",  # упрощение; euphoria_convergence флаг в L3-2
    },
    # --- Н-01 | RSI — КОНТРАРИАНСКИЙ ---
    "Н-01": {
        "NEUTRAL":          "NEUTRAL",
        "OVERSOLD":         "BULLISH",   # CONTRARIAN: перепроданность = разворот
        "EXTREME_OVERSOLD": "BULLISH",   # CONTRARIAN: исторический прецедент 2020
    },
    # --- Н-02 | Red Months — КОНТРАРИАНСКИЙ ---
    "Н-02": {
        "NORMAL":  "NEUTRAL",
        "RARE":    "BULLISH",   # CONTRARIAN: редкость = зона накопления
        "EXTREME": "BULLISH",   # CONTRARIAN: прецедент 2018 (финал медвежки)
    },
    # --- М-15 | Funding Rate 30d MA ---
    # FLOOR_ZONE → BULLISH: многолетний минимум фандинга, рынок платит лонгам;
    # Mozart (11.03.2026): исторически предшествует дну цикла.
    "М-15": {
        "FLOOR_ZONE": "BULLISH",
        "NEUTRAL":    "NEUTRAL",
    },
    # --- НВ-01 | Регрессия убывающих пиков ---
    # DESCENDING_STRONG/WEAK → BEARISH: нисходящий тренд swing highs.
    # Mozart (31.03.2026): 87k→84k→76k — паттерн активен.
    # ASCENDING → NEUTRAL: роста пиков нет в Mozart-паттернах (не BULLISH).
    'НВ-01': {
        'DESCENDING_STRONG': 'BEARISH',
        'DESCENDING_WEAK':   'BEARISH',
        'FLAT':              'NEUTRAL',
        'ASCENDING':         'NEUTRAL',
        'INSUFFICIENT_DATA': 'NEUTRAL',
    },
    # --- НВ-02 | PPI → CPI макро-фильтр ---
    # FALLING → BULLISH: замедление PPI → ожидается снижение CPI → ФРС смягчает.
    # RISING → BEARISH: рост PPI → рост CPI → ФРС удерживает ставку.
    # Mozart (13.05.2026): PPI опережает CPI на 1-3 месяца.
    'НВ-02': {
        'RISING':  'BEARISH',
        'FALLING': 'BULLISH',
        'FLAT':    'NEUTRAL',
        'MIXED':   'NEUTRAL',
    },
    # --- НВ-03 | BTC Dominance — ротация ликвидности ---
    # Источник: пост 10.05.2026 (НВ-03, PLAN_MOZART_PATTERNS.md ЧАСТЬ 4)
    # API: CoinGecko /api/v3/global → поле btc_dominance (%)
    # ROTATION_BTC → NEUTRAL (FORMALIZED): рост BTC.D амбивален (BTC-сезон вс риск-офф);
    #   Mozart не формулирует этот вектор явно; NEUTRAL исключает ложный сигнал.
    "НВ-03": {
        "ROTATION_ALTCOIN": "BULLISH",   # риск-аппетит возвращается: micro→mid→BTC цепочка
        "NEUTRAL":          "NEUTRAL",
        "ROTATION_BTC":     "NEUTRAL",   # FORMALIZED: нет явной интерпретации Mozart
    },
    # --- МБ-05 | LTH Realized Profit — давление распределения ---
    # Источник: пост 14.01.2026 (МБ-05, PLAN_MOZART_PATTERNS.md ЧАСТЬ 2)
    # HIGH_PRESSURE → BEARISH: Mozart явно: MA > $1B/день = «риски смещены в сторону медвежки».
    # MODERATE/LOW → NEUTRAL: недостаточно для сигнала; $500M FORMALIZED.
    "МБ-05": {
        "HIGH_PRESSURE": "BEARISH",
        "MODERATE":      "NEUTRAL",
        "LOW":           "NEUTRAL",
    },
}


# ---------------------------------------------------------------------------
# signal_polarity()
# ---------------------------------------------------------------------------

def signal_polarity(signal_id: str, label: str) -> str:
    """
    Возвращает полярность сигнала Mozart по его ID и метке зоны.

    Args:
        signal_id: Идентификатор сигнала ('М-01', 'Н-01', 'МБ-03' и т.д.).
        label:     Метка зоны, возвращённая classify-функцией Уровня 2.
                   Для М-09: str(detect_sth_rp_zscore_turning()) → 'True'/'False'.

    Returns:
        str: 'BULLISH' | 'NEUTRAL' | 'BEARISH'

    Raises:
        ValueError: Если signal_id не в таблице или label не в таблице сигнала.

    WHY ValueError вместо дефолта:
        Тихий возврат (например, NEUTRAL) при неизвестном ID/метке маскирует
        рассинхрон между mozart_signals.py и mozart_alignment.py. ValueError
        принудит разработчика заметить проблему при рефакторинге classify-функций.
    """
    if signal_id not in _POLARITY_TABLE:
        raise ValueError(
            f"signal_polarity: неизвестный signal_id={signal_id!r}. "
            f"Известные: {sorted(_POLARITY_TABLE)}"
        )
    signal_map = _POLARITY_TABLE[signal_id]
    if label not in signal_map:
        raise ValueError(
            f"signal_polarity: неизвестная метка={label!r} "
            f"для signal_id={signal_id!r}. "
            f"Известные метки: {sorted(signal_map)}"
        )
    return signal_map[label]


# ---------------------------------------------------------------------------
# Контрарианские сигналы — множество для быстрой проверки
# ---------------------------------------------------------------------------
#
# Контрарианский = полярность инвертирована относительно интуиции.
# Оркестратор помечает такие сигналы звёздочкой (*) в выводе.
# Хранится здесь, рядом с _POLARITY_TABLE — единая точка обслуживания.
#
# WHY set (не list): проверка `signal_id in _CONTRARIAN_IDS` — O(1).
# WHY только ID, не (ID, label): контрарианский — это свойство сигнала,
# а не конкретной метки. Оркестратору достаточно знать «этот сигнал особый».

_CONTRARIAN_IDS: frozenset[str] = frozenset({
    'Н-01',    # RSI: OVERSOLD/EXTREME_OVERSOLD → BULLISH (разворот)
    'Н-02',    # Red Months: RARE/EXTREME → BULLISH (исторические зоны дна)
    'М-10',    # LTH Realized Loss: EXTREME → BULLISH (исчерпание продавцов)
    'М-05',    # LTH NUPL: EUPHORIA → BEARISH (риск дистрибуции на вершине)
})


# ---------------------------------------------------------------------------
# AlignmentResult
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """
    Агрегированный вектор выравнивания сигналов Mozart.

    Поля:
        bullish:          ID сигналов с полярностью BULLISH.
        neutral:          ID сигналов с полярностью NEUTRAL.
        bearish:          ID сигналов с полярностью BEARISH.
        missing:          ID сигналов без данных (None / 'н/д' / API 403/404).
        score:            len(bullish) - len(bearish). Missing и neutral не влияют.
        verdict:          'BULLISH' / 'BEARISH' / 'MIXED' / 'NEUTRAL'.
        contrarian_flags: ID активированных контрарианских сигналов
                          (те, что в bullish или bearish И в _CONTRARIAN_IDS).
    """
    bullish:           list[str] = field(default_factory=list)
    neutral:           list[str] = field(default_factory=list)
    bearish:           list[str] = field(default_factory=list)
    missing:           list[str] = field(default_factory=list)
    score:             int       = 0
    verdict:           str       = 'NEUTRAL'
    contrarian_flags:  list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# build_alignment()
# ---------------------------------------------------------------------------

# Значения, которые означают «данных нет»
_MISSING_SENTINELS: frozenset[str] = frozenset({'н/д', 'н/д (403)', 'н/д (404)'})


def build_alignment(signals: dict[str, Optional[str]]) -> AlignmentResult:
    """
    Агрегирует словарь сигналов Mozart в AlignmentResult.

    Args:
        signals: { signal_id: label | None }
            label == None или в _MISSING_SENTINELS → missing.
            Иначе → signal_polarity(signal_id, label) → BULLISH/NEUTRAL/BEARISH.

    Returns:
        AlignmentResult с заполненными полями.

    Raises:
        ValueError: делегируется из signal_polarity при неизвестном ID или метке.

    WHY missing не влияет на score:
        н/д из-за 403/404 ≠ нейтральный рынок. Архитектурный принцип плана:
        отсутствие данных учитывается отдельно, не искажает счётчик.
    """
    bullish: list[str] = []
    neutral: list[str] = []
    bearish: list[str] = []
    missing: list[str] = []
    contrarian_flags: list[str] = []

    for signal_id, label in signals.items():
        # --- missing ---
        if label is None or label in _MISSING_SENTINELS:
            missing.append(signal_id)
            continue

        # --- полярность (ValueError при неизвестном ID или метке) ---
        polarity = signal_polarity(signal_id, label)

        if polarity == 'BULLISH':
            bullish.append(signal_id)
        elif polarity == 'BEARISH':
            bearish.append(signal_id)
        else:
            neutral.append(signal_id)

        # --- contrarian flag (только если сигнал directional) ---
        if polarity in ('BULLISH', 'BEARISH') and signal_id in _CONTRARIAN_IDS:
            contrarian_flags.append(signal_id)

    # --- score и verdict ---
    score = len(bullish) - len(bearish)
    total_directional = len(bullish) + len(bearish)

    if total_directional == 0:
        verdict = 'NEUTRAL'
    elif score >= 2:
        verdict = 'BULLISH'
    elif score <= -2:
        verdict = 'BEARISH'
    else:
        # score in {-1, 0, +1} при наличии хотя бы одного directional
        verdict = 'MIXED'

    return AlignmentResult(
        bullish=bullish,
        neutral=neutral,
        bearish=bearish,
        missing=missing,
        score=score,
        verdict=verdict,
        contrarian_flags=contrarian_flags,
    )
