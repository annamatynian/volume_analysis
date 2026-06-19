# mozart_signals.py
# Детерминированные сигналы по паттернам трейдера Mozart (Уровень 2).
# Пороги — только из mozart_config.MOZART_CONFIG, не хардкодятся здесь.

from __future__ import annotations

import numpy as np

from mozart_config import MOZART_CONFIG


# ---------------------------------------------------------------------------
# Н-01 | RSI(14) дневного ТФ
# ---------------------------------------------------------------------------

def calculate_rsi(closes, period: int = 14) -> float:
    """
    RSI по формуле Wilder (EMA-сглаживание, не простая MA).

    Args:
        closes: Последовательность цен закрытия (list или np.ndarray).
                Должна содержать более period элементов.
        period: Период RSI (по умолчанию 14).

    Returns:
        float: RSI в диапазоне [0.0, 100.0].

    Raises:
        ValueError: Если длина closes <= period (недостаточно данных).

    WHY Wilder, не простая MA: стандартная реализация RSI(14) использует
    EMA с alpha=1/period для сглаживания avg_gain/avg_loss. Простая MA
    даёт другие значения и несовместима с большинством торговых платформ.
    """
    arr = np.asarray(closes, dtype=float)

    if len(arr) <= period:
        raise ValueError(
            f"calculate_rsi: нужно минимум {period + 1} точек, получено {len(arr)}."
        )

    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Начальные средние — простая MA по первым `period` изменениям
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    # Wilder EMA по оставшимся изменениям
    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0.0:
        return 100.0

    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def classify_rsi_regime(rsi: float) -> str:
    """
    Классифицирует RSI по зонам паттерна Н-01.

    Зоны (строгое <):
        rsi < rsi_extreme_oversold                 → 'EXTREME_OVERSOLD'
        rsi_extreme_oversold <= rsi < rsi_oversold → 'OVERSOLD'
        rsi >= rsi_oversold                        → 'NEUTRAL'

    Пороги из MOZART_CONFIG — не хардкодятся.
    """
    extreme = MOZART_CONFIG["rsi_extreme_oversold"]
    oversold = MOZART_CONFIG["rsi_oversold"]

    if rsi < extreme:
        return "EXTREME_OVERSOLD"
    if rsi < oversold:
        return "OVERSOLD"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Н-02 | Красные месячные свечи подряд
# ---------------------------------------------------------------------------

def count_consecutive_red_months(ohlcv: list) -> int:
    """
    Считает количество последних подряд идущих красных месячных свечей.

    Args:
        ohlcv: Список свечей в формате ccxt:
               [timestamp, open, high, low, close, volume].
               Красная свеча: close (индекс 4) < open (индекс 1).

    Returns:
        int: Число последних подряд красных свечей (счёт с конца).
             0 если список пуст или последняя свеча не красная.

    WHY с конца: оркестратор отображает текущее состояние рынка,
    а не исторический максимум серии.
    """
    count = 0
    for candle in reversed(ohlcv):
        open_price  = candle[1]
        close_price = candle[4]
        if close_price < open_price:
            count += 1
        else:
            break
    return count


def classify_red_months_regime(count: int) -> str:
    """
    Классифицирует серию красных месячных свечей по паттерну Н-02.

    Зоны (строгое >=):
        count >= red_months_extreme → 'EXTREME'
        count >= red_months_rare    → 'RARE'
        иначе                       → 'NORMAL'

    Важная оговорка (пост 11.02.2026): 'EXTREME' не означает конец медвежки —
    указывает на экстремальную перепроданность по месячному ТФ.
    Пороги из MOZART_CONFIG — не хардкодятся.
    """
    extreme = MOZART_CONFIG["red_months_extreme"]
    rare    = MOZART_CONFIG["red_months_rare"]

    if count >= extreme:
        return "EXTREME"
    if count >= rare:
        return "RARE"
    return "NORMAL"


# ---------------------------------------------------------------------------
# МБ-03 | % STH в прибыли — зонная классификация + скорость изменения
# ---------------------------------------------------------------------------

def classify_sth_profit_zone(profit_pct: float) -> str:
    """
    Классифицирует % STH в прибыли по зонам паттерна МБ-03 (пост 09.09.2025).

    Линии плавающие (диапазоны из поста 09.09.2025, зафиксированы в MOZART_CONFIG):
      Синяя  (нейтральная зона): 51–59%
      Жёлтая (перегрев):          69–76%
      Красная (эйфория):          85–95%

    Возвращаемые зоны:
      BEAR             : < 51%    — большинство STH в убытке
      NEUTRAL          : 51–59%  — вокруг синей линии; сопротивление/поддержка
      NEUTRAL_BROKEN   : 59–69%  — нейтральная пробита вверх; отскок продолжается
      HEATED           : 69–76%  — жёлтая линия; редко в медвежьем рынке
      EUPHORIA_APPROACH: 76–85%  — переход к эйфории
      EUPHORIA         : ≥ 85%   — красная линия; подтверждение новой бычки

    Args:
        profit_pct: % STH в прибыли (поле profitLoss из /v1/profit-loss).

    Returns:
        str: Название зоны.
    """
    cfg = MOZART_CONFIG

    # Проверка сверху вниз — сначала самая сильная зона
    if profit_pct >= cfg["sth_profit_euphoria_min"]:
        return "EUPHORIA"           # ≥ 85%: красная линия, Over-heated
    if profit_pct >= cfg["sth_profit_heated_max"]:
        return "EUPHORIA_APPROACH"  # 76–85%: переход к эйфории
    if profit_pct >= cfg["sth_profit_heated_min"]:
        return "HEATED"             # 69–76%: жёлтая линия
    if profit_pct >= cfg["sth_profit_neutral_max"]:
        return "NEUTRAL_BROKEN"     # 59–69%: нейтральная пробита вверх
    if profit_pct >= cfg["sth_profit_neutral_min"]:
        return "NEUTRAL"            # 51–59%: вокруг синей линии
    return "BEAR"                   # < 51%: большинство STH в убытке


def build_sth_profit_signal(history: list) -> dict:
    """
    Строит комплексный сигнал по паттерну МБ-03 (Паттерн А + Паттерн Б).

    Args:
        history: Хронологический список значений profit_pct (старые → новые).
                 Минимум 1 элемент. Для дельты нужно ≥ window_days + 1 элементов.

    Returns:
        dict с ключами:
          "zone"               : str   — зона текущего значения (history[-1])
          "profit_pct"         : float — текущее значение history[-1]
          "delta_7d"           : float — изменение за последние window_days п.п.
          "is_dropping_sharply": bool  — дельта < -drop_threshold_7d

    Семантика delta_7d:
      Отрицательная = STH profit снижается.
      is_dropping_sharply = True: резкий пробой синей линии вниз =
        смена структуры по паттерну МБ-03 Паттерн Б (пост 09.09.2025).

    При недостаточной истории: delta_7d=0.0, is_dropping_sharply=False.
    """
    cfg     = MOZART_CONFIG
    window  = cfg["sth_profit_speed_window_days"]   # окно дельты (7 дней)
    thresh  = cfg["sth_profit_drop_threshold_7d"]   # порог «резкого» падения (FORMALIZED)

    current = float(history[-1])
    zone    = classify_sth_profit_zone(current)

    # Дельта: текущее − значение window дней назад
    if len(history) >= window + 1:
        delta_7d = current - float(history[-(window + 1)])
        is_dropping_sharply = delta_7d < -thresh
    else:
        # Недостаточно данных — не блокируем pipeline
        delta_7d            = 0.0
        is_dropping_sharply = False

    return {
        "zone":                zone,
        "profit_pct":          current,
        "delta_7d":            float(delta_7d),
        "is_dropping_sharply": bool(is_dropping_sharply),
    }


# ---------------------------------------------------------------------------
# М-01 | LTH SOPR — рубикон + светофор фаз + детектор разворота
# ---------------------------------------------------------------------------

def classify_lth_sopr_regime(sopr: float) -> str:
    """
    Классифицирует LTH SOPR по четырём зонам паттерна М-01 (пост 05.04.2026).

    Зоны (строгое >=, приоритет сверху вниз):
      sopr >= rubicon              → 'BULL'         — LTH продают в прибыль
      sopr >= early_bear           → 'EARLY_BEAR'   — 0–20% убыток, рубикон пройден
      sopr >= deep_bear            → 'MID_BEAR'     — 20–50% убыток, разгар
      sopr <  deep_bear            → 'CAPITULATION' — >50% убыток, кульминация

    Граница rubicon == 1.0:
      sopr == 1.0 → 'BULL' (выше или на уровне = рубикон не пробит).

    Пороги из MOZART_CONFIG — не хардкодятся.
    """
    rubicon    = MOZART_CONFIG["lth_sopr_rubicon"]
    early_bear = MOZART_CONFIG["lth_sopr_early_bear"]
    deep_bear  = MOZART_CONFIG["lth_sopr_deep_bear"]

    if sopr >= rubicon:
        return "BULL"
    if sopr >= early_bear:
        return "EARLY_BEAR"
    if sopr >= deep_bear:
        return "MID_BEAR"
    return "CAPITULATION"


def detect_lth_sopr_turning(history: list, window: int = 5) -> bool:
    """
    Паттерн В (пост 05.04.2026): SOPR перестаёт падать и начинает расти.

    True если:
      — в последних window значениях min достигнут ДО последней точки
        (= history[-1] > min(history[-window:]))

    False если:
      — SOPR ещё падает (min на последней позиции)
      — SOPR застыл на дне (last == min)
      — недостаточно данных (len(history) < window)

    WHY history[-1] > min(window): эквивалентно условию «мин достигнут до последней
    позиции И last > min»: если last > min, то min заведомо не на последней позиции.
    Отдельное поиска argmin избыточно.
    """
    if len(history) < window:
        return False

    w = history[-window:]
    return float(w[-1]) > min(float(v) for v in w)


# ---------------------------------------------------------------------------
# М-10 | LTH Realized Loss — зонный классификатор + % от исторического пика
# ---------------------------------------------------------------------------

def classify_lth_realized_loss(loss_usd: float) -> str:
    """
    Классифицирует реализованный убыток LTH по якорям паттерна М-10
    (пост 02.04.2026).

    ⚠️  Принимает raw значение из API BGeometrics (отрицательное — убыток).
    Вызывает abs(loss_usd) перед сравнением с якорями.

    Зоны (строгое >=, приоритет сверху вниз по abs):
      EXTREME          : abs >= anchor_cycle_target  (>= $500M)
      PEAK_FTX_RANGE   : abs >= anchor_2022_ftx      ($480M–$500M)
      MID_2022_RANGE   : abs >= anchor_2022_w1       ($300M–$480M)
      EARLY_2018_RANGE : abs >= anchor_2018          ($140M–$300M)
      BELOW_2018       : abs <  anchor_2018          (< $140M)

    Пороги из MOZART_CONFIG — не хардкодятся.
    """
    a2018    = MOZART_CONFIG["lth_loss_anchor_2018"]
    a2022_w1 = MOZART_CONFIG["lth_loss_anchor_2022_w1"]
    a2022ftx = MOZART_CONFIG["lth_loss_anchor_2022_ftx"]
    acycle   = MOZART_CONFIG["lth_loss_anchor_cycle_target"]

    val = abs(loss_usd)
    if val >= acycle:
        return "EXTREME"
    if val >= a2022ftx:
        return "PEAK_FTX_RANGE"
    if val >= a2022_w1:
        return "MID_2022_RANGE"
    if val >= a2018:
        return "EARLY_2018_RANGE"
    return "BELOW_2018"


def lth_loss_pct_of_historical_peak(loss_usd: float) -> float:
    """
    Процент текущего убытка LTH от исторического пика FTX-краша ($480M).

    Формула: abs(loss_usd) / anchor_2022_ftx * 100.0
    Знаменатель = anchor_2022_ftx, не anchor_cycle_target.
    Результат может превышать 100% (новый рекорд текущего цикла).

    ⚠️  Принимает оба знака входного значения — abs() вызывается внутри.
    """
    peak = float(MOZART_CONFIG["lth_loss_anchor_2022_ftx"])
    return abs(loss_usd) / peak * 100.0


# ---------------------------------------------------------------------------
# М-05 | LTH NUPL — рубикон + эйфория
# ---------------------------------------------------------------------------

def classify_lth_nupl_regime(nupl: float) -> str:
    """
    Классифицирует LTH NUPL по четырём зонам паттерна М-05
    (посты 05.04.2026, 15.05.2026).

    Зоны (приоритет сверху вниз):
      EUPHORIA : nupl >= euphoria (0.75)
                 Риск распределения LTH; высокая нереализованная прибыль,
                 верх к.ц. Порог взят из [LTH PAIN PROXY].
                 ⚠️ Mozart явно не называет 0.75 для LTH NUPL в этих постах.

      POSITIVE : rubicon + eps < nupl < euphoria
                 LTH в нереализованной прибыли; рынок здоров.

      RUBICON  : rubicon - eps <= nupl <= rubicon + eps
                 Граница прибыль↔убыток (±eps шума дневных данных).
                 Mozart: «Рубиконом является безубыток» (пост 05.04.2026,
                 аналогия с SOPR). Пересечение нуля — событие, не просто порог.

      BEAR     : nupl < rubicon - eps
                 LTH в убытке; повышенный риск вынужденных продаж.

    Границы:
      nupl == euphoria        → EUPHORIA (включительно).
      nupl == rubicon + eps   → RUBICON  (не POSITIVE; верхний край включительно).
      nupl == rubicon - eps   → RUBICON  (не BEAR; нижний край включительно).

    Пороги из MOZART_CONFIG — не хардкодятся.
    """
    euphoria = MOZART_CONFIG["lth_nupl_euphoria"]
    rubicon  = MOZART_CONFIG["lth_nupl_rubicon"]
    eps      = MOZART_CONFIG["lth_nupl_rubicon_eps"]

    if nupl >= euphoria:
        return "EUPHORIA"
    if nupl > rubicon + eps:
        return "POSITIVE"
    if nupl >= rubicon - eps:
        return "RUBICON"
    return "BEAR"


# ---------------------------------------------------------------------------
# М-02 | STH SOPR — рубикон безубытка
# ---------------------------------------------------------------------------

def classify_sth_sopr_regime(sopr: float) -> str:
    """
    Классифицирует STH SOPR по трём зонам паттерна М-02 (пост 16.04.2026).

    Зоны (приоритет сверху вниз):
      sopr >  rubicon + eps          → 'BULL'    — STH продают выше себестоимости
      sopr >= rubicon - eps          → 'RUBICON' — STH на безубытке (зона давления/поддержки)
      sopr <  rubicon - eps          → 'BEAR'    — STH фиксируют убыток, капитуляция

    Mozart (пост 16.04.2026):
      «нейтральная зона... сильная поддержка на бычьих,
       и сильное сопротивление на медвежьих рынках».

    Границы:
      sopr == rubicon + eps → RUBICON (не BULL; верхний край включительно).
      sopr == rubicon - eps → RUBICON (не BEAR; нижний край включительно).

    Пороги из MOZART_CONFIG — не хардкодятся.
    """
    rubicon = MOZART_CONFIG["sth_sopr_rubicon"]
    eps     = MOZART_CONFIG["sth_sopr_rubicon_eps"]

    if sopr > rubicon + eps:
        return "BULL"
    if sopr >= rubicon - eps:
        return "RUBICON"
    return "BEAR"


# ---------------------------------------------------------------------------
# М-02-Т | STH SOPR — детектор вектора пересечения рубикона
# ---------------------------------------------------------------------------

# Числовой ранг зон для сравнения направления движения.
# WHY отдельный dict, не enum: dict дешевле на импорт и прозрачнее в тестах;
# при добавлении зон достаточно обновить dict и константу.
_STH_SOPR_ZONE_RANK: dict[str, int] = {
    'BEAR':    0,
    'RUBICON': 1,
    'BULL':    2,
}


def detect_sth_sopr_turning(history: list, window: int = 5) -> 'str | None':
    """
    М-02-Т (пост 16.04.2026): детектор вектора пересечения рубикона STH SOPR.

    Принимает список зон ['BEAR'/'RUBICON'/'BULL'] — строки из
    classify_sth_sopr_regime(). Анализирует последние window элементов.

    Возвращает:
      'UPWARD'   — последняя зона выше минимума окна
                   (STH движутся к безубытку или пробивают его вверх)
      'DOWNWARD' — последняя зона ниже максимума окна
                   (STH теряют поддержку безубытка или уходят в капитуляцию)
      None       — нет направленного движения
                   (все зоны одинаковы или последняя == min и == max)

    WHY работает с зонами, не с float:
      classify_sth_sopr_regime() — единственный источник истины по зонам.
      Повторное сравнение с rubicon/eps внутри детектора дублировало бы
      production-логику в нарушение правила проекта.

    WHY отдельный ключ sth_sopr_turning_window (не lth_sopr_turning_window):
      Пороги разных сигналов меняются независимо — прецедент М-01-Т/М-09.

    WHY 'DOWNWARD' → BULLISH в полярности (контрарианский сигнал):
      Пробой STH SOPR вниз = капитуляция STH = дно близко.
      Аналог detect_lth_sopr_turning (True → BULLISH) и М-09.
    """
    if len(history) < window:
        return None

    w = history[-window:]
    ranks = [_STH_SOPR_ZONE_RANK.get(z, 1) for z in w]

    last    = ranks[-1]
    min_rank = min(ranks)
    max_rank = max(ranks)

    if last > min_rank:
        return 'UPWARD'
    if last < max_rank:
        return 'DOWNWARD'
    return None


# ---------------------------------------------------------------------------
# М-11 | ETF Flow — светофор спроса
# ---------------------------------------------------------------------------

def classify_etf_flow_regime(flow_btc: float) -> str:
    """
    Классифицирует суточный поток BTC в спотовые ETF по трём зонам паттерна М-11
    (пост 08.04.2026).

    Зоны (приоритет сверху вниз):
      INFLOW  : flow_btc >  significant
                Институциональный спрос активен; ETF-покупатель.

      NEUTRAL : -significant <= flow_btc <= significant
                Нет направленного сигнала; ETF-поток в пределах шума.
                Верхняя граница включительно: flow == significant → NEUTRAL.
                Нижняя граница включительно: flow == -significant → NEUTRAL.

      OUTFLOW : flow_btc <  -significant
                Нет нового институционального покупателя; ETF-продавец.
                Mozart (пост 08.04.2026): «нет нового покупателя» = ослабляет бычий тезис.

    WHY строгий > для INFLOW (не >=):
      Mozart оперирует бинарно «есть притоки / нет». Значение ровно на пороге считается
      шумом, не сигналом. По аналогии с LTH NUPL (верхняя граница RUBICON
      включительно снизу — FORMALIZED 2026-05-18).

    Порог из MOZART_CONFIG [«etf_flow_significant_btc»] — не хардкодится.
    """
    significant = float(MOZART_CONFIG["etf_flow_significant_btc"])

    if flow_btc > significant:
        return "INFLOW"
    if flow_btc >= -significant:
        return "NEUTRAL"
    return "OUTFLOW"


# ---------------------------------------------------------------------------
# М-03 | LTH MVRV — бинарный рубикон
# ---------------------------------------------------------------------------

def classify_lth_mvrv_regime(mvrv: float) -> str:
    """
    Классифицирует LTH MVRV по двум зонам паттерна М-03 (пост 25.02.2026).

    Зоны (бинарный рубикон, без eps):
      BULL : mvrv >= rubicon (1.0)
             LTH в нереализованной прибыли; среднестатистический LTH выше себестоимости.
             Мозарт: LTH не капитулируют принудительно — структурная поддержка.

      BEAR : mvrv < rubicon (1.0)
             LTH в убытке когортно; вынужденные продажи снижаются (пост 25.02.2026).

    Граница:
      mvrv == rubicon (1.0) → BULL (включительно; рубикон не пробит вниз).

    Нет eps: LTH MVRV — агрегированная метрика всей когорты LTH;
    дневные флуктуации менее значимы, чем у STH (высокая ротация).

    Порог из MOZART_CONFIG — не хардкодится.
    """
    rubicon = MOZART_CONFIG["lth_mvrv_rubicon"]

    if mvrv >= rubicon:
        return "BULL"
    return "BEAR"


# ---------------------------------------------------------------------------
# М-04 | STH MVRV — рубикон с eps-буфером
# ---------------------------------------------------------------------------

def classify_sth_mvrv_regime(mvrv: float) -> str:
    """
    Классифицирует STH MVRV по трём зонам паттерна М-04 (пост 16.04.2026).

    Зоны (приоритет сверху вниз):
      BULL    : mvrv > rubicon + eps
                STH продают с прибылью → давление продаж активно.

      NEUTRAL : rubicon - eps <= mvrv <= rubicon + eps
                STH у безубытка; нейтральная зона давления/поддержки.
                Верхняя граница включительно: mvrv == rubicon + eps → NEUTRAL.
                Нижняя граница включительно: mvrv == rubicon - eps → NEUTRAL.

      BEAR    : mvrv < rubicon - eps
                STH когортно в убытке → капитуляция.

    eps = MOZART_CONFIG["sth_mvrv_rubicon_eps"] = 0.02 (±2%, FORMALIZED).
    Мозарт не называет eps явно; 0.02 = типичный дневной шум STH MVRV.

    Пороги из MOZART_CONFIG — не хардкодятся.
    """
    rubicon = MOZART_CONFIG["sth_mvrv_rubicon"]
    eps     = MOZART_CONFIG["sth_mvrv_rubicon_eps"]

    if mvrv > rubicon + eps:
        return "BULL"
    if mvrv >= rubicon - eps:
        return "NEUTRAL"
    return "BEAR"


# ---------------------------------------------------------------------------
# М-06 | STH NUPL — рубикон капитуляции (асимметричный буфер)
# ---------------------------------------------------------------------------

def classify_sth_nupl_regime(nupl: float) -> str:
    """
    Классифицирует STH NUPL по трём зонам паттерна М-06 (пост 16.04.2026).

    Зоны (приоритет сверху вниз):
      POSITIVE     : nupl >  rubicon
                     STH когорта в нереализованной прибыли.
                     Mozart: «запас на рост по-прежнему имеет место» (пост 16.04.2026).

      RUBICON      : nupl >= rubicon - eps
                     STH на нуле или чуть ниже; зона максимального давления.
                     Верхняя граница включительно: nupl == 0.0 → RUBICON (не POSITIVE).
                     Нижняя граница включительно: nupl == -eps → RUBICON (не CAPITULATION).

      CAPITULATION : nupl <  rubicon - eps
                     STH когортно в убытке; смена режима давления.

    Асимметрия рубикона (отличие от classify_lth_nupl_regime):
      LTH NUPL: симметричный eps-буфер вокруг 0.0 (и сверху, и снизу).
      STH NUPL: eps только снизу; nupl == 0.0 → RUBICON, не POSITIVE.
      Нет зоны EUPHORIA (3 зоны вместо 4 у LTH NUPL).

    Пороги из MOZART_CONFIG — не хардкодятся.
    """
    rubicon = MOZART_CONFIG["sth_nupl_rubicon"]
    eps     = MOZART_CONFIG["sth_nupl_rubicon_eps"]

    if nupl > rubicon:
        return "POSITIVE"
    if nupl >= rubicon - eps:
        return "RUBICON"
    return "CAPITULATION"


# ---------------------------------------------------------------------------
# М-07 + М-08 | Cohort Flow — совместный переток LTH↔STH (пост 14.01.2026)
# ---------------------------------------------------------------------------

def classify_cohort_flow(lth_net_pos: float, sth_net_pos: float) -> str:
    """
    Классифицирует совместный переток монет между когортами LTH↔STH (пост 14.01.2026).

    4 квадранта по знаку (знак нуля считается положительным):

      lth >= 0 и sth <  0  → 'ACCUMULATION'  — STH продают → LTH, бычий фон
      lth <  0 и sth >= 0  → 'DISTRIBUTION'  — LTH продают → STH, медвежий фон
      lth >= 0 и sth >= 0  → 'BOTH_BUYING'   — обе когорты накапливают (редко)
      lth <  0 и sth <  0  → 'BOTH_SELLING'  — обе когорты продают (стресс)

    WHY совместно: Mozart: «LTH Net Pos + STH Net Pos должны анализироваться
    вместе — разнонаправленность подтверждает переток» (пост 14.01.2026).
    WHY знак нуля = положительный: нейтральная позиция когорты не является
    давлением продаж. Только отрицательное значение = фактическое распредение.
    WHY нет числовых порогов: Mozart оперирует качественно — знак
    достаточен для бинарного сигнала (пост 14.01.2026).
    """
    lth_positive = lth_net_pos >= 0.0
    sth_positive = sth_net_pos >= 0.0

    if lth_positive and not sth_positive:
        return "ACCUMULATION"
    if not lth_positive and sth_positive:
        return "DISTRIBUTION"
    if lth_positive and sth_positive:
        return "BOTH_BUYING"
    return "BOTH_SELLING"


# ---------------------------------------------------------------------------
# М-12 | HODL Waves — направление когорт (delta) (пост 13.05.2026)
# ---------------------------------------------------------------------------

def classify_hodl_wave_regime(
    age_1m_3m_current: float,
    age_1m_3m_prev: float,
    age_3m_6m_current: float,
    age_3m_6m_prev: float,
) -> str:
    """
    Классификация по направлению внутрикогортного сдвига (пост 13.05.2026).

    AGING        : age_1m_3m_current < age_1m_3m_prev
                   И age_3m_6m_current > age_3m_6m_prev
                   Монеты стареют из 1–3м в 3–6м = фаза накопления/холда.

    REJUVENATING : age_1m_3m_current > age_1m_3m_prev
                   И age_3m_6m_current < age_3m_6m_prev
                   Монеты молодеют из 3–6м в 1–3м = фаза движения/распределения.

    MIXED        : иначе — нет направленного сигнала.

    WHY строгое неравенство (< / >), не ≤/≥:
      Mozart описывает ИЗМЕНЕНИЕ направления когорт — нулевая дельта
      означает отсутствие перетока, а не его наличие (пост 13.05.2026).
    WHY нет числовых порогов: только знак изменения достаточен для
      бинарного сигнала по паттерну М-12.
    """
    aging = (
        age_1m_3m_current < age_1m_3m_prev
        and age_3m_6m_current > age_3m_6m_prev
    )
    rejuvenating = (
        age_1m_3m_current > age_1m_3m_prev
        and age_3m_6m_current < age_3m_6m_prev
    )

    if aging:
        return "AGING"
    if rejuvenating:
        return "REJUVENATING"
    return "MIXED"


# ---------------------------------------------------------------------------
# М-09 | STH Realized Price — детектор разворота Z-score (паттерн В)
# ---------------------------------------------------------------------------

def detect_sth_rp_zscore_turning(zscore_history: list[float], window: int = 5) -> bool:
    """
    Паттерн В (пост 05.04.2026): Z-score STH Realized Price перестаёт падать
    и начинает расти при стоячей цене.

    True если:
      — len(zscore_history) >= window
      — последний элемент > min(последних window элементов)

    False если:
      — недостаточно данных (len < window)
      — Z-score ещё падает (min на последней позиции)
      — Z-score застыл на дне (last == min)

    Аналог detect_lth_sopr_turning (М-01, ВЕТКА 1) — применяется к Z-score STH RP,
    а не к значению SOPR напрямую.

    WHY last > min(window): эквивалентно условию «мин достигнут до последней позиции
    И last > min»: если last > min, то min заведомо не на последней позиции.
    Отдельный поиск argmin избыточен.
    """
    if len(zscore_history) < window:
        return False
    w = zscore_history[-window:]
    return bool(float(w[-1]) > min(float(v) for v in w))


# ---------------------------------------------------------------------------
# МБ-02 | True Market Mean — «Зелёная линия», рубикон медвежьего рынка
# ---------------------------------------------------------------------------

def classify_true_market_mean_regime(price: float, tmm: float) -> str:
    """
    Классифицирует позицию цены относительно True Market Mean по паттерну МБ-02.

    Бинарный рубикон (Mozart, пост 25.02.2026):
        price >= tmm  →  'ABOVE'   (рубикон не пробит вниз)
        price <  tmm  →  'BELOW'   (медвежий рынок подтверждён)

    Граница включительно:
        price == tmm  →  'ABOVE'  (цена на рубиконе — не пробой)

    Нет буфера, нет AT-зоны (в отличие от МБ-01):
        TMM — «твёрдый рубикон»; Mozart описывает смену тренда только при
        реальном пробое вниз. Нахождение ровно на линии — не пробой.

    Args:
        price: Текущая цена BTC (USD).
        tmm:   True Market Mean (USD) — «зелёная линия» модели Mozart.

    Returns:
        str: 'ABOVE' | 'BELOW'

    WHY >= для ABOVE: ровно на TMM = рубикон не пробит (Mozart: «пробой вниз
        на объёмах» = смена тренда; само нахождение на линии ≠ пробой).
    WHY нет буфера: TMM — бинарный рубикон; буфер вносил бы AT-зону, которую
        Mozart не описывает. МБ-01 имеет буфер, МБ-02 — нет.
    """
    if price >= tmm:
        return "ABOVE"
    return "BELOW"


# ---------------------------------------------------------------------------
# МБ-04 | Supply in Loss — счётчик монет в убытке
# ---------------------------------------------------------------------------

def classify_supply_loss_regime(supply_loss_btc: float) -> str:
    """
    Классифицирует объём BTC в убытке по паттерну МБ-04.

    Зоны (пороги из MOZART_CONFIG):
        >= structural_trigger  (5M BTC)  →  'EXTREME'
        >= intermediate_trigger (3.5M)   →  'ELEVATED'
        >  0  (ниже 3.5M)               →  'INTERMEDIATE'
        <= 0                             →  'LOW'

    Границы включительно:
        supply_loss_btc == structural_trigger   →  'EXTREME'
        supply_loss_btc == intermediate_trigger →  'ELEVATED'

    Args:
        supply_loss_btc: Объём BTC в нереализованном убытке (float).
                         Источник: /v1/supply-loss → поле supplyLoss.

    Returns:
        str: 'EXTREME' | 'ELEVATED' | 'INTERMEDIATE' | 'LOW'

    WHY четыре зоны:
        EXTREME     — >= 5M: исторический триггер смены структурного тренда
                      (посты 02.04.2026, 08.04.2026). Пики 2019/2022.
        ELEVATED    — 3.5M–5M: активное давление убыточных монет над ценой;
                      Mozart: «3–3.5M = сопротивление для роста» (08.04.2026).
        INTERMEDIATE — (0, 3.5M): ниже ключевых Mozart-уровней, давление есть.
        LOW         — <= 0: теоретически чистый бычий рынок; в данных BTC
                      практически не встречается, но функция обязана быть
                      устойчивой к любому float-вводу.

    WHY >= для обеих верхних границ: достижение порога = активация сигнала
    (Mozart оперирует «коррекция к 5M» как событием, а не только как
    снижением ниже него).
    """
    structural   = float(MOZART_CONFIG["supply_loss_structural_trigger"])
    intermediate = float(MOZART_CONFIG["supply_loss_intermediate_trigger"])

    if supply_loss_btc >= structural:
        return "EXTREME"
    if supply_loss_btc >= intermediate:
        return "ELEVATED"
    if supply_loss_btc > 0:
        return "INTERMEDIATE"
    return "LOW"


# ---------------------------------------------------------------------------
# МБ-08 | One-Cycle Average — средняя когорты 2–4 года (пост 13.05.2026)
# ---------------------------------------------------------------------------

def count_consecutive_days_below(df, threshold: float) -> int:
    """
    Считает последовательные дни С КОНЦА df где df['close'] < threshold.

    Используется оркестратором для вычисления days_below перед передачей в
    classify_one_cycle_regime(..., days_below=N).

    Args:
        df:        DataFrame с колонкой 'close' (дневные klines).
        threshold: Пороговая цена (OCA в контексте МБ-08).

    Returns:
        int: Количество последовательных дней с конца где close < threshold.
             0 если df пуст или последний день >= threshold.

    WHY строгое <: аналогично classify_one_cycle_regime (price >= oca → ABOVE);
        close == threshold = рубикон не пробит, день не засчитывается.
    WHY с конца: оркестратор отображает ТЕКУЩУЮ непрерывную серию,
        а не исторический максимум; пробел сбрасывает счётчик.
    """
    count = 0
    for close in reversed(df['close'].tolist()):
        if float(close) < threshold:
            count += 1
        else:
            break
    return count


def calculate_one_cycle_average(
    age_2y_3y_rc_frac: float,
    age_3y_4y_rc_frac: float,
    realized_price: float,
    age_2y_3y_supply_btc: float,
    age_3y_4y_supply_btc: float,
    total_supply_btc: float,
) -> float:
    """
    Вычисляет One-Cycle Average — среднюю цену покупки когорты
    «держателей одного цикла» (Bitcoin — 2-4 года). Mozart, пост 13.05.2026.

    Источники данных:
      age_2y_3y_rc_frac, age_3y_4y_rc_frac:
          Доли от общего realized cap из /v1/realized-cap-hodl-waves.
          Диагностика 20.05.2026: 0.056 и 0.051.
      realized_price:
          Средняя цена покупки всех BTC («Синяя линия», МБ-01).
      age_2y_3y_supply_btc, age_3y_4y_supply_btc:
          Количество BTC в когортах из /v1/hodl-waves-supply.
          Диагностика 20.05.2026: 1,121,764 и 1,021,607 BTC.
      total_supply_btc:
          Сумма всех age_* полей из hodl-waves-supply (≈ 20.0M BTC).

    Формула:
      total_realized_cap    = realized_price × total_supply_btc
      cohort_realized_cap   = (rc_frac_2y3y + rc_frac_3y4y) × total_realized_cap
      cohort_supply         = supply_2y3y_btc + supply_3y4y_btc
      OCA                   = cohort_realized_cap / cohort_supply

    Упрощенная запись:
      OCA = (rc_frac_2y3y + rc_frac_3y4y) × realized_price × total_supply_btc
            ──────────────────────────────────────────────
                    supply_2y3y_btc + supply_3y4y_btc

    Returns:
        float: OCA в USD. 0.0 если cohort_supply <= 0 (защита от ZeroDivisionError).

    WHY нет порогов из MOZART_CONFIG: функция пуро математическая,
    пороги определяются формулой, а не конфигом.
    WHY два эндпоинта: realized-cap-hodl-waves даёт RC-доли (удельный вес),
    hodl-waves-supply даёт количество BTC (знаменатель) — оба необходимы.
    """
    cohort_supply = age_2y_3y_supply_btc + age_3y_4y_supply_btc
    if cohort_supply <= 0:
        return 0.0
    cohort_rc_frac = age_2y_3y_rc_frac + age_3y_4y_rc_frac
    return float(cohort_rc_frac * realized_price * total_supply_btc / cohort_supply)


def classify_one_cycle_regime(
    price: float,
    one_cycle_avg: float,
    days_below: int = 0,
) -> str:
    """
    Классифицирует позицию цены относительно One-Cycle Average (пост 13.05.2026).

    Зоны (приоритет сверху вниз):
      ABOVE          : price >= one_cycle_avg
                       Цена выше или на OCA — рубикон не пробит.

      TECHNICAL_BEAR : price < one_cycle_avg AND days_below <= confirmed_days
                       Mozart: «Техническая медвежка» — цена ниже OCA,
                       но ещё недостаточно долго.

      CONFIRMED_BEAR : price < one_cycle_avg AND days_below > confirmed_days
                       Mozart: «Настоящий медвежий рынок» — цена ниже OCA
                       более 2 месяцев (пост 13.05.2026).

    Границы:
      price == one_cycle_avg          → ABOVE          (рубикон строго <, не ≤)
      days_below == confirmed_days     → TECHNICAL_BEAR  (порог строго >, не ≥)
      price > one_cycle_avg с любым days_below → ABOVE (возвращение выше OCA снимает режим)

    Args:
        price:         Текущая цена BTC (USD).
        one_cycle_avg: One-Cycle Average в USD (из calculate_one_cycle_average).
        days_below:    Количество дней подряд цена ниже OCA (0 → независимость).

    Returns:
        str: 'ABOVE' | 'TECHNICAL_BEAR' | 'CONFIRMED_BEAR'

    WHY days_below параметр, не внутренний счётчик:
        Пурая функция не имеет состояния; оркестратор передаёт
        накопленный счётчик из исторического ряда (по аналогии с detect_lth_sopr_turning).
    """
    confirmed_days = int(MOZART_CONFIG["one_cycle_bear_confirmed_days"])

    if price >= one_cycle_avg:
        return "ABOVE"
    if days_below > confirmed_days:
        return "CONFIRMED_BEAR"
    return "TECHNICAL_BEAR"


# ---------------------------------------------------------------------------
# МБ-07 | MVRV Z-Score — макро позиционирование (пост 25.02.2026)
# ---------------------------------------------------------------------------

def classify_mvrv_zscore_regime(z: float) -> str:
    """
    Классифицирует MVRV Z-Score по пяти зонам паттерна МБ-07 (пост 25.02.2026).

    Endpoint: /v1/mvrv-zscore → поле mvrvZscore (float).

    Зоны (приоритет сверху вниз):
      PEAK    : z >  peak_threshold (7.0)
                Исторический топ цикла; в данных 2022–2026 не достигается.

      BULL    : z >= bull_threshold (3.0)
                Сильный бычьий рынок; рынок значительно выше Realized Price.

      NEUTRAL : z >= 0.0
                Рынок выше или ровно на Realized Price.
                Текущее (19.05.2026): z = 0.7759 → NEUTRAL.

      BEAR    : z >  bottom_threshold (-1.0)
                Рынок немного ниже Realized Price.
                Mozart: «Z < 0 = рынок ниже realized price» (пост 25.02.2026).

      BOTTOM  : z <= bottom_threshold (-1.0)
                Рынок значительно ниже Realized Price.
                Mozart: «зона исторического дна» в контексте МБ-01.

    Границы:
      z == peak_threshold  → BULL    (PEAK строго >, достижение порога ≠ экстремум)
      z == bull_threshold  → BULL    (нижняя граница включительно)
      z == 0.0             → NEUTRAL (рынок ровно на Realized Price = нейтраль)
      z == bottom_threshold→ BOTTOM  (включительно; достижение = сигнал дна)

    Пороги из MOZART_CONFIG — не хардкодятся.
    """
    peak   = float(MOZART_CONFIG["mvrv_zscore_peak"])
    bull   = float(MOZART_CONFIG["mvrv_zscore_bull"])
    bottom = float(MOZART_CONFIG["mvrv_zscore_bottom"])

    if z > peak:
        return "PEAK"
    if z >= bull:
        return "BULL"
    if z >= 0.0:
        return "NEUTRAL"
    if z > bottom:
        return "BEAR"
    return "BOTTOM"


# ---------------------------------------------------------------------------
# МБ-06 | Общий NUPL — зоны медвежьего рынка (пост 15.05.2026)
# ---------------------------------------------------------------------------

def classify_nupl_regime(nupl: float) -> str:
    """
    Классифицирует общий NUPL по пяти зонам паттерна МБ-06 (пост 15.05.2026).

    Endpoint: /v1/nupl → поле nupl (десятичная дробь; 0.3004 = 30.04%).

    Зоны (приоритет сверху вниз):
      BULL        : nupl >= bull_threshold (0.50)
                    Эйфория / распределение; исторически топы бычьего цикла.

      HOPE        : hope_threshold (0.25) <= nupl < bull_threshold
                    Оптимизм / восстановление.
                    Текущее (19.05.2026): nupl = 0.3004 → HOPE.

      EARLY_BEAR  : 0.0 <= nupl < hope_threshold (0.25)
                    Ранняя медвежка; Mozart's пик начала марта 2026 ~18% (0.18)
                    попадает сюда.

      BEAR        : bottom_target < nupl < 0.0
                    Рынок в нереализованном убытке; приближение к цели дна.

      BOTTOM_ZONE : nupl <= bottom_target (-0.40)
                    Mozart (пост 15.05.2026): «40%-ая доля нереализованных убытков»
                    = исторический целевой уровень дна текущего цикла.

    Границы:
      nupl == bull_threshold  → BULL        (включительно; достижение = активация)
      nupl == hope_threshold  → HOPE        (нижняя граница включительно)
      nupl == 0.0             → EARLY_BEAR  (рубикон нуля = нижняя граница)
      nupl == bottom_target   → BOTTOM_ZONE (включительно; достижение = сигнал дна)

    Пороги из MOZART_CONFIG — не хардкодятся.
    """
    bull   = float(MOZART_CONFIG["nupl_bull_threshold"])
    hope   = float(MOZART_CONFIG["nupl_hope_threshold"])
    bottom = float(MOZART_CONFIG["nupl_bottom_target"])

    if nupl >= bull:
        return "BULL"
    if nupl >= hope:
        return "HOPE"
    if nupl >= 0.0:
        return "EARLY_BEAR"
    if nupl > bottom:
        return "BEAR"
    return "BOTTOM_ZONE"


# ---------------------------------------------------------------------------
# МБ-01 | Realized Price — «Синяя линия» дна цикла
# ---------------------------------------------------------------------------

def classify_realized_price_regime(price: float, realized_price: float) -> str:
    """
    Классифицирует позицию цены относительно Realized Price по паттерну МБ-01.

    Зоны (буфер = MOZART_CONFIG["realized_price_buffer_pct"]):
        price > realized_price * (1 + buf)                     → 'ABOVE'
        realized_price * (1 - buf) <= price
            <= realized_price * (1 + buf)                      → 'AT'
        price < realized_price * (1 - buf)                     → 'BELOW'

    Граница AT включительно с обеих сторон:
        price == realized_price * (1 + buf) → 'AT'  (не 'ABOVE')
        price == realized_price * (1 - buf) → 'AT'  (не 'BELOW')

    Args:
        price:          Текущая цена BTC (USD).
        realized_price: Realized Price (USD) — «синяя линия» модели Mozart.

    Returns:
        str: 'ABOVE' | 'AT' | 'BELOW'

    WHY строгое > для ABOVE: ровно на верхней границе = ещё AT (включительно).
    WHY строгое < для BELOW: ровно на нижней границе = ещё AT (включительно).
    WHY буфер из конфига: изменение порога — только в mozart_config.py.
    """
    buf   = float(MOZART_CONFIG["realized_price_buffer_pct"])
    upper = realized_price * (1.0 + buf)
    lower = realized_price * (1.0 - buf)

    if price > upper:
        return "ABOVE"
    if price < lower:
        return "BELOW"
    return "AT"


# ---------------------------------------------------------------------------
# МБ-05 | LTH Realized Profit USD — порог медвежьего давления
# ---------------------------------------------------------------------------

def classify_lth_realized_profit_regime(profit_7d_ma_usd: float) -> str:
    """
    Классифицирует 7-дневную MA дневного LTH Realized Profit по паттерну МБ-05.

    Endpoint: /v1/realized-profit-lth-usd → поле realizedProfitLthUsd (float, USD).
    7-дневная MA вычисляется оркестратором перед передачей сюда.

    Зоны:
      HIGH   : profit_7d_ma_usd >  lth_profit_7d_ma_warning ($1B)
               Mozart (пост 14.01.2026): «риски смещены в сторону медвежки»
               LTH фиксируют прибыль агрессивно; давление распределения высокое.

      NORMAL : profit_7d_ma_usd <= lth_profit_7d_ma_warning
               Давление в норме, аномалии нет.

    Граница:
      profit_7d_ma_usd == warning → 'NORMAL' (строгий >; ровно $1B = ещё не HIGH)

    Args:
        profit_7d_ma_usd: 7-дневная MA LTH realized profit (USD/день, >= 0).

    Returns:
        str: 'HIGH' | 'NORMAL'

    WHY строгий > для HIGH:
        Mozart: «выше 1 млрд $» — строгое неравенство в цитате.
        Ровно $1B = граница не превышена, сигнал ещё не активирован.
    """
    warning = float(MOZART_CONFIG["lth_profit_7d_ma_warning"])
    if profit_7d_ma_usd > warning:
        return "HIGH"
    return "NORMAL"


# ---------------------------------------------------------------------------
# НВ-03 | BTC Dominance — ротация ликвидности (пост 10.05.2026)
# ---------------------------------------------------------------------------

def classify_btc_dominance_trend(btc_d_current: float, btc_d_30d_ago: float) -> str:
    """
    Классифицирует направление BTC Dominance за скользящий месяц (пост 10.05.2026).

    API: CoinGecko /api/v3/global → поле btc_dominance (float, %).

    Семантика зон:
      delta = btc_d_current - btc_d_30d_ago

      ROTATION_ALTCOIN : delta < -threshold
                         BTC.D значительно снизился за месяц — ликвидность ротирует в альты.
                         Mozart (пост 10.05.2026): «рынок часто идет по цепочке:
                         низкая капа → средняя → высокая».

      ROTATION_BTC     : delta > +threshold
                         BTC.D значительно вырос за месяц — ликвидность уходит из альтов в BTC.

      NEUTRAL          : -threshold <= delta <= +threshold
                         Движение BTC.D в пределах шума — направленной ротации нет.

    Границы (строгое < / >):
      delta == -threshold → NEUTRAL  (не ROTATION_ALTCOIN)
      delta == +threshold → NEUTRAL  (не ROTATION_BTC)

    WHY строгое < для ROTATION_ALTCOIN:
        Mozart (пост 10.05.2026): «снижение BTC.D > 2%» — строгое неравенство.
        Ровно на пороге = сигнал не активирован; ошибка → ложный ROTATION_ALTCOIN
        при каждом движении ровно на 2 п.п. (частое рыночное значение).

    Порог из MOZART_CONFIG [«btc_dominance_rotation_threshold_pct»] — не хардкодится.
    """
    threshold = float(MOZART_CONFIG["btc_dominance_rotation_threshold_pct"])
    delta = btc_d_current - btc_d_30d_ago

    if delta < -threshold:
        return "ROTATION_ALTCOIN"
    if delta > threshold:
        return "ROTATION_BTC"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# НВ-01 | Регрессия убывающих пиков (пост 31.03.2026)
# ---------------------------------------------------------------------------

def find_swing_highs(
    highs: list,
    dates: list,
    n: int,
) -> list:
    """
    Находит подтверждённые локальные максимумы в ряду highs.

    Пик i подтверждён если:
      highs[i] > max(highs[i-n : i])       -- строго выше левого окна
      highs[i] >= max(highs[i+1 : i+n+1])  -- не ниже правого окна

    Асимметричное сравнение (Gemini validation 2026-06-01):
      слево > -- предотвращает дублирование при плоском левом плече.
      справа >= -- детектирует пик при плоском правом плече (двойная вершина).

    WHY highs, не closes:
      Mozart называет 87k/84k/76k -- это high свечи, не close.

    Returns:
        list of (index: int, price: float, date)
    """
    peaks = []
    length = len(highs)
    for i in range(n, length - n):
        val = highs[i]
        left  = highs[i - n : i]
        right = highs[i + 1 : i + n + 1]
        # WHY > слева, >= справа: предотвращает дубликацию при двойной вершине.
        if left and right:
            if val > max(left) and val >= max(right):
                peaks.append((i, float(val), dates[i]))
    return peaks


def classify_descending_peaks(
    highs: list,
    dates: list,
) -> dict:
    """
    Классифицирует режим убывающих пиков (Mozart, пост 31.03.2026).

    Алгоритм:
      1. find_swing_highs
      2. Фильтр по max_days_between_peaks
      3. Последние K пиков
      4. < 3 пиков -> INSUFFICIENT_DATA
      5. Монотонность: all(Y[i] < Y[i-1])
      6. linregress(X, Y): slope, stderr
      7. FLAT если |slope| < 2*stderr, иначе DESCENDING/ASCENDING

    WHY монотонность вместо R2:
      R2 на 3-5 точках нестабилен. Gemini validation 2026-06-01.

    Returns:
        dict: regime, peaks_count, slope, projected_next, is_monotone
    """
    from scipy.stats import linregress

    n    = int(MOZART_CONFIG["swing_high_window"])
    k    = int(MOZART_CONFIG["swing_high_lookback_peaks"])
    mdbp = int(MOZART_CONFIG["max_days_between_peaks"])

    _insufficient: dict = {
        'regime': 'INSUFFICIENT_DATA',
        'peaks_count': 0,
        'slope': None,
        'projected_next': None,
        'is_monotone': None,
    }

    all_peaks = find_swing_highs(highs, dates, n)
    if not all_peaks:
        return _insufficient

    # Фильтр по интервалу между соседними пиками
    # WHY: пики с большим интервалом -- разные рыночные эпохи.
    def _days_between(p1, p2) -> int:
        d1, d2 = p1[2], p2[2]
        if hasattr(d1, 'date'):
            d1 = d1.date()
        if hasattr(d2, 'date'):
            d2 = d2.date()
        return abs((d2 - d1).days)

    filtered: list = [all_peaks[0]]
    for i in range(1, len(all_peaks)):
        if _days_between(all_peaks[i - 1], all_peaks[i]) <= mdbp:
            filtered.append(all_peaks[i])
        else:
            # Интервал превышен: сброс цепочки, стартуем новую
            filtered = [all_peaks[i]]

    peaks = filtered[-k:]

    if len(peaks) < 3:
        result = _insufficient.copy()
        result['peaks_count'] = len(peaks)
        return result

    Y = [p[1] for p in peaks]
    X = list(range(len(Y)))

    # Монотонность: каждый следующий < предыдущего
    is_monotone = all(Y[i] < Y[i - 1] for i in range(1, len(Y)))

    reg = linregress(X, Y)
    slope     = float(reg.slope)
    stderr    = float(reg.stderr)
    intercept = float(reg.intercept)
    projected_next = float(intercept + slope * len(Y))

    # Классификация по рекомендации Gemini (validation 2026-06-01):
    # знак slope -- главный признак, FLAT -- только когда slope близок к нулю.
    # WHY: stderr взрывается от немонотонности -- высокая волатильность
    # маскируется под флэт, если сравнивать |slope| с 2*stderr.
    price_scale = float(np.mean(Y)) if float(np.mean(Y)) != 0.0 else 1.0

    # Уровень 1: микро-флэт -- slope практически ноль относительно масштаба цены
    # WHY 0.001: движение < 0.1% от средней цены за шаг -- математический ноль.
    if abs(slope) / price_scale < 0.001:
        regime = 'FLAT'
    elif slope < 0 and is_monotone:
        # Уровень 2: строгий нисходящий тренд -- регрессия вниз + монотонность
        regime = 'DESCENDING_STRONG'
    elif slope < 0:
        # Уровень 3: общий вектор вниз, но есть нарушения
        # WHY 0.5*stderr: тотальный хаос -- slope < половины ошибки
        # (Gemini: в этом случае slope не имеет смысла как метрика)
        if stderr > 0 and abs(slope) < 0.5 * stderr:
            regime = 'FLAT'
        else:
            regime = 'DESCENDING_WEAK'
    else:
        regime = 'ASCENDING'

    return {
        'regime'        : regime,
        'peaks_count'   : len(peaks),
        'slope'         : slope,
        'projected_next': projected_next,
        'is_monotone'   : is_monotone,
    }


def classify_funding_rate_ma_regime(ma_value: float) -> str:
    """
    М-15 | Режим 30-дневной MA funding rate.

    Mozart (пост 11.03.2026):
      «30-дневная MA funding rate на многолетнем минимуме = предшествует дну цикла»

    Args:
        ma_value: Среднее значение funding rate за 30 дней (в долях).
                  API возвращает значения ~0.0001 (базовая ставка Binance = +0.0001).
                  8-часовой интервал: 30d × 3 записи/день = 90 записей для расчёта.

    Returns:
        'FLOOR_ZONE'  -- ma_value <= funding_rate_ma_floor:
                         Mozart-паттерн активен (многолетний минимум).
        'NEUTRAL'     -- ma_value > funding_rate_ma_floor:
                         Обычный диапазон.

    WHY включительная граница (порог включается в FLOOR_ZONE):
        ma_value == floor — рынок уже достиг порога; паттерн подтверждён.
        Аналогично lth_sopr_rubicon (PLAN_MOZART_PATTERNS.md).
    """
    from mozart_config import MOZART_CONFIG
    floor = MOZART_CONFIG["funding_rate_ma_floor"]

    if ma_value <= floor:
        return 'FLOOR_ZONE'
    return 'NEUTRAL'


def classify_lth_profit_regime(profit_ma7_usd: float) -> str:
    """
    МБ-05 | Режим давления LTH по 7-дневной MA реализованной прибыли.

    Mozart (пост 14.01.2026):
      "если давление продаж вновь сильно возрастёт
      (выше 1 млрд $ / день в среднем за 7 дней),
      то риски будут смещены в сторону медвежки"

    Args:
        profit_ma7_usd: 7-дневная MA дневной прибыли LTH, USD (всегда >= 0).

    Returns:
        'HIGH_PRESSURE'  — MA >= warning ($1B): медвежий риск по Mozart.
        'MODERATE'       — moderate < MA < warning ($500M–$1B):
                           умеренное давление. FORMALIZED.
        'LOW'            — MA <= moderate (<= $500M): давление слабое.

    WHY включительная граница warning (порог Mozart включается в HIGH_PRESSURE):
        Mozart оперирует "выше" — т.е. >= 1B активирует риск.
        Аналогично lth_sopr_rubicon: 1.0 включительно входит в BULL.

    WHY включительная граница moderate (порог включается в LOW):
        FORMALIZED-порог $500M — нижняя граница MODERATE-зоны.
        Равно порогу = ещё не умеренное давление — включается в LOW.
    """
    from mozart_config import MOZART_CONFIG
    warning  = MOZART_CONFIG['lth_profit_7d_ma_warning']
    moderate = MOZART_CONFIG['lth_profit_7d_ma_moderate']

    if profit_ma7_usd >= warning:
        return 'HIGH_PRESSURE'
    if profit_ma7_usd > moderate:
        return 'MODERATE'
    return 'LOW'

