"""
tests/test_mozart_alignment_polarity.py
=======================================
TDD — L3-1: signal_polarity() (docs/PLAN_MOZART_LEVEL3_SIGNAL_ALIGNMENT.md).

Контракт:
    signal_polarity(signal_id: str, label: str) -> str
        Возвращает 'BULLISH' / 'NEUTRAL' / 'BEARISH'.
        Таблица полярности зафиксирована в mozart_alignment.py (_POLARITY_TABLE).
        Изменение интерпретации Mozart — только через таблицу, не через формулу.

    signal_polarity(unknown_id, ...)   -> ValueError
    signal_polarity(known_id, unknown) -> ValueError

Правила:
  - WHY-комментарий к каждому assert: что сломается в production.
  - Контрарианские сигналы — отдельные тест-методы с развёрнутым WHY.
  - Нет числовых плейсхолдеров: входы — строки-метки, не числа.
  - Граничные случаи (неизвестный ID, неизвестная метка) — отдельные тесты.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mozart_alignment import signal_polarity  # RED: модуль не создан


# ===========================================================================
# М-01 | LTH SOPR
# ===========================================================================

class TestM01LthSopr:
    """Контракт М-01: classify_lth_sopr_regime → BULL/EARLY_BEAR/MID_BEAR/CAPITULATION."""

    def test_bull_is_bullish(self):
        result = signal_polarity("М-01", "BULL")
        assert result == "BULLISH", (
            # WHY: BULL = LTH продают выше себестоимости; build_alignment() считает
            # этот сигнал бычьим. Не-BULLISH → сигнал пропадает из bullish-счётчика
            # и занижает score в AlignmentResult.
        )

    def test_early_bear_is_neutral(self):
        result = signal_polarity("М-01", "EARLY_BEAR")
        assert result == "NEUTRAL", (
            # WHY: EARLY_BEAR = LTH прошли рубикон, но убыток < 20% — Mozart
            # трактует как переходную зону, не однозначно медвежью. BEARISH здесь
            # завысит медвежий счётчик и даст ложный BEARISH verdict.
        )

    def test_mid_bear_is_bearish(self):
        result = signal_polarity("М-01", "MID_BEAR")
        assert result == "BEARISH", (
            # WHY: MID_BEAR = LTH фиксируют 20–50% убыток; активное давление
            # продавцов. Не-BEARISH → сигнал не попадёт в bearish-счётчик.
        )

    def test_capitulation_is_bearish(self):
        result = signal_polarity("М-01", "CAPITULATION")
        assert result == "BEARISH", (
            # WHY: CAPITULATION = убыток >50%, кульминация; это медвежий сигнал
            # высшей интенсивности. Иная полярность нарушает контракт таблицы.
        )


# ===========================================================================
# М-02 | STH SOPR
# ===========================================================================

class TestM02SthSopr:
    """Контракт М-02: classify_sth_sopr_regime → BULL/RUBICON/BEAR."""

    def test_bull_is_bullish(self):
        assert signal_polarity("М-02", "BULL") == "BULLISH", (
            # WHY: STH продают выше себестоимости = здоровый спрос. Не-BULLISH → недосчёт.
        )

    def test_rubicon_is_neutral(self):
        assert signal_polarity("М-02", "RUBICON") == "NEUTRAL", (
            # WHY: Mozart: рубикон = зона давления/поддержки, не однозначный сигнал.
            # BEARISH здесь ложно ухудшит score при боковом рынке.
        )

    def test_bear_is_bearish(self):
        assert signal_polarity("М-02", "BEAR") == "BEARISH", (
            # WHY: BEAR = STH фиксируют убыток; медвежья структура подтверждена.
        )


# ===========================================================================
# М-03 | LTH MVRV (бинарный)
# ===========================================================================

class TestM03LthMvrv:
    """Контракт М-03: classify_lth_mvrv_regime → BULL/BEAR (без NEUTRAL)."""

    def test_bull_is_bullish(self):
        assert signal_polarity("М-03", "BULL") == "BULLISH", (
            # WHY: LTH MVRV выше рубикона = у держателей нереализованная прибыль.
        )

    def test_bear_is_bearish(self):
        assert signal_polarity("М-03", "BEAR") == "BEARISH", (
            # WHY: LTH MVRV ниже рубикона = LTH в убытке; риск вынужденных продаж.
        )


# ===========================================================================
# М-04 | STH MVRV
# ===========================================================================

class TestM04SthMvrv:
    """Контракт М-04: classify_sth_mvrv_regime → BULL/NEUTRAL/BEAR."""

    def test_bull_is_bullish(self):
        assert signal_polarity("М-04", "BULL") == "BULLISH"
        # WHY: STH MVRV выше 1 = покупки в прибыли; структурно бычий рынок.

    def test_neutral_is_neutral(self):
        assert signal_polarity("М-04", "NEUTRAL") == "NEUTRAL"
        # WHY: Переходная зона без направленного сигнала. Счётчик neutral.

    def test_bear_is_bearish(self):
        assert signal_polarity("М-04", "BEAR") == "BEARISH"
        # WHY: STH MVRV ниже 1 = новые покупатели в убытке; давление продаж.


# ===========================================================================
# М-05 | LTH NUPL
# ===========================================================================

class TestM05LthNupl:
    """Контракт М-05: POSITIVE/RUBICON/BEAR/EUPHORIA.
    Особый случай: EUPHORIA — контрарианский BEARISH (см. отдельный метод)."""

    def test_positive_is_bullish(self):
        assert signal_polarity("М-05", "POSITIVE") == "BULLISH"
        # WHY: LTH в нереализованной прибыли, рынок здоров.

    def test_rubicon_is_neutral(self):
        assert signal_polarity("М-05", "RUBICON") == "NEUTRAL"
        # WHY: Граница прибыль↔убыток; Mozart: переходная зона, не сигнал.

    def test_bear_is_bearish(self):
        assert signal_polarity("М-05", "BEAR") == "BEARISH"
        # WHY: LTH в убытке; риск вынужденных продаж усиливается.

    def test_euphoria_is_bearish_contrarian(self):
        result = signal_polarity("М-05", "EUPHORIA")
        assert result == "BEARISH", (
            # WHY (КОНТРАРИАНСКИЙ): LTH NUPL EUPHORIA означает максимальную
            # нереализованную прибыль у долгосрочных держателей — исторически
            # совпадает с вершиной цикла и началом дистрибуции (продаж в рынок).
            # Mozart использует это как предупреждение о перегреве, а не как
            # «всё хорошо». BULLISH здесь создаст ложный бычий сигнал именно
            # на вершине цикла, когда агрегатор должен видеть риск.
            # Таблица: М-05 EUPHORIA → BEARISH (строка "*** МБ-03").
        )


# ===========================================================================
# М-06 | STH NUPL
# ===========================================================================

class TestM06SthNupl:
    """Контракт М-06: POSITIVE/RUBICON/CAPITULATION."""

    def test_positive_is_bullish(self):
        assert signal_polarity("М-06", "POSITIVE") == "BULLISH"
        # WHY: STH в прибыли; краткосрочный спрос здоров.

    def test_rubicon_is_neutral(self):
        assert signal_polarity("М-06", "RUBICON") == "NEUTRAL"
        # WHY: STH на безубытке; зона давления без однозначного вектора.

    def test_capitulation_is_bearish(self):
        assert signal_polarity("М-06", "CAPITULATION") == "BEARISH"
        # WHY: STH массово фиксируют убыток; кульминация давления продаж.


# ===========================================================================
# М-07+08 | Cohort Flow (LTH/STH нетто-позиция)
# ===========================================================================

class TestM0708CohortFlow:
    """Контракт М-07+08: ACCUMULATION/BOTH_BUYING/DISTRIBUTION/BOTH_SELLING.
    BOTH_BUYING — нейтральный, не бычий (обе когорты накапливают без перетока)."""

    def test_accumulation_is_bullish(self):
        assert signal_polarity("М-07+08", "ACCUMULATION") == "BULLISH"
        # WHY: LTH накапливают, STH продают LTH → явный структурный переток.

    def test_both_buying_is_neutral(self):
        result = signal_polarity("М-07+08", "BOTH_BUYING")
        assert result == "NEUTRAL", (
            # WHY: Обе когорты покупают одновременно, без явного перетока между ними.
            # Mozart: отсутствие структуры перетока не подтверждает бычий тезис.
            # BULLISH здесь завысит бычий счётчик без реального подтверждения.
            # Таблица: BOTH_BUYING → NEUTRAL (** сноска в документе).
        )

    def test_distribution_is_bearish(self):
        assert signal_polarity("М-07+08", "DISTRIBUTION") == "BEARISH"
        # WHY: LTH продают, STH покупают → сигнал вершины/распределения.

    def test_both_selling_is_bearish(self):
        assert signal_polarity("М-07+08", "BOTH_SELLING") == "BEARISH"
        # WHY: Все когорты продают → паника/капитуляция без покупателей.


# ===========================================================================
# М-09 | STH RP Z-score turning (bool-результат → строка)
# ===========================================================================

class TestM09SthRpZscoreTurning:
    """Контракт М-09: detect_sth_rp_zscore_turning возвращает bool;
    при передаче в signal_polarity используется str(bool): 'True'/'False'."""

    def test_true_is_bullish(self):
        assert signal_polarity("М-09", "True") == "BULLISH"
        # WHY: Разворот Z-score вверх = Паттерн В Mozart; смена направления
        # подтверждает конец давления STH. Оркестратор должен передавать
        # str(detect_sth_rp_zscore_turning(...)) → 'True'/'False'.

    def test_false_is_neutral(self):
        assert signal_polarity("М-09", "False") == "NEUTRAL"
        # WHY: Разворот не подтверждён; сигнала нет, не медвежий. BEARISH
        # здесь создаст ложное медвежье давление от отсутствия паттерна.


# ===========================================================================
# М-10 | LTH Realized Loss
# ===========================================================================

class TestM10LthRealizedLoss:
    """Контракт М-10: BELOW_2018/EARLY_2018_RANGE/MID_2022_RANGE/PEAK_FTX_RANGE/EXTREME.
    Особый случай: EXTREME — контрарианский BULLISH (см. отдельный метод)."""

    def test_below_2018_is_neutral(self):
        assert signal_polarity("М-10", "BELOW_2018") == "NEUTRAL"
        # WHY: Убыток меньше исторических якорей; нет аномального давления.
        # Mozart не делает вывода о направлении при малом убытке.

    def test_early_2018_range_is_bearish(self):
        assert signal_polarity("М-10", "EARLY_2018_RANGE") == "BEARISH"
        # WHY: Начало давления убыточных LTH; соответствует ранней фазе 2018.

    def test_mid_2022_range_is_bearish(self):
        assert signal_polarity("М-10", "MID_2022_RANGE") == "BEARISH"
        # WHY: Активная капитуляция; уровни разгара медвежьего рынка 2022.

    def test_peak_ftx_range_is_bearish(self):
        assert signal_polarity("М-10", "PEAK_FTX_RANGE") == "BEARISH"
        # WHY: Исторический максимум убытков (FTX-крах); сильное медвежье давление.

    def test_extreme_is_bullish_contrarian(self):
        result = signal_polarity("М-10", "EXTREME")
        assert result == "BULLISH", (
            # WHY (КОНТРАРИАНСКИЙ): EXTREME = убытки LTH превысили все исторические
            # прецеденты ($500M+). Mozart интерпретирует это как сигнал исчерпания
            # продавцов: когда все, кто должен был продать, уже продали с убытком,
            # дальнейшее давление продаж иссякает → разворот вероятен.
            # Исторический прецедент: 2020 (COVID-крах) — экстремальный убыток
            # предшествовал восстановлению. BEARISH здесь означал бы, что агрегатор
            # продолжает считать сигнал медвежьим именно в момент, когда Mozart
            # видит разворот. Это прямо нарушает архитектурный контракт таблицы.
        )


# ===========================================================================
# М-11 | ETF Flow
# ===========================================================================

class TestM11EtfFlow:
    """Контракт М-11: INFLOW/NEUTRAL/OUTFLOW."""

    def test_inflow_is_bullish(self):
        assert signal_polarity("М-11", "INFLOW") == "BULLISH"
        # WHY: Притоки в ETF = институциональный спрос; структурно бычий факт.

    def test_neutral_is_neutral(self):
        assert signal_polarity("М-11", "NEUTRAL") == "NEUTRAL"
        # WHY: ETF-потоки сбалансированы; нет направленного давления.

    def test_outflow_is_bearish(self):
        assert signal_polarity("М-11", "OUTFLOW") == "BEARISH"
        # WHY: Оттоки = институциональные продажи; медвежье давление сверху.


# ===========================================================================
# М-12 | HODL Waves
# ===========================================================================

class TestM12HodlWaves:
    """Контракт М-12: AGING/MIXED/REJUVENATING."""

    def test_aging_is_bullish(self):
        assert signal_polarity("М-12", "AGING") == "BULLISH"
        # WHY: Монеты стареют = долгосрочные держатели не продают; дефицит предложения.

    def test_mixed_is_neutral(self):
        assert signal_polarity("М-12", "MIXED") == "NEUTRAL"
        # WHY: Нет явного направления когорт; переходная фаза без сигнала.

    def test_rejuvenating_is_bearish(self):
        assert signal_polarity("М-12", "REJUVENATING") == "BEARISH"
        # WHY: Монеты «молодеют» = LTH продают, предложение на рынке растёт.


# ===========================================================================
# МБ-03 | STH Profit Zone
# ===========================================================================

class TestMB03SthProfitZone:
    """Контракт МБ-03: BEAR/NEUTRAL/NEUTRAL_BROKEN/HEATED/EUPHORIA_APPROACH/EUPHORIA.
    EUPHORIA упрощена до BULLISH (флаг euphoria_convergence добавляется в L3-2).
    Таблица: NEUTRAL_BROKEN, HEATED, EUPHORIA → BULLISH; NEUTRAL → NEUTRAL; BEAR → BEARISH."""

    def test_bear_is_bearish(self):
        assert signal_polarity("МБ-03", "BEAR") == "BEARISH"
        # WHY: <51% STH в прибыли = большинство краткосрочных держателей в убытке;
        # медвежья структура.

    def test_neutral_is_neutral(self):
        assert signal_polarity("МБ-03", "NEUTRAL") == "NEUTRAL"
        # WHY: 51–59% в прибыли = вокруг синей линии; зона без направленного сигнала.

    def test_neutral_broken_is_bullish(self):
        assert signal_polarity("МБ-03", "NEUTRAL_BROKEN") == "BULLISH"
        # WHY: Нейтральная пробита вверх (59–69%); восстановление подтверждено.

    def test_heated_is_bullish(self):
        assert signal_polarity("МБ-03", "HEATED") == "BULLISH"
        # WHY: 69–76% (жёлтая линия) — редкость для медвежьего рынка; бычий признак.

    def test_euphoria_approach_is_bullish(self):
        assert signal_polarity("МБ-03", "EUPHORIA_APPROACH") == "BULLISH"
        # WHY: 76–85% — переход к эйфории; STH преимущественно в прибыли;
        # агрегатор считает это бычьим (контрарианский риск учитывается флагом
        # euphoria_convergence в L3-2, не через полярность).

    def test_euphoria_is_bullish_simplified(self):
        result = signal_polarity("МБ-03", "EUPHORIA")
        assert result == "BULLISH", (
            # WHY (УПРОЩЕНИЕ): МБ-03 EUPHORIA была бы контрарианским BEARISH только
            # при совпадении с М-05 EUPHORIA. Это единственный контекст-зависимый
            # случай в таблице. Решение (PLAN_MOZART_LEVEL3...md): упростить до BULLISH,
            # добавить отдельный флаг euphoria_convergence в AlignmentResult (L3-2).
            # BEARISH здесь означал бы, что signal_polarity нарушает принцип чистой
            # функции без состояния — она не должна знать о других сигналах.
        )


# ===========================================================================
# Н-01 | RSI — КОНТРАРИАНСКИЙ сигнал
# ===========================================================================

class TestH01RsiContrarian:
    """Контракт Н-01: NEUTRAL/OVERSOLD/EXTREME_OVERSOLD.
    Оба не-нейтральных значения → BULLISH (контрарианский сигнал).
    ВНИМАНИЕ: название 'OVERSOLD'/'EXTREME_OVERSOLD' интуитивно кажется медвежьим.
    Mozart использует их как разворотные (паттерн Н-01, пост февраля 2026)."""

    def test_neutral_is_neutral(self):
        assert signal_polarity("Н-01", "NEUTRAL") == "NEUTRAL"
        # WHY: RSI в нормальной зоне; нет экстремума → нет разворотного сигнала.

    def test_oversold_is_bullish_contrarian(self):
        result = signal_polarity("Н-01", "OVERSOLD")
        assert result == "BULLISH", (
            # WHY (КОНТРАРИАНСКИЙ): Mozart: «перепроданность на дневном RSI —
            # исторически предшествовала отскокам» (паттерн Н-01). OVERSOLD не
            # означает «ещё упадёт», а означает «экстремум достигнут, вероятен
            # отскок». BEARISH/NEUTRAL здесь означали бы, что агрегатор игнорирует
            # разворотный сигнал, который Mozart явно считает бычьим.
        )

    def test_extreme_oversold_is_bullish_contrarian(self):
        result = signal_polarity("Н-01", "EXTREME_OVERSOLD")
        assert result == "BULLISH", (
            # WHY (КОНТРАРИАНСКИЙ, СИЛЬНЫЙ): EXTREME_OVERSOLD = RSI ниже абсолютного
            # порога; Mozart (пост февраля 2026) ссылается на прецедент 2020 года
            # (COVID-крах) как на исторически надёжный разворотный сигнал.
            # Интуитивно «экстремальная перепроданность» звучит страшно, но именно
            # так Mozart использует этот сигнал — как самый сильный контрарианский
            # бычий. NEUTRAL/BEARISH здесь означали бы пропуск самого редкого и
            # ценного сигнала в таблице Уровня 3.
        )


# ===========================================================================
# Н-02 | Red Months — КОНТРАРИАНСКИЙ сигнал
# ===========================================================================

class TestH02RedMonthsContrarian:
    """Контракт Н-02: NORMAL/RARE/EXTREME.
    RARE и EXTREME → BULLISH (контрарианский: редкость красных месяцев = дно).
    Аналогично Н-01: 'страшное' название, бычья полярность."""

    def test_normal_is_neutral(self):
        assert signal_polarity("Н-02", "NORMAL") == "NEUTRAL"
        # WHY: Обычное число красных месяцев; нет экстремума → нет сигнала.

    def test_rare_is_bullish_contrarian(self):
        result = signal_polarity("Н-02", "RARE")
        assert result == "BULLISH", (
            # WHY (КОНТРАРИАНСКИЙ): Mozart (пост 11.02.2026): 4+ красных месяца подряд
            # — статистически редкое событие; исторически соответствовало зонам
            # накопления. Серия красных месяцев НЕ означает «дальше хуже», а означает
            # «экстремальная перепроданность на месячном ТФ». NEUTRAL/BEARISH здесь
            # нарушают контракт таблицы и пропускают разворотный сигнал.
        )

    def test_extreme_is_bullish_contrarian(self):
        result = signal_polarity("Н-02", "EXTREME")
        assert result == "BULLISH", (
            # WHY (КОНТРАРИАНСКИЙ, СИЛЬНЫЙ): EXTREME = ≥N месяцев красных подряд;
            # единственный исторический прецедент — 2018 год (финальная фаза
            # медвежьего рынка). Mozart: настолько длинная серия не продолжается
            # дальше, статистика разворота очень сильная. BEARISH здесь дал бы
            # максимальный медвежий сигнал в момент, когда Mozart говорит «дно».
        )


# ===========================================================================
# Граничные случаи — неизвестные ID и метки
# ===========================================================================

class TestUnknownInputs:
    """Контракт: неизвестные signal_id или label → ValueError."""

    def test_unknown_signal_id_raises(self):
        with pytest.raises(ValueError):
            signal_polarity("НЕСУЩЕСТВУЮЩИЙ", "BULL")
        # WHY: build_alignment() передаёт только известные ID из _POLARITY_TABLE.
        # Если ID не в таблице — это ошибка вызывающего кода, не тихий возврат NEUTRAL.
        # ValueError принудит разработчика заметить пропущенный сигнал, а не получить
        # молчаливо неверный score.

    def test_unknown_label_for_known_signal_raises(self):
        with pytest.raises(ValueError):
            signal_polarity("М-01", "НЕИЗВЕСТНАЯ_ЗОНА")
        # WHY: Если classify-функция вернула неожиданную метку (рефакторинг,
        # новая зона), build_alignment() должен упасть с понятной ошибкой, а не
        # тихо засчитать неверную полярность. Это защита от рассинхрона между
        # mozart_signals.py и mozart_alignment.py.

    def test_unknown_label_for_contrarian_signal_raises(self):
        with pytest.raises(ValueError):
            signal_polarity("Н-01", "ЗАБЫТАЯ_ЗОНА")
        # WHY: Контрарианские сигналы особенно опасны при тихой ошибке:
        # неверная полярность инвертирует сигнал незаметно. ValueError обязателен.

    def test_empty_signal_id_raises(self):
        with pytest.raises(ValueError):
            signal_polarity("", "BULL")
        # WHY: Пустая строка как ID — признак незаполненного поля в оркестраторе.
        # Тихий возврат создаст пустой ключ в AglinmentResult.missing или score.

    def test_empty_label_raises(self):
        with pytest.raises(ValueError):
            signal_polarity("М-01", "")
        # WHY: Пустая метка = API вернул пустое значение или classify-функция
        # зафейлила молча. Принудительный ValueError заставляет оркестратор
        # обработать н/д явно (передать в missing, не в bullish/bearish).


# ===========================================================================
# НВ-03 | BTC Dominance — ротация ликвидности
# ===========================================================================

class TestNV03BtcDominanceTrend:
    """Контракт НВ-03: classify_btc_dominance_trend → ROTATION_ALTCOIN/NEUTRAL/ROTATION_BTC."""

    def test_rotation_altcoin_is_bullish(self):
        result = signal_polarity("НВ-03", "ROTATION_ALTCOIN")
        assert result == "BULLISH", (
            # WHY: снижение BTC.D = ликвидность идёт в альты — возвращение риск-аппетита.
            # Mozart (пост 10.05.2026): «рынок часто идёт по цепочке: низкая капа →
            # средняя → высокая» — рост альтов предшествует росту средней
            # капы (SOL/LINK/ADA), который предшествует росту BTC. BEARISH/NEUTRAL здесь
            # пропустили бы опережающий сигнал в build_alignment().
        )

    def test_neutral_is_neutral(self):
        result = signal_polarity("НВ-03", "NEUTRAL")
        assert result == "NEUTRAL", (
            # WHY: движение BTC.D в пределах шума — сигнал не активирован;
            # BULLISH здесь = ложный +1 в score при отсутствии направленной ротации.
        )

    def test_rotation_btc_is_neutral(self):
        result = signal_polarity("НВ-03", "ROTATION_BTC")
        assert result == "NEUTRAL", (
            # WHY (FORMALIZED): рост BTC.D амбивален: может означать BTC-сезон
            # (хорошо для BTC) или риск-офф (плохо для рынка). Mozart не формулирует
            # этот вектор явно; NEUTRAL исключает ложный +1 или -1 в score.
        )

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError):
            signal_polarity("НВ-03", "НЕИЗВЕСТНАЯ")
        # WHY: если classify_btc_dominance_trend добавит новую зону без
        # обновления таблицы — ValueError принудит разработчика
        # заметить рассинхрон, а не получить тихий NEUTRAL.

