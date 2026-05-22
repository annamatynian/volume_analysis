# tests/test_calculate_vwap_deviation.py
# CO-3 | TDD calculate_vwap_deviation
#
# Ретроактивный TDD: функция уже реализована в volume_density.py.
# Эти тесты фиксируют контракт docstring как регрессию.
#
# Реализация (volume_density.py):
#   rolling_pv  = (df['close'] * df['vol']).rolling(window).sum()
#   rolling_vol = df['vol'].rolling(window).sum()
#   vwap = rolling_pv / rolling_vol
#   return (df['close'] - vwap) / vwap * 100
#
# Контракт:
#   - Принимает: df (колонки 'close', 'vol'), window: int = 20
#   - Возвращает: pd.Series того же индекса, что df
#   - Первые (window-1) строк → NaN (rolling().sum() с min_periods=window)
#   - window=1 → deviation = 0.0 везде (VWAP_t = close_t при одной точке)
#   - Результат в ПРОЦЕНТАХ (× 100), не в долях
#
# Правила проекта применены:
#   ✓ WHY-комментарий к каждому assert (что сломается в production)
#   ✓ Запрет воспроизведения production-логики: нет Σ(close×vol)/Σ(vol) в тестах
#   ✓ Нейтральные плейсхолдеры: 1.0 / 3.0 / 5.0 / 10.0 (не BTC-реалистичные)
#   ✓ Граничные значения — отдельные классы с явным WHY
#   ✓ Пороги из MOZART_CONFIG: vwap_window в конфиге отсутствует
#     (window — параметр функции, не signal-порог); _TEST_WINDOW = 3 — тестовый

import pytest
import pandas as pd
import numpy as np

from volume_density import calculate_vwap_deviation

try:
    from mozart_config import MOZART_CONFIG
except ImportError:
    MOZART_CONFIG = {}

# Малое тестовое окно: делает датасеты компактными (5-8 строк).
# НЕ из MOZART_CONFIG — window не signal-порог, а параметр функции.
# Этот константа управляет только тестами, не production.
_TEST_WINDOW = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(closes, vols=None, index=None):
    """Строит минимальный DataFrame с колонками 'close' и 'vol'.

    vols=None → нейтральный 1.0 для каждой строки (равные веса).
    index=None → RangeIndex(n).
    """
    n = len(closes)
    if vols is None:
        vols = [1.0] * n
    if index is None:
        index = pd.RangeIndex(n)
    return pd.DataFrame({"close": list(closes), "vol": list(vols)}, index=index)


# ---------------------------------------------------------------------------
# 1. Контракт типа возвращаемого значения
# ---------------------------------------------------------------------------

class TestReturnTypeContract:
    """Функция обязана возвращать pd.Series — ничто другое."""

    def test_result_is_pandas_series(self):
        """
        Возвращается pd.Series, а не ndarray или скаляр.

        WHY: весь downstream (pd.concat с OFI/flow-toxicity, .iloc-слайсинг,
        fillna, merge) работает через pd.Series с именованным индексом.
        np.ndarray не имеет индекса — молчаливый NaN-drift при join.
        """
        df = _make_df([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_vwap_deviation(df, window=_TEST_WINDOW)
        # WHY: isinstance(result, np.ndarray) пройдёт if-guard, но сломает merge
        assert isinstance(result, pd.Series), (
            f"Ожидался pd.Series, получен {type(result).__name__}"
        )

    def test_result_is_not_scalar_float(self):
        """
        Возвращается Series, а не одно число.

        WHY: скалярный float нарушит .loc-alignment в feature matrix —
        broadcast присвоит одно значение всем строкам вместо поточечного.
        """
        df = _make_df([1.0, 2.0, 3.0])
        result = calculate_vwap_deviation(df, window=_TEST_WINDOW)
        # WHY: скаляр сломает pd.concat с другими признаками
        assert not isinstance(result, (float, int, np.floating))

    def test_result_is_not_ndarray(self):
        """
        Возвращается pd.Series, не np.ndarray.

        WHY: ndarray не несёт временного индекса — при pd.concat
        получим позиционное выравнивание вместо label-based,
        сигнал сместится на произвольное число строк.
        """
        df = _make_df([1.0, 2.0, 3.0, 4.0])
        result = calculate_vwap_deviation(df, window=_TEST_WINDOW)
        # WHY: ndarray не содержит временного индекса — сломает join
        assert not isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# 2. Контракт индекса
# ---------------------------------------------------------------------------

class TestIndexContract:
    """Индекс результата обязан в точности совпадать с df.index."""

    def test_result_index_equals_df_rangeindex(self):
        """
        При стандартном RangeIndex индекс результата совпадает с df.index.

        WHY: несовпадение RangeIndex вызовет NaN-drift при pd.concat
        с другими feature-рядами (OFI, flow toxicity).
        """
        df = _make_df([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_vwap_deviation(df, window=_TEST_WINDOW)
        # WHY: несовпадение индекса → NaN в feature matrix после concat
        pd.testing.assert_index_equal(result.index, df.index)

    def test_result_index_equals_df_datetime_index(self):
        """
        При DatetimeIndex индекс результата совпадает с df.index.

        WHY: временной индекс критичен для join с OHLCV.
        Сдвиг даже на одну строку смещает момент сигнала входа в позицию.
        """
        idx = pd.date_range("2024-01-01", periods=6, freq="h")
        df = _make_df([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
        result = calculate_vwap_deviation(df, window=_TEST_WINDOW)
        # WHY: временной сдвиг → сигнал торгуется по цене другого момента
        pd.testing.assert_index_equal(result.index, df.index)

    def test_result_length_equals_input_length(self):
        """
        Длина результата равна длине входного DataFrame.

        WHY: усечённый вывод (len < n) сдвинет все downstream-индексы
        и создаст silent misalignment в feature matrix.
        """
        n = 8
        df = _make_df(list(range(1, n + 1)))
        result = calculate_vwap_deviation(df, window=_TEST_WINDOW)
        # WHY: любое усечение сломает выравнивание по строкам в feature matrix
        assert len(result) == n, (
            f"Длина результата {len(result)} ≠ длина входа {n}"
        )


# ---------------------------------------------------------------------------
# 3. Граничный тест: первые (window-1) строк → NaN
# ---------------------------------------------------------------------------

class TestNaNBoundary:
    """Неполное окно обязано давать NaN, а не 0 или экстраполированные числа."""

    def test_first_window_minus_1_rows_are_nan(self):
        """
        Первые (window-1) строк — NaN (окно неполное).

        WHY: нулевое заполнение неполного окна создаст ложный сигнал
        deviation=0.0 вместо «нет данных». Downstream z-score-нормализация
        включит эти строки в расчёт среднего и сожмёт реальные сигналы.
        """
        window = _TEST_WINDOW   # window=3 → строки 0,1 должны быть NaN
        df = _make_df([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_vwap_deviation(df, window=window)
        nan_zone = result.iloc[: window - 1]
        # WHY: NaN вместо 0.0 → downstream знает «данных нет», не «отклонение нулевое»
        assert nan_zone.isna().all(), (
            f"Строки 0..{window - 2} обязаны быть NaN; получено: {nan_zone.tolist()}"
        )

    def test_nan_zone_has_exact_size_window_minus_1(self):
        """
        Количество NaN строк равно ровно (window-1), не больше и не меньше.

        WHY: лишние NaN (nan_count > window-1) — off-by-one в min_periods,
        потеря первого валидного VWAP-сигнала.
        Меньше NaN (nan_count < window-1) — использование неполного окна,
        VWAP вычислен по меньшему числу точек, чем заявлено.
        """
        window = _TEST_WINDOW
        df = _make_df([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_vwap_deviation(df, window=window)
        nan_count = result.isna().sum()
        # WHY: точное количество NaN — контракт min_periods=window
        assert nan_count == window - 1, (
            f"Ожидалось {window - 1} NaN, получено {nan_count}"
        )

    def test_first_complete_window_row_is_not_nan(self):
        """
        Строка с индексом (window-1) — первое полное окно → не NaN.

        WHY: если первое полное окно возвращает NaN, весь ранний период
        отсутствует в сигнале и модель не видит момент первого VWAP.
        """
        window = _TEST_WINDOW   # индекс window-1 = 2 — первое полное окно
        df = _make_df([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_vwap_deviation(df, window=window)
        first_valid = result.iloc[window - 1]
        # WHY: NaN на первом полном окне → потеря начала сигнала без предупреждения
        assert not pd.isna(first_valid), (
            f"Строка {window - 1} (первое полное окно) не должна быть NaN; "
            f"получено {first_valid!r}"
        )

    def test_no_unexpected_nans_in_valid_zone(self):
        """
        В валидной зоне (от строки window-1 до конца) нет NaN.

        WHY: дыры внутри валидной зоны сломают rolling downstream-агрегации
        и молча создадут NaN-каскад в feature matrix.
        """
        window = _TEST_WINDOW
        df = _make_df([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = calculate_vwap_deviation(df, window=window)
        valid_zone = result.iloc[window - 1:]
        # WHY: NaN в середине валидной зоны → downstream rolling пропустит точки
        assert valid_zone.notna().all(), (
            f"Валидная зона содержит неожиданные NaN: {valid_zone.tolist()}"
        )


# ---------------------------------------------------------------------------
# 4. Граничный тест: равномерная цена → deviation = 0.0
# ---------------------------------------------------------------------------

class TestUniformPriceDeviation:
    """При постоянной цене VWAP ≡ close → deviation = 0 в любой точке."""

    def test_uniform_price_uniform_volume_gives_zero_deviation(self):
        """
        Постоянная цена + постоянный объём → deviation = 0.0 везде в валидной зоне.

        WHY: при постоянной цене VWAP = close по определению.
        Ненулевое отклонение означает численную ошибку в формуле —
        ложные сигналы на флэтовом рынке.
        """
        window = _TEST_WINDOW
        price = 5.0   # нейтральный плейсхолдер
        df = _make_df([price] * 7, vols=[2.0] * 7)
        result = calculate_vwap_deviation(df, window=window)
        valid = result.iloc[window - 1:]
        # WHY: VWAP = price при постоянной цене → (close - vwap) / vwap * 100 = 0
        assert (valid == 0.0).all(), (
            f"Постоянная цена → deviation=0.0; получено: {valid.tolist()}"
        )

    def test_uniform_price_varying_volume_gives_zero_deviation(self):
        """
        Постоянная цена + ПЕРЕМЕННЫЙ объём → deviation ≈ 0.0 везде.

        WHY: объём — лишь веса; Σ(price×vol)/Σ(vol) = price при любых ненулевых весах.
        Ненулевое отклонение указывает на ошибку нормализации весов — неверная
        weighted-average при изменяющемся vol.
        """
        window = _TEST_WINDOW
        price = 3.0
        vols  = [1.0, 5.0, 2.0, 8.0, 3.0, 1.0, 4.0]
        df = _make_df([price] * 7, vols=vols)
        result = calculate_vwap_deviation(df, window=window)
        valid = result.iloc[window - 1:]
        # WHY: переменный объём не меняет VWAP при постоянной цене
        assert (valid.abs() < 1e-10).all(), (
            f"Постоянная цена + переменный объём → deviation≈0; "
            f"получено: {valid.tolist()}"
        )


# ---------------------------------------------------------------------------
# 5. Граничные тесты: знак отклонения
# ---------------------------------------------------------------------------

class TestDeviationSign:
    """Знак deviation кодирует направление цены относительно VWAP."""

    def test_price_spike_above_history_gives_positive_deviation(self):
        """
        Цена долго низко, затем резкий скачок → deviation > 0.

        WHY: положительное отклонение = close торгуется с премией к VWAP
        (momentum-бычий сигнал). Инвертированный знак перевернёт long/short
        и приведёт к входу против тренда.
        """
        window = _TEST_WINDOW
        # Долго 1.0, затем 10.0 → VWAP в последнем окне < 10.0
        closes = [1.0, 1.0, 1.0, 1.0, 10.0]
        df = _make_df(closes)
        result = calculate_vwap_deviation(df, window=window)
        last = result.iloc[-1]
        # WHY: скачок цены вверх относительно истории окна → deviation > 0
        assert last > 0, (
            f"Скачок цены вверх должен дать deviation > 0; получено {last:.4f}"
        )

    def test_price_drop_below_history_gives_negative_deviation(self):
        """
        Цена долго высоко, затем резкий обвал → deviation < 0.

        WHY: отрицательное отклонение = дисконт к VWAP (медвежий сигнал).
        Ошибка знака → ложный short на oversold-активе, убыток.
        """
        window = _TEST_WINDOW
        # Долго 10.0, затем 1.0 → VWAP в последнем окне > 1.0
        closes = [10.0, 10.0, 10.0, 10.0, 1.0]
        df = _make_df(closes)
        result = calculate_vwap_deviation(df, window=window)
        last = result.iloc[-1]
        # WHY: обвал цены ниже истории окна → deviation < 0
        assert last < 0, (
            f"Обвал цены должен дать deviation < 0; получено {last:.4f}"
        )


# ---------------------------------------------------------------------------
# 6. Граничный тест: window = 1
# ---------------------------------------------------------------------------

class TestWindowOne:
    """При window=1 VWAP_t = close_t → deviation ≡ 0.0 без NaN-зоны."""

    def test_window_1_all_rows_zero_deviation(self):
        """
        window=1 + любые разные цены → deviation = 0.0 везде.

        WHY: при window=1 rolling(1).sum() = сама точка → VWAP_t = close_t.
        Ненулевое значение — off-by-one в скользящем окне или неверный min_periods.
        Тест намеренно использует РАЗНЫЕ цены, чтобы не перепутать с uniform-тестом.
        """
        closes = [1.0, 5.0, 2.0, 8.0, 3.0]
        df = _make_df(closes, vols=[1.0, 2.0, 1.0, 3.0, 1.0])
        result = calculate_vwap_deviation(df, window=1)
        # WHY: window=1 → VWAP=close → deviation=0 для любых цен
        assert (result == 0.0).all(), (
            f"window=1 → deviation=0.0 везде; получено: {result.tolist()}"
        )

    def test_window_1_produces_no_nan_rows(self):
        """
        window=1 → NaN-зона нулевой длины (window-1 = 0 строк).

        WHY: любой NaN при window=1 означает что min_periods задан неверно (> 1),
        и первые строки выпадут из feature matrix без предупреждения.
        """
        df = _make_df([1.0, 2.0, 3.0, 4.0])
        result = calculate_vwap_deviation(df, window=1)
        # WHY: window-1=0 → NaN-зоны нет; NaN = ошибка min_periods
        assert result.notna().all(), (
            f"window=1: NaN-зоны нет; получено: {result.tolist()}"
        )

    def test_window_1_result_length_equals_input(self):
        """
        window=1 → все n строк в результате.

        WHY: если реализация всегда удаляет window-1 строк даже при window=1,
        вывод будет усечён — выравнивание с feature matrix сломается.
        """
        n = 5
        df = _make_df(list(range(1, n + 1)))
        result = calculate_vwap_deviation(df, window=1)
        # WHY: window-1=0 → нет усечения, len(result) == n
        assert len(result) == n


# ---------------------------------------------------------------------------
# 7. Контракт масштаба: deviation в процентах, не в долях
# ---------------------------------------------------------------------------

class TestDeviationScale:
    """Результат должен быть в процентах (× 100), а не в долях."""

    def test_ten_percent_spike_gives_deviation_above_one(self):
        """
        ~10%-й скачок цены → deviation > 1.0 (в процентах), не 0.10 (в долях).

        WHY: downstream-пороги (например, «отклонение > 2%») сравниваются
        с числами в процентах. Если функция возвращает доли, порог 2
        никогда не будет достигнут при реальных движениях рынка < 100%
        → все сигналы пропадут молча.
        """
        window = _TEST_WINDOW
        # Цена растёт примерно на 10% от фонового уровня окна
        closes = [1.0, 1.0, 1.0, 1.0, 1.1]
        df = _make_df(closes)
        result = calculate_vwap_deviation(df, window=window)
        last = result.iloc[-1]
        # WHY: 10% в % → deviation > 1.0; в долях → 0.10 → порог 2% никогда не пройден
        assert last > 1.0, (
            f"~10%-й скачок → deviation > 1.0 (в %); получено {last:.4f}"
        )

    def test_neutral_market_gives_zero_not_one(self):
        """
        Нейтральный рынок (постоянная цена) → deviation = 0.0, не 1.0.

        WHY: если ошибочно вернуть (close/VWAP) без вычитания 1 и умножения на 100,
        результатом будет 1.0 вместо 0.0 → нейтральный рынок выглядит
        как «отклонение +100%».
        """
        window = _TEST_WINDOW
        closes = [2.0, 2.0, 2.0, 2.0, 2.0]
        df = _make_df(closes)
        result = calculate_vwap_deviation(df, window=window)
        valid = result.iloc[window - 1:]
        # WHY: (close - vwap) / vwap * 100 = 0, не close/vwap = 1
        assert (valid == 0.0).all(), (
            f"Нейтральный рынок → deviation=0.0, не 1.0; получено: {valid.tolist()}"
        )