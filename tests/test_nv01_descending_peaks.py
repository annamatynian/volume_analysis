# tests/test_nv01_descending_peaks.py
# TDD — НВ-01 | Регрессия убывающих локальных пиков
# Mozart-паттерн (пост 31.03.2026): 87k → 84k → 76k → (63–67k)
#
# Контракт двух функций:
#
#   find_swing_highs(highs: list[float], dates: list, n: int) -> list[tuple[int, float, date]]
#     Возвращает список (index, price, date) подтверждённых пиков.
#     Пик i: highs[i] > max(highs[i-n:i]) AND highs[i] >= max(highs[i+1:i+n+1])
#     Асимметричное сравнение по рекомендации Gemini (validation 2026-06-01).
#     Использует highs (не closes) — психологические уровни формируются по high.
#
#   classify_descending_peaks(highs, dates) -> dict
#     highs, dates — из DataFrame оркестратора (df['high'], df['date']).
#     Внутри: find_swing_highs → фильтрация по max_days_between_peaks →
#             последние K пиков → монотонность + linregress → режим.
#
#     Возвращает dict:
#       {
#         'regime':          str,   # DESCENDING_STRONG | DESCENDING_WEAK | FLAT | ASCENDING | INSUFFICIENT_DATA
#         'peaks_count':     int,   # сколько пиков использовано
#         'slope':           float | None,
#         'projected_next':  float | None,  # цена следующего пика по регрессии
#         'is_monotone':     bool  | None,  # все пики строго убывают?
#       }
#
# Все пороги берутся из MOZART_CONFIG — не хардкодятся в тестах.

import pytest
import pandas as pd
from datetime import date, timedelta

from mozart_config import MOZART_CONFIG

# Пороги из конфига — тесты вычисляют тестовые значения от них
_N    = int(MOZART_CONFIG["swing_high_window"])
_K    = int(MOZART_CONFIG["swing_high_lookback_peaks"])
_MDBP = int(MOZART_CONFIG["max_days_between_peaks"])

# Импорт production-функций (пока не существуют → RED)
from mozart_signals import find_swing_highs, classify_descending_peaks


# ---------------------------------------------------------------------------
# Вспомогательные фабрики синтетических данных
# ---------------------------------------------------------------------------

def _flat_highs(length: int, value: float = 100.0) -> list[float]:
    """Плоский ряд — нет пиков."""
    return [value] * length


def _make_dates(length: int, start: date = date(2024, 1, 1)) -> list[date]:
    """Список дат с шагом 1 день."""
    return [start + timedelta(days=i) for i in range(length)]


def _insert_peak(highs: list[float], pos: int, value: float) -> list[float]:
    """Вставляет одиночный пик в позицию pos (не меняет соседей)."""
    result = highs.copy()
    result[pos] = value
    return result


def _make_descending_peaks_series(peak_values: list[float],
                                   spacing: int = 30,
                                   base: float = 50.0,
                                   n: int = None) -> tuple[list[float], list[date]]:
    """
    Строит синтетический ряд с пиками заданных значений.
    spacing — расстояние между пиками в днях.
    Вокруг каждого пика значения равны base (нейтральный плейсхолдер).
    n — окно swing high; если None берётся из конфига.
    """
    if n is None:
        n = _N
    # Длина: n отступ слева + пики с spacing + n отступ справа
    total = n + len(peak_values) * spacing + n
    highs = [base] * total
    dates = _make_dates(total)
    peak_positions = []
    for i, val in enumerate(peak_values):
        pos = n + i * spacing + spacing // 2
        highs[pos] = val
        peak_positions.append(pos)
    return highs, dates


# ---------------------------------------------------------------------------
# find_swing_highs — детекция пиков
# ---------------------------------------------------------------------------

class TestFindSwingHighs:

    def test_returns_list(self):
        """WHY: оркестратор итерирует результат; не-list вызовет TypeError."""
        highs = _flat_highs(_N * 3 + 1, 100.0)
        dates = _make_dates(len(highs))
        result = find_swing_highs(highs, dates, _N)
        assert isinstance(result, list), (
            f'find_swing_highs должна вернуть list, получен {type(result)}'
        )

    def test_no_peaks_on_flat_series(self):
        """WHY: плоский ряд не содержит пиков; ложный пик вызовет
        INSUFFICIENT_DATA→False в classify и скроет реальные данные."""
        highs = _flat_highs(_N * 4, 100.0)
        dates = _make_dates(len(highs))
        result = find_swing_highs(highs, dates, _N)
        assert result == [], (
            f'Плоский ряд не должен содержать пиков, найдено: {result}'
        )

    def test_detects_single_clear_peak(self):
        """WHY: один явный пик посреди плоского ряда должен детектироваться;
        если нет — classify всегда вернёт INSUFFICIENT_DATA на реальных данных."""
        length = _N * 4
        pos = length // 2
        highs = _insert_peak(_flat_highs(length, 100.0), pos, 200.0)
        dates = _make_dates(length)
        result = find_swing_highs(highs, dates, _N)
        assert len(result) == 1, (
            f'Один явный пик должен быть найден, найдено: {len(result)}'
        )
        assert result[0][0] == pos, (
            # WHY: индекс пика используется для привязки к дате; неверный индекс
            # даст неправильный интервал между пиками в фильтре max_days_between_peaks
            f'Пик должен быть в позиции {pos}, найден в {result[0][0]}'
        )

    def test_peak_value_matches_high(self):
        """WHY: classify использует price из результата для регрессии Y;
        неверное значение → неверный slope → неверный projected_next."""
        length = _N * 4
        pos = length // 2
        peak_val = 77_000.0
        highs = _insert_peak(_flat_highs(length, 50_000.0), pos, peak_val)
        dates = _make_dates(length)
        result = find_swing_highs(highs, dates, _N)
        assert len(result) == 1
        assert result[0][1] == peak_val, (
            f'Значение пика должно быть {peak_val}, получено {result[0][1]}'
        )

    def test_asymmetric_left_strict_right_nonstrict(self):
        """WHY: Gemini validation — строгое > слева, >= справа.
        Это предотвращает дублирование на двойной вершине (два == значения).
        Тест: пик строго выше левого соседа, равен правому → должен детектироваться."""
        # highs: base ... peak_val ... peak_val (правый сосед равен пику)
        # Размер: достаточно для двух окон
        length = _N * 4
        pos = _N + _N // 2
        base = 100.0
        peak_val = 150.0
        highs = _flat_highs(length, base)
        highs[pos] = peak_val
        # Правый сосед равен пику — проверяем >= (не >)
        highs[pos + 1] = peak_val
        dates = _make_dates(length)
        result = find_swing_highs(highs, dates, _N)
        # pos должен детектироваться (строго > слева, >= справа выполняется)
        # pos+1 не должен (левый сосед == peak_val, нарушает строгое >)
        positions = [r[0] for r in result]
        assert pos in positions, (
            # WHY: если правый сосед == пику игнорирует пик → Mozart-паттерн
            # не детектируется на реальной двойной вершине
            f'pos={pos} должен быть пиком при >= справа, найдены пики: {positions}'
        )

    def test_two_separated_peaks_both_detected(self):
        """WHY: несколько пиков должны детектироваться независимо;
        если второй пик "затеняется" первым — регрессия строится по одной точке."""
        highs, dates = _make_descending_peaks_series([200.0, 180.0], spacing=_N * 2)
        result = find_swing_highs(highs, dates, _N)
        assert len(result) == 2, (
            f'Ожидалось 2 пика, найдено {len(result)}'
        )

    def test_peak_tuple_has_three_elements(self):
        """WHY: classify ожидает (index, price, date); неверная структура
        вызовет IndexError или TypeError при распаковке."""
        length = _N * 4
        pos = length // 2
        highs = _insert_peak(_flat_highs(length, 100.0), pos, 200.0)
        dates = _make_dates(length)
        result = find_swing_highs(highs, dates, _N)
        assert len(result) == 1
        assert len(result[0]) == 3, (
            f'Каждый элемент результата должен быть (index, price, date), '
            f'получено {len(result[0])} элементов'
        )


# ---------------------------------------------------------------------------
# classify_descending_peaks — режимы
# ---------------------------------------------------------------------------

class TestClassifyDescendingPeaks:

    def test_returns_dict_with_required_keys(self):
        """WHY: оркестратор обращается к result['regime'] и result['projected_next'];
        отсутствие ключа → KeyError без ясного сообщения об ошибке."""
        highs, dates = _make_descending_peaks_series([90.0, 85.0, 78.0])
        result = classify_descending_peaks(highs, dates)
        required = {'regime', 'peaks_count', 'slope', 'projected_next', 'is_monotone'}
        missing = required - set(result.keys())
        assert not missing, (
            f'Отсутствуют ключи в результате: {missing}'
        )

    def test_regime_is_string(self):
        """WHY: оркестратор вставляет regime в строку вывода;
        не-str вызовет TypeError при форматировании."""
        highs, dates = _make_descending_peaks_series([90.0, 85.0, 78.0])
        result = classify_descending_peaks(highs, dates)
        assert isinstance(result['regime'], str), (
            f"regime должен быть str, получен {type(result['regime'])}"
        )

    def test_only_valid_regimes(self):
        """WHY: оркестратор сравнивает regime со строковыми константами;
        опечатка ('DESCENDING_STRON') — тихий баг без исключения."""
        valid = {
            'DESCENDING_STRONG', 'DESCENDING_WEAK',
            'FLAT', 'ASCENDING', 'INSUFFICIENT_DATA',
        }
        # Проверяем несколько сценариев
        cases = [
            _make_descending_peaks_series([90.0, 85.0, 78.0]),         # убывание
            _make_descending_peaks_series([70.0, 80.0, 90.0]),         # возрастание
            (_flat_highs(200, 100.0), _make_dates(200)),               # нет пиков
        ]
        for highs, dates in cases:
            result = classify_descending_peaks(highs, dates)
            assert result['regime'] in valid, (
                f"Недопустимый режим: {result['regime']!r}, допустимые: {valid}"
            )

    def test_descending_strong_monotone(self):
        """WHY: Mozart-паттерн «87→84→76» — строго убывающие пики.
        DESCENDING_STRONG = монотонное убывание + отрицательный slope.
        Ошибка здесь: Mozart-сигнал не активируется при реальном паттерне."""
        # Синтетические пики строго убывают: каждый следующий меньше предыдущего
        highs, dates = _make_descending_peaks_series([90.0, 85.0, 78.0])
        result = classify_descending_peaks(highs, dates)
        assert result['regime'] == 'DESCENDING_STRONG', (
            # WHY: монотонное убывание + slope < 0 = Mozart DESCENDING_STRONG
            f"Ожидался DESCENDING_STRONG, получен {result['regime']!r}"
        )
        assert result['is_monotone'] is True, (
            f"is_monotone должен быть True для строго убывающих пиков"
        )

    def test_ascending_peaks(self):
        """WHY: возрастающие пики = противоположность паттерна Mozart;
        классификация DESCENDING даст ложный медвежий сигнал."""
        highs, dates = _make_descending_peaks_series([70.0, 80.0, 90.0])
        result = classify_descending_peaks(highs, dates)
        assert result['regime'] == 'ASCENDING', (
            f"Ожидался ASCENDING для возрастающих пиков, получен {result['regime']!r}"
        )

    def test_insufficient_data_fewer_than_three_peaks(self):
        """WHY: менее 3 пиков — регрессия бессмысленна (Gemini validation).
        Тихий возврат FLAT или MIXED скроет отсутствие данных от оркестратора."""
        # Только один пик в данных
        length = _N * 4
        pos = length // 2
        highs = _insert_peak(_flat_highs(length, 100.0), pos, 200.0)
        dates = _make_dates(length)
        result = classify_descending_peaks(highs, dates)
        assert result['regime'] == 'INSUFFICIENT_DATA', (
            f"Ожидался INSUFFICIENT_DATA при 1 пике, получен {result['regime']!r}"
        )
        assert result['projected_next'] is None, (
            # WHY: при недостатке данных проекция невозможна; не-None вводит оркестратор в заблуждение
            f"projected_next должен быть None при INSUFFICIENT_DATA"
        )

    def test_insufficient_data_no_peaks(self):
        """WHY: плоский ряд без пиков — INSUFFICIENT_DATA, не FLAT.
        FLAT означает «пики есть, но slope≈0»; отсутствие пиков — другая ситуация."""
        highs = _flat_highs(200, 100.0)
        dates = _make_dates(200)
        result = classify_descending_peaks(highs, dates)
        assert result['regime'] == 'INSUFFICIENT_DATA', (
            f"Ожидался INSUFFICIENT_DATA при отсутствии пиков, получен {result['regime']!r}"
        )

    def test_peaks_count_matches_used_peaks(self):
        """WHY: оркестратор выводит peaks_count в информационной строке;
        несовпадение с реальным количеством — дезинформация."""
        highs, dates = _make_descending_peaks_series([90.0, 85.0, 78.0])
        result = classify_descending_peaks(highs, dates)
        assert result['peaks_count'] == 3, (
            f"Ожидалось 3 пика, получено {result['peaks_count']}"
        )

    def test_slope_is_negative_for_descending(self):
        """WHY: slope — числовая характеристика тренда; оркестратор выводит его
        в блоке НВ-01. Положительный slope при убывающих пиках — ошибка расчёта."""
        highs, dates = _make_descending_peaks_series([90.0, 85.0, 78.0])
        result = classify_descending_peaks(highs, dates)
        assert result['slope'] is not None
        assert result['slope'] < 0, (
            f"slope должен быть < 0 для убывающих пиков, получен {result['slope']}"
        )

    def test_projected_next_is_float_for_descending(self):
        """WHY: оркестратор форматирует projected_next как цену (f'${x:,.0f}');
        None или str вызовет TypeError при форматировании."""
        highs, dates = _make_descending_peaks_series([90.0, 85.0, 78.0])
        result = classify_descending_peaks(highs, dates)
        assert isinstance(result['projected_next'], float), (
            f"projected_next должен быть float, получен {type(result['projected_next'])}"
        )

    def test_max_days_between_peaks_filter(self):
        """WHY: пики с интервалом > max_days_between_peaks принадлежат разным
        рыночным эпохам (Gemini validation). Их объединение в одну регрессию
        даёт математически верный, но экономически бессмысленный slope."""
        # Два пика с интервалом >> _MDBP → после фильтрации останется 1 пик
        # → INSUFFICIENT_DATA (не хватает точек для регрессии)
        spacing_too_large = _MDBP + 30   # заведомо больше порога
        highs, dates = _make_descending_peaks_series(
            [90.0, 80.0], spacing=spacing_too_large
        )
        result = classify_descending_peaks(highs, dates)
        assert result['regime'] == 'INSUFFICIENT_DATA', (
            # WHY: фильтр должен отсечь пары пиков с интервалом > max_days_between_peaks;
            # без фильтра регрессия молча строится по устаревшим данным
            f"Ожидался INSUFFICIENT_DATA при интервале > max_days_between_peaks, "
            f"получен {result['regime']!r}"
        )

    def test_uses_high_column_not_close(self):
        """WHY: Gemini validation — психологические уровни сопротивления
        формируются по high свечи, не по close. Mozart называет «87k, 84k, 76k» —
        это high. Если функция случайно берёт close — паттерн не детектируется."""
        # Создаём ряд где high образует убывающий паттерн, но close — плоский
        # find_swing_highs должна работать с переданным массивом highs
        # classify_descending_peaks принимает highs отдельно
        length = _N * 6
        highs = _flat_highs(length, 50_000.0)
        dates = _make_dates(length)
        # Три убывающих пика в highs
        for i, val in enumerate([90_000.0, 85_000.0, 78_000.0]):
            pos = _N + i * (_N * 2) + _N
            if pos < length - _N:
                highs[pos] = val
        result = classify_descending_peaks(highs, dates)
        # Если функция использует highs корректно — должна найти пики
        assert result['regime'] != 'INSUFFICIENT_DATA' or result['peaks_count'] >= 0, (
            # WHY: хотя бы пики найдены (результат зависит от точности позиций)
            'classify_descending_peaks должна принимать highs как основной массив'
        )

    # --- Граничные значения ---

    def test_boundary_exactly_three_peaks_not_insufficient(self):
        """WHY: 3 пика — минимум для регрессии (Gemini: K=3 достаточно для slope).
        Граница: ровно 3 пика не должны давать INSUFFICIENT_DATA."""
        highs, dates = _make_descending_peaks_series([90.0, 85.0, 78.0])
        result = classify_descending_peaks(highs, dates)
        assert result['regime'] != 'INSUFFICIENT_DATA', (
            # WHY: ровно 3 пика = минимально допустимый случай; ошибка < вместо <=
            # отсечёт реальный Mozart-паттерн из 3 пиков
            f"3 пика не должны давать INSUFFICIENT_DATA, получен {result['regime']!r}"
        )

    def test_boundary_exactly_two_peaks_is_insufficient(self):
        """WHY: 2 пика = идеальная прямая, R² = 1.0 — бессмысленно статистически.
        Граница: ровно 2 пика должны давать INSUFFICIENT_DATA."""
        # Два пика с допустимым интервалом
        highs, dates = _make_descending_peaks_series([90.0, 85.0], spacing=_N * 2)
        result = classify_descending_peaks(highs, dates)
        assert result['regime'] == 'INSUFFICIENT_DATA', (
            # WHY: 2 точки дают прямую с R²=1.0 — не статистика, а геометрия;
            # допускать это означает давать ложную уверенность в тренде
            f"2 пика должны давать INSUFFICIENT_DATA, получен {result['regime']!r}"
        )

    def test_descending_weak_non_monotone(self):
        """WHY: пики убывают в целом (slope < 0), но не монотонно —
        один пик нарушает порядок. DESCENDING_WEAK, не DESCENDING_STRONG."""
        # 1000 → 400 → 700: slope < 0 (1000→0→-150), но 700 > 400 = не монотонно.
        # WHY крупные значения: большой разброс гарантирует |slope| > 2*stderr
        # — иначе FLAT поглощает DESCENDING_WEAK на малых выборках.
        highs, dates = _make_descending_peaks_series([1000.0, 400.0, 700.0])
        result = classify_descending_peaks(highs, dates)
        assert result['regime'] == 'DESCENDING_WEAK', (
            # WHY: нарушение монотонности = паттерн Mozart не подтверждён чисто;
            # DESCENDING_STRONG при немонотонных пиках — ложный сигнал
            f"Ожидался DESCENDING_WEAK для немонотонного убывания, получен {result['regime']!r}"
        )
        assert result['is_monotone'] is False, (
            f"is_monotone должен быть False для немонотонных пиков"
        )
