"""
tests/test_mozart_alignment_build.py
=====================================
TDD — L3-2: build_alignment() + AlignmentResult
(docs/PLAN_MOZART_LEVEL3_SIGNAL_ALIGNMENT.md).

Контракт:
    build_alignment(signals: dict[str, str | None]) -> AlignmentResult

    signals: { signal_id: label | None }
        None или 'н/д' → missing (не влияет на score).
        Известные ID с известными метками → BULLISH/NEUTRAL/BEARISH.

    AlignmentResult.score    = len(bullish) - len(bearish)
    AlignmentResult.verdict:
        total_directional == 0          → 'NEUTRAL'
        score >= +2                     → 'BULLISH'
        score <= -2                     → 'BEARISH'
        score == +1 / -1 / 0 (но dir>0) → 'MIXED'
    AlignmentResult.contrarian_flags: ID контрарианских сигналов
        из bullish или bearish (т.е. активированных).

    ValueError если signals содержит неизвестный signal_id или метку
    (делегируется signal_polarity — контракт наследуется).

WHY missing ≠ NEUTRAL:
    н/д из-за 403/404 API — это отсутствие данных, а не нейтральный рынок.
    Включение в NEUTRAL завысило бы neutral-счётчик и могло сдвинуть verdict.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mozart_alignment import build_alignment, AlignmentResult


# ===========================================================================
# AlignmentResult — структура dataclass
# ===========================================================================

class TestAlignmentResultStructure:
    """Контракт: AlignmentResult имеет нужные поля с правильными типами."""

    def test_has_all_required_fields(self):
        result = build_alignment({"М-01": "BULL"})
        for field in ("bullish", "neutral", "bearish", "missing", "score", "verdict", "contrarian_flags"):
            assert hasattr(result, field), (
                # WHY: оркестратор обращается к result.bullish, result.score и т.д.
                # Отсутствие поля → AttributeError в print-блоке без подсказки.
                f"AlignmentResult не имеет поля '{field}'"
            )

    def test_lists_are_lists(self):
        result = build_alignment({"М-01": "BULL"})
        for field in ("bullish", "neutral", "bearish", "missing", "contrarian_flags"):
            assert isinstance(getattr(result, field), list), (
                # WHY: оркестратор делает len(result.bullish) и итерирует список.
                # Не-list (set, tuple) сломает форматирование join().
                f"Поле '{field}' должно быть list"
            )

    def test_score_is_int(self):
        result = build_alignment({"М-01": "BULL"})
        assert isinstance(result.score, int), (
            # WHY: оркестратор выводит f"+{result.score}" — float даст "+1.0".
            # verdict-логика использует >=2, <=−2 — с float работает, но тип
            # нарушает контракт.
        )

    def test_verdict_is_str(self):
        result = build_alignment({"М-01": "BULL"})
        assert isinstance(result.verdict, str), (
            # WHY: оркестратор выводит result.verdict в строку напрямую.
        )


# ===========================================================================
# Базовое распределение сигналов по спискам
# ===========================================================================

class TestSignalDistribution:
    """Контракт: каждый сигнал попадает ровно в один список."""

    def test_bullish_signal_goes_to_bullish_list(self):
        result = build_alignment({"М-01": "BULL"})
        assert "М-01" in result.bullish
        # WHY: М-01 BULL → BULLISH. Не в bullish → сигнал не учтён в score.

    def test_neutral_signal_goes_to_neutral_list(self):
        result = build_alignment({"М-01": "EARLY_BEAR"})
        assert "М-01" in result.neutral
        # WHY: М-01 EARLY_BEAR → NEUTRAL. Должен попасть в neutral, не в bearish.

    def test_bearish_signal_goes_to_bearish_list(self):
        result = build_alignment({"М-01": "MID_BEAR"})
        assert "М-01" in result.bearish
        # WHY: М-01 MID_BEAR → BEARISH. Должен уменьшать score, не нейтрализовать.

    def test_none_signal_goes_to_missing(self):
        result = build_alignment({"М-01": None})
        assert "М-01" in result.missing
        # WHY: None = API не вернул данные. Контракт: только в missing.

    def test_nd_string_goes_to_missing(self):
        result = build_alignment({"М-01": "н/д"})
        assert "М-01" in result.missing
        # WHY: 'н/д' — стандартный плейсхолдер оркестратора при 403/404.
        # Должен вести себя так же как None.

    def test_signal_is_in_exactly_one_list(self):
        result = build_alignment({"М-01": "BULL", "М-02": "RUBICON", "М-03": "BEAR"})
        all_ids = result.bullish + result.neutral + result.bearish + result.missing
        assert len(all_ids) == len(set(all_ids)), (
            # WHY: дублирование ID в двух списках означает двойной счёт в score.
            # Например М-01 в bullish И neutral → score завышен.
        )

    def test_none_not_in_bullish_neutral_bearish(self):
        result = build_alignment({"М-01": None})
        assert "М-01" not in result.bullish
        assert "М-01" not in result.neutral
        assert "М-01" not in result.bearish
        # WHY: н/д ≠ нейтральный рынок (архитектурный принцип плана).
        # Попадание в neutral исказит счётчик при 3–4 недоступных эндпоинтах.


# ===========================================================================
# Score — арифметика
# ===========================================================================

class TestScore:
    """Контракт: score = len(bullish) - len(bearish). Missing и neutral не влияют."""

    def test_score_all_bullish(self):
        result = build_alignment({"М-01": "BULL", "М-02": "BULL", "М-03": "BULL"})
        assert result.score == 3
        # WHY: 3 бычьих, 0 медвежьих → +3. Неправильная арифметика сдвинет verdict.

    def test_score_all_bearish(self):
        result = build_alignment({"М-01": "MID_BEAR", "М-02": "BEAR", "М-03": "BEAR"})
        assert result.score == -3
        # WHY: 0 бычьих, 3 медвежьих → −3.

    def test_score_mixed(self):
        result = build_alignment({
            "М-01": "BULL",
            "М-02": "BULL",
            "М-03": "BEAR",
        })
        assert result.score == 1
        # WHY: 2 − 1 = +1. Проверяет что вычитание, а не сложение.

    def test_neutral_does_not_affect_score(self):
        result_without = build_alignment({"М-01": "BULL", "М-02": "BEAR"})
        result_with = build_alignment({
            "М-01": "BULL", "М-02": "BEAR",
            "М-04": "NEUTRAL",  # нейтральный
        })
        assert result_with.score == result_without.score
        # WHY: NEUTRAL сигнал не должен изменять score. Иначе добавление
        # нейтрального сигнала неожиданно меняет verdict.

    def test_missing_does_not_affect_score(self):
        result_without = build_alignment({"М-01": "BULL", "М-02": "BEAR"})
        result_with = build_alignment({
            "М-01": "BULL", "М-02": "BEAR",
            "М-03": None,   # данные недоступны
        })
        assert result_with.score == result_without.score
        # WHY: ключевой архитектурный принцип (plan, раздел СТРУКТУРА).
        # н/д из-за 403 API ≠ нейтральный рынок. Missing не влияет на score.

    def test_score_zero_when_balanced(self):
        result = build_alignment({"М-01": "BULL", "М-02": "BEAR"})
        assert result.score == 0
        # WHY: 1 бычий − 1 медвежий = 0. Граничный случай для verdict MIXED.


# ===========================================================================
# Verdict — пороги
# ===========================================================================

class TestVerdict:
    """Контракт: verdict вычисляется строго по таблице из плана."""

    def test_verdict_neutral_when_no_directional(self):
        result = build_alignment({"М-01": "EARLY_BEAR"})  # → NEUTRAL
        assert result.verdict == "NEUTRAL"
        # WHY: total_directional == 0 → NEUTRAL. Все сигналы нейтральны — нет
        # основания давать BULLISH или BEARISH. MIXED здесь ложный вердикт.

    def test_verdict_neutral_when_all_missing(self):
        result = build_alignment({"М-01": None, "М-02": None})
        assert result.verdict == "NEUTRAL"
        # WHY: нет ни одного directional сигнала (all missing) → NEUTRAL.
        # MIXED предполагает противоречивые сигналы, а не отсутствие данных.

    def test_verdict_bullish_at_threshold(self):
        result = build_alignment({
            "М-01": "BULL", "М-02": "BULL",
            "М-03": "BEAR",
        })
        # score = +1, НЕ bullish verdict
        assert result.verdict == "MIXED"
        result2 = build_alignment({
            "М-01": "BULL", "М-02": "BULL",
        })
        # score = +2 → BULLISH
        assert result2.verdict == "BULLISH"
        # WHY: граница ±2 из плана. score=1 = перевес на 1 при нескольких
        # сигналах — статистический шум. score=2 = минимальный значимый перевес.

    def test_verdict_bearish_at_threshold(self):
        result = build_alignment({
            "М-01": "MID_BEAR", "М-02": "BEAR",
        })
        assert result.verdict == "BEARISH"
        # WHY: score = −2 → BEARISH (граница включительно).

    def test_verdict_mixed_score_plus_one(self):
        result = build_alignment({
            "М-01": "BULL",
            "М-02": "BEAR",
            "М-03": "BULL",
        })
        assert result.score == 1
        assert result.verdict == "MIXED"
        # WHY: перевес на 1 — недостаточно для BULLISH вердикта. Mozart:
        # противоречие должно быть явным (≥2). BULLISH здесь ложный сигнал.

    def test_verdict_mixed_score_zero_with_directional(self):
        result = build_alignment({
            "М-01": "BULL", "М-02": "BEAR",
        })
        assert result.score == 0
        assert result.verdict == "MIXED"
        # WHY: score=0 при наличии directional = противоречие сигналов.
        # NEUTRAL неверен (есть бычий И медвежий); MIXED — правильно.

    def test_verdict_bullish_ignores_neutral_and_missing(self):
        result = build_alignment({
            "М-01": "BULL", "М-02": "BULL",
            "М-04": "NEUTRAL",  # → NEUTRAL, не влияет (М-04 имеет зону NEUTRAL)
            "М-11": None,       # → missing, не влияет
        })
        assert result.verdict == "BULLISH"
        # WHY: score = 2 − 0 = +2. Neutral и missing не снижают score.
        # Правильный вердикт не должен ухудшаться от отсутствия данных.


# ===========================================================================
# Contrarian flags
# ===========================================================================

class TestContrarianFlags:
    """Контракт: contrarian_flags содержит ID активированных контрарианских сигналов."""

    def test_contrarian_signal_in_flags_when_activated(self):
        result = build_alignment({"Н-01": "EXTREME_OVERSOLD"})
        assert "Н-01" in result.contrarian_flags
        # WHY: Н-01 EXTREME_OVERSOLD — контрарианский. Оркестратор рядом
        # с сигналом должен вывести '* — контрарианский сигнал'.
        # Если не в flags — пометка не появится и читатель не поймёт,
        # почему 'страшное' название дало бычий сигнал.

    def test_non_contrarian_not_in_flags(self):
        result = build_alignment({"М-01": "BULL"})
        assert "М-01" not in result.contrarian_flags
        # WHY: М-01 — прямой сигнал, пометка не нужна. Лишняя * в выводе
        # оркестратора будет дезориентировать читателя.

    def test_contrarian_not_in_flags_when_neutral(self):
        result = build_alignment({"Н-01": "NEUTRAL"})
        assert "Н-01" not in result.contrarian_flags
        # WHY: контрарианский эффект проявляется только в экстремальной зоне.
        # Н-01 NEUTRAL = нет сигнала вообще → не помечать как контрарианский.

    def test_multiple_contrarian_flags(self):
        result = build_alignment({
            "Н-01": "OVERSOLD",
            "Н-02": "EXTREME",
            "М-10": "EXTREME",
        })
        assert "Н-01" in result.contrarian_flags
        assert "Н-02" in result.contrarian_flags
        assert "М-10" in result.contrarian_flags
        # WHY: одновременно несколько контрарианских — усиливает сигнал.
        # Все должны быть помечены для правильного вывода в оркестраторе.

    def test_bearish_contrarian_in_flags(self):
        result = build_alignment({"М-05": "EUPHORIA"})
        assert "М-05" in result.contrarian_flags
        # WHY: М-05 EUPHORIA → контрарианский BEARISH (не только бычьи
        # сигналы бывают контрарианскими). Пометка нужна в обоих направлениях.


# ===========================================================================
# Пустой и полный словарь сигналов
# ===========================================================================

class TestEdgeCases:

    def test_empty_signals_dict(self):
        result = build_alignment({})
        assert result.score == 0
        assert result.verdict == "NEUTRAL"
        assert result.bullish == []
        assert result.bearish == []
        # WHY: оркестратор может вызвать build_alignment({}) при полном
        # отказе API. Не должно быть исключений — должен вернуть пустой результат.

    def test_all_signals_present(self):
        """Все 14 ID из _POLARITY_TABLE — без исключений."""
        signals = {
            "М-01": "BULL",
            "М-02": "BULL",
            "М-03": "BEAR",
            "М-04": "NEUTRAL",
            "М-05": "POSITIVE",
            "М-06": "POSITIVE",
            "М-07+08": "ACCUMULATION",
            "М-09": "True",
            "М-10": "BELOW_2018",
            "М-11": "INFLOW",
            "М-12": "AGING",
            "МБ-03": "NEUTRAL",
            "Н-01": "NEUTRAL",
            "Н-02": "NORMAL",
        }
        result = build_alignment(signals)
        total = (len(result.bullish) + len(result.neutral)
                 + len(result.bearish) + len(result.missing))
        assert total == 14
        # WHY: каждый из 14 сигналов должен попасть ровно в один список.
        # total != 14 означает потерю или дублирование сигнала.

    def test_unknown_signal_id_raises(self):
        with pytest.raises(ValueError):
            build_alignment({"НЕСУЩЕСТВУЮЩИЙ": "BULL"})
        # WHY: build_alignment делегирует signal_polarity, которая кидает
        # ValueError при неизвестном ID. Контракт наследуется.

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError):
            build_alignment({"М-01": "НЕИЗВЕСТНАЯ"})
        # WHY: аналогично — неизвестная метка = рассинхрон classify-функций.
        # Тихий возврат NEUTRAL маскирует ошибку.
