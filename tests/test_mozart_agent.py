# tests/test_mozart_agent.py
# TDD для mozart_agent.py
# Тестируем только детерминированную логику — НЕ LLM-ответы

import pytest
from pathlib import Path
import tempfile
import os


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

MINIMAL_LOG = """\
[RUNNING DYNAMIC INSTITUTIONAL AUDIT]
Цена сейчас: $64,148

[SIGNAL ALIGNMENT]
==================================================================
  Бычьих   :  3  М-03 М-05* М-07+08
  Нейтр.   :  6  М-02 М-09 М-12 Н-01 Н-02 М-15
  Медвежьих:  7  М-01 М-04 М-06 М-10* М-11 НВ-01 НВ-02
  Н/Д      :  3  МБ-03 НВ-03 МБ-05
  Счёт     : -4
  Вердикт  : 🔴  BEARISH
==================================================================

[FINAL VERDICT]
==================================================================
Текущая цена: $64,148  |  Отклонение от POC: -6.6%
Рыночный режим: VALUE
==================================================================
"""

MINIMAL_LOG_BULLISH = """\
Цена сейчас: $95,000

[SIGNAL ALIGNMENT]
==================================================================
  Бычьих   :  8
  Нейтр.   :  3
  Медвежьих:  1
  Н/Д      :  0
  Счёт     : +7
  Вердикт  : 🟢  BULLISH
==================================================================

[FINAL VERDICT]
==================================================================
Текущая цена: $95,000  |  Отклонение от POC: +5.0%
==================================================================
"""


@pytest.fixture
def runs_dir(tmp_path):
    """Временная директория с тестовыми лог-файлами."""
    # Создаём несколько файлов с разными именами
    (tmp_path / "run_2026-05-01_10-00.txt").write_text("old log", encoding="utf-8")
    (tmp_path / "run_2026-06-10_23-46.txt").write_text("newer log", encoding="utf-8")
    (tmp_path / "run_2026-06-13_09-28.txt").write_text(MINIMAL_LOG, encoding="utf-8")
    # Нестандартный файл — не должен мешать
    (tmp_path / "run_alldays_ref_2026-04-15.txt").write_text("ref log", encoding="utf-8")
    return tmp_path


@pytest.fixture
def empty_runs_dir(tmp_path):
    """Пустая директория — нет ни одного лога."""
    return tmp_path


# ---------------------------------------------------------------------------
# TestFindLatestLog
# ---------------------------------------------------------------------------

class TestFindLatestLog:
    """find_latest_log(runs_dir) -> Path | None"""

    def test_returns_path_object(self, runs_dir):
        from mozart_agent import find_latest_log
        result = find_latest_log(runs_dir)
        # WHY: оркестратор читает файл через Path.read_text(); не-Path сломает вызов
        assert isinstance(result, Path)

    def test_returns_most_recent_by_name(self, runs_dir):
        from mozart_agent import find_latest_log
        result = find_latest_log(runs_dir)
        # WHY: имена run_YYYY-MM-DD_HH-MM.txt сортируются лексикографически = по времени
        # если вернём не последний — агент прочитает устаревший контекст
        assert result.name == "run_2026-06-13_09-28.txt"

    def test_returns_none_on_empty_dir(self, empty_runs_dir):
        from mozart_agent import find_latest_log
        result = find_latest_log(empty_runs_dir)
        # WHY: агент должен явно сообщить что логов нет, а не падать с исключением
        assert result is None

    def test_ignores_non_standard_filenames(self, runs_dir):
        from mozart_agent import find_latest_log
        result = find_latest_log(runs_dir)
        # WHY: run_alldays_ref_2026-04-15.txt не является стандартным run_YYYY-MM-DD
        # если его подхватить — агент получит нерелевантный контекст
        assert "alldays" not in result.name

    def test_accepts_string_path(self, runs_dir):
        from mozart_agent import find_latest_log
        result = find_latest_log(str(runs_dir))
        # WHY: пользователь может передать строку в CLI аргументе
        assert result is not None


# ---------------------------------------------------------------------------
# TestParseLogContext
# ---------------------------------------------------------------------------

class TestParseLogContext:
    """parse_log_context(text) -> dict с ключевыми фактами из лога"""

    def test_extracts_btc_price(self):
        from mozart_agent import parse_log_context
        ctx = parse_log_context(MINIMAL_LOG)
        # WHY: цена — первое что называет агент; если неверна — галлюцинация с первой строки
        assert ctx["price"] == 64148

    def test_extracts_alignment_verdict(self):
        from mozart_agent import parse_log_context
        ctx = parse_log_context(MINIMAL_LOG)
        # WHY: вердикт — главный вывод оркестратора; агент обязан его воспроизводить точно
        assert ctx["verdict"] == "BEARISH"

    def test_extracts_alignment_score(self):
        from mozart_agent import parse_log_context
        ctx = parse_log_context(MINIMAL_LOG)
        # WHY: счёт используется агентом при объяснении силы сигнала; ошибка = галлюцинация
        assert ctx["score"] == -4

    def test_extracts_bullish_count(self):
        from mozart_agent import parse_log_context
        ctx = parse_log_context(MINIMAL_LOG)
        # WHY: агент должен называть точное число бычьих сигналов из лога, не придумывать
        assert ctx["bullish_count"] == 3

    def test_extracts_bearish_count(self):
        from mozart_agent import parse_log_context
        ctx = parse_log_context(MINIMAL_LOG)
        # WHY: аналогично — медвежьи сигналы должны совпадать с логом
        assert ctx["bearish_count"] == 7

    def test_extracts_missing_count(self):
        from mozart_agent import parse_log_context
        ctx = parse_log_context(MINIMAL_LOG)
        # WHY: Н/Д сигналы не влияют на счёт — агент должен это учитывать корректно
        assert ctx["missing_count"] == 3

    def test_bullish_verdict(self):
        from mozart_agent import parse_log_context
        ctx = parse_log_context(MINIMAL_LOG_BULLISH)
        # WHY: разные вердикты должны парситься одинаково надёжно
        assert ctx["verdict"] == "BULLISH"
        assert ctx["score"] == 7
        assert ctx["price"] == 95000

    def test_returns_dict(self):
        from mozart_agent import parse_log_context
        ctx = parse_log_context(MINIMAL_LOG)
        # WHY: downstream код обращается к ctx["key"]; список или строка сломают агент
        assert isinstance(ctx, dict)

    def test_missing_alignment_block_returns_none_verdict(self):
        from mozart_agent import parse_log_context
        ctx = parse_log_context("Просто текст без alignment блока")
        # WHY: лог может быть неполным (обрыв записи); агент не должен падать
        assert ctx["verdict"] is None
        assert ctx["score"] is None


# ---------------------------------------------------------------------------
# TestBuildSystemPrompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    """build_system_prompt(log_text, skill_text) -> str"""

    def test_returns_string(self):
        from mozart_agent import build_system_prompt
        result = build_system_prompt(MINIMAL_LOG, "skill content")
        # WHY: системный промпт передаётся в LLM как строка; любой другой тип = ошибка API
        assert isinstance(result, str)

    def test_contains_log_text(self):
        from mozart_agent import build_system_prompt
        result = build_system_prompt(MINIMAL_LOG, "skill content")
        # WHY: без лога в промпте LLM не имеет фактической базы — галлюцинации неизбежны
        assert "64,148" in result

    def test_contains_skill_text(self):
        from mozart_agent import build_system_prompt
        result = build_system_prompt(MINIMAL_LOG, "skill content XYZ")
        # WHY: без скилла LLM не знает правила Mozart и первоисточники
        assert "skill content XYZ" in result

    def test_contains_anti_hallucination_instruction(self):
        from mozart_agent import build_system_prompt
        result = build_system_prompt(MINIMAL_LOG, "skill content")
        # WHY: ключевое требование — агент называет только числа из лога
        # без явного запрета LLM будет придумывать значения метрик
        assert "только" in result.lower() or "лог" in result.lower()

    def test_contains_no_trading_advice_instruction(self):
        from mozart_agent import build_system_prompt
        result = build_system_prompt(MINIMAL_LOG, "skill content")
        # WHY: принцип 2 — никаких торговых рекомендаций; запрет должен быть в промпте
        assert "торгов" in result.lower() or "позици" in result.lower()
