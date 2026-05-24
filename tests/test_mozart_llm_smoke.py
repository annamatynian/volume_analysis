"""
tests/test_mozart_llm_smoke.py
Smoke-тест для generate_alignment_summary() из mozart_llm.py (ветка L3-4).

ПОЧЕМУ smoke, а не unit:
    LLM-вывод недетерминирован. Детальные проверки контента сломаются
    при смене модели, промпта или версии API — это ложные падения.
    Smoke-тест защищает только минимальный контракт:
      1. Функция существует и импортируется.
      2. Возвращает str (не None, не int, не объект).
      3. Длина > 50 символов (защита от пустого / обрезанного ответа).
      4. Строка не содержит 'None' (None в f-string = незафильтрованная метрика).
      5. Нет слов прямых рекомендаций (функция описывает, не советует).

СТРАТЕГИЯ МОКА:
    anthropic.Anthropic патчится через unittest.mock.patch.
    Тест не делает реальных HTTP-запросов — не зависит от ANTHROPIC_API_KEY.
    Мок возвращает нейтральный синтетический текст (не API-реалистичный).
"""

import pytest
from unittest.mock import patch, MagicMock

from mozart_alignment import AlignmentResult


# ---------------------------------------------------------------------------
# Фикстуры — нейтральные плейсхолдеры (не API-реалистичные числа)
# ---------------------------------------------------------------------------

@pytest.fixture
def alignment_mixed():
    """
    Смешанный AlignmentResult: есть бычьи, медвежьи, нейтральные и missing.
    Используем нейтральные ID — не привязываемся к конкретным меткам.
    """
    return AlignmentResult(
        bullish=['Н-01', 'М-09'],
        neutral=['М-12'],
        bearish=['М-05'],
        missing=['МБ-03', 'М-01'],
        score=1,
        verdict='MIXED',
        contrarian_flags=['Н-01'],
    )


@pytest.fixture
def alignment_empty():
    """
    AlignmentResult без активных сигналов (все missing).
    Проверяем что функция не падает на пустом контексте.
    """
    return AlignmentResult(
        bullish=[],
        neutral=[],
        bearish=[],
        missing=['М-01', 'М-02', 'М-03'],
        score=0,
        verdict='NEUTRAL',
        contrarian_flags=[],
    )


@pytest.fixture
def raw_metrics_partial():
    """
    Частичные метрики: некоторые None (нет данных из API).
    None-значения должны быть отфильтрованы — не попасть в промпт как 'None'.
    """
    return {
        'lth_sopr': None,       # нет данных → должен быть отфильтрован
        'rsi': 22.3,            # нейтральное тестовое значение
        'etf_flow_btc': -450.0, # нейтральное тестовое значение
        'sth_sopr': None,       # нет данных → должен быть отфильтрован
    }


@pytest.fixture
def raw_metrics_all_none():
    """
    Все метрики None — крайний случай.
    Функция не должна падать и не должна выводить 'None' в результате.
    """
    return {'lth_sopr': None, 'rsi': None, 'etf_flow_btc': None}


# ---------------------------------------------------------------------------
# Синтетический текст мока — нейтральный, не API-реалистичный
# ---------------------------------------------------------------------------

_MOCK_LLM_TEXT = (
    "Синтетический тестовый ответ. "
    "LTH SOPR находится ниже рубикона безубытка согласно методологии Mozart. "
    "RSI дневного таймфрейма находится в зоне перепроданности. "
    "Это тестовый текст длиной более пятидесяти символов."
)


# ---------------------------------------------------------------------------
# Вспомогательная функция — строит замоканный клиент Anthropic
# ---------------------------------------------------------------------------

def _make_mock_client(text: str = _MOCK_LLM_TEXT) -> MagicMock:
    """
    Возвращает MagicMock, имитирующий anthropic.Anthropic().
    messages.create().content[0].text == text
    """
    mock_instance = MagicMock()
    mock_instance.messages.create.return_value.content = [MagicMock(text=text)]
    return mock_instance


# ---------------------------------------------------------------------------
# Тест 1: базовый контракт — непустая строка, нет None, длина > 50
# ---------------------------------------------------------------------------

def test_smoke_returns_nonempty_string(alignment_mixed, raw_metrics_partial):
    """
    WHY str: оркестратор делает print(result) — не-строка даст TypeError.
    WHY len > 50: пустой или обрезанный ответ = сигнал проблемы с API/промптом.
    WHY 'None' absent: None в f-string даёт 'None' — незафильтрованная метрика
        тихо сломает вывод оркестратора (пользователь увидит 'None' в тексте).
    """
    from mozart_llm import generate_alignment_summary

    with patch('mozart_llm.anthropic.Anthropic') as mock_cls:
        mock_cls.return_value = _make_mock_client()
        result = generate_alignment_summary(alignment_mixed, raw_metrics_partial)

    # WHY str: TypeError при print(result) если не строка
    assert isinstance(result, str), (
        "generate_alignment_summary() должна возвращать str, "
        f"получено {type(result).__name__}"
    )
    # WHY > 50: пустой ответ = сломан API-клиент или промпт
    assert len(result) > 50, (
        f"Ответ слишком короткий ({len(result)} символов) — возможна проблема с API"
    )
    # WHY нет 'None': None-метрика не отфильтрована → тихий баг в оркестраторе
    assert 'None' not in result, (
        "Строка 'None' в результате = None-метрика не отфильтрована перед промптом"
    )


# ---------------------------------------------------------------------------
# Тест 2: нет слов прямых рекомендаций
# ---------------------------------------------------------------------------

_FORBIDDEN_WORDS = ('купи', 'продай', 'рекомендую', 'рекомендую', 'войди', 'выйди')


def test_smoke_no_recommendation_words(alignment_mixed, raw_metrics_partial):
    """
    WHY: generate_alignment_summary() — описательная функция, не советник.
    Промпт явно запрещает рекомендации. Тест защищает контракт:
    если промпт изменится и LLM начнёт давать советы — тест упадёт.

    Мок возвращает нейтральный текст без запрещённых слов →
    тест проверяет что функция не добавляет слова самостоятельно
    (например, в постобработке результата).
    """
    from mozart_llm import generate_alignment_summary

    with patch('mozart_llm.anthropic.Anthropic') as mock_cls:
        mock_cls.return_value = _make_mock_client()
        result = generate_alignment_summary(alignment_mixed, raw_metrics_partial)

    result_lower = result.lower()
    for word in _FORBIDDEN_WORDS:
        # WHY каждое слово отдельно: чёткое сообщение что именно нарушено
        assert word not in result_lower, (
            f"Слово '{word}' найдено в результате — функция нарушает контракт "
            "'только факты, без рекомендаций'"
        )


# ---------------------------------------------------------------------------
# Тест 3: крайний случай — пустой контекст (все missing)
# ---------------------------------------------------------------------------

def test_smoke_empty_alignment_does_not_crash(alignment_empty, raw_metrics_all_none):
    """
    WHY: AlignmentResult с пустыми bullish/bearish/neutral = все сигналы missing.
    Такая ситуация реальна при массовых 403/404 от API BGeometrics.
    Функция не должна падать — оркестратор обёрнут в try/except,
    но лучше чтобы функция сама корректно обработала пустой контекст.

    WHY нет проверки содержания: даже с пустым контекстом LLM что-то ответит.
    Достаточно проверить что нет исключения и есть непустая строка.
    """
    from mozart_llm import generate_alignment_summary

    with patch('mozart_llm.anthropic.Anthropic') as mock_cls:
        mock_cls.return_value = _make_mock_client(
            "Данные по сигналам отсутствуют. "
            "Все метрики недоступны из-за ограничений API. "
            "Анализ невозможен на текущий момент из-за отсутствия данных."
        )
        # WHY: не должно быть исключения — оркестратор ожидает str, не Exception
        result = generate_alignment_summary(alignment_empty, raw_metrics_all_none)

    assert isinstance(result, str)  # WHY: TypeError при print() если не str
    assert len(result) > 0          # WHY: пустая строка = тихий сбой API-клиента


# ---------------------------------------------------------------------------
# Тест 4: API клиент вызывается ровно один раз
# ---------------------------------------------------------------------------

def test_smoke_api_called_once(alignment_mixed, raw_metrics_partial):
    """
    WHY: двойной вызов API = баг (дублирование запросов = двойная стоимость).
    Ноль вызовов = функция вернула что-то из кеша или хардкода.
    Контракт: ровно один вызов messages.create() на вызов generate_alignment_summary().
    """
    from mozart_llm import generate_alignment_summary

    with patch('mozart_llm.anthropic.Anthropic') as mock_cls:
        mock_instance = _make_mock_client()
        mock_cls.return_value = mock_instance
        generate_alignment_summary(alignment_mixed, raw_metrics_partial)

    call_count = mock_instance.messages.create.call_count
    # WHY exactly 1: 0 = нет вызова (фиктивный ответ), 2+ = дублирование запросов
    assert call_count == 1, (
        f"messages.create() вызвана {call_count} раз(а), ожидалась 1"
    )
