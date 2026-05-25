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
      6. Ровно 1 вызов models.generate_content() (нет дублирования запросов).

СТРАТЕГИЯ МОКА (google-genai SDK):
    mozart_llm.genai.Client патчится через unittest.mock.patch.
    client.models.generate_content().text == синтетический текст.
    Тест не делает реальных HTTP-запросов — не зависит от GOOGLE_API_KEY.
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
    "Синтетический тестовый ответ мока. "
    "LTH SOPR находится ниже рубикона безубытка согласно методологии Mozart. "
    "RSI дневного таймфрейма находится в зоне перепроданности. "
    "Это тестовый текст достаточной длины для прохождения smoke-теста."
)


# ---------------------------------------------------------------------------
# Вспомогательная функция — строит замоканный genai.Client
# ---------------------------------------------------------------------------

def _make_mock_client(text: str = _MOCK_LLM_TEXT) -> MagicMock:
    """
    Возвращает MagicMock, имитирующий genai.Client().
    client.models.generate_content(...).text == text
    """
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = text
    return mock_client


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

    with patch('mozart_llm.genai.Client') as mock_cls:
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

_FORBIDDEN_WORDS = ('купи', 'продай', 'рекомендую', 'войди', 'выйди')


def test_smoke_no_recommendation_words(alignment_mixed, raw_metrics_partial):
    """
    WHY: generate_alignment_summary() — описательная функция, не советник.
    Промпт явно запрещает рекомендации. Тест защищает контракт:
    если функция начнёт добавлять запрещённые слова в постобработке — тест упадёт.
    """
    from mozart_llm import generate_alignment_summary

    with patch('mozart_llm.genai.Client') as mock_cls:
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
    Функция не должна падать — достаточно проверить отсутствие исключения
    и наличие непустой строки.
    """
    from mozart_llm import generate_alignment_summary

    with patch('mozart_llm.genai.Client') as mock_cls:
        mock_cls.return_value = _make_mock_client(
            "Данные по сигналам отсутствуют. "
            "Все метрики недоступны из-за ограничений API. "
            "Анализ невозможен на текущий момент из-за отсутствия данных Mozart."
        )
        # WHY: не должно быть исключения — оркестратор ожидает str, не Exception
        result = generate_alignment_summary(alignment_empty, raw_metrics_all_none)

    assert isinstance(result, str)   # WHY: TypeError при print() если не str
    assert len(result) > 0           # WHY: пустая строка = тихий сбой API-клиента


# ---------------------------------------------------------------------------
# Тест 4: generate_content() вызывается ровно один раз
# ---------------------------------------------------------------------------

def test_smoke_api_called_once(alignment_mixed, raw_metrics_partial):
    """
    WHY: двойной вызов API = баг (дублирование запросов = двойная стоимость).
    Ноль вызовов = функция вернула хардкод или кеш без реального запроса.
    Контракт: ровно 1 вызов models.generate_content() на 1 вызов функции.
    """
    from mozart_llm import generate_alignment_summary

    with patch('mozart_llm.genai.Client') as mock_cls:
        mock_client = _make_mock_client()
        mock_cls.return_value = mock_client
        generate_alignment_summary(alignment_mixed, raw_metrics_partial)

    call_count = mock_client.models.generate_content.call_count
    # WHY exactly 1: 0 = нет вызова (фиктивный ответ), 2+ = дублирование запросов
    assert call_count == 1, (
        f"models.generate_content() вызвана {call_count} раз(а), ожидалась ровно 1"
    )
