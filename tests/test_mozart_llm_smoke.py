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


# ---------------------------------------------------------------------------
# Тест 5: нет слов прогноза цены
# ---------------------------------------------------------------------------

_FORBIDDEN_FORECAST = (
    'вырастет', 'упадёт', 'достигнет', 'пойдёт вверх', 'пойдёт вниз',
    'будет расти', 'будет падать', 'может вырасти', 'ожидается рост',
    'ожидается падение', 'продолжит рост', 'продолжит падение',
)


def test_smoke_no_forecast_words(alignment_mixed, raw_metrics_partial):
    """
    WHY: generate_alignment_summary() — описательная функция, не прогностическая.
    Промпт явно запрещает прогнозы цены. Тест ловит случай если постобработка
    или смена модели добавит прогнозные формулировки в вывод.
    Сначала проверяем что детекция работает (bad_mock содержит запрещённое слово),
    затем что clean_mock проходит без нарушений.
    """
    from mozart_llm import generate_alignment_summary

    # --- Шаг 1: убеждаемся что детекция вообще работает ---
    bad_text = (
        "LTH SOPR ниже 1.0. "
        "Согласно методологии Mozart цена вырастет до следующего уровня. "
        "ETF оттоки продолжают оказывать давление на рынок."
    )
    with patch('mozart_llm.genai.Client') as mock_cls:
        mock_cls.return_value = _make_mock_client(bad_text)
        bad_result = generate_alignment_summary(alignment_mixed, raw_metrics_partial)

    # WHY: если bad_text не содержит запрещённых слов — тест бесполезен
    assert any(w in bad_result.lower() for w in _FORBIDDEN_FORECAST), (
        "Детекция прогнозных слов не работает — bad_mock должен содержать хотя бы одно; "
        "проверь список _FORBIDDEN_FORECAST"
    )

    # --- Шаг 2: clean_mock не должен содержать прогнозных слов ---
    with patch('mozart_llm.genai.Client') as mock_cls:
        mock_cls.return_value = _make_mock_client()  # стандартный нейтральный мок
        clean_result = generate_alignment_summary(alignment_mixed, raw_metrics_partial)

    clean_lower = clean_result.lower()
    for word in _FORBIDDEN_FORECAST:
        # WHY каждое слово отдельно: чёткое сообщение о нарушении
        assert word not in clean_lower, (
            f"Слово прогноза '{word}' найдено в результате — функция нарушает контракт "
            "'только факты, без прогнозов цены'"
        )


# ---------------------------------------------------------------------------
# Тест 6: нет причинно-следственных выводов о будущем
# ---------------------------------------------------------------------------

_FORBIDDEN_CAUSAL_FUTURE = (
    'поэтому цена',
    'что приведёт к',
    'что вызовет',
    'следовательно цена',
    'обусловит рост',
    'обусловит падение',
    'повлечёт',
    'из-за чего цена',
    'это означает что цена',
)


def test_smoke_no_causal_future_language(alignment_mixed, raw_metrics_partial):
    """
    WHY: Mozart-анализ = описание факта метрики, не вывод о будущем.
    'LTH SOPR < 1 → что приведёт к росту' — запрещённый паттерн.
    'LTH SOPR < 1 — продажи ниже себестоимости' — разрешённый паттерн.
    Тест защищает границу между описанием (допустимо) и причинным прогнозом
    (запрещено).
    """
    from mozart_llm import generate_alignment_summary

    with patch('mozart_llm.genai.Client') as mock_cls:
        mock_cls.return_value = _make_mock_client()
        result = generate_alignment_summary(alignment_mixed, raw_metrics_partial)

    result_lower = result.lower()
    for phrase in _FORBIDDEN_CAUSAL_FUTURE:
        assert phrase not in result_lower, (
            f"Причинно-следственный прогноз '{phrase}' найден — нарушение контракта "
            "'описание фактов без выводов о будущем'"
        )


# ---------------------------------------------------------------------------
# Тест 7: промпт содержит ID активных сигналов
# ---------------------------------------------------------------------------

def test_prompt_contains_active_signal_ids(alignment_mixed, raw_metrics_partial):
    """
    WHY: generate_alignment_summary() должна передавать в промпт ID активных
    сигналов (bullish + bearish + neutral из AlignmentResult).
    Если ID отсутствуют — LLM анализирует без контекста Mozart и галлюцинирует.
    alignment_mixed: bullish=['Н-01','М-09'], neutral=['М-12'], bearish=['М-05'].
    Тест проверяет структуру промпта, а не вывод LLM.
    """
    from mozart_llm import generate_alignment_summary

    with patch('mozart_llm.genai.Client') as mock_cls:
        mock_client = _make_mock_client()
        mock_cls.return_value = mock_client
        generate_alignment_summary(alignment_mixed, raw_metrics_partial)

    call_args = mock_client.models.generate_content.call_args
    # contents может быть positional arg[1] или keyword arg 'contents'
    prompt = call_args.kwargs.get('contents') or (
        call_args.args[1] if len(call_args.args) > 1 else str(call_args)
    )

    for signal_id in ('Н-01', 'М-09', 'М-12', 'М-05'):
        # WHY: ID сигнала в промпте = LLM получила контекст Mozart для этого сигнала
        assert signal_id in prompt, (
            f"ID сигнала '{signal_id}' отсутствует в промпте — "
            "LLM не получила контекст Mozart, возможна галлюцинация"
        )


# ---------------------------------------------------------------------------
# Тест 8: контрарианские сигналы помечены в промпте
# ---------------------------------------------------------------------------

def test_prompt_marks_contrarian_signals(alignment_mixed, raw_metrics_partial):
    """
    WHY: контрарианский сигнал (Н-01: OVERSOLD → BULLISH) имеет инвертированную
    полярность. LLM должна знать что 'перепроданность → бычий' — паттерн Mozart,
    а не ошибка. Маркер [КОНТРАРИАНСКИЙ] обязан присутствовать в промпте рядом
    с ID сигнала-контрария. alignment_mixed.contrarian_flags = ['Н-01'].
    """
    from mozart_llm import generate_alignment_summary

    with patch('mozart_llm.genai.Client') as mock_cls:
        mock_client = _make_mock_client()
        mock_cls.return_value = mock_client
        generate_alignment_summary(alignment_mixed, raw_metrics_partial)

    call_args = mock_client.models.generate_content.call_args
    prompt = call_args.kwargs.get('contents') or (
        call_args.args[1] if len(call_args.args) > 1 else str(call_args)
    )

    # WHY: без маркера LLM интерпретирует Н-01 (RSI OVERSOLD) как медвежий сигнал
    assert 'КОНТРАРИАНСКИЙ' in prompt, (
        "Маркер 'КОНТРАРИАНСКИЙ' отсутствует в промпте — "
        "LLM не получила информацию об инвертированной полярности Н-01"
    )


# ---------------------------------------------------------------------------
# Тест 9: минимальная содержательная длина (калибровка 2026-05-25)
# ---------------------------------------------------------------------------

def test_smoke_minimum_content_length(alignment_mixed, raw_metrics_partial):
    """
    WHY 200 символов: живой прогон 2026-05-25 показал 5 абзацев (~700+ символов
    при 17 активных метриках). Порог 200 — нижняя граница содержательного ответа
    (≈ 1 полный абзац из ~35 слов). Ответ 51–199 символов = обрыв, не анализ.
    WHY не 700: мок возвращает синтетический текст, не реальный LLM-вывод.
    Проверка полноты реального вывода — ручная (docs/LLM_QUALITY_CHECKLIST.md).
    """
    from mozart_llm import generate_alignment_summary

    long_mock = _MOCK_LLM_TEXT * 3  # гарантированно > 200 символов

    with patch('mozart_llm.genai.Client') as mock_cls:
        mock_cls.return_value = _make_mock_client(long_mock)
        result = generate_alignment_summary(alignment_mixed, raw_metrics_partial)

    # WHY 200: меньше = обрыв (1 предложение ≠ анализ 13 активных сигналов Mozart)
    assert len(result) >= 200, (
        f"Результат {len(result)} символов — меньше минимального порога "
        "содержательного анализа (200). Возможен обрыв или пустой промпт."
    )


# ---------------------------------------------------------------------------
# Тест 10: ЛЛМ-БАГ-1 — форматирование больших float в промпте
# ---------------------------------------------------------------------------

def test_prompt_formats_large_floats(alignment_mixed):
    """
    WHY: raw_metrics с float 213722897.004 попадает в промпт как сырое число
    (строка '-213722897.004'), LLM воспроизводит нечитаемо: -213,722,897.00428572.
    Форматирование до '$213.7 млн' перед передачей в промпт делает вывод LLM
    читаемым. Тест проверяет call_args (промпт), а не вывод LLM.
    """
    from mozart_llm import generate_alignment_summary

    raw_metrics_with_large_float = {
        'realized_loss_lth_usd': -213_722_897.004,   # >= 1M → '$-213.7 млн'
        'rsi': 46.06115626812503,                    # < 1000 → '46.06'
    }

    with patch('mozart_llm.genai.Client') as mock_cls:
        mock_client = _make_mock_client()
        mock_cls.return_value = mock_client
        generate_alignment_summary(alignment_mixed, raw_metrics_with_large_float)

    call_args = mock_client.models.generate_content.call_args
    prompt = call_args.kwargs.get('contents') or (
        call_args.args[1] if len(call_args.args) > 1 else str(call_args)
    )

    # WHY: сырой long-float в промпте = LLM цитирует нечитаемое число в SUMMARY;
    # _fmt_metric() обязан заменить его на '$-213.7 млн' до передачи LLM
    assert '213722897' not in prompt, (
        "Сырой float 213722897 в промпте — _fmt_metric() не применён; "
        "LLM воспроизводит -213,722,897.00... вместо '$213.7 млн'"
    )
    # WHY: '213.7' в промпте = _fmt_metric() применён корректно для значений >=1M
    assert '213.7' in prompt, (
        "Форматированное значение '213.7' отсутствует в промпте — "
        "_fmt_metric() не добавлен в шаг 2 generate_alignment_summary()"
    )


# ---------------------------------------------------------------------------
# Тест 11: ЛЛМ-БАГ-3 — 'рост затруднён' отсутствует в SIGNAL_CONTEXT['М-11']
# ---------------------------------------------------------------------------

def test_signal_context_m11_no_causal_phrase():
    """
    WHY: 'рост затруднён' в SIGNAL_CONTEXT['М-11'] попадает в промпт LLM
    и воспроизводится как 'затрудненный рост' — пограничная причинно-следственная
    формулировка будущего, запрещённая правилами Mozart (чек-лист Ч.1.2).
    Тест проверяет что фраза убрана из SIGNAL_CONTEXT прежде чем попасть в промпт.
    """
    from mozart_llm import SIGNAL_CONTEXT

    m11_ctx = SIGNAL_CONTEXT.get('М-11') or ''

    # WHY: фраза в SIGNAL_CONTEXT → в промпте → LLM воспроизводит как прогнозную;
    # её отсутствие в словаре = LLM физически не может её галлюцинировать
    assert 'рост затруднён' not in m11_ctx, (
        "SIGNAL_CONTEXT['М-11'] содержит 'рост затруднён' — LLM воспроизводит "
        "как 'затрудненный рост' (пограничный прогноз по чек-листу Ч.1.2). "
        "Убрать фразу, оставить: 'Устойчивые оттоки = ключевой медвежий сигнал.'"
    )


# ---------------------------------------------------------------------------
# Тест 12: ЛЛМ-БАГ-2 — промпт содержит инструкцию покрытия КАЖДОГО сигнала
# ---------------------------------------------------------------------------

def test_prompt_contains_mandatory_coverage_instruction(
    alignment_mixed, raw_metrics_partial
):
    """
    WHY: М-12 HODL Waves систематически пропускается LLM при ≥13 активных
    сигналах (живые прогоны 2026-05-25 при 400 и 600 токенах). LLM обрабатывает
    сигналы по порядку и обрезает хвост при нехватке бюджета.
    Явная инструкция 'упомяни КАЖДЫЙ активный сигнал' устраняет пропуск.
    Тест проверяет call_args (промпт), а не недетерминированный вывод LLM.
    """
    from mozart_llm import generate_alignment_summary

    with patch('mozart_llm.genai.Client') as mock_cls:
        mock_client = _make_mock_client()
        mock_cls.return_value = mock_client
        generate_alignment_summary(alignment_mixed, raw_metrics_partial)

    call_args = mock_client.models.generate_content.call_args
    prompt = call_args.kwargs.get('contents') or (
        call_args.args[1] if len(call_args.args) > 1 else str(call_args)
    )

    # WHY: без слова 'КАЖДЫЙ' LLM пропускает хвостовые сигналы при сжатии вывода;
    # явная инструкция обязывает упомянуть каждый ID из АКТИВНЫЕ СИГНАЛЫ
    assert 'КАЖДЫЙ' in prompt or 'каждый' in prompt, (
        "Инструкция 'упомяни КАЖДЫЙ активный сигнал' отсутствует в промпте — "
        "М-12 и другие хвостовые сигналы пропускаются при ограниченном бюджете LLM"
    )


# ---------------------------------------------------------------------------
# Тест 13: max_output_tokens >= 900 (фикс ЛЛМ-БАГ-2: обрыв хвоста)
# ---------------------------------------------------------------------------

def test_api_called_with_sufficient_token_budget(alignment_mixed, raw_metrics_partial):
    """
    WHY: при 600 токенах LLM физически не вмещает 13 активных сигналов (~50
    токенов на сигнал). М-12 и Н-02 обрезаются последними. Минимальный бюджет
    для полного покрытия 13 сигналов — 900 токенов.
    Тест защищает контракт: generate_content() вызывается с max_output_tokens >= 900.
    Проверяется через call_args, а не вывод LLM.
    """
    from mozart_llm import generate_alignment_summary

    with patch('mozart_llm.genai.Client') as mock_cls:
        mock_client = _make_mock_client()
        mock_cls.return_value = mock_client
        generate_alignment_summary(alignment_mixed, raw_metrics_partial)

    call_args = mock_client.models.generate_content.call_args
    # config передаётся как keyword arg 'config'
    config = call_args.kwargs.get('config')
    actual_tokens = getattr(config, 'max_output_tokens', None)

    # WHY >= 900: при 13 сигналах × ~50 токенов = 650 минимум + заголовки;
    # 600 токенов доказанно обрезает М-12 и Н-02 (прогоны 2026-05-25, 2026-05-26)
    assert actual_tokens is not None and actual_tokens >= 900, (
        f"max_output_tokens={actual_tokens} — меньше 900; "
        "при 13 активных сигналах хвостовые (М-12, Н-02) обрезаются"
    )


# ---------------------------------------------------------------------------
# Тест 14: _fmt_metric не добавляет $ к BTC-метрикам (фикс нового бага)
# ---------------------------------------------------------------------------

def test_prompt_no_dollar_sign_on_btc_metrics(alignment_mixed):
    """
    WHY: _fmt_metric() применяла $ ко всем значениям >= 1M.
    BTC-метрики (lth_net_position_30d, sth_net_position_30d) — не USD.
    LLM получала '$1.0 млн' и воспроизводила как доллары, хотя это биткоины.
    Фикс: $ применяется только к ключам с суффиксом '_usd'.
    Тест проверяет call_args (промпт), а не вывод LLM.
    """
    from mozart_llm import generate_alignment_summary

    raw_btc_metrics = {
        'lth_net_position_30d': 1_029_914.0,   # BTC, '_usd' нет в ключе → без $
        'sth_net_position_30d': -1_012_192.0,  # BTC, '_usd' нет в ключе → без $
        'realized_loss_lth_usd': -191_000_000.0,  # USD → с $
    }

    with patch('mozart_llm.genai.Client') as mock_cls:
        mock_client = _make_mock_client()
        mock_cls.return_value = mock_client
        generate_alignment_summary(alignment_mixed, raw_btc_metrics)

    call_args = mock_client.models.generate_content.call_args
    prompt = call_args.kwargs.get('contents') or (
        call_args.args[1] if len(call_args.args) > 1 else str(call_args)
    )

    # WHY: '$1.0 млн' при BTC-ключе = LLM цитирует доллары вместо биткоинов;
    # lth_net_position_30d не имеет '_usd' → _fmt_metric(is_usd=False) → без $
    assert '$1.0' not in prompt, (
        "BTC-метрика lth_net_position_30d отформатирована как '$1.0 млн' — "
        "_fmt_metric применяет $ к non-USD ключу; добавить is_usd='_usd' in k"
    )
    assert '$-1.0' not in prompt, (
        "BTC-метрика sth_net_position_30d отформатирована как '$-1.0 млн' — "
        "_fmt_metric применяет $ к non-USD ключу"
    )
    # WHY: USD-ключ realized_loss_lth_usd ДОЛЖЕН получить $
    assert '$-191.0' in prompt, (
        "USD-метрика realized_loss_lth_usd не получила $ — "
        "_fmt_metric(is_usd=True) не применён для '_usd'-ключей"
    )
