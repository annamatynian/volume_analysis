# PLAN_MOZART_LEVEL3_SIGNAL_ALIGNMENT.md
# Мини-план Уровня 3 — агрегация и выравнивание сигналов Mozart
# Составлено: 2026-05-18
# Предшественник: PLAN_MOZART_TDD_LEVEL2.md (Уровень 2)
#
# КОНТЕКСТ:
# Уровень 1 (сбор метрик)    — реализован в volume_density.py (оркестратор).
# Уровень 2 (Mozart-разметка) — детерминированные classify-функции в mozart_signals.py.
#   Каждая функция возвращает метку зоны: 'BULL', 'RUBICON', 'INFLOW' и т.д.
# Уровень 3 (выравнивание)   — этот план. Отвечает на вопрос:
#   «Что говорят все сигналы Mozart вместе?»
#
# ПРОБЛЕМА УРОВНЯ 2 В ИЗОЛЯЦИИ:
#   Оркестратор выводит блоки отдельно. Аналитик вынужден вручную считать:
#   сколько сигналов бычьих, сколько медвежьих, есть ли совпадение.
#   Mozart в постах именно так рассуждает — через совпадение нескольких метрик.
#   Уровень 3 автоматизирует этот шаг.
#
# ГРАНИЦА ДЕТЕРМИНИРОВАННОГО И LLM:
#   Детерминированная часть (тестируется через TDD):
#     — полярность каждого сигнала (BULLISH / NEUTRAL / BEARISH)
#     — счётчики совпадений
#     — итоговый вектор выравнивания
#   LLM-часть (НЕ тестируется unit-тестами):
#     — текстовый вывод интерпретации
#     — контекст из постов Mozart
#   Детерминированная часть должна быть готова ДО подключения LLM.
#
# ПРЕДУСЛОВИЕ:
#   Уровень 3 запускается только после завершения всех 10 веток Уровня 2.
#   Частичная реализация бесполезна: неполный набор сигналов даёт искажённый
#   счётчик совпадений.

---

## АРХИТЕКТУРА

```
mozart_signals.py   ←── classify_*() функции (Уровень 2, уже есть)
        │
        ▼
mozart_alignment.py ←── новый модуль Уровня 3
  signal_polarity()     → dict[str, str]   # сигнал → 'BULLISH'/'NEUTRAL'/'BEARISH'
  build_alignment()     → AlignmentResult  # агрегированный вектор
        │
        ▼
volume_density.py   ←── оркестратор, новый блок [SIGNAL ALIGNMENT]
```

**Отдельный модуль `mozart_alignment.py`** — не добавлять в mozart_signals.py.
Причина: mozart_signals.py — чистые функции одного сигнала.
mozart_alignment.py — агрегатор, зависит от всех остальных.

---

## ТАБЛИЦА ПОЛЯРНОСТИ СИГНАЛОВ

Каждый classify-результат → полярность. Полярность зафиксирована здесь,
не вычисляется динамически. При изменении интерпретации Mozart — менять
только эту таблицу (аналог mozart_config.py для числовых порогов).

### Прямые сигналы (полярность прямая)

| ID    | Функция                    | BULLISH              | NEUTRAL                      | BEARISH                           |
|-------|----------------------------|----------------------|------------------------------|-----------------------------------|
| М-01  | classify_lth_sopr_regime   | BULL                 | EARLY_BEAR                   | MID_BEAR, CAPITULATION            |
| М-02  | classify_sth_sopr_regime   | BULL                 | RUBICON                      | BEAR                              |
| М-03  | classify_lth_mvrv_regime   | BULL                 | —                            | BEAR                              |
| М-04  | classify_sth_mvrv_regime   | BULL                 | NEUTRAL                      | BEAR                              |
| М-05  | classify_lth_nupl_regime   | POSITIVE             | RUBICON                      | BEAR, EUPHORIA*                   |
| М-06  | classify_sth_nupl_regime   | POSITIVE             | RUBICON                      | CAPITULATION                      |
| М-07+08| classify_cohort_flow      | ACCUMULATION         | BOTH_BUYING**                | DISTRIBUTION, BOTH_SELLING        |
| М-11  | classify_etf_flow_regime   | INFLOW               | NEUTRAL                      | OUTFLOW                           |
| М-12  | classify_hodl_wave_regime  | AGING                | MIXED                        | REJUVENATING                      |
| МБ-03 | classify_sth_profit_zone   | NEUTRAL_BROKEN,      | NEUTRAL                      | BEAR                              |
|       |                            | HEATED, EUPHORIA***  |                               |                                   |

### Контрарианские сигналы (полярность инвертированная)

⚠️ Важно: Mozart использует эти сигналы как контрарианские — экстремум = разворот.
Инвертируются при достижении экстремальной зоны.

| ID    | Функция                       | Зона          | Полярность в выравнивании   | Логика                              |
|-------|-------------------------------|---------------|-----------------------------|-------------------------------------|
| Н-01  | classify_rsi_regime           | NEUTRAL       | NEUTRAL                     | нет сигнала                         |
|       |                               | OVERSOLD      | BULLISH (контр.)            | Mozart: перепроданность = отскок    |
|       |                               | EXTREME_OVERSOLD | BULLISH (контр., сильный) | исторический прецедент (2020)       |
| Н-02  | classify_red_months_regime    | NORMAL        | NEUTRAL                     | нет сигнала                         |
|       |                               | RARE          | BULLISH (контр.)            | Mozart: 4+ красных = редкость       |
|       |                               | EXTREME       | BULLISH (контр., сильный)   | единственный прецедент 2018         |
| М-05  | classify_lth_nupl_regime      | EUPHORIA      | BEARISH (контр.)            | риск распределения LTH              |
| М-10  | classify_lth_realized_loss    | BELOW_2018    | NEUTRAL                     | норма, нет сигнала                  |
|       |                               | EARLY_2018    | BEARISH (слабый)            | начало давления                     |
|       |                               | MID_2022      | BEARISH                     | активная капитуляция                |
|       |                               | PEAK_FTX      | BEARISH (сильный)           | исторический максимум               |
|       |                               | EXTREME       | BULLISH (контр.)            | Mozart: экстремум убытков = дно     |
| МБ-03 | classify_sth_profit_zone      | EUPHORIA      | BEARISH (контр.)            | перегрев STH = риск разворота       |
| М-09  | detect_sth_rp_zscore_turning  | True          | BULLISH                     | паттерн В: смена направления        |
|       |                               | False         | NEUTRAL                     | ещё падает или нет данных           |

** BOTH_BUYING — нейтральный, не бычий: оба накапливают без перетока = нет подтверждения структуры.
*** МБ-03 EUPHORIA — контрарианский BEARISH если совпадает с LTH NUPL EUPHORIA.
    Иначе — BULLISH. Зависит от контекста. ⚠️ Это единственный контекст-зависимый случай.
    Решение при реализации: упростить до BULLISH, добавить отдельный флаг euphoria_convergence.

---

## СТРУКТУРА AlignmentResult

```python
@dataclass
class AlignmentResult:
    bullish:  list[str]   # ID сигналов с полярностью BULLISH
    neutral:  list[str]   # ID сигналов с полярностью NEUTRAL
    bearish:  list[str]   # ID сигналов с полярностью BEARISH
    missing:  list[str]   # ID сигналов без данных (API 403/404, н/д)
    score:    int         # bullish_count - bearish_count (от -N до +N)
    verdict:  str         # 'BULLISH' / 'NEUTRAL' / 'BEARISH' / 'MIXED'
    contrarian_flags: list[str]  # ID контрарианских сигналов (помечены отдельно)
```

### Вычисление verdict

```
score = len(bullish) - len(bearish)
total_directional = len(bullish) + len(bearish)  # без neutral и missing

if total_directional == 0:
    verdict = 'NEUTRAL'
elif score >= 2:
    verdict = 'BULLISH'
elif score <= -2:
    verdict = 'BEARISH'
elif score == 1 or score == -1:
    verdict = 'MIXED'   # перевес на 1 — недостаточно для вывода
else:  # score == 0
    verdict = 'MIXED'
```

Порог ±2 — FORMALIZED. Обоснование:
  При 10 сигналах score=1 означает 5 BULL vs 4 BEAR — статистический шум.
  score=2 означает 6 vs 4 — минимальный значимый перевес.
  При уточнении — менять только здесь (аналог mozart_config.py).

---

## ВЫВОД В ОРКЕСТРАТОРЕ

Новый блок в volume_density.py, добавляется последним:

```
════════════════════════════════════════════════
[SIGNAL ALIGNMENT]
  Бычьих  : 6  (М-01 М-02 М-04 М-11 Н-01* М-09)
  Нейтр.  : 2  (М-07+08 М-12)
  Медвежьих: 2  (М-05 МБ-03)
  Н/Д     : 1  (М-10 — эндпоинт 404)
  Счёт    : +4
  Вердикт : BULLISH
  * — контрарианский сигнал (OVERSOLD → бычий разворот)
════════════════════════════════════════════════
```

Блок строго описательный: счётчики и вердикт. Никаких «рынок вырастет».
Правило из архитектурного принципа проекта: оркестратор описывает факт.

---

## LLM-ИНТЕГРАЦИЯ (опционально, после детерминированной части)

### Что делает LLM
Принимает AlignmentResult + сырые значения метрик + контекст из постов Mozart.
Генерирует 3–5 предложений текстового резюме.

### Что LLM НЕ делает
- Не считает сигналы (это делает build_alignment())
- Не решает полярность (зафиксирована в таблице выше)
- Не даёт инвестиционных рекомендаций
- Не имеет памяти между запусками

### Архитектура вызова
```python
# mozart_llm.py (отдельный модуль)
def generate_alignment_summary(
    alignment: AlignmentResult,
    raw_metrics: dict,        # сырые значения для контекста
    mozart_context: str,      # выжимка из PLAN_MOZART_PATTERNS.md
) -> str:
    """
    Returns: 3–5 предложений текстового резюме.
    НЕ тестируется unit-тестами (недетерминированный вывод).
    Smoke-test: результат не пустой и не содержит 'None'.
    """
```

### Когда добавлять LLM
Только после того как детерминированная часть (mozart_alignment.py) работает
стабильно минимум 2 недели. LLM — надстройка, не замена.

---

## ПЛАН РЕАЛИЗАЦИИ (ветки)

```
ВЕТКА L3-1: Таблица полярности + signal_polarity()
  Файл: mozart_alignment.py (новый)
  TDD:  tests/test_mozart_alignment_polarity.py
  Что:  функция принимает (signal_id, label) → 'BULLISH'/'NEUTRAL'/'BEARISH'
  Важно: контрарианские случаи — отдельные тесты с WHY

ВЕТКА L3-2: build_alignment() + AlignmentResult
  Файл: mozart_alignment.py (расширение)
  TDD:  tests/test_mozart_alignment_build.py
  Что:  принимает dict всех сигналов, возвращает AlignmentResult
  Важно: missing сигналы (None, н/д) — не считаются ни BULL ни BEAR

ВЕТКА L3-3: блок [SIGNAL ALIGNMENT] в оркестраторе
  Файл: volume_density.py
  TDD:  оркестратор не unit-тестируется; интеграционная проверка вручную
  Что:  вызов build_alignment() + форматированный вывод

ВЕТКА L3-4 (опционально): LLM-резюме
  Файл: mozart_llm.py (новый)
  TDD:  smoke-test только (недетерминированный вывод)
  Что:  generate_alignment_summary() → str
```

---

## КРИТЕРИИ ВКЛЮЧЕНИЯ В L3-1

Перед стартом ВЕТКИ L3-1 должны быть закрыты все 10 веток Уровня 2.
Проверка:
  - `pytest --tb=no -q` → 0 failed
  - Все 10 classify-функций существуют в mozart_signals.py
  - Таблица полярности выше актуальна (сверить с последними постами Mozart)

---

## ПРАВИЛА (наследуются из Уровня 2)

1. Пороги полярности — только в mozart_alignment.py (аналог mozart_config.py)
2. TDD: RED подтверждён pytest перед написанием кода
3. edit_file: dryRun=true перед apply
4. py_compile после каждого production-изменения
5. Контрарианские сигналы — отдельные тесты с развёрнутым WHY
6. missing сигналы никогда не влияют на score (отдельный список, не NEUTRAL)
   WHY: н/д из-за 403 API ≠ нейтральный рынок; это отсутствие данных
