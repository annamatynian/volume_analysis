# PLAN_REMAINING_TASKS.md
# Актуальный план оставшихся задач
# Составлено: 2026-05-20
# Текущее состояние: 735 passed, 76 skipped, 0 failed
#
# КАК ПОЛЬЗОВАТЬСЯ ЭТИМ ДОКУМЕНТОМ:
#   Читать в начале каждой сессии. Статус обновлять после каждого закрытия задачи.
#   Для деталей паттернов → docs/PLAN_MOZART_PATTERNS.md (первоисточник).
#   Для деталей TDD Level 2 → docs/PLAN_MOZART_TDD_LEVEL2.md (всё закрыто, архив).
#   Для деталей Level 3 → docs/PLAN_MOZART_LEVEL3_SIGNAL_ALIGNMENT.md.

---

## СТАТУС: ЧТО УЖЕ ЗАКРЫТО (не трогать)

### PLAN_MOZART_TDD_LEVEL2.md — ПОЛНОСТЬЮ ЗАКРЫТ ✅
Все 10 веток, все classify/detect функции в mozart_signals.py, все тесты.

| Ветка | Паттерн | Тест-файл |
|---|---|---|
| 1 | М-01 LTH SOPR | test_mozart_lth_sopr.py |
| 2 | М-10 LTH Realized Loss | test_mozart_lth_loss.py |
| 3 | М-05 LTH NUPL | test_mozart_lth_nupl.py |
| 4 | М-02 STH SOPR | test_mozart_sth_sopr.py |
| 5 | М-11 ETF Flow | test_mozart_etf_flow.py |
| 6 | М-03+М-04 LTH/STH MVRV | test_mozart_mvrv.py |
| 7 | М-06 STH NUPL | test_mozart_sth_nupl.py |
| 8 | М-07+М-08 Cohort Flow | test_mozart_cohort_flow.py |
| 9 | М-12 HODL Waves | test_mozart_hodl_waves.py |
| 10 | М-09 STH RP Z-score turning | test_mozart_sth_realized_price.py |

Дополнительно закрыто: Н-01 RSI (test_mozart_rsi.py), Н-02 Red Months
(test_mozart_red_months.py), МБ-03 STH Profit (test_mozart_sth_profit.py),
CO-3 calculate_vwap_deviation (test_calculate_vwap_deviation.py + smoke).

---

## БЛОК 1 — ТЕХНИЧЕСКИЙ ДОЛГ (приоритет: сначала)

### CO-6 | Отсутствуют slug/column тесты для 6 методов onchain_client.py (Блок 2) — ЗАКРЫТ ✅ (2026-05-21, 726p/76s/0f)
Добавлен: класс TestBlok2MethodSlugsAndColumns в test_onchain_client.py, 14 тестов.
Production-код не менялся — все 6 методов были корректны.
**Обнаружено:** 2026-05-20, сессия L3-1
**Суть:** Методы добавлены в Блоке 2 без соответствующих тестов в test_onchain_client.py.
  При рефакторинге slug легко перепутать молча (нет 404-исключения, просто неверные данные).
  Риск реализуется при интеграции в оркестратор (L3-3).
**Методы без тестов:**
  get_realized_price()           slug: realized-price       col: realized_price
  get_true_market_mean()         slug: true-market-mean     col: true_market_mean
  get_supply_loss()              slug: supply-loss          col: supply_loss
  get_nupl()                     slug: nupl                 col: nupl
  get_mvrv_zscore()              slug: mvrv-zscore          col: mvrv_zscore
  get_realized_cap_hodl_waves()  slug: realized-cap-hodl-waves  col: multi (age_*)
**Что добавить:** класс TestBlok2MethodSlugsAndColumns в test_onchain_client.py
  — 2 теста на каждый простой метод (slug + column)
  — для get_realized_cap_hodl_waves(): дополнительно float-каст + удаление unixTs
  Итого ~14 тестов. Делать по общему TDD-шаблону из TestExpansionMethodSlugs.
**Когда закрывать:** перед или сразу после L3-3 (до интеграции в оркестратор).

### CO-5 | realized_loss_lth_usd 404 — ЗАКРЫТ ✅ (2026-05-20, 540p/76s/0f)
**Диагностика завершена. Результат `_diag_realized_loss.py`:**
  - /v1/realized_loss_lth_usd → 404 (paywall/удалён на free tier)
  - /v1/realized_loss_lth_btc → 404 (paywall/удалён на free tier)
  - /v1/realized_loss_lth     → 200 OK ✅ (работает, 1440 записей)
    Поле: `realizedLossLth` (float, знак минус = убыток)
    Диапазон: 2022-05-20 … 2026-05-19
    Последнее значение: -415,157,636.61 (~$415M/день, USD-деноминация без суффикса)
**ВЫВОД:** Endpoint жив, но _usd и _btc варианты за paywall.
  Базовый `/v1/realized_loss_lth` возвращает USD-значения без суффикса.
**Действия для фикса `onchain_client.py`:**
  1. URL:   `realized-loss-lth-usd` → `realized_loss_lth`
  2. Поле:  `realizedLossLthUsd`    → `realizedLossLth`
  3. Тип:   float (не str), знак уже корректный (отрицательный = убыток)
  4. После фикса: запустить `_diag_lth_loss_live.py` end-to-end
**Файлы:** `onchain_client.py`
**Документ-источник:** docs/PLAN_MOZART_PATTERNS.md → М-10

---

## БЛОК 2 — НОВЫЕ ENDPOINTS (приоритет: HIGH)

> Для каждого: (1) диагностика → (2) метод в onchain_client.py → (3) TDD

**Правило диагностики:** ≤5 запросов, следовать RULES_API_DIAGNOSTICS.md.
Сверять имена полей с `BGeometrics_docs.json` (файл в корне проекта).

### МБ-01 | realized-price → «Синяя линия» дна цикла — ЗАКРЫТ ✅
**Паттерн Mozart:** цена ниже Realized Price = зона исторического дна ±20%
  (пост 25.02.2026). Самый сильный уровень из трёхлинейной модели.
**Endpoint:** `/v1/realized-price`, поле `realizedPrice`
**Что сделано:**
  - `get_realized_price()` в `onchain_client.py`
  - `classify_realized_price_regime(price, realized_price) -> str` в `mozart_signals.py`
    зоны: ABOVE / AT (±20% буфер) / BELOW
  - `tests/test_mozart_realized_price.py`
**Диагностика:** 200 OK, 1461 запись, поле realizedPrice

### МБ-02 | true-market-mean → «Зелёная линия», рубикон медвежьего рынка
**Статус:** ✅ ЗАКРЫТ (2026-05-20)
**Что сделано:**
  - `get_true_market_mean()` в `onchain_client.py` (шаблон `get_realized_price()`)
  - `classify_true_market_mean_regime(price, tmm) -> str` в `mozart_signals.py`
    (бинарный рубикон: price >= tmm → 'ABOVE', price < tmm → 'BELOW', без буфера)
  - `tests/test_mozart_true_market_mean.py`: 7 тестов, RED→GREEN
  - `volume_density.py`: блок `[МБ-02 | TRUE MARKET MEAN]` + строка в `[FINAL VERDICT]`
**Ключевое решение:** Без AT-зоны и буфера (TMM = чёткий рубикон);
  граница price == tmm → 'ABOVE' (рубикон не пробит вниз).

### МБ-04 | supply-loss → Счётчик монет в убытке — ЗАКРЫТ ✅
**Паттерн Mozart:** 5M монет = структурный сигнал смены тренда; 3–3.5M = промежуточный
  (пост 02.04.2026, пост 08.04.2026).
**Endpoint:** `/v1/supply-loss`, поле `supplyLoss`
**Что сделано:**
  - `get_supply_loss()` в `onchain_client.py`
  - `classify_supply_loss_regime(supply_loss_btc: float) -> str` в `mozart_signals.py`
    зоны: EXTREME (≥5M) / ELEVATED (≥3.5M) / INTERMEDIATE (>0) / LOW (≤0)
  - `tests/test_mozart_supply_loss.py`: 13 тестов
**Диагностика:** 7.641M BTC → EXTREME; поле API: supplyLoss (float)

### МБ-06 | nupl → Общий NUPL, 30-дн MA, цель дна
**Статус:** ✅ ЗАКРЫТ (2026-05-20)
**Что сделано:**
  - `get_nupl()` в `onchain_client.py`
  - `classify_nupl_regime(nupl: float) -> str` в `mozart_signals.py`
    зоны: BULL (≥0.50) / HOPE (≥0.25) / EARLY_BEAR (≥0.0) / BEAR (>-0.40) / BOTTOM_ZONE (≤-0.40)
  - `tests/test_mozart_nupl.py`: 19 тестов, RED→GREEN
**Ключевые факты:** поле `nupl`, десятичная дробь (0.3004 = 30.04%);
  текущее 0.3004 → HOPE; пик EARLY_BEAR марта 2026 ~0.18 попадает в EARLY_BEAR;
  цель дна Mozart -0.40 = BOTTOM_ZONE.
**Примечание:** интеграция в оркестратор — отдельным проходом после МБ-08.
**Документ-источник:** docs/PLAN_MOZART_PATTERNS.md → МБ-06

### МБ-07 | mvrv-zscore → Макро позиционирование относительно истории
**Статус:** ✅ ЗАКРЫТ (2026-05-20)
**Что сделано:**
  - `get_mvrv_zscore()` в `onchain_client.py`
  - `classify_mvrv_zscore_regime(z: float) -> str` в `mozart_signals.py`
    зоны: PEAK (>7) / BULL (≥3) / NEUTRAL (≥0) / BEAR (>-1) / BOTTOM (≤-1)
  - `tests/test_mozart_mvrv_zscore.py`: 19 тестов, RED→GREEN
**Ключевые факты:** поле `mvrvZscore`, float;
  диапазон в данных [-0.36, 3.35]; текущее 0.7759 → NEUTRAL.
**Примечание:** интеграция в оркестратор — отдельным проходом после МБ-08.
**Документ-источник:** docs/PLAN_MOZART_PATTERNS.md → МБ-07

### МБ-08 | realized-cap-hodl-waves → One-Cycle Average
**Статус:** ✅ ЗАКРЫТ (2026-05-20)
**Что сделано:**
  - `get_realized_cap_hodl_waves()` в `onchain_client.py` (multi-column, RC-доли)
  - `calculate_one_cycle_average(rc_2y3y, rc_3y4y, realized_price, supply_2y3y, supply_3y4y, total_supply) -> float`
  - `classify_one_cycle_regime(price, one_cycle_avg, days_below=0) -> str`
    зоны: ABOVE / TECHNICAL_BEAR / CONFIRMED_BEAR (> 60 дней)
  - `tests/test_mozart_one_cycle.py`: 17 тестов, RED→GREEN
**Ключевые факты:**
  - `/v1/realized-cap-hodl-waves`: RC-доли (0–1, строки->каст float); age_2y_3y=0.056, age_3y_4y=0.051
  - `/v1/hodl-waves-supply` (уже в get_hodl_waves): age_2y_3y=1.12M BTC, age_3y_4y=1.02M BTC
  - `/v1/hodl-waves` → 404 (paywall); supply через `hodl-waves-supply`
  - OCA ≈ realized_price на текущих данных (RC-доля ≈ supply-доля для когорты)
**Примечание:** интеграция в оркестратор — отдельным проходом после закрытия всех функций.
**Документ-источник:** docs/PLAN_MOZART_PATTERNS.md → МБ-08

---

## БЛОК 3 — LEVEL 3: SIGNAL ALIGNMENT (приоритет: после Блока 2)

**Предусловие:** Блок 2 закрыт (все новые endpoints работают).
**Архитектура:** docs/PLAN_MOZART_LEVEL3_SIGNAL_ALIGNMENT.md (читать перед стартом).

### L3-1 | signal_polarity() — таблица полярности — ЗАКРЫТ ✅ (2026-05-21, 679p/76s/0f)
**Файл:** mozart_alignment.py
**Тест-файл:** tests/test_mozart_alignment_polarity.py — 53 теста
**Суть:** (signal_id, label) → 'BULLISH'/'NEUTRAL'/'BEARISH'
Таблица полярности: 14 сигналов, контрарианские с развёрнутым WHY.

### L3-2 | build_alignment() + AlignmentResult — ЗАКРЫТ ✅ (2026-05-21, 712p/76s/0f)
**Файл:** mozart_alignment.py (расширение)
**Тест-файл:** tests/test_mozart_alignment_build.py — 33 теста + smoke
**Суть:** dict всех сигналов → AlignmentResult(bullish, neutral, bearish, missing, score, verdict)
Вердикт: score ≥ +2 → BULLISH; ≤ −2 → BEARISH; иначе → MIXED/NEUTRAL.
missing (API 404/403) → не влияет на score, отдельный список.

### L3-3 | блок [SIGNAL ALIGNMENT] в оркестраторе
**Файл:** volume_density.py
**Суть:** вызов build_alignment() + форматированный вывод последним блоком.
Без unit-тестов; интеграционная проверка вручную.

### L3-4 | LLM-резюме (опционально)
**Файл:** mozart_llm.py (новый)
**Суть:** generate_alignment_summary(alignment, raw_metrics, mozart_context) → str
Только smoke-test (недетерминированный вывод).
Добавлять только после того как L3-1..L3-3 работают стабильно ≥2 недели.

---

## ПОРЯДОК ВЫПОЛНЕНИЯ

```
[СЕЙЧАС]
  CO-5  Диагностика realized_loss_lth_usd 404       ✅ ЗАКРЫТ

[БЛОК 2 — в порядке приоритета]
  МБ-01  realized-price           ✅ ЗАКРЫТ
  МБ-02  true-market-mean         ✅ ЗАКРЫТ
  МБ-04  supply-loss              ✅ ЗАКРЫТ  (7.641M BTC → EXTREME, поле API: supplyLossBtc)
  МБ-05  realized-profit-lth-usd  ✅ ЗАКРЫТ  (slug с дефисами, 1440 записей, USD)
  МБ-06  nupl                     ✅ ЗАКРЫТ  (0.3004 = HOPE; поле nupl, десятичная дробь)
  МБ-07  mvrv-zscore              ✅ ЗАКРЫТ  (0.7759 = NEUTRAL; поле mvrvZscore)
  МБ-08  realized-cap-hodl-waves  ✅ ЗАКРЫТ  (OCA = RC-доля × realized_price × total / cohort_supply)

[БЛОК 3 — после Блока 2]
  L3-1  signal_polarity()                ✅ ЗАКРЫТ  (53 теста)
  L3-2  build_alignment() + AlignmentResult  ✅ ЗАКРЫТ  (33 теста + smoke)
  L3-3  блок [SIGNAL ALIGNMENT] в оркестраторе   ✅ ЗАКРЫТ (2026-05-23, 745p/76s/0f)
  L3-4  LLM-резюме (mozart_llm.py)               ✅ ЗАКРЫТ (2026-05-24, 749p/76s/0f)
  НВ-03 BTC Dominance (CoinGecko)                ✅ ЗАКРЫТ (2026-05-25, 778p/76s/0f)
  БАГ   datetime import в volume_density.py      ✅ ЗАКРЫТ (2026-05-25)
  БАГ   Mozart LLM gemini-2.0-flash quota=0      ✅ ЗАКРЫТ (2026-05-25, → gemini-2.5-flash)
```

---

## КЛЮЧЕВЫЕ ДОКУМЕНТЫ — СПРАВОЧНИК

| Документ | Назначение | Когда читать |
|---|---|---|
| docs/PLAN_MOZART_PATTERNS.md | Первоисточник всех паттернов Mozart с цитатами из постов | Перед каждым новым паттерном |
| docs/PLAN_MOZART_TDD_LEVEL2.md | Архив Level 2 (все ветки закрыты) | Только при вопросах по уже реализованным функциям |
| docs/PLAN_MOZART_LEVEL3_SIGNAL_ALIGNMENT.md | Архитектура агрегации сигналов (Level 3) | Перед стартом Блока 3 |
| RULES_API_DIAGNOSTICS.md | Правила диагностики endpoints (≤5 запросов) | Перед каждой новой диагностикой |
| BGeometrics_docs.json | Актуальный список endpoints BGeometrics | При диагностике (сверка имён полей) |
| mozart_config.py | Все числовые пороги паттернов | При написании тестов и функций |
| mozart_signals.py | Все classify/detect функции Level 2 | При добавлении новых функций |

---

## ПРАВИЛА (действуют для всех задач)

1. **Читать этот документ в начале сессии** — не PLAN_MOZART_TDD_LEVEL2.md (он архив)
2. Пороги — только из mozart_config.py, не хардкодить в тестах
3. TDD: RED подтверждён pytest → только потом GREEN
4. WHY-комментарий к каждому assert
5. Нейтральные плейсхолдеры в тестах (не API-реалистичные числа)
6. Граничные значения — отдельные тесты
7. py_compile после каждого изменения production-кода
8. Обновлять СТАТУС в этом документе после каждого закрытия задачи
