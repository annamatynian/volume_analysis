# PLAN_MOZART_TDD_LEVEL2.md
# План TDD Уровня 2 — детерминированные сигналы по паттернам Mozart
# Составлено: 2026-05-18
# Источник паттернов: PLAN_MOZART_PATTERNS.md
#
# КОНТЕКСТ:
# Уровень 1 (сбор метрик) — реализован в volume_density.py (оркестратор).
# Уровень 2 (Mozart-разметка) — детерминированные classify-функции в mozart_signals.py.
#   Каждая функция: числовые пороги только из mozart_config.py,
#   TDD: RED подтверждён pytest → GREEN → py_compile.
# Уровень 3 (LLM-агент) — будущее, после готовности Уровня 2.
#
# КРИТЕРИЙ ВКЛЮЧЕНИЯ В ЭТОТ ПЛАН:
#   Endpoint подтверждён рабочим И метод в onchain_client.py уже существует.
#   Паттерны с непротестированными endpoints — в отдельном плане (PLAN_MOZART_NEW_ENDPOINTS.md).
#
# ТЕКУЩЕЕ СОСТОЯНИЕ:
#   388 passed, 76 skipped, 0 failed (2026-05-18)
#   mozart_signals.py: calculate_rsi, classify_rsi_regime (Н-01),
#                       count_consecutive_red_months, classify_red_months_regime (Н-02),
#                       classify_sth_profit_zone, build_sth_profit_signal (МБ-03)
#   mozart_config.py:  пороги Н-01, Н-02, МБ-03, МБ-04, МБ-05, МБ-08

---

## ПОРЯДОК ВЕТОК (приоритет сверху вниз)

```
ВЕТКА 1: М-01 LTH SOPR рубикон + светофор фаз         ✅ ЗАКРЫТА (2026-05-17, +19 тестов)
ВЕТКА 2: М-10 LTH Realized Loss исторические якоря    ✅ ЗАКРЫТА (2026-05-17, +N тестов)
ВЕТКА 3: М-05 LTH NUPL рубикон + эйфория              ✅ ЗАКРЫТА (2026-05-18, +14 тестов)
ВЕТКА 4: М-02 STH SOPR рубикон                        ← СЛЕДУЮЩАЯ
ВЕТКА 5: М-11 ETF Flow светофор спроса
ВЕТКА 6: М-03 LTH MVRV + М-04 STH MVRV (бинарные, вместе)
ВЕТКА 7: М-06 STH NUPL рубикон
ВЕТКА 8: М-07 + М-08 Cohort Flow (совместный анализ)
ВЕТКА 9: М-12 HODL Waves направление когорт (delta)
ВЕТКА 10: М-09 STH Realized Price паттерн В (Z-score turning)
```

М-13 LTH Pain Proxy — пропустить, уже реализован в оркестраторе.

---

## ВЕТКА 1 | М-01 | LTH SOPR — Рубикон + Светофор фаз

### Источник паттерна
PLAN_MOZART_PATTERNS.md М-01, посты 05.04.2026.

### Что добавить в mozart_config.py
```python
# М-01 | LTH SOPR (пост 05.04.2026)
"lth_sopr_rubicon":       1.0,   # граница бычий↔медвежий
"lth_sopr_early_bear":    0.80,  # начало ранней медвежки (0–20% убыток)
"lth_sopr_deep_bear":     0.50,  # кульминация (40–50% убыток, скорр. цикл)
```

### Что написать в mozart_signals.py
```python
def classify_lth_sopr_regime(sopr: float) -> str:
    """
    Зоны (строгое <, приоритет сверху вниз):
      sopr >= rubicon              → 'BULL'        — LTH продают в прибыль
      sopr >= early_bear           → 'EARLY_BEAR'  — 0–20% убыток, рубикон пройден
      sopr >= deep_bear            → 'MID_BEAR'    — 20–50% убыток, разгар
      sopr <  deep_bear            → 'CAPITULATION'— >50% убыток, кульминация
    Граница rubicon == 1.0 строго:
      sopr == 1.0 → 'BULL' (выше или на уровне = не пробит вниз).
    """

def detect_lth_sopr_turning(history: list[float], window: int = 5) -> bool:
    """
    Паттерн В: SOPR перестаёт падать и начинает расти при стоячей цене.
    True если последние window значений монотонно не убывают (min достигнут
    ранее чем последний элемент).
    WHY: сигнал дна — не абсолютное значение, а смена производной (пост 05.04.2026).
    """
```

### Тест-файл
`tests/test_mozart_lth_sopr.py`

### Граничные тесты обязательно
- `sopr == 1.0` → BULL (граница рубикона: включительно сверху)
- `sopr == 0.80` → EARLY_BEAR (граница: включительно снизу)
- `sopr == 0.50` → MID_BEAR (не CAPITULATION — граница включительно снизу)
- `sopr = 0.499` → CAPITULATION
- Detect turning: история убывает потом растёт → True; монотонно падает → False

### Примечание
Метод `get_lth_sopr()` уже в клиенте и кэше. Пороги 0.50/0.80 — из поста
с поправкой Mozart на сглаживание цикла (исторически 50–60% → 40–50%).

---

## ВЕТКА 2 | М-10 | LTH Realized Loss — Исторические якоря

### Источник паттерна
PLAN_MOZART_PATTERNS.md М-10, посты 02.04.2026 и 03.04.2026.

### ⚠️ Важно: знак значения
Диагностика 2026-05-18: `realizedLossLthUsd = -2.467489232E7`
API возвращает **отрицательные** числа (убыток). В классификаторе
принимаем абсолютное значение (`abs(loss_usd)`).

### Что добавить в mozart_config.py
```python
# М-10 | LTH Realized Loss — исторические якоря (пост 02.04.2026)
# Все значения в USD/день, сравниваем с abs(realizedLossLthUsd).
"lth_loss_anchor_2018":   140_000_000,   # ~$140M — завершение медвежки 2018
"lth_loss_anchor_2022_w1": 300_000_000,  # ~$300M — начало капитуляции 2022
"lth_loss_anchor_2022_ftx": 480_000_000, # ~$480M — пик FTX краха
"lth_loss_anchor_cycle_target": 500_000_000,  # >$500M — ожидаемый пик текущего цикла
```

### Что написать в mozart_signals.py
```python
def classify_lth_realized_loss(loss_usd: float) -> str:
    """
    Принимает raw значение из API (отрицательное или положительное),
    сравнивает abs(loss_usd) с историческими якорями (пост 02.04.2026).

    Зоны:
      abs < anchor_2018              → 'BELOW_2018'      — ниже исторических прецедентов
      abs < anchor_2022_w1           → 'EARLY_2018_RANGE'— уровень завершения медвежки 2018
      abs < anchor_2022_ftx          → 'MID_2022_RANGE'  — уровень начала капитуляции 2022
      abs < anchor_cycle_target      → 'PEAK_FTX_RANGE'  — уровень краха FTX
      abs >= anchor_cycle_target     → 'EXTREME'         — превышает ожидаемый пик цикла

    WHY abs(): API возвращает убыток как отрицательное число.
    WHY якоря, не произвольные пороги: Mozart явно называет $140M/$300M/$480M/$500M
    как ориентиры для определения стадии медвежьего рынка (пост 02.04.2026).
    """

def lth_loss_pct_of_historical_peak(loss_usd: float) -> float:
    """
    % текущего убытка от исторического пика $480M (крах FTX).
    WHY: Mozart сравнивает текущее (~$200M) с прошлым ($480M) — пост 02.04.2026.
    Используется в оркестраторе как дополнительный контекст.
    Returns: float 0–100+ (может превышать 100% при новом историческом пике).
    """
```

### Тест-файл
`tests/test_mozart_lth_loss.py`

### Граничные тесты обязательно
- `loss_usd = -140_000_001` → EARLY_2018_RANGE (abs чуть выше якоря 2018)
- `loss_usd = -300_000_000` → MID_2022_RANGE (на границе)
- `loss_usd = +250_000_000` → функция должна работать с положительным abs тоже
- `pct_of_peak(-480_000_000)` → ровно 100.0%
- `pct_of_peak(-960_000_000)` → 200.0% (новый исторический пик)

---

## ВЕТКА 3 | М-05 | LTH NUPL — Рубикон + Эйфория

### Источник паттерна
PLAN_MOZART_PATTERNS.md М-05, посты 05.04.2026 и 15.05.2026.

### Что добавить в mozart_config.py
```python
# М-05 | LTH NUPL (посты 05.04.2026, 15.05.2026)
"lth_nupl_euphoria":   0.75,  # эйфория LTH → риск распределения
"lth_nupl_rubicon":    0.0,   # граница прибыль↔убыток (рубикон)
# Примечание: Mozart явно не называет 0.75 для LTH NUPL в этих постах.
# Значение взято из нашей реализации LTH Pain Proxy и соответствует общей логике.
# При обновлении анализа — пересмотреть.
```

### Что написать в mozart_signals.py
```python
def classify_lth_nupl_regime(nupl: float) -> str:
    """
    Зоны:
      nupl >= euphoria  → 'EUPHORIA'   — риск распределения LTH
      nupl >= rubicon   → 'POSITIVE'   — LTH в нереализованной прибыли
      nupl == 0 (±eps)  → 'RUBICON'    — граница смены рынка
      nupl <  rubicon   → 'BEAR'       — LTH в убытке, давление снижено
    WHY отдельная RUBICON зона: Mozart трактует переход через 0 как событие,
    а не просто порог (пост 05.04.2026). Для отображения в оркестраторе.
    eps = 0.005 (полпроцента — шум дневных данных).
    """
```

### Тест-файл
`tests/test_mozart_lth_nupl.py`

---

## ВЕТКА 4 | М-02 | STH SOPR — Рубикон безубытка

### Источник паттерна
PLAN_MOZART_PATTERNS.md М-02, пост 16.04.2026.

### Что добавить в mozart_config.py
```python
# М-02 | STH SOPR (пост 16.04.2026)
"sth_sopr_rubicon": 1.0,  # безубыток STH = зона давления/поддержки
```

### Что написать в mozart_signals.py
```python
def classify_sth_sopr_regime(sopr: float) -> str:
    """
    Зоны:
      sopr > rubicon  → 'BULL'     — STH продают в прибыль
      sopr == rubicon → 'RUBICON'  — STH на безубытке (сопротивление / поддержка)
      sopr < rubicon  → 'BEAR'     — STH капитулируют
    eps = 0.005 для зоны RUBICON.
    WHY: в медвежьем рынке STH SOPR == 1.0 = сильное сопротивление (пост 16.04.2026).
    """
```

### Тест-файл
`tests/test_mozart_sth_sopr.py`

### Примечание
М-02 — самый простой паттерн. Можно объединить с М-04 STH MVRV в одну ветку
если время ограничено.

---

## ВЕТКА 5 | М-11 | ETF Flow — Светофор спроса

### Источник паттерна
PLAN_MOZART_PATTERNS.md М-11, пост 08.04.2026.

### Что добавить в mozart_config.py
```python
# М-11 | ETF Flow (пост 08.04.2026)
# Mozart оперирует бинарно: притоки / оттоки. Нейтральная зона — наш выбор.
# 500 BTC/день = ~$40M при $80k — порог "значимого" потока (FORMALIZED).
"etf_flow_significant_btc": 500,  # BTC/день — выше = значимый сигнал
```

### Что написать в mozart_signals.py
```python
def classify_etf_flow_regime(flow_btc: float) -> str:
    """
    Зоны:
      flow_btc >  significant  → 'INFLOW'   — институциональный спрос
      flow_btc >= -significant → 'NEUTRAL'  — нет направленного сигнала
      flow_btc <  -significant → 'OUTFLOW'  — нет нового покупателя

    WHY significant=500 BTC: Mozart использует качественно "есть / нет притоков"
    (пост 08.04.2026). Порог 500 BTC (~$40M при $80k) — FORMALIZED, отсекает шум.
    Если Mozart уточнит числовой порог — перенести в конфиг.
    """
```

### Тест-файл
`tests/test_mozart_etf_flow.py`

---

## ВЕТКА 6 | М-03 + М-04 | LTH MVRV + STH MVRV — Бинарные рубиконы

### Источник паттерна
PLAN_MOZART_PATTERNS.md М-03 и М-04, посты 25.02.2026 и 16.04.2026.

### Что добавить в mozart_config.py
```python
# М-03 | LTH MVRV (пост 25.02.2026)
"lth_mvrv_rubicon": 1.0,  # < 1.0 = LTH в убытке, давление продаж снижено

# М-04 | STH MVRV (пост 16.04.2026)
"sth_mvrv_rubicon": 1.0,  # ≈ 1.0 = STH у безубытка = нейтральная зона
```

### Что написать в mozart_signals.py
```python
def classify_lth_mvrv_regime(mvrv: float) -> str:
    """
    Бинарный рубикон 1.0 (пост 25.02.2026):
      mvrv >= rubicon → 'BULL'  — LTH в нереализованной прибыли
      mvrv <  rubicon → 'BEAR'  — LTH в убытке, продажи вынужденные
    """

def classify_sth_mvrv_regime(mvrv: float) -> str:
    """
    Рубикон 1.0 (пост 16.04.2026):
      mvrv > rubicon + eps  → 'BULL'    — давление продаж от STH
      mvrv >= rubicon - eps → 'NEUTRAL' — STH у безубытка = зона давления
      mvrv <  rubicon - eps → 'BEAR'    — STH капитулируют
    eps = 0.02 (2% — типичный дневной шум STH MVRV).
    """
```

### Тест-файл
`tests/test_mozart_mvrv.py`

---

## ВЕТКА 7 | М-06 | STH NUPL — Рубикон капитуляции

### Источник паттерна
PLAN_MOZART_PATTERNS.md М-06, пост 16.04.2026.

### Что добавить в mozart_config.py
```python
# М-06 | STH NUPL (пост 16.04.2026)
"sth_nupl_rubicon": 0.0,  # < 0 = STH когорта в убытке = капитуляция
```

### Что написать в mozart_signals.py
```python
def classify_sth_nupl_regime(nupl: float) -> str:
    """
    Зоны:
      nupl >  rubicon → 'POSITIVE'    — STH в нереализованной прибыли
      nupl >= -eps    → 'RUBICON'     — STH на нуле, максимальное давление
      nupl <  -eps    → 'CAPITULATION'— STH в убытке когортно
    eps = 0.005.
    WHY: Mozart: "запас на рост по-прежнему имеет место" при NUPL > 0
    (пост 16.04.2026). Переход в отрицательную зону = смена режима.
    """
```

### Тест-файл
`tests/test_mozart_sth_nupl.py`

---

## ВЕТКА 8 | М-07 + М-08 | Cohort Flow — Совместный переток LTH↔STH

### Источник паттерна
PLAN_MOZART_PATTERNS.md М-07 и М-08, пост 14.01.2026.

### Что добавить в mozart_config.py
```python
# М-07+М-08 | Cohort Flow (пост 14.01.2026)
# Mozart использует качественно: знак LTH и STH net position.
# Числового порога не называет — знак достаточен для классификации.
# (нет числовых констант для добавления в конфиг)
```

### Что написать в mozart_signals.py
```python
def classify_cohort_flow(lth_net_pos: float, sth_net_pos: float) -> str:
    """
    Совместная классификация перетока монет LTH↔STH (пост 14.01.2026).

    Зоны:
      lth > 0 и sth < 0  → 'ACCUMULATION'  — STH продают → LTH, бычий фон
      lth < 0 и sth > 0  → 'DISTRIBUTION'  — LTH продают → STH, медвежий фон
      lth > 0 и sth > 0  → 'BOTH_BUYING'   — оба накапливают (редко)
      lth < 0 и sth < 0  → 'BOTH_SELLING'  — оба продают (стресс)

    WHY совместно: Mozart: "LTH Net Pos + STH Net Pos должны анализироваться
    вместе — разнонаправленность подтверждает переток" (пост 14.01.2026).
    Знак нуля считается положительным (накопление).
    """
```

### Тест-файл
`tests/test_mozart_cohort_flow.py`

### Примечание
Данные из `get_lth_net_position_change_30d()` и `get_sth_net_position_change_30d()`.
Оба метода уже в клиенте. Функция принимает два числа — простейший TDD.

---

## ВЕТКА 9 | М-12 | HODL Waves — Направление когорт (delta)

### Источник паттерна
PLAN_MOZART_PATTERNS.md М-12, пост 13.05.2026.

### Сложность
Mozart не даёт числовых порогов. Логика — направление изменения когорт
(монеты стареют → age_1m_3m падает, age_3m_6m растёт = накопление).
Нужна история (минимум 2 точки).

### Что добавить в mozart_config.py
```python
# М-12 | HODL Waves (пост 13.05.2026)
# Нет числовых порогов от Mozart — используем знак изменения.
# (нет констант для конфига)
```

### Что написать в mozart_signals.py
```python
def classify_hodl_wave_regime(
    age_1m_3m_current: float,
    age_1m_3m_prev: float,
    age_3m_6m_current: float,
    age_3m_6m_prev: float,
) -> str:
    """
    Классификация по направлению внутрикогортного сдвига (пост 13.05.2026):

      age_1m_3m падает и age_3m_6m растёт → 'AGING'        — монеты стареют = накопление
      age_1m_3m растёт и age_3m_6m падает → 'REJUVENATING' — монеты молодеют = распределение
      иначе                                → 'MIXED'        — нет направленного сигнала

    WHY знак: Mozart описывает переток между когортами качественно.
    Числового порога нет — знак изменения достаточен для бинарного сигнала.
    """
```

### Тест-файл
`tests/test_mozart_hodl_waves.py`

---

## ВЕТКА 10 | М-09 | STH Realized Price — Паттерн В (Z-score turning)

### Источник паттерна
PLAN_MOZART_PATTERNS.md М-09 Паттерн В, пост 05.04.2026.

### Контекст
Паттерны А (главное сопротивление) и Б (линия снижается) уже реализованы
в оркестраторе как `[PROGNOSTIC STRESS-TEST]`. Паттерн В требует
отдельной функции детекции смены тренда по Z-score.

### Что добавить в mozart_config.py
```python
# М-09 | STH Realized Price — Z-score turning (пост 05.04.2026)
"sth_rp_zscore_turning_window": 5,  # дней для детекции смены направления
```

### Что написать в mozart_signals.py
```python
def detect_sth_rp_zscore_turning(zscore_history: list[float], window: int = 5) -> bool:
    """
    Паттерн В: Z-score STH RP начинает расти при стоячей цене.
    Аналог detect_lth_sopr_turning — применяется к Z-score,
    не к самому значению STH RP.

    True если в последних window точках min достигнут до последней точки
    и последняя точка выше min (началось восстановление).
    """
```

### Тест-файл
`tests/test_mozart_sth_realized_price.py`

---

## ПОСЛЕ ЭТИХ ВЕТОК — новые endpoints

После завершения всех 10 веток выше — диагностика и TDD для:

| ID | Endpoint | Приоритет | Метод в клиенте |
|---|---|---|---|
| МБ-01 | `realized-price` | HIGH | ❌ добавить |
| МБ-02 | `true-market-mean` | HIGH | ❌ добавить |
| МБ-04 | `supply-loss` | HIGH | ❌ добавить |
| МБ-05 | `realized-profit-lth-usd` | HIGH | ❌ добавить |
| МБ-06 | `nupl` | MEDIUM | ❌ добавить |
| МБ-07 | `mvrv-zscore` | MEDIUM | ❌ добавить |
| МБ-08 | `realized-cap-hodl-waves` | MEDIUM | ❌ добавить |

Для каждого: сначала диагностика (≤5 запросов, RULES_API_DIAGNOSTICS.md),
затем метод в клиенте, затем TDD.

---

## ШАБЛОН ENTRY POINT ДЛЯ КАЖДОЙ ВЕТКИ

```
Прочитай PLAN_MOZART_TDD_LEVEL2.md.
Текущее состояние: XXX passed, 76 skipped, 0 failed.
Задача ветки: [ВЕТКА N | М-XX | название].
Начни с добавления порогов в mozart_config.py,
затем TDD: тест → RED (pytest подтверждён) → функция → GREEN → py_compile.
```

---

## ПРАВИЛА КАЖДОЙ ВЕТКИ (без исключений)

1. Пороги — только из `mozart_config.py`, не хардкодить в тестах и сигналах
2. TDD: RED подтверждён pytest перед написанием кода
3. `edit_file`: dryRun=true перед apply, anchor ≥ 4 уникальных строки
4. PowerShell: команды по одной (не &&)
5. `py_compile` после каждого production-изменения
6. Граничные значения — отдельные тесты с WHY-комментарием
7. Смотреть PLAN_MOZART_PATTERNS.md перед каждой веткой —
   убеждаться что пороги взяты из первоисточника (поста Mozart), не придуманы
