# PLAN: On-Chain Expansion — следующая ветка разработки

**Дата составления:** 2026-05-10  
**Состояние на момент планирования:** 285 passed, 76 skipped, 0 failed  
**Источник данных:** BGeometrics API (bitcoin-data.com/api/v1) — 318 эндпоинтов  
**Лимит free tier:** 10 запросов/час  
**Кэш:** 23-часовой parquet в `data/onchain_cache/`

---

## Текущий [HOLDER STRUCTURE] — что уже работает

| Эндпоинт | Колонка | Последнее значение | Статус |
|---|---|---|---|
| `lth-mvrv` | `lth_mvrv` | 1.73 | ✅ кэш + fallback |
| `sth-mvrv` | `sth_mvrv` | 1.04 | ✅ кэш + fallback |
| `lth-sopr` | `lth_sopr` | 1.79 | ✅ кэш + fallback |
| `lth-net-position-change-30d-btc` | `lth_net_position_30d` | +1,105,844 BTC | ✅ кэш + fallback |
| `sth-net-position-change-30d-btc` | `sth_net_position_30d` | −970,781 BTC | ✅ кэш + fallback |

**Что отсутствует из holder-картины:** STH SOPR, MTH-когорта (1–6 мес)

---

## ЗАДАЧИ — по приоритету

---

### ПРИОРИТЕТ 1 — `sth-sopr` (замыкает SOPR-картину)

**Почему:** у нас есть LTH SOPR = 1.79 (LTH продают в прибыль), но нет STH SOPR.
Без него непонятно: краткосрочники капитулируют или тоже продают в плюс?
Пара LTH SOPR + STH SOPR — стандартная аналитическая связка.

**Технические задачи:**

1. **`onchain_client.py`** — добавить метод:
   ```python
   async def get_sth_sopr(self, start_date=None, end_date=None) -> pd.DataFrame:
       """STH SOPR. Returns DataFrame [date, sth_sopr]."""
       return await self._fetch_timeseries('sth-sopr', 'sth_sopr', start_date, end_date)
   ```
   → Проверить JSON структуру эндпоинта перед написанием теста.  
   → Запустить `debug_holder_api.py` или временный скрипт: `GET /v1/sth-sopr`

2. **`onchain_cache.py`** — добавить метод `get_sth_sopr()` с metric-ключом `'sth-sopr'`

3. **`tests/test_onchain_client.py`** — добавить тест:
   - `test_get_sth_sopr_returns_correct_columns` — проверить `['date', 'sth_sopr']`

4. **`tests/test_onchain_cache.py`** — добавить класс `TestSthSoprCache`:
   - `test_missing_cache_calls_api_and_saves`
   - `test_api_error_falls_back_to_stale_cache`

5. **`volume_density.py`** — добавить в блок `[HOLDER STRUCTURE]`:
   ```python
   _sth_sopr_df = await _hc.get_sth_sopr(start_date=_h_start, end_date=_h_end)
   _sth_sopr_v  = _hlast(_sth_sopr_df, 'sth_sopr')
   ```
   И в print-блок:
   ```
   STH SOPR        : {_hfmt(_sth_sopr_v)}  (>1.0 = STH продают в прибыль; <1.0 = капитуляция STH)
   ```

**TDD-цикл:** тест → RED (подтверждён пользователем) → код → GREEN  
**Запрос к API считается:** 1 req (сохранится в кэш на 23ч)

---

### ПРИОРИТЕТ 2 — `exchange-netflow-btc` (давление продаж на POC)

**Почему:** приток BTC на биржи = продавцы несут монеты → давление на цену.
Отток = накопление вне бирж. Прямая связь с Volume Profile: если POC совпадает
с периодом биржевого оттока — это усиливает сигнал накопления.

**Технические задачи:**

1. Проверить JSON структуру: `GET /v1/exchange-netflow-btc` (ожидаем `[d, unixTs, значение]`)

2. **`onchain_client.py`** — добавить метод:
   ```python
   async def get_exchange_netflow(self, ...) -> pd.DataFrame:
       """Exchange Net Flow (BTC). Positive = inflow, Negative = outflow."""
       return await self._fetch_timeseries('exchange-netflow-btc', 'exchange_netflow_btc', ...)
   ```

3. **`onchain_cache.py`** — добавить `get_exchange_netflow()`, metric-ключ `'exchange-netflow-btc'`

4. **Тесты** — аналогично п.4 из Приоритета 1

5. **`volume_density.py`** — новый блок вывода `[EXCHANGE FLOWS]` после `[HOLDER STRUCTURE]`:
   ```
   Exchange Netflow 30d : {значение} BTC  (+ = приток на биржи, − = отток)
   ```

6. **Опционально:** рассмотреть добавление сигнала в `evaluate_poc_quality()`:
   - `exchange_netflow < -X` при `volume_w_score > 60%` → тег `FAIR_VALUE_MAGNET_OUTFLOW`
   - Только если порог объективно выводится из данных (не произвольный)

**Запрос к API считается:** 1 req

---

### ПРИОРИТЕТ 3 — `nupl-lth` + `nupl-sth` (нереализованная P/L по когортам)

**Почему:** NUPL (Net Unrealized Profit/Loss) точнее MVRV — учитывает реальные
позиции, а не просто среднюю цену. LTH NUPL < 0 = LTH в убытке (медвежий рынок),
LTH NUPL > 0.75 = эйфория (риск разворота).

**Технические задачи:**

1. Проверить JSON структуру обоих эндпоинтов

2. **`onchain_client.py`** — добавить `get_nupl_lth()` и `get_nupl_sth()`

3. **`onchain_cache.py`** — добавить оба метода, уникальные metric-ключи

4. **Тесты** — клиент + кэш для каждого метода

5. **`volume_density.py`** — добавить в `[HOLDER STRUCTURE]`:
   ```
   LTH NUPL        : {значение}  (> 0 = прибыль, < 0 = убыток, > 0.75 = эйфория)
   STH NUPL        : {значение}  (< 0 = капитуляция STH)
   ```

**Запрос к API считается:** 2 req

---

### ПРИОРИТЕТ 4 — `etf-flow-btc` (институциональные потоки)

**Почему:** с 2024 года BTC ETF — значимый игрок. Дни с сильным ETF-оттоком
совпадают с продажами на рынке. Приток в ETF при тестировании POC = институциональная поддержка.

**Технические задачи:**

1. Проверить JSON структуру: `GET /v1/etf-flow-btc`

2. **`onchain_client.py`** — добавить `get_etf_flow()`

3. **`onchain_cache.py`** — добавить `get_etf_flow()`, metric-ключ `'etf-flow-btc'`

4. **Тесты** — клиент + кэш

5. **`volume_density.py`** — добавить в `[EXCHANGE FLOWS]` (создан в Приоритете 2):
   ```
   ETF Flow (день)  : {значение} BTC  (+ = приток в ETF, − = отток из ETF)
   ```

**Запрос к API считается:** 1 req

---

### ПРИОРИТЕТ 5 — `hodl-waves-supply` (MTH-покрытие)

**Почему:** единственный источник данных по medium-term holders (1–6 мес).
Позволяет видеть где монеты концентрируются по возрасту — прямая связь с POC.

**Особенность:** этот эндпоинт скорее всего возвращает несколько колонок
(одна на каждый возрастной диапазон). Нужна предварительная диагностика JSON.

**Технические задачи:**

1. **Диагностика первая:** `GET /v1/hodl-waves-supply` — посмотреть все колонки

2. После диагностики — решить какие диапазоны брать:
   - `1m-3m` (1–3 месяца) — краткий MTH
   - `3m-6m` (3–6 месяцев) — основной MTH
   - `6m-12m` (6–12 месяцев) — длинный MTH, граница LTH

3. **`onchain_client.py`** — `get_hodl_waves()`

4. **`onchain_cache.py`** — `get_hodl_waves()`, metric-ключ `'hodl-waves-supply'`

5. **Тесты** — клиент + кэш

6. **`volume_density.py`** — новый блок `[HODL WAVES]` или расширение `[HOLDER STRUCTURE]`

**Запрос к API считается:** 1 req  
**Риск:** JSON структура неизвестна, может быть сложной — оценить после диагностики

---

## Бюджет API запросов

| Приоритет | Эндпоинт | req при холодном старте |
|---|---|---|
| Текущие (5 штук) | уже кэшированы | 0 |
| P1: sth-sopr | 1 | 1 |
| P2: exchange-netflow-btc | 1 | 1 |
| P3: nupl-lth + nupl-sth | 2 | 2 |
| P4: etf-flow-btc | 1 | 1 |
| P5: hodl-waves-supply | 1 | 1 |
| **Итого новых** | | **6 req** |
| **Итого всего** | | **6 req (кэш 23ч)** |

Лимит 10 req/hour — **6 новых укладываются** при первом запуске.  
После первого запуска все кэшируются на 23 часа — повторные запуски: 0 req.

---

## Обязательная диагностика перед каждым приоритетом

Перед написанием теста — проверить JSON структуру эндпоинта:

```python
# Шаблон диагностического скрипта
import requests
r = requests.get('https://bitcoin-data.com/api/v1/<endpoint>')
import json; data = r.json()
print(type(data), len(data) if isinstance(data, list) else '')
print(data[0] if isinstance(data, list) else list(data.keys()))
```

Это критично: MVRV-эндпоинты имели `[d, unixTs, значение]` — 3 поля.
Новые могут иметь другую структуру. Без диагностики — невалидный RED тест.

---

## Архитектурные принципы (не менять)

- Все новые методы через `_fetch_timeseries()` в `onchain_client.py`
- Каждый метод — уникальный metric-ключ в кэше (нет коллизий parquet)
- `CachedBGeometricsClient` передаётся в `OnChainValidator` — не raw client
- Оркестратор: только описательные комментарии, без интерпретаций
- Все пороговые суждения → `@pytest.mark.skip` или за пределами оркестратора

---

## Правила сессии (стандартные)

1. Читать NEXT_SESSION + Правила работы и написания тестов.txt
2. TDD: тест → RED (pytest подтверждён) → код → GREEN
3. `edit_file`: dryRun=true перед apply; anchor ≥ 4 уникальных строки
4. PowerShell: команды по одной (не `&&`)
5. `py_compile` после каждого изменения production-кода
6. Статистически верифицировать данные перед написанием теста (для числовых контрактов)
