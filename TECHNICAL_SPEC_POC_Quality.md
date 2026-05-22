# Техническое задание: POC Quality Score
## Модернизация volume_density.py — адаптация HFT-логики

**Дата оригинала:** 2026-04-02 (Draft v1.0)
**Дата последнего обновления:** 2026-04-08 (сессия 14, актуализация)
**Статус:** ✅ Этапы 1–5 реализованы. Архитектура эволюционировала от score к теговой системе.

---

## Архитектурный принцип

**Incremental Changes Only.** Новые функции добавляются поверх существующих модулей.
Запрещено переписывать `liquidity_density_audit()` целиком.
Каждый компонент реализуется как отдельная чистая функция → тестируется изолированно → подключается в оркестратор.

---

## ЭТАП 1 — P1: Absorption Detection ✅ DONE

**Функция:** `detect_absorption_days(df, atr_value) → df`
**Статус:** реализована, тесты GREEN, подключена в оркестратор.

Признаки дня поглощения (все три обязательны):
- `vol > rolling_mean(20) * 1.5` — аномальный объём
- `(close - low) / (high - low) > 0.70` — закрытие в верхних 30% = бычье
- `close < close.shift(5)` — нисходящий контекст

**Интеграция:** после `apply_time_decay()`, добавляет колонку `absorption`.
Отчёт: блок `[ABSORPTION DETECTION]`, счётчик дней у POC ± 1.5 ATR.

---

## ЭТАП 2 — P1: BGeometrics on-chain эндпоинты ✅ DONE

**Реализовано:**
- `BGeometricsClient.get_realized_loss_lth_usd()` в `onchain_client.py`
- `OnChainValidator.check_capitulation_signal()` в `onchain_validator.py`

**Порог капитуляции:** $300M/день × 3 дня подряд.
**Отчёт:** блок `[ON-CHAIN: LTH REALIZED LOSS]` с баргафиком и флагом ДА/НЕТ.

---

## ЭТАП 3 — P1: POC Quality Score ✅ DONE (с изменением архитектуры)

### Исходный план (score-based)
`calculate_poc_quality_score()` — взвешенный скор [0.0–1.0], реализована и покрыта тестами.
Классификация: `score > 0.65 → FAIR_VALUE_MAGNET`, `< 0.35 → RESISTANCE_TRAP`, иначе `NEUTRAL`.

### Актуальная архитектура: теговая система
`evaluate_poc_quality()` — **заменила** `calculate_poc_quality_score()` в оркестраторе.
Вместо взвешенного score — явные теги с приоритетом:

```
FAIR_VALUE_*  теги (бычьи):
  FAIR_VALUE_MAGNET_ABSORPTION    absorption_ratio > 0.4
  FAIR_VALUE_MAGNET_CAPITULATION  capitulation_confirmed=True
  FAIR_VALUE_MAGNET_STH_PRESSURE  z_score > 1.0

RESISTANCE_*  теги (медвежьи):
  RESISTANCE_TRAP_DELTA    delta_context_score < 0.35 AND volume_w_score > 60%
  RESISTANCE_TRAP_OI       oi_regime == 'STRONG_BEAR'
  RESISTANCE_TRAP_FUNDING  funding_regime == 'POSITIVE_EXTREME'

Информационные теги (не влияют на label):
  LTH_CAPITULATION_ZONE   proxy_sopr < 0.60
  BULLISH_DIVERGENCE       oi_regime == 'LIQUIDATION' AND absorption_ratio > 0.3
```

**Правило агрегации:**
- Только FAIR_VALUE_* → `FAIR_VALUE_MAGNET`
- Только RESISTANCE_* → `RESISTANCE_TRAP`
- Конфликт или нет тегов → `NEUTRAL`

`calculate_poc_quality_score()` оставлена в коде как legacy (используется в отдельных тестах).

---

## ЭТАП 4 — P2: Open Interest ✅ DONE

**Функции:**
- `classify_oi_regime(price_change_pct, oi_change_pct) → str` — чистая функция
- `load_oi_history(csv_path, ...) → pd.DataFrame` — загрузка CSV Binance Vision
- `classify_funding_regime(funding_pct) → str` — режим ставки фандинга
- `classify_market_regime(oi_regime, funding_regime) → str` — агрегированный режим

**Матрица OI-режимов** (порог ±1%):

| Price | OI | Режим |
|---|---|---|
| ↑ >+1% | ↑ >+1% | STRONG_BULL |
| ↑ >+1% | ≤+1% | WEAK_BULL |
| ↓ <−1% | ↑ >+1% | STRONG_BEAR |
| ↓ <−1% | ≤+1% | LIQUIDATION |

**Источник данных:** исторический CSV Binance Vision (`BTCUSDT-metrics-daily.csv`),
а не `fetch_open_interest_history()` (ccxt даёт только ~30 дней).

**Влияние на теговую архитектуру:**
- `STRONG_BEAR` → тег `RESISTANCE_TRAP_OI`
- `LIQUIDATION` + поглощение → тег `BULLISH_DIVERGENCE`

**Матрица `classify_market_regime()`:**
- `LIQUIDATION` (любой funding) → `CAPITULATION`
- `STRONG_BULL` + `POSITIVE_EXTREME` → `OVERHEATED_BULL`
- `STRONG_BULL` иначе → `BULL`
- `STRONG_BEAR` + `POSITIVE_EXTREME` → `BEAR_SQUEEZE`
- `STRONG_BEAR` иначе → `BEAR`
- `WEAK_BULL` + negative funding → `BEAR`
- Всё остальное → `NEUTRAL`

**Отчёты:** блоки `[OI REGIME]`, `[FUNDING RATE REGIME]`, строка `Рыночный режим` в `[FINAL VERDICT]`.

---

## ЭТАП 5 — P3: Delta в бинах Volume Profile ✅ DONE

### Исходный план
`calculate_bin_delta()` из `fetch_trades()` — медленно, P3.

### Реализованная архитектура: гибридный горизонт (aggTrades)

**Два горизонта анализа дельты:**

| Горизонт | Источник | Длина | Цель |
|---|---|---|---|
| Anchor period | aggTrades monthly ZIP (Binance Vision) | ~4 месяца | Структурная память уровня |
| Reaction period | aggTrades daily ZIP (Binance Vision) | 14 дней | Краткосрочное давление |

**Pipeline:**
```
get_anchor_months()          → список (year, month) где цена была у POC
download_anchor_month()      → ZIP aggTrades за месяц
build_delta_cache()          → CVD slope → parquet (idempotent)
download_reaction_month()    → daily ZIP за последние 14 дней
build_reaction_delta()       → суммарная дельта за reaction period
calculate_cvd_in_zone()      → CVD + slope для зоны POC ± 1.5 ATR
calculate_delta_context_score() → sigmoid(0.6×tanh(slope/1000) + 0.4×tanh(delta/500))
```

**Дополнительно — Delta Volume Profile (klines 1m):**
```
download_klines_day()          → 1m klines ZIP (Binance Vision)
build_klines_delta_cache()     → poc_bin_delta per day → parquet
build_delta_profile()          → профиль с колонкой delta
```

**Результат:** `delta_context_score ∈ [0.0, 1.0]`
- `> 0.5` → buy_vol доминирует → бычий контекст
- `< 0.5` → sell_vol доминирует → медвежий контекст
- `= 0.5` → нейтрально

**Влияние на `evaluate_poc_quality()`:**
- `delta_context_score < 0.35` при `volume_w_score > 60%` → тег `RESISTANCE_TRAP_DELTA`

**Отчёты:** блоки `[DELTA PROFILE]` и `[DELTA CONTEXT]`.

---

## Функции вне исходного ТЗ (добавлены по ходу разработки)

| Функция | Откуда | Назначение |
|---|---|---|
| `classify_funding_regime()` | Сессия ~8 | Режим ставки фандинга Binance USDM |
| `classify_market_regime()` | Сессия ~12 | Агрегация OI + funding → единый режим |
| `calculate_lth_pain_proxy()` | Сессия ~10 | VWMA-155 прокси SOPR для LTH-когорты |
| `extract_sub_levels()` | Сессия ~9 | HVN внутри широких DBSCAN-зон |
| `get_anchor_months()` | Сессия ~11 | Отбор месяцев для aggTrades download |
| `calculate_delta_context_score()` | Сессия ~11 | Агрегация двух дельта-горизонтов |
| `load_oi_history()` | Сессия ~8 | CSV Binance Vision вместо ccxt API |
| `calculate_basis_spread()` | Сессия 14 | Спред spot vs futures (CONTANGO/BACKWARDATION/FLAT) |

---

## Актуальная карта функций (сессия 14)

| Функция | В оркестраторе | Тесты |
|---|---|---|
| `calculate_atr()` | ✅ | ✅ |
| `apply_time_decay()` | ✅ | ✅ |
| `build_profile()` | ✅ | ✅ |
| `calculate_value_area()` | ✅ | ✅ |
| `find_liquidity_clusters()` | ✅ | ✅ |
| `detect_absorption_days()` | ✅ | ✅ |
| `extract_sub_levels()` | ✅ | ✅ |
| `calculate_poc_quality_score()` | ❌ legacy | ✅ |
| `evaluate_poc_quality()` | ✅ активна | ✅ |
| `classify_oi_regime()` | ✅ | ✅ |
| `load_oi_history()` | ✅ | ✅ |
| `calculate_bin_delta()` | ❌ | ✅ |
| `classify_funding_regime()` | ✅ | ✅ |
| `classify_market_regime()` | ✅ | ✅ |
| `calculate_lth_pain_proxy()` | ✅ | ✅ |
| `get_anchor_months()` | ✅ | ✅ |
| `calculate_delta_context_score()` | ✅ | ✅ |
| `calculate_basis_spread()` | ✅ | ✅ |
| `load_aggtrades_zip()` | внутри pipeline | ✅ |
| `calculate_cvd_in_zone()` | внутри pipeline | ✅ |
| `load_klines_zip()` | внутри pipeline | ✅ |
| `build_delta_profile()` | внутри pipeline | ✅ |

**Внешние модули:** `onchain_client`, `onchain_validator`, `download_anchor_data`,
`delta_cache`, `pruning` — все реализованы.

---

## Backlog (следующие кандидаты)

- `classify_volume_type(candle)` — тип свечи: absorption / exhaustion / breakout / neutral
- `calculate_liquidation_heatmap(price_range, oi_df)` — концентрация ликвидаций по ценовым уровням
- `detect_spoofing_pattern(order_book_snapshot)` — детект крупных заявок с быстрой отменой

---

## Definition of Done (неизменно)

- [ ] Функция на уровне модуля (не вложенная)
- [ ] pytest покрывает контракт (не воспроизводит логику внутри теста)
- [ ] Тест для граничного случая (пустой df, нулевой объём, None)
- [ ] Интеграция в оркестратор не ломает существующий вывод
- [ ] NEXT_SESSION.md обновлён
