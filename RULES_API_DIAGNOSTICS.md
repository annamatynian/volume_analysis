# RULES_API_DIAGNOSTICS.md
# Правила написания диагностических скриптов для BGeometrics API
# Зафиксировано: 2026-05-18 (инцидент: 429 при диагностике nupl-sth)

---

## КОНТЕКСТ

BGeometrics API (bitcoin-data.com) — **бесплатный тариф**:
- Лимит: **10 запросов/час** (X-RateLimit-Limit-Hour: 10)
- Счётчик сбрасывается по X-RateLimit-Reset-Hour (Unix timestamp)
- **404-ответы ТОЖЕ считаются в лимит** — сервер обработал запрос

Инцидент 2026-05-18: диагностический скрипт с циклом по 11 endpoint'ам
исчерпал весь часовой лимит за один запуск. Endpoint'ы 10–12 вернули 429.

---

## ПРАВИЛА

### 1. Максимум 5 endpoint'ов за один диагностический скрипт

```python
# ❌ ПЛОХО — 11 endpoint'ов убивают весь часовой лимит
ENDPOINTS = ["ep1", "ep2", "ep3", "ep4", "ep5",
             "ep6", "ep7", "ep8", "ep9", "ep10", "ep11"]

# ✅ ХОРОШО — 5 endpoint'ов, остаток лимита сохранён
ENDPOINTS = ["ep1", "ep2", "ep3", "ep4", "ep5"]
```

### 2. Проверять X-RateLimit-Remaining-Hour перед следующим запросом

```python
r = session.get(url, timeout=10)
remaining = int(r.headers.get('X-RateLimit-Remaining-Hour', 99))
print(f"Remaining: {remaining}/hour")
if remaining <= 2:
    print("СТОП: лимит почти исчерпан — прерываем цикл")
    break
```

### 3. Разбивать диагностику на тематические группы, не один большой скрипт

```
# Вместо одного скрипта на 11 endpoint'ов — два скрипта по 5:
_diag_group_A.py  → nupl-sth, sth-nupl, utxos-in-profit-pct, sth-mvrv, sth-realized-price
_diag_group_B.py  → realized-loss-lth-usd, lth-realized-loss, realized-loss, ...
# Запускать с перерывом 1 час между группами.
```

### 4. Никакого "детального блока" после основного цикла

Скрипт выше делал повторный запрос к nupl-sth в "детальном блоке" —
это дополнительный запрос сверх цикла. Итого: 12 запросов вместо 11.

```python
# ❌ ПЛОХО — двойной запрос к одному endpoint'у
for ep in ENDPOINTS:
    r = session.get(f"{BASE}/{ep}")
    ...

print(">>> Детально: nupl-sth")
r = session.get(f"{BASE}/nupl-sth")   # ← уже был в цикле!
```

### 5. Сохранять raw-ответ в файл, не делать повторный запрос

Если нужен детальный разбор конкретного endpoint'а — сохрани тело
первого ответа и разбирай его локально:

```python
responses = {}
for ep in ENDPOINTS:
    r = session.get(f"{BASE}/{ep}", timeout=10)
    responses[ep] = {"status": r.status_code, "body": r.text}

# Детальный разбор — из уже полученных данных, без нового запроса
import json
body = responses["nupl-sth"]["body"]
parsed = json.loads(body)
print(f"Длина: {len(parsed)}, первый: {parsed[0]}, последний: {parsed[-1]}")
```

### 6. Smoke-скрипты удалять сразу после использования

```powershell
# Сразу после запуска:
del _diag_nupl_sth.py
```

Правило уже есть в `Правила работы и написания тестов.txt` —
здесь продублировано для контекста API-диагностики.

---

## ШАБЛОН ПРАВИЛЬНОГО ДИАГНОСТИЧЕСКОГО СКРИПТА

```python
# _diag_ТЕМА.py — удалить после: del _diag_ТЕМА.py
# Макс. 5 endpoint'ов. Сохраняем ответы, не делаем повторных запросов.

import requests, json

BASE = "https://bitcoin-data.com/api/v1"
session = requests.Session()

ENDPOINTS = [
    "ep-one",
    "ep-two",
    "ep-three",
    # не более 5
]

responses = {}
for ep in ENDPOINTS:
    r = session.get(f"{BASE}/{ep}", timeout=10)
    remaining = int(r.headers.get('X-RateLimit-Remaining-Hour', 99))
    responses[ep] = {"status": r.status_code, "body": r.text, "remaining": remaining}
    print(f"{r.status_code} | remaining={remaining} | {ep} | {r.text[:120]}")
    if remaining <= 2:
        print("СТОП: лимит почти исчерпан")
        break

# Детальный разбор — из уже полученных данных
for ep, data in responses.items():
    if data["status"] == 200:
        try:
            parsed = json.loads(data["body"])
            print(f"\n{ep}: тип={type(parsed).__name__}, "
                  f"{'длина='+str(len(parsed)) if isinstance(parsed, list) else 'ключи='+str(list(parsed.keys()))}")
            if isinstance(parsed, list) and parsed:
                print(f"  первый: {parsed[0]}")
                print(f"  последний: {parsed[-1]}")
        except Exception as e:
            print(f"\n{ep}: не JSON — {e}")
```

---

## БЫСТРАЯ СПРАВКА

| Заголовок ответа | Смысл |
|---|---|
| `X-RateLimit-Limit-Hour` | Лимит запросов в час (= 10 на free tier) |
| `X-RateLimit-Remaining-Hour` | Сколько запросов осталось в текущем часу |
| `X-RateLimit-Reset-Hour` | Unix timestamp сброса счётчика |

Если получили 429 — подождать до Reset-Hour и продолжить.
Конвертация: `datetime.fromtimestamp(reset_hour_value)`.
