# mozart_agent.py
# Интерактивный аналитический агент по методологии Mozart.
# Читает последний лог оркестратора, отвечает на вопросы в диалоговом режиме.
#
# Запуск:
#   .\venv\Scripts\python.exe mozart_agent.py           # Gemini 2.5 Flash (по умолчанию)
#   .\venv\Scripts\python.exe mozart_agent.py --claude  # Claude API

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

RUNS_DIR = Path(__file__).parent / "runs"
SKILL_PATH = Path(__file__).parent / "docs" / "SKILL_MOZART_ANALYSIS.md"

# Паттерн стандартных лог-файлов: run_YYYY-MM-DD_HH-MM.txt
_LOG_PATTERN = re.compile(r"^run_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}\.txt$")


# ---------------------------------------------------------------------------
# find_latest_log
# ---------------------------------------------------------------------------

def find_latest_log(runs_dir: str | Path) -> Path | None:
    """Находит последний стандартный лог-файл в директории runs/.

    Стандартный формат: run_YYYY-MM-DD_HH-MM.txt
    Нестандартные файлы (run_alldays_ref_*, run_clusters_* и т.д.) игнорируются.

    Args:
        runs_dir: путь к директории с логами (str или Path).

    Returns:
        Path к последнему файлу или None если файлов нет.
    """
    runs_dir = Path(runs_dir)
    candidates = [
        f for f in runs_dir.iterdir()
        if f.is_file() and _LOG_PATTERN.match(f.name)
    ]
    if not candidates:
        return None
    # Лексикографическая сортировка по имени = хронологическая (YYYY-MM-DD_HH-MM)
    return max(candidates, key=lambda f: f.name)


# ---------------------------------------------------------------------------
# parse_log_context
# ---------------------------------------------------------------------------

def parse_log_context(text: str) -> dict:
    """Извлекает ключевые факты из текста лога оркестратора.

    Парсит только то что явно написано в логе — никаких вычислений или предположений.

    Args:
        text: полный текст лога оркестратора.

    Returns:
        dict с ключами:
            price         : int | None   — текущая цена BTC
            verdict       : str | None   — BEARISH / BULLISH / MIXED / NEUTRAL
            score         : int | None   — числовой счёт alignment
            bullish_count : int | None   — число бычьих сигналов
            bearish_count : int | None   — число медвежьих сигналов
            missing_count : int | None   — число Н/Д сигналов
    """
    ctx: dict = {
        "price": None,
        "verdict": None,
        "score": None,
        "bullish_count": None,
        "bearish_count": None,
        "missing_count": None,
    }

    # Цена BTC: "Цена сейчас: $64,148" или "Текущая цена: $64,148"
    price_match = re.search(
        r"(?:Цена сейчас|Текущая цена)\s*:\s*\$([0-9,]+)",
        text,
    )
    if price_match:
        ctx["price"] = int(price_match.group(1).replace(",", ""))

    # Вердикт: "Вердикт  : 🔴  BEARISH" или "🟢  BULLISH" и т.д.
    verdict_match = re.search(
        r"Вердикт\s*:\s*[^\w]*(BEARISH|BULLISH|MIXED|NEUTRAL)",
        text,
    )
    if verdict_match:
        ctx["verdict"] = verdict_match.group(1)

    # Счёт: "Счёт     : -4" или "+7"
    score_match = re.search(r"Счёт\s*:\s*([+-]?\d+)", text)
    if score_match:
        ctx["score"] = int(score_match.group(1))

    # Бычьих: "Бычьих   :  3"
    bull_match = re.search(r"Бычьих\s*:\s*(\d+)", text)
    if bull_match:
        ctx["bullish_count"] = int(bull_match.group(1))

    # Медвежьих: "Медвежьих:  7"
    bear_match = re.search(r"Медвежьих\s*:\s*(\d+)", text)
    if bear_match:
        ctx["bearish_count"] = int(bear_match.group(1))

    # Н/Д: "Н/Д      :  3"
    missing_match = re.search(r"Н/Д\s*:\s*(\d+)", text)
    if missing_match:
        ctx["missing_count"] = int(missing_match.group(1))

    return ctx


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

def build_system_prompt(log_text: str, skill_text: str) -> str:
    """Строит системный промпт для LLM-агента.

    Содержит:
    - роль и ограничения агента
    - полный текст лога оркестратора (единственный источник фактов)
    - скилл SKILL_MOZART_ANALYSIS.md (правила интерпретации Mozart)
    - явный запрет на галлюцинации и торговые рекомендации

    Args:
        log_text:   полный текст последнего лога оркестратора.
        skill_text: содержимое docs/SKILL_MOZART_ANALYSIS.md.

    Returns:
        str — готовый системный промпт.
    """
    return f"""Ты — аналитический агент по методологии Mozart.
Твоя роль: разъяснять метрики и сигналы из вывода оркестратора на основе постов Mozart.

═══════════════════════════════════════════════════════════════
СТРОГИЕ ПРАВИЛА (нарушение недопустимо):

1. ТОЛЬКО ФАКТЫ ИЗ ЛОГА: называй только числа и значения которые явно присутствуют
   в тексте лога оркестратора ниже. Если числа нет в логе — говори прямо:
   «этих данных нет в текущем логе».

2. БЕЗ ТОРГОВЫХ РЕКОМЕНДАЦИЙ: никаких советов купить/продать/держать позицию.
   Совокупность сигналов описывай фактически, не как призыв к действию.

3. ПЕРВОИСТОЧНИК: при интерпретации сигнала называй пост Mozart и его дату.
   Если Mozart этого не определял — говори прямо: «Mozart этого не определял».

4. БЕЗ ДОМЫСЛОВ: не интерпретируй что Mozart «имел в виду» — только то что написано явно.
   Не применяй внешние знания о крипторынке вне системы Mozart.

5. ЗАТУХАНИЕ СИГНАЛА: при ссылке на исторические пороги всегда упоминай —
   каждый цикл пиковые значения метрик ~на 15–20% меньше предыдущего.
═══════════════════════════════════════════════════════════════

МЕТОДОЛОГИЯ MOZART (правила интерпретации сигналов):
{skill_text}

═══════════════════════════════════════════════════════════════
ТЕКУЩИЙ ЛОГ ОРКЕСТРАТОРА (единственный источник фактов для этой сессии):
═══════════════════════════════════════════════════════════════
{log_text}
═══════════════════════════════════════════════════════════════
"""


# ---------------------------------------------------------------------------
# LLM-провайдеры
# ---------------------------------------------------------------------------

def _make_gemini_client():
    """Инициализирует Gemini 2.5 Flash клиент."""
    from google import genai
    from google.genai import types
    from dotenv import load_dotenv
    import os
    load_dotenv()
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return client, types


def _ask_gemini(client_tuple, system_prompt: str, history: list[dict]) -> str:
    """Отправляет запрос к Gemini 2.5 Flash."""
    client, types = client_tuple
    # Собираем содержимое: системный промпт + история диалога
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            max_output_tokens=1500,
        ),
        contents=contents,
    )
    return response.text


def _make_claude_client():
    """Инициализирует Anthropic Claude клиент."""
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()
    return anthropic.Anthropic()


def _ask_claude(client, system_prompt: str, history: list[dict]) -> str:
    """Отправляет запрос к Claude API."""
    messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
    ]
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1500,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Диалоговый цикл
# ---------------------------------------------------------------------------

def run_agent(use_claude: bool = False) -> None:
    """Запускает интерактивный диалоговый агент."""

    # 1. Найти последний лог
    log_path = find_latest_log(RUNS_DIR)
    if log_path is None:
        print("[ОШИБКА] Логи не найдены в директории runs/")
        print("Запусти сначала: .\\venv\\Scripts\\python.exe -W ignore volume_density.py")
        sys.exit(1)

    log_text = log_path.read_text(encoding="utf-8")
    ctx = parse_log_context(log_text)

    # 2. Загрузить скилл
    if not SKILL_PATH.exists():
        print(f"[ОШИБКА] Файл скилла не найден: {SKILL_PATH}")
        sys.exit(1)
    skill_text = SKILL_PATH.read_text(encoding="utf-8")

    # 3. Системный промпт
    system_prompt = build_system_prompt(log_text, skill_text)

    # 4. Инициализация LLM
    provider = "Claude API" if use_claude else "Gemini 2.5 Flash"
    if use_claude:
        llm_client = _make_claude_client()
        ask_fn = lambda history: _ask_claude(llm_client, system_prompt, history)
    else:
        llm_client = _make_gemini_client()
        ask_fn = lambda history: _ask_gemini(llm_client, system_prompt, history)

    # 5. Приветствие
    price_str = f"${ctx['price']:,}" if ctx["price"] else "н/д"
    verdict_str = ctx["verdict"] or "н/д"
    score_str = str(ctx["score"]) if ctx["score"] is not None else "н/д"

    print()
    print("=" * 70)
    print("  MOZART ANALYSIS AGENT")
    print(f"  Провайдер : {provider}")
    print(f"  Лог       : {log_path.name}")
    print(f"  BTC       : {price_str}")
    print(f"  Вердикт   : {verdict_str}  (счёт: {score_str})")
    print("=" * 70)
    print("  Задавай вопросы по метрикам и сигналам Mozart.")
    print("  Команды: 'exit' или 'quit' — выход | 'лог' — показать имя лога")
    print("=" * 70)
    print()

    # 6. Диалоговый цикл
    history: list[dict] = []

    while True:
        try:
            user_input = input("Ты: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "выход"):
            print("Выход.")
            break

        if user_input.lower() == "лог":
            print(f"[Лог: {log_path}]")
            continue

        history.append({"role": "user", "content": user_input})

        print("\nАгент: ", end="", flush=True)
        try:
            answer = ask_fn(history)
        except Exception as exc:
            print(f"[ОШИБКА LLM: {exc}]")
            history.pop()  # убираем вопрос без ответа из истории
            continue

        print(answer)
        print()
        history.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mozart Analysis Agent")
    parser.add_argument(
        "--claude",
        action="store_true",
        help="Использовать Claude API вместо Gemini 2.5 Flash",
    )
    args = parser.parse_args()
    run_agent(use_claude=args.claude)
