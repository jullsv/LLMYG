#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
from PyPDF2 import PdfReader
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os
API_KEY = os.getenv("YANDEX_API_KEY", "")
FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")

API_CONFIG = {
    "url": "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
    "model_uri": f"gpt://{FOLDER_ID}/yandexgpt",
    "temperature": 0.3,
    "max_tokens": 2000,
    "delay_between_requests": 1
}


def load_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    raise ValueError("Только PDF и TXT файлы")


def _make_request(system_prompt: str, user_prompt: str) -> Optional[str]:
    prompt = {
        "modelUri": API_CONFIG["model_uri"],
        "completionOptions": {
            "stream": False,
            "temperature": API_CONFIG["temperature"],
            "maxTokens": API_CONFIG["max_tokens"]
        },
        "messages": [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": user_prompt}
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {API_KEY}"
    }
    try:
        response = requests.post(
            API_CONFIG["url"],
            headers=headers,
            json=prompt,
            timeout=30
        )
        if response.status_code != 200:
            return None
        result = response.json()
        if 'result' in result and 'alternatives' in result['result']:
            return result['result']['alternatives'][0]['message']['text'].strip()
        return None
    except requests.exceptions.RequestException:
        return None


def extract_statements(text: str) -> List[str]:
    system_prompt = (
        "Вы - аналитик текста. Найдите в тексте ключевые фактические утверждения. "
        "Выведите только утверждения, по одному в строке, без нумерации, "
        "пояснений, маркеров и дополнительного текста."
    )
    user_prompt = f"Текст для анализа:\n{text[:3000]}\n\nИзвлеки ключевые утверждения:"
    response = _make_request(system_prompt, user_prompt)
    if response:
        lines = [line.strip() for line in response.split("\n") if line.strip() and len(line.strip()) > 10]
        return lines
    return []


def extract_binary_opposition(statement: str, context: str) -> Optional[Tuple[str, str]]:
    system_prompt = (
        "Вы - эксперт по методу деконструкции. "
        "Найдите в утверждении бинарную оппозицию - пару понятий, где одно доминирует над другим. "
        "Примеры: свобода/детерминизм, разум/чувство, автономия/контроль. "
        "Выведите ответ в формате: 'понятие1/понятие2' без пояснений."
    )
    user_prompt = (
        f"Утверждение: {statement}\nКонтекст: {context[:500]}\n\n"
        f"Найдите бинарную оппозицию в формате 'доминирующее/подчинённое':"
    )
    response = _make_request(system_prompt, user_prompt)
    if response and '/' in response:
        parts = response.strip().split('/')
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return (parts[0].strip(), parts[1].strip())
    return None


def invert_hierarchy(statement: str, opposition: Tuple[str, str]) -> str:
    system_prompt = (
        "Вы — аналитик метода деконструкции. "
        "Сформулируйте утверждение, которое инвертирует бинарную оппозицию: "
        "покажите, что доминирующее понятие на самом деле зависит от подчинённого. "
        "Это не простое отрицание, а выявление скрытой взаимозависимости."
    )
    user_prompt = (
        f"Исходное утверждение: {statement}\n"
        f"Бинарная оппозиция: {opposition[0]}/{opposition[1]}\n\n"
        f"Сформулируйте деконструктивное утверждение (инверсия иерархии):"
    )
    response = _make_request(system_prompt, user_prompt)
    return response if response else f"Не {statement.lower()}"


def find_contradictory_fragments(source_text: str, inverted_assertion: str) -> List[Dict]:
    system_prompt = (
        "Вы — аналитик текста в методе деконструкции. "
        "Найдите фрагменты исходного текста, которые НЕОЖИДАННО подтверждают обратное утверждение. "
        "Это выявляет скрытые смыслы и внутренние противоречия автора. "
        "Выведите ответ в формате JSON: "
        '[{"fragment": "цитата", "explanation": "почему это подтверждает обратное"}]'
    )
    user_prompt = (
        f"ИСХОДНЫЙ ТЕКСТ:\n{source_text[:2500]}\n\n"
        f"ОБРАТНОЕ (ДЕКОНСТРУКТИВНОЕ) УТВЕРЖДЕНИЕ:\n{inverted_assertion}\n\n"
        f"Найдите фрагменты, подтверждающие это обратное утверждение:"
    )
    response = _make_request(system_prompt, user_prompt)
    if response:
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except json.JSONDecodeError:
            pass
    return []


def deconstruct_statement(statement: str, source_text: str) -> Optional[Dict]:
    opposition = extract_binary_opposition(statement, source_text)
    if not opposition:
        return None
    inverted = invert_hierarchy(statement, opposition)
    fragments = find_contradictory_fragments(source_text, inverted)
    return {
        "statement": statement,
        "opposition": {"dominant": opposition[0], "subordinate": opposition[1]},
        "inverted_assertion": inverted,
        "supporting_fragments": fragments
    }


def deconstruct_text(text: str, max_statements: int = 3) -> List[Dict]:
    statements = extract_statements(text)
    results = []
    for stmt in statements[:max_statements]:
        result = deconstruct_statement(stmt, text)
        if result:
            results.append(result)
    return results


def analyze_file(file_path: str, max_statements: int = 1) -> List[Dict]:
    text = load_text(file_path)
    return deconstruct_text(text, max_statements)


def save_results(results: List[Dict], output_path: str):
    with open(f"{output_path}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(f"{output_path}.txt", "w", encoding="utf-8") as f:
        for idx, res in enumerate(results, 1):
            f.write(f"Исходное утверждение: {res['statement']}\n")
            f.write(f"Бинарная оппозиция: {res['opposition']['dominant']} / {res['opposition']['subordinate']}\n")
            f.write(f"Инверсия иерархии: {res['inverted_assertion']}\n")
            f.write("Фрагменты, подтверждающие инверсию:\n")
            for frag in res['supporting_fragments']:
                f.write(f"  • \"{frag['fragment']}\"\n    → {frag['explanation']}\n")
            f.write("\n")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    file_path = script_dir / "soznanie-i-samosoznanie.pdf"
    results = analyze_file(str(file_path))
    save_results(results, str(script_dir / "deconstruction_results"))
    print(f"Завершено. Проанализировано утверждений: {len(results)}")