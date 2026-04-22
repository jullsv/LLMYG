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
    "model_uri": f"gpt://{FOLDER_ID}/yandexgpt/latest",
    "temperature": 0.3,
    "max_tokens": 2000,
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
    raise ValueError("только PDF и TXT файлы")


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
            print(f"Ошибка API: {response.status_code} - {response.text}")
            return None
        result = response.json()
        if 'result' in result and 'alternatives' in result['result']:
            return result['result']['alternatives'][0]['message']['text'].strip()
        return None
    except Exception as e:
        print(f"Ошибка сети: {e}")
        return None


def extract_statements(text: str) -> List[str]:
    system_prompt = (
        "Ты читаешь текст. Найди в нём самую важную мысль автора. "
        "Напиши каждую мысль отдельной строкой. "
        "Без номеров, без пояснений, просто мысли."
    )
    user_prompt = f"Текст:\n{text[:3000]}\n\nКакие главные мысли здесь есть?"
    response = _make_request(system_prompt, user_prompt)
    if response:
        return [line.strip() for line in response.split("\n") if len(line.strip()) > 10]
    return []


def extract_binary_opposition(statement: str, context: str) -> Optional[Tuple[str, str]]:
    system_prompt = (
        "Проанализируй текст, выдели в нем два понятия. "
        "Эти два понятия должны быть существенны и значимы для текста, то есть их использование должно составлять существенную часть этого текста. "
        "При этом эти два понятия  должны  либо логически, либо семантически друг другу противопоставлены. "
        "Примеры: правда/ложь, свобода/правила, ум/чувства. "
        "Напиши их через слэш: Главное/Второстепенное."
    )
    user_prompt = (
        f"Мысль автора: '{statement}'\n\n"
        f"Текст вокруг:\n{context[:800]}\n\n"
        f"Какие два слова здесь спорят? Напиши в формате Главное/Второстепенное:"
    )
    response = _make_request(system_prompt, user_prompt)
    if response and '/' in response:
        parts = response.strip().split('/')
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return (parts[0].strip(), parts[1].strip())
    return None


def invert_hierarchy(statement: str, opposition: Tuple[str, str]) -> str:
    system_prompt = (
        "Автор считает, что {A} важнее {B}. "
        "Покажи, почему {B} может быть не просто противоположностью {A}, "
        "а необходимым условием для {A}. "
        "Важно: не смешивай {B} с другими понятиями — работай только с этой парой."
    )
    formatted_sys = system_prompt.format(A=opposition[0], B=opposition[1])
    user_prompt = f"Почему {opposition[1]} необходимо для {opposition[0]}?"
    response = _make_request(formatted_sys, user_prompt)
    return response if response else f"{opposition[1]} необходимо для {opposition[0]}."


def find_contradictory_fragments(source_text: str, inverted_assertion: str) -> List[Dict]:
    system_prompt = (
        "Автор текста утверждает что-то, но его собственные слова могут показывать ограничения этого утверждения. "
        "Я дам тебе альтернативный взгляд (инверсию). "
        "Найди в тексте фразы, которые, если посмотреть внимательно, показывают: "
        "автор не учитывает что-то важное, или его позиция имеет границы. "
        "Не ищи 'подтверждения' — ищи моменты, где текст сам себя ставит под вопрос. "
        "Ответ в формате JSON: "
        '[{"fragment": "цитата", "explanation": "какое ограничение это вскрывает"}]'
    )
    user_prompt = (
        f"Текст:\n{source_text[:2500]}\n\n"
        f"Альтернативный взгляд: {inverted_assertion}\n\n"
        f"Найди фразы, показывающие ограничения исходной позиции:"
    )
    response = _make_request(system_prompt, user_prompt)
    # ... парсинг как раньше
    if response:
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
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


def analyze_file(file_path: str, max_statements: int = 1) -> List[Dict]:
    text = load_text(file_path)
    statements = extract_statements(text)
    results = []
    for stmt in statements[:max_statements]:
        res = deconstruct_statement(stmt, text)
        if res:
            results.append(res)
    return results


def save_results(results: List[Dict], output_path: str):
    with open(f"{output_path}.txt", "w", encoding="utf-8") as f:
        for res in results:
            f.write(f"Ключевое утверждение: {res['statement']}\n")
            f.write(f"Бинарные оппозиции: {res['opposition']['dominant']} : {res['opposition']['subordinate']}\n")
            f.write(f"Инверсия: {res['inverted_assertion']}\n")
            f.write("Ограничения исходной позиции:\n") 
            for frag in res['supporting_fragments']:
                f.write(f"  • \"{frag['fragment']}\"\n")
                f.write(f"    → {frag['explanation']}\n")  
            f.write("\n")

if __name__ == "__main__":
    FILENAME = "ii.txt"
    script_dir = Path(__file__).parent
    target_file = script_dir / FILENAME
    
    if target_file.exists():
        print(f"Анализ файла: {FILENAME}...")
        final_results = analyze_file(str(target_file))
        save_results(final_results, str(script_dir / "analysis_ii"))
        print(f"Результаты сохранены в analysis_ii.txt")
    else:
        print(f"Файл {FILENAME} не найден.")
