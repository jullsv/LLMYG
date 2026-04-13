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
        "Найдите в тексте ключевые фактические утверждения. "
        "Выведите только утверждения, по одному в строке, без нумерации, "
        "пояснений, маркеров и дополнительного текста."
    )
    user_prompt = f"Текст для чтения:\n{text[:3000]}\n\nО чем самом главном говорит автор?"
    response = _make_request(system_prompt, user_prompt)
    if response:
        return [line.strip() for line in response.split("\n") if len(line.strip()) > 10]
    return []

def extract_binary_opposition(statement: str, context: str) -> Optional[Tuple[str, str]]:
    system_prompt = (
        "Проанализируй текст и выдели в нем два ключевых понятия. Эти понятия должны быть фундаментом текста — на них должна держаться вся логика автора."
        "Важно: эти понятия должны быть противопоставлены друг другу (логически или по смыслу), как день и ночь или порядок и хаос. "
        "Одно из них автор считает главным, а второе — второстепенным."
        "Примеры: истина/ложь, природа/культура, оригинал/копия."
        "Выведи только пару в формате: 'Главное/Второстепенное'."
    )
    user_prompt = (
        f"ТЕКСТ:\n{context[:1000]}\n\n"
         f"На основе этого текста найди главную оппозицию (противопоставление) для утверждения: '{statement}'"
    )
    response = _make_request(system_prompt, user_prompt)
    if response and '/' in response:
        parts = response.split('/')
        if len(parts) == 2:
            return (parts[0].strip(), parts[1].strip())
    return None

def invert_hierarchy(statement: str, opposition: Tuple[str, str]) -> str:
    system_prompt = (
        "У нас есть два слова: {A} и {B}. Автор текста думает, что {A} — это самый главный босс. "
        "Давай докажем, что автор ошибается! Напиши одно умное предложение о том, почему на самом деле {B} важнее, "
        "и почему без {B} твой главный босс {A} вообще не сможет существовать."
    )
    # Форматируем промпт внутри функции
    formatted_sys = system_prompt.format(A=opposition[0], B=opposition[1])
    user_prompt = f"Почему {opposition[1]} важнее, чем {opposition[0]}?"
    response = _make_request(formatted_sys, user_prompt)
    return response if response else f"На самом деле {opposition[1]} важнее."

def find_contradictory_fragments(source_text: str, inverted_assertion: str) -> List[Dict]:
    system_prompt = (
        "Автор текста очень хочет, чтобы мы верили в его главную мысль. "
        "Но он мог случайно проговориться. "
        "Я дам тебе предложение (которое спорит с автором). "
        "Найди в тексте такие кусочки или фразы, которые (если на них посмотреть хитро) подтверждают не автора, а наше спорное предложение. "
        "Выдай ответ в формате JSON: [{'fragment': 'цитата', 'explanation': 'почему это секретное доказательство'}]"
    )
    user_prompt = f"Текст:\n{source_text[:3000]}\n\n Секретная Мысль: {inverted_assertion}"
    response = _make_request(system_prompt, user_prompt)
    if response:
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != 0:
                return json.loads(response[start:end])
        except:
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
            f.write(f"Утверждение ключевое: {res['statement']}\n")
            f.write(f"Бинарные оппозиции {res['opposition']['dominant']} (главный) vs {res['opposition']['subordinate']} (скрытый)\n")
            f.write(f"Инверсия: {res['inverted_assertion']}\n")
            f.write("Доказательства:\n")
            for frag in res['supporting_fragments']:
                f.write(f"   - Цитата: \"{frag['fragment']}\"\n")
                f.write(f"     Почему это важно: {frag['explanation']}\n")
            f.write("-" * 30 + "\n")

if __name__ == "__main__":
    FILENAME = "mw.pdf" 
    script_dir = Path(__file__).parent
    target_file = script_dir / FILENAME
    
    if target_file.exists():
        print(f" деконструкциz файла: {FILENAME}...")
        final_results = analyze_file(str(target_file))
        save_results(final_results, str(script_dir / "final_analysis"))
        print(f"Результаты в файле final_analysis.txt")
    else:
        print(f"Файл {FILENAME} не найден.")
