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
    "max_tokens": 8000,
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


def extract_core_argument(text: str) -> str:
    system_prompt = "Сформулируй главную идею этого текста в одном предложении. Без пояснений."
    user_prompt = f"Текст:\n{text[:4000]}\n\nГлавная идея:"
    return _make_request(system_prompt, user_prompt) or ""


def extract_statements(text: str, core_argument: str = "") -> List[str]:
    if core_argument:
        system_prompt = (
            "Найди 2-3 ключевых утверждения, которые раскрывают главную идею текста. "
            "Напиши каждое утверждение отдельной строкой. Без номеров и пояснений."
        )
        user_prompt = f"Главная идея: {core_argument}\n\nТекст:\n{text[:3000]}\n\nКлючевые утверждения:"
    else:
        system_prompt = (
            "Найди в тексте 2-3 самые важные мысли автора. "
            "Напиши каждую мысль отдельной строкой. Без номеров и пояснений."
        )
        user_prompt = f"Текст:\n{text[:3000]}\n\nГлавные мысли:"
    response = _make_request(system_prompt, user_prompt)
    if response:
        return [line.strip() for line in response.split("\n") if len(line.strip()) > 10]
    return []


def extract_binary_opposition(statement: str, context: str, core_argument: str = "") -> Optional[Tuple[str, str]]:
    if core_argument:
        system_prompt = (
            "В тексте есть два понятия, которые спорят друг с другом. "
            "Одно автор считает главным, второе — второстепенным. "
            "Найди эту пару в контексте главной идеи текста. "
            "Примеры: истина/ложь, сознание/имитация, понимание/правила. "
            "Напиши через слэш: Главное/Второстепенное."
        )
        user_prompt = (
            f"Главная идея: {core_argument}\n\n"
            f"Утверждение: '{statement}'\n\n"
            f"Контекст:\n{context[:800]}\n\n"
            f"Найди оппозицию в формате Главное/Второстепенное:"
        )
    else:
        system_prompt = (
            "В тексте есть два слова, которые спорят друг с другом. "
            "Одно слово автор любит больше, второе — меньше. "
            "Найди эту пару слов. "
            "Примеры: правда/ложь, свобода/правила, ум/чувства. "
            "Напиши их через слэш: Главное/Второстепенное."
        )
        user_prompt = (
            f"Утверждение: '{statement}'\n\n"
            f"Контекст:\n{context[:800]}\n\n"
            f"Какие два слова здесь спорят? Напиши в формате Главное/Второстепенное:"
        )
    response = _make_request(system_prompt, user_prompt)
    if response and '/' in response:
        parts = response.strip().split('/')
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return (parts[0].strip(), parts[1].strip())
    return None


def invert_hierarchy(statement: str, opposition: Tuple[str, str], core_argument: str = "") -> str:
    if core_argument:
        system_prompt = (
            "Автор утверждает: {core}. "
            "Он считает, что {A} важнее {B}. "
            "Покажи, почему само это различение может быть не онтологической границей, "
            "а культурной или концептуальной установкой, которую можно поставить под вопрос. "
            "Не соглашайся с автором — найди точку, где его аргумент опирается на неочевидное допущение."
        )
        formatted = system_prompt.format(A=opposition[0], B=opposition[1], core=core_argument)
        user_prompt = f"Какое допущение в аргументе автора можно поставить под вопрос?"
    else:
        system_prompt = (
            "Автор считает, что {A} важнее {B}. "
            "Покажи, почему {B} может быть не просто противоположностью {A}, "
            "а необходимым условием для {A}. "
            "Важно: не смешивай {B} с другими понятиями — работай только с этой парой."
        )
        formatted = system_prompt.format(A=opposition[0], B=opposition[1])
        user_prompt = f"Почему {opposition[1]} необходимо для {opposition[0]}?"
    response = _make_request(formatted, user_prompt)
    return response if response else f"Различение {opposition[0]}/{opposition[1]} может быть условным"


def find_contradictory_fragments(source_text: str, inverted_assertion: str, core_argument: str = "") -> List[Dict]:
    if core_argument:
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
            f"Главная идея автора: {core_argument}\n\n"
            f"Текст:\n{source_text[:2500]}\n\n"
            f"Альтернативный взгляд: {inverted_assertion}\n\n"
            f"Найди фразы, показывающие ограничения исходной позиции:"
        )
    else:
        system_prompt = (
            "Автор написал текст и хочет, чтобы мы поверили в одну мысль. "
            "Но иногда автор сам пишет такие фразы, которые (если посмотреть внимательно) говорят обратное. "
            "Я дам тебе 'хитрую мысль'. Найди в тексте фразы, которые её как бы подтверждают. "
            "Ответ дай в формате JSON: "
            '[{"fragment": "цитата из текста", "explanation": "почему это подтверждает хитрую мысль"}]'
        )
        user_prompt = (
            f"Текст:\n{source_text[:2500]}\n\n"
            f"Хитрая мысль: {inverted_assertion}\n\n"
            f"Найди фразы в тексте, которые её подтверждают:"
        )
    response = _make_request(system_prompt, user_prompt)
    if response:
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
    return []


def validate_fragment(fragment: str, inverted_assertion: str, full_text: str) -> str:
    system_prompt = (
        "Проверь, действительно ли эта цитата подтверждает альтернативный взгляд. "
        "Если связь слабая — предложи более релевантный фрагмент из текста. "
        "Ответь кратко: 'подтверждает' / 'слабая связь' / 'не подтверждает'."
    )
    user_prompt = (
        f"Цитата: {fragment}\n"
        f"Альтернативный взгляд: {inverted_assertion}\n"
        f"Полный текст: {full_text[:4000]}\n\n"
        f"Оцени связь:"
    )
    return _make_request(system_prompt, user_prompt) or "не подтверждает"


def deconstruct_statement(statement: str, source_text: str, core_argument: str = "") -> Optional[Dict]:
    opposition = extract_binary_opposition(statement, source_text, core_argument)
    if not opposition:
        return None
    inverted = invert_hierarchy(statement, opposition, core_argument)
    fragments = find_contradictory_fragments(source_text, inverted, core_argument)
    validated_fragments = []
    for frag in fragments[:3]:
        validation = validate_fragment(frag['fragment'], inverted, source_text)
        if validation == "подтверждает":
            validated_fragments.append(frag)
    if not validated_fragments:
        validated_fragments = fragments[:2]
    return {
        "statement": statement,
        "core_argument": core_argument,
        "opposition": {"dominant": opposition[0], "subordinate": opposition[1]},
        "inverted_assertion": inverted,
        "supporting_fragments": validated_fragments
    }


def analyze_file(file_path: str, max_statements: int = 2) -> List[Dict]:
    text = load_text(file_path)
    core_argument = extract_core_argument(text)
    statements = extract_statements(text, core_argument)
    results = []
    for stmt in statements[:max_statements]:
        res = deconstruct_statement(stmt, text, core_argument)
        if res:
            results.append(res)
    return results


def save_results(results: List[Dict], output_path: str):
    with open(f"{output_path}.txt", "w", encoding="utf-8") as f:
        for res in results:
            f.write(f"Ключевое утверждение: {res['statement']}\n")
            f.write(f"Главная идея текста: {res['core_argument']}\n")
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
        save_results(final_results, str(script_dir / "final_analysis"))
        print(f"Результаты сохранены в final_analysis.txt")
    else:
        print(f"Файл {FILENAME} не найден.")
