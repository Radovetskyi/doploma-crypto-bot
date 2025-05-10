import streamlit as st
from difflib import get_close_matches

# Функція для завантаження даних з текстового файлу
def load_faq_data(file_path="faq.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Розбиваємо текст на питання-відповіді
    faq = {}
    sections = content.split("\n\n")
    for section in sections:
        lines = section.split(":\n", 1)
        if len(lines) == 2:
            question, answer = lines
            faq[question.strip()] = answer.strip()
    return faq

# Функція для пошуку найкращого збігу
def find_best_answer(user_query, faq_data):
    questions = list(faq_data.keys())
    closest_match = get_close_matches(user_query, questions, n=1, cutoff=0.5)
    if closest_match:
        return faq_data[closest_match[0]]
    else:
        return "Вибачте, я не знайшов відповіді на ваше запитання."

# Завантажуємо FAQ дані
faq_data = load_faq_data()

# Інтерфейс Streamlit
st.title("Криптовалютний Асистент (FAQ)")

# Ввід користувача
user_input = st.text_input("Введіть ваше запитання:")

# Відображення відповіді
if user_input:
    answer = find_best_answer(user_input, faq_data)
    st.write(f"Відповідь: {answer}")
