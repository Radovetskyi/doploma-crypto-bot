import matplotlib
matplotlib.use('Agg')

import telebot
import yfinance as yf
import matplotlib.pyplot as plt
import io
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from difflib import get_close_matches
import pandas as pd
from statistics import mean
import os
from dotenv import load_dotenv

load_dotenv()

# Ініціалізація бота з API-ключем

bot = telebot.TeleBot(os.environ.get("API_KEY_OPENAI"))

# Завантаження FAQ з файлу
def load_faq_data(file_path="faq.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    faq = {}
    sections = content.split("\n\n")
    for section in sections:
        lines = section.split(":\n", 1)
        if len(lines) == 2:
            question, answer = lines
            faq[question.strip()] = answer.strip()
    return faq

# Пошук найближчої відповіді
def find_best_answer(user_query, faq_data):
    questions = list(faq_data.keys())
    closest_match = get_close_matches(user_query, questions, n=1, cutoff=0.5)
    if closest_match:
        return faq_data[closest_match[0]]
    else:
        return "Вибачте, я не знайшов відповіді на ваше запитання."

# Функція для побудови графіку криптовалют
def plot_crypto(symbol, period="1y"):
    data = yf.download(symbol, period=period)

    print(data) 

    if data.empty:
        return None, None, None, None, None

    # Видалення другого рівня MultiIndex
    data.columns = data.columns.droplevel(1)  # Змінено на droplevel(1)

    # Перетворення стовпця 'Close' на числові значення
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

    highest_price = data['High'].max()
    lowest_price = data['Low'].min()
    average_price = data['Close'].mean()
    total_volume = data['Volume'].sum()

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'], label=f"{symbol} (Close Price)")
    plt.xlabel("Дата", fontsize=8)
    plt.ylabel("Ціна (USD)")
    plt.xticks(rotation=45, fontsize=8)
    plt.title(f"Графік {symbol} за {period}")
    plt.legend()
    plt.grid(True)

    # Збереження графіку у буфер пам'яті
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return buf, highest_price, lowest_price, average_price, total_volume


# Функція для створення клавіатури з кнопками
def create_crypto_keyboard():
    keyboard = InlineKeyboardMarkup()
    buttons = [
        InlineKeyboardButton("BTC-USD", callback_data="BTC-USD"),
        InlineKeyboardButton("ETH-USD", callback_data="ETH-USD"),
        InlineKeyboardButton("BNB-USD", callback_data="BNB-USD"),
        InlineKeyboardButton("SOL-USD", callback_data="SOL-USD")
    ]
    keyboard.add(*buttons)
    return keyboard

# Завантаження FAQ
faq_data = load_faq_data()

# Обробка команди /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Вітаю! Я ваш помічник з криптовалют. Виберіть графік криптовалюти:", reply_markup=create_crypto_keyboard())

# Обробка натискання кнопок
@bot.callback_query_handler(func=lambda call: True)
def handle_callback_query(call):
    symbol = call.data
    result = plot_crypto(symbol)
    
    if result[0] is None:  # Якщо buf == None, означає, що дані не були знайдені
        bot.send_message(call.message.chat.id, f"Не вдалося знайти дані для {symbol}.")
        return

    buf, highest_price, lowest_price, average_price, total_volume = result
    
    response_message = (f"Графік для {symbol}\n"
                        f"Найвища ціна: {highest_price:.2f} USD\n"
                        f"Найнижча ціна: {lowest_price:.2f} USD\n"
                        f"Середня ціна: {average_price:.2f} USD\n"
                        f"Загальний об'єм: {total_volume:.0f}\n")
    
    bot.send_photo(call.message.chat.id, buf, caption=response_message)
    bot.answer_callback_query(call.id)


# Обробка текстових повідомлень
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_query = message.text.strip()
    answer = find_best_answer(user_query, faq_data)
    bot.reply_to(message, f"Відповідь: {answer}")

# Запуск бота
if __name__ == "__main__":
    bot.polling()
