import matplotlib
matplotlib.use('Agg')

import telebot
import yfinance as yf
import matplotlib.pyplot as plt
import io
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from difflib import get_close_matches
import mplfinance as mpf
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import statsmodels.api as sm
from itertools import product
import warnings
import os
from dotenv import load_dotenv

load_dotenv()

# Ініціалізація бота з API-ключем

bot = telebot.TeleBot(os.environ.get("API_KEY_TELEGRAM"))

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
def plot_crypto(symbol, period="1mo"):
    data = yf.download(symbol, period=period)
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
    return buf

def clean_yfinance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        # Приводимо колонки до однорівневого списку, відкидаючи верхній рівень
        df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        
        # Переконаймося, що індекс — це datetime
        df.index = pd.to_datetime(df.index)

        # Сортуємо за датою (опціонально)
        df.sort_index(inplace=True)

        return df

def plot_crypto_with_arima(symbol, period="max", interval="1mo", forecast_periods=1):
        data = yf.download(symbol, period=period, interval=interval)
        data = clean_yfinance_dataframe(data)
        df = data[['Close']].dropna()

        if len(df) < 24:
            raise ValueError("Недостатньо даних для сезонного ARIMA. Потрібно щонайменше 24 місяці.")

        # BoxCox трансформація
        df['Close_box'], lmbda = boxcox(df['Close'])
        df['prices_box_diff'] = df['Close_box'] - df['Close_box'].shift(12)
        df['prices_box_diff2'] = df['prices_box_diff'] - df['prices_box_diff'].shift(1)
        df.dropna(inplace=True)

        # Підбір найкращих параметрів SARIMA
        Ps = range(0, 2)
        qs = range(0, 2)
        ps = range(0, 2)
        Qs = range(0, 2)
        D = 1
        d = 1

        parameters = list(product(ps, qs, Ps, Qs))
        best_aic = float("inf")
        best_model = None

        warnings.filterwarnings('ignore')

        for param in parameters:
            try:
                model = sm.tsa.statespace.SARIMAX(
                    df['Close_box'],
                    order=(param[0], d, param[1]),
                    seasonal_order=(param[2], D, param[3], 12)
                ).fit(disp=False)

                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = model
            except:
                continue

        # Прогноз: історичний + майбутній
        pred_in_sample = best_model.predict(start=0, end=len(df)-1)
        forecast_future = best_model.predict(start=len(df), end=len(df)+forecast_periods-1)

        forecast_in_sample = inv_boxcox(pred_in_sample, lmbda)
        forecast_future = inv_boxcox(forecast_future, lmbda)

        df['Forecast'] = forecast_in_sample
        future_dates = [df.index[-1] + pd.DateOffset(months=i) for i in range(1, forecast_periods + 1)]
        future_df = pd.DataFrame(forecast_future, index=future_dates, columns=['Forecast'])

        # Візуалізація
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label='Історичні дані')
        plt.plot(df.index, df['Forecast'], label='ARIMA (in-sample)', linestyle='--')
        plt.plot(future_df.index, future_df['Forecast'], label='Прогноз', color='red')
        plt.title(f"{symbol}: історія та прогноз")
        plt.xlabel("Дата")
        plt.ylabel("Ціна (USD)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)

        # Збереження графіка в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        # Результат прогнозу і рекомендація
        last_price = df['Close'].iloc[-1]
        forecasted_price = forecast_future[-1]
        change_pct = ((forecasted_price - last_price) / last_price) * 100

        if change_pct > 3:
            recommendation = "Купувати актив"
        elif change_pct < -3:
            recommendation = "Продавати актив"
        else:
            recommendation = "Утримувати актив"

        print(f"\n--- {symbol} ---")
        print(f"Остання ціна: {last_price:.2f} USD")
        print(f"Прогноз через {forecast_periods} міс.: {forecasted_price:.2f} USD")
        print(f"Зміна: {change_pct:.2f}%")
        print(f"Рекомендація: {recommendation}")

        return buf

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
    buf = plot_crypto(symbol)
    bot.send_photo(call.message.chat.id, buf, caption=f"Графік для {symbol}")
    bot.answer_callback_query(call.id)

# Обробка текстових повідомлень
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_query = message.text.strip()  # Не перетворюємо в верхній регістр
    answer = find_best_answer(user_query, faq_data)
    bot.reply_to(message, f"Відповідь: {answer}")



# Запуск бота
if __name__ == "__main__":
    bot.polling()
