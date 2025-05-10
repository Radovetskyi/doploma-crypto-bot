import os
from openai import Client  # Правильний імпорт клієнта
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dotenv import load_dotenv
import os

load_dotenv('ProgAgents\.env')
# Ініціалізація клієнта OpenAI
client = Client(api_key=os.environ.get("API_KEY_OPENAI"))

# Функція для отримання відповіді від чат-бота
def get_chat_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ви - помічник із криптовалют та трейдингу."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Сталася помилка: {e}"

# Функція для завантаження та відображення графіка криптовалюти
def plot_crypto_price(ticker):
    try:
        data = yf.download(ticker, period="1mo", interval="1d")
        if not data.empty:
            fig, ax = plt.subplots()
            ax.plot(data.index, data["Close"], label=f"{ticker} (Ціна закриття)")

            # Форматування осі X для кращої читабельності
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Мітки раз на тиждень
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Формат дати
            plt.xticks(rotation=45, fontsize=8)  # Поворот міток і зменшення шрифту

            ax.set_xlabel("Дата")
            ax.set_ylabel("Ціна (USD)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("Не вдалося завантажити дані для вибраної криптовалюти.")
    except Exception as e:
        st.error(f"Сталася помилка під час завантаження даних: {e}")

# Інтерфейс Streamlit
st.title("Криптовалютний Асистент")

# Ввід користувача
user_input = st.text_input("Введіть ваше питання про криптовалюту або трейдинг:")

# Отримання та відображення відповіді
if user_input:
    answer = get_chat_response(user_input)
    st.write(f"Бот: {answer}")

# Вибір криптовалюти для візуалізації
st.subheader("Візуалізація цін криптовалют")
crypto_choice = st.selectbox(
    "Оберіть криптовалюту для перегляду графіка:",
    options=["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"],
    index=0
)

# Відображення графіка для вибраної криптовалюти
if crypto_choice:
    plot_crypto_price(crypto_choice)
