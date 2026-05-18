import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np  # Додаємо для коректної ініціалізації порожніх числових значень

# Налаштування сторінки
st.set_page_config(page_title="AI Оптимізатор Навантаження", layout="wide")
st.title("ШІ-Агент Оптимізації Навантаження")

# Налаштування API
st.sidebar.subheader("Налаштування ШІ")
api_key_input = st.sidebar.text_input("Введіть Gemini API Key", type="password", value="AIzaSyBdQB44ln6G5vizl4Pr0beWhIpEEQ0AoyY")

# Системний Промпт
SYSTEM_PROMPT = """Ти - АІ агент для оптимізації енергоспоживання. Твоє завдання — аналізувати дані енергоспоживання, знаходити пікові навантаження, виявляти аномальні періоди, показувати графіки та формувати рекомендації для优化ції споживання електроенергії.
Як відповідати:
1. Якщо користувач просить "повний аналіз", виконай покроково такі 6 кроків:
   - Проаналізуй надані дані енергоспоживання.
   - Визнач пікові періоди навантаження.
   - Вияв можливі аномалії або нерівномірності споживання.
   - Поясни можливі причини виникнення піків.
   - Запропонуй рекомендації щодо оптимізації навантаження.
   - Сформуй короткий висновок.
2. Якщо користувач просить щось одне зі списку (наприклад, "порахуй середнє значення", "покажи тільки пікові навантаження", "знайди аномалію", чи задає будь яке інше конкретне питання):
   - Дай чітку, коротку відповідь тільки на те, про що тебе запитали, використовуючи надану статистичну вибірку."""

st.sidebar.subheader("Дані споживання")
uploaded_file = st.sidebar.file_uploader("Завантажте CSV файл", type=['csv'])

# Ініціалізація історії чату
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Привіт! Завантажте файл з даними та напишіть мені, що з ними зробити. Ви можете попросити 'повний аналіз', 'знайди пікові навантаження'"}
    ]


# Функція для виклику Gemini
def get_gemini_analysis(query, data_summary, api_key):
    if not api_key:
        return "Помилка: Будь ласка, вкажіть коректний API ключ у бічній панелі."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        full_prompt = f"{SYSTEM_PROMPT}\n\nОсь статистична вибірка з даних користувача:\n{data_summary}\n\nЗапит користувача: {query}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Помилка при запиті до Gemini API: {str(e)}"


# Основна логіка
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    time_col = df.columns[0]
    val_col = df.columns[1]
    df = df.sort_values(by=time_col).head(200).reset_index(drop=True)

    with st.expander("Попередній перегляд завантажених даних (перші 200 годин)"):
        st.dataframe(df.head())
        chart_df = df.copy().set_index(time_col)
        st.line_chart(chart_df[val_col])

    # Відображення історії чату
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "chart_type" in msg:
                if msg["chart_type"] == "line":
                    st.line_chart(msg["chart_data"])
                elif msg["chart_type"] == "scatter":
                    st.scatter_chart(msg["chart_data"])

    # Ввід повідомлення користувачем
    if prompt := st.chat_input("Напишіть запит..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            prompt_lower = prompt.lower()

            # Математичні обчислення
            mean_val = df[val_col].mean()
            std_val = df[val_col].std()
            threshold_90 = df[val_col].quantile(0.90)

            peaks_df = df[df[val_col] > threshold_90]
            anomalies_df = df[(df[val_col] > mean_val + 2 * std_val) | (df[val_col] < mean_val - 2 * std_val)]

            # Пакет даних для передачі в LLM
            data_summary = f"""
            Період з {df[time_col].min()} по {df[time_col].max()}
            Загальна кількість записів: {len(df)}
            Середнє значення споживання: {mean_val:.2f}
            Максимальне значення: {df[val_col].max():.2f}
            Мінімальне значення: {df[val_col].min():.2f}
            Пікові періоди (топ-10%):
            {peaks_df[[time_col, val_col]].head(10).to_string(index=False)}
            Виявлені математичні аномалії (поза 2 сигмами):
            {anomalies_df[[time_col, val_col]].to_string(index=False) if not anomalies_df.empty else "Аномалій не знайдено"}
            """

            with st.spinner("Агент обробляє запит..."):
                gemini_response = get_gemini_analysis(prompt, data_summary, api_key_input)
                st.write(gemini_response)

                # Створюємо словник для збереження
                msg_to_save = {"role": "assistant", "content": gemini_response}

                if "пік" in prompt_lower or "пікові" in prompt_lower:
                    base_chart_df = df.copy().set_index(time_col)
                    plot_df = pd.DataFrame(index=base_chart_df.index)
                    plot_df["Базове споживання"] = base_chart_df[val_col].astype(float)

                    # КЛЮЧОВЕ ВИПРАВЛЕННЯ: Задаємо тип float і порожнє значення як NaN
                    plot_df["Пікові точки"] = np.nan
                    plot_df["Пікові точки"] = plot_df["Пікові точки"].astype(float)

                    # Безпечно мапимо значення
                    plot_df.loc[peaks_df[time_col], "Пікові точки"] = peaks_df[val_col].values

                    st.line_chart(plot_df)
                    msg_to_save["chart_type"] = "line"
                    msg_to_save["chart_data"] = plot_df

                elif "аномал" in prompt_lower or "точка" in prompt_lower:
                    if not anomalies_df.empty:
                        plot_data = anomalies_df.copy().set_index(time_col)[val_col]
                        st.scatter_chart(plot_data)
                        msg_to_save["chart_type"] = "scatter"
                        msg_to_save["chart_data"] = plot_data
                    else:
                        st.info("Математичних аномалій для побудови точкового графіка не виявлено.")

                # Зберігаємо в історію в самому кінці успішного блоку
                st.session_state.messages.append(msg_to_save)
else:
    st.info("💡 Будь ласка, завантажте CSV файл з даними про енергоспоживання у боковій панелі.")