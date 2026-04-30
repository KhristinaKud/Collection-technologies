import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Центр
df_center = pd.read_csv('Amplitudes.csv')
df_center['Sound pressure level (dB)'] = pd.to_numeric(df_center['Sound pressure level (dB)'], errors='coerce')
df_center = df_center.replace([np.inf, -np.inf], np.nan).dropna()

# Робота
df_work = pd.read_csv('Amplitudes2.csv')
df_work['Sound pressure level (dB)'] = pd.to_numeric(df_work['Sound pressure level (dB)'], errors='coerce')
df_work = df_work.replace([np.inf, -np.inf], np.nan).dropna()

def print_location_stats(name, dataframe):
    mean_val = dataframe['Sound pressure level (dB)'].mean()
    max_val = dataframe['Sound pressure level (dB)'].max()
    min_val = dataframe['Sound pressure level (dB)'].min()

    print(f"--- {name} ---")
    print(f"Середній рівень шуму: {mean_val:.2f} дБ.")
    print(f"Максимальний пік: {max_val:.2f} дБ.")
    print(f"Мінімальне значення: {min_val:.2f} дБ.\n")

# Вивід у консоль
print_location_stats("Центральна частина міста", df_center)
print_location_stats("Робоче приміщення", df_work)

# Візуалізація
plt.figure(figsize=(12, 6))
plt.plot(df_center['Time (s)'], df_center['Sound pressure level (dB)'], label='Центр', color='royalblue', alpha=0.8)
plt.plot(df_work['Time (s)'], df_work['Sound pressure level (dB)'], label='Робота', color='darkorange', alpha=0.6)

plt.title('Порівняльний аналіз рівня шуму: Центр vs Робота')
plt.xlabel('Час (секунди)')
plt.ylabel('Рівень звукового тиску (дБ)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()