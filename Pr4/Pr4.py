import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft

df = pd.read_csv('block_1.csv')
#  дані лише одного  домогосподарства
first_household = df['LCLid'].unique()[0]
household_data = df[df['LCLid'] == first_household].copy()
# Перетворення текстового Nall та Nan
household_data['energy(kWh/hh)'] = pd.to_numeric(household_data['energy(kWh/hh)'], errors='coerce')
# Заповнення пробілів за допомогою методу ffill
signal = household_data['energy(kWh/hh)'].ffill().values[:1024]

# Швидке перетворення Фур'є
fourier_result = np.abs(fft(signal))
# Дискретне вейвлет-перетворення (родина Добеші, тип 4)
wavelet_name = 'db4'
cA, cD = pywt.dwt(signal, wavelet_name)

# Візуалізація
fig, axs = plt.subplots(4, 1, figsize=(12, 10))
axs[0].plot(signal, color='black')
axs[0].set_title("Оригінальний сигнал споживання")
# Відкидаємо симетрію та нульову частоту для Фур'є
axs[1].plot(fourier_result[1:512], color='red')
axs[1].set_title("Спектр Фур'є")
axs[2].plot(cA, color='blue')
axs[2].set_title(f"Апроксимація вейвлетом")
axs[3].plot(cD, color='orange')
axs[3].set_title(f"Деталізація вейвлетом")
plt.tight_layout()
plt.show()