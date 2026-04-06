import pandas as pd
import numpy as np
import numexpr as ne
import time

np.random.seed(0)
n = 1_000_000
df = pd.DataFrame({
    'voltage': np.random.uniform(210, 240, n),
    'current': np.random.uniform(0, 30, n),
    'power_factor': np.random.uniform(0.7, 1.0, n),
    'hours': np.random.uniform(0.1, 1.0, n)
})
csv_filename = "energ_dataset.csv"
df.to_csv(csv_filename, index=False)

#  Спосіб через eval()
v_list = df['voltage'].tolist()
c_list = df['current'].tolist()
p_list = df['power_factor'].tolist()
h_list = df['hours'].tolist()
start_eval = time.time()
res_eval = [eval(f"{v} * {c} * {p} * {h}") for v, c, p, h in zip(v_list, c_list, p_list, h_list)]
total_energy_eval = sum(res_eval)
end_eval = time.time()

# Спосіб через Pandas eval()
start_pandas = time.time()
res_pandas = df.eval("voltage * current * power_factor * hours")
total_energy_pandas = res_pandas.sum()
end_pandas = time.time()

# Спосіб через NumExpr
voltage = df['voltage'].values
current = df['current'].values
power_factor = df['power_factor'].values
hours = df['hours'].values
start_numexpr = time.time()
res_numexpr = ne.evaluate("voltage * current * power_factor * hours")
total_energy_numexpr = res_numexpr.sum()
end_numexpr = time.time()

# векторізований спосіб
start_vectorized = time.time()
res_vectorized = df['voltage'] * df['current'] * df['power_factor'] * df['hours']
total_energy_vectorized = res_vectorized.sum()
end_vectorized = time.time()

print("\n Результати часу виконання ")
print(f"1. Standard eval(): {end_eval - start_eval:.4f} сек")
print(f"2. Pandas eval(): {end_pandas - start_pandas:.4f} сек")
print(f"3. NumExpr: {end_numexpr - start_numexpr:.4f} сек")
print(f"4. Vectorized: {end_vectorized - start_vectorized:.4f} сек")

print("\n Перевірка точності ")
print(f"Сума Standard eval: {total_energy_eval:,.2f}")
print(f"Сума Pandas eval: {total_energy_pandas:,.2f}")
print(f"Сума NumExpr: {total_energy_numexpr:,.2f}")
print(f"Сума Vectorized: {total_energy_vectorized:,.2f}")