import pandas as pd

# Читання даних
df_products = pd.read_csv('amazon_products.csv')
df_orders = pd.read_csv('amazon_orders.csv')

# Об'єднання таблиці (Merge)
df_merged = pd.merge(df_orders, df_products, on='product_id', how='left')

# Якщо в колонці category стоїть NaN, product_id не знайдено в довіднику
missing_data = df_merged[df_merged['category'].isna()]

total_orders = len(df_orders)
missing_count = len(missing_data)
missing_share = (missing_count / total_orders) * 100

print("\nПриклад об'єднаних даних:")
print(df_merged.head())
print(f"Всього замовлень: {total_orders}")
print(f"Замовлень без опису в базі: {missing_count}")
print(f"Відсоток помилок: {missing_share:.2f}%")