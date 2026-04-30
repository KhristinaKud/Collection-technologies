import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 1. Підготовка даних
file_path = 'insurance.csv'
df_manual = pd.read_csv(file_path)
df_sklearn = df_manual.copy()
print("Реалізація вручну")
# Label Encoding
df_manual['sex'] = df_manual['sex'].map({'female': 0, 'male': 1})
df_manual['smoker'] = df_manual['smoker'].map({'no': 0, 'yes': 1})
# One-Hot Encoding
for region in df_manual['region'].unique():
    df_manual[f'region_{region}'] = (df_manual['region'] == region).astype(int)
df_manual_final = df_manual.drop('region', axis=1)
print(df_manual_final.head(3))
print("\n Використання sklearn")
# Label Encoding
le = LabelEncoder()
df_sklearn['sex'] = le.fit_transform(df_sklearn['sex'])
df_sklearn['smoker'] = le.fit_transform(df_sklearn['smoker'])
# One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
region_encoded = ohe.fit_transform(df_sklearn[['region']])
# Об'єднання
df_sklearn_final = pd.concat([df_sklearn.drop(columns=['region']), region_encoded], axis=1)
print(df_sklearn_final.head(3))