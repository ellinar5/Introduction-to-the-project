import pandas as pd
import numpy as np

# 1. Генерация синтетических данных
np.random.seed(42)  # Для воспроизводимости

# Создаём DataFrame 
df = pd.DataFrame({
    'Name': np.random.choice(['Alice', 'Bob', 'Charlie', 'David'], size=100), #генерирует случайные числа с равномерным распределением от 0 до 1 (выбирает случайные элементы из списка с равной вероятностью, 100 раз)
    'Score': np.random.choice([10, 20, 30, 40], size=100)
})
print("Первоначальныый DataFrame:")
print(df.head(11))
print("\n")

# 2. Заменяем значения в столбце 'Score'
df['Score'] = df['Score'].replace({10: 100, 20: 200})

# возвращаем первые 10 строк (включая ноль будет 11)
print("DataFrame с заменой:")
print(df.head(11))

