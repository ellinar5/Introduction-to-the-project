import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DF = {
    'date': ['30.09', '01.10', '02.10', '03.10', '04.10', '05.10', '06.10', '07.10', '08.10', '09.10'],
    'time': ['06:00', '12:30', '18:45', '09:15', '15:20', '21:00', '03:30', '09:45', '15:00', '21:30'],
    'temperature': ['0.9', '0.0', '7.5', '5.0', '3.4', '2.2', '1.5', '1.7', '0.3', '1.0']
}
df = pd.DataFrame(DF)
df.insert(1, '', '')  # визуальный разделитель (пустой столбец)

# Проверочная функция для количества столбцов
def check_columns_count(dataframe):
    
    expected_cols = len(dataframe.columns)

    for index in range(len(dataframe)):
        
        row_length = len(dataframe.iloc[index])
        if row_length != expected_cols:
            
            raise ValueError(f"В строке {index+1}: количество элементов отличается.")
    
    return True

# показ первоначального DataFrame
def show_initial_dataframe():
    
    return df

#Собирает datetime (подставляет текущий год) и переводит температуру в float.
def prepare(df_in):
    year = pd.Timestamp.now().year
    combined = df_in['date'].str.strip() + ' ' + df_in['time'].str.strip() + f' {year}'
    dfc = df_in.copy()

    dfc['datetime'] = pd.to_datetime(combined, format="%d.%m %H:%M %Y", errors='coerce')

    if dfc['datetime'].isna().any():

        dfc['datetime'] = pd.to_datetime(combined, dayfirst=True, errors='coerce')

    if dfc['datetime'].isna().any():

        raise ValueError("Ошибка преобразования даты/времени. Проверьте формат.")

    dfc['temperature'] = pd.to_numeric(dfc['temperature'], errors='coerce')

    if dfc['temperature'].isna().any():

        raise ValueError("Ошибка: есть нечисловые значения в температуре")

    return dfc

#добавляем шум на данные
def add_noise(df_in, std=1.0, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, std, size=len(df_in))
    dfc = df_in.copy()
    dfc['temperature_noisy'] = dfc['temperature'].to_numpy(dtype=float) + noise

    # Проверка распределения
    print(f"[ШУМ] mean={noise.mean():.3f}, std={noise.std():.3f}")
    print(f"[ШУМ] Доля |шум| ≤ 1°C: {(np.mean(np.abs(noise) <= 1) * 100):.1f}%")

    # Проверяем длину массива
    if len(noise) != len(df_in):
        raise ValueError("Ошибка: длина массива шума не совпадает с числом строк")

    return dfc

#Вычисление скользящего среднего
def moving_average(df_in, col='temperature_noisy', k=5):
    values = pd.to_numeric(df_in[col], errors='coerce')
    ma = values.rolling(window=k, min_periods=k).mean().to_numpy(dtype=float)

    # Проверка: первые k-1 значений должны быть NaN
    nan_ok = np.all(np.isnan(ma[:k-1])) if k > 1 else True
    print(f"[MA] Проверка NaN до окна k: {nan_ok}")

    if not nan_ok:
        raise AssertionError(f"Ошибка: первые {k-1} значений MA должны быть NaN")

    if len(ma) != len(values):
        raise AssertionError("Ошибка: длина MA не совпадает с исходным рядом")

    return ma

#Построение трех графиков
def plot_series(df_in, ma, time_col='datetime', col='temperature_noisy'):
    if time_col not in df_in.columns:
        raise KeyError(f"Нет столбца {time_col} для построения графика")

    x = df_in[time_col]
    y = pd.to_numeric(df_in[col], errors='coerce').to_numpy(dtype=float)

    if len(x) != len(ma):
        raise ValueError("Ошибка: длина x и MA не совпадает")

    fig, axs = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
    axs[0].plot(x, y); axs[0].set_title('Исходный ряд')
    axs[1].plot(x, ma); axs[1].set_title('Скользящее среднее')
    axs[2].plot(x, y, label='Исходный'); axs[2].plot(x, ma, label='MA', linewidth=2); axs[2].legend()
    plt.show()
    print("[OK] Графики построены")


#разница между рядом и MA
def diff_report(orig, ma):
    orig = np.asarray(orig, dtype=float)
    ma = np.asarray(ma, dtype=float)
    if orig.shape != ma.shape:
        raise ValueError("Ошибка: размеры массива исходного и MA не совпадают")

    diff = orig - ma
    valid = ~np.isnan(diff)
    if valid.sum() == 0:
        print("[DIFF] Нет значений для сравнения (все NaN)")
    else:
        print(f"[DIFF] mean={np.nanmean(diff):.4f}, std={np.nanstd(diff):.4f}, "
              f"min={np.nanmin(diff):.4f}, max={np.nanmax(diff):.4f}")
    return diff



if __name__ == "__main__":
    
    # Показываем изначальный DataFrame
    initial_df = show_initial_dataframe()
    print("Исходный DataFrame:\n", initial_df.to_string(index=False))
    print("\n")

    df_p = prepare(df)
    print("\nПосле подготовки:")
    print(df_p[['date','time','datetime','temperature']])

    print("\n")
    df_n = add_noise(df_p, std=1.0, seed=42)
    ma = moving_average(df_n, 'temperature_noisy', k=5)

    print("\nПервые 10 MA:")
    print(ma[:10])

    plot_series(df_n, ma, time_col='datetime', col='temperature_noisy')

    diff = diff_report(df_n['temperature_noisy'].to_numpy(dtype=float), ma)
    print("\nПервые 10 значений разницы:")
    print(diff[:10])