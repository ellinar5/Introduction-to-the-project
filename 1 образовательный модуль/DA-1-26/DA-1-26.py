import pandas as pd

# Данные для DataFrame
DF = {
    'id': range(1, 6),
    'commands': ['Alpine F1 Team', 'Aston Martin F1 team', 'Mercedes AMG Petronas Motorsport', 'McLaren Formula 1 Team', 'Atlassian Williams Racing'],
    'points': [10, 20, 30, 40, 50]  
}

df = pd.DataFrame(DF) # Создаем DataFrame


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


# Вторая функция - изменение ID в первом столбце
def manipulate_id_column(dataframe):

    try:
        check_columns_count(dataframe)  # Проверка перед изменением
        dataframe_copy = dataframe.copy()
        original_ids = set(dataframe_copy['id'])
        new_ids = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        changed_ids = dataframe_copy['id'].replace(new_ids)

        if any(changed_ids.isna()):
            raise ValueError("Некоторые значения ID были потеряны при замене!")
        return dataframe_copy.assign(id=changed_ids)
    
    except Exception as e:
        print(f"Ошибка при изменении ID: {e}")
        return None


# Третья функция - изменение команд во втором столбце
def manipulate_values_column(dataframe):
    try:
        check_columns_count(dataframe)  # Проверка перед изменением
        dataframe_copy = dataframe.copy()
        original_values = set(dataframe_copy['commands'])
        new_values = {'McLaren Formula 1 Team': 'Scuderia Ferrari HP'}
        changed_values = dataframe_copy['commands'].replace(new_values)
        
        if any(changed_values.isna()):
            raise ValueError("Некоторые значения команд были потеряны при замене!")
        return dataframe_copy.assign(commands=changed_values)
    
    except Exception as e:
        print(f"Ошибка при изменении команд: {e}")
        return None


# Четвертая функция - одновременная замена команд и очков
def manipulate_values_and_points_columns(dataframe):
    try:
        check_columns_count(dataframe)  # Проверка перед изменением
        dataframe_copy = dataframe.copy()
        original_values = set(dataframe_copy['commands'])
        new_values = {'McLaren Formula 1 Team': 'Scuderia Ferrari HP'}
        changed_values = dataframe_copy['commands'].replace(new_values)

        if any(changed_values.isna()):
            raise ValueError("Некоторые значения команд были потеряны при замене!")
            
        original_points = set(dataframe_copy['points'])
        new_points = {10: 100, 20: 200, 30: 300, 40: 4000, 50: 500}
        changed_points = dataframe_copy['points'].replace(new_points)

        if any(changed_points.isna()):
            raise ValueError("Некоторые значения очков были потеряны при замене!")
            
        return dataframe_copy.assign(commands=changed_values, points=changed_points)
   
    except Exception as e:
        print(f"Ошибка при одновременной замене: {e}")
        return None


if __name__ == "__main__":
    
    # Показываем изначальный DataFrame
    initial_df = show_initial_dataframe()
    print("Исходный DataFrame:\n", initial_df.to_string(index=False))
    print("\n")

    # Результат второй функции (замена ID)
    modified_ids = manipulate_id_column(initial_df)
    if modified_ids is not None:
        print("DataFrame после изменения ID:\n", modified_ids.to_string(index=False))
    else:
        print("ID не удалось изменить.\n")
    print("\n")

    # Результат третьей функции (замена команд)
    modified_names = manipulate_values_column(initial_df)
    if modified_names is not None:
        print("DataFrame после изменения команд:\n", modified_names.to_string(index=False))
    else:
        print("Названия команд не удалось изменить.\n")
    print("\n")

    # Результат четвертой функции (одновременная замена команд и очков)
    final_result = manipulate_values_and_points_columns(initial_df)
    if final_result is not None:
        print("Итоговый DataFrame после изменений:\n", final_result.to_string(index=False))
    else:
        print("Одновременная замена не удалась.\n")