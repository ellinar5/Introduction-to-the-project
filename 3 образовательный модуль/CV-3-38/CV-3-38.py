import cv2
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:/Users/User/Downloads/Tesseract-OCR/tesseract.exe"

# =========================
# ЧЕТКИЕ ПУТИ К ДАТАСЕТУ
# =========================
PAPER_DIR = r"E:/УЧЕБА/УЧЕБА НГУ/2 КУРС/1 СЕМЕСТР/1.ОСНОВНАЯ/Введение в проект/3 блок/dataset/paper"
SCREEN_DIR = r"E:/УЧЕБА/УЧЕБА НГУ/2 КУРС/1 СЕМЕСТР/1.ОСНОВНАЯ/Введение в проект/3 блок/dataset/screen"

# =========================
# ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
# =========================
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка загрузки: {image_path}")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Белый фон: высокая яркость + низкая насыщенность
    white_mask = (v > 200) & (s < 40)
    white_ratio = np.sum(white_mask) / (img.shape[0] * img.shape[1])

    # Яркие блики
    bright_ratio = np.sum(v > 240) / (img.shape[0] * img.shape[1])

    # Шум белого фона
    v_blur = cv2.GaussianBlur(v, (5, 5), 0)
    white_std = np.std(v_blur[white_mask]) if np.sum(white_mask) > 0 else 0

    return [white_ratio, bright_ratio, white_std]

# =========================
# СБОР ДАТАСЕТА
# =========================
X = []  # признаки
y = []  # метки (0 — бумага, 1 — экран)

for file in os.listdir(PAPER_DIR):
    path = os.path.join(PAPER_DIR, file)
    feats = extract_features(path)
    if feats:
        X.append(feats)
        y.append(0)

for file in os.listdir(SCREEN_DIR):
    path = os.path.join(SCREEN_DIR, file)
    feats = extract_features(path)
    if feats:
        X.append(feats)
        y.append(1)

print(f"Загружено изображений: {len(X)}")

# =========================
# ОБУЧЕНИЕ МОДЕЛИ
# =========================
model = LogisticRegression()
model.fit(X, y)

print("Модель успешно обучена")

# =========================
# ПРОВЕРКА ИЗОБРАЖЕНИЯ
# =========================
def predict_image(image_path):
    feats = extract_features(image_path)
    if feats is None:
        return

    pred = model.predict([feats])[0]
    prob = model.predict_proba([feats])[0]

    print("\n=== АНАЛИЗ ИЗОБРАЖЕНИЯ ===")
    print(f"Белый фон (доля): {feats[0]:.3f}")
    print(f"Яркие блики (доля): {feats[1]:.3f}")
    print(f"Шум белого фона (std): {feats[2]:.2f}")

    if abs(prob[0] - prob[1]) < 0.15:
        print("РЕЗУЛЬТАТ: ⚠️ ПОДОЗРЕНИЕ")
    else:
        print("РЕЗУЛЬТАТ:", "БУМАГА" if pred == 0 else "ЭКРАН")

    print(f"Вероятность бумаги: {prob[0]:.2f}")
    print(f"Вероятность экрана: {prob[1]:.2f}")

# =========================
# НОВАЯ УМНАЯ ФУНКЦИЯ ИТОГА
# =========================
def extract_total_price_smart(image_path):

    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    data = pytesseract.image_to_data(
        gray, lang="rus", output_type=pytesseract.Output.DICT
    )

    keywords = [
        "ИТОГ",
        "ИТОГО",
        "ИТОГОК",
        "ОПЛАТЕ",
        "СУЧЕТОМ",
        "СКИДКИ"
    ]

    # ищем слово "ИТОГ"
    for i, word in enumerate(data["text"]):
        if any(k in word.upper().replace(" ", "") for k in keywords):
            y = data["top"][i]
            region_top = max(0, int(y - 0.12 * h))
            region_bottom = min(h, int(y + 0.12 * h))

            roi = gray[region_top:region_bottom, :]

            text = pytesseract.image_to_string(roi, lang="rus")
            text = text.replace(",", ".")

            match = re.search(r"([0-9]{2,6}\.[0-9]{2})", text)
            if match:
                return float(match.group(1))

    return None

# =========================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =========================
test_image = r"E:/УЧЕБА/УЧЕБА НГУ/2 КУРС/1 СЕМЕСТР/1.ОСНОВНАЯ/Введение в проект/3 блок/1407412961_1371258957.jpg"

predict_image(test_image)

price = extract_total_price_smart(test_image)

if price:
    print(f"💰 ИТОГОВАЯ СУММА: {price}")
else:
    print("⚠️ ИТОГОВАЯ СУММА НЕ НАЙДЕНА")