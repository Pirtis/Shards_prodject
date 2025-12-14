import keyboard
import sounddevice as sd
import numpy as np
import os
import torch
import json
# FIX для PyInstaller + transformers + tqdm
import sys
import types

# создаем фиктивный модуль tqdm для transformers
if getattr(sys, "frozen", False):
    import importlib
    import tqdm
    sys.modules["tqdm"] = tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vosk import Model, KaldiRecognizer


MODEL_PATH = "voice_model/vosk-model-small-ru-0.22"
SAMPLE_RATE = 16000
DURATION = 5

print("====================================")
print("  Голосовой модуль Barotrauma v1.0")
print("====================================")
print("Загрузка модели VOSK...")

model = Model(MODEL_PATH)
print("VOSK загружен.")
print("Нажми Q чтобы начать запись.")
print("------------------------------------")


def record_audio():
    print("[LOG] Запись голоса 5 секунд...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype=np.int16)
    sd.wait()
    print("[LOG] Запись завершена.")
    return audio


def recognize_vosk(audio_data):
    """
    audio_data — numpy array int16.
    """
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    # Vosk хочет bytes
    audio_bytes = audio_data.tobytes()

    if rec.AcceptWaveform(audio_bytes):
        result = rec.Result()
    else:
        result = rec.FinalResult()

    text = json.loads(result).get("text", "")
    return text.strip()


def load_classifier():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "command_model")  # папка рядом с EXE

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to("cpu")
    return tokenizer, model


def classify_text(model, tokenizer, text, label_map):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, padding=True,
                           max_length=64).to("cpu")

        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()

    return label_map.get(pred_id, f"Класс {pred_id}")

while True:
    print("[LOG] Ожидание нажатия клавиши q...")
    keyboard.wait("q")  # та же клавиша, что ёq

    print("[LOG] q нажата. Начинаю запись...")
    audio = record_audio()

    print("[LOG] Распознаю текст...")

    text = recognize_vosk(audio)

    print("[LOG] Распознанный текст:", text)

    # Загружаем классификатор
    tokenizer, clf_model = load_classifier()

    # метки
    label_map = {
        0: "сражение",
        1: "Включение питания реактора",
        2: "Вылючение питания реактора",
        3: "хил",
        4: "Навигация к пункту назначения",
        5: "Навигация назад",
        6: "Отстранение",
        7: "Очистить элементы",
        8: "Подождите",
        9: "пожар",
        10: "Ремонт механических систем",
        11: "Ремонт повреждённых систем",
        12: "Ремонт электрических систем",
        13: "следование",
        14: "Управление оружием",
        15: "Устранить утечки"
    }

    # Классификация
    prediction = classify_text(clf_model, tokenizer, text, label_map)
    print("------------------------------------")
    print("[LOG] Команда:", prediction)
    print("------------------------------------")

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(prediction)

    print("[LOG] Команда сохранена в output.txt")
    print("------------------------------------")
