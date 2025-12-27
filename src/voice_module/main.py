import keyboard
import sounddevice as sd
import numpy as np
import json
import sys
import torch

from config import SAMPLE_RATE, DURATION, MODEL_PATH, OUTPUT_FILE, TRIGGER_KEY
from spec_detect import detect_spec
from command_detect import detect_command
from output import save

from vosk import Model, KaldiRecognizer


def record_audio():
    print("[LOG] Запись голоса 5 секунд...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16
    )
    sd.wait()
    print("[LOG] Запись завершена.")
    return audio


def recognize_vosk(model, audio_data):
    print("[LOG] Распознаю текст...")
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    audio_bytes = audio_data.tobytes()

    if rec.AcceptWaveform(audio_bytes):
        result = rec.Result()
    else:
        result = rec.FinalResult()

    text = json.loads(result).get("text", "")
    print("[LOG] Распознанный текст:", text)
    return text.strip()


def main():
    print("====================================")
    print("  Голосовой модуль Barotrauma v1.0")
    print("====================================")
    print("Загрузка модели VOSK...")

    # ⚠️ КРИТИЧНО: модель создаётся здесь → логи Vosk 1 в 1
    model = Model(MODEL_PATH)

    print("VOSK загружен.")
    print(f"Нажми {TRIGGER_KEY.upper()} чтобы начать запись.")
    print("------------------------------------")

    while True:
        print(f"[LOG] Ожидание нажатия клавиши {TRIGGER_KEY}...")
        keyboard.wait(TRIGGER_KEY)

        print(f"[LOG] {TRIGGER_KEY} нажата. Начинаю запись...")
        audio = record_audio()

        text = recognize_vosk(model, audio)

        spec = detect_spec(text)
        print("------------------------------------")
        print("[LOG] Распознанная специализация:", spec)

        command = detect_command(text)
        print("------------------------------------")
        print("[LOG] Команда:", command)
        print("------------------------------------")

        save(command, spec)

        print("[LOG] Команда сохранена в output.txt")
        print("====================================")


if __name__ == "__main__":
    main()
