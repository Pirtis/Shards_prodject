import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import keyboard
import json
import datetime
import os
import time
import threading

# ---------------- Настройки ----------------
MODEL_PATH = r"E:\Education\4 course 1 semester\Course project\Shards_prodject\Code\Task\vosk-model-small-ru-0.22"  # путь к вашей модели
samplerate = 16000
channels = 1
max_duration = 15  # сек
OUTPUT_DIR = "E:/Education/4 course 1 semester/Course project/Shards_prodject/Code/Task/recorded_texts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ------------------------------------------

# Загружаем модель
model = Model(MODEL_PATH)

# Глобальные переменные
recording = False
record_start_time = None
rec = None
audio_frames = []
text_buffer = ""

# Callback для RawInputStream
def audio_callback(indata, frames, time_info, status):
    global rec, text_buffer, recording
    audio_frames.append(np.frombuffer(indata, dtype=np.int16))
    if recording and rec:
        if rec.AcceptWaveform(audio_frames[-1].tobytes()):
            res = json.loads(rec.Result())
            text = res.get("text", "")
            if text:
                print(text, end=" ", flush=True)
                text_buffer += text + " "

# Начало записи
def start_record(event):
    global recording, record_start_time, rec, text_buffer
    if not recording:
        recording = True
        record_start_time = time.time()
        text_buffer = ""
        rec = KaldiRecognizer(model, samplerate)
        print("Запись началась...")

# Окончание записи
def stop_record(event):
    global recording, rec, text_buffer
    if recording:
        recording = False
        # Получаем остаток текста
        if rec:
            final_res = json.loads(rec.FinalResult())
            final_text = final_res.get("text", "")
            if final_text:
                print(final_text, flush=True)
                text_buffer += final_text
        print("\nЗапись завершена")
        # Сохраняем текст
        if text_buffer.strip():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(OUTPUT_DIR, f"speech_{timestamp}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text_buffer.strip())
            print(f"Текст сохранён в файл: {filename}")

# Контроль лимита времени
def monitor_duration():
    global recording, record_start_time
    while True:
        if recording and record_start_time:
            if time.time() - record_start_time > max_duration:
                print("\nДостигнут лимит 15 секунд.")
                keyboard.release("q")  # принудительно "отпускаем" клавишу
        time.sleep(0.05)

# Привязка клавиши
keyboard.on_press_key("q", start_record)
keyboard.on_release_key("q", stop_record)

# Поток для контроля лимита времени
threading.Thread(target=monitor_duration, daemon=True).start()

# Запуск RawInputStream
with sd.RawInputStream(samplerate=samplerate,
                       blocksize=8000,
                       dtype='int16',
                       channels=channels,
                       callback=audio_callback):
    print("Нажмите и удерживайте 'q' для записи...")
    keyboard.wait()  # ждём событий клавиатуры
