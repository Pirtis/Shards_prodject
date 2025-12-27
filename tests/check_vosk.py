import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import keyboard
import json
import datetime
import os
import time


MODEL_PATH = r"E:\Education\4 course 1 semester\Course project\Shards_prodject\Code\Task\Model\vosk-model-small-ru-0.22"
model = Model(MODEL_PATH)
samplerate = 16000
channels = 1
max_duration = 15
OUTPUT_DIR = "E:/Education/4 course 1 semester/Course project/Shards_prodject/Code/Task/Test/Recorded_texts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def record_while_q():
    rec = KaldiRecognizer(model, samplerate)
    audio_frames = []
    recording = False
    start_time = None


    print("Удерживайте 'q' для записи (макс 15 сек)")


    def callback(indata, frames, time_info, status):
        audio_frames.append(np.frombuffer(indata, dtype=np.int16))
        if rec.AcceptWaveform(audio_frames[-1].tobytes()):
            pass


    try:
        with sd.RawInputStream(samplerate=samplerate, blocksize=8000,
                               dtype='int16', channels=channels,
                               callback=callback):
            while True:
                if keyboard.is_pressed('q'):
                    if not recording:
                        recording = True
                        start_time = time.time() 
                        print("Запись началась...")
                    elapsed = time.time() - start_time
                    if elapsed > max_duration:
                        print("Достигнут лимит 15 секунд.")
                        break
                else:
                    if recording:
                        print("Запись остановлена.")
                        break
                # time.sleep(0.01) # чтобы не грузить CPU
    except KeyboardInterrupt:
        print("Прервано пользователем")


    if audio_frames:
        audio_np = np.concatenate(audio_frames, axis=0)
        rec.AcceptWaveform(audio_np.tobytes())
        result = json.loads(rec.FinalResult())
        text = result.get("text", "")
        print("Распознано:", text)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"speech_{timestamp}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Текст сохранён в файл: {filename}")
    else:
        print("Аудио не было записано.")


while True:
    record_while_q()
