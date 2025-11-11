import whisper
import sounddevice as sd
import numpy as np

# Параметры записи
duration = 5  # секунд
sample_rate = 16000


model = whisper.load_model("base")

def record_audio(duration, sample_rate):
    print("Говори...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    print("Запись завершена.")
    return np.squeeze(audio)

def recognize_speech(audio, sample_rate):
    result = model.transcribe(audio, fp16=False, language="ru")
    return result["text"]

if __name__ == "__main__":
    audio = record_audio(duration, sample_rate)
    text = recognize_speech(audio, sample_rate)
    print("Распознанный текст:")
    print(text)
