import sounddevice as sd
import numpy as np
import json
from vosk import KaldiRecognizer, Model
from config import SAMPLE_RATE, DURATION, MODEL_PATH

model = Model(MODEL_PATH)

def record_audio():
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16
    )
    sd.wait()
    return audio

def recognize(audio_data):
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    audio_bytes = audio_data.tobytes()

    if rec.AcceptWaveform(audio_bytes):
        result = rec.Result()
    else:
        result = rec.FinalResult()

    return json.loads(result).get("text", "").strip()
