import torch
import sounddevice as sd
import numpy as np
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import keyboard


model_dir = "./whisper_finetuned" 
processor = WhisperProcessor.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


def transcribe_audio(waveform, sr=16000):
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0)
    

    if sr != 16000:
        waveform = torchaudio.functional.resample(torch.tensor(waveform), sr, 16000).numpy()
    

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
    

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


SAMPLING_RATE = 16000
DURATION = 5  


print("Нажмите пробел для начала записи речи...")


while True:
    try:
        if keyboard.is_pressed("space"):
            print("Запись началась...")
            waveform = sd.rec(int(DURATION * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1)
            sd.wait()
            waveform = waveform.flatten()
            
            print("Распознавание...")
            text = transcribe_audio(waveform, SAMPLING_RATE)
            print("Распознанный текст:", text)
            
            print("\nНажмите пробел для новой записи...")
    except KeyboardInterrupt:
        print("Выход...")
        break