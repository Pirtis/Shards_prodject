
from transformers import pipeline
import numpy as np


clf = pipeline("text-classification", model="command_model_rubert_X_v2", device=0)
labels = np.load("command_model_rubert_X_v3/label_names.npy", allow_pickle=True)


for i, cmd in enumerate(labels):
    print(f"   {i:2d} → {cmd}")


while True:
    text = input("Ты: ").strip()
    if text.lower() in ["выход"]:
        break
    if not text:
        continue

    result = clf(text)[0]
    label_id = int(result['label'].split('_')[-1])
    command = labels[label_id]
    confidence = result['score']

    print(f"ИИ: {command}")
    print(f"    Уверенность: {confidence:.1%}")