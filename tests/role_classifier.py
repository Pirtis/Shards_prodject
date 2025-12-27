import keyboard
import sounddevice as sd
import numpy as np
# import whisper
import os
import torch
import sys
import types


if getattr(sys, "frozen", False):
    import importlib
    import tqdm

    sys.modules["tqdm"] = tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification

DURATION = 5
SAMPLE_RATE = 16000
print("====================================")
print("  Голосовой модуль Barotrauma v1.0")
print("====================================")
print("Загрузка модели Whisper...")

# model = whisper.load_model("base")
print("Whisper загружен.")
print("Нажми q чтобы начать запись.")
print("------------------------------------")


def record_audio():
    print("[LOG] Запись голоса 5 секунд...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype=np.float32)
    sd.wait()
    print("[LOG] Запись завершена.")
    return np.squeeze(audio)


def load_classifier():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "command_model")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to("cpu")
    return tokenizer, model


def classify_text(model, tokenizer, text, label_map):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, padding=True,
                           max_length=128).to("cpu")

        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()

    return label_map.get(pred_id, f"Класс {pred_id}")


# Классификация
import pickle
from torch import nn


class TextTokenizer:
    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.vocab_size = 0

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            words = str(text).lower().split()
            sequence = []
            for word in words:
                if word in self.word_index:
                    sequence.append(self.word_index[word])
            sequences.append(sequence)
        return sequences


class SpecializationClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super(SpecializationClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        out = torch.cat((hidden_forward, hidden_backward), dim=1)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.batch_norm1(out)
        out = self.relu(self.fc2(out))
        out = self.batch_norm2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


def load_role_classifier():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        model_path = os.path.join(base_dir, "best_specialization_model_8classes.pth")
        tokenizer_path = os.path.join(base_dir, "tokenizer.pkl")
        label_encoder_path = os.path.join(base_dir, "label_encoder.pkl")

        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        model_config = {
            "vocab_size": 1685,
            "embedding_dim": 100,
            "hidden_dim": 128,
            "output_dim": 8,
            "n_layers": 2,
            "dropout": 0.3
        }

        model = SpecializationClassifier(**model_config)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, tokenizer, label_encoder
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки модели ролей: {e}")
        return None, None, None


def pad_sequences(sequences, maxlen=50, padding='post', value=0):
    result = []
    for seq in sequences:
        if len(seq) > maxlen:
            seq = seq[:maxlen]
        else:
            pad_length = maxlen - len(seq)
            if padding == 'post':
                seq = seq + [value] * pad_length
            else:
                seq = [value] * pad_length + seq
        result.append(seq)
    return result


def predict_role(model, tokenizer, label_encoder, text):
    if model is None or tokenizer is None or label_encoder is None:
        return "Модель не загружена"
    try:
        model.eval()
        sequences = tokenizer.texts_to_sequences([text])
        if not sequences or len(sequences[0]) == 0:
            return "Текст не распознан"
        sequence = sequences[0]
        padded_sequence = pad_sequences([sequence], maxlen=50)[0]
        input_tensor = torch.tensor([padded_sequence]).long()
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
        predicted_role = label_encoder.inverse_transform([predicted_class])[0]
        return predicted_role
    except Exception as e:
        return "Ошибка предсказания"


def save_role_to_file(role, filename="out.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(role)
        print(f"[SUCCESS] Роль сохранена в {filename}: {role}")
    except Exception as e:
        print(f"[ERROR] Ошибка сохранения файла: {e}")


role_model, role_tokenizer, label_encoder = load_role_classifier()
if role_model is not None:
    print("[INFO] Модель классификации ролей загружена")
else:
    print("[INFO] Модель классификации ролей не загружена")
# Конец классификации

while True:
    print("[LOG] Ожидание нажатия клавиши q...")
    keyboard.wait("q")

    print("[LOG] q нажата. Начинаю запись...")
    # audio = record_audio()

    print("[LOG] Распознаю текст...")
    # result = model.transcribe(audio, fp16=False, language="ru")
    # text = result["text"].strip()
    text = 'Капитан туши пожар'
    print("[LOG] Распознанный текст:", text)

    # Классификация роли
    if role_model is not None:
        role_prediction = predict_role(role_model, role_tokenizer, label_encoder, text)
        print("------------------------------------")
        print(f"Роль: {role_prediction}")
        print("------------------------------------")
        save_role_to_file(role_prediction)
    else:
        print("------------------------------------")
        print("Модель ролей не загружена")
        print("------------------------------------")