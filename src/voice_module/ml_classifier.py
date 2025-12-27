import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Метки классов
LABEL_MAP = {
    0: "сражение",
    1: "Включение питания реактора",
    2: "Выключение питания реактора",
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

_tokenizer = None
_model = None

def _load_model():
    global _tokenizer, _model

    if _model is not None:
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "command_model")

    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModelForSequenceClassification.from_pretrained(model_path)
    _model.to("cpu")
    _model.eval()

def classify_with_model(text: str) -> str:
    _load_model()

    with torch.no_grad():
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )

        logits = _model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()

    return LABEL_MAP.get(pred_id, f"Класс {pred_id}")
