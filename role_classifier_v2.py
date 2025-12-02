import torch
import pickle
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("====================================")
print("  Barotrauma Role Classifier")
print("====================================")

# Загрузка модели
checkpoint = torch.load('role_classifier.pth', map_location='cpu')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
label_map = json.load(open('label_map.json', 'r', encoding='utf-8'))

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint['model_config']['model_name'],
    num_labels=checkpoint['model_config']['num_labels']
)

# Исправление ключей
state_dict = checkpoint['model_state_dict']
fixed_dict = {k.replace('transformer.', ''): v for k, v in state_dict.items()}
model.load_state_dict(fixed_dict, strict=False)
model.eval()
model.to('cpu')

print("Модель загружена успешно!")
print("\n" + "=" * 40)


def predict(text):
    """Предсказание роли для текста"""
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )

        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs).item()
        confidence = probs[0][pred_id].item()
        role = label_map.get(str(pred_id), f"Класс {pred_id}")

        return role, confidence


def save_result(role, confidence, filename="out.txt"):
    """Сохранение результата"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{role}")
    return True


# Основной цикл
print("\nИнструкция:")
print("1. Введите текст команды для классификации")
print("2. Введите 'exit' для выхода")
print("-" * 40)

while True:
    user_input = input("\nВведите текст команды: ").strip()

    if not user_input:
        continue

    if user_input.lower() == 'exit':
        print("Выход из программы.")
        break

    else:
        text = user_input

    # Предсказание
    role, confidence = predict(text)

    print(f"\n{'=' * 30}")
    print(f"Текст: {text}")
    print(f"Роль: {role}")
    print(f"Уверенность: {confidence:.2%}")
    print('=' * 30)

    # Сохранение
    if save_result(role, confidence):
        print("✓ Результат сохранен в out.txt")