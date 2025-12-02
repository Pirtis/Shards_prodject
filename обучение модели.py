import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import json
import pickle


class RoleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class RoleClassifier(nn.Module):
    """
    Обертка для модели с сохранением конфигурации
    """

    def __init__(self, model_name='cointegrated/rubert-tiny', num_labels=8):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )


def prepare_data(csv_path='Barotrauma_dataset_full.csv'):
    """
    Подготовка данных для обучения
    """
    # Загружаем данные
    df = pd.read_csv(csv_path, encoding='utf-8')

    print(f"Загружено {len(df)} примеров")
    print("Первые 5 строк данных:")
    print(df.head())

    # Проверяем наличие нужных колонок
    if 'text' not in df.columns or 'specialization' not in df.columns:
        raise ValueError("В датасете должны быть колонки 'text' и 'specialization'")

    # Кодируем метки (роли)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(df['specialization'])

    # Создаем маппинг меток
    label_map = {i: label for i, label in enumerate(label_encoder.classes_)}

    print(f"\nНайдено {len(label_map)} уникальных ролей:")
    for idx, role in label_map.items():
        print(f"  {idx}: {role}")

    # Разделяем данные
    X_train, X_val, y_train, y_val = train_test_split(
        df['text'].values,
        labels_encoded,
        test_size=0.2,
        random_state=42,
        stratify=labels_encoded
    )

    print(f"\nРазмер тренировочной выборки: {len(X_train)}")
    print(f"Размер валидационной выборки: {len(X_val)}")

    return X_train, X_val, y_train, y_val, label_map, label_encoder


def train_model(X_train, y_train, X_val, y_val, label_map, model_name='cointegrated/rubert-tiny'):
    """
    Обучение модели
    """
    print(f"\nИспользуется модель: {model_name}")
    print(f"Количество классов: {len(label_map)}")

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Создаем нашу модель
    model = RoleClassifier(model_name=model_name, num_labels=len(label_map))

    # Создаем датасеты
    train_dataset = RoleDataset(X_train, y_train, tokenizer)
    val_dataset = RoleDataset(X_val, y_val, tokenizer)

    # Создаем DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Настройка обучения
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    model.to(device)

    # Используем torch.optim.AdamW
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 5

    # Тренировочные метрики
    train_losses = []
    val_losses = []
    accuracies = []

    # Обучение
    for epoch in range(num_epochs):
        print(f"\n{'=' * 50}")
        print(f"Эпоха {epoch + 1}/{num_epochs}")
        print('=' * 50)

        # Обучение
        model.train()
        train_loss = 0
        train_batches = 0

        for batch in tqdm(train_loader, desc="Обучение", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            train_loss += loss.item()
            train_batches += 1

            loss.backward()
            optimizer.step()

        # Валидация
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Валидация", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                # Вычисляем точность
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        accuracies.append(accuracy)

        print(f"\nРезультаты эпохи {epoch + 1}:")
        print(f"  Средняя loss обучения: {avg_train_loss:.4f}")
        print(f"  Средняя loss валидации: {avg_val_loss:.4f}")
        print(f"  Точность валидации: {accuracy:.4f}")

    # Выводим итоговые результаты
    print(f"\n{'=' * 50}")
    print("ИТОГИ ОБУЧЕНИЯ")
    print('=' * 50)
    print(f"Финальная точность: {accuracies[-1]:.4f}")
    print(f"Лучшая точность: {max(accuracies):.4f} (эпоха {accuracies.index(max(accuracies)) + 1})")

    return model, tokenizer


def save_model_pth(model, tokenizer, label_map, label_encoder, output_dir='role_model_pth'):
    """
    Сохранение модели в формате .pth
    """
    # Создаем директорию, если не существует
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем модель в формате .pth
    model_path = os.path.join(output_dir, 'role_classifier.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_name': model.model_name,
            'num_labels': model.num_labels
        },
        'label_map': label_map,
        'class_names': label_encoder.classes_.tolist()
    }, model_path)

    # Сохраняем токенизатор отдельно (через pickle)
    tokenizer_path = os.path.join(output_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    # Сохраняем маппинг меток в JSON для удобства
    with open(os.path.join(output_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # Создаем простой файл с кодом для использования модели (всё в одном)
    usage_code = '''# role_model_pth/usage_example.py
# Код для использования обученной модели

import torch
import pickle
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_role_model(model_dir='role_model_pth'):
    """
    Загрузка модели определения ролей из .pth файла

    Args:
        model_dir: путь к папке с моделью

    Returns:
        словарь с моделью, токенизатором и label_map
    """
    # Загружаем сохраненные данные
    checkpoint = torch.load(f'{model_dir}/role_classifier.pth', map_location='cpu')

    # Загружаем токенизатор
    with open(f'{model_dir}/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Загружаем конфигурацию модели
    model_config = checkpoint['model_config']

    # Создаем модель с той же архитектурой
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config['model_name'],
        num_labels=model_config['num_labels']
    )

    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cpu')
    model.eval()

    # Загружаем маппинг меток
    label_map = checkpoint['label_map']

    return {
        'model': model,
        'tokenizer': tokenizer,
        'label_map': label_map
    }

def predict_role(model_dict, text):
    """
    Предсказание роли для текста

    Args:
        model_dict: словарь с моделью, токенизатором и label_map
        text: текст для классификации

    Returns:
        tuple: (роль, уверенность)
    """
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    label_map = model_dict['label_map']

    with torch.no_grad():
        # Токенизация
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )

        # Предсказание
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_id].item()

        # Получаем название роли
        predicted_role = label_map.get(str(predicted_id), f"Класс {predicted_id}")

        return predicted_role, confidence

# Пример использования:
if __name__ == "__main__":
    # Загрузка модели
    print("Загрузка модели...")
    model_dict = load_role_model()

    # Тестовые примеры
    test_texts = [
        "Срочно, капитан, механика барахлит!",
        "Доктор, нужна медицинская помощь!",
        "Инженер, проверьте систему"
    ]

    print("\\nТестирование модели:")
    print("=" * 50)

    for text in test_texts:
        role, confidence = predict_role(model_dict, text)
        print(f"Текст: {text}")
        print(f"Роль: {role} (уверенность: {confidence:.2%})")
        print()
'''

    # Сохраняем код использования в папку модели
    usage_path = os.path.join(output_dir, 'usage_example.py')
    with open(usage_path, 'w', encoding='utf-8') as f:
        f.write(usage_code)

    # Создаем еще более простой вариант для вашего основного кода
    simple_loader_code = '''# role_model_pth/simple_loader.py
# Простейшая загрузка модели для вашего голосового модуля

import torch
import pickle

def load_role_classifier_simple(model_dir='role_model_pth'):
    """
    Простая загрузка модели - возвращает всё необходимое для работы
    """
    # Загружаем модель
    checkpoint = torch.load(f'{model_dir}/role_classifier.pth', map_location='cpu')

    # Загружаем токенизатор
    with open(f'{model_dir}/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Загружаем конфигурацию
    model_config = checkpoint['model_config']

    # Импортируем здесь, чтобы не было зависимостей в основном коде
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Создаем и загружаем модель
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config['model_name'],
        num_labels=model_config['num_labels']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cpu')
    model.eval()

    # Маппинг меток
    label_map = checkpoint['label_map']

    # Функция предсказания
    def predict(text):
        with torch.no_grad():
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            )

            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()

            predicted_role = label_map.get(str(predicted_id), f"Класс {predicted_id}")

            return predicted_role, confidence

    return predict

# Использование в вашем коде:
# from simple_loader import load_role_classifier_simple
# predict = load_role_classifier_simple()
# role, confidence = predict("Капитан, помогите!")
'''

    simple_loader_path = os.path.join(output_dir, 'simple_loader.py')
    with open(simple_loader_path, 'w', encoding='utf-8') as f:
        f.write(simple_loader_code)

    print(f"\nМодель сохранена в папку: {output_dir}")
    print("\nОсновные файлы:")
    print(f"  ✓ role_classifier.pth - основная модель (веса + конфиг)")
    print(f"  ✓ tokenizer.pkl - токенизатор")
    print(f"  ✓ label_map.json - маппинг меток")
    print(f"  ✓ usage_example.py - пример использования")
    print(f"  ✓ simple_loader.py - простой загрузчик для вашего кода")

    print("\nКак использовать в вашем голосовом модуле:")
    print("=" * 60)
    print("""
# 1. Скопируйте папку role_model_pth рядом с вашим скриптом
# 2. В начале вашего скрипта добавьте:

def load_role_model():
    import torch
    import pickle
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Загружаем модель
    checkpoint = torch.load('role_model_pth/role_classifier.pth', map_location='cpu')

    # Загружаем токенизатор
    with open('role_model_pth/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Создаем модель
    model_config = checkpoint['model_config']
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config['model_name'],
        num_labels=model_config['num_labels']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cpu')
    model.eval()

    label_map = checkpoint['label_map']

    return model, tokenizer, label_map

# 3. Используйте в вашем коде:

model, tokenizer, label_map = load_role_model()

def predict_role(text):
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )

        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_id].item()

        predicted_role = label_map.get(str(predicted_id), f"Класс {predicted_id}")

        return predicted_role, confidence

# 4. В основном цикле:
# role, confidence = predict_role(распознанный_текст)
""")
    print("=" * 60)


def test_model_with_examples(model, tokenizer, label_map):
    """
    Тестирование модели на примерах
    """
    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ НА ПРИМЕРАХ")
    print("=" * 50)

    test_examples = [
        "Срочно, капитан, механика барахлит, действуй!",
        "Капитан, всё гремит, почини механику!",
        "Механик, проверь двигатель!",
        "Доктор, нужна помощь!",
        "Инженер, система повреждена!"
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    for example in test_examples:
        with torch.no_grad():
            inputs = tokenizer(
                example,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            ).to(device)

            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()

            predicted_role = label_map.get(str(predicted_id), f"Класс {predicted_id}")

            print(f"\nПример: {example}")
            print(f"Предсказанная роль: {predicted_role}")
            print(f"Уверенность: {confidence:.2%}")

            # Показываем топ-3 предсказания
            top_k = 3
            probs, indices = torch.topk(probabilities[0], top_k)
            print(f"Топ-{top_k} предсказания:")
            for i in range(top_k):
                role = label_map.get(str(indices[i].item()), f"Класс {indices[i].item()}")
                prob = probs[i].item()
                print(f"  {i + 1}. {role}: {prob:.2%}")


def main():
    """
    Основная функция обучения
    """
    print("=" * 70)
    print("ОБУЧЕНИЕ МОДЕЛИ ОПРЕДЕЛЕНИЯ РОЛЕЙ ДЛЯ BAROTRAUMA")
    print("Формат сохранения: .pth")
    print("=" * 70)

    try:
        # Подготовка данных
        print("\n[1/4] ПОДГОТОВКА ДАННЫХ...")
        X_train, X_val, y_train, y_val, label_map, label_encoder = prepare_data()

        # Обучение модели
        print("\n[2/4] ОБУЧЕНИЕ МОДЕЛИ...")
        model, tokenizer = train_model(X_train, y_train, X_val, y_val, label_map)

        # Тестирование модели
        print("\n[3/4] ТЕСТИРОВАНИЕ МОДЕЛИ...")
        test_model_with_examples(model.transformer, tokenizer, label_map)

        # Сохранение модели в .pth формате
        print("\n[4/4] СОХРАНЕНИЕ МОДЕЛИ...")
        save_model_pth(model, tokenizer, label_map, label_encoder)

        print("\n" + "=" * 70)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 70)

        print("\n📁 Папка с моделью: role_model_pth")
        print("📄 Основной файл: role_classifier.pth")
        print("\nДля использования в вашем коде просто:")
        print("1. Скопируйте папку 'role_model_pth' рядом с вашим скриптом")
        print("2. Используйте код из simple_loader.py или напишите свой загрузчик")

    except FileNotFoundError as e:
        print(f"\n❌ ОШИБКА: Файл не найден: {e}")
        print("Убедитесь, что файл 'Barotrauma_dataset_full.csv' находится в текущей директории")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()