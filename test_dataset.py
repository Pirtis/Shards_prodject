from transformers import BertTokenizer, BertForSequenceClassification,DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report

model_path = "command_model_rubert_X_v3"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

labels = np.load(f"{model_path}/label_names.npy", allow_pickle=True)

def predict_batch(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=64)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).numpy()
        predictions = np.argmax(probabilities, axis=-1)

    return predictions


import pandas as pd

test_df = pd.read_csv("test_dataset.csv")


true_labels = test_df['label'].astype('category').cat.codes.values
texts = test_df['text'].tolist()

# Предсказания модели
pred_labels = predict_batch(texts)


true_labels_str = [labels[i] for i in true_labels]
pred_labels_str = [labels[i] for i in pred_labels]


accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, average='weighted')
precision, recall, f1_per_class, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None)

print(f"Accuracy: {accuracy:.3f}")
print(f"F1 (weighted): {f1:.3f}")
print("\nDetailed Classification Report:")
print(classification_report(true_labels_str, pred_labels_str, target_names=labels))