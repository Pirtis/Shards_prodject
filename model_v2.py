from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate
import numpy as np
import os
import csv

df = pd.read_csv("dataset_pachanil.csv", encoding='utf-8-sig')
print(f"Загружено {len(df)} строк, классов: {df['label'].nunique()}")

labels = df['label'].astype('category').cat.categories
os.makedirs("command_model", exist_ok=True)
np.save("command_model/label_names.npy", labels)

df['label'] = df['label'].astype('category').cat.codes
num_labels = df['label'].nunique()

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)


tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=num_labels)

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding=True, max_length=64)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)


accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    preds, labels_eval = eval_pred
    preds = np.argmax(preds, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels_eval)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels_eval, average="weighted")["f1"]
    }


class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, log_file="training_metrics.csv"):
        self.log_file = log_file
        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "train_loss", "eval_loss", "eval_accuracy", "eval_f1"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                getattr(state, "epoch", ""),
                getattr(state, "global_step", ""),
                logs.get("loss", ""),
                logs.get("eval_loss", ""),
                logs.get("eval_accuracy", ""),
                logs.get("eval_f1", "")
            ])


args = TrainingArguments(
    output_dir="out",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    eval_strategy="epoch",
    save_strategy="no",
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[MetricsLoggerCallback]
)


print("TRAINING.....................")
trainer.train()


trainer.save_model("command_model")
tokenizer.save_pretrained("command_model")

print("NOW REALLY VSE")
