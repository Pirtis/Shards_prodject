from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback, DataCollatorWithPadding
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate
import numpy as np
import os
import csv

df = pd.read_csv("dataset.csv", encoding='utf-8-sig')

labels = df['label'].astype('category').cat.categories
df['label'] = df['label'].astype('category').cat.codes
num_labels = len(labels)

os.makedirs("command_model_rubert_X_v3", exist_ok=True)
np.save("command_model_rubert_X_v3/label_names.npy", labels)

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModelForSequenceClassification.from_pretrained("cointegrated/rubert-tiny2", num_labels=num_labels)

def tokenize(batch):

    return tokenizer(batch['text'], truncation=True, padding=True, max_length=64)

train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    acc = evaluate.load("accuracy").compute(predictions=preds, references=labels)
    f1 = evaluate.load("f1").compute(predictions=preds, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

class MetricsLogger(TrainerCallback):
    def __init__(self, log_file="training_metrics.csv"):
        self.log_file = log_file
        self.last_train_loss = None

        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "eval_loss", "eval_accuracy", "eval_f1"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs and "eval_loss" not in logs:
            self.last_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        epoch = state.epoch

        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                self.last_train_loss,
                metrics.get("eval_loss"),
                metrics.get("eval_accuracy"),
                metrics.get("eval_f1")
            ])


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

args = TrainingArguments(
    output_dir="out",
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), MetricsLogger()]
)

trainer.train()
trainer.save_model("command_model_rubert_X_v3")
tokenizer.save_pretrained("command_model_rubert_X_v3")

