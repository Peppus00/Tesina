import sys
#!{sys.executable} -m pip install spacy
#!{sys.executable} -m pip install nltk
#!{sys.executable} -m pip install statsmodels
#!{sys.executable} -m pip install transformers
#!{sys.executable} -m pip install keras
#!{sys.executable} -m pip install tensorflow
#!{sys.executable} -m pip install xgboost
#!{sys.executable} -m pip install transformers[torch]
#!{sys.executable} -m pip install accelerate -U

import transformers
import accelerate
print(transformers.__version__)
print(accelerate.__version__)


#from Packages import *
df = pd.read_csv('result_df.csv')

import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np

# Funzione per pulire le etichette
def clean_labels(label):
    if pd.isna(label):
        return label
    label = str(label)
    label = label.strip("[]")
    label = label.replace('"', '')
    label = label.replace("'", "")
    return label

df['labels'] = df['labels'].astype(str).apply(clean_labels)
df = df[df['labels'].notnull()]
df = df[df['labels'] != 'nan']

mlb = MultiLabelBinarizer()
df['labels'] = df['labels'].apply(lambda x: x.split(','))

# Definizione della classe Dataset
class AmazonReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# Suddivisione in train, validation e test set
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Tokenizzazione
model_name = "indigo-ai/BERTino"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Tokenizza i dataset
train_encodings = tokenizer(list(train_df['reviewDescription']), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(list(val_df['reviewDescription']), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(test_df['reviewDescription']), truncation=True, padding=True, max_length=512)

# Crea i dataset
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_df['labels'])
val_labels = mlb.transform(val_df['labels'])
test_labels = mlb.transform(test_df['labels'])

train_dataset = AmazonReviewDataset(train_encodings, train_labels)
val_dataset = AmazonReviewDataset(val_encodings, val_labels)
test_dataset = AmazonReviewDataset(test_encodings, test_labels)

# Carica il modello
model = DistilBertForSequenceClassification.from_pretrained(model_name, 
                                                            num_labels=len(mlb.classes_),
                                                            problem_type="multi_label_classification")

from torch.utils.data import DataLoader

# Parametri di training senza early stopping
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=64,
    warmup_steps=100,
    weight_decay=0.1,
    learning_rate=5e-5,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",  # Strategia di valutazione
    save_strategy="epoch",        # Strategia di salvataggio da corrispondere con evaluation_strategy
    load_best_model_at_end=True   # Carica il miglior modello alla fine del training
)

from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(pred):
    logits = pred.predictions[0]  # Prende il primo elemento della tuple se predictions è una tuple
    preds = logits > 0.35  # Utilizza una soglia per convertire i logit in etichette binarie
    labels = pred.label_ids
    precision = precision_score(labels, preds, average='micro')
    recall = recall_score(labels, preds, average='micro')
    f1 = f1_score(labels, preds, average='micro')
    metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    print(metrics)
    return metrics


# Inizializzazione del Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Implementazione manuale dell'early stopping basato su F1
best_f1 = None
patience = 3
patience_counter = 0

for epoch in range(training_args.num_train_epochs):
    trainer.train()
    eval_results = trainer.evaluate()
    
    eval_f1 = eval_results["eval_f1"]  # Assicurati che 'eval_f1' sia il nome corretto
    
    if best_f1 is None or eval_f1 > best_f1:  # Confronto basato su F1
        best_f1 = eval_f1
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Stopping early based on F1 score at epoch {epoch+1}")
        break

# Funzione per trovare la soglia ottimale
def find_optimal_threshold(model, val_dataset):
    val_loader = DataLoader(val_dataset, batch_size=16)
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    precision, recall, thresholds = precision_recall_curve(all_labels.ravel(), all_logits.ravel())
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold

# Trova la soglia ottimale
optimal_threshold = find_optimal_threshold(model, val_dataset)
print(f"Optimal threshold: {optimal_threshold}")

# Puoi utilizzare optimal_threshold per fare predizioni sul test set o in applicazioni future

# Valutazione sul test set (opzionale)
test_results = trainer.evaluate(test_dataset)
print(test_results)





#PER SALVARE 
import json
# Salva il modello
model_save_path = "./modello_bertino_cluster"
trainer.save_model(model_save_path)

# Valuta il modello sul dataset di valutazione (o sul test set, se lo preferisci)
results = trainer.evaluate()

# Salva le metriche di valutazione in un file JSON
evaluation_metrics_path = "/modello_bertino_cluster/evaluation_metrics.json"
with open(evaluation_metrics_path, "w") as f:
    json.dump(results, f)

# Se desideri, puoi anche valutare il modello sul test set e salvare quelle metriche
# test_results = trainer.evaluate(test_dataset)
# test_metrics_path = "/path/to/test_metrics.json"
# with open(test_metrics_path, "w") as f:
#     json.dump(test_results, f)
