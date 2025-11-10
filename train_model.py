import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import torch.nn as nn
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from models import CustomTokenizer, BertClassifier

class ConcertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    return accuracy_score(trues, preds), classification_report(trues, preds, digits=4)

def main():
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    texts = [item['text'] for item in dataset]
    labels = [item['label'] for item in dataset]
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    bert_model_name = 'DeepPavlov/rubert-base-cased-sentence'
    tokenizer = CustomTokenizer(bert_model_name)
    train_dataset = ConcertDataset(X_train, y_train, tokenizer)
    test_dataset = ConcertDataset(X_test, y_test, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertClassifier(bert_model_name)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    total_steps = len(train_loader) * 5
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)
    for epoch in range(5):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        test_acc, _ = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/5 | Train loss: {train_loss:.4f} | Test acc: {test_acc:.4f}")
    test_acc, report = evaluate(model, test_loader, device)
    print(f"\nFinal accuracy: {test_acc:.4f}\n{report}")
    model.bert.save_pretrained("concert_classifier_peft")
    tokenizer.save_pretrained("tokenizer")

    with open("model_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "bert_model_name": bert_model_name,
            "num_classes": 2,
            "max_length": 128
        }, f, ensure_ascii=False, indent=2)

    print("LoRA adapters saved to concert_classifier_lora/")
    print("Tokenizer saved to tokenizer/")
    print("Config saved to model_config.json")


if __name__ == "__main__":
    main()
