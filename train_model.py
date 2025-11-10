import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, criterion, device):
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
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, predictions), classification_report(true_labels, predictions)

def main():
    print("Загрузка датасета...")
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    texts = [item['text'] for item in dataset]
    labels = [item['label'] for item in dataset]
    print(f"Всего примеров: {len(texts)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Обучающая выборка: {len(X_train)}")
    print(f"Тестовая выборка: {len(X_test)}")
    
    bert_model_name = 'DeepPavlov/rubert-base-cased-sentence'
    print(f"\nЗагрузка токенизатора и модели BERT: {bert_model_name}")
    
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    tokenizer = CustomTokenizer(bert_tokenizer)
    print("Создан кастомный токенизатор с использованием spaCy")
    print("Обработка включает: токенизацию, лемматизацию, POS-теги, именованные сущности")
    
    train_dataset = ConcertDataset(X_train, y_train, tokenizer)
    test_dataset = ConcertDataset(X_test, y_test, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    model = BertClassifier(bert_model_name)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    num_epochs = 5
    print(f"\nНачало обучения на {num_epochs} эпох...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc, _ = evaluate(model, test_loader, device)
        print(f"Эпоха {epoch+1}/{num_epochs}")
        print(f"  Потери на обучении: {train_loss:.4f}")
        print(f"  Точность на тесте: {test_acc:.4f}")
    
    print("\nФинальная оценка модели:")
    test_acc, test_report = evaluate(model, test_loader, device)
    print(f"Точность: {test_acc:.4f}")
    print("\nОтчет классификации:")
    print(test_report)
    
    print("\nСохранение модели...")
    torch.save(model.state_dict(), 'concert_classifier.pth')
    tokenizer.save_pretrained('tokenizer')
    with open('model_config.json', 'w', encoding='utf-8') as f:
        json.dump({
            'bert_model_name': bert_model_name,
            'num_classes': 2,
            'max_length': 128
        }, f, ensure_ascii=False, indent=2)
    print("Модель сохранена в concert_classifier.pth")
    print("Токенизатор сохранен в папку tokenizer/")
    print("Конфигурация сохранена в model_config.json")

if __name__ == "__main__":
    main()
