import torch
from transformers import BertTokenizer
import json
from models import CustomTokenizer, BertClassifier

def predict(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence

def main():
    print("Загрузка конфигурации модели...")
    with open('model_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("Загрузка токенизатора...")
    bert_tokenizer = BertTokenizer.from_pretrained('tokenizer')
    tokenizer = CustomTokenizer(bert_tokenizer)
    print("Создан кастомный токенизатор с использованием spaCy")
    
    print("Загрузка модели...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertClassifier(config['bert_model_name'], config['num_classes'])
    model.load_state_dict(torch.load('concert_classifier.pth', map_location=device))
    model.to(device)
    model.eval()
    print(f"Модель загружена и готова к использованию (устройство: {device})\n")
    
    test_texts = [
        "Хочу купить билет на концерт Metallica",
        "Нужно сходить в магазин за продуктами",
        "Есть билеты на концерт в эту субботу?",
        "Завтра иду на работу рано утром",
        "Планирую посетить концерт классической музыки"
    ]
    
    print("Примеры предсказаний:\n")
    for text in test_texts:
        pred, confidence = predict(text, model, tokenizer, device)
        label = "Желание посетить концерт" if pred == 1 else "Не связано с концертом"
        print(f"Текст: {text}")
        print(f"Предсказание: {label}")
        print(f"Уверенность: {confidence:.4f}\n")

if __name__ == "__main__":
    main()
