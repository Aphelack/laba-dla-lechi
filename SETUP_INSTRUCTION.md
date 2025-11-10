# Инструкция по установке и запуску обученной модели

## Системные требования

- Python 3.8 или выше
- 4 ГБ свободного места на диске
- 8 ГБ оперативной памяти (рекомендуется)
- GPU не требуется (модель работает на CPU)

## Шаг 1: Клонирование репозитория

```bash
git clone <URL_репозитория>
cd laba-dla-lechi
```

## Шаг 2: Создание виртуального окружения

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Для Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

## Шаг 3: Установка зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Шаг 4: Установка модели spaCy для русского языка

```bash
python -m spacy download ru_core_news_sm
```

## Шаг 5: Проверка установки

Проверьте, что все файлы модели на месте:
- `concert_classifier.pth` - веса обученной модели
- `tokenizer/` - папка с токенизатором
- `model_config.json` - конфигурация модели
- `dataset.json` - датасет для обучения

## Шаг 6: Запуск предсказаний

```bash
python predict.py
```

Программа загрузит модель и выполнит предсказания на тестовых примерах.

## Использование модели в своем коде

```python
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import json
import nlp_processing

class CustomTokenizer:
    def __init__(self, bert_tokenizer):
        self.bert_tokenizer = bert_tokenizer
    
    def preprocess_with_spacy(self, text):
        result = nlp_processing.process_text(text)
        lemmas = ' '.join(result['lemmas'])
        pos_info = ' '.join([f"{pos}" for _, pos, _ in result['pos_tags']])
        entities = ' '.join([ent for ent, _ in result['entities']])
        processed_text = f"{lemmas} {pos_info} {entities}"
        return processed_text
    
    def __call__(self, text, add_special_tokens=True, max_length=128, 
                 padding='max_length', truncation=True, return_tensors='pt'):
        processed_text = self.preprocess_with_spacy(text)
        return self.bert_tokenizer(
            processed_text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )

class BertClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes=2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        logits = self.linear(dropout_output)
        return logits

with open('model_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

bert_tokenizer = BertTokenizer.from_pretrained('tokenizer')
tokenizer = CustomTokenizer(bert_tokenizer)
device = torch.device('cpu')
model = BertClassifier(config['bert_model_name'], config['num_classes'])
model.load_state_dict(torch.load('concert_classifier.pth', map_location=device))
model.to(device)
model.eval()

text = "Хочу купить билет на концерт"
encoding = tokenizer(text, add_special_tokens=True, max_length=128, 
                     padding='max_length', truncation=True, return_tensors='pt')

with torch.no_grad():
    logits = model(encoding['input_ids'], encoding['attention_mask'])
    pred = torch.argmax(logits, dim=1).item()

print("Желание посетить концерт" if pred == 1 else "Не связано с концертом")
```

## Возможные проблемы и решения

### Ошибка при установке torch

Если возникают проблемы с установкой PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Ошибка при загрузке модели BERT

Если модель не загружается, проверьте интернет-соединение и повторите попытку.

### Недостаточно памяти

Уменьшите batch_size в коде или используйте меньшую модель BERT.

## Структура проекта

```
laba-dla-lechi/
├── concert_classifier.pth      # Обученная модель
├── tokenizer/                  # Токенизатор BERT
├── model_config.json           # Конфигурация
├── dataset.json                # Датасет
├── models.py                   # Модели (CustomTokenizer, BertClassifier)
├── nlp_processing.py           # Анализ текста с spaCy
├── predict.py                  # Скрипт предсказаний
├── train_model.py              # Скрипт обучения
├── requirements.txt            # Зависимости
└── README.md                   # Документация
```

## Контакты

При возникновении вопросов обращайтесь к разработчику проекта.
