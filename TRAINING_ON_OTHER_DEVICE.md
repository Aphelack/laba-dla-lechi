# Инструкция для обучения на другом устройстве

## Подготовка на текущем устройстве

### 1. Генерация датасета (если еще не сделано)

```bash
source .venv/bin/activate
python generate_dataset.py
```

Будет создан файл `dataset.json` со 100 примерами (50 положительных, 50 отрицательных).

### 2. Копирование на другое устройство

Скопируйте следующие файлы на целевое устройство:

**Обязательные файлы для обучения:**
- `dataset.json` - датасет
- `models.py` - модели (CustomTokenizer, BertClassifier)
- `nlp_processing.py` - обработка текста
- `train_model.py` - скрипт обучения
- `requirements.txt` - зависимости
- `README.md` - документация

**Опционально (для полного понимания):**
- `generate_dataset.py` - как был создан датасет
- `conditions.md` - условия лабораторной
- `METHODOLOGY.md` - методология
- `SETUP_INSTRUCTION.md` - инструкция по установке

## Установка на новом устройстве

### 1. Системные требования

- Python 3.8 или выше
- pip (менеджер пакетов Python)
- 8 ГБ RAM минимум
- 5 ГБ свободного места на диске
- Интернет-соединение для загрузки моделей

### 2. Создание виртуального окружения

**Linux/Mac:**
```bash
cd путь/к/проекту
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
cd путь\к\проекту
python -m venv .venv
.venv\Scripts\activate
```

### 3. Установка зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Этот процесс займет 5-10 минут, будут установлены:
- PyTorch (~800 МБ)
- Transformers
- spaCy
- scikit-learn
- другие зависимости

### 4. Установка модели spaCy для русского языка

```bash
python -m spacy download ru_core_news_sm
```

### 5. Проверка датасета

```bash
python nlp_processing.py
```

Эта команда покажет примеры токенизации, лемматизации, частеречной разметки, синтаксических зависимостей и именованных сущностей из датасета.

### 6. Запуск обучения

```bash
python train_model.py
```

**Что происходит во время обучения:**

1. Загрузка датасета (dataset.json)
2. Разделение на обучающую (80%) и тестовую (20%) выборки
3. Загрузка модели BERT (DeepPavlov/rubert-base-cased-sentence)
4. Создание кастомного токенизатора с использованием spaCy для:
   - Токенизации
   - Лемматизации
   - Частеречной разметки (POS-tagging)
   - Распознавания именованных сущностей (NER)
5. Обучение на 5 эпохах
6. Оценка качества модели
7. Сохранение обученной модели

**Время обучения:**
- На CPU: 10-20 минут
- На GPU: 2-5 минут

**Результаты обучения:**

После завершения будут созданы файлы:
- `concert_classifier.pth` - веса обученной модели (~470 МБ)
- `tokenizer/` - папка с токенизатором
- `model_config.json` - конфигурация модели

### 7. Тестирование модели

```bash
python predict.py
```

Скрипт загрузит обученную модель и покажет примеры предсказаний на тестовых фразах.

## Возможные проблемы и решения

### Проблема: Ошибка при установке PyTorch

**Решение:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Проблема: Недостаточно памяти при обучении

**Решение:** Уменьшите batch_size в файле train_model.py

Найдите строку:
```python
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```

Измените на:
```python
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
```

И аналогично для test_loader.

### Проблема: Не загружается модель BERT

**Причина:** Нет интернет-соединения или заблокирован доступ к huggingface.co

**Решение:** 
1. Проверьте интернет-соединение
2. Попробуйте использовать VPN
3. Или предварительно скачайте модель на другом устройстве

### Проблема: Модель spaCy не найдена

**Решение:**
```bash
python -m spacy download ru_core_news_sm --user
```

## Использование обученной модели

После обучения модель можно использовать для предсказаний:

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

with open('model_config.json', 'r') as f:
    config = json.load(f)

bert_tokenizer = BertTokenizer.from_pretrained('tokenizer')
tokenizer = CustomTokenizer(bert_tokenizer)
device = torch.device('cpu')
model = BertClassifier(config['bert_model_name'], config['num_classes'])
model.load_state_dict(torch.load('concert_classifier.pth', map_location=device))
model.to(device)
model.eval()

text = "Хочу купить билет на концерт любимой группы"
encoding = tokenizer(text, add_special_tokens=True, max_length=128,
                     padding='max_length', truncation=True, return_tensors='pt')

with torch.no_grad():
    logits = model(encoding['input_ids'].to(device), 
                   encoding['attention_mask'].to(device))
    pred = torch.argmax(logits, dim=1).item()
    probs = torch.softmax(logits, dim=1)
    confidence = probs[0][pred].item()

result = "Желание посетить концерт" if pred == 1 else "Не связано с концертом"
print(f"{result} (уверенность: {confidence:.2%})")
```

## Перенос обученной модели на другое устройство

Для использования уже обученной модели на другом устройстве скопируйте:

1. `concert_classifier.pth`
2. `tokenizer/` (всю папку)
3. `model_config.json`
4. `predict.py`
5. `requirements.txt`

Установите зависимости и запустите:
```bash
pip install -r requirements.txt
python predict.py
```

## Дополнительная информация

### Архитектура модели

```
Входной текст
    ↓
Токенизация (spaCy)
    ↓
Лемматизация (spaCy)
    ↓
POS-теггинг (spaCy)
    ↓
NER (spaCy)
    ↓
Объединение результатов
    ↓
BERT Tokenizer
    ↓
BERT Encoder (rubert-base-cased-sentence)
    ↓
Pooler Output (768 размерность)
    ↓
Dropout (p=0.3)
    ↓
Linear Layer (768 → 2)
    ↓
Softmax
    ↓
Предсказание (0 или 1)
```

### Метрики качества

После обучения в консоли будут выведены:
- Accuracy (точность)
- Precision (точность по каждому классу)
- Recall (полнота)
- F1-score (гармоническое среднее precision и recall)

Хорошие значения: > 0.85 для всех метрик.

### Советы по улучшению качества

1. Увеличить размер датасета до 500-1000 примеров
2. Увеличить количество эпох обучения (до 10)
3. Использовать более крупную модель BERT
4. Добавить аугментацию данных
5. Настроить гиперпараметры (learning rate, batch size)

## Контакты

При возникновении вопросов обращайтесь к преподавателю или разработчику проекта.
