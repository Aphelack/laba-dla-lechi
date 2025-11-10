# Лабораторная работа №1: Обработка естественного языка

## Вариант 5: Желание посетить концерт

Проект реализует систему обработки естественного языка для определения желания пользователя посетить концерт.

## Описание

Система анализирует текст на русском языке и определяет, выражает ли он намерение посетить концерт. 

Используются следующие методы обработки естественного языка:
- Токенизация
- Лемматизация
- Частеречная разметка (POS-tagging)
- Разбор синтаксических зависимостей
- Распознавание именованных сущностей (NER)

## Быстрый старт

### Обучение на своей машине

```bash
git clone <URL>
cd laba-dla-lechi
git checkout dataset-gen
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download ru_core_news_sm
python generate_dataset.py
python nlp_processing.py
python train_model.py
```

### Перенос на другое устройство

После обучения скопируйте следующие файлы на другое устройство:
- `concert_classifier.pth`
- `tokenizer/` (вся папка)
- `model_config.json`
- `predict.py`
- `requirements.txt`

На новом устройстве:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict.py
```

## Структура файлов

- `generate_dataset.py` - генерация обучающего датасета
- `nlp_processing.py` - демонстрация работы spaCy (токенизация, лемматизация, POS, зависимости, NER)
- `models.py` - определение моделей (CustomTokenizer, BertClassifier)
- `train_model.py` - обучение модели классификации
- `predict.py` - использование обученной модели для предсказаний
- `dataset.json` - сгенерированный датасет (100 примеров)

## Примеры использования

### Анализ текста с помощью spaCy

```bash
python nlp_processing.py
```

Вывод покажет токенизацию, леммы, части речи, зависимости и именованные сущности.

### Обучение модели

```bash
python train_model.py
```

### Предсказание

```bash
python predict.py
```

## Документация

- `SETUP_INSTRUCTION.md` - подробная инструкция по установке и запуску
- `METHODOLOGY.md` - описание методологии выполнения работы

## Технологии

- Python 3.8+
- spaCy - обработка естественного языка
- PyTorch - глубокое обучение
- Transformers - модель BERT
- scikit-learn - метрики качества

## Требования

Минимальные:
- Python 3.8+
- 4 ГБ свободного места
- 8 ГБ RAM

Рекомендуемые:
- Python 3.10+
- 8 ГБ свободного места
- 16 ГБ RAM
- GPU (опционально, ускоряет обучение)
