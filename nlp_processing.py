import spacy
import json

nlp = spacy.load('ru_core_news_sm')

def tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]

def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

def pos_tagging(text):
    doc = nlp(text)
    return [(token.text, token.pos_, token.tag_) for token in doc]

def dependency_parsing(text):
    doc = nlp(text)
    return [(token.text, token.dep_, token.head.text) for token in doc]

def named_entity_recognition(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def process_text(text):
    tokens = tokenize(text)
    lemmas = lemmatize(text)
    pos_tags = pos_tagging(text)
    dependencies = dependency_parsing(text)
    entities = named_entity_recognition(text)
    
    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "dependencies": dependencies,
        "entities": entities
    }

def main():
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print("Анализ нескольких примеров из датасета:\n")
    
    for i, item in enumerate(dataset[:3]):
        print(f"Пример {i+1}: {item['text']}")
        print(f"Метка: {'Положительный (концерт)' if item['label'] == 1 else 'Отрицательный'}\n")
        
        result = process_text(item['text'])
        
        print("Токены:", result['tokens'])
        print("Леммы:", result['lemmas'])
        print("Части речи:", [(t, p) for t, p, _ in result['pos_tags']])
        print("Зависимости:", result['dependencies'][:5])
        print("Именованные сущности:", result['entities'])
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
