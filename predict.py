import torch
import sys
import json
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel, PeftConfig
from models import CustomTokenizer


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
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence


def main():
    print("Загрузка конфигурации модели...")
    with open("model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    tokenizer = CustomTokenizer(config["bert_model_name"])

    peft_model_dir = "concert_classifier_peft"

    peft_config = PeftConfig.from_pretrained(peft_model_dir)
    base_model = BertForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        num_labels=config["num_classes"]
    )
    model = PeftModel.from_pretrained(base_model, peft_model_dir)
    model.to(device)
    model.eval()

    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        pred, confidence = predict(input_text, model, tokenizer, device)
        label = "Желание посетить концерт" if pred == 1 else "Не связано с концертом"
        print(f"Текст: {input_text}")
        print(f"Предсказание: {label}")
        print(f"Уверенность: {confidence:.4f}")
    else:
        raise

if __name__ == "__main__":
    main()
