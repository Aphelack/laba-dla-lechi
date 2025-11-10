import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import nlp_processing
import spacy


class CustomTokenizer:
    def __init__(self, bert_tokenizer_name: str):
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_name)
        self.nlp = spacy.load("ru_core_news_sm")

    def spacy_tokenize(self, text: str):
        doc = self.nlp(text)
        return [token.text for token in doc]

    def __call__(self, text: str, max_length=128, padding='max_length',
                 truncation=True, return_tensors='pt', add_special_tokens=True):

        tokens = self.spacy_tokenize(text)

        encoding = self.bert_tokenizer(
            tokens,
            is_split_into_words=True,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors
        )
        return encoding

    def save_pretrained(self, path: str):
        self.bert_tokenizer.save_pretrained(path)


class BertClassifier(nn.Module):
    def __init__(self, bert_model_name: str, num_classes: int = 2,
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super().__init__()

        self.bert = BertForSequenceClassification.from_pretrained(
            bert_model_name, num_labels=num_classes
        )

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            bias="none"
        )

        self.bert = get_peft_model(self.bert, lora_config)

        self.bert.print_trainable_parameters()

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
