import torch
import torch.nn as nn
from transformers import BertModel
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
    
    def save_pretrained(self, path):
        self.bert_tokenizer.save_pretrained(path)

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
