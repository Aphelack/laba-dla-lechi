import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import nlp_processing
import spacy
from spellchecker import SpellChecker
import os
import requests
import time
from artists_api import MusicGroupChecker

class CustomTokenizer:
    def __init__(self, bert_tokenizer_name: str, 
                 dictionary_path: str = "russian_proper_dict.txt",
                 use_api: bool = True):
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_name)
        self.nlp = spacy.load("ru_core_news_sm")
        self.spell_checker = SpellChecker(language='ru')
        self._load_custom_dictionary(dictionary_path)
        
        self.use_api = use_api
        if use_api:
            self.music_checker = MusicGroupChecker()

        self.static_groups = set()
        self._load_static_groups()

    def _load_custom_dictionary(self, dictionary_path: str):
        """Load custom dictionary for spell checking"""
        if os.path.exists(dictionary_path):
            try:
                with open(dictionary_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                words = []
                for line in lines:
                    parts = line.strip().split('\t')
                    if parts and parts[0]:
                        words.append(parts[0])
                self.spell_checker.word_frequency.load_words(words)
                print(f"Loaded {len(words)} words from custom dictionary")
            except Exception as e:
                print(f"Error loading dictionary: {e}")

    def _load_static_groups(self):
        static_file = "music_groups.txt"
        with open(static_file, 'r', encoding='utf-8') as f:
            groups = [line.strip().lower() for line in f if line.strip()]
        self.static_groups = set(groups)

    def spacy_tokenize(self, text: str):
        doc = self.nlp(text)
        return [token.text for token in doc]

    def preprocess_with_spacy(self, text: str):
        result = nlp_processing.process_text(text)
        tokens = result['lemmas']
        return tokens

    def correct_spelling(self, text: str) -> str:
        words = text.split()
        corrected_words = []
        
        for word in words:
            if word not in self.spell_checker:
                correction = self.spell_checker.correction(word)
                if correction and correction != word:
                    corrected_words.append(correction)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)

    def replace_music_groups(self, text: str) -> str:
        """Simple music group replacement - check all words and pairs"""
        
        words = text.split()
        if len(words) < 2:
            return text

        single_word_groups = []
        for i, word in enumerate(words):
            if len(word) > 2:
                if self._is_music_group(word):
                    single_word_groups.append(i)

        pair_groups = []
        for i in range(len(words) - 1):
            pair = f"{words[i]} {words[i+1]}"
            if self._is_music_group(pair):
                pair_groups.append((i, i+1))

        for i in sorted(single_word_groups, reverse=True):
            skip = False
            for start, end in pair_groups:
                if i >= start and i <= end:
                    skip = True
                    break
            if not skip:
                words[i] = "музыкальная_группа"

        for start, end in sorted(pair_groups, key=lambda x: x[0], reverse=True):
            words[start:end+1] = ["музыкальная_группа"]
        
        return ' '.join(words)

    def _is_music_group(self, name: str) -> bool:
        """Check if a name is a music group"""

        if name.lower() in self.static_groups:
            return True
        
        if self.use_api:
            try:
                return self.music_checker.check_group(name)
            except:
                return name.lower() in self.static_groups
    
        return name.lower() in self.static_groups

    def __call__(self, text: str, max_length=128, padding='max_length',
                 truncation=True, return_tensors='pt', add_special_tokens=True, 
                 correct_spelling: bool = True):
        if correct_spelling:
            text = self.correct_spelling(text)

        text = self.replace_music_groups(text)
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
