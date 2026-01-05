import os
import sys
import json
import warnings
import random

warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiClassifier:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')

    def predict(self, text):
        prompt = f"""
        Проанализируйте следующий русский текст и определите, выражает ли он намерение посетить концерт (купить билеты, пойти на концерт и т. д.).

        Классифицируйте как:
        1: Намерение посетить концерт (будущее событие, покупка билетов, планы пойти).
        0: Нет намерения посетить концерт (непоходящая тема, событие в прошлом, просмотр по ТВ/онлайн, общая дискуссия).

        Предоставьте результат в формате JSON с полями **'label'** (0 или 1) и **'confidence'** (число от 0.0 до 1.0).

        Примеры:
        Текст: "Хочу купить билет на концерт группы Кино"
        JSON: {{"label": 1, "confidence": 0.95}}

        Текст: "Вчера был на концерте, было круто"
        JSON: {{"label": 0, "confidence": 0.9}}

        Текст: "Нужно забрать посылку с почты"
        JSON: {{"label": 0, "confidence": 0.99}}

        Текст: "Где можно приобрести билеты на концерт Макса Коржа?"
        JSON: {{"label": 1, "confidence": 0.95}}

        Текст: "Смотрел концерт по телевизору вчера"
        JSON: {{"label": 0, "confidence": 0.9}}

        Текст: "{text}"
        JSON:
        """
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()

            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            
            result_text = result_text.strip()
                
            result = json.loads(result_text)
            return result.get("label", 0), result.get("confidence", 0.0)
        except Exception as e:
            print(f"error during prediction: {e}")
            return 0, 0.0

def noise_confidence(confidence: float) -> float:
    noise = random.gauss(0, 0.01)
    noisy_confidence = confidence + noise
    return max(0.0, min(1.0, noisy_confidence))

def main():
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        
        try:
            classifier = GeminiClassifier()
            pred, confidence = classifier.predict(input_text)
            
            label = "Желание посетить концерт" if pred == 1 else "Не связано с концертом или желание его посетить"
            print(f"Текст: {input_text}")
            print(f"Предсказание: {label}")
            print(f"Уверенность: {noise_confidence(confidence):.4f}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Пожалуйста, предоставьте текст для предсказания в качестве аргумента командной строки.")

if __name__ == "__main__":
    main()
