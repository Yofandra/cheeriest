from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import SentenceTransformer, util
import json
import os
import random
import numpy as np
import torch
import re
import logging

class ActionProvideMotivation(Action):
    def name(self) -> Text:
        return "action_provide_motivation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        motivational_quotes = [
            "Kamu lebih kuat dari yang kamu kira.",
            "Hari buruk tidak berarti kehidupan yang buruk.",
            "Semua ini akan berlalu, percayalah."
        ]
        dispatcher.utter_message(text=random.choice(motivational_quotes))
        return []

class ActionGreetUser(Action):
    def name(self) -> Text:
        return "action_greet_user"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        last_greet = tracker.latest_message.get("text").lower()

        greetings = {
            "hey": "Hello!!",
            "hello": "Haii!!",
            "hi": "Halo!!",
            "hai": "Halo!!",
            "halo": "Hai!!",
            "yo": "Yo!",
            "sup": "Sup?",
            "holla": "Holla!!",
            "bonjour": "Bonjour!!",
            "hola": "Hola!!",
            "ciao": "Ciao!!",
            "namaste": "Namaste!!",
            "salam": "Salam!!",
            "konnichiwa": "Konnichiwa!!",
            "anyeong": "Anyeong!!",
            "annyeong": "Annyeong!!",
            "aloha": "Aloha!!",
            "assalamualaikum": "Waalaikumsalam",
        }

        response = greetings.get(last_greet, "Maaf, saya tidak mengerti. Bisa ulangi?")

        dispatcher.utter_message(text=response)
        return []

class ActionByeUser(Action):
    def name(self) -> Text:
        return "action_bye_user"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        last_greet = tracker.latest_message.get("text").lower()

        goodbyes = {
            "good bye": "Good bye!!",
            "bye": "Bye!!",
            "goodbye": "Good Bye!!",
            "sampai jumpa": "Sampai jumpa!",
            "adios": "Adios!!",
            "wassalamualaikum": "Waalaikumsalam",
        }

        response = goodbyes.get(last_greet, "Maaf, saya tidak mengerti. Bisa ulangi?")

        dispatcher.utter_message(text=response)
        return []
    
class ActionSemanticResponse(Action):
    def name(self) -> Text:
        return "action_semantic_response"

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.data_path = 'dataset/dataset_psych8k.json'
        self.embedding_path = 'dataset/dataset_psych8k_embeddings.npy'
        self.threshold = 0.6  # Threshold untuk skor kemiripan
        self.messages = []
        self.responses = []
        self.embeddings = None
        self._load_data()

    def _load_data(self):
        # Memuat data pesan dan respons
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        def normalize_text(text):
            text = re.sub(r'[^\w\s]', '', text)
            return text.strip().lower()
        self.messages = [normalize_text(item['message']) for item in data]
        self.responses = [item['response'] for item in data]

        # Memuat atau menghitung embedding
        if os.path.exists(self.embedding_path):
            self.embeddings = np.load(self.embedding_path)
        else:
            self.embeddings = self.model.encode(self.messages, convert_to_numpy=True)
            np.save(self.embedding_path, self.embeddings)

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        def normalize_text(text):
            text = re.sub(r'[^\w\s]', '', text)
            return text.strip().lower()
        user_input = normalize_text(tracker.latest_message.get('text'))
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)

        # Menghitung skor kemiripan
        scores = util.cos_sim(user_embedding, self.embeddings)[0]
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]

        if best_score >= self.threshold:
            best_response = self.responses[best_idx]
            dispatcher.utter_message(text=best_response)
        else:
            dispatcher.utter_message(text="Maaf, saya tidak menemukan jawaban yang sesuai. Silakan coba pertanyaan lain.")

        return []