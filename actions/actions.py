from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

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
            "makasih": "Sama-sama!!",
            "sampai jumpa": "Sampai jumpa!",
            "adios": "Adios!!",
            "terimakasih": "Sama-sama!!",
            "wassalamualaikum": "Waalaikumsalam",
        }

        response = goodbyes.get(last_greet, "Maaf, saya tidak mengerti. Bisa ulangi?")

        dispatcher.utter_message(text=response)
        return []