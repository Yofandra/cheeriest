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
