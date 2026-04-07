from typing import List, Tuple

from rasa_sdk import Tracker


class ChatHistoryBuilder:
    def build(self, tracker: Tracker) -> Tuple[str, str]:
        user_message = tracker.latest_message.get("text", "")
        history_list: List[str] = []

        for event in tracker.events:
            if event.get("event") == "user":
                history_list.append(f"Nguoi dung: {event.get('text')}")
            elif event.get("event") == "bot":
                history_list.append(f"Bot: {event.get('text')}")

        chat_history = (
            "\n".join(history_list[-5:-1])
            if len(history_list) > 1
            else "Chua co lich su tro chuyen."
        )
        return user_message, chat_history
