from pathlib import Path
from typing import Any, Dict, List, Text

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import google.generativeai as genai
import posthog
from dotenv import load_dotenv
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from actions.query_writer import QueryRewriter
from actions.services import (
    ChatHistoryBuilder,
    ChromaRepository,
    ContextBuilder,
    GeminiAnswerService,
    RetrievalService,
)

posthog.disabled = True
posthog.capture = lambda *args, **kwargs: None

load_dotenv()

GEN_API_KEY = os.getenv("GEN_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_PATH = Path(
    os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db_rebuilt"))
).resolve()
LEGACY_CHROMA_DB_PATH = (BASE_DIR / "chroma_db").resolve()

print(f"GEN_API_KEY: {GEN_API_KEY}")
print(f"LLM_MODEL: {LLM_MODEL}")
print("DB path:", CHROMA_DB_PATH)

genai.configure(api_key=GEN_API_KEY.strip())


class ActionRAGRetrieval(Action):
    """Hanh dong lay du lieu tu ChromaDB va tra loi bang Gemini API."""

    def __init__(self):
        rewriter = QueryRewriter()
        repository = ChromaRepository(
            db_path=CHROMA_DB_PATH,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_model_name=EMBEDDING_MODEL,
            legacy_db_path=LEGACY_CHROMA_DB_PATH,
        )

        self.history_builder = ChatHistoryBuilder()
        self.retrieval_service = RetrievalService(repository=repository, rewriter=rewriter)
        self.context_builder = ContextBuilder(self.retrieval_service)
        self.answer_service = GeminiAnswerService(LLM_MODEL)

    def name(self) -> Text:
        return "action_rag_retrieval"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_message, chat_history = self.history_builder.build(tracker)

        if not user_message:
            dispatcher.utter_message(text="Xin loi, toi khong hieu cau hoi cua ban.")
            return []

        try:
            collection = self.retrieval_service.repository.load_collection()
            if collection is None:
                dispatcher.utter_message(
                    text=(
                        "Xin loi, he thong chua san sang. "
                        "ChromaDB dang loi duong dan hoac khong tuong thich schema, "
                        "vui long kiem tra lai du lieu va version chromadb."
                    )
                )
                return []

            docs = self.retrieval_service.retrieve(user_message, top_k=8)
            print(f"Found {len(docs)} documents from ChromaDB.")

            context = self.context_builder.build(docs)
            response = self.answer_service.generate_answer(
                user_message,
                context,
                chat_history,
            )
            dispatcher.utter_message(text=response)

        except Exception as exc:
            print(f"[ERROR action_rag_retrieval]: {exc}")
            dispatcher.utter_message(
                text="Xin loi, da xay ra loi khi xu ly cau hoi cua ban."
            )

        return []


if __name__ == "__main__":
    rewriter = QueryRewriter()
    test_query = "AI la gi?"
    print("Original query:", test_query)
    print("Rewritten query:", rewriter.rewrite(test_query))
