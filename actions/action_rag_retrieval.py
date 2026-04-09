from pathlib import Path
from typing import Any, Dict, List, Text
import os

# ===== ENV SETUP =====
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

# Disable tracking
posthog.disabled = True
posthog.capture = lambda *args, **kwargs: None

load_dotenv()

# ===== ENV VARIABLES =====
GEN_API_KEY = os.getenv("GEN_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_PATH = Path(
    os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db_rebuilt"))
).resolve()
LEGACY_CHROMA_DB_PATH = (BASE_DIR / "chroma_db").resolve()

# Safe config (tránh crash nếu thiếu key)
if GEN_API_KEY:
    genai.configure(api_key=GEN_API_KEY.strip())
else:
    print("[WARNING] GEN_API_KEY is not set!")

print(f"[INFO] LLM_MODEL: {LLM_MODEL}")
print(f"[INFO] CHROMA_DB_PATH: {CHROMA_DB_PATH}")


# ===== ACTION =====
class ActionRAGRetrieval(Action):
    """RAG retrieval + Gemini answer (Production-ready, lazy load)."""

    def __init__(self):
        # ❗ KHÔNG load nặng ở đây
        self.initialized = False
        self.history_builder = ChatHistoryBuilder()

        # lazy objects
        self.retrieval_service = None
        self.context_builder = None
        self.answer_service = None

    def name(self) -> Text:
        return "action_rag_retrieval"

    def _lazy_init(self):
        """Khởi tạo RAG pipeline khi cần (tránh block startup)."""
        if self.initialized:
            return

        print("[INFO] Initializing RAG pipeline...")

        try:
            rewriter = QueryRewriter()

            repository = ChromaRepository(
                db_path=CHROMA_DB_PATH,
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_model_name=EMBEDDING_MODEL,
                legacy_db_path=LEGACY_CHROMA_DB_PATH,
            )

            self.retrieval_service = RetrievalService(
                repository=repository, rewriter=rewriter
            )
            self.context_builder = ContextBuilder(self.retrieval_service)
            self.answer_service = GeminiAnswerService(LLM_MODEL)

            self.initialized = True
            print("[INFO] RAG pipeline initialized successfully")

        except Exception as e:
            print(f"[ERROR] Failed to initialize RAG pipeline: {e}")
            self.initialized = False

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        user_message, chat_history = self.history_builder.build(tracker)

        if not user_message:
            dispatcher.utter_message(text="Xin lỗi, tôi không hiểu câu hỏi của bạn.")
            return []

        # 🔥 Lazy init tại runtime
        self._lazy_init()

        if not self.initialized:
            dispatcher.utter_message(
                text="Hệ thống đang khởi động, vui lòng thử lại sau vài giây."
            )
            return []

        try:
            # Load collection khi cần
            collection = self.retrieval_service.repository.load_collection()
            if collection is None:
                dispatcher.utter_message(
                    text="Không thể tải dữ liệu. Vui lòng kiểm tra ChromaDB."
                )
                return []

            docs = self.retrieval_service.retrieve(user_message, top_k=5)
            print(f"[INFO] Retrieved {len(docs)} docs")

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
                text="Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn."
            )

        return []