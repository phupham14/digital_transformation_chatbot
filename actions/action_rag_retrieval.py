from pathlib import Path
from typing import Any, Text, Dict, List, Optional

import chromadb
import google.generativeai as genai
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==================== Configuration ====================
GEN_API_KEY = "AIzaSyBsaCNp1FcDKOvC4qZXJdzF02u-JEQ__0c"
LLM_MODEL = "models/gemini-2.5-flash"
CHROMA_COLLECTION_NAME = "digital_transformation_handbook"
EMBEDDING_MODEL = "BAAI/bge-m3"
CHROMA_DB_PATH = Path(__file__).resolve().parent.parent / "chroma_db"

# ==================== Initialize Models ====================
genai.configure(api_key=GEN_API_KEY.strip())
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def _load_collection() -> Optional[Any]:
    """Khoi tao collection theo duong dan tuyet doi de tranh sai cwd."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        return client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    except Exception as e:
        print(f"[WARNING] ChromaDB initialization failed: {e}")
        return None


class ActionRAGRetrieval(Action):
    """Hanh dong lay du lieu tu ChromaDB va tra loi bang Gemini API."""

    def name(self) -> Text:
        return "action_rag_retrieval"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # Lay cau hoi tu tin nhan cuoi cung
        user_message = tracker.latest_message.get("text", "")

        if not user_message:
            dispatcher.utter_message(text="Xin loi, toi khong hieu cau hoi cua ban.")
            return []

        history_list = []
        user_messages_only = []

        # Duyet qua cac su kien (events) de lay tin nhan cu
        for event in tracker.events:
            if event.get("event") == "user":
                text = event.get("text")
                history_list.append(f"Nguoi dung: {text}")
                user_messages_only.append(text)
            elif event.get("event") == "bot":
                history_list.append(f"Bot: {event.get('text')}")

        # Lay 4 dong chat gan nhat (tuong duong 2 luot hoi dap)
        chat_history_str = (
            "\n".join(history_list[-5:-1])
            if len(history_list) > 1
            else "Chua co lich su tro chuyen."
        )

        # Ghep cau hoi truoc do cua nguoi dung voi cau hien tai de ChromaDB hieu ngu canh
        search_query = user_message
        if len(user_messages_only) >= 2:
            last_user_msg = user_messages_only[-2]
            search_query = f"{last_user_msg}. {user_message}"

        try:
            # Kiem tra ChromaDB co san khong
            collection = _load_collection()
            if collection is None:
                dispatcher.utter_message(
                    text=(
                        "Xin loi, he thong chua san sang. "
                        "ChromaDB dang loi duong dan hoac khong tuong thich schema, "
                        "vui long kiem tra lai du lieu va version chromadb."
                    )
                )
                return []

            # Buoc 1: Lay du lieu tu ChromaDB
            docs = self._search_chromadb(collection, search_query, top_k=10)

            # Buoc 2: Rerank ket qua
            docs = self._rerank_documents(search_query, docs, top_k=5)

            # Buoc 3: Tao context va goi Gemini API
            context = self._build_context(docs)
            response = self._generate_answer(user_message, context, chat_history_str)

            # Gui phan hoi tong hop
            dispatcher.utter_message(text=response)

        except Exception as e:
            print(f"[ERROR action_rag_retrieval]: {e}")
            dispatcher.utter_message(
                text="Xin loi, da xay ra loi khi xu ly cau hoi cua ban."
            )

        return []

    def _search_chromadb(
        self, collection: Any, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Tim kiem cac tai lieu lien quan tu ChromaDB."""
        # Embedding cau hoi
        query_embedding = embedding_model.encode([query]).tolist()

        # Tim kiem
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)

        # Format lai ket qua
        docs = []
        if results["documents"] and len(results["documents"]) > 0:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                docs.append(
                    {
                        "text": doc,
                        "page": meta.get("page"),
                        "chapter": meta.get("chapter"),
                        "type": meta.get("type"),
                    }
                )

        return docs

    def _rerank_documents(
        self, query: str, docs: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank cac tai lieu dua tren do lien quan."""
        if not docs:
            return []

        pairs = [[query, d["text"]] for d in docs]
        scores = reranker.predict(pairs)

        # Sort theo score
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [r[0] for r in ranked[:top_k]]

    def _build_context(self, docs: List[Dict[str, Any]]) -> str:
        """Xay dung context tu cac tai lieu."""
        context_parts = []
        for d in docs:
            page_info = f"(Trang {d['page']})" if d.get("page") else ""
            context_parts.append(f"{page_info} {d['text']}")

        return "\n\n".join(context_parts)

    def _generate_answer(self, query: str, context: str, chat_history: str) -> str:
        """Goi Gemini API de tao cau tra loi."""
        prompt = f"""Ban la tro ly AI chuyen ve chuyen doi so. Hay tra loi dua tren context duoc cung cap.

YEU CAU:
- Chi su dung thong tin trong [Context], KHONG tu y them kien thuc ben ngoai.
- Neu context khong chua thong tin lien quan, tra loi: "Toi khong tim thay thong tin phu hop trong tai lieu."
- Tra loi ngan gon (3-5 cau), ro rang, de hieu.
- Khong lap lai y, khong viet dai dong.
- Neu co nhieu thong tin giong nhau, hay tong hop lai.
- Trich dan so trang o CUOI DOAN (khong lap lai nhieu lan trong tung cau).
- Uu tien cach dien dat tu nhien, tranh cac an du khong pho bien.

NGU CANH:
- Su dung [Lich su tro chuyen] de hieu ro cau hoi hien tai (dac biet voi cac tu nhu "no", "cai do", "van de tren").

Lich su tro chuyen gan nhat:
{chat_history}

Context:
{context}

Cau hoi: {query}

Tra loi:
"""

        try:
            model = genai.GenerativeModel(LLM_MODEL)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"[ERROR] Gemini API call failed: {e}")
            return "Xin loi, da xay ra loi khi xu ly cau hoi cua ban."

print("DB path:", CHROMA_DB_PATH)