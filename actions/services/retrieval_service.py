import re
import unicodedata
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from actions.query_writer import QueryRewriter


class ChromaRepository:
    def __init__(
        self,
        db_path,
        collection_name: str,
        embedding_model_name: str,
        legacy_db_path=None,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.legacy_db_path = legacy_db_path
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self._collection: Optional[Any] = None
        self._all_docs_cache: Optional[List[Dict[str, Any]]] = None

    def load_collection(self) -> Optional[Any]:
        if self._collection is not None:
            return self._collection

        candidate_paths = [self.db_path]
        if self.legacy_db_path and self.legacy_db_path not in candidate_paths:
            candidate_paths.append(self.legacy_db_path)

        settings = Settings(anonymized_telemetry=False)
        for db_path in candidate_paths:
            if not db_path.exists():
                continue

            try:
                client = chromadb.PersistentClient(path=str(db_path), settings=settings)
                self._collection = client.get_collection(self.collection_name)
                print(f"[INFO] Loaded Chroma collection from: {db_path}")
                return self._collection
            except Exception as exc:
                print(f"[WARNING] ChromaDB initialization failed at {db_path}: {exc}")

        return None

    def embed_query(self, query: str):
        return self.embedding_model.encode(
            "Represent this sentence for searching relevant passages: " + query,
            normalize_embeddings=True,
        )

    def query_documents(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        collection = self.load_collection()
        if collection is None:
            return []

        query_embedding = self.embed_query(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k, 6),
        )

        docs: List[Dict[str, Any]] = []
        if not results.get("documents"):
            return docs

        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            docs.append(
                {
                    "text": doc,
                    "page": meta.get("page"),
                    "chapter": meta.get("chapter"),
                    "type": meta.get("type"),
                    "_distance": distance,
                }
            )

        return docs

    def get_all_docs(self) -> List[Dict[str, Any]]:
        if self._all_docs_cache is not None:
            return self._all_docs_cache

        collection = self.load_collection()
        if collection is None:
            return []

        results = collection.get(include=["documents", "metadatas"])
        docs: List[Dict[str, Any]] = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            docs.append(
                {
                    "text": doc,
                    "page": meta.get("page"),
                    "chapter": meta.get("chapter"),
                    "type": meta.get("type"),
                }
            )

        self._all_docs_cache = docs
        return docs


class RetrievalService:
    def __init__(self, repository: ChromaRepository, rewriter: QueryRewriter):
        self.repository = repository
        self.rewriter = rewriter

    def retrieve(self, raw_query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        rewritten_query = self.rewriter.rewrite(raw_query)
        candidate_queries = list(dict.fromkeys([raw_query.strip(), rewritten_query.strip()]))
        candidates: Dict[str, Dict[str, Any]] = {}

        for query in candidate_queries:
            for doc in self.repository.query_documents(query, top_k=top_k):
                key = f"{doc.get('page')}::{doc['text'][:120]}"
                best = candidates.get(key)
                if best is None or doc["_distance"] < best["_distance"]:
                    candidates[key] = doc

        for doc in self._keyword_search(raw_query, top_k=top_k):
            key = f"{doc.get('page')}::{doc['text'][:120]}"
            if key not in candidates:
                candidates[key] = {**doc, "_distance": 1.0}

        query_keywords = self._extract_keywords(raw_query)
        canonical_terms = self._extract_canonical_terms(rewritten_query)
        topic_phrase = self._extract_topic_phrase(raw_query)
        is_definition_query = self._is_definition_query(raw_query)

        rescored = []
        for doc in candidates.values():
            lexical_score = self._score_document(
                doc,
                query_keywords=query_keywords,
                canonical_terms=canonical_terms,
                topic_phrase=topic_phrase,
                is_definition_query=is_definition_query,
            )
            embedding_score = max(0.0, 1.0 - float(doc["_distance"]))
            rescored.append((lexical_score + embedding_score, doc))

        rescored.sort(key=lambda item: item[0], reverse=True)

        final_docs: List[Dict[str, Any]] = []
        for _, doc in rescored:
            cleaned_doc = {
                "text": doc["text"],
                "page": doc.get("page"),
                "chapter": doc.get("chapter"),
                "type": doc.get("type"),
            }
            if cleaned_doc not in final_docs:
                final_docs.append(cleaned_doc)
            if len(final_docs) >= top_k:
                break

        return final_docs

    def _strip_accents(self, text: str) -> str:
        text = text.replace("đ", "d").replace("Đ", "D")
        return "".join(
            ch
            for ch in unicodedata.normalize("NFD", text)
            if unicodedata.category(ch) != "Mn"
        )

    def _normalize_for_match(self, text: str) -> str:
        text = self._strip_accents(text.lower())
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _compact_text(self, text: str) -> str:
        return self._normalize_for_match(text).replace(" ", "")

    def _extract_keywords(self, query: str) -> List[str]:
        stop_words = {
            "la",
            "gi",
            "khai",
            "niem",
            "dinh",
            "nghia",
            "trong",
            "cua",
            "ve",
            "cho",
            "toi",
            "duoc",
            "hay",
            "what",
            "is",
            "meaning",
            "of",
        }
        tokens = self._normalize_for_match(query).split()
        return [token for token in tokens if len(token) > 1 and token not in stop_words]

    def _extract_canonical_terms(self, rewritten_query: str) -> List[str]:
        rewritten_normalized = self._normalize_for_match(rewritten_query)
        canonical_terms = []

        for term in set(self.rewriter.synonyms.values()):
            normalized_term = self._normalize_for_match(term)
            if normalized_term in rewritten_normalized:
                canonical_terms.append(normalized_term)

        return canonical_terms

    def _extract_topic_phrase(self, query: str) -> str:
        normalized_query = self._normalize_for_match(query)
        normalized_query = re.sub(
            r"\b(la gi|khai niem|dinh nghia|what is|meaning of)\b",
            " ",
            normalized_query,
        )
        return re.sub(r"\s+", " ", normalized_query).strip()

    def _is_definition_query(self, query: str) -> bool:
        normalized_query = self._normalize_for_match(query)
        markers = ("la gi", "khai niem", "dinh nghia", "what is", "meaning of")
        return any(marker in normalized_query for marker in markers)

    def is_noisy_doc(self, text: str) -> bool:
        dot_runs = text.count("...") + text.count("___")
        words = re.findall(r"\w+", text, re.UNICODE)
        alpha_words = sum(1 for word in words if any(ch.isalpha() for ch in word))
        return dot_runs >= 3 or alpha_words < 12

    def _score_document(
        self,
        doc: Dict[str, Any],
        query_keywords: List[str],
        canonical_terms: List[str],
        topic_phrase: str,
        is_definition_query: bool,
    ) -> float:
        normalized_text = self._normalize_for_match(doc["text"])
        compact_text = self._compact_text(doc["text"])
        score = 0.0

        keyword_hits = sum(1 for keyword in query_keywords if keyword in normalized_text)
        score += keyword_hits * 1.2

        for term in canonical_terms:
            if term in normalized_text:
                score += 5.0
            elif self._compact_text(term) in compact_text:
                score += 5.0

        if topic_phrase:
            if topic_phrase in normalized_text:
                score += 6.0
            elif self._compact_text(topic_phrase) in compact_text:
                score += 6.0

            if (
                doc.get("type") == "definition"
                and self._compact_text(topic_phrase) in compact_text
                and "la gi" in normalized_text
            ):
                score += 4.0

        if is_definition_query and doc.get("type") == "definition":
            score += 2.5

        if self.is_noisy_doc(doc["text"]):
            score -= 3.0

        return score

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_keywords = self._extract_keywords(query)
        canonical_terms = self._extract_canonical_terms(self.rewriter.rewrite(query))
        topic_phrase = self._extract_topic_phrase(query)
        is_definition_query = self._is_definition_query(query)

        scored_docs = []
        for doc in self.repository.get_all_docs():
            score = self._score_document(
                doc,
                query_keywords=query_keywords,
                canonical_terms=canonical_terms,
                topic_phrase=topic_phrase,
                is_definition_query=is_definition_query,
            )
            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]


class ContextBuilder:
    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval_service = retrieval_service

    def build(self, docs: List[Dict[str, Any]], limit: int = 3) -> str:
        context_parts = []
        for doc in docs[:limit]:
            if self.retrieval_service.is_noisy_doc(doc["text"]):
                continue
            page_info = f"(Trang {doc['page']})" if doc.get("page") else ""
            context_parts.append(f"{page_info} {doc['text']}")

        return "\n\n".join(context_parts)
