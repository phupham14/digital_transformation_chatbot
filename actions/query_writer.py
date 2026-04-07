import re
import unicodedata


class QueryRewriter:
    def __init__(self):
        self.synonyms = {
            "ai": "trí tuệ nhân tạo",
            "artificial intelligence": "trí tuệ nhân tạo",
            "big data": "dữ liệu lớn",
            "cloud": "điện toán đám mây",
            "cloud computing": "điện toán đám mây",
            "iot": "internet vạn vật",
            "internet of things": "internet vạn vật",
            "digital transformation": "chuyển đổi số",
            "cybersecurity": "an toàn thông tin",
            "machine learning": "học máy",
            "deep learning": "học sâu",
        }
        self.definition_markers = (
            "là gì",
            "khái niệm",
            "định nghĩa",
            "what is",
            "meaning of",
        )

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

    def normalize(self, query: str) -> str:
        q = query.replace("\\", " ").strip().lower()
        q = re.sub(r"\s+", " ", q)
        return q

    def expand_synonym(self, query: str) -> str:
        expanded_parts = [query]
        normalized_query = self._normalize_for_match(query)

        for source, target in self.synonyms.items():
            source_normalized = self._normalize_for_match(source)
            pattern = rf"(?<!\w){re.escape(source_normalized)}(?!\w)"
            if source_normalized and re.search(pattern, normalized_query):
                expanded_parts.append(target)

        return " ".join(dict.fromkeys(expanded_parts))

    def add_context(self, query: str) -> str:
        normalized_query = self._normalize_for_match(query)
        has_definition_marker = any(
            self._normalize_for_match(marker) in normalized_query
            for marker in self.definition_markers
        )

        if not has_definition_marker:
            query = f"{query} là gì"

        return query

    def rewrite(self, query: str) -> str:
        q = self.normalize(query)
        q = self.expand_synonym(q)
        q = self.add_context(q)
        return q
