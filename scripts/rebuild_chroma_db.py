from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb
import posthog
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

posthog.disabled = True
posthog.capture = lambda *args, **kwargs: None

load_dotenv()

# ==================== CONFIG ====================
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR = Path(
    os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db_rebuilt"))
).resolve()
LEGACY_DB_PATH = Path(
    os.getenv(
        "CHROMA_LEGACY_DB_PATH",
        str(BASE_DIR / "chroma_db" / "chroma.sqlite3"),
    )
).resolve()
COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME", "digital_transformation_handbook"
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
BATCH_SIZE = int(os.getenv("CHROMA_REBUILD_BATCH_SIZE", "32"))


# ==================== LOAD OLD DATA ====================
def load_legacy_rows(db_path: Path) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    rows = cur.execute(
        """
        SELECT
            e.embedding_id,
            MAX(CASE WHEN em.key = 'chroma:document' THEN em.string_value END) AS document,
            MAX(CASE WHEN em.key = 'page' THEN em.int_value END) AS page,
            MAX(CASE WHEN em.key = 'chapter' THEN em.string_value END) AS chapter,
            MAX(CASE WHEN em.key = 'type' THEN em.string_value END) AS doc_type
        FROM embeddings e
        LEFT JOIN embedding_metadata em ON em.id = e.id
        GROUP BY e.id, e.embedding_id
        ORDER BY e.id
        """
    ).fetchall()

    conn.close()

    documents = []
    for embedding_id, document, page, chapter, doc_type in rows:
        if not document:
            continue

        metadata = {"type": doc_type or "general"}
        if page is not None:
            metadata["page"] = int(page)
        if chapter:
            metadata["chapter"] = chapter

        documents.append(
            {
                "id": str(embedding_id),
                "document": document,
                "metadata": metadata,
            }
        )

    return documents


# ==================== EMBEDDING ====================
def embed_documents(model, docs: List[str]):
    texts = [
        "Represent this sentence for retrieval: " + doc
        for doc in docs
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


# ==================== REBUILD ====================
def rebuild_collection(rows: List[Dict]) -> None:
    # Only clear the chosen output directory so we do not accidentally wipe another DB.
    if CHROMA_DB_DIR.exists():
        shutil.rmtree(CHROMA_DB_DIR)

    print(f"Creating new ChromaDB at: {CHROMA_DB_DIR}")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(COLLECTION_NAME)

    model = SentenceTransformer(EMBEDDING_MODEL)

    total = len(rows)

    for start in range(0, total, BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]

        documents = [item["document"] for item in batch]
        ids = [item["id"] for item in batch]
        metadatas = [item["metadata"] for item in batch]

        embeddings = embed_documents(model, documents)

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        print(f"Indexed {start + len(batch)}/{total}")

    print(f"\nDONE! Total documents: {collection.count()}")


# ==================== MAIN ====================
def main():
    if not LEGACY_DB_PATH.exists():
        raise FileNotFoundError(f"Legacy DB not found: {LEGACY_DB_PATH}")

    print(f"Loading legacy data from: {LEGACY_DB_PATH}")
    rows = load_legacy_rows(LEGACY_DB_PATH)

    if not rows:
        raise RuntimeError("No documents found!")

    print(f"Loaded {len(rows)} documents")

    rebuild_collection(rows)


if __name__ == "__main__":
    main()
