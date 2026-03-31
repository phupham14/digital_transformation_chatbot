from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_DIR = Path("chroma_db")
LEGACY_DB_PATH = CHROMA_DB_DIR / "chroma.sqlite3"
COLLECTION_NAME = "digital_transformation_handbook"
EMBEDDING_MODEL = "BAAI/bge-m3"
BATCH_SIZE = 32


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
                "id": embedding_id,
                "document": document,
                "metadata": metadata,
            }
        )

    return documents


def rebuild_collection(rows: List[Dict]) -> None:
    if CHROMA_DB_DIR.exists():
        shutil.rmtree(CHROMA_DB_DIR)

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL)

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        documents = [item["document"] for item in batch]
        ids = [item["id"] for item in batch]
        metadatas = [item["metadata"] for item in batch]
        embeddings = model.encode(documents, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        print(f"Indexed {start + len(batch)}/{len(rows)} documents")

    print(f"Done. Collection '{COLLECTION_NAME}' now has {collection.count()} documents.")


def main() -> None:
    if not LEGACY_DB_PATH.exists():
        raise FileNotFoundError(f"Legacy database not found: {LEGACY_DB_PATH}")

    rows = load_legacy_rows(LEGACY_DB_PATH)
    if not rows:
        raise RuntimeError("No legacy documents found to rebuild.")

    print(f"Recovered {len(rows)} non-empty documents from legacy SQLite.")
    rebuild_collection(rows)


if __name__ == "__main__":
    main()
