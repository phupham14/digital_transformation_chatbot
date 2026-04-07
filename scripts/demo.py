import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_PATH = Path(
    os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
).resolve()
COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME", "digital_transformation_handbook"
)

client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = client.get_collection(COLLECTION_NAME)

results = collection.query(
    query_texts=["chuyen doi so la gi"],
    n_results=3,
)

print(f"DB path: {CHROMA_DB_PATH}")
print(results)
