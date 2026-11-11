import os
import atexit
from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_qdrant import QdrantVectorStore
from app.rag.embeddings import get_embedding_model, EMBEDDING_DIMENSION

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "egypt_real_estate")
QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_storage")


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        client = QdrantClient(url=QDRANT_URL)
    else:
        client = QdrantClient(path=QDRANT_LOCAL_PATH)

    # Close cleanly BEFORE Python tears down sys.modules on Windows
    atexit.register(client.close)
    return client


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    vector_size: int = EMBEDDING_DIMENSION,
) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
            ),
        )
        print(f"[Qdrant] Created collection '{collection_name}' (dim={vector_size})")


def get_vector_store(collection_name: str = COLLECTION_NAME) -> QdrantVectorStore:
    client = get_qdrant_client()
    ensure_collection_exists(client, collection_name=collection_name)
    embeddings = get_embedding_model()
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )