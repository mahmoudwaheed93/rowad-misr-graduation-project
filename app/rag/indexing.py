from uuid import uuid5, NAMESPACE_DNS
from tqdm import tqdm
from app.rag.ingestion import ingest_csv_to_json, PROCESSED_JSON_PATH
from app.rag.chunking import load_property_documents
from app.rag.vector_store import get_vector_store


def index_properties(
    rebuild_json: bool = False,
    batch_size: int = 256,
    limit: int | None = None,
) -> None:
    if rebuild_json or not PROCESSED_JSON_PATH.exists():
        ingest_csv_to_json(limit=limit)

    documents = load_property_documents(chunk_long_descriptions=False)
    if limit:
        documents = documents[:limit]

    vector_store = get_vector_store()

    print(f"[Indexing] Indexing {len(documents)} properties into Qdrant (batch_size={batch_size})...")
    for i in tqdm(range(0, len(documents), batch_size), desc="Upserting batches"):
        batch = documents[i : i + batch_size]
        ids = [
            str(uuid5(NAMESPACE_DNS, f"prop_{doc.metadata.get('property_id')}_{i+idx}"))
            for idx, doc in enumerate(batch)
        ]
        vector_store.add_documents(documents=batch, ids=ids)

    print("[Indexing] Completed successfully.")


if __name__ == "__main__":
    # Set limit=1000 for a quick first test, or limit=None for all 39,700+ listings
    index_properties(rebuild_json=True, limit=1000)