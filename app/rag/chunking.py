import json
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROCESSED_JSON_PATH = Path("data/processed/property_documents.json")


def load_property_documents(
    json_path: Path = PROCESSED_JSON_PATH,
    chunk_long_descriptions: bool = False,
    max_chunk_size: int = 1500,
    chunk_overlap: int = 150,
) -> List[Document]:
    if not json_path.exists():
        raise FileNotFoundError(f"Processed JSON not found at {json_path.resolve()}. Run ingestion.py first.")

    with open(json_path, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)

    docs: List[Document] = [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in raw_docs
    ]

    if not chunk_long_descriptions:
        return docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunked_docs: List[Document] = []
    for doc in docs:
        if len(doc.page_content) <= max_chunk_size:
            chunked_docs.append(doc)
        else:
            header = (
                f"Title: {doc.metadata.get('title')} | "
                f"Location: {doc.metadata.get('location')} | "
                f"Price: {doc.metadata.get('price')} EGP\n"
            )
            sub_docs = splitter.create_documents([doc.page_content], metadatas=[doc.metadata])
            for sub in sub_docs:
                if not sub.page_content.startswith("Title:"):
                    sub.page_content = header + sub.page_content
                chunked_docs.append(sub)

    return chunked_docs