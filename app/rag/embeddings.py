import os
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

# Point directly to your local HF cache if not already set globally
os.environ.setdefault("HF_HOME", r"D:\LLMCache\hub")

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


@lru_cache(maxsize=1)
def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": device,
            "local_files_only": True,  # Loads 100% offline from D:\LLMCache\hub
        },
        encode_kwargs={"normalize_embeddings": True},
    )