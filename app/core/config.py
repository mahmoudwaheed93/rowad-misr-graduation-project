from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Aqar-AI"
    VERSION: str = "0.1.0"
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "meta-llama/llama-3.3-70b-instruct"
    QDRANT_URL: str | None = None
    QDRANT_LOCAL_PATH: str = "./qdrant_storage"
    QDRANT_COLLECTION: str = "egypt_real_estate"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
