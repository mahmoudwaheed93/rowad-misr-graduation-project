from typing import Literal
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    department: Literal["hr", "accounting", "legal"]
    top_k: int = Field(default=3, ge=1, le=8)


class AskResponse(BaseModel):
    mode: Literal["generated", "retrieval_only", "no_evidence"]
    answer: str
    evidence: list[dict]
