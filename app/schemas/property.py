from typing import List, Optional, Any
from pydantic import BaseModel, Field

class PropertyFilter(BaseModel):
    query: str = Field(..., description="Semantic search query in Arabic or English")
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    bedrooms: Optional[int] = None
    min_area: Optional[float] = None
    location_keyword: Optional[str] = None
    property_type: Optional[str] = None
    k: int = Field(6, ge=1, le=20)

class PropertyMetadata(BaseModel):
    property_id: Any = ""
    title: Optional[str] = None
    property_type: Optional[str] = None
    location: Optional[str] = None
    price: Optional[float] = None
    area_sqm: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    furnished: Optional[str] = None
    payment_method: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = "Property Finder"

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2)
    session_id: str = Field(default="default_session")

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    properties: List[PropertyMetadata] = []