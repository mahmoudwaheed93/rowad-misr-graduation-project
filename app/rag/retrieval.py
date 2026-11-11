from typing import List, Optional
from langchain_core.documents import Document
from qdrant_client.http import models as rest
from app.rag.vector_store import get_vector_store


def build_qdrant_filter(
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    bedrooms: Optional[int] = None,
    min_area: Optional[float] = None,
    location_keyword: Optional[str] = None,
    property_type: Optional[str] = None,
) -> Optional[rest.Filter]:
    must_conditions = []

    if min_price is not None or max_price is not None:
        must_conditions.append(
            rest.FieldCondition(
                key="metadata.price",
                range=rest.Range(gte=min_price, lte=max_price),
            )
        )

    if bedrooms is not None:
        must_conditions.append(
            rest.FieldCondition(
                key="metadata.bedrooms",
                match=rest.MatchValue(value=bedrooms),
            )
        )

    if min_area is not None:
        must_conditions.append(
            rest.FieldCondition(
                key="metadata.area_sqm",
                range=rest.Range(gte=min_area),
            )
        )

    if location_keyword:
        must_conditions.append(
            rest.FieldCondition(
                key="metadata.location",
                match=rest.MatchText(text=location_keyword),
            )
        )

    if property_type:
        must_conditions.append(
            rest.FieldCondition(
                key="metadata.property_type",
                match=rest.MatchText(text=property_type),
            )
        )

    return rest.Filter(must=must_conditions) if must_conditions else None


def retrieve_properties(
    query: str,
    k: int = 5,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    bedrooms: Optional[int] = None,
    min_area: Optional[float] = None,
    location_keyword: Optional[str] = None,
    property_type: Optional[str] = None,
) -> List[Document]:
    vector_store = get_vector_store()
    qdrant_filter = build_qdrant_filter(
        min_price=min_price,
        max_price=max_price,
        bedrooms=bedrooms,
        min_area=min_area,
        location_keyword=location_keyword,
        property_type=property_type,
    )

    return vector_store.similarity_search(
        query=query,
        k=k,
        filter=qdrant_filter,
    )