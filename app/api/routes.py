from fastapi import APIRouter, HTTPException
from app.schemas.property import PropertyFilter, ChatRequest, ChatResponse, PropertyMetadata
from app.rag.retrieval import retrieve_properties
from app.rag.generation import answer_property_query
from app.agents.real_estate_agent import run_agent_chat

router = APIRouter(prefix="/api/v1", tags=["Aqar-AI"])

@router.post("/search", response_model=list[PropertyMetadata])
def search_listings(payload: PropertyFilter):
    docs = retrieve_properties(
        query=payload.query,
        k=payload.k,
        min_price=payload.min_price,
        max_price=payload.max_price,
        bedrooms=payload.bedrooms,
        min_area=payload.min_area,
        location_keyword=payload.location_keyword,
        property_type=payload.property_type,
    )
    return [PropertyMetadata(**d.metadata) for d in docs]

@router.post("/rag", response_model=ChatResponse)
def rag_query(payload: PropertyFilter):
    result = answer_property_query(
        query=payload.query,
        k=payload.k,
        min_price=payload.min_price,
        max_price=payload.max_price,
        bedrooms=payload.bedrooms,
        location_keyword=payload.location_keyword,
    )
    return ChatResponse(
        session_id="stateless_rag",
        answer=result["answer"],
        properties=[PropertyMetadata(**m) for m in result["retrieved_properties"]],
    )

@router.post("/chat", response_model=ChatResponse)
def agent_chat(payload: ChatRequest):
    try:
        answer = run_agent_chat(message=payload.message, session_id=payload.session_id)
        return ChatResponse(session_id=payload.session_id, answer=answer, properties=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
