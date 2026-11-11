import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.rag.retrieval import retrieve_properties

load_dotenv()

SYSTEM_PROMPT = """You are Aqar-AI (عقار ذكي), an expert Egyptian Real Estate AI Advisor.
Answer the user's request using the retrieved property listings below.
- State Title, Location, Price (in EGP), Area (sqm), Bedrooms/Bathrooms, and Source.
- If the user asks to calculate installments or ROI, compute the 10% down payment and monthly/quarterly installments over 8 years clearly.
- Respond in the same language as the user's question (Arabic or English).

Retrieved Property Listings:
{context}
"""

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.LLM_MODEL or os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct"),
        api_key=settings.LLM_API_KEY or os.getenv("LLM_API_KEY"),
        base_url=settings.LLM_BASE_URL or os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0.2,
    )

def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "No matching properties found."
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"--- Property #{i} (ID: {doc.metadata.get('property_id')}) ---\n{doc.page_content}")
    return "\n\n".join(formatted)

def answer_property_query(
    query: str,
    k: int = 6,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    bedrooms: Optional[int] = None,
    location_keyword: Optional[str] = None,
) -> dict:
    docs = retrieve_properties(
        query=query,
        k=k,
        min_price=min_price,
        max_price=max_price,
        bedrooms=bedrooms,
        location_keyword=location_keyword,
    )
    # Fallback to pure semantic search if strict metadata filters matched 0 rows
    if not docs:
        docs = retrieve_properties(query=query, k=k)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = prompt | get_llm() | StrOutputParser()
    answer = chain.invoke({
        "context": format_docs(docs),
        "question": query,
    })

    return {
        "query": query,
        "answer": answer,
        "retrieved_properties": [d.metadata for d in docs],
    }