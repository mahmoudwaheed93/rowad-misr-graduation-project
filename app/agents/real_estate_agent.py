import re
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from app.rag.generation import get_llm, answer_property_query
from app.tools.property_tools import (
    search_egypt_properties,
    calculate_mortgage_installment,
    submit_property_purchase_order,
)
import app.tools.property_tools as prop_tools
from app.memory.session_memory import memory_saver

AGENT_SYSTEM_PROMPT = """You are Aqar-AI (عقار ذكي), an autonomous Egyptian Real Estate Buying & Search Agent connected to Aqarmap Egypt, Nawy, and Property Finder Egypt.
- When users search for properties, call `search_egypt_properties` and include each property's Source Portal and clickable URL.
- When users ask about installments, call `calculate_mortgage_installment`.
- When users ask to BUY, BOOK, or RESERVE a property (e.g., 'Buy property...', 'Reserve this unit', 'احجز العقار'), immediately call `submit_property_purchase_order` to create the reservation order and give them the Order ID, Down Payment breakdown, and direct portal checkout link.
- Always reply in the user's language (Arabic or English)."""

tools = [search_egypt_properties, calculate_mortgage_installment, submit_property_purchase_order]


def _extract_quick_filters(message: str):
    beds = None
    max_p = None
    bed_match = re.search(r'(\d+)\s*(?:bedroom|bedrooms|bed|beds|غرف|غرفة)', message, re.IGNORECASE)
    if bed_match:
        beds = int(bed_match.group(1))
    if "٣ غرف" in message:
        beds = 3
    price_match = re.search(r'(?:under|below|max|تحت|أقل من)\s*(\d+(?:\.\d+)?)\s*(?:m|million|مليون)?', message, re.IGNORECASE)
    if price_match:
        val = float(price_match.group(1))
        max_p = val * 1_000_000 if val < 100 else val
    return beds, max_p


def run_agent_chat(message: str, session_id: str = "default_session") -> str:
    # Deterministic Buy/Reserve handler if user clicks "Buy / Reserve Now" button or asks to buy
    if any(k in message.lower() for k in ["buy property", "reserve unit", "book property", "شراء العقار", "احجز العقار"]):
        id_m = re.search(r"ID:\s*([^\s|]+)", message)
        price_m = re.search(r"priced at\s*(\d+(?:\.\d+)?)", message, re.IGNORECASE)
        src_m = re.search(r"from\s*(Nawy|Aqarmap|Property Finder)", message, re.IGNORECASE)
        title_m = re.search(r"titled:\s*(.+?)\s*\(ID:", message, re.IGNORECASE)
        return submit_property_purchase_order.invoke({
            "property_id": id_m.group(1) if id_m else "PROP-101",
            "property_title": title_m.group(1) if title_m else message[:60],
            "price_egp": float(price_m.group(1)) if price_m else 6500000.0,
            "source_portal": src_m.group(1) if src_m else "Nawy",
        })

    try:
        agent = create_react_agent(
            model=get_llm(),
            tools=tools,
            prompt=SystemMessage(content=AGENT_SYSTEM_PROMPT),
            checkpointer=memory_saver,
        )
        config = {"configurable": {"thread_id": session_id}}
        response = agent.invoke({"messages": [HumanMessage(content=message)]}, config=config)
        content = response["messages"][-1].content
        if content and str(content).strip():
            if not prop_tools.LAST_RETRIEVED_META:
                search_egypt_properties.invoke({"query": message})
            return str(content)
    except Exception as e:
        print(f"[Agent Fallback] {e}")

    beds, max_p = _extract_quick_filters(message)
    search_egypt_properties.invoke({"query": message, "bedrooms": beds, "max_price": max_p})
    rag_res = answer_property_query(query=message, k=6, bedrooms=beds, max_price=max_p)
    return rag_res["answer"]
