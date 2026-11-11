import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

RAW_CSV_PATH = Path("D:\\users\\propertyfinder.csv")
PROCESSED_JSON_PATH = Path("data/processed/property_documents.json")

def _clean_number(val: Any, as_int: bool = False) -> Optional[float]:
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return int(val) if as_int else float(val)
    cleaned = re.sub(r"[^\d.]", "", str(val))
    if not cleaned:
        return None
    try:
        num = float(cleaned)
        return int(num) if as_int else num
    except ValueError:
        return None

def detect_source(row: pd.Series) -> str:
    """Detects where the row came from based on specific text or URL patterns."""
    url = str(row.get("url", "")).lower()
    source_val = str(row.get("source", "")).lower()
    
    if "propertyfinder" in url or "propertyfinder" in source_val:
        return "Property Finder"
    elif "nawy" in url or "nawy" in source_val:
        return "Nawy"
    elif "aqarmap" in url or "aqarmap" in source_val:
        return "Aqarmap"
    return "Unknown Agent"

def validate_and_normalize(row: pd.Series, idx: int) -> Dict[str, Any]:
    """Validates and processes data, aligning to the unified data spec."""
    source = detect_source(row)
    
    # 1. Base property identification
    prop_id = str(row.get("id", row.get("property_id", f"gen_{source[:2]}_{idx}"))).strip()
    title = str(row.get("title", row.get("name", "Listing"))).strip()
    location = str(row.get("location", row.get("address", "Egypt"))).strip()
    prop_type = str(row.get("property_type", row.get("type", "Property"))).strip()

    # 2. Extract and validate prices
    price = _clean_number(row.get("price", row.get("price_egp", row.get("amount"))))
    area = _clean_number(row.get("area", row.get("area_sqm", row.get("size"))))
    bedrooms = _clean_number(row.get("bedrooms", row.get("beds")), as_int=True)
    bathrooms = _clean_number(row.get("bathrooms", row.get("baths")), as_int=True)
    
    # 3. Clean amenities and extra payload attributes
    amenities = str(row.get("amenities", row.get("features", ""))).strip()
    url = str(row.get("url", row.get("link", ""))).strip()
    description = str(row.get("description", row.get("details", ""))).strip()
    payment = str(row.get("payment_method", row.get("payment", ""))).strip()

    return {
        "property_id": prop_id,
        "title": title,
        "property_type": prop_type,
        "location": location,
        "price": price,
        "area_sqm": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "payment_method": payment,
        "amenities": amenities,
        "url": url,
        "description": description,
        "source": source
    }

def format_prompt_block(meta: Dict[str, Any]) -> str:
    """Combines key information with source tracking for accurate RAG prompts."""
    parts = [
        f"Title: {meta['title']}",
        f"Source Partner: {meta['source']}",
        f"Property Type: {meta['property_type']}",
        f"Location: {meta['location']}",
        f"Price: {meta['price']:,.0f} EGP" if meta.get("price") else "Price: On request EGP",
        f"Area: {meta['area_sqm']} sqm" if meta.get("area_sqm") else "",
        f"Bedrooms: {meta['bedrooms']}" if meta.get("bedrooms") is not None else "",
        f"Bathrooms: {meta['bathrooms']}" if meta.get("bathrooms") is not None else "",
        f"Payment Plan: {meta['payment_method']}" if meta.get("payment_method") else "",
        f"Amenities: {meta['amenities']}" if meta.get("amenities") else "",
        f"Details: {meta['description']}" if meta.get("description") else "",
    ]
    return "\n".join([p for p in parts if p])

def ingest_all_sources(csv_path: Path = RAW_CSV_PATH, output_path: Path = PROCESSED_JSON_PATH, limit: int | None = None) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Unified multi-source CSV not found at {csv_path.resolve()}")

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    if limit:
        df = df.head(limit)

    raw_docs = []
    seen_signatures = set()

    print(f"[Ingestion Agent] Reading raw records...")
    for idx, row in df.iterrows():
        norm = validate_and_normalize(row, idx)
        
        # Validation filter: Require basic pricing and location
        if not norm["price"] or norm["price"] < 50000:
            continue
        if not norm["location"] or norm["location"].lower() in ["nan", "unknown", ""]:
            continue
            
        # Deduplication step [cite: 6]
        identity_key = f"{norm['title']}_{norm['price']}_{norm['location']}".lower()
        if identity_key in seen_signatures:
            continue
        seen_signatures.add(identity_key)

        prompt_str = format_prompt_block(norm)
        raw_docs.append({
            "id": norm["property_id"],
            "page_content": prompt_str,
            "metadata": norm
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(raw_docs, f, ensure_ascii=False, indent=2)

    print(f"[Data Processing & Validation] Filtered & validated {len(raw_docs)} unique items.")
    return raw_docs

if __name__ == "__main__":
    ingest_all_sources(limit=1000)