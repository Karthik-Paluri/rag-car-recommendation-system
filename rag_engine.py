# -------------------------------------------------------
# RAG ENGINE - COMPLETE AND CLEAN VERSION
# -------------------------------------------------------

import json
import numpy as np
import faiss
import re
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder

# -------------------------------------------------------
# ---------------- LOAD DATA ----------------------------
# -------------------------------------------------------

def load_dataset(path="cars_RAG.jsonl"):
    cars = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cars.append(json.loads(line))
    return cars

cars = load_dataset()
texts = [c["text"] for c in cars]

# -------------------------------------------------------
# ---------------- EMBEDDINGS ---------------------------
# -------------------------------------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embs = embed_model.encode(texts, normalize_embeddings=True).astype("float32")

# -------------------------------------------------------
# ---------------- FAISS -------------------------------
# -------------------------------------------------------

dim = embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embs)

# -------------------------------------------------------
# ---------------- CROSS ENCODER ------------------------
# -------------------------------------------------------

cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -------------------------------------------------------
# ---------------- RETRIEVE FUNCTION --------------------
# -------------------------------------------------------

# Retrieve top_k candidate cars using FAISS similarity search.
def retrieve(query, top_k=100):
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        item = cars[idx].copy()
        item["score"] = float(score)
        results.append(item)

    return results

# -------------------------------------------------------
# ---------------- RERANK FUNCTION ----------------------
# -------------------------------------------------------

def rerank(query, candidates, final_k=5):
    pairs = [(query, c["text"]) for c in candidates]
    scores = cross_model.predict(pairs)

    for c, s in zip(candidates, scores):
        c["_cross_score"] = float(s)

    sorted_items = sorted(candidates, key=lambda x: x["_cross_score"], reverse=True)
    return sorted_items[:final_k]

# -------------------------------------------------------
# ---------- PRICE FILTER EXTRACTION --------------------
# -------------------------------------------------------

def extract_price_constraints(query):
    q = query.lower().replace(",", "")

    # BETWEEN range (15k to 25k, 15-25k)
    match = re.search(r'(\d+)\s*(k|000)?\s*[-toand]+\s*(\d+)\s*(k|000)?', q)
    if match:
        low = int(match.group(1)) * (1000 if match.group(2) else 1)
        high = int(match.group(3)) * (1000 if match.group(4) else 1)
        return {"type": "between", "min": low, "max": high}

    # LESS THAN / UNDER
    match = re.search(r'(less than|under|below|<)\s*(\d+)\s*(k|000)?', q)
    if match:
        val = int(match.group(2)) * (1000 if match.group(3) else 1)
        return {"type": "max", "value": val}

    # MORE THAN / ABOVE
    match = re.search(r'(more than|over|above|greater than|>|plus)\s*(\d+)\s*(k|000)?', q)
    if match:
        val = int(match.group(2)) * (1000 if match.group(3) else 1)
        return {"type": "min", "value": val}

    # AROUND / NEAR
    match = re.search(r'(around|near|approx)\s*(\d+)\s*(k|000)?', q)
    if match:
        center = int(match.group(2)) * (1000 if match.group(3) else 1)
        return {"type": "around", "center": center, "range": 3000}

    # Brief formats like 40k, <20k, >30k
    match = re.search(r'([<>])\s*(\d+)\s*(k|000)?', q)
    if match:
        val = int(match.group(2)) * (1000 if match.group(3) else 1)
        if match.group(1) == ">":
            return {"type": "min", "value": val}
        else:
            return {"type": "max", "value": val}

    # Single number like "20k budget"
    match = re.search(r'(\d+)\s*(k|000)?', q)
    if match:
        center = int(match.group(1)) * (1000 if match.group(2) else 1)
        return {"type": "around", "center": center, "range": 3000}

    return None


def apply_price_filter(candidates, rule):
    if not rule:
        return candidates

    t = rule["type"]

    if t == "min":  
        return [c for c in candidates if c["meta"]["price_eur"] >= rule["value"]]

    if t == "max":  
        return [c for c in candidates if c["meta"]["price_eur"] <= rule["value"]]

    if t == "between":
        return [
            c for c in candidates
            if rule["min"] <= c["meta"]["price_eur"] <= rule["max"]
        ]

    if t == "around":
        low = rule["center"] - rule["range"]
        high = rule["center"] + rule["range"]
        return [
            c for c in candidates
            if low <= c["meta"]["price_eur"] <= high
        ]

    return candidates

# -------------------------------------------------------
# ---------------- OLLAMA CHAT --------------------------
# -------------------------------------------------------

# Requires Ollama running locally with qwen2.5:7b model installed
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

def ollama_chat(prompt, max_tokens=200, temperature=0.1):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True)
    resp.raise_for_status()

    full_text = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        if "message" in data:
            full_text += data["message"]["content"]

    return full_text


# -------------------------------------------------------
# ---------------- MAIN ASK FUNCTION --------------------
# -------------------------------------------------------

def generate_answer(query, reranked, top_k_for_llm=3):
    items = reranked[:top_k_for_llm]

    # Build compact context for LLM reasoning
    ctx_lines = []
    for i, r in enumerate(items, start=1):
        m = r["meta"]
        ctx_lines.append(
            f"{i}. {m['model']} | {m['price_eur']} EUR | "
            f"{m['mileage_km']} km | {m['fuel_type']} | {m['car_link']}"
        )
    context = "\n".join(ctx_lines)

    prompt = f"""
You are a car recommendation assistant. 

User query:
{query}

Top candidate cars:
{context}

Give a highly accurate recommendation of 3 cars based on the user's query.
Format:
- 2–3 sentences of reasoning explaining why these cars fit the user's needs.
"""

    reasoning = ollama_chat(prompt, max_tokens=200)

    # Build structured results for Streamlit cards
    results = []
    for r in items:
        m = r["meta"]
        results.append({
            "model": m["model"],
            "price": f"{m['price_eur']} EUR",
            "mileage": f"{m['mileage_km']} km",
            "fuel": m["fuel_type"],
            "link": m["car_link"],
        })

    return {
        "reasoning": reasoning.strip(),
        "cars": results
    }



def ask(query):
    retrieved = retrieve(query, top_k=100)

    price_rule = extract_price_constraints(query)
    if price_rule:
        retrieved = apply_price_filter(retrieved, price_rule)

    if not retrieved:
        return {
            "reasoning": f"No cars match your filters: {query}",
            "cars": []
        }

    reranked = rerank(query, retrieved, final_k=5)
    answer = generate_answer(query, reranked, top_k_for_llm=3)
    return answer

# -------------------------------------------------------   
# ---------------- TEST ASK FUNCTION --------------------
# -------------------------------------------------------
if __name__ == "__main__":
    query = "Looking for a accident free petrol car around 30000 euros"
    print(ask(query))