
#!/usr/bin/env python3
# FULL evaluation pipeline script (same as provided earlier)

import argparse
import json
import os
import time
from typing import List, Dict, Any

import numpy as np

from sentence_transformers import SentenceTransformer
import spacy

nlp = spacy.load("en_core_web_sm")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def text_embeddings(texts):
    return embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def cosine(a, b):
    return float(np.dot(a, b))

def doc_entities(text: str):
    doc = nlp(text)
    return [{"text": e.text, "label": e.label_} for e in doc.ents]

def top_k_similar(embed, ctx_embs, k=3):
    sims = np.dot(ctx_embs, embed)
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def relevance_and_completeness(user_msg, assistant, contexts):
    texts = [user_msg, assistant] + [c["text"] for c in contexts]
    embs = text_embeddings(texts)
    eu, ea = embs[0], embs[1]
    ectx = embs[2:]

    relevance = cosine(eu, ea)

    if len(ectx) == 0:
        ctx_support = 0.0
        top = []
    else:
        k = min(3, len(ectx))
        idx, sims = top_k_similar(ea, ectx, k)
        ctx_support = float(np.mean(sims))
        top = [{"index": int(i), "similarity": float(s)} for i, s in zip(idx, sims)]

    doc = nlp(user_msg)
    keywords = set()
    for ent in doc.ents:
        keywords.add(ent.text.strip())
    for chunk in doc.noun_chunks:
        kw = chunk.text.strip()
        if len(kw.split()) <= 4:
            keywords.add(kw)

    hits = 0
    for kw in keywords:
        if kw.lower() in assistant.lower():
            hits += 1
        else:
            ek = text_embeddings([kw])[0]
            if cosine(ek, ea) > 0.65:
                hits += 1

    completeness = hits / len(keywords) if keywords else 1.0

    return {
        "relevance_score": float(relevance),
        "context_support_score": float(ctx_support),
        "top_contexts": top,
        "keywords_extracted": list(keywords),
        "completeness_score": float(completeness),
    }

def hallucination_check(assistant, contexts):
    ents = doc_entities(assistant)
    if not ents:
        return {"entities_found": [], "unsupported_entities": [], "hallucination_rate": 0.0}

    ctx_texts = [c["text"] for c in contexts]
    if not ctx_texts:
        return {
            "entities_found": ents,
            "unsupported_entities": [e["text"] for e in ents],
            "hallucination_rate": 1.0,
        }

    ctx_embs = text_embeddings(ctx_texts)
    unsupported = []
    support_map = {}

    for ent in ents:
        t = ent["text"]
        literal = any(t.lower() in c.lower() for c in ctx_texts)
        if literal:
            support_map[t] = {"supported": True, "method": "literal"}
            continue

        ee = text_embeddings([t])[0]
        sims = np.dot(ctx_embs, ee)
        best = float(np.max(sims))
        idx = int(np.argmax(sims))

        if best > 0.72:
            support_map[t] = {
                "supported": True,
                "method": "semantic",
                "similarity": best,
                "context_excerpt": ctx_texts[idx][:300],
            }
        else:
            unsupported.append(t)
            support_map[t] = {"supported": False, "similarity": best}

    rate = len(unsupported) / len(ents)
    return {
        "entities_found": ents,
        "unsupported_entities": unsupported,
        "hallucination_rate": float(rate),
        "support_map": support_map,
    }

def evaluate(chat_json, ctx_json):
    msgs = chat_json["messages"]
    last_u = last_a = None
    for m in reversed(msgs):
        if m["role"] == "assistant" and last_a is None:
            last_a = m["content"]
        if m["role"] == "user" and last_u is None:
            last_u = m["content"]
        if last_u and last_a:
            break

    contexts = [
        {"id": c.get("id", str(i)), "text": c.get("text", "")}
        for i, c in enumerate(ctx_json.get("contexts", []))
    ]

    rel = relevance_and_completeness(last_u, last_a, contexts)
    hall = hallucination_check(last_a, contexts)

    composite = max(
        0.0,
        min(
            1.0,
            0.4 * rel["relevance_score"]
            + 0.3 * rel["context_support_score"]
            + 0.2 * rel["completeness_score"]
            + 0.1 * (1 - hall["hallucination_rate"]),
        ),
    )

    return {
        "relevance": rel,
        "hallucination": hall,
        "composite_score": float(composite),
        "metadata": {"num_contexts": len(contexts)},
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chat", required=True)
    p.add_argument("--contexts", required=True)
    p.add_argument("--out", default="evaluation.json")
    a = p.parse_args()

    chat = load_json(a.chat)
    ctx = load_json(a.contexts)

    t0 = time.time()
    result = evaluate(chat, ctx)
    result["runtime_ms"] = (time.time() - t0) * 1000

    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)

    print("Composite:", result["composite_score"])
    print("Output ->", a.out)

if __name__ == "__main__":
    main()
