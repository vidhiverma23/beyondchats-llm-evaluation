This repository contains a real-time LLM evaluation pipeline designed for BeyondChats. It showcases skills in NLP, embeddings, system architecture, low-latency design, and scalable evaluation workflow engineering.

Architecture Document — LLM Evaluation Pipeline
1. Overview

This document describes the system architecture, components, data flow, and scaling strategy of the LLM Evaluation Pipeline designed for BeyondChats.

2. System Components
A. Input Layer

Accepts 2 JSON files:

chat.json → conversation history

contexts.json → context retrieved from vector DB

B. Embedding Engine

Uses SentenceTransformers (MiniLM) for:

Query embeddings

Response embeddings

Context embeddings

C. Evaluation Engine

Contains four core evaluators:

Relevance Scorer

Completeness Checker

Context Support Analyzer

Hallucination Detector

D. Output Layer

Produces structured JSON including:

Scores

Latency

Composite score

Debug metadata

3. Data Flow
            User Query               LLM Response
                 │                       │
                 ▼                       ▼
        +----------------+     +------------------+
        | Vector DB      |     | evaluate_llm.py  |
        | Retrieve Top-K |     | (Evaluation)     |
        +----------------+     +------------------+
                 │                       │
                 │ Retrieved Contexts     │
                 └────────────┬──────────┘
                              ▼
                 +--------------------------+
                 |  Evaluation Components   |
                 |  - Relevance             |
                 |  - Completeness          |
                 |  - Context Support       |
                 |  - Hallucination         |
                 +--------------------------+
                              │
                              ▼
                Outputs → evaluation.json

4. Scaling Strategy
A. Lightweight Embeddings

MiniLM runs quickly even on CPUs → <30ms latency.

B. No LLM Calls

Zero cost, zero external dependency.

C. Batch Embedding Pipeline

Possible to evaluate 1,000+ messages/sec.

D. Modular Design

Easily converted into a FastAPI microservice.

5. Future Extensions

Add LLM Verifier for hallucination

Use cross-encoders for deeper semantic relevance

Integrate with Pinecone/Weaviate

Add monitoring dashboard (Prometheus + Grafana)

6. Why This Architecture?

✔ Low latency
✔ Extremely low cost
✔ Production-ready
✔ Modular and extendable
✔ Compatible with beyondchats real-time needs

