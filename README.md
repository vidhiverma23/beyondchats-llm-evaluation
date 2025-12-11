-- This repository contains a real-time LLM evaluation pipeline designed for BeyondChats. It showcases skills in NLP, embeddings, system architecture, low-latency design, and scalable evaluation workflow engineering.

LLM Evaluation Pipeline — BeyondChats Internship Assignment
By Vidhi Verma

This repository contains a production-ready LLM Response Evaluation Pipeline designed to evaluate AI responses along the following dimensions:

Response Relevance

Completeness

Hallucination Detection

Context Support

Latency & Cost Estimation

The system evaluates LLM outputs in real time using embeddings-based semantic analysis and lightweight heuristics to maintain low latency and low cost, even at massive scale.

>Features

Relevance scoring: Semantic similarity between user query and model response

Context Support: Checks if the generated response is grounded in retrieved context

Completeness estimation: Ensures all parts of the user query are addressed

Hallucination detection: Flags unsupported claims

Composite scoring: A weighted system for overall reliability

Fast, scalable, low-cost architecture

Embeddings-based pipeline (sentence-transformers)

>System Architecture
High-Level Design 
flowchart 
    A[User Query] --> B[Vector DB Retrieval]
    B --> C[Retrieved Contexts]
    A --> D[LLM Response]

    C --> E[Evaluation Engine]
    D --> E[Evaluation Engine]

    E --> F[Relevance Score]
    E --> G[Completeness Score]
    E --> H[Context Support Score]
    E --> I[Hallucination Analysis]
    E --> J[Latency & Cost]

    J --> K[Final Evaluation JSON]

>Directory Structure
beyondchats-llm-evaluation/
│
├── evaluate_llm.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── examples/
│     ├── chat.json
│     ├── contexts.json
│     ├── chat2.json
│     └── contexts2.json
│
└── docs/
      └── architecture.md

 Installation & Setup
1. Clone the repo
git clone https://github.com/YOUR_USERNAME/beyondchats-llm-evaluation.git
cd beyondchats-llm-evaluation

2. Create Python 3.11 Virtual Environment
python3.11 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

4. Run the evaluator
python evaluate_llm.py --chat examples/chat.json --contexts examples/contexts.json


Output:

Composite: 0.87
Output -> evaluation.json

>Example Evaluation Output
{
  "relevance": {
    "relevance_score": 0.92,
    "context_support_score": 0.88,
    "completeness_score": 1.0
  },
  "hallucination": {
    "hallucination_rate": 0.0,
    "unsupported_entities": []
  },
  "composite_score": 0.95
}

>Scaling Strategy 

This pipeline is optimized for millions of requests/day:

1️ Embeddings-based evaluation

SentenceTransformers MiniLM (~22 ms inference)

Fully local → zero cost per query

2️ No LLM calls during evaluation

Avoids API charges

Avoids high latency

3️ Vector comparisons (cosine similarity)

O(1) per evaluation

Extremely fast

4️ Tiered evaluation

Lightweight checks first

Heavy checks only on flagged responses

5️ GPU batching 

Evaluate 1,000+ responses per second

>Future Enhancements 

Add Cross-Encoder for deeper semantic relevance

Add LLM-based claim extraction

Add web search-backed fact validation

Convert into a microservice using FastAPI

Add queue-based async pipeline (Redis + Celery)

Add Elasticsearch / Pinecone integration