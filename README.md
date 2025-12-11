
# Full LLM Evaluation Pipeline

This repo contains:
- evaluate_llm.py (full real-time evaluation script)
- requirements.txt
- Example JSON files

Supports:
- Relevance scoring
- Completeness
- Hallucination detection
- Context support
- Composite scoring

Run:
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python evaluate_llm.py --chat chat.json --contexts contexts.json
```
