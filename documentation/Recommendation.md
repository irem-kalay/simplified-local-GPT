# Production Deployment Recommendation
## Local Wikipedia RAG Assistant

---

## 1. Executive Summary

The current system is a well-structured local prototype. Moving it to production requires addressing five key areas: infrastructure, model serving, scalability, reliability, and security. This document outlines a practical path from the current single-machine setup to a production-grade deployment, with recommendations scaled to the likely use case (small-to-medium internal tool or educational platform).

---

## 2. Current Architecture Limitations

| Limitation | Impact in Production |
|---|---|
| Ollama on localhost | Cannot serve concurrent users; no horizontal scaling |
| ChromaDB in SQLite mode | Not designed for high-throughput concurrent reads/writes |
| Streamlit single-process server | Stateful; sessions are not isolated across workers |
| No authentication | Any user with network access can query the system |
| Fixed 40-entity knowledge base | Knowledge goes stale; no update mechanism |
| No caching | Every query hits the LLM, even for repeated questions |
| No logging or monitoring | Impossible to measure quality or debug failures in production |

---

## 3. Recommended Production Stack

### 3.1 Model Serving

**Replace Ollama with a dedicated inference server.**

| Option | When to use |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | GPU server; best throughput via continuous batching |
| [Ollama with multiple replicas behind a load balancer](https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image) | Low-cost horizontal scaling via Docker |
| [TGI (Text Generation Inference)](https://github.com/huggingface/text-generation-inference) | Hugging Face stack; streaming, quantization support |

For a privacy-first or air-gapped deployment, **vLLM on a single GPU server** is the most practical option. It supports OpenAI-compatible endpoints, making it a drop-in replacement for the `ollama.generate()` calls already in the codebase.

**Recommended model upgrade:** Move from Mistral 7B to **Mistral 7B Instruct v0.3** or **Llama 3.1 8B Instruct** for better instruction-following and reduced hallucination.

### 3.2 Vector Database

**Replace ChromaDB (SQLite backend) with a production vector database.**

| Option | Notes |
|---|---|
| [Qdrant](https://qdrant.tech/) | Docker-deployable, fast, strong metadata filtering |
| [Weaviate](https://weaviate.io/) | GraphQL API, built-in embedding support |
| [pgvector](https://github.com/pgvector/pgvector) | PostgreSQL extension; reuses existing DB infrastructure |

**Recommendation: Qdrant** — it has a near-identical metadata filtering API to ChromaDB, making migration straightforward. It supports gRPC for low-latency retrieval and has a Docker image for self-hosted deployment.

Migration effort: The `retrieve_context()` function in `rag_engine.py` would need only minor changes to use the Qdrant client instead of ChromaDB.

### 3.3 Application Layer

**Replace Streamlit with a proper web framework for production.**

Streamlit is excellent for prototyping but is not designed for multi-user production use. Recommended migration path:

- **Backend API:** FastAPI + Uvicorn workers (async, handles concurrency)
- **Frontend:** React or Next.js chat interface (or keep Streamlit for internal tools with ≤ 10 concurrent users)

If the audience is small and internal, **Streamlit Community Cloud** or a Docker-deployed Streamlit instance behind nginx is acceptable.

### 3.4 Caching

Add a response cache to avoid redundant LLM calls for repeated questions:

- **Redis** — cache `(query_hash, query_type) → answer` with a 24-hour TTL
- Cache at the `answer_question()` boundary in `rag_engine.py`
- Expected hit rate for FAQ-style usage: 30–50%

### 3.5 Authentication and Access Control

- Add **OAuth2 / SSO** (e.g., Google Workspace or institutional SSO) if deployed for a team
- For a public-facing tool: add rate limiting (e.g., 20 queries/hour per IP via `slowapi`)
- Protect the Ollama/vLLM endpoint behind the application layer — never expose it directly

---

## 4. Knowledge Base Maintenance

The current knowledge base is a one-time Wikipedia snapshot. In production:

| Approach | Description |
|---|---|
| Scheduled re-ingestion | Cron job runs `ingest.py` + `embed_and_store.py` weekly/monthly |
| Incremental updates | Store `ingestion_date` per document; only re-fetch articles older than N days |
| Change detection | Compare Wikipedia revision IDs before re-ingesting |
| Entity expansion | Allow administrators to add new entities via a simple config file or UI |

The SQLite + ChromaDB pipeline already supports clean re-ingestion (`DELETE` + re-insert). Scheduling this as a background job is straightforward.

---

## 5. Observability

Add the following before going to production:

- **Structured logging:** Log query, query_type, retrieval latency, generation latency, and answer length for every request
- **Metrics:** Track p50/p95 latency, error rate, cache hit rate (Prometheus + Grafana)
- **Feedback loop:** Add thumbs-up/thumbs-down buttons in the UI; store feedback in SQLite for quality evaluation
- **Retrieval quality monitoring:** Periodically run a fixed evaluation set (e.g., the example queries from the homework PDF) and compare answers to expected outputs

---

## 6. Containerization

Package the system with Docker for consistent, reproducible deployment:

```dockerfile
# Suggested multi-stage build
FROM python:3.11-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS app
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Use `docker-compose` to orchestrate:
- `app` service (Streamlit / FastAPI)
- `ollama` service (model inference)
- `qdrant` service (vector database)
- `redis` service (response cache)

---

## 7. Scalability Roadmap

| Phase | Changes | Expected Capacity |
|---|---|---|
| **Current (prototype)** | Ollama + ChromaDB + Streamlit on one machine | 1 concurrent user |
| **Phase 1 (small team)** | Docker Compose + Redis cache | 5–10 concurrent users |
| **Phase 2 (department)** | vLLM on GPU server + Qdrant + FastAPI backend | 50–100 concurrent users |
| **Phase 3 (institution)** | Kubernetes + auto-scaling inference pods + managed vector DB | 500+ concurrent users |

---

## 8. Security Considerations

- Never commit `data/` (SQLite + ChromaDB files) to version control — add to `.gitignore`
- Sanitize user inputs before passing to the LLM prompt (prompt injection defense)
- Pin all dependency versions in `requirements.txt` for reproducible builds and vulnerability auditing
- Run the application as a non-root user inside Docker
- Enable HTTPS (TLS termination at nginx or a load balancer) for any non-localhost deployment

---

## 9. Cost Estimate (Self-Hosted)

| Configuration | Hardware | Monthly Cost (est.) |
|---|---|---|
| Prototype | Developer laptop (CPU) | $0 |
| Small team (Phase 1) | 1× VPS, 16 GB RAM, 8 vCPU | ~$50–80/month |
| Department (Phase 2) | 1× A10G GPU server (24 GB VRAM) | ~$300–600/month |

These estimates assume self-hosted infrastructure. A managed GPU cloud (AWS, GCP, Azure) would be higher but provides SLA guarantees.

---

## 10. Summary of Priority Recommendations

1. **Containerize** the application with Docker Compose (highest immediate value)
2. **Add Redis caching** for repeated queries (quick win, significant latency reduction)
3. **Migrate vector DB to Qdrant** when expecting more than a handful of concurrent users
4. **Implement structured logging and a feedback button** before any real-user deployment
5. **Schedule weekly re-ingestion** to keep the knowledge base fresh
6. **Add authentication** before exposing to any network beyond localhost