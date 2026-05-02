# Production Deployment Recommendation
## Local Wikipedia RAG Assistant

---

## 1. Executive Summary

The current system is a functional single-user localhost prototype. Moving it to production requires addressing four key areas: infrastructure (where it runs), scalability (how many users and entities it can handle), observability (how you know it is working), and cost (what it takes to operate). This document covers the recommended path and the tradeoffs involved.

---

## 2. Current Architecture vs. Production Architecture

| Layer | Current (Localhost) | Recommended (Production) |
|-------|---------------------|--------------------------|
| LLM | Mistral via Ollama on laptop | Dedicated GPU server or managed inference (e.g., vLLM on a GPU instance) |
| Embedding model | all-MiniLM-L6-v2 on CPU | Same model on CPU workers, or upgraded to a larger model if quality requires it |
| Vector database | ChromaDB (file-based SQLite backend) | ChromaDB with a proper HTTP server, or migrate to Qdrant / Weaviate for better horizontal scaling |
| Document database | SQLite flat file | PostgreSQL or managed database service |
| Application server | `streamlit run app.py` (single process) | Containerised app behind a load balancer (e.g., Docker + NGINX) |
| Session state | Streamlit in-memory session state | Redis or a database-backed session store for multi-user support |
| Response cache | Python dict in-process | Redis with TTL-based expiration |
| Deployment | Manual run on laptop | CI/CD pipeline (GitHub Actions) → Docker image → cloud VM or Kubernetes |

---

## 3. Recommended Stack for Production

### 3.1 Inference Server
Replace Ollama running on a laptop with a dedicated inference server:

- **Option A — Self-hosted GPU server:** Deploy Mistral (or a larger model such as Mistral-7B-Instruct) using [vLLM](https://github.com/vllm-project/vllm) on a machine with an NVIDIA GPU (minimum: RTX 3090 / A10G). vLLM supports continuous batching, which dramatically increases throughput under concurrent load.
- **Option B — Managed inference:** If GPU hardware is unavailable or cost-prohibitive, use a privacy-respecting managed inference provider (e.g., Groq, Together AI, or Anyscale) that hosts open-source models. This reintroduces an external API dependency but removes hardware management burden.

**Recommendation:** Option A for a university or enterprise setting where data privacy is paramount; Option B for a startup or prototype moving fast.

### 3.2 Vector Database
ChromaDB's embedded SQLite backend is not designed for concurrent writes or large-scale production use. Recommended migration path:

- **Short term:** Run ChromaDB in HTTP server mode (`chromadb.HttpClient`) behind a persistent volume. This decouples the database from the application process and allows horizontal scaling of the app layer.
- **Long term:** Migrate to **Qdrant** (self-hosted, excellent Python client, supports filtering and payload indexing natively) or **Weaviate** (supports hybrid BM25 + vector search, which would improve retrieval quality for named-entity queries).

### 3.3 Document Database
Replace SQLite with **PostgreSQL**:
- Supports concurrent reads and writes
- Full-text search via `tsvector` can complement vector search for exact-match entity lookups
- Managed options (e.g., Supabase, AWS RDS) reduce operational overhead

### 3.4 Application Layer
- Containerise the Streamlit app with Docker
- Run behind NGINX as a reverse proxy with TLS termination
- Use `gunicorn` or a process manager to keep the app alive under load
- For multi-user support, replace Streamlit's in-memory session state with a Redis-backed store, or migrate the UI to a proper web framework (FastAPI + React) where session management is explicit

### 3.5 Knowledge Base Updates
Production deployments need a refresh pipeline:
- Schedule `ingest.py` and `embed_and_store.py` as nightly or weekly cron jobs
- Use an incremental update strategy: hash each Wikipedia page's content and skip re-ingestion if the hash has not changed
- Version the ChromaDB collection (e.g., `wikipedia_entities_v2`) and perform blue-green swaps to avoid downtime during re-indexing

---

## 4. Scalability Considerations

### 4.1 Data Scale
The current system handles 40 entities (~1,000 chunks). At 10,000+ entities:
- ChromaDB's HNSW index remains performant (sub-100ms queries) up to millions of vectors
- SQLite becomes a bottleneck for concurrent writes during ingestion — migrate to PostgreSQL
- Embedding generation should be parallelised across multiple workers (e.g., Celery task queue)

### 4.2 User Scale
At more than a few concurrent users:
- LLM inference is the primary bottleneck. A single Mistral-7B instance on one GPU handles ~5–10 concurrent streaming requests before latency degrades
- Horizontal scaling requires multiple GPU instances behind a load balancer
- The response cache (currently per-process) must be centralised in Redis so all app instances share it

### 4.3 Query Latency Targets

| Step | Current (laptop CPU) | Production Target (GPU server) |
|------|---------------------|-------------------------------|
| Query condensation (LLM) | ~1–3s | ~0.3–0.5s |
| Query classification (LLM) | ~1–3s | ~0.3–0.5s |
| Embedding generation | ~0.1s | ~0.05s |
| ChromaDB vector search | ~0.1s | ~0.05s |
| LLM answer generation (streaming) | ~5–20s | ~2–5s |
| **Total end-to-end** | **~7–26s** | **~3–6s** |

---

## 5. Observability and Monitoring

Production systems need visibility into failure and degradation:

- **Structured logging:** Replace `print()` statements with Python's `logging` module, writing JSON logs to stdout. Ingest into a log aggregation service (e.g., Loki, Datadog).
- **Metrics:** Instrument with Prometheus counters and histograms for: query latency per step, cache hit rate, retrieval result count, LLM error rate.
- **Alerting:** Alert on: LLM latency p95 > 10s, error rate > 5%, ChromaDB connection failures.
- **Tracing:** Add distributed tracing (e.g., OpenTelemetry) to follow a query through classification → retrieval → generation as a single trace.

---

## 6. Security Considerations

- **Input validation:** Sanitise user input before passing to the LLM prompt to prevent prompt injection attacks.
- **Rate limiting:** Limit queries per user per minute to prevent abuse and protect LLM inference capacity.
- **Authentication:** Add basic authentication or SSO if the system is exposed beyond a trusted network.
- **Data residency:** If the knowledge base contains sensitive documents, ensure the vector database and LLM inference server are deployed in the appropriate geographic region.

---

## 7. Cost Estimate (Self-Hosted GPU Option)

| Component | Specification | Estimated Monthly Cost |
|-----------|--------------|----------------------|
| GPU inference server | 1× NVIDIA A10G (24GB VRAM) on AWS g5.xlarge | ~$600–$900 |
| PostgreSQL | AWS RDS db.t3.medium | ~$50 |
| Redis (cache + sessions) | AWS ElastiCache t3.micro | ~$20 |
| Application server | 2× t3.medium EC2 instances | ~$60 |
| Storage (EBS) | 100 GB for models + databases | ~$10 |
| **Total** | | **~$740–$1,040/month** |

For a university project or low-traffic internal tool, a single GPU instance (e.g., a used RTX 3090 workstation on-premises) can reduce this to near-zero ongoing cost.

---

## 8. Recommended Migration Roadmap

| Phase | Duration | Actions |
|-------|----------|---------|
| **Phase 1 — Containerise** | 1 week | Dockerise the app, add docker-compose with ChromaDB HTTP server and Redis |
| **Phase 2 — GPU inference** | 1 week | Deploy vLLM on a GPU instance, point `LLM_MODEL` at the new endpoint |
| **Phase 3 — Database migration** | 1 week | Migrate SQLite to PostgreSQL, add incremental ingestion pipeline |
| **Phase 4 — Observability** | 1 week | Add structured logging, Prometheus metrics, Grafana dashboard |
| **Phase 5 — Scale testing** | 1 week | Load test with k6 or Locust, tune vLLM concurrency and ChromaDB HNSW parameters |

---

## 9. Summary

The current localhost prototype demonstrates all core RAG concepts correctly and is well-structured for a single-user research tool. The primary changes needed for production are replacing Ollama-on-laptop with a dedicated GPU inference server, moving ChromaDB to HTTP server mode or Qdrant, replacing SQLite with PostgreSQL, and centralising the response cache in Redis. These changes address concurrency, latency, and reliability at scale without requiring a fundamental redesign of the retrieval or generation pipeline.