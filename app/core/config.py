"""Centralized runtime configuration.

Every tunable named in PRD_TRD.md (§4 requirements, §8 stack, §10 roadmap) lives here as an
environment-backed setting. No pipeline module may hardcode a threshold, timeout, model name,
or path inline — see PRD_TRD.md §9.8 (original TRD) design principle, carried into this freeze.
"""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Secrets / external services ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    api_key: str = Field(default="change_me_local_dev", alias="API_KEY")
    eval_key: str = Field(default="change_me_eval", alias="EVAL_KEY")

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    sqlite_path: str = Field(default="./data/app.db", alias="SQLITE_PATH")

    # --- Models ---
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    llm_model: str = Field(default="claude-sonnet-5", alias="LLM_MODEL")
    spacy_model: str = Field(default="en_core_web_sm", alias="SPACY_MODEL")

    # --- Corpus / data locations (REQ-CORPUS-01, REQ-CORPUS-02) ---
    corpus_root: str = Field(default="data/corpus", alias="CORPUS_ROOT")
    processed_dir: str = Field(default="data/processed", alias="PROCESSED_DIR")
    default_corpus_id: str = Field(default="default", alias="DEFAULT_CORPUS_ID")

    # --- Ingestion chunking (scripts/ingest_corpus.py) ---
    chunk_max_tokens: int = Field(default=200, gt=0, alias="CHUNK_MAX_TOKENS")
    chunk_overlap_tokens: int = Field(default=40, ge=0, alias="CHUNK_OVERLAP_TOKENS")

    # --- Embedding call policy (original TRD §9.4: 500ms timeout, 1 retry, 200ms backoff) ---
    embedding_timeout_ms: int = Field(default=500, gt=0, alias="EMBEDDING_TIMEOUT_MS")
    embedding_retry_backoff_ms: int = Field(default=200, ge=0, alias="EMBEDDING_RETRY_BACKOFF_MS")

    # --- Retrieval Controller (REQ-CTRL-01/02/03) ---
    stability_threshold: float = Field(default=0.90, alias="STABILITY_THRESHOLD")
    max_wait_chunks: int = Field(default=6, alias="MAX_WAIT_CHUNKS")

    # --- Multi-Intent Decomposer (REQ-INTENT-01/02) ---
    merge_threshold: float = Field(default=0.85, alias="MERGE_THRESHOLD")

    # --- Evidence fusion / reranking (REQ-EVID-01..04) ---
    dedup_threshold: float = Field(default=0.95, alias="DEDUP_THRESHOLD")
    min_relevance: float = Field(default=0.35, alias="MIN_RELEVANCE")
    k_dense: int = Field(default=20, alias="K_DENSE")
    k_sparse: int = Field(default=20, alias="K_SPARSE")
    rerank_candidates: int = Field(default=30, alias="RERANK_CANDIDATES")
    final_k: int = Field(default=6, alias="FINAL_K")
    max_total_evidence: int = Field(default=15, alias="MAX_TOTAL_EVIDENCE")
    evidence_token_budget: int = Field(default=3000, alias="EVIDENCE_TOKEN_BUDGET")
    max_concurrent_retrievals: int = Field(default=8, alias="MAX_CONCURRENT_RETRIEVALS")
    retrieval_timeout_ms: int = Field(default=800, alias="RETRIEVAL_TIMEOUT_MS")
    rrf_k: int = Field(default=60, alias="RRF_K")

    # --- Grounding (REQ-GROUND-01/02/03) ---
    entailment_min: float = Field(default=0.30, alias="ENTAILMENT_MIN")

    # --- Session refinement (REQ-SESS-01/02) ---
    refinement_threshold: float = Field(default=0.75, alias="REFINEMENT_THRESHOLD")

    # --- Session lifecycle (REQ-SEC-01/02) ---
    session_ttl_seconds: int = Field(default=1800, alias="SESSION_TTL_SECONDS")

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_max_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_MAX_TOKENS")
        return self


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton. Use this accessor, never instantiate Settings() directly,
    so every module observes the same configuration and tests can override via env vars cleanly.
    """
    return Settings()
