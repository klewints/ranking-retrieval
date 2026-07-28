ARCHITECTURE

This document expands the architecture summary from README.md and provides more detailed design notes.

1. High-level decomposition

- Data Layer
  - Raw CSVs (data/raw/*) are consumed by scripts/process_data.py.
  - Processed outputs (data/processed/tracks_cleaned.csv, user_interactions.csv) are canonical artifacts used by SearchService and RetrievalService.

- Search Layer
  - SearchIndex: builds candidate lists from processed table columns using heuristics for field names.
  - SearchMatcher: RapidFuzz based text normalization and scoring utilities.
  - SearchEngine: orchestrates correction and category-scoped matching.

- Retrieval Layer
  - TwoTowerEmbeddings and LightGCNEmbeddings: wrappers that either load model checkpoints (torch) and expose embedding matrices.
  - FaissIndex: optional index loader that reads a binary FAISS index and a pickled mapping of indices->track IDs.
  - CandidateGenerator: maps search hits to track IDs and expands candidates via FAISS when available.
  - RankingModelWrapper: wraps a small feed-forward network for candidate scoring.

- API Layer
  - FastAPI application (backend/api/main.py) registers routers and constructs application singletons in lifespan.
  - Routers provide simple, schema-driven endpoints described in docs/API.md.

2. Data flow for a typical recommendation request (/recommend)

- Request arrives with user_id and/or query
- If query present: SearchService.search(query) -> search candidates
- RetrievalService.retrieve_by_user(user_id) and/or retrieve_by_search(search_results) produce candidate track IDs
- RecommendationService._score_candidates builds feature vectors using embeddings, popularity and genre overlap
- RankingModelWrapper (if present) scores features; else fallback dot-product heuristic is used
- Top K results returned

3. Startup behavior

- Application constructs SearchService (loads processed CSV) and SearchEngine index in FastAPI lifespan
- Retrieves optional artifacts: Two-Tower, LightGCN, Ranking model. All loads are attempted and failures are logged as warnings but do not stop the server
- RetrievalService attempts to load FAISS index. If FAISS artifacts are missing or faiss library is not installed, retrieval endpoints will fall back to popularity-based candidates or raise errors depending on the call

4. Observability & failure modes

- Missing processed dataset: SearchService raises FileNotFoundError at startup -> application will log and may not become fully ready.
- Missing faiss or faiss files: FaissIndex.load() raises RuntimeError/FileNotFoundError. RetrievalService.is_ready() will return False.
- Missing ranking model: RecommendationService still operates using a dot-product heuristic.

5. Extensibility points

- Add a persistent text index (e.g., Whoosh/Elasticsearch/Meili) to replace SearchIndex for very large catalogs.
- Implement EmbeddingStore to host model-based embedding inference for items or to connect to an external embedding service.
- Replace persistence of artifacts with a more robust artifact store (S3, Model Registry) and make loading asynchronous.

