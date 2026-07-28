# Music Recommendation — ranking-retrieval

Professional overview

This repository contains an open-source research-grade implementation of a music search and retrieval system focused on combining fuzzy search with vector-based retrieval and a lightweight ranking stage. It assembles deterministic data-processing pipelines, a RapidFuzz-powered search layer, optional retrieval with Two-Tower and LightGCN embeddings, FAISS-based nearest-neighbour serving, and a FastAPI backend that exposes search and recommendation endpoints.

Problem statement

Music discovery systems must return relevant results quickly for ambiguous queries (typos, partial names) while also supporting personalized recommendations at scale. This project demonstrates a pragmatic, modular approach that mixes classic information-retrieval techniques (fuzzy matching over normalized catalog fields) with embedding-based retrieval and re-ranking.

Objectives

- Provide a deterministic data ingestion and feature engineering pipeline to produce a canonical tracks dataset and user interactions matrix.
- Implement a robust fuzzy-search layer that can correct and match user queries across tracks, artists, albums and genres.
- Offer a retrieval layer that can use precomputed embeddings (Two-Tower and LightGCN) and FAISS for fast nearest-neighbour lookup.
- Provide a simple ranking model (trainable) to reorder retrieval candidates and blend search relevance, popularity and embedding similarity.
- Expose the functionality via a clean FastAPI service for experimentation and extension.

Current implementation status

- Data processing: Implemented (scripts/process_data.py and backend/services/preprocessing.py)
- Search (RapidFuzz + SearchIndex): Implemented (backend/search/*)
- Retrieval: Implemented as modular components with working Two-Tower and LightGCN model wrappers and FAISS index scaffolding. FAISS usage is optional and requires the faiss library and pre-trained artifacts to be present (not included in repo).
- Ranking: Implemented as a small PyTorch model plus a training script; if a trained model is absent, recommendation service falls back to a dot-product heuristic.
- API: Implemented with FastAPI endpoints for /search, /recommend, /similar and /health.

Core capabilities

- Deterministic data ingestion and cleaning for Spotify and Last.fm CSVs
- Candidate extraction (tracks, artists, albums, genres)
- Typos and query correction using RapidFuzz
- Candidate generation from search results or user embeddings
- Optional FAISS-backed retrieval using precomputed item embeddings
- Trainable Two-Tower and LightGCN model trainers (scripts included)
- Ranking model training and inference (fallback heuristic when absent)

System architecture

Below is a high-level architecture diagram showing the implemented components.

```mermaid
flowchart LR
  subgraph Data
    RAW_SPOTIFY[data/raw/spotify_tracks CSVs]
    RAW_LASTFM[data/raw/last_fm CSVs]
    PROCESS[scripts/process_data.py]
    PROCESSED[data/processed: tracks_cleaned.csv, user_interactions.csv]
  end

  subgraph Backend[Backend (FastAPI)]
    SearchSvc[SearchService]
    RetrievalSvc[RetrievalService]
    RecommendSvc[RecommendationService]
    API[FastAPI routers]
  end

  subgraph SearchLayer[Search Layer]
    SearchIndex[SearchIndex]
    Fuzzy[SearchMatcher (RapidFuzz)]
  end

  subgraph RetrievalLayer[Retrieval Layer]
    TwoTower[TwoTowerEmbeddings (torch)]
    LightGCN[LightGCNEmbeddings (torch)]
    FAISS[FaissIndex (optional faiss lib)]
    CandidateGen[DefaultCandidateGenerator]
    Ranking[RankingModelWrapper (optional)]
  end

  RAW_SPOTIFY --> PROCESS --> PROCESSED
  RAW_LASTFM --> PROCESS --> PROCESSED
  PROCESSED --> SearchSvc
  SearchSvc --> SearchLayer
  SearchSvc --> RetrievalSvc
  RetrievalSvc --> RetrievalLayer
  RetrievalSvc --> CandidateGen
  CandidateGen --> FAISS
  RetrievalSvc --> RecommendSvc
  RecommendSvc --> API
  SearchSvc --> API

  classDef optional fill:#f9f,stroke:#333,stroke-dasharray: 5 5
  class FAISS,TwoTower,LightGCN,Ranking optional
```

Explanation of subsystems

- Backend: FastAPI application located at backend/api/main.py. The application constructs SearchService and RetrievalService at startup and attempts to load optional retrieval artifacts (Two-Tower, LightGCN, Ranking, FAISS).

- Search: Implemented in backend/search. SearchIndex builds candidate lists from processed tracks CSV columns (track, artist, album, genres). SearchMatcher uses RapidFuzz to score and correct queries. SearchEngine orchestrates correction + per-category matching.

- Retrieval: Implemented under backend/retrieval. Two model wrappers are provided:
  - TwoTowerEmbeddings: loads a saved PyTorch Two-Tower model checkpoint and exposes user/item embedding matrices.
  - LightGCNEmbeddings: loads precomputed user/item embeddings that come from a LightGCN training run.
  - FaissIndex: reads a FAISS binary file and a pickled track-id mapping; optional — Faiss must be installed to use it.
  - CandidateGenerator: DefaultCandidateGenerator maps search candidates to track IDs and optionally queries FAISS using embeddings.

- API: Routers live under backend/api/routers. Endpoints:
  - GET /search — fuzzy search returning corrected_query and category-tagged matches
  - GET /recommend — recommendation endpoint combining user-based and query-based signals
  - GET /similar — fetch similar tracks using FAISS and item embeddings
  - GET /health — simple readiness indicators

Planned Components (not implemented / placeholders)

- EmbeddingStore: backend/retrieval/embedding_store.py is a placeholder; loading and runtime embedding retrieval are NotImplemented.
- No pre-trained model binaries or FAISS artifacts are checked in. Training scripts are provided, but running them requires data and GPU/torch environment.
- Production-grade features (authentication, metrics, batching, async workers) are not included.

Repository structure

- backend/: Python package containing the API, search, retrieval, and service layers.
  - backend/api/: FastAPI application and routers
  - backend/search/: search index and fuzzy matching logic
  - backend/retrieval/: retrieval models, FAISS index wrappers, candidate generation and ranking
  - backend/services/: preprocessing, search wrapper and recommendation service
  - backend/config.py: path and numeric configuration constants

- data/: storage for raw and processed CSVs. The repository includes sample raw CSVs in data/raw/ but the processed dataset is produced by scripts/process_data.py.

- models/: intended location for saved model artifacts (Two-Tower, LightGCN, ranking, FAISS index)

- scripts/: convenience scripts to process data, train models and build FAISS index.

- tests/: pytest-based unit tests covering search, preprocessing, retrieval scaffolding and config assertions.

Data pipeline

Datasets
- Raw inputs expected: data/raw/spotify_tracks/*.csv and data/raw/last_fm/*.csv
- Process outputs: data/processed/tracks_cleaned.csv and data/processed/user_interactions.csv

Preprocessing
- Implemented in backend/services/preprocessing.py and exposed via scripts/process_data.py.
- Normalizes textual fields (unicode NFKD, ASCII transliteration, lowercasing), deduplicates by normalized (artist,track), creates display-friendly title-cased fields.

Feature engineering
- Genre lists parsed into genres_list (list of normalized genre names)
- Artist frequency and numeric features computed (popularity, duration)
- Genre MultiLabelBinarizer, LabelEncoder for artist and StandardScaler for numeric features are created and saved to artifacts (genre_encoder.pkl, artist_encoder.pkl, feature_scaler.pkl)

Encoders / scalers
- Persisted using pickle into data/processed/ per backend/config paths

Interaction matrix
- Extracted from Last.fm-style user play records by mapping user + (artist_clean, track_clean) -> track_id from the merged tracks table.

Generated artefacts
- data/processed/tracks_cleaned.csv (canonical track table)
- data/processed/user_interactions.csv (aggregated playcounts)
- data/processed/*.pkl (genre, artist encoder, scaler)

Search engine

Implemented design
- SearchIndex (backend/search/search_index.py) inspects processed track DataFrame and builds candidate lists for categories: track, artist, album, genre. For genres it flattens lists and various string formats.
- SearchMatcher (backend/search/fuzzy_search.py) uses RapidFuzz's WRatio and token-based scorers to provide robust similarity metrics and a correction routine.
- SearchEngine (backend/search/search_engine.py) runs correction then per-category matching and returns a merged list of SearchResultItems sorted by score.

RapidFuzz
- Used for matching and suggestion. Score thresholds and correction thresholds are configurable in backend/config.py (SEARCH_SCORE_THRESHOLD, CORRECTION_SCORE_THRESHOLD).

Search indexing
- In-memory index built from processed CSV; no on-disk search index is used. For production, an inverted-index or dedicated text index could be added.

Query correction
- _correct_query uses RapidFuzz extractOne against all candidates and only replaces the query when similarity exceeds CORRECTION_SCORE_THRESHOLD.

Ranking
- Per-result ranking within SearchEngine sorts by (score, category) descending. The RecommendationService uses a separate training-capable ranking model to score candidates.

API behavior and supported search types
- /search accepts q parameter and returns corrected_query + results where each result has name, category and score. Supported categories: track, artist, album, genre.

Limitations
- Search is in-memory and therefore limited by the host memory and the processed dataset size.
- No language-aware normalization beyond ASCII transliteration and whitespace/punctuation handling.

Retrieval layer

Current implementation
- Two-Tower: Torch-based model class (backend/retrieval/two_tower.py) with helpers to save/load a checkpoint that includes model_state, mapping dicts and embedding matrices. TwoTowerEmbeddings exposes user/item embeddings as numpy arrays for retrieval.
- LightGCN: Model and helpers under backend/retrieval/lightgcn.py. Training script computes adjacency, propagates embeddings and persists precomputed embeddings.
- FAISS: FaissIndex wraps reading a binary FAISS index and a pickled track-id list, and exposes query() -> (track_ids, distances).
- Candidate generation: DefaultCandidateGenerator maps search result names to track_ids using exact and substring matches over normalized display fields. It optionally queries FAISS using item embeddings for nearest-neighbour expansion.

Notes and Partial Implementations
- EmbeddingStore is a placeholder and raises NotImplementedError: it is not used by default but indicates future plans for a runtime embedding service.
- The repository provides training scripts but no prebuilt model artifacts. Running training produces model files under models/ and the FAISS index builder requires faiss to be installed.

API documentation (endpoints)

All routers are implemented under backend/api/routers. The following endpoints are present and match the FastAPI schemas (backend/api/schemas.py).

1) GET /search
- Method: GET
- Path: /search?q={query}
- Description: Run a fuzzy search across tracks, artists, albums and genres. Also returns a corrected_query if a close candidate is found.
- Parameters: q (string, required)
- Example request: GET /search?q=taylr%20swft
- Example response:
  {
    "corrected_query": "Taylor Swift",
    "results": [
      {"name": "Taylor Swift", "category": "artist", "score": 98.3},
      {"name": "Blank Space", "category": "track", "score": 76.5}
    ]
  }
- Possible errors: 400 for invalid inputs (handled by FastAPI), 503 if service not available at startup, 500 on unexpected errors.

2) GET /recommend
- Method: GET
- Path: /recommend?user_id={user_id}&q={query}&limit={limit}
- Description: Return blended recommendations. If user_id provided, retrieval uses user embeddings; if query provided, search-based candidates are added. Results are scored using the ranking model if available, otherwise a dot-product heuristic is used.
- Parameters: user_id (optional), q (optional), limit (int, default 20)
- Example request: GET /recommend?user_id=user1&limit=10
- Example response: {"user_id": "user1", "query": null, "corrected_query": "", "results": [{...}]}
- Possible errors: 400 if neither user_id nor q provided, 503 if recommendation service unavailable, 500 on unexpected errors.

3) GET /similar
- Method: GET
- Path: /similar?track_id={track_id}&limit={limit}
- Description: Return tracks similar to a given track_id using item embeddings + FAISS when available. Falls back to popular tracks when FAISS or embeddings are unavailable.
- Parameters: track_id (required), limit (int, default 20)
- Example request: GET /similar?track_id=123
- Example response: {"track_id":"123","results":[{"track_id":"456","track":"...","artist":"...","score":0.12}, ...]}

4) GET /health
- Method: GET
- Path: /health
- Description: Returns health and readiness flags: search_ready, retrieval_ready, ranking_ready.

Configuration

Key configuration variables are defined in backend/config.py. Important values:
- ROOT_DIR, DATA_DIR, PROCESSED_DATA_DIR
- TRACKS_CLEANED_PATH (data/processed/tracks_cleaned.csv)
- USER_INTERACTIONS_PATH (data/processed/user_interactions.csv)
- MODEL_DIR (models/)
- TWO_TOWER_MODEL_PATH, LIGHTGCN_MODEL_PATH, RANKING_MODEL_PATH (expected model artifact paths)
- FAISS_INDEX_DIR, FAISS_INDEX_PATH, FAISS_TRACK_IDS_PATH
- Default numeric settings: DEFAULT_SEARCH_LIMIT, MAX_SEARCH_RESULTS, SEARCH_SCORE_THRESHOLD, CORRECTION_SCORE_THRESHOLD, EMBEDDING_DIMENSION, RETRIEVAL_TOP_K, RETRIEVAL_CANDIDATE_LIMIT, RECOMMENDATION_TOP_K

Important runtime notes
- Some Config attributes referenced in code (e.g. EMBEDDING_MODEL_PATH, RETRIEVAL_MODEL_PATH, LIGHTGCN_MODEL_PATH) must exist or be provided before attempting to load models. The application attempts to load optional models at startup and logs warnings if they are absent.

Installation

This project targets modern Python (3.10+) and uses PyTorch. Minimal steps to run locally:

1) Create a virtual environment

   python -m venv .venv
   .\.venv\Scripts\activate   (Windows)
   source .venv/bin/activate   (Linux/macOS)

2) Install dependencies

   pip install -r requirements.txt

   Notes:
   - FAISS is optional but required for FAISS-based retrieval and index building; install platform-specific faiss if needed.
   - PyTorch installation depends on your CUDA availability — follow instructions at https://pytorch.org/

3) Prepare data

   Place raw CSVs in data/raw/spotify_tracks and data/raw/last_fm and run:

   python scripts/process_data.py

   This writes data/processed/tracks_cleaned.csv and user_interactions.csv and saves encoder artifacts.

4) (Optional) Train models

   - Two-Tower: python scripts/train_two_tower.py
   - LightGCN: python scripts/train_lightgcn.py
   - Ranking: python scripts/train_ranking.py

5) (Optional) Build FAISS index

   python scripts/build_faiss.py

6) Run API

   python -m backend.api.main

Testing

- Tests use pytest and are in tests/.
- Test coverage focuses on search fuzzy matching, preprocessing and basic retrieval scaffolding.
- To run tests: pip install -r requirements-dev.txt && pytest -q

Current project status (feature matrix)

| Component | Status |
|-----------|--------|
| Data Engineering | Complete |
| Search Engine | Complete |
| RapidFuzz integration | Complete |
| Retrieval (FAISS scaffolding) | Partial (requires faiss & model artifacts) |
| Two-Tower model | Implemented (training + load/save) |
| LightGCN model | Implemented (training + load/save) |
| Ranking model | Implemented (training + load/save), fallback heuristic when missing |
| EmbeddingStore runtime | Not Implemented (placeholder) |
| Recommendation Service | Complete (uses above pieces when available) |
| React Frontend | Not Present |

Roadmap

Completed
- Deterministic data pipeline
- RapidFuzz-based search and correction
- Retrieval scaffolding with Two-Tower and LightGCN loaders

In Progress
- Packaging pre-trained artifacts (not included)
- Optional FAISS index builder integration test on CI

Planned
- EmbeddingStore implementation for runtime retrieval
- Productionization (async workers, metrics, CI packaging)

Future improvements
- Replace naive in-memory search with a production text index for large catalogs
- Add streaming ingestion and incremental FAISS updates
- Add thorough unit and integration tests for retrieval with real model artefacts

Development notes

- Architectural decisions favor modularity: search and retrieval are separate to allow hybrid strategies.
- RapidFuzz was chosen for fast, precise fuzzy matching without external infra.
- FAISS is optional to keep the repository runnable without heavy dependencies.

Code quality review (high-level findings)

- Several Config attributes are referenced but not defined consistently (e.g. EMBEDDING_MODEL_PATH, LIGHTGCN_MODEL_PATH references). This leads to potential AttributeError at runtime when code attempts to access Config.<missing>.
- EmbeddingStore is explicitly a placeholder and raises NotImplementedError.
- tests/test_config.py expects Config.TRACKS_CLEANED_PATH to exist; tests assume processed data presence which requires running scripts/process_data.py before tests in many environments.
- requirements.txt appears large and contains many development notebooks-related packages; consider trimming to runtime essentials for production usage.

Where to start contributing

- To reproduce the system locally: run scripts/process_data.py with small sample CSVs (there are sample CSVs under data/raw/). Then run the API.
- To enable full retrieval tests: train Two-Tower (scripts/train_two_tower.py), build FAISS (scripts/build_faiss.py) and place artifacts under models/.

License & Contribution

This repository contains example research code. Add a LICENSE file as appropriate for your project before publishing publicly.


---

For more detailed developer-level documentation, see the docs/ directory.