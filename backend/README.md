# `lowkey` Backend

Song recommendation and underground music discovery vector database engine for **lowkey**.

## Overview
- **10,000 Track Dataset:** Curated Spotify tracks with rich audio features (energy, valence, tempo, danceability) and popularity scores ($0\text{--}100$), with a focus on lowkey/underground tracks ($\le 35$ popularity).
- **Vector Database:** ChromaDB with Cosine similarity indexing.
- **Embeddings:** ONNX-based `all-MiniLM-L6-v2` embedding model for semantic search and mood matching under strict RAM constraints ($< 150\,\text{MB}$).

---

## Directory Structure
```
backend/
├── data/
│   ├── songs_10k.csv           # 10,000 clean tracks with audio features & metadata
│   └── chroma_db/              # ChromaDB vector store (ignored in git)
├── fetch_data.py               # Dataset acquisition & preprocessing
├── ingest_chroma.py            # Low-RAM streaming batch embedding & ChromaDB upsert
├── verify_lowkey.py            # Automated test assertions & sample queries
├── test_results.json           # Verification test run output
└── requirements.txt            # Python dependencies
```

---

## Setup & Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ingest Tracks into Vector Database
```bash
python ingest_chroma.py
```

### 3. Run Verification Tests & Queries
```bash
python verify_lowkey.py
```
This runs live semantic search queries with lowkey popularity filters and outputs the results to `test_results.json`.
