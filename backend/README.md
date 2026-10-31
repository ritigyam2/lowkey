# `lowkey` Backend

Song recommendation and underground music discovery vector database engine for **lowkey**.

## Overview
- **10,000 Track Dataset:** Curated Spotify tracks with rich audio features (energy, valence, tempo, danceability) and popularity scores ($0\text{--}100$), with a focus on lowkey/underground tracks ($\le 35$ popularity).
- **#nowplaying-RS Sequence Dataset:** 20,631 unique tracks deduplicated from `train_SEQ.csv` for 11-dimensional audio feature similarity and sequence-aware experiments.
- **Audio Similarity HNSW Engine:** High-performance Approximate Nearest Neighbor (ANN) vector graph using `hnswlib` with Cosine distance indexing over normalized 11-D acoustic traits.
- **Semantic Text Vector Store:** ChromaDB with ONNX-based `all-MiniLM-L6-v2` dense embeddings for natural language vibe matching under strict RAM constraints ($< 150\,\text{MB}$).

---

## Directory Structure
```
backend/
├── data/
│   ├── songs_10k.csv                  # 10,000 clean tracks with audio features & metadata
│   ├── Train-Test Splits/             # #nowplaying-RS Context and Sequence splits
│   │   ├── Context/                   # train.csv, test.csv
│   │   └── Sequence/                  # train_SEQ.csv (253k rows), test_SEQ.csv
│   ├── chroma_db/                     # ChromaDB vector store (ignored in git)
│   └── models/                        # Serialized HNSW index & scalers (ignored in git)
│       ├── audio_hnsw.bin             # HNSW binary vector graph (Cosine space)
│       ├── audio_scaler.joblib        # Fitted MinMaxScaler (clip=True)
│       ├── audio_track_ids.json       # Index-to-track_id array
│       └── audio_track_features.json  # Raw 11-feature metadata lookup
├── audio_hnsw_engine.py               # HNSW 11-D audio similarity engine
├── test_audio_hnsw.py                 # Automated verification harness for HNSW engine
├── audio_hnsw_test_output.json        # Output report of top-5 audio similarity search
├── fetch_data.py                      # Dataset acquisition & preprocessing (10k dataset)
├── ingest_chroma.py                   # Low-RAM streaming batch embedding & ChromaDB upsert
├── verify_lowkey.py                   # ChromaDB semantic query verification harness
├── test_results.json                  # ChromaDB verification test run output
└── requirements.txt                   # Python dependencies
```

---

## Setup & Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Audio Similarity Retrieval using HNSW (ANN)
To build/load the HNSW vector graph on the 11 audio dimensions and execute the query search for the input track:
```bash
python test_audio_hnsw.py
```
Outputs the top-5 acoustically similar tracks with cosine similarity scores and detailed audio traits to `audio_hnsw_test_output.json`.

### 3. Ingest Semantic Tracks into ChromaDB
```bash
python ingest_chroma.py
```

### 4. Run Semantic Vibe Verification Tests
```bash
python verify_lowkey.py
```
This runs live semantic search queries with lowkey popularity filters and outputs the results to `test_results.json`.
