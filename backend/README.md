# LOWKEY Backend

This directory contains the Python components for the two retrieval experiments described in the repository README.

## Components

- `audio_hnsw_engine.py` — prepares 11-dimensional audio feature vectors and queries an HNSW index using cosine distance
- `test_audio_hnsw.py` — builds or loads the acoustic index and runs a sample similarity query
- `fetch_data.py` — downloads and prepares the 10,000-track dataset
- `ingest_chroma.py` — creates or updates a ChromaDB collection from track metadata and audio attributes
- `verify_lowkey.py` — runs example natural-language queries against the collection
- `requirements.txt` — Python dependencies

## Data and Generated Files

The scripts read datasets from `data/`. The HNSW index, scaler, track metadata files, and ChromaDB collection are generated locally and stored under `data/models/` and `data/chroma_db/`. These generated artifacts are not tracked by Git.

## Setup

From this directory:

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Acoustic similarity:

```bash
python test_audio_hnsw.py
```

Semantic collection ingestion:

```bash
python ingest_chroma.py
```

Example semantic queries:

```bash
python verify_lowkey.py
```

See the [repository README](../README.md) for the design overview and current limitations.
