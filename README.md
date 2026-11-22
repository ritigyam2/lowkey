# LOWKEY

A music-discovery prototype that explores two ways to retrieve tracks: acoustic similarity and text-based search.

The project is implemented as a set of Python backend scripts. It does not currently include a web application or a public recommendation API.

## Retrieval Approaches

### Acoustic similarity

The acoustic retrieval path uses 11 track features:

- Danceability and energy
- Valence and tempo
- Loudness and acousticness
- Instrumentalness and liveness
- Speechiness, key, and mode

The pipeline scales the feature values, applies L2 normalisation, and indexes the resulting vectors with HNSW using cosine distance. The implementation is in `backend/audio_hnsw_engine.py`.

### Text-based retrieval

The semantic path builds a text description from track metadata and selected audio attributes. ChromaDB stores the embeddings and supports natural-language queries with metadata filters, including track popularity. The ingestion and query scripts are:

- `backend/ingest_chroma.py`
- `backend/verify_lowkey.py`

These are separate retrieval paths; the current implementation does not combine them into a hybrid ranking model.

## Data

The backend uses two datasets included in the repository:

- `backend/data/Train-Test Splits/Sequence/train_SEQ.csv` for acoustic similarity
- `backend/data/songs_10k.csv` for text-based retrieval

The generated HNSW model files and ChromaDB database are stored locally under `backend/data/models/` and `backend/data/chroma_db/`; they are not committed.

## Setup

Python 3.12 is used by the GitHub Actions workflow.

```bash
git clone https://github.com/ritigyam2/lowkey.git
cd lowkey/backend

python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Verification Scripts

Build or load the acoustic index and run the sample query:

```bash
python test_audio_hnsw.py
```

Create or update the ChromaDB collection:

```bash
python ingest_chroma.py
```

Run the text-query examples:

```bash
python verify_lowkey.py
```

The scripts write sample outputs to `audio_hnsw_test_output.json` and `test_results.json`.

## Validation and Limitations

The repository includes example query outputs and a CI workflow that runs syntax checks and the acoustic verification script.

The saved outputs are examples, not a formal recommendation-quality evaluation. The project does not currently report recall@k, precision@k, or human relevance judgments. Text retrieval is based on track metadata and audio attributes rather than audio-waveform understanding, and the two retrieval paths have not yet been evaluated as a combined recommender.

## Project Structure

```text
backend/
├── data/
│   ├── Train-Test Splits/
│   └── songs_10k.csv
├── audio_hnsw_engine.py
├── test_audio_hnsw.py
├── fetch_data.py
├── ingest_chroma.py
├── verify_lowkey.py
└── requirements.txt
```

## License

See [LICENSE](LICENSE).
