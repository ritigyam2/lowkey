# lowkey
> **Break out of the algorithmic echo chamber.** An underground, indie, and low-stream song discovery engine .

---

## The Concept
Mainstream streaming services (Spotify, Apple Music) optimize for listener retention by looping viral hits, trending charts, and songs you already know. Even algorithmic features like *Discover Weekly* rarely surface tracks under 50,000 streams. Overlooked B-sides, bedroom-pop artists, and niche sub-genres are buried in the long tail.

**lowkey** is built to surface hidden gems:
- **Zero Echo Chamber:** Discovers music based on semantic vibe and sonic properties rather than stream count.
- **Underground-First:** Actively prioritizes tracks with low popularity scores ($\le 35/100$) so listeners discover music they have genuinely never heard.
- **Smart Semantic Search:** Query by natural mood, aesthetic, or vibe (e.g., *"melancholic bedroom pop with soft lo-fi guitar"* or *"dark late-night underground techno"*).

---

## Architecture & Tech Stack

```
┌─────────────────────────┐     ┌──────────────────────────┐     ┌────────────────────────┐
│ Hugging Face Spotify DB │ ──> │ Low-RAM Batch Embedder   │ ──> │ ChromaDB Vector Store  │
│ (114k Tracks Dataset)   │     │ (ONNX all-MiniLM-L6-v2)  │     │ (10,000 Vectors)       │
└─────────────────────────┘     └──────────────────────────┘     └───────────┬────────────┘
                                                                             │
                                ┌──────────────────────────┐                 │
                                │ Semantic Query Harness   │ <───────────────┘
                                │ & Popularity Filter      │
                                └──────────────────────────┘
```

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) (v1.5+) | Embedded vector storage with persistent SQLite + HNSW index using Cosine space (`hnsw:space: cosine`). |
| **Embedding Model** | `all-MiniLM-L6-v2` (ONNX Runtime) | Lightweight 384-dimensional dense semantic text embeddings. Avoids heavy PyTorch memory overhead. |
| **Low-RAM Ingestion** | Custom Python Engine | Streams tracks in batches of 500 with explicit garbage collection (`gc.collect()`), keeping memory footprint under **150MB RAM**. |
| **Data Processing** | `pandas`, `requests` | Cleans, deduplicates, and structures audio metrics and popularity thresholds. |

---

## Dataset Information

* **Source:** [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (~114k tracks across 114 genres) by Maharshi Pandya on Hugging Face.
* **Processed Dataset:** [`backend/data/songs_10k.csv`](backend/data/songs_10k.csv) (10,000 clean, unique tracks).
* **Sampling Distribution:**
  * **60% Lowkey Tracks (6,000 songs):** Strictly filtered to tracks with popularity $\le 35$ ($0\text{--}35$).
  * **40% Diverse Tracks (4,000 songs):** Balanced across 113 unique genres for rich contrast and discovery.
* **Indexed Metadata Attributes:**
  * `track_name` (str) & `artist` (str)
  * `genre` (str) — e.g. *shoegaze, idm, ambient, bedroom-pop, minimal-techno, afrobeat, emo*
  * `popularity` (int, $0\text{--}100$)
  * `is_lowkey` (bool) — `True` if popularity $\le 35$
  * `energy`, `valence` (mood positivity), `tempo` (BPM), and `danceability` (float)
  * `spotify_id` (str) for playback and playlist export

---

## Repository Structure

```
lowkey/
├── .gitignore                   # Ignores large raw caches and binary Chroma sqlite DBs
├── LICENSE                      # Project license
├── README.md                    # Main project overview & documentation
└── backend/
    ├── data/
    │   └── songs_10k.csv        # Preprocessed 10k tracks ready for indexing (1.8 MB)
    ├── fetch_data.py            # Dataset acquisition, sanitization & 10k sampling
    ├── ingest_chroma.py         # Batch embedder & persistent ChromaDB upsert engine
    ├── verify_lowkey.py         # Test harness with live semantic queries & assertions
    ├── test_results.json        # Output report of sample recommendation queries
    ├── requirements.txt         # Backend Python dependencies
    └── README.md                # Backend-specific guide
```

---

## Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/ritigyam2/lowkey.git
cd lowkey/backend
pip install -r requirements.txt
```

### 2. Ingest Tracks into Vector Database
Embeds and saves the 10,000 songs into ChromaDB (`backend/data/chroma_db/`):
```bash
python ingest_chroma.py
```
> *Batch size is set to 500 to maintain minimal RAM usage throughout the process.*

### 3. Verify & Run Test Search Queries
```bash
python verify_lowkey.py
```
Validates the 10,000-vector index count and performs test queries with popularity filters, saving the results to `backend/test_results.json`.

---

## Verified Search Examples

Below are actual results returned from our verification test harness (`verify_lowkey.py`):

### Query 1: *"melancholic dreamy bedroom pop with soft vocals and lo-fi guitar"*
> **Filter:** `where={"is_lowkey": True}` (Popularity $\le 35$)

| Rank | Track | Artist | Genre | Popularity | Similarity Score |
| :---: | :--- | :--- | :--- | :---: | :---: |
| 1 | **Sweet Dreams** | Cameron's Bedtime Classics | `disney` | **19/100** | 0.6179 |
| 2 | **Dr Dream** | Melorman | `idm` | **12/100** | 0.6102 |
| 3 | **Watery Brown Noise** | White Noise Baby Sleep | `sleep` | **34/100** | 0.5693 |
| 4 | **Send Me the Pillow That You Dream On** | Johnny Tillotson | `rock-n-roll` | **32/100** | 0.5671 |
| 5 | **Back to My Bed** | Elderbrook | `house` | **0/100** | 0.5574 |

### Query 2: *"dark late night underground electronic techno club"*
> **Filter:** `where={"popularity": {"$lte": 25}}`

| Rank | Track | Artist | Genre | Popularity | Similarity Score |
| :---: | :--- | :--- | :--- | :---: | :---: |
| 1 | **Midnight Escape Finale** | Kick Puncher | `club` | **15/100** | 0.5980 |
| 2 | **Analog Feel (Greenhaven DJs Remix)** | Cosmic Gate; Greenhaven DJs | `german` | **0/100** | 0.5355 |
| 3 | **Freaks of the Night** | Fabio Fusco; Marc Deal | `trance` | **0/100** | 0.5334 |
| 4 | **2,20 Minutes Foot Walk** | Tak Dec | `minimal-techno` | **17/100** | 0.5207 |
| 5 | **Charlie Hotel 155** | Underground Resistance | `detroit-techno`| **5/100** | 0.5196 |

---

## Roadmap
- [x] Phase 1: Curate & sanitize 10,000-track dataset with popularity and audio traits.
- [x] Phase 2: Implement low-RAM ONNX embedding pipeline with ChromaDB.
- [x] Phase 3: Automated verification & similarity search harness with lowkey filters.
- [ ] Phase 4: Build FastAPI REST API endpoints (`/recommend`, `/vibe-search`, `/similar`).
- [ ] Phase 5: Develop Web / Mobile frontend UI with 30s audio previews and 1-click Spotify playlist export.
