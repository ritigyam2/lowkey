import os
import gc
import time
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
CSV_PATH = os.path.join(BASE_DIR, "data", "songs_10k.csv")
COLLECTION_NAME = "lowkey_songs"
BATCH_SIZE = 500

def create_song_document(row):
    """
    Creates a rich semantic text document describing the song.
    Embeddings will be generated from this text for natural language & vibe queries.
    """
    album = str(row.get('album_name', 'Single/Album')).strip()
    return (
        f"Track: {row['track_name']} | "
        f"Artist: {row['artists']} | "
        f"Genre: {row['track_genre']} | "
        f"Album: {album} | "
        f"Vibe: Energy {row['energy']:.2f}, Mood/Valence {row['valence']:.2f}, Tempo {row['tempo']:.1f} BPM, Danceability {row['danceability']:.2f} | "
        f"Popularity: {int(row['popularity'])}/100"
    )

def ingest_songs():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}. Please run fetch_data.py first.")

    print(f"[*] Initializing Persistent ChromaDB at: {CHROMA_PATH}")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    # Initialize Persistent Chroma Client
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Chroma default uses all-MiniLM-L6-v2 running on lightweight ONNX (< 100MB RAM)
    emb_fn = embedding_functions.DefaultEmbeddingFunction()

    # Get or create collection configured for Cosine similarity
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"[*] Reading dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    total_records = len(df)
    print(f"[*] Total tracks to ingest: {total_records}")

    start_time = time.time()
    
    # Ingest in chunks of BATCH_SIZE to preserve RAM
    for i in range(0, total_records, BATCH_SIZE):
        batch_df = df.iloc[i : i + BATCH_SIZE]
        batch_start = time.time()

        ids = [str(row['track_id']) for _, row in batch_df.iterrows()]
        documents = [create_song_document(row) for _, row in batch_df.iterrows()]
        
        metadatas = [
            {
                "track_id": str(row['track_id']),
                "track_name": str(row['track_name']),
                "artist": str(row['artists']),
                "album": str(row.get('album_name', '')),
                "genre": str(row['track_genre']),
                "popularity": int(row['popularity']),
                "is_lowkey": bool(row['is_lowkey']),
                "danceability": float(row.get('danceability', 0.0)),
                "energy": float(row.get('energy', 0.0)),
                "valence": float(row.get('valence', 0.0)),
                "tempo": float(row.get('tempo', 0.0)),
                "spotify_id": str(row['track_id'])
            }
            for _, row in batch_df.iterrows()
        ]

        # Upsert into ChromaDB
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        batch_time = time.time() - batch_start
        processed = min(i + BATCH_SIZE, total_records)
        print(f"[{processed:>5}/{total_records}] Ingested batch {i // BATCH_SIZE + 1} ({len(batch_df)} tracks) in {batch_time:.2f}s")

        # Explicit low-RAM cleanup
        del ids, documents, metadatas, batch_df
        gc.collect()

    total_time = time.time() - start_time
    final_count = collection.count()
    print(f"\n[+] Ingestion Complete in {total_time:.1f}s!")
    print(f"[+] Total indexed items in '{COLLECTION_NAME}' collection: {final_count}")
    return final_count

if __name__ == "__main__":
    ingest_songs()

