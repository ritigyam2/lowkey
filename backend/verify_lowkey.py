import os
import json
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
COLLECTION_NAME = "lowkey_songs"
TEST_OUTPUT_PATH = os.path.join(BASE_DIR, "test_results.json")

def verify_and_test():
    print("=" * 60)
    print("   LOWKEY VECTOR DATABASE VERIFICATION & TEST HARNESS   ")
    print("=" * 60)

    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"ChromaDB not found at {CHROMA_PATH}. Run ingest_chroma.py first.")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn
    )

    total_count = collection.count()
    print(f"[*] Verified collection: '{COLLECTION_NAME}'")
    print(f"[*] Total indexed tracks: {total_count}")
    
    # Assert count satisfies target
    assert total_count >= 10000, f"Expected at least 10,000 items, got {total_count}"
    print(f"[PASSED] Verification Check: {total_count} / 10,000 songs stored.")

    # Test Scenarios
    test_scenarios = [
        {
            "test_name": "Lowkey Underground Bedroom Pop",
            "prompt": "melancholic dreamy bedroom pop with soft vocals and lo-fi guitar",
            "filter": {"is_lowkey": True},
            "n_results": 5
        },
        {
            "test_name": "Strict Popularity Filter (Popularity <= 25)",
            "prompt": "dark late night underground electronic techno club",
            "filter": {"popularity": {"$lte": 25}},
            "n_results": 5
        },
        {
            "test_name": "Acoustic Folk & Indie Discovery",
            "prompt": "peaceful warm acoustic indie folk coffee shop",
            "filter": {"is_lowkey": True},
            "n_results": 5
        }
    ]

    test_report = {
        "status": "SUCCESS",
        "total_songs_indexed": total_count,
        "collection_name": COLLECTION_NAME,
        "distance_metric": "cosine",
        "tests": []
    }

    print("\n" + "=" * 60)
    print("RUNNING LIVE SEMANTIC QUERIES WITH LOWKEY FILTERS:")
    print("=" * 60)

    for scenario in test_scenarios:
        prompt = scenario["prompt"]
        where_filter = scenario["filter"]
        n_res = scenario["n_results"]

        print(f"\n[Test: {scenario['test_name']}]")
        print(f"Prompt:  \"{prompt}\"")
        print(f"Filter:  {where_filter}")

        query_res = collection.query(
            query_texts=[prompt],
            n_results=n_res,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        ids = query_res["ids"][0]
        metadatas = query_res["metadatas"][0]
        distances = query_res["distances"][0]

        for rank, (t_id, meta, dist) in enumerate(zip(ids, metadatas, distances), start=1):
            song_info = {
                "rank": rank,
                "track_id": t_id,
                "track_name": meta.get("track_name"),
                "artist": meta.get("artist"),
                "genre": meta.get("genre"),
                "popularity": meta.get("popularity"),
                "is_lowkey": meta.get("is_lowkey"),
                "similarity_score": round(1.0 - dist, 4),
                "cosine_distance": round(dist, 4)
            }
            formatted_results.append(song_info)
            pop_badge = f"[Lowkey | Pop: {meta.get('popularity')}]" if meta.get("is_lowkey") else f"[Pop: {meta.get('popularity')}]"
            print(f"  {rank}. \"{meta.get('track_name')}\" - {meta.get('artist')} ({meta.get('genre')}) {pop_badge} (Similarity: {song_info['similarity_score']})")

        test_report["tests"].append({
            "test_name": scenario["test_name"],
            "query": prompt,
            "filter": where_filter,
            "top_matches": formatted_results
        })

    # Save to test_results.json
    with open(TEST_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[+] All tests passed! Detailed test results exported to: {TEST_OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    verify_and_test()

