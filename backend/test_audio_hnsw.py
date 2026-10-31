import os
import json
import logging
from audio_hnsw_engine import AudioHNSWEngine, AUDIO_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestAudioHNSW")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "audio_hnsw_test_output.json")

# Cleaned input song payload assigned by Senior SWE
CLEANED_INPUT_PAYLOAD = {
    "danceability": 0.9802,
    "energy": 0.9793,
    "valence": 0.5625,
    "bpm": 116.69,
    "loudness_db": -20.82,
    "acousticness": 0.0071,
    "instrumentalness": 0.1396,
    "liveness": 0.0,
    "speechiness": 0.0,
    "key_int": 0,
    "mode": 1
}


def run_verification():
    print("=" * 80)
    print("   LOWKEY - AUDIO SIMILARITY RETRIEVAL (HNSW / ANN) VERIFICATION HARNESS   ")
    print("=" * 80)

    engine = AudioHNSWEngine()

    print("[*] Initializing HNSW Audio Engine (Loading / Building Index)...")
    total_tracks = engine.build_index(force_rebuild=False)
    print(f"[+] Total unique tracks in HNSW index: {total_tracks}")
    assert total_tracks == 20631, f"Expected 20,631 unique tracks, got {total_tracks}"

    print("\n[*] Input Query Payload:")
    print(json.dumps(CLEANED_INPUT_PAYLOAD, indent=2))

    print("\n[*] Executing Approximate Nearest Neighbor (ANN) Cosine Search (Top-5)...")
    recommendations = engine.find_similar_tracks(CLEANED_INPUT_PAYLOAD, top_k=5)

    # Assertions
    assert len(recommendations) == 5, f"Expected 5 recommendations, got {len(recommendations)}"
    
    similarities = [r["cosine_similarity"] for r in recommendations]
    for i in range(len(similarities) - 1):
        assert similarities[i] >= similarities[i + 1], f"Similarity scores not monotonically decreasing: {similarities}"

    for r in recommendations:
        assert 0.0 <= r["cosine_similarity"] <= 1.0, f"Invalid similarity score: {r['cosine_similarity']}"
        assert len(r["track_id"]) == 32, f"Expected 32-char hex track_id, got: {r['track_id']}"
        for feat in AUDIO_FEATURES:
            assert feat in r["audio_features"], f"Missing audio feature '{feat}' in track {r['track_id']}"

    print("\n" + "=" * 80)
    print("TOP-5 ACOUSTICALLY SIMILAR TRACKS (HNSW / COSINE SIMILARITY):")
    print("=" * 80)
    header = f"{'Rank':<5} | {'Track ID':<34} | {'Cosine Sim':<10} | {'Dance':<7} | {'Energy':<7} | {'Tempo (BPM)':<11} | {'Loudness':<8}"
    print(header)
    print("-" * len(header))

    for r in recommendations:
        feats = r["audio_features"]
        print(
            f"{r['rank']:<5} | "
            f"{r['track_id']:<34} | "
            f"{r['cosine_similarity']:<10.4f} | "
            f"{feats['danceability']:<7.4f} | "
            f"{feats['energy']:<7.4f} | "
            f"{feats['tempo']:<11.2f} | "
            f"{feats['loudness']:<8.2f}"
        )

    # Prepare export report
    output_report = {
        "status": "SUCCESS",
        "total_unique_tracks_indexed": total_tracks,
        "input_query_payload": CLEANED_INPUT_PAYLOAD,
        "mapped_features_order": AUDIO_FEATURES,
        "hnsw_config": {
            "space": "cosine",
            "dim": engine.dim,
            "M": engine.M,
            "ef_construction": engine.ef_construction,
            "ef_search": engine.ef_search
        },
        "top_5_recommendations": recommendations
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[+] All verification assertions PASSED successfully!")
    print(f"[+] Detailed output exported to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()

