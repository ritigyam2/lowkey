import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import hnswlib
import joblib
from sklearn.preprocessing import MinMaxScaler, normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AudioHNSWEngine")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "Train-Test Splits", "Sequence", "train_SEQ.csv")
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "data", "models")

AUDIO_FEATURES: List[str] = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "loudness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "key",
    "mode"
]

FIELD_MAPPINGS: Dict[str, str] = {
    "bpm": "tempo",
    "loudness_db": "loudness",
    "key_int": "key"
}


class AudioHNSWEngine:
    """
    Approximate Nearest Neighbor (ANN) audio similarity engine using HNSW graphs.
    Extracts 11-dimensional audio features, normalizes them using MinMaxScaler,
    projects into L2-normalized space, and performs fast Cosine Similarity queries.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        dim: int = 11,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50
    ):
        self.data_path = data_path or DEFAULT_DATA_PATH
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.index_path = os.path.join(self.model_dir, "audio_hnsw.bin")
        self.scaler_path = os.path.join(self.model_dir, "audio_scaler.joblib")
        self.track_map_path = os.path.join(self.model_dir, "audio_track_ids.json")
        self.track_features_path = os.path.join(self.model_dir, "audio_track_features.json")

        self.index: Optional[hnswlib.Index] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.track_ids: List[str] = []
        self.track_features: Dict[str, Dict[str, float]] = {}

    def is_built(self) -> bool:
        """Check if all persistent artifacts exist on disk."""
        return (
            os.path.exists(self.index_path)
            and os.path.exists(self.scaler_path)
            and os.path.exists(self.track_map_path)
            and os.path.exists(self.track_features_path)
        )

    def load_and_preprocess_dataset(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Loads train_SEQ.csv, deduplicates by track_id, fills nulls,
        and extracts the 11 audio feature matrix.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Source dataset not found at: {self.data_path}")

        logger.info(f"Loading raw dataset from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        initial_count = len(df)

        # Deduplicate strictly on track_id
        df_unique = df.drop_duplicates(subset=["track_id"]).copy().reset_index(drop=True)
        unique_count = len(df_unique)
        logger.info(f"Deduplicated dataset: {initial_count} total rows -> {unique_count} unique tracks.")

        # Ensure all 11 columns exist and fill missing values with 0.0
        for col in AUDIO_FEATURES:
            if col not in df_unique.columns:
                raise KeyError(f"Required audio feature column '{col}' missing from dataset.")
            df_unique[col] = pd.to_numeric(df_unique[col], errors="coerce").fillna(0.0)

        feature_matrix = df_unique[AUDIO_FEATURES].to_numpy(dtype=np.float32)
        return df_unique, feature_matrix

    def build_index(self, force_rebuild: bool = False) -> int:
        """
        Builds the MinMaxScaler, transforms vectors to L2-normalized cosine space,
        constructs the HNSW graph index, and persists all artifacts.
        """
        if not force_rebuild and self.is_built():
            logger.info("Found existing artifacts on disk. Loading without rebuilding...")
            self.load_index()
            return len(self.track_ids)

        os.makedirs(self.model_dir, exist_ok=True)
        df_unique, feature_matrix = self.load_and_preprocess_dataset()
        num_elements = len(df_unique)

        logger.info("Fitting MinMaxScaler with clipping in range [0, 1]...")
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=True)
        X_scaled = self.scaler.fit_transform(feature_matrix)

        logger.info("Applying L2 normalization for exact Cosine Similarity indexing...")
        X_norm = normalize(X_scaled, norm="l2", axis=1).astype(np.float32)

        logger.info(f"Initializing HNSW Cosine Index (dim={self.dim}, M={self.M}, ef_construction={self.ef_construction})...")
        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.init_index(max_elements=num_elements, ef_construction=self.ef_construction, M=self.M)

        logger.info(f"Building HNSW vector graph across {num_elements} audio vectors...")
        labels = np.arange(num_elements)
        self.index.add_items(X_norm, labels)
        self.index.set_ef(self.ef_search)

        # Save track metadata
        self.track_ids = df_unique["track_id"].astype(str).tolist()
        self.track_features = {
            row["track_id"]: {col: float(row[col]) for col in AUDIO_FEATURES}
            for _, row in df_unique.iterrows()
        }

        # Persist artifacts
        logger.info(f"Saving HNSW binary index to: {self.index_path}")
        self.index.save_index(self.index_path)

        logger.info(f"Saving scaler to: {self.scaler_path}")
        joblib.dump(self.scaler, self.scaler_path)

        logger.info(f"Saving track ID mappings to: {self.track_map_path}")
        with open(self.track_map_path, "w", encoding="utf-8") as f:
            json.dump(self.track_ids, f)

        logger.info(f"Saving track features dictionary to: {self.track_features_path}")
        with open(self.track_features_path, "w", encoding="utf-8") as f:
            json.dump(self.track_features, f)

        logger.info(f"[+] HNSW Audio Index build complete. {num_elements} tracks successfully indexed.")
        return num_elements

    def load_index(self) -> None:
        """Loads persistent HNSW index, scaler, and metadata from disk."""
        if not self.is_built():
            logger.info("Artifacts not found. Triggering automatic index build...")
            self.build_index()
            return

        logger.info("Loading HNSW model artifacts from disk...")
        self.scaler = joblib.load(self.scaler_path)

        with open(self.track_map_path, "r", encoding="utf-8") as f:
            self.track_ids = json.load(f)

        with open(self.track_features_path, "r", encoding="utf-8") as f:
            self.track_features = json.load(f)

        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.load_index(self.index_path, max_elements=len(self.track_ids))
        self.index.set_ef(self.ef_search)
        logger.info(f"[+] Successfully loaded HNSW index ({len(self.track_ids)} tracks).")

    def preprocess_query(self, query_payload: Dict[str, Any]) -> np.ndarray:
        """
        Maps input query keys (e.g. bpm -> tempo, loudness_db -> loudness, key_int -> key),
        fills missing defaults, scales via fitted MinMaxScaler, and applies L2 normalization.
        """
        if self.scaler is None:
            raise RuntimeError("Engine scaler is not initialized. Call load_index() or build_index() first.")

        # Standardize query dict
        normalized_dict = {}
        for k, v in query_payload.items():
            mapped_key = FIELD_MAPPINGS.get(k, k)
            normalized_dict[mapped_key] = v

        # Extract 11 features in exact order
        feature_vector = []
        for feature in AUDIO_FEATURES:
            val = normalized_dict.get(feature)
            if val is None:
                val = 0.0
            feature_vector.append(float(val))

        vector_2d = np.array([feature_vector], dtype=np.float32)

        # Apply MinMaxScaler & L2 norm
        scaled_vec = self.scaler.transform(vector_2d)
        norm_vec = normalize(scaled_vec, norm="l2", axis=1).astype(np.float32)
        return norm_vec

    def find_similar_tracks(self, query_payload: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes ANN search using HNSW and returns the Top-K most acoustically
        similar tracks enriched with metadata and cosine similarity metrics.
        """
        if self.index is None:
            self.load_index()

        query_vector = self.preprocess_query(query_payload)
        labels, distances = self.index.knn_query(query_vector, k=top_k)

        results = []
        for rank, (label_idx, dist) in enumerate(zip(labels[0], distances[0]), start=1):
            track_id = self.track_ids[label_idx]
            # In hnswlib cosine space, distance = 1 - cos(theta)
            cosine_similarity = round(float(np.clip(1.0 - dist, 0.0, 1.0)), 4)
            cosine_distance = round(float(dist), 4)

            track_info = {
                "rank": rank,
                "track_id": track_id,
                "cosine_similarity": cosine_similarity,
                "cosine_distance": cosine_distance,
                "audio_features": self.track_features.get(track_id, {})
            }
            results.append(track_info)

        return results


if __name__ == "__main__":
    engine = AudioHNSWEngine()
    engine.build_index()

