import os
import sys
import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "songs_10k.csv")
DATASET_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"

def fetch_and_prepare_dataset(target_count=10000):
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_csv_path = os.path.join(DATA_DIR, "raw_spotify_dataset.csv")

    if not os.path.exists(raw_csv_path):
        print(f"[*] Downloading Spotify 114k dataset from Hugging Face...")
        response = requests.get(DATASET_URL, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(raw_csv_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        done = int(50 * downloaded / total_size)
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
                        sys.stdout.flush()
        print("\n[+] Download complete.")
    else:
        print(f"[+] Found cached raw dataset at {raw_csv_path}")

    print("[*] Processing and sanitizing dataset...")
    # Load dataset
    df = pd.read_csv(raw_csv_path)

    # Standardize and drop missing essential fields
    df = df.dropna(subset=['track_id', 'track_name', 'artists', 'track_genre', 'popularity'])
    df = df.drop_duplicates(subset=['track_id'])

    # Ensure types are correct
    df['popularity'] = df['popularity'].astype(int)
    for col in ['danceability', 'energy', 'valence', 'tempo', 'acousticness']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Add lowkey domain flag (True if popularity <= 35)
    df['is_lowkey'] = df['popularity'] <= 35

    print(f"[*] Total unique valid tracks available: {len(df)}")
    print(f"[*] Lowkey / underground tracks (popularity <= 35): {df['is_lowkey'].sum()}")

    # To guarantee a healthy mix for testing discovery while prioritizing the lowkey premise:
    # 60% lowkey tracks (popularity <= 35), 40% other tracks across diverse genres
    lowkey_needed = int(target_count * 0.60)
    mainstream_needed = target_count - lowkey_needed

    lowkey_pool = df[df['is_lowkey']]
    other_pool = df[~df['is_lowkey']]

    sampled_lowkey = lowkey_pool.sample(n=min(lowkey_needed, len(lowkey_pool)), random_state=42)
    sampled_other = other_pool.sample(n=min(mainstream_needed, len(other_pool)), random_state=42)

    df_final = pd.concat([sampled_lowkey, sampled_other]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    if len(df_final) < target_count:
        # Fallback if needed
        remaining = target_count - len(df_final)
        leftovers = df[~df['track_id'].isin(df_final['track_id'])].sample(n=remaining, random_state=42)
        df_final = pd.concat([df_final, leftovers]).reset_index(drop=True)

    df_final = df_final.iloc[:target_count]

    # Save to final 10k CSV
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"[+] Successfully generated {OUTPUT_CSV} with {len(df_final)} tracks!")
    print(f"    - Lowkey tracks: {df_final['is_lowkey'].sum()}")
    print(f"    - Unique genres represented: {df_final['track_genre'].nunique()}")
    print(f"    - Popularity range: {df_final['popularity'].min()} - {df_final['popularity'].max()}")
    return OUTPUT_CSV

if __name__ == "__main__":
    fetch_and_prepare_dataset(10000)

