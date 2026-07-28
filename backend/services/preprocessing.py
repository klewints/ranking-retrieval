import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
import pickle
import re
import unicodedata

# Deterministic helper functions

def normalize_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    s = str(text)
    # Normalize unicode
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    # Lowercase
    s = s.lower()
    # Replace common conjunctions
    s = s.replace('&', ' and ')
    # Remove punctuation except internal apostrophes
    s = re.sub(r"[^a-z0-9' ]+", ' ', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def title_case_name(name: str) -> str:
    # Title case while keeping some small words lowercase except first
    if not name:
        return ''
    s = name.strip()
    # Use simple title case after normalizing spaces
    parts = s.split()
    parts = [p.capitalize() for p in parts]
    return ' '.join(parts)


def read_csvs_from_dir(directory: str) -> List[pd.DataFrame]:
    p = Path(directory)
    if not p.exists():
        return []
    dfs = []
    for f in sorted(p.glob('*.csv')):
        try:
            df = pd.read_csv(f)
            df['__source_file'] = str(f.name)
            dfs.append(df)
        except Exception:
            # skip unreadable files
            continue
    return dfs


def _find_column(df: pd.DataFrame, candidate_names: list[str]) -> str | None:
    lower_columns = {col.lower(): col for col in df.columns}
    for candidate in candidate_names:
        if candidate in lower_columns:
            return lower_columns[candidate]
    return None


def clean_tracks_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    # Standard columns if present: track_name/title/name, artist_name/artist, album
    df = df.copy()
    rename_map = {}
    # heuristics for title
    title_col = _find_column(df, ['track_name', 'title', 'name', 'song'])
    if title_col:
        rename_map[title_col] = 'track'
    artist_col = _find_column(df, ['artist_name', 'artist', 'artists'])
    if artist_col:
        rename_map[artist_col] = 'artist'
    album_col = _find_column(df, ['album_name', 'album'])
    if album_col:
        rename_map[album_col] = 'album'
    # popularity
    popularity_col = _find_column(df, ['popularity', 'track_popularity'])
    if popularity_col:
        rename_map[popularity_col] = 'popularity'
    # duration
    duration_col = _find_column(df, ['duration_ms', 'duration'])
    if duration_col:
        rename_map[duration_col] = 'duration_ms'
    # genres - may be absent
    genres_col = _find_column(df, ['genres', 'genre', 'track_genres'])
    if genres_col:
        rename_map[genres_col] = 'genres'

    df = df.rename(columns=rename_map)

    # Keep only relevant columns plus others
    # Ensure track and artist exist
    if 'track' not in df.columns or 'artist' not in df.columns:
        # If missing essential columns, return empty dataframe
        return pd.DataFrame(columns=['track', 'artist', 'album', 'popularity', 'duration_ms', 'genres', '__source_file'])

    # Drop duplicates using normalized artist+track
    df['track_norm'] = df['track'].apply(normalize_text)
    df['artist_norm'] = df['artist'].apply(normalize_text)
    df = df.drop_duplicates(subset=['artist_norm', 'track_norm'])

    # Trim whitespace and normalize text fields
    df['track_clean'] = df['track'].apply(lambda x: normalize_text(x))
    df['artist_clean'] = df['artist'].apply(lambda x: normalize_text(x))
    df['track_display'] = df['track_clean'].apply(title_case_name)
    df['artist_display'] = df['artist_clean'].apply(title_case_name)

    # Album
    if 'album' in df.columns:
        df['album_clean'] = df['album'].apply(lambda x: normalize_text(x))
        df['album_display'] = df['album_clean'].apply(title_case_name)
    else:
        df['album_clean'] = ''
        df['album_display'] = ''

    # Genres: normalize into list of labels
    if 'genres' in df.columns:
        def split_genres(val):
            if pd.isna(val):
                return []
            if isinstance(val, list):
                items = val
            else:
                items = re.split(r'[;|,]', str(val))
            items = [normalize_text(i) for i in items if normalize_text(i)]
            return sorted(set(items))
        df['genres_list'] = df['genres'].apply(split_genres)
    else:
        df['genres_list'] = [[] for _ in range(len(df))]

    # Popularity: coerce to numeric, clip
    if 'popularity' in df.columns:
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
        df['popularity'] = df['popularity'].fillna(0).astype(float)
    else:
        df['popularity'] = 0.0

    # Duration: convert ms to seconds if looks like ms
    if 'duration_ms' in df.columns:
        df['duration_ms'] = pd.to_numeric(df['duration_ms'], errors='coerce')
        df['duration_sec'] = df['duration_ms'].fillna(0).astype(float) / 1000.0
    else:
        df['duration_sec'] = 0.0

    # Keep columns
    keep_cols = ['track', 'artist', 'album', 'track_display', 'artist_display', 'album_display', 'track_clean', 'artist_clean', 'album_clean', 'genres_list', 'popularity', 'duration_sec', '__source_file']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = '' if c.endswith('_display') or c.endswith('_clean') or c in ['album', 'track', 'artist'] else 0.0
    return df[keep_cols]


def merge_datasets(spotify_dfs: List[pd.DataFrame], lastfm_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    # Clean each and concatenate
    cleaned = []
    for df in spotify_dfs:
        c = clean_tracks_dataframe(df, source='spotify')
        if not c.empty:
            cleaned.append(c)
    for df in lastfm_dfs:
        c = clean_tracks_dataframe(df, source='lastfm')
        if not c.empty:
            cleaned.append(c)
    if not cleaned:
        return pd.DataFrame(columns=['track', 'artist', 'album', 'track_display', 'artist_display', 'album_display', 'track_clean', 'artist_clean', 'album_clean', 'genres_list', 'popularity', 'duration_sec', '__source_file'])
    merged = pd.concat(cleaned, ignore_index=True)
    # Standardize artist names by grouping normalized forms
    merged = merged.drop_duplicates(subset=['artist_clean', 'track_clean'])
    merged = merged.reset_index(drop=True)
    merged['track_id'] = merged.index.astype(str)
    return merged


def engineer_features(merged: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = merged.copy()
    # Artist frequency
    artist_counts = df['artist_clean'].value_counts().to_dict()
    df['artist_freq'] = df['artist_clean'].map(artist_counts).fillna(0).astype(float)

    # Genre encoding: MultiLabelBinarizer
    mlb = MultiLabelBinarizer(sparse_output=False)
    genres_matrix = mlb.fit_transform(df['genres_list'])
    genres_df = pd.DataFrame(genres_matrix, columns=[f'genre__{g}' for g in mlb.classes_]) if mlb.classes_.size else pd.DataFrame()
    df = pd.concat([df.reset_index(drop=True), genres_df.reset_index(drop=True)], axis=1)

    # Artist encoder
    artist_encoder = LabelEncoder()
    df['artist_encoded'] = artist_encoder.fit_transform(df['artist_clean'].astype(str)) if not df.empty else []

    # Numerical scaling
    scaler = StandardScaler()
    numeric_cols = ['popularity', 'duration_sec', 'artist_freq']
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    artifacts = {
        'genre_encoder': mlb,
        'artist_encoder': artist_encoder,
        'feature_scaler': scaler,
    }
    return df, artifacts


def create_user_interactions(lastfm_dfs: List[pd.DataFrame], merged: pd.DataFrame) -> pd.DataFrame:
    # Expect lastfm_dfs contain user and track/artist info
    records = []
    for df in lastfm_dfs:
        df = df.copy()
        lower_map = {col.lower(): col for col in df.columns}
        if 'user' in lower_map:
            user_col = lower_map['user']
        elif 'user_id' in lower_map:
            user_col = lower_map['user_id']
        elif 'username' in lower_map:
            user_col = lower_map['username']
        else:
            user_col = None

        artist_col = _find_column(df, ['artist', 'artist_name', 'artists'])
        track_col = _find_column(df, ['track', 'track_name', 'name', 'song'])

        if user_col is None or artist_col is None or track_col is None:
            continue

        df_local = df[[user_col, artist_col, track_col]].copy()
        df_local.columns = ['user', 'artist', 'track']
        df_local['artist_clean'] = df_local['artist'].apply(normalize_text)
        df_local['track_clean'] = df_local['track'].apply(normalize_text)
        # playcount if present
        if 'playcount' in lower_map:
            count_col = lower_map['playcount']
            df_local['playcount'] = pd.to_numeric(df[count_col], errors='coerce').fillna(1).astype(int)
        else:
            df_local['playcount'] = 1
        records.append(df_local)
    if not records:
        return pd.DataFrame(columns=['user', 'track_id', 'artist', 'track', 'playcount'])
    interactions = pd.concat(records, ignore_index=True)
    # Map to merged track_id by artist_clean + track_clean
    merged_map = merged.set_index(['artist_clean', 'track_clean'])['track_id'].to_dict()
    interactions['track_id'] = interactions.apply(lambda r: merged_map.get((r['artist_clean'], r['track_clean']), ''), axis=1)
    interactions = interactions[interactions['track_id'] != '']
    if interactions.empty:
        return pd.DataFrame(columns=['user', 'track_id', 'artist', 'track', 'playcount'])
    interactions = interactions[['user', 'track_id', 'artist', 'track', 'playcount']]
    interactions = interactions.groupby(['user', 'track_id', 'artist', 'track'], as_index=False)['playcount'].sum()
    return interactions


def save_artifacts(output_dir: str, df_tracks: pd.DataFrame, interactions: pd.DataFrame, artifacts: Dict[str, Any]):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tracks_path = out / 'tracks_cleaned.csv'
    interactions_path = out / 'user_interactions.csv'
    # Write CSVs using UTF-8 to avoid encoding issues on Windows
    tracks_path.write_text(df_tracks.to_csv(index=False), encoding='utf-8')
    interactions_path.write_text(interactions.to_csv(index=False), encoding='utf-8')

    # save encoders
    with open(out / 'genre_encoder.pkl', 'wb') as f:
        pickle.dump(artifacts['genre_encoder'], f)
    with open(out / 'artist_encoder.pkl', 'wb') as f:
        pickle.dump(artifacts['artist_encoder'], f)
    with open(out / 'feature_scaler.pkl', 'wb') as f:
        pickle.dump(artifacts['feature_scaler'], f)


def process_all(raw_spotify_dir: str, raw_lastfm_dir: str, output_dir: str) -> Dict[str, Any]:
    spotify_dfs = read_csvs_from_dir(raw_spotify_dir)
    lastfm_dfs = read_csvs_from_dir(raw_lastfm_dir)
    merged = merge_datasets(spotify_dfs, lastfm_dfs)
    engineered, artifacts = engineer_features(merged)
    interactions = create_user_interactions(lastfm_dfs, merged)
    save_artifacts(output_dir, engineered, interactions, artifacts)
    return {
        'tracks': engineered,
        'interactions': interactions,
        'artifacts': artifacts,
    }
