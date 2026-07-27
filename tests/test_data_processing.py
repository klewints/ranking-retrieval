import os
from pathlib import Path
import pandas as pd
import tempfile
from backend.services import preprocessing


def test_process_creates_outputs(tmp_path):
    # Prepare raw spotify
    raw_spotify = tmp_path / 'raw_spotify'
    raw_lastfm = tmp_path / 'raw_lastfm'
    out_processed = tmp_path / 'processed'
    raw_spotify.mkdir()
    raw_lastfm.mkdir()
    out_processed.mkdir()

    spotify_rows = [
        {'track_name': 'Blank Space', 'artist_name': 'Taylor Swift', 'album_name': '1989', 'popularity': 90, 'duration_ms': 231000, 'genres': 'Pop'},
        {'track_name': 'Love Story', 'artist_name': 'Taylor Swift', 'album_name': 'Fearless', 'popularity': 85, 'duration_ms': 235000, 'genres': 'Country;Pop'},
    ]
    lastfm_rows = [
        {'user': 'user1', 'artist': 'Taylor Swift', 'track': 'Blank Space', 'playcount': 3},
        {'user': 'user2', 'artist': 'Unknown Artist', 'track': 'Some Song', 'playcount': 1},
    ]
    spotify_df = pd.DataFrame(spotify_rows)
    lastfm_df = pd.DataFrame(lastfm_rows)
    spotify_df.to_csv(raw_spotify / 'spotify_sample.csv', index=False)
    lastfm_df.to_csv(raw_lastfm / 'lastfm_sample.csv', index=False)

    res = preprocessing.process_all(str(raw_spotify), str(raw_lastfm), str(out_processed))
    tracks = res['tracks']
    interactions = res['interactions']

    # Tracks should contain the two spotify tracks
    assert any('Blank Space' in s for s in tracks['track_display'].astype(str))
    assert any('Taylor Swift' in s for s in tracks['artist_display'].astype(str))
    # interactions should map Blank Space
    assert not interactions.empty
    assert 'user1' in interactions['user'].values


def test_remove_duplicates_and_normalization(tmp_path):
    raw_spotify = tmp_path / 'raw_spotify'
    raw_lastfm = tmp_path / 'raw_lastfm'
    out_processed = tmp_path / 'processed'
    raw_spotify.mkdir()
    raw_lastfm.mkdir()
    out_processed.mkdir()

    # Duplicate rows with different casing and whitespace
    spotify_rows = [
        {'track_name': '  blank SPACE', 'artist_name': 'TAYLOR swift', 'album_name': '1989', 'popularity': 90, 'duration_ms': 231000, 'genres': 'Pop'},
        {'track_name': 'Blank Space', 'artist_name': 'Taylor Swift', 'album_name': '1989', 'popularity': 90, 'duration_ms': 231000, 'genres': 'Pop'},
    ]
    spotify_df = pd.DataFrame(spotify_rows)
    spotify_df.to_csv(raw_spotify / 'spotify_dups.csv', index=False)

    res = preprocessing.process_all(str(raw_spotify), str(raw_lastfm), str(out_processed))
    tracks = res['tracks']
    # duplicates removed
    assert tracks['track_clean'].nunique() == 1
    # normalized display
    assert any('Blank Space' == s for s in tracks['track_display'])
