from backend.services.search_service import SearchService
import pandas as pd
from pathlib import Path


def test_search_typo_tolerance(tmp_path):
    processed = tmp_path / 'processed'
    processed.mkdir()
    tracks = [
        {'track_display': 'Blank Space', 'artist_display': 'Taylor Swift', 'album_display': '1989', 'genres_list': "['pop']"},
        {'track_display': 'Bad Guy', 'artist_display': 'Billie Eilish', 'album_display': 'When We All Fall Asleep', 'genres_list': "['pop']"},
        {'track_display': 'Roar', 'artist_display': 'Katy Perry', 'album_display': 'Prism', 'genres_list': "['pop']"},
    ]
    df = pd.DataFrame(tracks)
    df.to_csv(processed / 'tracks_cleaned.csv', index=False)

    svc = SearchService(processed_dir=str(processed))
    # typo queries
    res1 = svc.search('taylr swft')
    assert len(res1) > 0
    assert any('Taylor' in r['artist'] for r in res1)

    res2 = svc.search('bilie elish')
    assert any('Billie' in r['artist'] for r in res2)

    res3 = svc.search('katy pery')
    assert any('Katy' in r['artist'] for r in res3)
