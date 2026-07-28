from pathlib import Path

import pandas as pd

from backend.services.search_service import SearchService


def test_search_typo_tolerance(tmp_path: Path):
    processed = tmp_path / "processed"
    processed.mkdir()
    tracks = [
        {
            "track_display": "Blank Space",
            "artist_display": "Taylor Swift",
            "album_display": "1989",
            "genres_list": "['pop']",
        },
        {
            "track_display": "Bad Guy",
            "artist_display": "Billie Eilish",
            "album_display": "When We All Fall Asleep",
            "genres_list": "['pop']",
        },
        {
            "track_display": "Roar",
            "artist_display": "Katy Perry",
            "album_display": "Prism",
            "genres_list": "['pop']",
        },
    ]
    df = pd.DataFrame(tracks)
    df.to_csv(processed / "tracks_cleaned.csv", index=False)

    svc = SearchService(processed_dir=str(processed))

    res1 = svc.search("taylr swft")
    assert res1["corrected_query"] == "Taylor Swift"
    assert any(
        item["category"] == "artist" and "Taylor" in item["name"]
        for item in res1["results"]
    )

    res2 = svc.search("bilie elish")
    assert res2["corrected_query"] == "Billie Eilish"
    assert any(
        item["category"] == "artist" and "Billie" in item["name"]
        for item in res2["results"]
    )

    res3 = svc.search("katy pery")
    assert res3["corrected_query"] == "Katy Perry"
    assert any(
        item["category"] == "artist" and "Katy" in item["name"]
        for item in res3["results"]
    )


def test_search_ranks_exact_match_first(tmp_path: Path):
    processed = tmp_path / "processed"
    processed.mkdir()
    tracks = [
        {
            "track_display": "All Too Well",
            "artist_display": "Taylor Swift",
            "album_display": "Red",
            "genres_list": "['pop']",
        },
        {
            "track_display": "Red",
            "artist_display": "Taylor Swift",
            "album_display": "Red",
            "genres_list": "['pop']",
        },
    ]
    df = pd.DataFrame(tracks)
    df.to_csv(processed / "tracks_cleaned.csv", index=False)

    svc = SearchService(processed_dir=str(processed))
    results = svc.search("Red")
    assert results["results"]
    assert results["results"][0]["score"] == 100.0
    assert any(
        result["category"] == "track" and result["name"] == "Red"
        for result in results["results"]
    )


def test_search_includes_genres(tmp_path: Path):
    processed = tmp_path / "processed"
    processed.mkdir()
    tracks = [
        {
            "track_display": "Shape of You",
            "artist_display": "Ed Sheeran",
            "album_display": "Divide",
            "genres_list": "['pop', 'dance']",
        }
    ]
    df = pd.DataFrame(tracks)
    df.to_csv(processed / "tracks_cleaned.csv", index=False)

    svc = SearchService(processed_dir=str(processed))
    results = svc.search("pop")
    assert any(
        item["category"] == "genre" and item["name"].lower() == "pop"
        for item in results["results"]
    )
