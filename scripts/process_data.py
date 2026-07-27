"""Data processing script: reads raw CSVs from data/raw/spotify_tracks and data/raw/last_fm,
cleans and merges them, engineers features, and writes outputs to data/processed/.

This script is deterministic and safe to run multiple times.
"""
from pathlib import Path
import sys
import logging

# Ensure project root is on sys.path so imports like 'backend.services' work when running scripts
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services import preprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('process_data')

ROOT = Path(__file__).resolve().parents[1]
RAW_SPOTIFY = ROOT / 'data' / 'raw' / 'spotify_tracks'
RAW_LASTFM = ROOT / 'data' / 'raw' / 'last_fm'
OUT = ROOT / 'data' / 'processed'


def main():
    logger.info('Starting data processing')
    result = preprocessing.process_all(str(RAW_SPOTIFY), str(RAW_LASTFM), str(OUT))
    tracks = result.get('tracks')
    interactions = result.get('interactions')
    logger.info(f'Wrote tracks_cleaned.csv with {len(tracks)} rows')
    logger.info(f'Wrote user_interactions.csv with {len(interactions)} rows')


if __name__ == '__main__':
    main()
