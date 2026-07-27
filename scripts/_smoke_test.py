import sys
sys.path.append('.')
try:
    from backend.services import preprocessing
    from backend.services import search_service
    print('imports OK')
    res = preprocessing.process_all('data/raw/spotify_tracks','data/raw/last_fm','data/processed')
    print('process_all OK, tracks rows=', len(res['tracks']), 'interactions=', len(res['interactions']))
    svc = search_service.SearchService(processed_dir='data/processed')
    print('SearchService loaded, tracks=', len(svc.tracks))
except Exception as e:
    print('ERROR', e)
    raise
