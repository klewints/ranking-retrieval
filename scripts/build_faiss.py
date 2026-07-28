"""Build a FAISS index for song embeddings."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import Config
from backend.retrieval.two_tower import TwoTowerEmbeddings


def load_two_tower(path: Path) -> TwoTowerEmbeddings:
    return TwoTowerEmbeddings.load(path)


def build_index(output_dir: Path, source_model_path: Path) -> None:
    two_tower = load_two_tower(source_model_path)
    embeddings = two_tower.item_embeddings.astype('float32')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError('FAISS library is required to build the index.') from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))

    index_path = output_dir / 'faiss_index.bin'
    track_ids_path = output_dir / 'track_ids.pkl'
    faiss.write_index(index, str(index_path))
    with open(track_ids_path, 'wb') as handle:
        pickle.dump(two_tower.index_to_item, handle)

    print(f'FAISS index saved to {index_path}')
    print(f'Track ID mapping saved to {track_ids_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build FAISS index for song embeddings')
    parser.add_argument('--source', default=str(Config.TWO_TOWER_MODEL_PATH), help='Path to Two-Tower model')
    parser.add_argument('--output-dir', default=str(Config.FAISS_INDEX_DIR), help='Directory to save FAISS model')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(Path(args.output_dir), Path(args.source))


if __name__ == '__main__':
    main()
