from __future__ import annotations

import re
from typing import Dict, List, Optional

from rapidfuzz import fuzz, process

from backend.config import Config

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value.strip().lower())


class SearchMatcher:
    def __init__(self, score_threshold: int = Config.SEARCH_SCORE_THRESHOLD, correction_threshold: int = Config.CORRECTION_SCORE_THRESHOLD):
        self.score_threshold = score_threshold
        self.correction_threshold = correction_threshold

    def score(self, query: str, choice: str) -> float:
        query_text = normalize_text(query)
        choice_text = normalize_text(choice)

        if not query_text or not choice_text:
            return 0.0

        if query_text == choice_text:
            return 100.0

        scores = [
            fuzz.WRatio(query_text, choice_text),
            fuzz.partial_ratio(query_text, choice_text),
            fuzz.token_sort_ratio(query_text, choice_text),
            fuzz.token_set_ratio(query_text, choice_text),
        ]

        return float(max(scores))

    def find_matches(self, query: str, choices: List[str], limit: int) -> List[Dict[str, float]]:
        if not query or not choices:
            return []

        normalized_query = normalize_text(query)
        raw_matches = process.extract(
            normalized_query,
            choices,
            scorer=fuzz.WRatio,
            processor=normalize_text,
            limit=max(limit, 50),
        )

        scored: List[Dict[str, float]] = []

        for item, raw_score, _ in raw_matches:
            score = self.score(query, item)
            if score < self.score_threshold:
                continue
            scored.append({"name": item, "score": round(score, 2)})

        scored.sort(key=lambda match: match["score"], reverse=True)
        return scored[:limit]

    def find_best_match(self, query: str, choices: List[str]) -> Optional[Dict[str, float]]:
        if not query or not choices:
            return None

        normalized_query = normalize_text(query)
        best = process.extractOne(
            normalized_query,
            choices,
            scorer=fuzz.WRatio,
            processor=normalize_text,
        )

        if best is None:
            return None

        name, _, _ = best
        score = self.score(query, name)

        if score < self.correction_threshold:
            return None

        return {"name": name, "score": round(score, 2)}
