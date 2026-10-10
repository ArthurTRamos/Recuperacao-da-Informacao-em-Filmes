"""Índice termo -> documento -> frequência."""
from collections import Counter

from .Preprocess import preprocess


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.document_length = {}
        self.n_documents = 0
        self.avgdl = 0.0

    @property
    def n_movies(self):
        return self.n_documents

    def build(self, movies):
        self.index.clear()
        self.document_length.clear()
        for movie in movies:
            if isinstance(movie, dict):
                movie_id, title, overview = movie['id'], movie['title'], movie['overview']
            else:
                movie_id, title, overview = movie[:3]
            if movie_id in self.document_length:
                raise ValueError(f"ID de filme duplicado: {movie_id}")
            words = preprocess(f"{title} {overview}")
            self.document_length[movie_id] = len(words)
            for word, frequency in Counter(words).items():
                self.index.setdefault(word, {})[movie_id] = frequency
        self.n_documents = len(self.document_length)
        self.avgdl = (sum(self.document_length.values()) / self.n_documents
                      if self.n_documents else 0.0)
        return self

    def get_term_frequency(self, term):
        return sum(self.index.get(term, {}).values())

    def get_k_most_frequent(self, k):
        return sorted(((word, sum(postings.values()))
                       for word, postings in self.index.items()),
                      key=lambda item: (-item[1], item[0]))[:k]
