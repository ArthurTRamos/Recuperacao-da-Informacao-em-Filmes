"""BM25: IDF, saturação de frequência e normalização pelo tamanho."""
import math

from .Preprocess import preprocess


class BM25:
    def __init__(self, inverted_index, k1=1.5, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Use k1 > 0 e 0 <= b <= 1.")
        self.inverted_index = inverted_index
        self.k1, self.b = k1, b

    def _term_score(self, word, movie_id):
        index = self.inverted_index
        postings = index.index.get(word, {})
        tf = postings.get(movie_id, 0)
        if not tf or not index.avgdl:
            return 0.0
        idf = math.log1p((index.n_documents - len(postings) + 0.5)
                        / (len(postings) + 0.5))
        dl = index.document_length[movie_id]
        return idf * tf * (self.k1 + 1) / (
            tf + self.k1 * (1 - self.b + self.b * dl / index.avgdl))

    def score(self, query, movie_id):
        return sum(self._term_score(word, movie_id)
                   for word in set(preprocess(query)))

    def search(self, query, top_k=5):
        if top_k < 1:
            raise ValueError("top_k deve ser positivo.")
        scores = {}
        for word in set(preprocess(query)):
            for movie_id in self.inverted_index.index.get(word, {}):
                scores[movie_id] = scores.get(movie_id, 0.0) + self._term_score(word, movie_id)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
