"""Fusão de rankings BM25 e Sentence-BERT com Reciprocal Rank Fusion."""


class Hybrid:
    def __init__(self, bm25, semantic, rrf_k=60):
        if rrf_k < 0:
            raise ValueError("rrf_k deve ser não negativo.")
        self.bm25, self.semantic = bm25, semantic
        self.rrf_k = rrf_k

    def search(self, query, top_k=5, candidate_k=30, query_vector=None):
        if candidate_k < top_k:
            raise ValueError("O tamanho dos rankings deve ser >= top_k.")

        lexical = self.bm25.search(query, candidate_k)
        semantic = self.semantic.search(query, candidate_k, query_vector=query_vector)

        fused_scores = {}
        for ranking in (lexical, semantic):
            for rank, (movie_id, _score) in enumerate(ranking, start=1):
                fused_scores[movie_id] = fused_scores.get(movie_id, 0.0) + (
                    1.0 / (self.rrf_k + rank))

        return sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
