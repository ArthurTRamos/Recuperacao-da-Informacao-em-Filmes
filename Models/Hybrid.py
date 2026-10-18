"""Fusão de rankings BM25 e Sentence-BERT com Reciprocal Rank Fusion."""


class Hybrid:
    def __init__(self, bm25, semantic, rrf_k=60):
        if rrf_k < 0:
            raise ValueError("rrf_k deve ser não negativo.")
        self.bm25, self.semantic = bm25, semantic
        self.rrf_k = rrf_k

    def search(self, query, top_k=5, candidate_k=30, query_vector=None):
        return [(movie_id, score) for movie_id, score, _, _ in self.search_with_ranks(
            query, top_k, candidate_k, query_vector)]

    def search_with_ranks(self, query, top_k=5, candidate_k=30, query_vector=None):
        """Retorna a fusão e as posições anteriores de cada documento."""
        if candidate_k < top_k:
            raise ValueError("O tamanho dos rankings deve ser >= top_k.")

        lexical = self.bm25.search(query, candidate_k)
        semantic = self.semantic.search(query, candidate_k, query_vector=query_vector)

        fused_scores = {}
        source_ranks = ({movie_id: rank for rank, (movie_id, _) in enumerate(lexical, 1)},
                        {movie_id: rank for rank, (movie_id, _) in enumerate(semantic, 1)})
        for ranking in (lexical, semantic):
            for rank, (movie_id, _score) in enumerate(ranking, start=1):
                fused_scores[movie_id] = fused_scores.get(movie_id, 0.0) + (
                    1.0 / (self.rrf_k + rank))

        ordered = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [(movie_id, score, source_ranks[0].get(movie_id),
                 source_ranks[1].get(movie_id)) for movie_id, score in ordered]
