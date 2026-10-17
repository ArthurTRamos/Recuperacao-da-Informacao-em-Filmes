"""Serviço de recuperação compartilhado pela interface Streamlit e pela CLI."""
from .BM25 import BM25
from .Corpus import DEFAULT_CACHE
from .Hybrid import Hybrid
from .InvertedIndex import InvertedIndex
from .SentenceBERT import DEFAULT_MODEL, SentenceBERT


class Retrieval:
    def __init__(self, movies, cache_dir=DEFAULT_CACHE, model_name=DEFAULT_MODEL):
        self.movies = {movie['id']: movie for movie in movies}
        self.bm25 = BM25(InvertedIndex().build(movies))
        self.semantic = SentenceBERT(movies, cache_dir, model_name)
        self.hybrid = Hybrid(self.bm25, self.semantic)

    def compare(self, query, top_k=5, candidate_k=30):
        if not query.strip():
            raise ValueError("Digite uma consulta sobre o filme que procura.")
        if top_k < 1 or candidate_k < top_k:
            raise ValueError("Use top_k positivo e candidatos >= top_k.")
        if self.semantic.embeddings is None:
            self.semantic.load()
        vector = self.semantic.encode_query(query)
        rankings = {
            'BM25': self.bm25.search(query, top_k),
            'Sentence-BERT': self.semantic.search(query, top_k, query_vector=vector),
            'BM25 + Sentence-BERT (RRF)': self.hybrid.search(
                query, top_k, candidate_k, vector),
        }
        results = {}
        for name, items in rankings.items():
            results[name] = []
            for rank, (movie_id, score) in enumerate(items, 1):
                result = {**self.movies[movie_id], 'score': score, 'rank': rank}
                if name == 'BM25':
                    result['matched_terms'] = self.bm25.matching_query_terms(query, movie_id)
                results[name].append(result)
        return results
