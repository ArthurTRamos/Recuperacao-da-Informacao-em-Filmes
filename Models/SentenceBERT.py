"""Embeddings offline e recuperação por cosseno (vetores normalizados)."""
import json
from pathlib import Path

import numpy as np

from .Corpus import DEFAULT_CACHE, corpus_fingerprint, document_text

DEFAULT_MODEL = 'intfloat/multilingual-e5-base'


class SentenceBERT:
    def __init__(self, movies, cache_dir=DEFAULT_CACHE, model_name=DEFAULT_MODEL):
        self.movies = movies
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.encoder = None
        self.embeddings = None
        self.positions = {movie['id']: i for i, movie in enumerate(movies)}

    def _encoder(self, offline=True):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(
                self.model_name, cache_folder=str(self.cache_dir / 'encoder'),
                local_files_only=offline, device='cpu')
        return self.encoder

    def prepare(self, batch_size=32):
        """Única fase que baixa o modelo e codifica os filmes."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Reutiliza primeiro uma cópia local para que atualizar o corpus
            # não dependa de rede quando o modelo já foi baixado.
            encoder = self._encoder(offline=True)
        except (OSError, ValueError):
            self.encoder = None
            encoder = self._encoder(offline=False)
        vectors = encoder.encode(
            [f'passage: {document_text(movie)}' for movie in self.movies],
            batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        metadata = json.dumps({'version': 2, 'model': self.model_name,
                               'corpus': corpus_fingerprint(self.movies),
                               'ids': [movie['id'] for movie in self.movies]})
        target = self.cache_dir / 'embeddings.npz'
        temporary = self.cache_dir / 'embeddings.tmp.npz'
        np.savez_compressed(temporary, embeddings=vectors, metadata=metadata)
        temporary.replace(target)
        self.embeddings = vectors
        return target

    def load(self):
        target = self.cache_dir / 'embeddings.npz'
        if not target.exists():
            raise ValueError("Embeddings ausentes. Execute: python main.py --prepare")
        with np.load(target, allow_pickle=False) as cache:
            metadata = json.loads(str(cache['metadata'].item()))
            vectors = cache['embeddings'].copy()
        expected = {'version': 2, 'model': self.model_name,
                    'corpus': corpus_fingerprint(self.movies),
                    'ids': [movie['id'] for movie in self.movies]}
        if metadata != expected:
            raise ValueError("Dataset ou modelo mudou. Execute novamente --prepare.")
        if (vectors.ndim != 2 or vectors.shape[0] != len(self.movies)
                or vectors.shape[1] == 0 or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
            raise ValueError("Embeddings inválidos. Execute novamente --prepare.")
        self.embeddings = vectors
        return self

    def encode_query(self, query):
        return np.asarray(self._encoder().encode(
            f'query: {query}', normalize_embeddings=True,
            convert_to_numpy=True), dtype=np.float32)

    def search(self, query, top_k=5, candidate_ids=None, query_vector=None):
        if top_k < 1:
            raise ValueError("top_k deve ser positivo.")
        if not query.strip():
            return []
        if self.embeddings is None:
            self.load()
        ids = list(self.positions) if candidate_ids is None else list(dict.fromkeys(candidate_ids))
        if not ids:
            return []
        vector = self.encode_query(query) if query_vector is None else query_vector
        positions = [self.positions[movie_id] for movie_id in ids]
        scores = np.clip(self.embeddings[positions] @ vector, -1, 1)
        return sorted(((movie_id, float(score)) for movie_id, score in zip(ids, scores)),
                      key=lambda item: (-item[1], item[0]))[:top_k]
