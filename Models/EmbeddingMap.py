"""PCA dos filmes e da consulta no mesmo plano, sem alterar o ranking."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmbeddingProjection:
    coordinates: np.ndarray
    mean: np.ndarray
    components: np.ndarray
    retained_variance: float

    def transform(self, vector):
        return (np.asarray(vector) - self.mean) @ self.components.T


def project_embeddings(embeddings):
    """Ajusta PCA apenas nos documentos; a consulta usa a mesma transformação."""
    vectors = np.asarray(embeddings, dtype=np.float64)
    if (vectors.ndim != 2 or not all(vectors.shape)
            or not np.isfinite(vectors).all()):
        raise ValueError('O mapa precisa de uma matriz de embeddings finitos e não vazia.')
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    _, singular_values, directions = np.linalg.svd(centered, full_matrices=False)
    components = np.zeros((2, vectors.shape[1]))
    count = min(2, len(directions))
    components[:count] = directions[:count]
    variance = singular_values ** 2
    retained = float(variance[:2].sum() / variance.sum()) if variance.sum() else 0.0
    return EmbeddingProjection(centered @ components.T, mean, components, retained)
