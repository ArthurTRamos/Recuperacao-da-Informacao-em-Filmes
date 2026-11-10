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


def radial_coordinates(projection, query_vector, similarities):
    """Preserva a distância à consulta; usa o PCA apenas para escolher ângulos.

    As similaridades devem ser os cossenos dos embeddings completos normalizados.
    Distâncias entre filmes neste mapa não representam as distâncias originais.
    """
    scores = np.asarray(similarities, dtype=np.float64)
    if (scores.shape != (len(projection.coordinates),)
            or not np.isfinite(scores).all()):
        raise ValueError('Informe um cosseno finito para cada filme do mapa.')
    query = np.asarray(query_vector)
    if query.shape != projection.mean.shape or not np.isfinite(query).all():
        raise ValueError('A consulta deve ter as mesmas dimensões dos filmes.')
    offsets = projection.coordinates - projection.transform(query)
    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    # Pontos coincidentes no PCA não têm direção: distribua-os deterministicamente.
    coincident = np.linalg.norm(offsets, axis=1) <= 1e-12
    angles[coincident] = np.linspace(0, 2 * np.pi, coincident.sum(), endpoint=False)
    radii = np.sqrt(2 * (1 - np.clip(scores, -1, 1)))
    return radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
