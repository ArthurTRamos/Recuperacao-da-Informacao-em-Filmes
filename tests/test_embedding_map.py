import unittest

import numpy as np

from Models.EmbeddingMap import project_embeddings


class EmbeddingMapTests(unittest.TestCase):
    def test_planar_distances_and_query_use_the_same_projection(self):
        # A plane embedded in four dimensions should lose no information.
        vectors = np.array([[1, 2, 3, 4], [2, 2, 4, 4],
                            [1, 4, 3, 6], [3, 6, 5, 8]], dtype=float)
        projection = project_embeddings(vectors)
        np.testing.assert_allclose(projection.transform(vectors), projection.coordinates)
        original_distances = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=2)
        map_distances = np.linalg.norm(
            projection.coordinates[:, None] - projection.coordinates[None, :], axis=2)
        np.testing.assert_allclose(map_distances, original_distances, atol=1e-10)
        self.assertAlmostEqual(projection.retained_variance, 1.0)
        query = (vectors[0] + vectors[1]) / 2
        np.testing.assert_allclose(projection.transform(query),
                                   projection.coordinates[:2].mean(axis=0), atol=1e-10)

    def test_single_movie_and_identical_embeddings_remain_finite(self):
        for vectors in [np.ones((1, 4)), np.ones((3, 4))]:
            projection = project_embeddings(vectors)
            self.assertEqual(projection.coordinates.shape, (len(vectors), 2))
            self.assertTrue(np.isfinite(projection.coordinates).all())
            self.assertEqual(projection.retained_variance, 0.0)

    def test_invalid_embeddings_are_rejected(self):
        for vectors in [[], [[]], [1, 2], [[np.nan]], [[np.inf]]]:
            with self.assertRaises(ValueError):
                project_embeddings(vectors)


if __name__ == '__main__':
    unittest.main()
