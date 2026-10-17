import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from main import main
from Models.BM25 import BM25
from Models.Corpus import load_movies
from Models.Hybrid import Hybrid
from Models.InvertedIndex import InvertedIndex
from Models.Preprocess import preprocess
from Models.Retrieval import Retrieval
from Models.SentenceBERT import SentenceBERT

MOVIES = [
    {'id': 11, 'title': 'Espaço', 'overview': 'Viagem espacial'},
    {'id': 22, 'title': 'Amizade', 'overview': 'Amigos em uma jornada'},
    {'id': 33, 'title': 'Espaço distante', 'overview': 'Aventura no espaço'},
]


class ControlledEncoder:
    """Separa filme lexical forte (11) de filme semanticamente forte (33)."""
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(texts)
        if isinstance(texts, str):
            return np.array([0, 1], dtype=np.float32)
        return np.array([[1, 0], [0.6, 0.8], [0, 1]], dtype=np.float32)


class LexicalTests(unittest.TestCase):
    def test_portuguese_and_unicode(self):
        self.assertEqual(preprocess('Os filmes, FILME!'), ['film', 'film'])
        self.assertEqual(preprocess('ação'), preprocess('ac\u0327a\u0303o'))
        self.assertEqual(preprocess('a de e o'), [])

    def test_formula_postings_and_rebuild(self):
        index = InvertedIndex().build([(1, 'gato', 'gato'), (2, 'cachorro', '')])
        bm25 = BM25(index)
        expected = math.log(2) * 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 2 / 1.5))
        self.assertAlmostEqual(bm25.score('gato', 1), expected)
        self.assertEqual(bm25.search('gato'), [(1, expected)])
        self.assertEqual(bm25.search('dinossauro'), [])
        index.build([(3, 'peixe', '')])
        self.assertEqual(index.n_documents, 1)
        self.assertNotIn(1, index.document_length)
        self.assertEqual(bm25.search('gato'), [])

    def test_empty_corpus_and_duplicate_ids(self):
        index = InvertedIndex().build([])
        self.assertEqual(BM25(index).search('aventura'), [])
        index.build([(1, '', '')])
        self.assertEqual(BM25(index).score('aventura', 1), 0)
        with self.assertRaises(ValueError):
            index.build([(1, 'x', ''), (1, 'y', '')])


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = SentenceBERT(MOVIES, self.directory.name)
        self.encoder = ControlledEncoder()
        self.model.encoder = self.encoder
        self.model.prepare()

    def test_cache_round_trip_cosine_and_online_calls(self):
        reloaded = SentenceBERT(MOVIES, self.directory.name).load()
        reloaded.encoder = self.encoder
        result = reloaded.search('ideia')
        self.assertEqual([movie_id for movie_id, _ in result], [33, 22, 11])
        self.assertAlmostEqual(result[1][1], 0.8, places=6)
        self.assertEqual(len(self.encoder.calls), 2)
        self.assertIsInstance(self.encoder.calls[-1], str)

    def test_stale_corpus_order_and_model_rejected(self):
        for movies, name in [
            (list(reversed(MOVIES)), self.model.model_name),
            ([{**MOVIES[0], 'overview': 'Mudou'}, *MOVIES[1:]], self.model.model_name),
            (MOVIES, 'outro-modelo'),
        ]:
            with self.assertRaisesRegex(ValueError, 'mudou'):
                SentenceBERT(movies, self.directory.name, name).load()

    def test_invalid_vectors_rejected(self):
        target = Path(self.directory.name) / 'embeddings.npz'
        with np.load(target, allow_pickle=False) as data:
            metadata = data['metadata'].copy()
        np.savez_compressed(target, metadata=metadata, embeddings=np.zeros((3, 2)))
        with self.assertRaisesRegex(ValueError, 'inválidos'):
            self.model.load()

    def test_hybrid_excludes_semantic_winner_when_not_a_candidate(self):
        # A consulta só corresponde lexicalmente ao filme 11.
        bm25 = BM25(InvertedIndex().build(MOVIES))
        hybrid = Hybrid(bm25, self.model)
        self.assertEqual(self.model.search('viagem')[0][0], 33)
        self.assertEqual(hybrid.search('viagem', top_k=1, candidate_k=2)[0][0], 11)
        self.assertEqual(hybrid.search('xyz'), [])
        # Ao incluir ambos os candidatos, o filme 33 sobe pelo cosseno.
        self.assertEqual(hybrid.search('espaço', top_k=1, candidate_k=2)[0][0], 33)
        with self.assertRaises(ValueError):
            hybrid.search('espaço', top_k=3, candidate_k=1)

    def test_compare_reuses_one_query_vector_and_preserves_metadata(self):
        service = Retrieval(MOVIES, self.directory.name)
        service.semantic.encoder = self.encoder
        rankings = service.compare('espaço', top_k=2, candidate_k=2)
        self.assertEqual(len(self.encoder.calls), 2)  # corpus + uma consulta
        self.assertEqual(rankings['Sentence-BERT'][0]['id'], 33)
        self.assertEqual(rankings['BM25 + Sentence-BERT (RRF)'][0]['id'], 33)
        self.assertEqual(rankings['BM25'][0]['rank'], 1)
        with self.assertRaises(ValueError):
            service.compare(' ')

    def test_missing_cache_actionable(self):
        with self.assertRaisesRegex(ValueError, '--prepare'):
            SentenceBERT(MOVIES, Path(self.directory.name) / 'missing').load()


class ApplicationTests(unittest.TestCase):
    def test_real_dataset_and_cli_without_encoder(self):
        movies = load_movies()
        self.assertGreater(len(movies), 0)
        with patch('Modelos.SentenceBERT.SentenceBERT._encoder', side_effect=AssertionError('encoder loaded')):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(['--query', 'espaço', '--method', 'bm25', '--json'])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(output.getvalue())['BM25'])

    def test_import_dataset_builder_has_no_network_side_effects(self):
        with patch('requests.get', side_effect=AssertionError('network')):
            import Dataset.DatasetBuilder  # noqa: F401

    def test_bad_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'movies.csv'
            path.write_text('id,title,overview\n1,Filme,Sinopse\n1,Outro,Sinopse\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicado'):
                load_movies(path)

    def test_repeated_movies_keep_first_id_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'movies.csv'
            source = 'id,title,overview\n1,Filme,Sinopse\n2,Filme,Sinopse\n'
            path.write_text(source, encoding='utf-8')
            self.assertEqual([movie['id'] for movie in load_movies(path)], [1])
            self.assertEqual(path.read_text(encoding='utf-8'), source)


if __name__ == '__main__':
    unittest.main()
