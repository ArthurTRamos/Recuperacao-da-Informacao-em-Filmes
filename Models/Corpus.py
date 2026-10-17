"""Leitura e identificação do corpus usado por todos os modelos."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / 'Dataset' / 'movies_ptbr.csv'
DEFAULT_CACHE = Path(__file__).resolve().parent / 'artefatos'


def load_movies(path=DEFAULT_DATASET):
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        if not {'id', 'title', 'overview'}.issubset(reader.fieldnames or []):
            raise ValueError("O CSV deve conter id, title e overview.")
        movies, ids, documents = [], set(), set()
        for line, row in enumerate(reader, 2):
            try:
                movie_id = int(row['id'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ID inválido na linha {line}.") from exc
            if movie_id in ids:
                raise ValueError(f"ID duplicado na linha {line}: {movie_id}")
            if not row['title'] or not row['overview']:
                raise ValueError(f"Título ou sinopse ausente na linha {line}.")
            ids.add(movie_id)
            document = (row['title'].strip().casefold(), row['overview'].strip().casefold(),
                        (row.get('genres') or '').strip().casefold())
            if document in documents:
                continue
            documents.add(document)
            movies.append({**row, 'id': movie_id})
    if not movies:
        raise ValueError("O dataset está vazio.")
    return movies


def document_text(movie):
    genres = movie.get('genres') or ''
    if isinstance(genres, (list, tuple)):
        genres = ', '.join(str(genre) for genre in genres if genre)
    else:
        genres = str(genres).replace('|', ', ').strip()
    genre_context = f"Gêneros: {genres}. " if genres else ''
    return f"{genre_context}{movie['title']}. {movie['overview']}"


def corpus_fingerprint(movies):
    data = [(movie['id'], document_text(movie)) for movie in movies]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()
