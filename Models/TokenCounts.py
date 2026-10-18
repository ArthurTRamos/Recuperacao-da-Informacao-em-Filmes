"""Exporta a contagem de tokens do texto de cada filme usado pelo SBERT."""
import csv
from pathlib import Path

from .Corpus import DEFAULT_CACHE, DEFAULT_DATASET, ROOT, document_text, load_movies
from .SentenceBERT import DEFAULT_MODEL, SentenceBERT

DEFAULT_OUTPUT = ROOT / 'Dataset' / 'semantic_token_counts.csv'
FIELDS = ['id', 'title', 'genres', 'tokens_generated', 'tokens_used',
          'max_sequence_length', 'truncated', 'model_name']


def export_token_counts(movies, output=DEFAULT_OUTPUT, cache_dir=DEFAULT_CACHE,
                        model_name=DEFAULT_MODEL):
    encoder = SentenceBERT(movies, cache_dir, model_name)._encoder(offline=True)
    tokenizer = encoder.tokenizer
    max_length = encoder.max_seq_length
    rows = []

    for movie in movies:
        token_ids = tokenizer.encode(f'passage: {document_text(movie)}',
                                     add_special_tokens=True, truncation=False)
        generated = len(token_ids)
        rows.append({
            'id': movie['id'],
            'title': movie['title'],
            'genres': movie.get('genres', ''),
            'tokens_generated': generated,
            'tokens_used': min(generated, max_length),
            'max_sequence_length': max_length,
            'truncated': generated > max_length,
            'model_name': model_name,
        })

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    return target, rows


def export_dataset_token_counts(output=DEFAULT_OUTPUT, dataset=DEFAULT_DATASET,
                                cache_dir=DEFAULT_CACHE, model_name=DEFAULT_MODEL):
    return export_token_counts(load_movies(dataset), output, cache_dir, model_name)
