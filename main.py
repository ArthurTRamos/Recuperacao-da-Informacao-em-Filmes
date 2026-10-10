"""Coordenador da interface Streamlit, preparação offline e consultas CLI."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from Modelos.Corpus import DEFAULT_CACHE, DEFAULT_DATASET, load_movies
from Modelos.Retrieval import Retrieval
from Modelos.SentenceBERT import DEFAULT_MODEL

EXPLANATIONS = {
    'BM25': 'Busca palavras em comum no título e na sinopse. Palavras mais raras pesam mais.',
    'Sentence-BERT': 'Compara o sentido da pesquisa com os filmes, mesmo quando as palavras são diferentes.',
    'BM25 + Sentence-BERT (RRF)': 'Combina as posições nos dois rankings. Filmes presentes em qualquer lista podem entrar no resultado.',
}


def result_text(items, score_label='Similaridade de cosseno'):
    if not items:
        return 'Nenhum filme encontrado. Experimente outra consulta.'
    return '\n\n'.join(
        f"{item['rank']}. {item['title']} ({item.get('release_date', '')})\n"
        f"{score_label}: {item['score']:.4f}\n{item['overview']}" for item in items)


def launch_web(args):
    """Abre a interface Streamlit e repassa sua configuração de corpus/modelo."""
    app_path = Path(__file__).with_name('app.py')
    environment = os.environ.copy()
    environment.update({
        'FILMES_DATASET': str(Path(args.dataset).resolve()),
        'FILMES_CACHE_DIR': str(Path(args.cache_dir).resolve()),
        'FILMES_MODEL': args.model,
    })
    command = [sys.executable, '-m', 'streamlit', 'run',
               '--server.headless=true', str(app_path)]
    return subprocess.call(command, env=environment)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=str, default=str(DEFAULT_DATASET))
    parser.add_argument('--cache-dir', type=str, default=str(DEFAULT_CACHE))
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--prepare', action='store_true', help='Baixa o modelo e pré-calcula embeddings.')
    parser.add_argument('--query', help='Executa a consulta no terminal; sem esta opção, abre o Streamlit.')
    parser.add_argument('--method', choices=['all', 'bm25'], default='all')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--candidates', type=int, default=30)
    parser.add_argument('--json', action='store_true', help='Imprime rankings como JSON na CLI.')
    args = parser.parse_args(argv)
    if args.top_k < 1 or args.candidates < 1:
        parser.error('--top-k e --candidates devem ser positivos.')
    if args.prepare:
        try:
            service = Retrieval(load_movies(args.dataset), args.cache_dir, args.model)
            path = service.semantic.prepare()
            print(f'Embeddings de {len(service.movies)} filmes salvos em {path}')
            return 0
        except Exception as exc:
            print(f'Erro ao preparar embeddings: {exc}', file=sys.stderr)
            return 1
    if args.query is None:
        return launch_web(args)
    try:
        service = Retrieval(load_movies(args.dataset), args.cache_dir, args.model)
        if not args.query.strip():
            raise ValueError('Digite uma consulta não vazia.')
        if args.method == 'bm25':
            rankings = {'BM25': [{**service.movies[movie_id], 'score': score, 'rank': rank}
                                for rank, (movie_id, score) in enumerate(
                                    service.bm25.search(args.query, args.top_k), 1)]}
        else:
            rankings = service.compare(args.query, args.top_k, args.candidates)
        if args.json:
            print(json.dumps(rankings, ensure_ascii=False, indent=2))
        else:
            labels = {'BM25': 'Pontuação BM25', 'Sentence-BERT': 'Similaridade de cosseno',
                      'BM25 + Sentence-BERT (RRF)': 'Pontuação RRF'}
            for name, items in rankings.items():
                print(f'\n{name}\n{EXPLANATIONS[name]}\n{result_text(items, labels[name])}')
            print('\nAs pontuações BM25, cosseno e RRF têm escalas distintas. '
                  'Compare as posições, não os valores entre métodos.')
        return 0
    except Exception as exc:
        print(f'Erro: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
