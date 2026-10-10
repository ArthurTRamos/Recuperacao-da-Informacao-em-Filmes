"""Coleta filmes populares do TMDB sem expor a credencial no código."""
import argparse
import csv
import os
from pathlib import Path

import requests

FIELDS = ['id', 'title', 'overview', 'release_date', 'vote_average', 'poster_path']


def collect_movies(api_key, pages=14):
    movies = {}
    for page in range(1, pages + 1):
        response = requests.get(
            'https://api.themoviedb.org/3/movie/popular',
            params={'api_key': api_key, 'language': 'pt-BR', 'page': page},
            timeout=30)
        # Não imprime a URL da resposta, pois ela contém a credencial.
        if not response.ok:
            raise RuntimeError(f'TMDB retornou HTTP {response.status_code} na página {page}.')
        payload = response.json()
        if not isinstance(payload.get('results'), list):
            raise ValueError(f'Resposta inválida na página {page}.')
        for item in payload['results']:
            if item.get('overview') and item.get('title'):
                poster = item.get('poster_path')
                movies[item['id']] = {
                    'id': item['id'], 'title': item['title'], 'overview': item['overview'],
                    'release_date': item.get('release_date', '')[:4],
                    'vote_average': item.get('vote_average', ''),
                    'poster_path': f'https://image.tmdb.org/t/p/w500{poster}' if poster else '',
                }
    return list(movies.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages', type=int, default=14)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('movies_ptbr.csv'))
    args = parser.parse_args()
    if not 1 <= args.pages <= 500:
        parser.error('Use entre 1 e 500 páginas.')
    api_key = os.environ.get('TMDB_API_KEY')
    if not api_key:
        parser.error('Defina a variável de ambiente TMDB_API_KEY.')
    try:
        movies = collect_movies(api_key, args.pages)
    except requests.RequestException:
        parser.exit(1, 'Falha de conexão com o TMDB. Tente novamente.\n')
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
    if not movies:
        parser.exit(1, 'Nenhum filme retornado; o arquivo existente foi preservado.\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp.csv')
    with temporary.open('w', encoding='utf-8', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(movies)
    temporary.replace(args.output)
    print(f'{len(movies)} filmes salvos em {args.output}. Refaça os embeddings com --prepare.')


if __name__ == '__main__':
    main()
