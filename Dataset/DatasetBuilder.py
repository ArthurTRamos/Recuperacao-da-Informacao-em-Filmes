"""Coleta filmes populares do TMDB sem expor a credencial no código."""
import argparse
import base64
import csv
import json
import os
import re
from pathlib import Path

import requests

FIELDS = ['id', 'title', 'overview', 'release_date', 'vote_average', 'poster_path']


def api_v3_key_from_jwt(token):
    """Obtém a chave v3 que o TMDB inclui no campo aud de alguns JWTs."""
    try:
        payload = token.split('.')[1]
        payload += '=' * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError, json.JSONDecodeError):
        return None
    audience = claims.get('aud')
    if isinstance(audience, str) and re.fullmatch(r'[a-fA-F0-9]{32}', audience):
        return audience
    return None


def collect_movies(api_key, pages=38):
    movies = {}
    # Alguns JWTs do TMDB carregam a chave v3 no campo aud. Nesse caso, use-a
    # como api_key, tal como no exemplo da API v3; outros JWTs usam Bearer.
    embedded_key = api_v3_key_from_jwt(api_key) if api_key.count('.') == 2 else None
    v3_key = embedded_key or (None if api_key.count('.') == 2 else api_key)
    headers = (None if v3_key else {'Authorization': f'Bearer {api_key}'})
    for page in range(1, pages + 1):
        params = {'language': 'pt-BR', 'page': page}
        if v3_key:
            params['api_key'] = v3_key
        response = requests.get(
            'https://api.themoviedb.org/3/movie/popular',
            params=params, headers=headers,
            timeout=30)
        # Não imprime a URL da resposta, pois ela contém a credencial.
        if not response.ok:
            try:
                detail = response.json().get('status_message', '')
            except ValueError:
                detail = ''
            message = f'TMDB retornou HTTP {response.status_code} na página {page}.'
            if detail:
                message += f' {detail}'
            raise RuntimeError(message)
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
        writer = csv.DictWriter(target, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(movies)
    temporary.replace(args.output)
    print(f'{len(movies)} filmes salvos em {args.output}. Refaça os embeddings com --prepare.')


if __name__ == '__main__':
    main()
