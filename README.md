# Cinema em busca

Aplicação Streamlit para comparar buscas de filmes em português com BM25,
Sentence-BERT com multilingual-E5-base e a fusão RRF. Os modelos estão em `Models/`.

## Instalação e uso

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --prepare  # baixa o modelo e calcula embeddings uma vez
python main.py            # abre a aplicação web
```

Para consultar pelo terminal:

```bash
python main.py --query "uma aventura no espaço"
python main.py --query "viagem no tempo" --method bm25
```

O primeiro comando compara as três estratégias; `--method bm25` funciona sem
embeddings. Após alterar o dataset, execute `python main.py --prepare` novamente.
Exporte os tokens usados por filme com `python main.py --token-counts`; o CSV é
salvo em `Dataset/semantic_token_counts.csv`. O arquivo informa tokens gerados,
tokens usados após truncagem e o limite do modelo. Gere os embeddings primeiro;
esse comando usa o modelo já baixado e substitui as contagens anteriores.

## Estratégias

- **BM25:** busca termos em comum e dá mais peso aos raros.
- **Sentence-BERT:** compara a similaridade de cosseno entre a consulta e cada filme.
- **RRF:** funde as posições dos rankings BM25 e Sentence-BERT com `1 / (60 + posição)`.

BM25 e Sentence-BERT usam título, sinopse e gêneros. Para a busca semântica,
os campos são rotulados e o E5 recebe `query:` nas consultas e `passage:` nos
filmes. O coletor salva gêneros em português no campo `genres`; após atualizar
o CSV ou trocar o modelo, execute `python main.py --prepare` para recalcular os embeddings.

As pontuações têm escalas diferentes; compare as posições. No híbrido, `--candidates`
define a profundidade de cada ranking antes da fusão (padrão: 30).

## Atualizar o dataset

Defina `TMDB_API_KEY` com uma chave v3 ou token de leitura do TMDB (necessário criação de conta no site da API do TMDB). O coletor aceita chave v3 diretamente e extrai a chave do campo `aud` de JWTs compatíveis.

```bash
export TMDB_API_KEY="sua_chave_ou_token"  # PowerShell: $env:TMDB_API_KEY="..."
python Dataset/DatasetBuilder.py --pages 38
python main.py --prepare
```

O script atualiza `Dataset/movies_ptbr.csv`; `--pages` define quantas páginas
populares baixar. O CSV deve conter `id`, `title` e `overview`.

## Testes

```bash
python -m unittest discover -s tests -v
```
