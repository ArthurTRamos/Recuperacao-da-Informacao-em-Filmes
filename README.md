# Cinema em busca

Aplicação Python para experimentar três estratégias de recuperação em títulos e
sinopses de filmes em português. `main.py` coordena a interface web Streamlit, a
preparação offline e as consultas pelo terminal. A implementação dos modelos fica
em `Modelos/` e a página em `app.py`.

## Executar

Requer Python 3.10 ou superior. Na raiz do projeto:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --prepare
python main.py
```

O segundo comando abre a página no navegador. Para iniciar o Streamlit diretamente,
use `streamlit run app.py`. No Windows, ative o ambiente com
`.venv\Scripts\activate`. A interface web também aceita `--dataset`, `--cache-dir`
e `--model` quando iniciada com `python main.py`.

A preparação inicial precisa de internet para baixar o modelo multilíngue
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Ela salva o encoder
em `Modelos/artefatos/encoder/` e os embeddings normalizados em
`Modelos/artefatos/embeddings.npz`. As consultas carregam o encoder local com
`local_files_only=True`, sem baixar modelos nem recodificar o corpus.
Execute `--prepare` novamente ao alterar títulos, sinopses, IDs, ordem ou modelo.
O cache é validado por hash do corpus, modelo e IDs para impedir rankings desalinhados.

```bash
python main.py --query "uma aventura no espaço para salvar a humanidade"
python main.py --query "amizade improvável" --top-k 5 --candidates 30 --json
# BM25 funciona sem preparar embeddings ou instalar/carregar Sentence Transformers:
python main.py --query "viagem no espaço" --method bm25
```

`--dataset`, `--cache-dir` e `--model` permitem usar outros arquivos e modelos.
Use os mesmos valores na preparação e nas consultas. `--candidates` define a
quantidade de resultados considerados em cada ranking para a fusão e deve ser
maior ou igual a `--top-k`.

## Como comparar

| Estratégia | Recuperação | Pontuação |
| --- | --- | --- |
| BM25 | Palavras em comum, com índice invertido | IDF + frequência com saturação + tamanho do documento |
| Sentence-BERT | Todos os filmes, pelo sentido da frase | Similaridade de cosseno |
| BM25 + Sentence-BERT (RRF) | Combina os rankings lexicais e semânticos independentes | Reciprocal Rank Fusion (RRF) |

BM25 é um modelo lexical da família de ponderação por frequência e raridade;
sua fórmula difere do TF-IDF clássico. O pré-processamento usa stopwords e
stemming em português com Snowball, sem baixar recursos NLTK. Sentence-BERT
recebe título e sinopse originais, preservando contexto, acentos e pontuação.
Vetores de norma 1 permitem calcular cosseno pelo produto escalar.

Os valores BM25 e cosseno **não têm a mesma escala** nem representam probabilidades.
Todos os rankings ordenam a pontuação em ordem decrescente; empates usam o ID.
BM25 retorna apenas filmes que compartilham algum termo. SBERT ranqueia o corpus
todo. RRF soma `1 / (k + posição)` para cada filme em cada ranking (por padrão,
`k = 60`), sem somar pontuações BM25 e cosseno, cujas escalas são diferentes.
Assim, um filme só lexical ou só semântico ainda pode aparecer na lista fundida;
quem aparece bem posicionado em ambos tende a subir. O híbrido fica vazio quando
as duas listas não têm resultados. `--candidates` controla a profundidade de
ambas as listas antes da fusão.

Para uma atividade com adolescentes, compare um título exato, uma descrição da
história e uma paráfrase com sinônimos. Observe quais filmes mudam de posição e
varie o número de candidatos do híbrido. Não há uma estratégia sempre vencedora;
para medir qualidade, é preciso julgar a relevância dos resultados para as consultas.
O carregamento ignora repetições de título e sinopse, mantendo a primeira ocorrência
e preservando o CSV original (259 registros, 215 filmes distintos). Não há curadoria
ou classificação etária automática.

## Dataset e TMDB

`Dataset/movies_ptbr.csv` deve conter `id`, `title` e `overview`; ano, nota e
URL de pôster são metadados opcionais. O coletor preserva IDs do TMDB, remove
duplicatas e filmes sem sinopse e só executa quando chamado explicitamente.
A credencial é lida do ambiente:

```bash
export TMDB_API_KEY="sua_chave"
python Dataset/DatasetBuilder.py --pages 14
python main.py --prepare
```

## Validação

```bash
python -m unittest discover -s tests -v
```

Os testes usam um encoder controlado para verificar cosseno, persistência,
invalidação do cache, limites da fusão e reutilização do vetor da consulta,
sem internet. Para validar os pesos reais, prepare os embeddings e execute uma
consulta com `--method all` (padrão).

Referências: [Sentence Transformers](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)
e [modelo multilíngue](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
