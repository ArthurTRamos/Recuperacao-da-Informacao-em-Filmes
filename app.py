"""Interface web em Streamlit para comparar os três rankings de filmes."""
import html
import os

import altair as alt
import streamlit as st

from Models.Corpus import DEFAULT_CACHE, DEFAULT_DATASET, load_movies
from Models.EmbeddingMap import project_embeddings
from Models.Retrieval import Retrieval
from Models.SentenceBERT import DEFAULT_MODEL
from main import EXPLANATIONS

st.set_page_config(page_title='Batalha de Algoritmos', page_icon='⚔️', layout='wide')

EXAMPLES = [
    ('💡 Semântica · Procurando Nemo',
     'um pai atravessa o oceano para reencontrar seu filho',
     'Descreva a história sem dizer o título. Observe quem encontra a ideia.'),
    ('🔎 Léxica · Harry Potter',
     'Harry Potter e a Câmara Secreta',
     'Use o título exato. Observe as palavras destacadas pelo BM25.'),
    ('🌍 Geral · Toy Story',
     'brinquedos amigos em uma aventura de animação para toda a família',
     'Misture tema e gêneros. Compare os resultados e as posições no RRF.'),
]
METHODS = [
    ('BM25', '🔴 BM25 · Palavra-chave', 'Pontuação BM25'),
    ('Sentence-BERT', '🔵 Sentence-BERT · Semântico', 'Similaridade'),
    ('BM25 + Sentence-BERT (RRF)', '🟢 Híbrido · RRF', 'Pontuação RRF'),
]


@st.cache_resource(show_spinner='Carregando o catálogo de filmes…')
def get_retrieval(dataset_path, cache_dir, model_name):
    return Retrieval(load_movies(dataset_path), cache_dir, model_name)


@st.cache_data(show_spinner='Desenhando o mapa dos filmes…')
def get_projection(embeddings):
    return project_embeddings(embeddings)


def choose_example(value):
    st.session_state['query'] = value
    st.session_state['run_search'] = True


def show_embedding_map(service, rankings, query, query_vector):
    st.subheader('Um mapa de significados')
    st.write('Cada ponto é um filme. O computador transforma título, gêneros e sinopse '
             'em uma lista de números chamada embedding. Aqui, resumimos esses números '
             'em um mapa para explorar ideias parecidas.')
    try:
        if service.semantic.embeddings is None:
            service.semantic.load()
        projection = get_projection(service.semantic.embeddings)
    except Exception as exc:
        st.info(f'Não foi possível montar o mapa: {exc}')
        return

    hits = rankings.get('Sentence-BERT', []) if rankings else []
    hits_by_id = {movie['id']: movie for movie in hits}
    records = []
    for movie, coordinates in zip(service.semantic.movies, projection.coordinates):
        hit = hits_by_id.get(movie['id'])
        records.append({
            'x': float(coordinates[0]), 'y': float(coordinates[1]),
            'title': movie['title'], 'genres': movie.get('genres', ''),
            'rank': hit['rank'] if hit else None,
            'score': hit['score'] if hit else None,
            'group': 'Resultado semântico' if hit else 'Outros filmes',
        })
    if query_vector is not None:
        coordinates = projection.transform(query_vector)
        records.append({'x': float(coordinates[0]), 'y': float(coordinates[1]),
                        'title': f'Sua busca: {query}', 'genres': '',
                        'rank': None, 'score': None, 'group': 'Sua busca'})
    else:
        st.info('Faça uma busca para colocar sua ideia no mapa e destacar os resultados.')

    nearby_only = st.checkbox('Mostrar só a consulta e os resultados semânticos',
                              disabled=not hits)
    if nearby_only and hits:
        records = [record for record in records if record['group'] != 'Outros filmes']

    base = alt.Chart(alt.Data(values=records)).encode(
        x=alt.X('x:Q', axis=None, scale=alt.Scale(zero=False)),
        y=alt.Y('y:Q', axis=None, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip('title:N', title='Filme / consulta'),
                 alt.Tooltip('genres:N', title='Gêneros'),
                 alt.Tooltip('rank:Q', title='Posição semântica', format='.0f'),
                 alt.Tooltip('score:Q', title='Cosseno original', format='.4f')])
    others = base.transform_filter(alt.datum.group == 'Outros filmes').mark_circle(
        size=45, color='#94a3b8', opacity=0.4)
    selected = base.transform_filter(alt.datum.group != 'Outros filmes').mark_point(
        filled=True, opacity=1, stroke='white', strokeWidth=1).encode(
            color=alt.Color('group:N', title='Legenda',
                            scale=alt.Scale(domain=['Resultado semântico', 'Sua busca'],
                                            range=['#2563eb', '#f97316'])),
            shape=alt.Shape('group:N', title='Legenda',
                            scale=alt.Scale(domain=['Resultado semântico', 'Sua busca'],
                                            range=['circle', 'diamond'])),
            size=alt.condition(alt.datum.group == 'Sua busca', alt.value(300), alt.value(140)))
    labels = base.transform_filter(
        (alt.datum.group == 'Sua busca')
        | ((alt.datum.group == 'Resultado semântico') & (alt.datum.rank <= 3))
    ).mark_text(align='left', dx=10, dy=-12, limit=260).encode(
        text=alt.condition(alt.datum.group == 'Sua busca', alt.value('Sua busca'),
                           alt.Text('title:N')))
    chart = (others + selected + labels).properties(height=520).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.caption('Azul: melhores resultados semânticos · Losango laranja: sua busca · '
               'Cinza: restante do catálogo. Passe o mouse nos pontos; use a roda '
               'para aproximar e arraste para explorar.')
    st.info('Este mapa é uma aproximação: ao resumir tantos números em apenas dois, '
            'algumas distâncias mudam. O ranking é calculado com o embedding completo, '
            'por similaridade de cosseno; a ordem visual pode ser diferente.')
    if hits:
        st.markdown('**Mais parecidos no espaço completo dos embeddings**')
        st.dataframe([{'Posição': movie['rank'], 'Filme': movie['title'],
                       'Cosseno': round(movie['score'], 4)} for movie in hits],
                     hide_index=True, use_container_width=True)
    with st.expander('Como este mapa foi feito?'):
        st.write(f'Cada embedding tem {service.semantic.embeddings.shape[1]} dimensões. '
                 f'Usamos PCA para resumi-las em duas, preservando '
                 f'{projection.retained_variance:.1%} da variação dos filmes. '
                 'A projeção é ajustada nos filmes e aplicada também à consulta; '
                 'os filmes permanecem no mesmo lugar quando a busca muda. '
                 'Os eixos não representam gêneros específicos.')


def show_movie(movie, method, score_label):
    with st.container(border=True):
        poster, details = st.columns([1, 2.4], gap='small')
        with poster:
            if movie.get('poster_path'):
                try:
                    st.image(movie['poster_path'], width=105)
                except Exception:
                    st.caption('🎬')
            else:
                st.caption('🎬')
        with details:
            title = movie['title']
            year = movie.get('release_date', '')
            st.markdown(f"**{movie['rank']}. {title}** ({year})" if year
                        else f"**{movie['rank']}. {title}**")
            st.caption(f"{score_label}: {movie['score']:.4f}")
            if method == 'BM25':
                terms = movie.get('matched_terms', [])
                st.markdown('**🔎 Palavras da busca encontradas**')
                if terms:
                    badges = ''.join(
                        '<span style="display:inline-block;background:#fee2e2;color:#991b1b;'
                        'border:1px solid #fecaca;border-radius:999px;padding:2px 9px;'
                        'margin:2px 4px 2px 0;font-size:0.85rem">'
                        f'{html.escape(term)}</span>' for term in terms)
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.caption('Nenhum termo da consulta correspondeu a este filme.')
            elif method == 'Sentence-BERT':
                # Cosine is in [-1, 1] for normalized vectors. Map it linearly
                # to a display-only 0-100 index, not a calibrated probability.
                index = max(0.0, min(100.0, (movie['score'] + 1.0) * 50.0))
                st.progress(index / 100.0,
                            text=f'Índice cosseno: {index:.1f}% (não é probabilidade)')
            elif method == 'BM25 + Sentence-BERT (RRF)':
                cutoff = movie.get('rank_cutoff', 0)
                bm25_rank = movie.get('bm25_rank')
                semantic_rank = movie.get('sentence_bert_rank')
                bm25_position = f'#{bm25_rank}' if bm25_rank else f'fora do top {cutoff}'
                semantic_position = (f'#{semantic_rank}' if semantic_rank
                                     else f'fora do top {cutoff}')
                st.info(f'Posições antes da fusão — BM25: **{bm25_position}** · '
                        f'Sentence-BERT: **{semantic_position}**', icon='📍')
            genres = movie.get('genres') or ''
            if genres:
                st.caption('🎭 ' + ', '.join(g.strip() for g in genres.split('|') if g.strip()))
            with st.expander('Ver sinopse'):
                st.write(movie['overview'])


def main():
    st.markdown(
        "<h1 style='text-align:center'>⚔️ Batalha de Algoritmos: BM25 vs. "
        "Sentence-BERT vs. RRF</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;font-size:1.12rem'>Veja como motores de busca "
        "clássicos e baseados em IA interpretam sua pesquisa.</p>",
        unsafe_allow_html=True)

    try:
        service = get_retrieval(
            os.environ.get('FILMES_DATASET', str(DEFAULT_DATASET)),
            os.environ.get('FILMES_CACHE_DIR', str(DEFAULT_CACHE)),
            os.environ.get('FILMES_MODEL', DEFAULT_MODEL),
        )
    except Exception as exc:
        st.error(f'Não foi possível carregar o catálogo: {exc}')
        st.stop()

    st.caption(f"Catálogo: {len(service.movies)} filmes. Escolha uma ideia ou escreva "
               'uma descrição com suas próprias palavras.')
    example_columns = st.columns(3)
    for column, (label, value, hint) in zip(example_columns, EXAMPLES):
        column.button(label, use_container_width=True,
                      on_click=choose_example, args=(value,))
        column.caption(hint)

    if 'query' not in st.session_state:
        st.session_state['query'] = ''
    with st.form('movie_search', clear_on_submit=False):
        search_column, button_column = st.columns([5, 1])
        with search_column:
            st.text_input('Sua pesquisa', key='query',
                          placeholder='Ex.: uma aventura para salvar a humanidade',
                          label_visibility='collapsed')
        with button_column:
            submitted = st.form_submit_button('🔎 BUSCAR', use_container_width=True)
        control_columns = st.columns(2)
        top_k = control_columns[0].slider('Filmes exibidos por estratégia', 1,
                                          len(service.movies), min(5, len(service.movies)))
        candidate_k = control_columns[1].slider(
            'Posições consideradas por cada busca antes do RRF', top_k,
            len(service.movies), min(max(30, top_k), len(service.movies)))

    should_search = submitted or st.session_state.pop('run_search', False)
    if should_search:
        query = st.session_state['query'].strip()
        if not query:
            st.warning('Digite uma pesquisa ou escolha um exemplo.')
            st.session_state.pop('rankings', None)
            st.session_state.pop('query_vector', None)
        else:
            try:
                with st.spinner('Comparando os três métodos… A primeira consulta pode demorar.'):
                    if service.semantic.embeddings is None:
                        service.semantic.load()
                    vector = service.semantic.encode_query(query)
                    st.session_state['rankings'] = service.compare(
                        query, top_k=top_k, candidate_k=candidate_k, query_vector=vector)
                st.session_state['query_vector'] = vector
                st.session_state['last_query'] = query
            except Exception as exc:
                st.error(f'Não foi possível executar a busca: {exc}')
                st.session_state.pop('rankings', None)
                st.session_state.pop('query_vector', None)

    rankings = st.session_state.get('rankings')
    ranking_tab, map_tab = st.tabs(['🏆 Comparar buscas', '🗺️ Mapa de significados'])
    with map_tab:
        show_embedding_map(service, rankings, st.session_state.get('last_query', ''),
                           st.session_state.get('query_vector'))
    with ranking_tab:
        show_rankings(rankings)


def show_rankings(rankings):
    if rankings:
        st.divider()
        st.caption(f"Resultados para: **{st.session_state.get('last_query', '')}**")
        st.caption('BM25, cosseno e RRF têm escalas diferentes; compare as posições, '
                   'não os valores entre estratégias.')
        columns = st.columns(3, gap='medium')
        for column, (key, title, score_label) in zip(columns, METHODS):
            with column:
                st.subheader(title)
                st.caption(EXPLANATIONS[key])
                items = rankings.get(key, [])
                if not items:
                    st.info('Nenhum filme encontrado para esta pesquisa.')
                for movie in items:
                    show_movie(movie, key, score_label)


if __name__ == '__main__':
    main()
