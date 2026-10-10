"""Interface web em Streamlit para comparar os três rankings de filmes."""
import os

import streamlit as st

from Modelos.Corpus import DEFAULT_CACHE, DEFAULT_DATASET, load_movies
from Modelos.Retrieval import Retrieval
from Modelos.SentenceBERT import DEFAULT_MODEL
from main import EXPLANATIONS

st.set_page_config(page_title='Batalha de Algoritmos', page_icon='⚔️', layout='wide')

EXAMPLES = [
    ('🧱 Bonecos de plástico com vida', 'bonecos de brinquedo ganham vida'),
    ('🧙‍♂️ Escola de magia e cicatriz', 'um jovem vai para uma escola de magia'),
    ('🚗 Viagem no tempo com carro', 'viajar no tempo usando um carro'),
]
METHODS = [
    ('BM25', '🔴 BM25 · Palavra-chave', 'Pontuação BM25'),
    ('Sentence-BERT', '🔵 Sentence-BERT · Semântico', 'Similaridade'),
    ('BM25 + Sentence-BERT (RRF)', '🟢 Híbrido · RRF', 'Pontuação RRF'),
]


@st.cache_resource(show_spinner='Carregando o catálogo de filmes…')
def get_retrieval(dataset_path, cache_dir, model_name):
    return Retrieval(load_movies(dataset_path), cache_dir, model_name)


def choose_example(value):
    st.session_state['query'] = value
    st.session_state['run_search'] = True


def show_movie(movie, score_label):
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
    for column, (label, value) in zip(example_columns, EXAMPLES):
        column.button(label, use_container_width=True,
                      on_click=choose_example, args=(value,))

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
                                          len(service.movies), 5)
        candidate_k = control_columns[1].slider(
            'Posições consideradas por cada busca antes do RRF', top_k,
            len(service.movies), min(max(30, top_k), len(service.movies)))

    should_search = submitted or st.session_state.pop('run_search', False)
    if should_search:
        query = st.session_state['query'].strip()
        if not query:
            st.warning('Digite uma pesquisa ou escolha um exemplo.')
            st.session_state.pop('rankings', None)
        else:
            try:
                with st.spinner('Comparando os três métodos… A primeira consulta pode demorar.'):
                    st.session_state['rankings'] = service.compare(
                        query, top_k=top_k, candidate_k=candidate_k)
                st.session_state['last_query'] = query
            except Exception as exc:
                st.error(f'Não foi possível executar a busca: {exc}')
                st.session_state.pop('rankings', None)

    rankings = st.session_state.get('rankings')
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
                    show_movie(movie, score_label)


if __name__ == '__main__':
    main()
