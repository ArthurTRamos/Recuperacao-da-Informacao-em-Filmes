# Roteiro de apresentação — cerca de 5 minutos

1. **O desafio (30 segundos).** “Você já esqueceu o nome de um filme, mas lembrou
   da história? Hoje vamos experimentar três jeitos de procurar filmes.” Peça que
   as crianças imaginem como descreveriam um filme sem dizer seu nome.

2. **Procurando pela ideia (1 minuto).** Clique em **Semântica · Procurando Nemo**.
   Leia “um pai atravessa o oceano para reencontrar seu filho” e pergunte qual filme
   elas imaginam. “A busca semântica compara ideias representadas por números.
   Ela pode encontrar uma história mesmo quando usamos palavras diferentes.”
   Compare as primeiras posições e abra a sinopse. O resultado depende do catálogo;
   os exemplos não garantem que um método sempre vencerá.

3. **Procurando pelas palavras (1 minuto).** Clique em **Léxica · Harry Potter**.
   “Agora sabemos o título: Harry Potter e a Câmara Secreta. O BM25 procura palavras
   da nossa busca nos filmes.” Mostre os destaques das palavras encontradas.
   Observe que outros métodos também podem acertar o título exato.

4. **Juntando as pistas (1 minuto).** Clique em **Geral · Toy Story**.
   “Podemos misturar assunto, tipo de filme e palavras que lembramos.” Compare os
   três rankings. “O RRF junta as posições das duas listas. Estar bem colocado nas
   duas ajuda um filme a subir.” Mostre as posições anteriores indicadas no híbrido.

5. **Visitando o mapa (1 minuto).** Abra **Mapa de significados**. “Cada ponto é um
   filme; o losango laranja é nossa busca. Os pontos azuis são os resultados
   semânticos.” Passe o mouse e aproxime o mapa. “É como resumir um mundo com muitas
   direções em uma folha: algumas informações se perdem. Por isso, a lista usa os
   números completos, e não só a distância que vemos aqui.”

6. **Uma busca da turma (30 segundos).** Peça uma nova descrição de filme e faça
   a consulta. “Qual pista ajudou mais: as palavras ou a ideia? Procurar de jeitos
   diferentes pode ajudar quando a primeira tentativa não encontra o que queremos.”

Antes da aula, execute `python main.py` e teste os três atalhos. Caso o dataset
ou modelo tenha mudado, gere os embeddings com `python main.py --prepare`.
