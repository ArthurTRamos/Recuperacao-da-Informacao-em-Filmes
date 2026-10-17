"""Pré-processamento lexical em português, sem downloads ao importar."""
import re
import unicodedata

from nltk.stem.snowball import PortugueseStemmer

STOPWORDS = frozenset("""a ao aos aquela aquelas aquele aqueles aquilo as até com
como da das de dela delas dele deles depois do dos e é ela elas ele eles em entre
era eram essa essas esse esses esta estas este estes eu foi foram há isso isto
já lhe lhes mais mas me mesmo meu meus minha minhas muito na nas nem no nos nós
nossa nossas nosso nossos num numa o os ou para pela pelas pelo pelos por qual
quando que quem se sem ser seu seus sua suas também te tem ter tu um uma umas
uns você vocês à às não são está estão sobre onde""".split())
stemmer = PortugueseStemmer()


def lower_case_normalization(words):
    return [word.lower() for word in words]


def tokenization(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).lower())


def remove_stopwords(words):
    return [word for word in words if word not in STOPWORDS]


def stemming(words):
    return [stemmer.stem(word) for word in words]


def preprocess(text):
    return stemming(remove_stopwords(tokenization(text)))
