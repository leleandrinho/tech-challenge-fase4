"""Embeddings: cada texto vira um vetor normalizado"""
from sentence_transformers import SentenceTransformer

from src import config


def carregar_modelo_embeddings():
    return SentenceTransformer(config.MODELO_EMBEDDINGS)


def gerar_embeddings(textos, modelo):
    """Matriz (n_textos x dimensões). Usada uma vez, na indexação"""
    return modelo.encode(textos, normalize_embeddings=True, show_progress_bar=True)


def embedar_consulta(consulta, modelo):
    """Vetor da pergunta, usada a cada consulta"""
    return modelo.encode([consulta], normalize_embeddings=True)[0]
