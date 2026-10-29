"""Armazenamento e busca vetorial

Os vetores ficam numa matriz numpy (uma linha por avaliação), salva em disco.
Como estão normalizados, o produto interno é o cosseno. A busca é exata (mesmo resultado
de um FAISS IndexFlatIP) e aceita uma máscara para filtrar por metadados.
"""
import numpy as np

from src import config
from src.documents import carregar_documentos, salvar_documentos


def salvar_indice(documentos, vetores):
    config.PASTA_INDICE.mkdir(exist_ok=True)
    salvar_documentos(documentos, config.ARQ_DOCUMENTOS)
    np.save(config.ARQ_EMBEDDINGS, vetores)


def carregar_indice():
    if not config.ARQ_EMBEDDINGS.exists():
        raise FileNotFoundError("Índice não encontrado. Rode o python: scripts/build_index.py")
    documentos = carregar_documentos(config.ARQ_DOCUMENTOS)
    vetores = np.load(config.ARQ_EMBEDDINGS)
    return documentos, vetores


def buscar_vetorial(vetor_consulta, vetores, k, mascara):
    """Top-k por similaridade de cosseno, só entre os documentos permitidos pela máscara.

    Retorna uma lista de (posição, similaridade), da mais para a menos parecida.
    """
    similaridades = vetores @ vetor_consulta
    similaridades = np.where(mascara, similaridades, -np.inf)   # documentos filtrados ficam de fora
    melhores = np.argsort(-similaridades)[:k]
    return [(int(i), float(similaridades[i])) for i in melhores if similaridades[i] > -np.inf]
