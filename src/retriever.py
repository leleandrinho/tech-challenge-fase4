"""Recuperação híbrida: BM25 (lexical) + busca vetorial (semântica) fundidas por RRF.

- tokenização sem acentos e sem stopwords (comentários informais: "nao chegou" x "não chegou");
- só entram no RRF os documentos com score BM25 > 0;
- filtros por metadados (nota, estado, categoria, atraso);
- processamento da pergunta (limpeza + filtros deduzidos de termos como "insatisfeitos").
"""
import re
import unicodedata

import numpy as np

from src import config
from src.embeddings import embedar_consulta
from src.vectorstore import buscar_vetorial

# criando stopword, palavras como "não", "sem" e "muito" ficam porque carregam sentimento.
STOPWORDS = {"a", "o", "as", "os", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
             "um", "uma", "e", "que", "com", "para", "por", "se", "ao", "aos", "foi", "ser", "eu",
             "meu", "minha", "mais", "mas", "como", "ou", "ja", "isso", "esse", "essa"}


def remover_acentos(texto):
    letras = unicodedata.normalize("NFD", texto)
    return "".join(c for c in letras if unicodedata.category(c) != "Mn")


def tokenizar_lexical(texto):
    """deixar minúsculas, sem acentos, só palavras alfanuméricas, sem stopwords."""
    palavras = re.findall(r"[a-z0-9\-]+", remover_acentos(texto.lower()))
    return [p for p in palavras if p not in STOPWORDS and len(p) > 1]


def rrf(rankings, k=config.RRF_K):
    """funde rankings somando 1/(k + posição)."""
    pontos = {}
    for ranking in rankings:
        for posicao, idx in enumerate(ranking):
            pontos[idx] = pontos.get(idx, 0.0) + 1.0 / (k + posicao + 1)
    return sorted(pontos.items(), key=lambda par: -par[1])


# processamento da pergunta
def processar_pergunta(pergunta):
    """Limpa a pergunta e deduz filtros de nota a partir de termos-chave"""
    consulta = " ".join(pergunta.split())
    filtros = {}
    if not config.USAR_FILTROS_AUTOMATICOS:
        return consulta, filtros

    texto = remover_acentos(consulta.lower())
    nota = re.search(r"\bnota\s*(?:igual a\s*|de\s*)?([1-5])\b", texto)

    if nota:
        filtros["nota_min"] = int(nota.group(1))
        filtros["nota_max"] = int(nota.group(1))
    elif re.search(r"insatisfeit|nota[s]? baix|avaliac\w* negativ|reclama", texto):
        filtros["nota_max"] = config.NOTA_MAX_NEGATIVA
    elif re.search(r"elogi|(?<!in)satisfeit|nota[s]? alta|avaliac\w* positiv", texto):
        filtros["nota_min"] = config.NOTA_MIN_POSITIVA
    return consulta, filtros


# filtros e buscas
def passa_filtros(doc, filtros):
    """True se o documento atende a todos os filtros (nota_min, nota_max, estado, categoria, atraso)."""
    if "nota_min" in filtros and doc["nota"] < filtros["nota_min"]:
        return False
    if "nota_max" in filtros and doc["nota"] > filtros["nota_max"]:
        return False
    if "estado" in filtros and doc.get("customer_state") != filtros["estado"].upper():
        return False
    if "categoria" in filtros and filtros["categoria"].lower() not in (doc.get("categorias") or "").lower():
        return False
    if "atraso" in filtros and doc.get("entregue_com_atraso") is not filtros["atraso"]:
        return False
    return True


def ranking_bm25(consulta, bm25, mascara, n):
    """Posições dos n documentos com maior score BM25, ignora os que não têm nenhum termo em comum"""
    scores = np.array(bm25.get_scores(tokenizar_lexical(consulta)))
    scores = np.where(mascara, scores, -np.inf)
    melhores = np.argsort(-scores)[:n]
    return [int(i) for i in melhores if scores[i] > 0]


def recuperar(consulta, recursos, k=config.TOP_K, filtros=None, modo="hibrido"):
    """Retorna as k avaliações mais relevantes

    'hibrido' (BM25 + vetorial + RRF), 'vetorial' ou 'bm25'
    """
    documentos = recursos["documentos"]
    vetores = recursos["vetores"]
    filtros = filtros or {}

    mascara = np.array([passa_filtros(doc, filtros) for doc in documentos])
    vetor = embedar_consulta(consulta, recursos["modelo_emb"])

    # as duas buscas
    resultado_vet = buscar_vetorial(vetor, vetores, config.CANDIDATOS_POR_BUSCA, mascara)
    rank_vet = [posicao for posicao, _ in resultado_vet]
    rank_lex = ranking_bm25(consulta, recursos["bm25"], mascara, config.CANDIDATOS_POR_BUSCA)

    # escolha da estratégia
    if modo == "vetorial":
        escolhidos = rank_vet[:k]
    elif modo == "bm25":
        escolhidos = rank_lex[:k]
    else:
        escolhidos = [posicao for posicao, _ in rrf([rank_lex, rank_vet])[:k]]

    evidencias = []
    for posicao in escolhidos:
        doc = dict(documentos[posicao])
        doc["posicao"] = posicao
        doc["similaridade"] = float(vetores[posicao] @ vetor)
        doc["rank_vetorial"] = rank_vet.index(posicao) + 1 if posicao in rank_vet else None
        doc["rank_bm25"] = rank_lex.index(posicao) + 1 if posicao in rank_lex else None
        evidencias.append(doc)

    return {
        "evidencias": evidencias,
        "melhor_similaridade": resultado_vet[0][1] if resultado_vet else 0.0,
        "n_docs_filtrados": int(mascara.sum()),
        "modo": modo,
    }
