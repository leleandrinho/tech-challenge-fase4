"""Pipeline RAG completo: pergunta -> processamento -> recuperação -> guardrail -> contexto -> resposta.

Nenhuma função aqui imprime: elas devolvem um dicionário com todas as etapas, e quem chama
(a CLI, a avaliação) decide como mostrar.
"""
import time

from rank_bm25 import BM25Okapi

from src import config
from src.context import montar_contexto
from src.embeddings import carregar_modelo_embeddings
from src.guardrails import avaliar_evidencias, extrair_citacoes, filtrar_evidencias, modelo_recusou
from src.retriever import processar_pergunta, recuperar, tokenizar_lexical
from src.vectorstore import carregar_indice


def carregar_recursos(com_llm=True):
    """Carrega tudo o que o pipeline precisa, uma única vez. Retorna um dicionário."""
    documentos, vetores = carregar_indice()
    recursos = {
        "documentos": documentos,
        "vetores": vetores,
        "bm25": BM25Okapi([tokenizar_lexical(d["texto"]) for d in documentos]),
        "modelo_emb": carregar_modelo_embeddings(),
        "tok": None,
        "llm": None,
    }
    if com_llm:
        from src.generator import carregar_modelo_geracao
        recursos["tok"], recursos["llm"] = carregar_modelo_geracao()
    return recursos


def responder(pergunta, recursos, k=config.TOP_K, filtros=None, gerar=True, modo="hibrido"):
    """Executa o pipeline. 'filtros' (opcional) soma-se aos filtros deduzidos da pergunta."""
    r = {"pergunta": pergunta, "recusada": False, "motivo": None, "aviso": None,
         "resposta": None, "citacoes": [], "evidencias_usadas": [], "contexto": None}

    # 1. processamento da pergunta
    consulta, filtros_auto = processar_pergunta(pergunta)
    filtros_finais = {**filtros_auto, **(filtros or {})}
    r["consulta"] = consulta
    r["filtros"] = filtros_finais

    # 2. recuperação (embedding da pergunta + BM25 + vetorial + RRF)
    t0 = time.perf_counter()
    recuperacao = recuperar(consulta, recursos, k, filtros_finais, modo)
    r["t_recuperacao"] = time.perf_counter() - t0
    r["recuperacao"] = recuperacao

    # 3. guardrail antes da geração
    suficiente, motivo = avaliar_evidencias(recuperacao)
    r["motivo"] = motivo
    r["motivo_guardrail"] = motivo
    if not suficiente:
        r["recusada"] = True
        r["resposta"] = config.MSG_SEM_EVIDENCIA
        return r

    # 4. contexto
    evidencias = filtrar_evidencias(recuperacao["evidencias"])
    r["evidencias_usadas"] = evidencias
    r["contexto"] = montar_contexto(pergunta, evidencias)
    if not gerar:
        return r

    # 5. geração e checagem da resposta
    from src.generator import gerar_resposta
    resposta, segundos = gerar_resposta(r["contexto"], recursos["tok"], recursos["llm"])
    r["resposta"] = resposta
    r["t_geracao"] = segundos
    r["citacoes"] = extrair_citacoes(resposta, len(evidencias))

    if modelo_recusou(resposta):
        r["recusada"] = True
        r["motivo"] = "o modelo indicou que as avaliações não respondem à pergunta"
    elif not r["citacoes"]:
        r["aviso"] = "a resposta não trouxe citações válidas; trate como não fundamentada"
    return r
