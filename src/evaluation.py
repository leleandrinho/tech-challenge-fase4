"""Avaliação do RAG: qualidade da recuperação, guardrails e citações da resposta.

As perguntas de avaliação ficam aqui mesmo, no começo do arquivo. Cada uma tem um tipo:
  - "valida":          a base deve ter evidências para responder;
  - "fora_de_escopo":  nada a ver com avaliações de e-commerce (o sistema deve recusar);
  - "sem_evidencia":   tema plausível, mas que as avaliações não cobrem (o sistema deve recusar).

Gabarito das perguntas válidas: uma avaliação recuperada conta como RELEVANTE se o texto contém
alguma das palavras_chave (sem diferenciar acentos) e, quando existe "nota_esperada", se a nota
está dentro da faixa. Campos opcionais: "filtros" (ex.: {"categoria": "informatica"}).
"""
import time

import numpy as np

from src import config
from src.pipeline import responder
from src.retriever import processar_pergunta, recuperar, remover_acentos

PERGUNTAS_AVALIACAO = [
    # ---------------------------------------------------------------- válidas
    {"pergunta": "Quais reclamações sobre atraso na entrega aparecem com mais frequência?",
     "tipo": "valida",
     "palavras_chave": ["atraso", "atrasou", "atrasado", "demora", "demorou", "prazo"]},
    {"pergunta": "O que os clientes relatam sobre produtos que chegaram quebrados ou com defeito?",
     "tipo": "valida",
     "palavras_chave": ["quebrado", "quebrou", "defeito", "danificado", "avariado", "amassado"]},
    {"pergunta": "Existem reclamações de pedidos que nunca chegaram?",
     "tipo": "valida",
     "palavras_chave": ["nao recebi", "nao chegou", "nunca chegou", "nao entregue", "nao foi entregue"]},
    {"pergunta": "O que os clientes mais elogiam na experiência de compra?",
     "tipo": "valida",
     "palavras_chave": ["otimo", "excelente", "recomendo", "perfeito", "parabens", "satisfeito"],
     "nota_esperada": (4, 5)},
    {"pergunta": "Quais elogios aparecem sobre a rapidez da entrega?",
     "tipo": "valida",
     "palavras_chave": ["rapid", "antes do prazo", "chegou antes", "ligeiro"]},
    {"pergunta": "O que clientes insatisfeitos dizem sobre o atendimento do vendedor?",
     "tipo": "valida",
     "palavras_chave": ["atendimento", "vendedor", "loja", "resposta", "responder", "contato"],
     "nota_esperada": (1, 2)},
    {"pergunta": "Há reclamações de que o produto é diferente do anunciado?",
     "tipo": "valida",
     "palavras_chave": ["diferente", "anunciado", "descricao", "propaganda", "foto"]},
    {"pergunta": "Quais problemas aparecem nas avaliações com nota 1?",
     "tipo": "valida",
     "palavras_chave": ["nao", "atras", "defeito", "quebr", "problema", "pessimo", "ruim"],
     "nota_esperada": (1, 1)},
    {"pergunta": "O que os clientes relatam sobre o prazo de entrega?",
     "tipo": "valida",
     "palavras_chave": ["prazo", "entrega", "atras", "demor", "rapid"]},
    {"pergunta": "Quais problemas são relatados em produtos de informática?",
     "tipo": "valida",
     "palavras_chave": ["defeito", "nao funciona", "nao liga", "travando", "quebrou", "problema"],
     "filtros": {"categoria": "informatica"}},

    # ------------------------------------------------------- fora de escopo
    {"pergunta": "Qual a capital da França?", "tipo": "fora_de_escopo"},
    {"pergunta": "Como funciona um motor de foguete?", "tipo": "fora_de_escopo"},
    {"pergunta": "Quem ganhou a Copa do Mundo de 2002?", "tipo": "fora_de_escopo"},
    {"pergunta": "Qual é a melhor receita de bolo de chocolate?", "tipo": "fora_de_escopo"},

    # --------------------------------------------------------- sem evidência
    {"pergunta": "Os clientes reclamam do aplicativo de celular da loja?", "tipo": "sem_evidencia"},
    {"pergunta": "O que dizem sobre a política de devolução para compras feitas no exterior?",
     "tipo": "sem_evidencia"},
    {"pergunta": "Há relatos de problemas com o programa de fidelidade e pontos?", "tipo": "sem_evidencia"},
]


def eh_relevante(doc, item):
    """True se a avaliação recuperada atende ao gabarito da pergunta."""
    texto = remover_acentos(doc["texto"].lower())
    if not any(palavra in texto for palavra in item["palavras_chave"]):
        return False
    if "nota_esperada" in item:
        minimo, maximo = item["nota_esperada"]
        if not minimo <= doc["nota"] <= maximo:
            return False
    return True


def avaliar_recuperacao(recursos, modo, nome):
    """Mede precision@k, hit rate e latência de uma estratégia de recuperação (como na aula)."""
    validas = [item for item in PERGUNTAS_AVALIACAO if item["tipo"] == "valida"]
    k = config.TOP_K
    precisoes, acertos, latencias = [], 0, []

    print("-" * 70)
    print(f"ESTRATÉGIA: {nome}")
    for item in validas:
        consulta, filtros_auto = processar_pergunta(item["pergunta"])
        filtros = {**filtros_auto, **item.get("filtros", {})}

        t0 = time.perf_counter()
        resultado = recuperar(consulta, recursos, k, filtros, modo)
        latencias.append(time.perf_counter() - t0)

        evidencias = resultado["evidencias"]
        relevantes = sum(eh_relevante(e, item) for e in evidencias)
        precisao = relevantes / len(evidencias) if evidencias else 0.0
        precisoes.append(precisao)
        if relevantes > 0:
            acertos += 1
        print(f'  [P@{k}={precisao:.2f}] "{item["pergunta"][:60]}..."')

    resumo = {"nome": nome,
              "precision": float(np.mean(precisoes)),
              "hit_rate": acertos / len(validas),
              "latencia_ms": 1000 * float(np.mean(latencias))}
    print(f"  precision@{k} média = {resumo['precision']:.2f}   "
          f"hit rate = {acertos}/{len(validas)}   latência média = {resumo['latencia_ms']:.0f} ms")
    return resumo


def avaliar_guardrails(recursos, gerar=False):
    """Mede a taxa de recusa correta e de falsa recusa. Com gerar=True, mede também as citações."""
    print("-" * 70)
    print("GUARDRAILS" + (" (com geração)" if gerar else " (só a checagem antes da geração)"))

    similaridades = {"valida": [], "nao_valida": []}
    recusas_corretas = falsas_recusas = com_citacao = respondidas = 0
    n_validas = sum(item["tipo"] == "valida" for item in PERGUNTAS_AVALIACAO)
    n_nao_validas = len(PERGUNTAS_AVALIACAO) - n_validas

    for item in PERGUNTAS_AVALIACAO:
        r = responder(item["pergunta"], recursos, filtros=item.get("filtros"), gerar=gerar)
        similaridade = r["recuperacao"]["melhor_similaridade"]
        valida = item["tipo"] == "valida"

        if valida:
            similaridades["valida"].append(similaridade)
            if r["recusada"]:
                falsas_recusas += 1
            elif gerar:
                respondidas += 1
                com_citacao += len(r["citacoes"]) > 0
        else:
            similaridades["nao_valida"].append(similaridade)
            if r["recusada"]:
                recusas_corretas += 1

        correto = r["recusada"] != valida
        situacao = "ok  " if correto else "ERRO"
        acao = "recusou " if r["recusada"] else "respondeu"
        print(f'  [{situacao}] {item["tipo"]:14s} {acao}  sim={similaridade:.2f}  "{item["pergunta"][:48]}..."')

    print(f"\n  recusa correta (fora de escopo / sem evidência): {recusas_corretas}/{n_nao_validas}")
    print(f"  falsa recusa (pergunta válida recusada):         {falsas_recusas}/{n_validas}")
    if gerar and respondidas:
        print(f"  respostas válidas com citação válida:            {com_citacao}/{respondidas}")

    print("\n  CALIBRAÇÃO DO LIMIAR (melhor similaridade de cada pergunta)")
    print(f"    maior similaridade entre perguntas NÃO válidas: {max(similaridades['nao_valida']):.2f}")
    print(f"    menor similaridade entre perguntas válidas:     {min(similaridades['valida']):.2f}")
    print(f"    LIMIAR_MELHOR_SIMILARIDADE atual (config.py):   {config.LIMIAR_MELHOR_SIMILARIDADE:.2f}")
    print("    (o ideal é um limiar entre a maior 'não válida' e a menor 'válida'; se elas se sobrepõem,")
    print("     nenhum limiar é perfeito: escolha o que equilibra recusa correta e falsa recusa)")
    if not gerar:
        print("\n  Obs.: sem geração só a primeira camada é testada; perguntas 'sem_evidencia' com")
        print("  similaridade alta só serão recusadas pelo modelo (rode com --com-geracao).")
