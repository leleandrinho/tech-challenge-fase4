"""Guardrails: reconhecer quando NÃO há evidência suficiente para responder.

Duas camadas:
  1. ANTES da geração: decide se vale chamar o LLM (similaridade e quantidade de evidências).
     Se não vale, o sistema recusa sem gerar nada, então não há como alucinar.
  2. DEPOIS da geração: confere se a resposta traz citações válidas ou se o modelo recusou.
"""
import re

from src import config


def filtrar_evidencias(evidencias):
    """Mantém só as avaliações com similaridade mínima, o resto é ruído e não vai ao prompt"""
    return [e for e in evidencias if e["similaridade"] >= config.LIMIAR_EVIDENCIA]


def avaliar_evidencias(recuperacao):
    """Retorna (suficiente, motivo) com base no resultado da recuperação."""
    if not recuperacao["evidencias"]:
        return False, "nenhuma avaliação recuperada (filtros restritivos ou sem termos em comum)"

    melhor = recuperacao["melhor_similaridade"]
    if melhor < config.LIMIAR_MELHOR_SIMILARIDADE:
        return False, (f"melhor similaridade {melhor:.2f} < limiar {config.LIMIAR_MELHOR_SIMILARIDADE:.2f} "
                       "(pergunta provavelmente fora do escopo da base)")

    n_relevantes = len(filtrar_evidencias(recuperacao["evidencias"]))
    if n_relevantes < config.MIN_EVIDENCIAS:
        return False, (f"apenas {n_relevantes} avaliação(ões) com similaridade >= {config.LIMIAR_EVIDENCIA:.2f} "
                       f"(mínimo: {config.MIN_EVIDENCIAS}); evidências insuficientes")

    return True, f"{n_relevantes} evidências relevantes (melhor similaridade {melhor:.2f})"


def extrair_citacoes(resposta, n_evidencias):
    """Lê a linha 'Fontes: [1, 3]' e devolve só os números que existem de fato no contexto."""
    linha = re.search(r"Fontes?:\s*(.+)", resposta, re.IGNORECASE)
    if not linha:
        return []
    numeros = {int(n) for n in re.findall(r"\d+", linha.group(1))}
    return sorted(n for n in numeros if 1 <= n <= n_evidencias)


def modelo_recusou(resposta):
    return "não encontrei essa informação" in resposta.lower()
