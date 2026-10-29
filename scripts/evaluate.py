"""Avaliação do pipeline RAG: compara estratégias de recuperação e testa os guardrails.

Uso:
    python scripts/evaluate.py                  # só recuperação e guardrails (rápido, sem LLM)
    python scripts/evaluate.py --com-geracao    # também gera as respostas e confere as citações
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.evaluation import PERGUNTAS_AVALIACAO, avaliar_guardrails, avaliar_recuperacao
from src.pipeline import carregar_recursos

parser = argparse.ArgumentParser()
parser.add_argument("--com-geracao", action="store_true", help="carrega o LLM e avalia também as respostas")
args = parser.parse_args()

print("Carregando índice e modelos...")
recursos = carregar_recursos(com_llm=args.com_geracao)

n_validas = sum(item["tipo"] == "valida" for item in PERGUNTAS_AVALIACAO)
print(f"\nGABARITO: {len(PERGUNTAS_AVALIACAO)} perguntas ({n_validas} válidas, "
      f"{len(PERGUNTAS_AVALIACAO) - n_validas} fora de escopo ou sem evidência)")

# 1. recuperação: compara as três estratégias
resumos = []
for modo, nome in [("vetorial", "VETORIAL pura"),
                   ("bm25", "BM25 pura (lexical)"),
                   ("hibrido", "HÍBRIDA (BM25 + vetorial + RRF)")]:
    resumos.append(avaliar_recuperacao(recursos, modo, nome))

print("\n" + "=" * 70)
print(f"RESUMO DA RECUPERAÇÃO (top-{config.TOP_K})")
print(f"  {'estratégia':34s} {'precision':>10s} {'hit rate':>9s} {'latência':>9s}")
for r in resumos:
    print(f"  {r['nome']:34s} {r['precision']:>10.2f} {r['hit_rate']:>9.0%} {r['latencia_ms']:>7.0f} ms")

# 2. guardrails (e, opcionalmente, citações da resposta)
print()
avaliar_guardrails(recursos, gerar=args.com_geracao)

print("\nCUSTO desta avaliação: R$ 0,00 — embeddings, busca e LLM 100% locais.")
