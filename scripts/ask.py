"""CLI do pipeline RAG: mostra cada etapa

Uso:
    python scripts/ask.py                                  # modo interativo
    python scripts/ask.py "Quais problemas de entrega?"    # uma pergunta só
    python scripts/ask.py --estado SP --categoria beleza_saude "O que elogiam?"
    python scripts/ask.py --so-recuperacao "..."           # sem chamar o LLM (rápido)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.context import rotulo_evidencia
from src.pipeline import carregar_recursos, responder

LINHA = "-" * 78


def resumo(texto, n=95):
    texto = texto.replace("\n", " ")
    return texto if len(texto) <= n else texto[:n].rstrip() + "..."


def mostrar(r):
    rec = r["recuperacao"]
    print("\n" + "=" * 78)
    print(f'PERGUNTA: "{r["pergunta"]}"')

    print("\n[1] PROCESSAMENTO DA PERGUNTA")
    print(f'    consulta: "{r["consulta"]}"')
    print(f"    filtros de metadados: {r['filtros'] or 'nenhum'}")

    print(f"\n[2] RECUPERAÇÃO — {rec['modo']} (BM25 + vetorial + RRF)")
    print(f"    documentos elegíveis após filtro: {rec['n_docs_filtrados']:,}   ({r['t_recuperacao'] * 1000:.0f} ms)")
    for i, e in enumerate(rec["evidencias"], 1):
        print(f"    [{i}] cos={e['similaridade']:.2f}  vet#{e['rank_vetorial'] or '-'} "
              f"bm25#{e['rank_bm25'] or '-'}  ({rotulo_evidencia(e)})")
        print(f'        "{resumo(e["texto"])}"')

    recusou_antes = r["recusada"] and not r["evidencias_usadas"]
    print(f"\n[3] GUARDRAIL — {'RECUSA' if recusou_antes else 'ok'}: {r['motivo_guardrail']}")

    if r["contexto"]:
        print(f"\n[4] CONTEXTO — {len(r['evidencias_usadas'])} avaliações enviadas ao LLM "
              f"({len(r['contexto']):,} caracteres)")

    print("\n" + LINHA)
    tempo = f" ({r['t_geracao']:.1f} s)" if "t_geracao" in r else ""
    print(f"[5] RESPOSTA{tempo}:")
    print(r["resposta"] or "(geração desligada: --so-recuperacao)")
    if r["recusada"] and r["evidencias_usadas"]:
        print(f"\n  RECUSA PELO MODELO: {r['motivo']}")
    if r["aviso"]:
        print(f"\n  ATENÇÃO: {r['aviso']}")

    if r["citacoes"]:
        print("\n" + LINHA)
        print("[6] EVIDÊNCIAS CITADAS NA RESPOSTA:")
        for n in r["citacoes"]:
            e = r["evidencias_usadas"][n - 1]
            print(f"  [{n}] ({rotulo_evidencia(e)})")
            print(f'      "{e["texto"]}"')


parser = argparse.ArgumentParser(description="Voice of Customer Intelligence com RAG (Olist)")
parser.add_argument("pergunta", nargs="?", help="se omitida, abre o modo interativo")
parser.add_argument("--k", type=int, default=config.TOP_K, help="avaliações recuperadas")
parser.add_argument("--sem-filtros", action="store_true", help="desliga os filtros automáticos por nota")
parser.add_argument("--estado", help="filtra por estado do cliente, ex.: SP")
parser.add_argument("--categoria", help="filtra por categoria do produto, ex.: beleza_saude")
parser.add_argument("--so-recuperacao", action="store_true", help="não chama o LLM")
args = parser.parse_args()

if args.sem_filtros:
    config.USAR_FILTROS_AUTOMATICOS = False

filtros = {}
if args.estado:
    filtros["estado"] = args.estado
if args.categoria:
    filtros["categoria"] = args.categoria

print("Carregando índice e modelos (só na primeira vez)...")
recursos = carregar_recursos(com_llm=not args.so_recuperacao)
print(f"Pronto: {len(recursos['documentos']):,} avaliações indexadas.")

def perguntar(texto):
    resultado = responder(texto, recursos, k=args.k, filtros=filtros, gerar=not args.so_recuperacao)
    mostrar(resultado)


if args.pergunta:
    perguntar(args.pergunta)
else:
    while True:
        texto = input("\nPergunta (ou 'sair'): ").strip()
        if texto.lower() == "sair":
            break
        if texto:
            perguntar(texto)
