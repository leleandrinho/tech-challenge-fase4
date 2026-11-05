"""Indexação (roda uma vez): base_rag.csv -> documentos -> embeddings -> índice salvo em indice/.

Uso:
    python scripts/build_index.py                 # base completa
    python scripts/build_index.py --limite 2000   # amostra pequena, para testar rápido
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.documents import montar_documentos
from src.embeddings import carregar_modelo_embeddings, gerar_embeddings
from src.ingestion import carregar_base_rag
from src.vectorstore import salvar_indice

parser = argparse.ArgumentParser()
parser.add_argument("--limite", type=int, help="usa só as N primeiras avaliações (teste rápido)")
args = parser.parse_args()

print("INGESTÃO — lendo a base tratada")
df = carregar_base_rag()
if args.limite:
    df = df.head(args.limite)
documentos = montar_documentos(df)
print(f"  {len(documentos):,} avaliações (1 avaliação = 1 documento, sem chunking)")

print(f"\nEMBEDDINGS — modelo {config.MODELO_EMBEDDINGS}")
modelo = carregar_modelo_embeddings()
t0 = time.perf_counter()
vetores = gerar_embeddings([d["texto"] for d in documentos], modelo)
print(f"  matriz de embeddings: {vetores.shape} em {time.perf_counter() - t0:.0f} s")
print(f"  -> {vetores.shape[0]:,} documentos, cada um virou um vetor de {vetores.shape[1]} dimensões")

print("\nINDEXAÇÃO — salvando em disco")
salvar_indice(documentos, vetores)
print(f"  {config.ARQ_DOCUMENTOS.relative_to(config.RAIZ)}")
print(f"  {config.ARQ_EMBEDDINGS.relative_to(config.RAIZ)}")
print("\nPronto. Agora use: python scripts/ask.py")
