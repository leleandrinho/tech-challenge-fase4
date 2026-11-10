"""Ingestão: lê a base já tratada pelo notebook preparacao_bases"""

import pandas as pd

from src import config

def carregar_base_rag():
    if not config.ARQ_BASE_RAG.exists():
        raise FileNotFoundError(f"{config.ARQ_BASE_RAG} não encontrado. "
                                "Rode antes o notebook preparacao_bases.ipynb.")

    df = pd.read_csv(config.ARQ_BASE_RAG)

    # realizando limpeza (novamente) das avaliações para que nenhum documento vazio ou repetido chega no índice
    df = df.dropna(subset=["texto", "nota"])
    df = df[df["texto"].str.strip() != ""]
    df = df.drop_duplicates(subset="doc_id")
    df["nota"] = df["nota"].astype(int)

    # no CSV a coluna pode vir como True/False ou texto, vamos ajustar
    if "entregue_com_atraso" in df.columns:
        df["entregue_com_atraso"] = df["entregue_com_atraso"].map(
            {True: True, False: False, "True": True, "False": False})

    return df.reset_index(drop=True)
