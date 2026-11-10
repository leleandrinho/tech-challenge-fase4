"""Documentos: cada avaliação vira um documento (texto + metadados)

não havera tratamento em chunking pois avaliações de e-commerce já são textos curtos,
então a unidade de recuperação é a própria avaliação. O identificador é o doc_id (= order_id)
"""
import json


def montar_documentos(df):
    """transformamos de DF para uma lista de dicionários (valores nulos viram None)"""
    return json.loads(df.to_json(orient="records", force_ascii=False))


# Salvando esses documentos
def salvar_documentos(documentos, caminho):
    caminho.parent.mkdir(exist_ok=True)
    caminho.write_text(json.dumps(documentos, ensure_ascii=False), encoding="utf-8")

# Função para ler documentos
def carregar_documentos(caminho):
    return json.loads(caminho.read_text(encoding="utf-8"))
