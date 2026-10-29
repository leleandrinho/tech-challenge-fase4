"""Configurações do pipeline RAG: todos os parâmetros ficam aqui"""

from pathlib import Path

# caminhos
RAIZ = Path(__file__).resolve().parent.parent
ARQ_BASE_RAG = RAIZ / "data" / "processed" / "base_rag.csv"   # gerado pelo notebook preparacao_bases
PASTA_INDICE = RAIZ / "indice"                                # gerado pelo build_index.py
ARQ_DOCUMENTOS = PASTA_INDICE / "documentos.json"
ARQ_EMBEDDINGS = PASTA_INDICE / "embeddings.npy"

# modelos - mesmos utilizados no hands on
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" #
MODELO_GERACAO = "Qwen/Qwen2.5-1.5B-Instruct"

# recuperação
TOP_K = 6                   # avaliações enviadas ao LLM (colocar no readme que por se tratar de perguntas agregadas, estamos colocando mais que na aulas)
CANDIDATOS_POR_BUSCA = 30   # candidatos que cada busca (BM25 e vetorial) entrega para o RRF
RRF_K = 60                  # constante do RRF

# Filtros por nota deduzidos da pergunta (ex: "clientes insatisfeitos" -> nota <= 2)
USAR_FILTROS_AUTOMATICOS = True
NOTA_MAX_NEGATIVA = 2
NOTA_MIN_POSITIVA = 4

# guardrails
# Valores iniciais: calibrar com perguntas válidas e fora do escopo (ver evaluation.py)
LIMIAR_MELHOR_SIMILARIDADE = 0.35   # a melhor avaliação recuperada precisa atingir este cosseno
LIMIAR_EVIDENCIA = 0.25             # só avaliações com cosseno >= isto contam como evidência
MIN_EVIDENCIAS = 2                  # mínimo de evidências para tentar responder

MSG_SEM_EVIDENCIA = "Não encontrei essa informação nas avaliações da base de conhecimento."

# contexto e geração
MAX_CHARS_EVIDENCIA = 400    # trunca avaliações muito longas no prompt
MAX_NEW_TOKENS = 300
