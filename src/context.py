"""Construção do contexto: transforma as evidências recuperadas no texto enviado ao LLM"""
from src import config


def rotulo_evidencia(e):
    """Metadados curtos que acompanham cada avaliação"""
    partes = [f"nota {e['nota']}"]
    if e.get("customer_state"):
        partes.append(e["customer_state"])
    if e.get("categoria_principal") not in (None, "", "sem_itens", "sem_categoria"):
        partes.append(e["categoria_principal"])
    partes.append(f"id {e['doc_id'][:8]}")
    return " | ".join(partes)


def montar_contexto(pergunta, evidencias):
    """Avaliações numeradas + pergunta"""
    blocos = []
    for i, e in enumerate(evidencias, 1):
        texto = e["texto"]
        if len(texto) > config.MAX_CHARS_EVIDENCIA:
            texto = texto[:config.MAX_CHARS_EVIDENCIA].rstrip() + "..."
        if (e.get("n_ocorrencias") or 1) > 1:
            texto += f"  (comentário idêntico repetido por {int(e['n_ocorrencias'])} clientes)"
        blocos.append(f"[{i}] ({rotulo_evidencia(e)})\n{texto}")
    return "Avaliações de clientes:\n\n" + "\n\n".join(blocos) + f"\n\nPergunta: {pergunta}"
