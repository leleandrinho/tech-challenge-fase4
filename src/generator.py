"""Geração com grounding: LLM local (Qwen) responde só com base nas avaliações numeradas."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from src import config

hf_logging.set_verbosity_error()  # silencia avisos da biblioteca

SISTEMA = f"""Você é um analista de Voz do Cliente (Voice of Customer) de um e-commerce.
Responda APENAS com base nas avaliações numeradas fornecidas, de forma curta e direta,
destacando os temas e padrões que aparecem nelas. Não invente absolutamente nada.
Você vê apenas uma amostra das avaliações: nunca afirme percentuais, totais e nem "a maioria dos clientes".
Termine SEMPRE com uma linha no formato: Fontes: [n, n]
Se as avaliações não contiverem a resposta, diga exatamente:
"{config.MSG_SEM_EVIDENCIA}" Não invente nada.

Exemplo de resposta:
Os clientes relatam atraso na entrega e produtos que chegaram com defeito.
Fontes: [1, 3, 4]"""


def carregar_modelo_geracao():
    tok = AutoTokenizer.from_pretrained(config.MODELO_GERACAO)
    llm = AutoModelForCausalLM.from_pretrained(
        config.MODELO_GERACAO,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    return tok, llm


def gerar_resposta(mensagem_usuario, tok, llm):
    """Retorna texto da resposta, segundos gastos na geração"""
    msgs = [{"role": "system", "content": SISTEMA}, {"role": "user", "content": mensagem_usuario}]
    entrada = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt").to(llm.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        saida = llm.generate(**entrada, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    n_prompt = entrada["input_ids"].shape[1]
    texto = tok.decode(saida[0][n_prompt:], skip_special_tokens=True).strip()
    return texto, time.perf_counter() - t0
