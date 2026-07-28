import os
from nanovllm import LLM, SamplingParams, StreamingDetokenizer
from transformers import AutoTokenizer

path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
llm = LLM(path, enforce_eager=False, max_model_len=4096)
tok = AutoTokenizer.from_pretrained(path)
detok = StreamingDetokenizer(tok)
prompt = tok.apply_chat_template([{"role": "user", "content": "introduce yourself"}],
                                 tokenize=False, add_generation_prompt=True)
for ev in llm.stream([prompt], SamplingParams(temperature=0.6, max_tokens=256)):
    print(detok.feed(ev.seq_id, ev.token_id), end="", flush=True)
    if ev.finished: print(detok.flush(ev.seq_id))