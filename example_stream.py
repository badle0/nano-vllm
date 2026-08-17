import os
import sys

from nanovllm import LLM, SamplingParams, StreamingDetokenizer
from transformers import AutoTokenizer

path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
llm = LLM(path, enforce_eager=False, max_model_len=4096)
tok = AutoTokenizer.from_pretrained(path)
detok = StreamingDetokenizer(tok)
prompt = tok.apply_chat_template([{"role": "user", "content": "introduce yourself"}],
                                 tokenize=False, add_generation_prompt=True)
rendered = ""
with llm.stream(
    [prompt],
    SamplingParams(temperature=0.6, max_tokens=256),
) as stream:
    for event in stream:
        rendered = detok.feed(event.seq_id, event.token_id).apply(rendered)
        if event.finished:
            rendered = detok.flush(event.seq_id).apply(rendered)
        preview = rendered.replace("\n", "\\n")
        sys.stdout.write("\r\033[2K" + preview)
        sys.stdout.flush()
print()
