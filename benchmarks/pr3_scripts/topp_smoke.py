import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
tok = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=False, max_model_len=4096)

prompts = [
    tok.apply_chat_template([{"role": "user", "content": "introduce yourself"}],
                            tokenize=False, add_generation_prompt=True),
    tok.apply_chat_template([{"role": "user", "content": "list all prime numbers within 100"}],
                            tokenize=False, add_generation_prompt=True),
]
params = [SamplingParams(temperature=0.6, top_k=50, top_p=0.9, max_tokens=64),
          SamplingParams(temperature=0.6, top_p=0.8,           max_tokens=64)]
for o in llm.generate(prompts, params):
    print(o["text"][:200], "\n---")
print("top_p flowed through LLM -> Sequence -> prepare_sample -> fence -> Sampler (k+p and p-only rows): OK")
