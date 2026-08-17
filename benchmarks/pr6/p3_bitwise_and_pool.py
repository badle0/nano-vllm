import os, random, torch
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import reset_context

llm = LLM(
    os.path.expanduser("~/huggingface/Qwen3-0.6B"), enforce_eager=False,
    max_model_len=4096, max_num_batched_tokens=512,
)
random.seed(0)
GSP = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)   # greedy: on this branch now
PROMPTS = ["The capital of France is", "def fibonacci(n):"]
greedy = lambda: [o["token_ids"] for o in llm.generate(PROMPTS, GSP, use_tqdm=False)]

base = greedy()                                        # decode graphs, pre-capture

llm.add_request([random.randint(1000, 150000) for _ in range(3968)],
                SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True))
seqs, _ = llm.scheduler.schedule()
ids, pos = llm.model_runner.prepare_prefill(seqs)
model = llm.model_runner.model
with torch.inference_mode():
    eager_out = model(ids, pos).clone()                # doubles as the capture warmup
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=llm.model_runner.graph_pool):   # join the decode pool
        graph_out = model(ids, pos)
    torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
print("P3a bitwise graph==eager:", torch.equal(graph_out, eager_out),
      "| allclose:", torch.allclose(graph_out.float(), eager_out.float(), atol=1e-3))
reset_context(); llm.scheduler.cancel_all()

after = greedy()                                       # decode graphs, post-pooled-capture
print("P3b decode graphs intact:", after == base)
