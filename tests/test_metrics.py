# tests/test_metrics.py
import importlib.util, pathlib

_p = pathlib.Path(__file__).resolve().parents[1] / "nanovllm/metrics.py"
_s = importlib.util.spec_from_file_location("metrics", _p)
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
compute_metrics = _m.compute_metrics

class Stub:
    def __init__(self, arrival, sched, first, finish, token_times, np_, nc):
        self.arrival_time, self.first_scheduled_time = arrival, sched
        self.first_token_time, self.finish_time = first, finish
        self.token_times = token_times
        self.num_prompt_tokens, self.num_completion_tokens = np_, nc

def test_exact_values():
    m = compute_metrics(Stub(10.0, 10.5, 11.0, 12.0, [11.0, 11.4, 12.0], 100, 3))
    assert abs(m["ttft"] - 1.0) < 1e-12
    assert abs(m["queue_time"] - 0.5) < 1e-12
    assert abs(m["e2e_latency"] - 2.0) < 1e-12
    assert abs(m["mean_itl"] - 0.5) < 1e-12
    assert abs(m["max_itl"] - 0.6) < 1e-12
    assert len(m["itls"]) == 2
    assert abs(m["itls"][0] - 0.4) < 1e-12 and abs(m["itls"][1] - 0.6) < 1e-12
    assert m["num_prompt_tokens"] == 100 and m["num_completion_tokens"] == 3

def test_single_token_sequence():
    m = compute_metrics(Stub(0.0, 0.1, 0.2, 0.2, [0.2], 5, 1))
    assert m["itls"] == [] and m["mean_itl"] == 0.0 and m["max_itl"] == 0.0
    assert m["ttft"] == m["e2e_latency"] == 0.2

def test_engine_metrics_invariants(llm):
    from nanovllm import SamplingParams
    outs = llm.generate(["The capital of France is", "def fibonacci(n):", "In 1969"],
                        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8))
    for o in outs:
        m = o["metrics"]
        assert m["queue_time"] >= 0 and m["ttft"] >= m["queue_time"]
        assert m["e2e_latency"] >= m["ttft"]
        assert m["num_completion_tokens"] == len(o["token_ids"]) == 8
        assert len(m["itls"]) == 7 and all(g >= 0 for g in m["itls"])
    # per-request distinctness: arrival-relative stamps must differ across requests
    assert len({outs[i]["metrics"]["ttft"] for i in range(3)}) > 1 or \
           len({outs[i]["metrics"]["queue_time"] for i in range(3)}) > 1