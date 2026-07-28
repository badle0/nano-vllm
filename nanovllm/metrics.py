def compute_metrics(seq) -> dict:
    itls = [b - a for a, b in zip(seq.token_times, seq.token_times[1:])]
    return {
        "ttft": seq.first_token_time - seq.arrival_time,
        "queue_time": seq.first_scheduled_time - seq.arrival_time,
        "e2e_latency": seq.finish_time - seq.arrival_time,
        "mean_itl": sum(itls) / len(itls) if itls else 0.0,
        "max_itl": max(itls) if itls else 0.0,
        "itls": itls,
        "num_prompt_tokens": seq.num_prompt_tokens,
        "num_completion_tokens": seq.num_completion_tokens,
    }