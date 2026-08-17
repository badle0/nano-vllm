def compute_metrics(seq, delivery_time: float | None = None) -> dict:
    """Compute durations with explicit engine and caller boundaries.

    Engine durations start after tokenization when the sequence is ready for the
    scheduler. Submission durations start at the public API boundary. Caller E2E
    is present only when a caller-facing delivery timestamp is supplied.
    """
    itls = [b - a for a, b in zip(seq.token_times, seq.token_times[1:])]
    metrics = {
        "engine_queue_time": seq.first_scheduled_time - seq.engine_arrival_time,
        "engine_ttft": seq.first_token_time - seq.engine_arrival_time,
        "engine_e2e": seq.finish_time - seq.engine_arrival_time,
        "engine_mean_itl": sum(itls) / len(itls) if itls else 0.0,
        "engine_max_itl": max(itls) if itls else 0.0,
        "engine_itls": itls,
        "submission_to_engine": seq.engine_arrival_time - seq.submission_time,
        "submission_to_first_token": seq.first_token_time - seq.submission_time,
        "submission_to_engine_finish": seq.finish_time - seq.submission_time,
        "num_prompt_tokens": seq.num_prompt_tokens,
        "num_completion_tokens": seq.num_completion_tokens,
    }
    if delivery_time is not None:
        metrics["engine_finish_to_delivery"] = delivery_time - seq.finish_time
        metrics["caller_e2e"] = delivery_time - seq.submission_time
    return metrics
