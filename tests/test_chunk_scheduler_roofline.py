from benchmarks.chunked_prefill_tail import scheduler_roofline as roofline
from benchmarks.chunked_prefill_tail.validate_scheduler_roofline import validate


def test_scheduler_correctness_contracts_pass():
    result = roofline._correctness_contracts()
    assert result["gates"]["all_pass"]
    assert result["capacity"]["queue_unchanged"]
    assert result["nonpositive_remaining"]["waiter_unscheduled"]


def test_scheduler_roofline_uses_fixed_release_shapes():
    parser = roofline._parser()
    args = parser.parse_args([])
    assert roofline.BACKLOG_SIZES == (0, 100_000, 500_000)
    assert args.warmup == 2_000
    assert args.samples == 20_000


def test_small_scheduler_measurement_preserves_backlog_and_nonnegative_work():
    row = roofline._measure(backlog_size=17, warmup=3, samples=100)
    assert row["post_state"] == {
        "running": 2,
        "waiting": 17,
        "mid_chunk_seq": False,
        "all_scheduled_counts_nonnegative": True,
    }
    assert len(row["raw_nanoseconds"]) == 100
    assert row["microseconds"]["median"] > 0


def test_validator_requires_exactly_five_artifacts(tmp_path):
    try:
        validate([])
    except ValueError as error:
        assert "exactly five" in str(error)
    else:
        raise AssertionError("validator accepted an empty evidence set")
