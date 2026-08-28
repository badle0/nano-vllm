import copy
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = Path(__file__).with_name("run_speculative_v2_lifecycle.py")
SPEC = importlib.util.spec_from_file_location(
    "run_speculative_v2_lifecycle",
    HARNESS_PATH,
)
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)
V0_RUNNER_PATH = Path(__file__).with_name("run_speculative_v0_golden.py")
V0_SPEC = importlib.util.spec_from_file_location(
    "run_speculative_v0_golden_for_schema_test",
    V0_RUNNER_PATH,
)
V0_RUNNER = importlib.util.module_from_spec(V0_SPEC)
V0_SPEC.loader.exec_module(V0_RUNNER)


def test_require_is_an_explicit_runtime_check():
    try:
        HARNESS.require(False, "sentinel failure")
    except AssertionError as error:
        assert str(error) == "sentinel failure"
    else:
        raise AssertionError("require(False) did not fail")


def test_harness_rejects_optimized_python_before_argument_parsing():
    environment = os.environ.copy()
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(REPO_ROOT),
        inherited_pythonpath,
    )))
    completed = subprocess.run(
        [sys.executable, "-O", str(HARNESS_PATH)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    output = completed.stdout + completed.stderr
    assert "refuses optimized Python" in output
    assert "the following arguments are required" not in output


def test_physical_parameter_bytes_deduplicates_shared_storage():
    class SharedStorageModule(nn.Module):
        def __init__(self):
            super().__init__()
            storage_owner = torch.arange(8, dtype=torch.float32)
            self.full = nn.Parameter(storage_owner)
            self.tied_view = nn.Parameter(storage_owner[:4])
            self.independent = nn.Parameter(
                torch.arange(3, dtype=torch.float16)
            )

    model = SharedStorageModule()

    naive_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    expected_physical_bytes = 8 * 4 + 3 * 2
    assert naive_bytes > expected_physical_bytes
    assert (
        HARNESS.physical_parameter_bytes(model)
        == expected_physical_bytes
    )


def test_expected_cache_shape_uses_explicit_head_dim_lazily():
    hf_config = SimpleNamespace(
        head_dim=80,
        num_hidden_layers=12,
        num_key_value_heads=8,
    )

    assert HARNESS.expected_cache_shape(
        hf_config,
        num_blocks=7,
        block_size=256,
        world_size=2,
    ) == (2, 12, 7, 256, 4, 80)


def _audit(
    *,
    eager,
    joint_block_bytes,
    profiled_graph_allocated_bytes,
    profiled_graph_reserved_bytes,
    profiled_graph_peak_allocated_bytes,
    profiled_graph_peak_reserved_bytes,
    memory_budget_bytes,
    used_before_kv_bytes,
    warmup_transient_bytes,
    target_warmup_transient_bytes,
    draft_warmup_transient_bytes,
    reservation_bytes,
    post_init_budget_headroom_bytes,
    allocated_before_kv_bytes,
    target_kv_bytes,
    draft_kv_bytes,
    final_graph_allocated_bytes,
    post_graph_allocated_bytes=0,
    post_graph_reserved_bytes=0,
):
    graph_margin = (
        0
        if eager
        else max(
            HARNESS.MIN_GRAPH_ALLOCATOR_MARGIN_BYTES,
            joint_block_bytes,
        )
    )
    graph_ownership = max(
        profiled_graph_allocated_bytes,
        profiled_graph_reserved_bytes,
    )
    graph_peak = max(
        graph_ownership,
        profiled_graph_peak_allocated_bytes,
        profiled_graph_peak_reserved_bytes,
    )
    graph_construction_reservation = graph_peak + graph_margin
    runtime_reservation = (
        graph_ownership + warmup_transient_bytes + reservation_bytes
    )
    sizing_overhead = max(
        graph_construction_reservation,
        runtime_reservation,
    )
    allocated_after_kv = (
        allocated_before_kv_bytes + target_kv_bytes + draft_kv_bytes
    )
    reserved_after_kv = allocated_after_kv
    allocated_after_graph = (
        allocated_after_kv + final_graph_allocated_bytes
    )
    reserved_after_graph = (
        reserved_after_kv + final_graph_allocated_bytes
    )
    post_init_allocated_bytes = (
        allocated_after_graph + post_graph_allocated_bytes
    )
    post_init_reserved_bytes = (
        reserved_after_graph + post_graph_reserved_bytes
    )
    return SimpleNamespace(
        workspace_plan=SimpleNamespace(
            reservation_bytes=reservation_bytes,
        ),
        joint_block_bytes=joint_block_bytes,
        profiled_graph_allocated_bytes=profiled_graph_allocated_bytes,
        profiled_graph_reserved_bytes=profiled_graph_reserved_bytes,
        profiled_graph_peak_allocated_bytes=(
            profiled_graph_peak_allocated_bytes
        ),
        profiled_graph_peak_reserved_bytes=(
            profiled_graph_peak_reserved_bytes
        ),
        memory_budget_bytes=memory_budget_bytes,
        used_before_kv_bytes=used_before_kv_bytes,
        warmup_transient_bytes=warmup_transient_bytes,
        target_warmup_transient_bytes=target_warmup_transient_bytes,
        draft_warmup_transient_bytes=draft_warmup_transient_bytes,
        post_init_budget_headroom_bytes=post_init_budget_headroom_bytes,
        allocated_before_kv_bytes=allocated_before_kv_bytes,
        target_kv_bytes=target_kv_bytes,
        draft_kv_bytes=draft_kv_bytes,
        final_graph_allocated_bytes=final_graph_allocated_bytes,
        allocated_after_kv_before_graph_bytes=allocated_after_kv,
        reserved_after_kv_before_graph_bytes=reserved_after_kv,
        allocated_after_graph_before_pretouch_bytes=(
            allocated_after_graph
        ),
        reserved_after_graph_before_pretouch_bytes=reserved_after_graph,
        post_init_allocated_bytes=post_init_allocated_bytes,
        post_init_reserved_bytes=post_init_reserved_bytes,
        graph_allocator_margin_bytes=graph_margin,
        graph_construction_reservation_bytes=(
            graph_construction_reservation
        ),
        runtime_reservation_bytes=runtime_reservation,
        sizing_overhead_bytes=sizing_overhead,
    )


def test_independent_graph_memory_reconciliation_prices_all_components():
    mib = 1024**2
    audit = _audit(
        eager=False,
        joint_block_bytes=80 * mib,
        profiled_graph_allocated_bytes=13 * mib,
        profiled_graph_reserved_bytes=12 * mib,
        profiled_graph_peak_allocated_bytes=30 * mib,
        profiled_graph_peak_reserved_bytes=25 * mib,
        memory_budget_bytes=1000 * mib,
        used_before_kv_bytes=100 * mib,
        warmup_transient_bytes=25 * mib,
        target_warmup_transient_bytes=15 * mib,
        draft_warmup_transient_bytes=20 * mib,
        reservation_bytes=50 * mib,
        post_init_budget_headroom_bytes=400 * mib,
        allocated_before_kv_bytes=100 * mib,
        target_kv_bytes=40 * mib,
        draft_kv_bytes=20 * mib,
        final_graph_allocated_bytes=5 * mib,
        post_graph_allocated_bytes=2 * mib,
        post_graph_reserved_bytes=9 * mib,
    )

    result = HARNESS.independent_memory_reconciliation(
        audit,
        enforce_eager=False,
    )

    assert result["profiled_graph_ownership_bytes"] == 13 * mib
    assert result["profiled_graph_peak_bytes"] == 30 * mib
    assert result["graph_allocator_margin_bytes"] == 80 * mib
    assert result["graph_construction_reservation_bytes"] == 110 * mib
    assert result["runtime_reservation_bytes"] == 88 * mib
    assert result["sizing_overhead_bytes"] == 110 * mib
    assert result["sizing_usable_bytes"] == 790 * mib
    assert result["automatic_num_blocks"] == 9
    assert result["modeled_runtime_headroom_bytes"] == 325 * mib
    assert result["kv_allocated_increment_bytes"] == 60 * mib
    assert result["kv_accounted_bytes"] == 60 * mib
    assert result["final_graph_allocated_increment_bytes"] == 5 * mib
    assert result["final_graph_reserved_increment_bytes"] == 5 * mib
    assert result["post_init_allocated_increment_bytes"] == 2 * mib
    assert result["post_init_reserved_increment_bytes"] == 9 * mib
    assert result["total_allocated_increment_bytes"] == 67 * mib
    assert result["total_accounted_increment_bytes"] == 67 * mib


def test_independent_eager_memory_reconciliation_keeps_activation_peak():
    mib = 1024**2
    audit = _audit(
        eager=True,
        joint_block_bytes=8 * mib,
        profiled_graph_allocated_bytes=0,
        profiled_graph_reserved_bytes=0,
        profiled_graph_peak_allocated_bytes=0,
        profiled_graph_peak_reserved_bytes=0,
        memory_budget_bytes=200 * mib,
        used_before_kv_bytes=40 * mib,
        warmup_transient_bytes=25 * mib,
        target_warmup_transient_bytes=10 * mib,
        draft_warmup_transient_bytes=15 * mib,
        reservation_bytes=20 * mib,
        post_init_budget_headroom_bytes=100 * mib,
        allocated_before_kv_bytes=40 * mib,
        target_kv_bytes=16 * mib,
        draft_kv_bytes=8 * mib,
        final_graph_allocated_bytes=0,
    )

    result = HARNESS.independent_memory_reconciliation(
        audit,
        enforce_eager=True,
    )

    assert result["graph_allocator_margin_bytes"] == 0
    assert result["graph_construction_reservation_bytes"] == 0
    assert result["runtime_reservation_bytes"] == 45 * mib
    assert result["sizing_overhead_bytes"] == 45 * mib
    assert result["sizing_usable_bytes"] == 115 * mib
    assert result["automatic_num_blocks"] == 14
    assert result["modeled_runtime_headroom_bytes"] == 55 * mib


def test_source_tree_fingerprint_includes_uncommitted_content(tmp_path):
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    untracked = tmp_path / "untracked.txt"
    tracked.write_text("one\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.txt"],
        cwd=tmp_path,
        check=True,
    )
    untracked.write_text("alpha\n", encoding="utf-8")

    first = HARNESS.source_tree_sha256(tmp_path)
    untracked.write_text("beta\n", encoding="utf-8")
    second = HARNESS.source_tree_sha256(tmp_path)

    assert len(first) == 64
    assert first != second


def test_validate_unchanged_detects_same_status_different_content(tmp_path):
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "spec-v2@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Spec V2 Test"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "base"],
        cwd=tmp_path,
        check=True,
    )
    tracked.write_text("first dirty payload\n", encoding="utf-8")
    before = HARNESS.git_snapshot(tmp_path)
    tracked.write_text("second dirty payload\n", encoding="utf-8")
    after = HARNESS.git_snapshot(tmp_path)

    assert before["status_porcelain_v1"] == after["status_porcelain_v1"]
    with pytest.raises(AssertionError, match="source_tree_sha256"):
        HARNESS.validate_unchanged(before, after)


def test_model_snapshot_hashes_metadata_and_inventories_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text(
        '{"version":"1"}\n',
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    resolved, snapshot = HARNESS.model_snapshot(str(tmp_path))

    assert resolved == tmp_path.resolve()
    assert snapshot["metadata_files"]["config.json"]["sha256"] == (
        HARNESS.sha256_file(tmp_path / "config.json")
    )
    assert snapshot["weight_files"] == [
        {
            "name": "model.safetensors",
            "size_bytes": 7,
            "sha256": HARNESS.sha256_file(
                tmp_path / "model.safetensors"
            ),
        }
    ]
    assert snapshot["total_weight_bytes"] == 7

    (tmp_path / "model.safetensors").write_bytes(b"WEIGHTS")
    _, changed = HARNESS.model_snapshot(str(tmp_path))
    assert changed["weight_files"][0]["size_bytes"] == 7
    assert (
        changed["weight_files"][0]["sha256"]
        != snapshot["weight_files"][0]["sha256"]
    )


def test_write_json_exclusive_never_overwrites(tmp_path):
    output = tmp_path / "evidence.json"
    HARNESS.write_json_exclusive(output, {"run": 1})

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        HARNESS.write_json_exclusive(output, {"run": 2})

    assert json.loads(output.read_text(encoding="utf-8")) == {"run": 1}


def test_write_json_exclusive_rejects_symlink_without_touching_target(tmp_path):
    target = tmp_path / "target.json"
    target.write_text('{"original":true}\n', encoding="utf-8")
    output = tmp_path / "evidence.json"
    output.symlink_to(target)

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        HARNESS.write_json_exclusive(output, {"replacement": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "original": True
    }


def test_write_json_exclusive_cleans_temporary_after_link_failure(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "evidence.json"

    def fail_link(*args, **kwargs):
        raise OSError("injected link failure")

    monkeypatch.setattr(HARNESS.os, "link", fail_link)
    with pytest.raises(OSError, match="injected link failure"):
        HARNESS.write_json_exclusive(output, {"run": 1})

    assert not output.exists()
    assert list(tmp_path.glob(".evidence.json.*.tmp")) == []


def test_write_json_exclusive_has_one_concurrent_winner(tmp_path):
    output = tmp_path / "evidence.json"

    def publish(value):
        try:
            HARNESS.write_json_exclusive(output, {"run": value})
        except RuntimeError:
            return "lost"
        return "won"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (1, 2)))

    assert sorted(outcomes) == ["lost", "won"]
    assert json.loads(output.read_text(encoding="utf-8"))["run"] in (1, 2)


def _retained_gpu_snapshot(**overrides):
    gpu_uuid = "GPU-773e0633-edb0-6c38-1d2b-d232f9109126"
    identity = {
        "name": HARNESS.RETAINED_GPU_NAME,
        "gpu_uuid": gpu_uuid,
        "driver_version": "570.133.20",
        "total_memory_mib": "40960",
    }
    snapshot = {
        "phase": "before",
        "index": 0,
        "name": HARNESS.RETAINED_GPU_NAME,
        "uuid": gpu_uuid.removeprefix("GPU-"),
        "compute_capability": [8, 0],
        "total_memory_bytes": 42406903808,
        "multiprocessor_count": 108,
        "nvidia_smi_query_id": gpu_uuid,
        "nvidia_smi_raw_lines": [],
        "nvidia_smi_parsed_rows": [identity],
        "nvidia_smi_returncode": 0,
        "nvidia_smi_stderr": "",
        "compute_apps_raw_lines": [],
        "compute_apps": [],
        "compute_apps_returncode": 0,
        "compute_apps_stderr": "",
        "own_process_namespace_ids": [100, 200],
        "own_gpu_process_ids": [200],
        "foreign_compute_apps": [],
    }
    snapshot.update(overrides)
    return snapshot


def test_gpu_csv_parsers_preserve_quoted_process_names():
    identity = HARNESS.parse_gpu_identity_rows(
        'NVIDIA A100-SXM4-40GB, GPU-1, 570.1, 40960\n'
    )
    apps = HARNESS.parse_compute_app_rows(
        'GPU-1, 123, "python, worker", 10\n'
    )

    assert identity[0]["gpu_uuid"] == "GPU-1"
    assert apps[0]["process_name"] == "python, worker"
    assert apps[0]["pid"] == 123
    assert HARNESS.normalize_gpu_uuid("abc") == "GPU-abc"
    assert HARNESS.normalize_gpu_uuid("GPU-abc") == "GPU-abc"


def test_retained_gpu_gate_accepts_exact_a100_endpoint():
    HARNESS.validate_retained_gpu_snapshot(
        _retained_gpu_snapshot(),
        phase="before",
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": "NVIDIA H100 80GB HBM3"}, "requires NVIDIA A100"),
        ({"compute_capability": [9, 0]}, "requires compute capability"),
        ({"nvidia_smi_parsed_rows": []}, "returned 0 rows"),
        (
            {
                "nvidia_smi_parsed_rows": [{
                    "name": HARNESS.RETAINED_GPU_NAME,
                    "gpu_uuid": "GPU-other",
                    "driver_version": "570.133.20",
                    "total_memory_mib": "40960",
                }]
            },
            "identities differ",
        ),
        ({"compute_apps_returncode": 1}, "consumer query failed"),
    ],
)
def test_retained_gpu_gate_rejects_bad_identity_or_query(overrides, message):
    with pytest.raises(AssertionError, match=message):
        HARNESS.validate_retained_gpu_snapshot(
            _retained_gpu_snapshot(**overrides),
            phase="before",
        )


def test_retained_gpu_gate_accepts_namespace_alias_and_rejects_foreign_pid():
    own_row = {
        "gpu_uuid": "GPU-773e0633-edb0-6c38-1d2b-d232f9109126",
        "pid": 200,
        "pid_raw": "200",
        "process_name": "python",
        "used_memory_mib": 10,
        "used_memory_raw": "10",
    }
    HARNESS.validate_retained_gpu_snapshot(
        _retained_gpu_snapshot(compute_apps=[own_row]),
        phase="before",
    )

    foreign_row = {**own_row, "pid": 300, "pid_raw": "300"}
    with pytest.raises(AssertionError, match="foreign compute consumers"):
        HARNESS.validate_retained_gpu_snapshot(
            _retained_gpu_snapshot(
                compute_apps=[foreign_row],
                foreign_compute_apps=[foreign_row],
            ),
            phase="before",
        )


def test_retained_gpu_gate_rejects_changed_endpoint_identity():
    before = _retained_gpu_snapshot()
    after = copy.deepcopy(before)
    after["nvidia_smi_parsed_rows"][0]["driver_version"] = "571.0"

    with pytest.raises(AssertionError, match="identity changed"):
        HARNESS.validate_same_gpu_endpoints(before, after)


def _v0_comparator_fixture():
    runner_commit = "1" * 40
    model = {
        "argument": "/model",
        "resolved_path": "/model",
        "metadata_files": {
            "config.json": {"size_bytes": 2, "sha256": "2" * 64}
        },
        "weight_files": [
            {
                "name": "model.safetensors",
                "size_bytes": 7,
                "sha256": "3" * 64,
            }
        ],
        "total_weight_bytes": 7,
    }
    tokenizer = {
        "class": "transformers.Tokenizer",
        "length": 3,
        "vocab_size": 3,
        "vocab_entries": 3,
        "vocab_sha256": "4" * 64,
        "bos_token_id": None,
        "eos_token_id": 2,
        "pad_token_id": None,
        "unk_token_id": None,
        "all_special_ids": [2],
        "backend_json_bytes": 17,
        "backend_json_sha256": "5" * 64,
    }
    outputs = (("x", (10, 11)), ("y", (12, 13)))
    trace = ((True, ((4, 0, 4, True),)), (False, ((5, 4, 1, False),)))
    args = SimpleNamespace(
        mode="eager",
        gpu_memory_utilization=0.5,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=4,
    )
    source_snapshot = {
        "head": HARNESS.CANONICAL_V0_COMMIT,
        "detached": True,
        "dirty": False,
    }
    runner_snapshot = {
        "head": runner_commit,
        "detached": False,
        "dirty": False,
    }
    document = {
        "schema": HARNESS.V0_EVIDENCE_SCHEMA,
        "source": {
            "expected_commit": HARNESS.CANONICAL_V0_COMMIT,
            "git_before": source_snapshot,
            "git_after": source_snapshot,
            "unchanged_during_run": True,
        },
        "runner": {
            "expected_commit": runner_commit,
            "git_before": runner_snapshot,
            "git_after": runner_snapshot,
            "script_sha256_before": "6" * 64,
            "script_sha256_after": "6" * 64,
            "unchanged_during_run": True,
        },
        "model": {
            "before": model,
            "after": model,
            "unchanged_during_run": True,
        },
        "tokenizer": tokenizer,
        "run": {
            "mode": "eager",
            "requested_engine_config": {
                "enforce_eager": True,
                "gpu_memory_utilization": 0.5,
                "max_model_len": 512,
                "max_num_batched_tokens": 512,
                "max_num_seqs": 4,
                "tensor_parallel_size": 1,
                "kvcache_block_size": 256,
                "num_kvcache_blocks": -1,
                "top_p_backend": "exact",
                "disable_python_gc": False,
            },
            "workload": {
                "prompts_token_ids": [[1, 2, 3, 4], [7, 8, 9]],
                "sampling_params": {
                    "temperature": 0.0,
                    "max_tokens": 4,
                    "ignore_eos": True,
                    "top_k": -1,
                    "top_p": 1.0,
                },
            },
            "comparison_outputs": HARNESS.nested_dict(outputs),
            "scheduler_trace": HARNESS.nested_dict(trace),
        },
        "randomness": {"seed": 20260828, "sampling_mode": "greedy"},
    }
    return document, runner_commit, model, tokenizer, outputs, trace, args


def _write_v0_document(path, document):
    path.write_text(
        json.dumps(document, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return HARNESS.sha256_file(path)


def test_v0_comparator_accepts_only_the_registered_equivalent_fixture(tmp_path):
    document, runner, model, tokenizer, outputs, trace, args = (
        _v0_comparator_fixture()
    )
    path = tmp_path / "v0.json"
    digest = _write_v0_document(path, document)

    result = HARNESS.validate_v0_golden(
        path,
        expected_sha256=digest,
        expected_runner_commit=runner,
        args=args,
        target_model_evidence=model,
        tokenizer_evidence=tokenizer,
        outputs=outputs,
        scheduler_trace=trace,
    )

    assert result["outputs_match"] is True
    assert result["scheduler_trace_match"] is True
    assert result["sha256"] == digest


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda d: d.update(schema="wrong"), "wrong evidence schema"),
        (
            lambda d: d["model"]["before"]["weight_files"][0].update(
                sha256="f" * 64
            ),
            "model artifacts differ",
        ),
        (
            lambda d: d["tokenizer"].update(vocab_sha256="f" * 64),
            "runtime tokenizer differs",
        ),
        (
            lambda d: d["run"]["requested_engine_config"].update(
                max_model_len=511
            ),
            "engine setting differs",
        ),
        (
            lambda d: d["run"]["requested_engine_config"].update(
                kvcache_block_size=128
            ),
            "fixed engine contract differs",
        ),
        (
            lambda d: d["randomness"].update(seed=0),
            "randomness contract differs",
        ),
        (
            lambda d: d["run"]["workload"].update(prompts_token_ids=[[1]]),
            "workload differs",
        ),
        (
            lambda d: d["run"].update(comparison_outputs=[]),
            "outputs differ",
        ),
        (
            lambda d: d["run"].update(scheduler_trace=[]),
            "scheduler trace differs",
        ),
    ],
)
def test_v0_comparator_rejects_mismatched_contract_cells(
    tmp_path,
    mutation,
    message,
):
    document, runner, model, tokenizer, outputs, trace, args = (
        _v0_comparator_fixture()
    )
    document = copy.deepcopy(document)
    mutation(document)
    path = tmp_path / "v0.json"
    digest = _write_v0_document(path, document)

    with pytest.raises(AssertionError, match=message):
        HARNESS.validate_v0_golden(
            path,
            expected_sha256=digest,
            expected_runner_commit=runner,
            args=args,
            target_model_evidence=model,
            tokenizer_evidence=tokenizer,
            outputs=outputs,
            scheduler_trace=trace,
        )


def test_v0_comparator_rejects_artifact_hash_mismatch(tmp_path):
    document, runner, model, tokenizer, outputs, trace, args = (
        _v0_comparator_fixture()
    )
    path = tmp_path / "v0.json"
    _write_v0_document(path, document)

    with pytest.raises(AssertionError, match="artifact hash mismatch"):
        HARNESS.validate_v0_golden(
            path,
            expected_sha256="0" * 64,
            expected_runner_commit=runner,
            args=args,
            target_model_evidence=model,
            tokenizer_evidence=tokenizer,
            outputs=outputs,
            scheduler_trace=trace,
        )


def test_v0_and_v2_runtime_tokenizer_snapshots_share_one_schema():
    class Backend:
        @staticmethod
        def to_str():
            return '{"type":"fake"}'

    class Tokenizer:
        backend_tokenizer = Backend()
        vocab_size = 2
        bos_token_id = None
        eos_token_id = 1
        pad_token_id = None
        unk_token_id = None
        all_special_ids = [1]

        @staticmethod
        def get_vocab():
            return {"a": 0, "<eos>": 1}

        @staticmethod
        def __len__():
            return 2

    tokenizer = Tokenizer()

    assert HARNESS.runtime_tokenizer_snapshot(tokenizer) == (
        V0_RUNNER.runtime_tokenizer_snapshot(tokenizer)
    )
