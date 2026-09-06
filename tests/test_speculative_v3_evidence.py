import importlib.util
import copy
import builtins
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

import _speculative_v3_evidence as EVIDENCE
import compare_speculative_v3_cache_neutrality as COMPARATOR
import compare_speculative_v3_output_control as OUTPUT_COMPARATOR


def load_script(name):
    path = TESTS_ROOT / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROUTE_RUNNER = load_script("run_speculative_v3_route_compile.py")
CACHE_RUNNER = load_script("run_speculative_v3_cache_neutrality.py")


def git(repo, *args):
    return subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def initialize_repository(path):
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "evidence@example.invalid")
    git(path, "config", "user.name", "Evidence Test")
    (path / "nanovllm").mkdir()
    (path / "nanovllm" / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (path / "tracked.txt").write_text("tracked-a\n", encoding="utf-8")
    git(path, "add", ".")
    git(path, "commit", "-qm", "fixture")
    return path


@pytest.mark.parametrize("producer", [
    "7fec9993d5e4e0e06fec22e3973dfc203fdbd2d8",
    "e8e0452f99727958077b51f340a5375a090e6884",
])
def test_real_v3_implementation_pin_matches_frozen_producer_tree(producer):
    # Historical certification pins the producer, not all future HEADs.
    snapshot = {"nanovllm_tree": git(REPO_ROOT, "rev-parse", f"{producer}:nanovllm")}
    binding = EVIDENCE.validate_implementation_binding(REPO_ROOT, snapshot)

    assert binding == {
        "commit": "7fec9993d5e4e0e06fec22e3973dfc203fdbd2d8",
        "tree": "5820fb685d76b549fe21117baa31b3b32ebae14b",
        "parent": "e252086ee625ea8aefdb63f3affc093380f40064",
        "nanovllm_tree": "52398af379f767708a0b804646f4b490fa8323ad",
    }


def test_real_v3_binding_rejects_a_changed_runtime_including_later_heads():
    with pytest.raises(AssertionError, match="producer commit changes"):
        EVIDENCE.validate_implementation_binding(REPO_ROOT, {"nanovllm_tree": "0" * 40})
    snapshot = EVIDENCE.git_snapshot(REPO_ROOT)
    if snapshot["nanovllm_tree"] == EVIDENCE.IMPLEMENTATION_NANOVLLM_TREE:
        EVIDENCE.validate_implementation_binding(REPO_ROOT, snapshot)
    else:
        with pytest.raises(AssertionError, match="producer commit changes"):
            EVIDENCE.validate_implementation_binding(REPO_ROOT, snapshot)


def test_real_v3_runtime_import_registry_matches_checkout():
    registry = {
        "nanovllm": (ROUTE_RUNNER.nanovllm, "nanovllm/__init__.py"),
        "LLM": (ROUTE_RUNNER.LLM, "nanovllm/llm.py"),
        "LLMEngine": (
            ROUTE_RUNNER.LLMEngine,
            "nanovllm/engine/llm_engine.py",
        ),
        "ModelRunner": (
            ROUTE_RUNNER.ModelRunner,
            "nanovllm/engine/model_runner.py",
        ),
        "Scheduler": (
            ROUTE_RUNNER.Scheduler,
            "nanovllm/engine/scheduler.py",
        ),
        "Sampler": (ROUTE_RUNNER.Sampler, "nanovllm/layers/sampler.py"),
        "SamplingParams": (
            ROUTE_RUNNER.SamplingParams,
            "nanovllm/sampling_params.py",
        ),
        "DraftRouteRegistry": (
            ROUTE_RUNNER.DraftRouteRegistry,
            "nanovllm/engine/speculative_routes.py",
        ),
    }

    identities = EVIDENCE.runtime_import_identities(REPO_ROOT, registry)
    assert set(identities) == set(registry)
    assert identities["LLM"]["path"] == "nanovllm/llm.py"
    assert all(identity["matches_head"] for identity in identities.values())

    wrong = dict(registry)
    wrong["LLM"] = (
        ROUTE_RUNNER.LLM,
        "nanovllm/engine/llm_engine.py",
    )
    with pytest.raises(AssertionError, match="runtime import LLM came from"):
        EVIDENCE.runtime_import_identities(REPO_ROOT, wrong)


def retained_provenance_fixture(repo_root):
    commit = "c" * 40
    source = {
        "repository": str(repo_root),
        "head": commit,
        "tree": "d" * 40,
        "nanovllm_tree": EVIDENCE.IMPLEMENTATION_NANOVLLM_TREE,
        "branch": "fixture",
        "detached": False,
        "clean": True,
        "status_porcelain_v1": [],
        "status_sha256": EVIDENCE.sha256_bytes(b""),
        "source_tree_sha256": "e" * 64,
    }

    def identity(path):
        return {
            "path": path,
            "sha256": "1" * 64,
            "head_blob": "a" * 40,
            "head_blob_sha256": "1" * 64,
            "matches_head": True,
        }

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
    environment = {
        "software": {
            "python": "3.12",
            "python_optimize": 0,
            "torch_dynamo_disable": False,
            "torch_dynamo_suppress_errors": False,
        },
        "hardware": {"device_count": 1},
        "selected_environment": {
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor-a",
            "TRITON_CACHE_DIR": "/tmp/triton-a",
            "CUDA_VISIBLE_DEVICES": "0",
            "TORCHDYNAMO_SUPPRESS_ERRORS": None,
        },
    }
    provenance = {
        "implementation": {
            "commit": EVIDENCE.IMPLEMENTATION_COMMIT,
            "tree": EVIDENCE.IMPLEMENTATION_TREE,
            "parent": EVIDENCE.IMPLEMENTATION_PARENT,
            "nanovllm_tree": EVIDENCE.IMPLEMENTATION_NANOVLLM_TREE,
        },
        "producer_commit": commit,
        "source": {"before": source, "after": copy.deepcopy(source), "unchanged": True},
        "source_files": {
            "before": {
                "runner": identity("tests/runner.py"),
                "helper": identity("tests/_speculative_v3_evidence.py"),
            },
            "after": {
                "runner": identity("tests/runner.py"),
                "helper": identity("tests/_speculative_v3_evidence.py"),
            },
            "unchanged": True,
        },
        "runtime_imports": {
            "before": {"Runtime": identity("nanovllm/runtime.py")},
            "after": {"Runtime": identity("nanovllm/runtime.py")},
            "unchanged": True,
        },
        "models": {
            role: {
                "before": copy.deepcopy(model),
                "after": copy.deepcopy(model),
                "unchanged": True,
            }
            for role in ("target", "draft")
        },
        "environment": {
            "before": environment,
            "after": copy.deepcopy(environment),
            "unchanged": True,
        },
        "invocation": [
            str(repo_root / "tests" / "runner.py"),
            "--retained",
            "--expected-commit",
            commit,
        ],
        "cwd": str(repo_root),
        "retention_requested": True,
        "retention_eligible": True,
    }
    return provenance, source, model, identity


def validate_fixture_provenance(
    monkeypatch,
    tmp_path,
    *,
    mutate=None,
):
    repo_root = tmp_path / "repo"
    provenance, source, model, identity = retained_provenance_fixture(repo_root)
    if mutate is not None:
        mutate(provenance)

    def fake_source_identity(path, observed_repo_root):
        assert observed_repo_root == repo_root
        resolved = Path(path).resolve()
        if resolved == Path(EVIDENCE.__file__).resolve():
            relative = "tests/_speculative_v3_evidence.py"
        else:
            relative = resolved.relative_to(repo_root.resolve()).as_posix()
        return identity(relative)

    def fake_model_snapshot(argument):
        return Path("/model"), copy.deepcopy(model)

    monkeypatch.setattr(EVIDENCE, "source_file_identity", fake_source_identity)
    monkeypatch.setattr(EVIDENCE, "model_snapshot", fake_model_snapshot)
    return EVIDENCE.validate_retained_provenance(
        provenance,
        repo_root=repo_root,
        current_source=source,
        expected_commit="c" * 40,
        expected_runner_path="tests/runner.py",
        expected_model_roles=("target", "draft"),
        expected_runtime_imports={"Runtime": "nanovllm/runtime.py"},
    )


def mutate_environment_endpoints(provenance, section, **updates):
    for endpoint in ("before", "after"):
        provenance["environment"][endpoint][section].update(updates)


def test_retained_provenance_accepts_complete_independently_bound_record(
    monkeypatch, tmp_path
):
    validated = validate_fixture_provenance(monkeypatch, tmp_path)
    assert validated["producer_commit"] == "c" * 40
    assert set(validated["models"]) == {"target", "draft"}


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda value: value.update(producer_commit="f" * 40),
            "producer commit",
        ),
        (
            lambda value: value.update(retention_requested=False),
            "retained mode",
        ),
        (
            lambda value: value["source"]["before"].update(clean=False),
            "source changed",
        ),
        (
            lambda value: value["source_files"]["before"]["runner"].update(
                sha256="9" * 64
            ),
            "source files changed",
        ),
        (
            lambda value: value["runtime_imports"]["before"].clear(),
            "runtime imports changed",
        ),
        (
            lambda value: value["runtime_imports"]["before"]["Runtime"].update(
                path="nanovllm/wrong.py"
            ),
            "runtime imports changed",
        ),
        (
            lambda value: value["models"]["target"]["before"][
                "weight_files"
            ][0].update(sha256="9" * 64),
            "target model changed",
        ),
        (
            lambda value: value["invocation"].remove("--retained"),
            "omitted --retained",
        ),
        (
            lambda value: mutate_environment_endpoints(
                value, "software", python_optimize=1
            ),
            "optimized Python",
        ),
        (
            lambda value: mutate_environment_endpoints(
                value, "software", torch_dynamo_suppress_errors=True
            ),
            "suppressed TorchDynamo errors",
        ),
        (
            lambda value: mutate_environment_endpoints(
                value,
                "selected_environment",
                TORCHDYNAMO_SUPPRESS_ERRORS="0",
            ),
            "TORCHDYNAMO_SUPPRESS_ERRORS",
        ),
        (
            lambda value: mutate_environment_endpoints(
                value,
                "selected_environment",
                TORCHINDUCTOR_CACHE_DIR=str(value["source"]["before"]["repository"]),
            ),
            "outside the source checkout",
        ),
    ),
)
def test_retained_provenance_rejects_fabricated_or_changed_fields(
    monkeypatch,
    tmp_path,
    mutation,
    match,
):
    with pytest.raises(AssertionError, match=match):
        validate_fixture_provenance(
            monkeypatch,
            tmp_path,
            mutate=mutation,
        )


def test_validated_pair_identity_ignores_only_isolated_cache_paths():
    value = {
        "environment": {
            "selected_environment": {
                "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor-a",
                "TRITON_CACHE_DIR": "/tmp/triton-a",
                "CUDA_VISIBLE_DEVICES": "0",
            }
        }
    }
    other = copy.deepcopy(value)
    other["environment"]["selected_environment"].update(
        TORCHINDUCTOR_CACHE_DIR="/tmp/inductor-b",
        TRITON_CACHE_DIR="/tmp/triton-b",
    )
    assert COMPARATOR.normalized_validated_provenance(
        value
    ) == COMPARATOR.normalized_validated_provenance(other)

    other["environment"]["selected_environment"]["CUDA_VISIBLE_DEVICES"] = "1"
    assert COMPARATOR.normalized_validated_provenance(
        value
    ) != COMPARATOR.normalized_validated_provenance(other)


def test_paired_producer_compiler_caches_must_be_disjoint():
    def record(inductor, triton):
        return {
            "environment": {
                "selected_environment": {
                    "TORCHINDUCTOR_CACHE_DIR": inductor,
                    "TRITON_CACHE_DIR": triton,
                }
            }
        }

    first = record("/tmp/a-inductor", "/tmp/a-triton")
    second = record("/tmp/b-inductor", "/tmp/b-triton")
    assert EVIDENCE.require_independent_compiler_cache_roots(
        (first, second)
    ) == [
        (Path("/tmp/a-inductor"), Path("/tmp/a-triton")),
        (Path("/tmp/b-inductor"), Path("/tmp/b-triton")),
    ]

    reused = record("/tmp/a-inductor", "/tmp/c-triton")
    with pytest.raises(AssertionError, match="reused or nested"):
        EVIDENCE.require_independent_compiler_cache_roots((first, reused))

    nested = record("/tmp/a-inductor/nested", "/tmp/d-triton")
    with pytest.raises(AssertionError, match="reused or nested"):
        EVIDENCE.require_independent_compiler_cache_roots((first, nested))


def valid_compiler_environment(monkeypatch):
    for name in (
        "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
        "TORCHINDUCTOR_AUTOGRAD_CACHE",
        "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE",
        "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE",
        "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE",
        "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO",
        "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_REMOTE_PGO",
    ):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("TORCH_LOGS", "recompiles,graph_breaks")
    monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)
    monkeypatch.delenv("TORCHDYNAMO_SUPPRESS_ERRORS", raising=False)


@pytest.mark.parametrize(
    "name",
    (
        "TORCH_COMPILE_DISABLE",
        "TORCHDYNAMO_DISABLE",
        "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO",
        "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_REMOTE_PGO",
    ),
)
def test_route_compiler_gate_rejects_disabled_or_pgo_execution(
    monkeypatch, name
):
    valid_compiler_environment(monkeypatch)
    ROUTE_RUNNER.require_compiler_environment()
    monkeypatch.setenv(name, "1")
    with pytest.raises(AssertionError, match=name):
        ROUTE_RUNNER.require_compiler_environment()


@pytest.mark.parametrize("value", ("0", "1"))
def test_route_compiler_gate_rejects_suppressed_errors(monkeypatch, value):
    valid_compiler_environment(monkeypatch)
    monkeypatch.setenv("TORCHDYNAMO_SUPPRESS_ERRORS", value)
    with pytest.raises(AssertionError, match="TORCHDYNAMO_SUPPRESS_ERRORS"):
        ROUTE_RUNNER.require_compiler_environment()


def test_retained_runtime_gate_rejects_optimized_python(monkeypatch):
    monkeypatch.setattr(
        EVIDENCE.sys,
        "flags",
        SimpleNamespace(optimize=1),
    )
    with pytest.raises(AssertionError, match="optimized Python"):
        EVIDENCE.require_retained_runtime_environment()


def test_route_retained_configuration_is_canonical():
    args = ROUTE_RUNNER.parse_args(
        [
            "--model",
            "/model",
            "--mode",
            "eager",
            "--retained",
        ]
    )
    ROUTE_RUNNER.require_retained_configuration(args)
    for name in ROUTE_RUNNER.RETAINED_CONFIGURATION:
        changed = copy.copy(args)
        value = getattr(changed, name)
        setattr(changed, name, value + 1)
        with pytest.raises(AssertionError, match=name):
            ROUTE_RUNNER.require_retained_configuration(changed)


def test_route_compiler_snapshot_must_be_nonvacuous():
    empty = {"counters": {}, "cache_manifest": []}
    with pytest.raises(AssertionError, match="no compiled graphs"):
        ROUTE_RUNNER.require_nonvacuous_compiler_snapshot(empty)

    compiled = {
        "counters": {"'stats'": {"'unique_graphs'": 1}},
        "cache_manifest": [
            {"root": "inductor", "path": "graph.py", "bytes": 1}
        ],
    }
    ROUTE_RUNNER.require_nonvacuous_compiler_snapshot(compiled)


def test_route_harness_rejects_optimized_python_before_argparse():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(REPO_ROOT), environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-O",
            str(TESTS_ROOT / "run_speculative_v3_route_compile.py"),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    output = completed.stdout + completed.stderr
    assert "route certification refuses optimized Python" in output
    assert "the following arguments are required" not in output


def test_route_cache_roots_must_be_distinct_non_nested_and_external(tmp_path):
    repo = tmp_path / "repo"
    model = tmp_path / "model"
    repo.mkdir()
    model.mkdir()
    valid_inductor = tmp_path / "inductor"
    valid_triton = tmp_path / "triton"
    ROUTE_RUNNER.validate_cache_root_isolation(
        valid_inductor,
        valid_triton,
        repo_root=repo,
        model_roots=(model,),
    )

    invalid_pairs = (
        (valid_inductor, valid_inductor),
        (valid_inductor, valid_inductor / "nested"),
        (repo / "cache", valid_triton),
        (valid_inductor, model / "cache"),
    )
    for inductor, triton in invalid_pairs:
        with pytest.raises(AssertionError):
            ROUTE_RUNNER.validate_cache_root_isolation(
                inductor,
                triton,
                repo_root=repo,
                model_roots=(model,),
            )


def test_shared_compiler_cache_roots_are_explicit_canonical_and_isolated(
    monkeypatch, tmp_path
):
    inductor = tmp_path / "inductor"
    triton = tmp_path / "triton"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(inductor))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(triton))

    roots = EVIDENCE.compiler_cache_roots_from_environment(required=True)
    assert roots == (inductor.resolve(), triton.resolve())
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(inductor.resolve())
    EVIDENCE.validate_compiler_cache_root_isolation(
        roots,
        repo_root=tmp_path / "repo",
        model_roots=(tmp_path / "model",),
    )

    monkeypatch.setenv("TRITON_CACHE_DIR", str(inductor / "nested"))
    with pytest.raises(AssertionError, match="nested"):
        EVIDENCE.compiler_cache_roots_from_environment(required=True)

    monkeypatch.delenv("TRITON_CACHE_DIR")
    with pytest.raises(AssertionError, match="explicit Inductor and Triton"):
        EVIDENCE.compiler_cache_roots_from_environment(required=True)


def test_route_wrapper_rolls_back_draft_and_ordinary_decode(
    monkeypatch,
):
    calls = []

    class Scheduler:
        def schedule(self):
            calls.append("schedule")
            return [SimpleNamespace()], False

        def capture_decode_schedule_rollback(self, seqs, *, is_prefill):
            calls.append("capture")
            return "ordinary-rollback"

        def rollback_failed_decode_schedule(self, rollback):
            calls.append(("ordinary", rollback))

    engine = SimpleNamespace(scheduler=Scheduler())

    def rollback_draft(plan, error):
        calls.append(("draft", plan, str(error)))

    engine._rollback_failed_draft_state = rollback_draft

    def fail_after_plan(engine, *, plans, **kwargs):
        plans.append("draft-plan")
        raise RuntimeError("route failed")

    monkeypatch.setattr(
        ROUTE_RUNNER,
        "_force_scheduled_route_cycle",
        fail_after_plan,
    )
    with pytest.raises(RuntimeError, match="route failed"):
        ROUTE_RUNNER.force_route_cycle(
            engine,
            effective_k=2,
            cache_roots=(),
            capture_ledger={},
            run_id="run-id",
            record_index=0,
        )

    assert calls == [
        "schedule",
        "capture",
        ("draft", "draft-plan", "route failed"),
        ("ordinary", "ordinary-rollback"),
    ]


def test_guarded_interval_combined_event_order(monkeypatch):
    events = []
    monkeypatch.setattr(
        ROUTE_RUNNER.torch.compiler,
        "set_stance",
        lambda stance: events.append(stance),
    )
    original_print = builtins.print

    def recorded_print(*args, **kwargs):
        events.append(args[0])

    monkeypatch.setattr(builtins, "print", recorded_print)
    try:
        with ROUTE_RUNNER.guarded_draft_interval("run=x,batch=1"):
            events.append("body")
    finally:
        monkeypatch.setattr(builtins, "print", original_print)

    assert events == [
        "fail_on_recompile",
        "V3_DRAFT_INTERVAL_BEGIN run=x,batch=1",
        "body",
        "V3_DRAFT_INTERVAL_END run=x,batch=1",
        "default",
    ]


def output_control_fixture(side, mode):
    stages = (
        ("prefill", 7, 0),
        ("first_target_decode", 0, 2),
        ("repeated_target_decode", 0, 2),
    )
    steps = []
    for stage_index, (stage, prefill, decode) in enumerate(stages):
        events = [
            {
                "stage": stage,
                "seq_id": seq_id,
                "token_id": 100 * stage_index + seq_id,
                "finished": False,
            }
            for seq_id in (1, 2)
        ]
        steps.append(
            {
                "stage": stage,
                "num_prefill_tokens": prefill,
                "num_decode_tokens": decode,
                "events": events,
            }
        )
    flattened = [event for step in steps for event in step["events"]]
    rng = {"cpu_sha256": "a" * 64, "cuda_sha256": "b" * 64}
    phase_calls = (
        {name: 0 for name in OUTPUT_COMPARATOR.DRAFT_PHASES}
        if side == "off"
        else {
            "_construct_draft_model": 1,
            "warmup_draft_model": 1,
            "capture_draft_cudagraph": 0 if mode == "eager" else 2,
            "_pretouch_draft_eager_prefill": 0 if mode == "eager" else 1,
            "_pretouch_draft_routes": 1,
        }
    )
    calls = []
    if side == "on":
        for index in range(2):
            calls.append(
                {
                    "route": {
                        "schema": "draft-discard-v1",
                        "execution_mode": (
                            "eager_dynamic" if mode == "eager" else "cuda_graph"
                        ),
                        "batch_bucket": 2,
                        "effective_k": 2,
                        "catchup_family": (
                            "paged_eager_dynamic_v1" if index == 0 else "none"
                        ),
                        "sampler_envelope": (
                            "exact_all_compositions_worst_case_v1"
                        ),
                    },
                    "catchup_tokens": 7 if index == 0 else 0,
                    "proposal_token_ids": [[10, 11], [20, 21]],
                    "graph_decode_steps": 0 if mode == "eager" else 2,
                    "eager_decode_steps": 2 if mode == "eager" else 0,
                    "rng_before": copy.deepcopy(rng),
                    "rng_after": copy.deepcopy(rng),
                    "rng_neutral": True,
                }
            )
    return {
        "schema": OUTPUT_COMPARATOR.INPUT_SCHEMA,
        "side": side,
        "mode": mode,
        "seed": 20260828,
        "model": "/model",
        "draft_model_argument": "/model",
        "configured_k": 2,
        "configuration": {
            "max_model_len": 512,
            "max_num_batched_tokens": 512,
            "max_num_seqs": 2,
            "gpu_memory_utilization": 0.5,
            "top_p_backend": "exact",
            "tensor_parallel_size": 1,
        },
        "workload": {
            "prompt_token_ids": [[10, 11, 12, 13], [20, 21, 22]],
            "sampling": {
                "temperature": 0.8,
                "top_k": 8,
                "top_p": 0.9,
                "max_tokens": 6,
                "ignore_eos": True,
            },
        },
        "retention_eligible": True,
        "provenance": {"retention_eligible": True},
        "draft_phase_calls": phase_calls,
        "live_draft_resource_attributes": {
            name: side == "on"
            for name in OUTPUT_COMPARATOR.DRAFT_RESOURCE_ATTRIBUTES
        },
        "draft_owned_instance_attributes": (
            []
            if side == "off"
            else list(OUTPUT_COMPARATOR.DRAFT_OWNED_ATTRIBUTES)
        ),
        "runtime_draft_calls": calls,
        "rng_snapshots": {
            name: copy.deepcopy(rng)
            for name in (
                "after_init",
                "after_prefill",
                "after_first_target_decode",
                "after_repeated_target_decode",
            )
        },
        "steps": steps,
        "authoritative_target_events": flattened,
        "target_token_ids_by_seq": {
            str(seq_id): [
                event["token_id"]
                for event in flattened
                if event["seq_id"] == seq_id
            ]
            for seq_id in (1, 2)
        },
    }


@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_output_control_v2_raw_schema_accepts_exact_off_on_contract(mode):
    OUTPUT_COMPARATOR.validate_raw_artifact(
        output_control_fixture("off", mode),
        "off",
    )
    OUTPUT_COMPARATOR.validate_raw_artifact(
        output_control_fixture("on", mode),
        "on",
    )


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda value: value["draft_phase_calls"].update(
                capture_draft_cudagraph=1
            ),
            "phase counts",
        ),
        (
            lambda value: value["steps"][0]["events"][0].update(seq_id=99),
            "sequence ID",
        ),
        (
            lambda value: value["runtime_draft_calls"][0]["rng_after"].update(
                cuda_sha256="c" * 64
            ),
            "changed target RNG",
        ),
    ),
)
def test_output_control_v2_raw_schema_rejects_claim_corruption(
    mutation,
    match,
):
    value = output_control_fixture("on", "graph")
    mutation(value)
    with pytest.raises(AssertionError, match=match):
        OUTPUT_COMPARATOR.validate_raw_artifact(value, "on")


def test_source_tree_hash_covers_untracked_and_symlink_payloads(tmp_path):
    repo = initialize_repository(tmp_path / "repo")
    baseline = EVIDENCE.source_tree_sha256(repo)
    untracked = repo / "untracked.txt"
    untracked.write_text("untracked-a\n", encoding="utf-8")
    with_untracked = EVIDENCE.source_tree_sha256(repo)
    assert with_untracked != baseline

    link = repo / "link"
    link.symlink_to("tracked.txt")
    with_link = EVIDENCE.source_tree_sha256(repo)
    assert with_link != with_untracked

    link.unlink()
    link.symlink_to("untracked.txt")
    assert EVIDENCE.source_tree_sha256(repo) != with_link


def test_git_snapshot_ignores_hostile_git_environment(tmp_path, monkeypatch):
    expected_repo = initialize_repository(tmp_path / "expected")
    foreign_repo = initialize_repository(tmp_path / "foreign")
    expected_head = git(expected_repo, "rev-parse", "HEAD")
    monkeypatch.setenv("GIT_DIR", str(foreign_repo / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(foreign_repo))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(foreign_repo / ".git" / "objects"))

    snapshot = EVIDENCE.git_snapshot(expected_repo)

    assert snapshot["repository"] == str(expected_repo.resolve())
    assert snapshot["head"] == expected_head


def test_git_snapshot_ignores_hostile_git_config_injection(tmp_path, monkeypatch):
    repo = initialize_repository(tmp_path / "repo")
    expected = EVIDENCE.git_snapshot(repo)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "/does/not/exist")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/does/not/exist")
    monkeypatch.setenv("GIT_SHALLOW_FILE", "/does/not/exist")

    assert EVIDENCE.git_snapshot(repo) == expected


def make_model(path):
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    (path / "tokenizer.json").write_text('{"fixture":true}\n', encoding="utf-8")
    (path / "model-00002-of-00002.safetensors").write_bytes(b"weights-B")
    (path / "model-00001-of-00002.safetensors").write_bytes(b"weights-A")
    return path


def test_model_snapshot_hashes_all_shards_and_detects_equal_size_mutation(tmp_path):
    model = make_model(tmp_path / "model")
    _, before = EVIDENCE.model_snapshot(str(model))
    assert [item["name"] for item in before["weight_files"]] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert before["total_weight_bytes"] == len(b"weights-A") + len(b"weights-B")

    (model / "model-00001-of-00002.safetensors").write_bytes(b"weights-Z")
    _, after = EVIDENCE.model_snapshot(str(model))
    assert before["weight_files"][0]["size_bytes"] == after["weight_files"][0]["size_bytes"]
    assert (
        EVIDENCE.model_content_identity(before)
        != EVIDENCE.model_content_identity(after)
    )


@pytest.mark.parametrize("missing", ["config", "weights"])
def test_model_snapshot_rejects_incomplete_model(tmp_path, missing):
    model = tmp_path / "model"
    model.mkdir()
    if missing != "config":
        (model / "config.json").write_text("{}\n", encoding="utf-8")
    if missing != "weights":
        (model / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(AssertionError):
        EVIDENCE.model_snapshot(str(model))


def test_output_preflight_rejects_source_model_cache_and_existing(tmp_path):
    model = make_model(tmp_path / "model")
    cache = tmp_path / "cache"
    cache.mkdir()
    safe = tmp_path / "evidence" / "result.json"
    assert EVIDENCE.validate_output_paths(
        (safe,),
        repo_root=REPO_ROOT,
        model_roots=(model,),
        cache_roots=(cache,),
    ) == (safe.resolve(),)

    for forbidden in (
        REPO_ROOT / "evidence.json",
        model / "evidence.json",
        cache / "evidence.json",
    ):
        with pytest.raises(AssertionError):
            EVIDENCE.validate_output_paths(
                (forbidden,),
                repo_root=REPO_ROOT,
                model_roots=(model,),
                cache_roots=(cache,),
            )

    safe.parent.mkdir()
    safe.write_text("already here\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="overwrite"):
        EVIDENCE.validate_output_paths((safe,), repo_root=REPO_ROOT)


def test_exclusive_json_writer_does_not_follow_existing_symlink(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("preserve me\n", encoding="utf-8")
    link = tmp_path / "result.json"
    link.symlink_to(target)

    with pytest.raises(AssertionError, match="symlink"):
        EVIDENCE.write_json_exclusive(link, {"changed": True})

    assert target.read_text(encoding="utf-8") == "preserve me\n"


def test_exclusive_writer_rejects_dangling_symlink(tmp_path):
    missing = tmp_path / "missing.json"
    link = tmp_path / "result.json"
    link.symlink_to(missing)

    with pytest.raises(AssertionError, match="symlink"):
        EVIDENCE.write_json_exclusive(link, {"changed": True})

    assert link.is_symlink()
    assert not missing.exists()


def test_cache_runner_keeps_dangling_output_lexical_until_validation(tmp_path):
    output = tmp_path / "result.json"
    output.symlink_to(tmp_path / "missing.json")
    requested_output, requested_tensor = CACHE_RUNNER.requested_output_paths(
        output
    )

    assert requested_output == output
    assert requested_tensor == tmp_path / "result.tensors.pt"
    with pytest.raises(AssertionError, match="symlink"):
        EVIDENCE.validate_output_paths(
            (requested_output, requested_tensor),
            repo_root=REPO_ROOT,
        )


@pytest.mark.parametrize(
    "payload",
    (
        b'{"field":1,"field":2}',
        b'{"field":NaN}',
        b'{"field":Infinity}',
        b'{"field":-Infinity}',
    ),
)
def test_strict_json_rejects_duplicates_and_nonfinite_values(payload):
    with pytest.raises((AssertionError, ValueError)):
        EVIDENCE.load_strict_json_bytes(payload)


def test_registered_file_rejects_symlink_before_read(tmp_path):
    target = tmp_path / "target.bin"
    target.write_bytes(b"registered")
    link = tmp_path / "link.bin"
    link.symlink_to(target)

    with pytest.raises(AssertionError, match="symlink"):
        EVIDENCE.read_registered_file(link, max_bytes=100)


def test_cache_sidecar_digest_is_checked_before_torch_load(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    raw_path = root / "result.json"
    raw_path.write_text("{}\n", encoding="utf-8")
    sidecar = root / "result.tensors.pt"
    sidecar.write_bytes(b"not a tensor archive")
    value = {
        "mode": "eager",
        "draft_cache_fill": "zero",
        "tensor_artifact": {
            "path": sidecar.name,
            "format": "torch-save-weights-only",
            "size_bytes": sidecar.stat().st_size,
            "sha256": "0" * 64,
            "record_count": 9,
        },
    }
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(COMPARATOR.torch, "load", forbidden_load)
    with pytest.raises(AssertionError, match="SHA-256"):
        COMPARATOR.load_sidecar(raw_path, value, [], root)
    assert not called


def test_tensor_hash_domain_separates_shape():
    flat = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    matrix = flat.reshape(2, 2)
    assert flat.view(torch.uint8).numpy().tobytes() == matrix.view(torch.uint8).numpy().tobytes()
    assert COMPARATOR.tensor_sha256(flat) != COMPARATOR.tensor_sha256(matrix)


def tensor_records():
    records = []
    for index, record_id in enumerate(COMPARATOR.RECORD_IDS):
        logits = torch.tensor(
            [[index, index + 1, -index, 0.5, -0.5]],
            dtype=torch.bfloat16,
        )
        probabilities = torch.zeros((1, 5), dtype=torch.float32)
        probabilities.scatter_(
            -1,
            logits.argmax(dim=-1, keepdim=True),
            1.0,
        )
        records.append(
            {
                "record_id": record_id,
                "logits": logits,
                "probabilities": probabilities,
            }
        )
    return records


def write_cache_fixture(root, *, mutate_sidecar_record=False):
    root.mkdir()
    raw_path = root / "result.json"
    tensor_path = root / "result.tensors.pt"
    sidecar_records = tensor_records()
    raw_records = []
    for record in sidecar_records:
        logits = record["logits"]
        probabilities = record["probabilities"]
        raw_records.append(
            {
                "record_id": record["record_id"],
                "logits_sha256": COMPARATOR.tensor_sha256(logits),
                "probabilities_sha256": COMPARATOR.tensor_sha256(probabilities),
                "logits_dtype": "torch.bfloat16",
                "probabilities_dtype": "torch.float32",
                "logits_shape": [1, 5],
                "probabilities_shape": [1, 5],
                "logits_finite": True,
                "probabilities_finite": True,
                "probability_row_sums": [1.0],
                "token_ids": probabilities.argmax(dim=-1).tolist(),
            }
        )
    if mutate_sidecar_record:
        sidecar_records[1]["logits"][0, 0] += 1
    EVIDENCE.write_torch_exclusive(
        tensor_path,
        {
            "schema": COMPARATOR.INPUT_SCHEMA,
            "mode": "eager",
            "draft_cache_fill": "zero",
            "records": sidecar_records,
        },
    )

    route = {
        "schema": "draft-discard-v1",
        "execution_mode": "eager_dynamic",
        "batch_bucket": 1,
        "effective_k": 3,
        "catchup_family": "paged_eager_dynamic_v1",
        "sampler_envelope": "exact_all_compositions_worst_case_v1",
    }

    def cycle(offset, positions, catchup, *, boundary=False):
        selected = raw_records[offset : offset + 3]
        proposal = [record["token_ids"][0] for record in selected]
        target_table = [0] if boundary else [1, 2]
        reservation_table = [0, 1] if boundary else list(target_table)
        value = {
            "route": route,
            "effective_k": 3,
            "catchup_tokens": catchup,
            "proposal_input_positions": positions,
            "proposed_token_ids": [proposal],
            "target_token_ids": [proposal[0]],
            "graph_decode_steps": 0,
            "eager_decode_steps": 3,
            "sampler_records": selected,
            "rng_before_draft": {"cpu": "a" * 64, "cuda": "b" * 64},
            "rng_after_draft": {"cpu": "a" * 64, "cuda": "b" * 64},
            "filled_physical_blocks": sorted(set(reservation_table)),
            "tables_after_target_schedule": [target_table],
            "tables_during_reservation": [reservation_table],
            "tables_after_handoff": [target_table],
            "temporary_blocks": 1 if boundary else 0,
        }
        return value

    raw = {
        "schema": COMPARATOR.INPUT_SCHEMA,
        "mode": "eager",
        "seed": 20260828,
        "draft_cache_fill": "zero",
        "model": "/model",
        "draft_model": "/model",
        "configuration": {
            "num_speculative_tokens": 3,
            "gpu_memory_utilization": 0.5,
            "max_model_len": 512,
            "max_num_batched_tokens": 512,
            "max_num_seqs": 1,
            "tensor_parallel_size": 1,
            "top_p_backend": "exact",
        },
        "workload": {
            "neutral_boundary_warmup": {
                "length": 255,
                "salt": 911,
                "fill": "zero",
                "proposal_positions": [255, 256, 257],
                "sampler_records_discarded": 3,
            },
            "boundary_prompt": {
                "length": 255,
                "salt": 31,
                "proposal_positions": [255, 256, 257],
            },
            "shared_prefix_prompt": {
                "length": 257,
                "salt": 73,
                "proposal_positions": [257, 258, 259],
            },
            "sampling": {
                "temperature": 0.0,
                "top_k": -1,
                "top_p": 1.0,
                "ignore_eos": True,
            },
        },
        "measurement_warmup": {
            "fill": "zero",
            "prompt_length": 255,
            "prompt_salt": 911,
            "proposal_input_positions": [255, 256, 257],
            "catchup_tokens": 255,
            "temporary_blocks": 1,
            "proposed_token_ids": [[1, 2, 3]],
            "target_token_ids": [1],
            "sampler_records_discarded": 3,
        },
        "boundary": cycle(0, [255, 256, 257], 255, boundary=True),
        "shared_prefix": {
            "cold": cycle(3, [257, 258, 259], 257),
            "shared_prefix": cycle(6, [257, 258, 259], 257),
        },
        "all_logits_finite": True,
        "all_probabilities_finite": True,
        "retention_eligible": True,
        "provenance": {"retention_eligible": True},
        "tensor_artifact": {
            "path": tensor_path.name,
            "format": "torch-save-weights-only",
            "size_bytes": tensor_path.stat().st_size,
            "sha256": EVIDENCE.sha256_file(tensor_path),
            "record_count": 9,
        },
    }
    EVIDENCE.write_json_exclusive(raw_path, raw)
    return raw_path


def test_cache_fixture_loads_with_portable_exact_record_binding(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(COMPARATOR, "EXPECTED_VOCAB_SIZE", 5)
    raw_path = write_cache_fixture(tmp_path / "fixture")

    raw, tensors, descriptor = COMPARATOR.load_artifact(
        raw_path,
        "zero",
        raw_path.parent,
    )

    assert raw["schema"] == COMPARATOR.INPUT_SCHEMA
    assert [record["record_id"] for record in tensors] == list(
        COMPARATOR.RECORD_IDS
    )
    assert descriptor["path"] == "result.json"
    assert descriptor["tensor_sidecar"]["path"] == "result.tensors.pt"


def test_sidecar_records_are_bound_to_raw_json_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(COMPARATOR, "EXPECTED_VOCAB_SIZE", 5)
    raw_path = write_cache_fixture(
        tmp_path / "fixture",
        mutate_sidecar_record=True,
    )

    with pytest.raises(AssertionError, match="logits do not match JSON hash"):
        COMPARATOR.load_artifact(raw_path, "zero", raw_path.parent)


def test_cache_poison_comparison_requires_bitwise_logits_equality():
    zero = tensor_records()
    nan = tensor_records()
    nan[0]["logits"][0, 0] = torch.nextafter(
        nan[0]["logits"][0, 0],
        torch.tensor(float("inf"), dtype=torch.bfloat16),
    )
    assert float((zero[0]["logits"].float() - nan[0]["logits"].float()).abs().max()) < 0.03125

    with pytest.raises(AssertionError, match="zero/NaN logits differ"):
        COMPARATOR.compare_tensor_records(zero, nan)


def test_guarded_draft_interval_restores_stance_on_success_and_error(
    monkeypatch, capsys
):
    stances = []
    monkeypatch.setattr(
        ROUTE_RUNNER.torch.compiler,
        "set_stance",
        stances.append,
    )

    with ROUTE_RUNNER.guarded_draft_interval("batch=1,bucket=1,k=1,catchup=4"):
        pass
    assert stances == ["fail_on_recompile", "default"]
    stderr = capsys.readouterr().err
    assert stderr.splitlines() == [
        "V3_DRAFT_INTERVAL_BEGIN batch=1,bucket=1,k=1,catchup=4",
        "V3_DRAFT_INTERVAL_END batch=1,bucket=1,k=1,catchup=4",
    ]

    stances.clear()
    with pytest.raises(RuntimeError, match="draft failed"):
        with ROUTE_RUNNER.guarded_draft_interval(
            "batch=2,bucket=2,k=2,catchup=0"
        ):
            raise RuntimeError("draft failed")
    assert stances == ["fail_on_recompile", "default"]
    stderr = capsys.readouterr().err
    assert "V3_DRAFT_INTERVAL_END batch=2,bucket=2,k=2,catchup=0" in stderr
