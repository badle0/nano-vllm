import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.chunked_prefill_tail.aggregate_certification import (
    LoadedArtifact,
    aggregate_artifacts,
    load_artifact,
    validate_run_payload,
    write_archive,
)
from benchmarks.chunked_prefill_tail.common import model_identity
from benchmarks.chunked_prefill_tail.full_completion_cert import (
    KIND,
    PROTOCOL,
    _run_workload,
    _validate_args as validate_run_args,
    build_parser as build_run_parser,
    certification_workload,
)


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeCuda:
    def __init__(self):
        self.seed = None

    def manual_seed_all(self, seed):
        self.seed = seed

    def synchronize(self):
        pass

    def reset_peak_memory_stats(self):
        pass

    def memory_allocated(self):
        return 100

    def memory_reserved(self):
        return 200

    def max_memory_allocated(self):
        return 300

    def max_memory_reserved(self):
        return 400


class FakeTorch:
    def __init__(self):
        self.seed = None
        self.cuda = FakeCuda()

    def manual_seed(self, seed):
        self.seed = seed


class FakeEngine:
    def __init__(self, tau=256, max_itl_ms=5.0):
        config = SimpleNamespace(
            max_num_batched_tokens=tau,
            max_num_seqs=min(512, tau),
            max_model_len=4096,
            gpu_memory_utilization=0.8,
            enforce_eager=False,
            tensor_parallel_size=1,
            disable_python_gc=True,
            kvcache_block_size=256,
            num_kvcache_blocks=1000,
            hf_config=SimpleNamespace(vocab_size=20_000),
        )
        self.model_runner = SimpleNamespace(
            config=config,
            varlen_miss=0,
            varlen_graphs={(256, 64): object()},
            graph_bs=[1, 2, 4, 8, 16, 32, 64],
        )
        self.scheduler = SimpleNamespace(
            block_manager=SimpleNamespace(
                free_block_ids=list(range(1000)),
                used_block_ids=[],
            )
        )
        self.max_itl_s = max_itl_ms / 1000.0
        self.added_prompts = []
        self.warmup_prompts = None
        self.long_admitted = False
        self.finished = False
        self.steps = 0

    def generate(self, prompts, params, use_tqdm):
        assert use_tqdm is False
        assert params.ignore_eos is True
        assert params.max_tokens == 4
        self.warmup_prompts = prompts
        return [{"token_ids": [1, 2, 3, 4]} for _ in prompts]

    def add_request(self, prompt, params):
        assert params.temperature == 0.6
        assert params.max_tokens == 256
        assert params.ignore_eos is True
        assert params.top_k == -1
        assert params.top_p == 1.0
        seq_id = len(self.added_prompts)
        self.added_prompts.append(prompt)
        if len(self.added_prompts) == 18:
            self.long_admitted = True
        return seq_id

    def _step(self):
        self.steps += 1
        if self.long_admitted:
            return SimpleNamespace(num_prefill_tokens=240, num_decode_tokens=16)
        if self.steps == 1:
            return SimpleNamespace(num_prefill_tokens=256, num_decode_tokens=0)
        return SimpleNamespace(num_prefill_tokens=0, num_decode_tokens=16)

    def step_with_metrics(self):
        step_output = self._step()
        outputs = []
        if self.long_admitted:
            itls = [self.max_itl_s] * 255
            for seq_id in range(18):
                prompt_tokens = 64 if seq_id < 16 else 2048
                token_ids = [1000 + seq_id] * 256
                ttft = 0.010 if seq_id < 16 else 0.050
                e2e = ttft + sum(itls)
                outputs.append((seq_id, token_ids, {
                    "engine_queue_time": 0.001,
                    "engine_ttft": ttft,
                    "engine_e2e": e2e,
                    "engine_mean_itl": self.max_itl_s,
                    "engine_max_itl": self.max_itl_s,
                    "engine_itls": itls,
                    "submission_to_engine": 0.0,
                    "submission_to_first_token": ttft,
                    "submission_to_engine_finish": e2e,
                    "num_prompt_tokens": prompt_tokens,
                    "num_completion_tokens": 256,
                }))
            self.finished = True
        signed = (
            step_output.num_prefill_tokens
            if step_output.num_prefill_tokens
            else -step_output.num_decode_tokens
        )
        return outputs, signed

    def is_finished(self):
        return self.finished


class FakeClock:
    def __init__(self):
        self.values = iter((0.0, 1.0, 2.0))

    def __call__(self):
        return next(self.values)


def fake_pin(commit="a" * 40, source_marker="base"):
    source_digest = hashlib.sha256(source_marker.encode()).hexdigest()
    aggregate = hashlib.sha256()
    aggregate.update(b"source.py\0")
    aggregate.update(str(len(source_marker)).encode("ascii"))
    aggregate.update(b"\0")
    aggregate.update(bytes.fromhex(source_digest))
    return {
        "git": {
            "branch": "fix/chunked-prefill-tail",
            "clean": True,
            "commit": commit,
            "status": [],
            "tree": "c" * 40,
        },
        "source": {
            "aggregate_sha256": aggregate.hexdigest(),
            "algorithm": "sha256(path\\0size\\0sha256(file))",
            "file_count": 1,
            "files": [{
                "path": "source.py",
                "bytes": len(source_marker),
                "sha256": source_digest,
            }],
        },
    }


def fake_environment():
    return {
        "python": "3.12.0",
        "platform": "test",
        "torch": "2.10.0+cu128",
        "cuda_build": "12.8",
        "transformers": "4.test",
        "flash_attn": "2.test",
        "gpu": "NVIDIA A100-SXM4-40GB",
        "gpu_total_memory_bytes": 40 * 2**30,
        "compute_capability": [8, 0],
        "driver": "test",
        "python_gc_enabled": True,
        "environment_variables": {
            "CUDA_VISIBLE_DEVICES": None,
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/test",
        },
    }


def build_payload(tmp_path, seed, tau=256, max_itl_ms=5.0):
    engine = FakeEngine(tau=tau, max_itl_ms=max_itl_ms)
    run = _run_workload(
        engine,
        FakeSamplingParams,
        FakeTorch(),
        seed,
        clock=FakeClock(),
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text("{}\n")
    (model_dir / "weights.bin").write_bytes(b"weights")
    pin = fake_pin()
    output = f"run-{seed}.json"
    payload = {
        "schema_version": 1,
        "kind": KIND,
        "protocol": PROTOCOL,
        "argv": [
            "python",
            "full_completion_cert.py",
            "--model",
            str(model_dir),
            "--tau",
            str(tau),
            "--seed",
            str(seed),
            "--expected-commit",
            "a" * 40,
            "--expected-source-sha256",
            pin["source"]["aggregate_sha256"],
            "--output",
            output,
        ],
        "arguments": {
            "model": str(model_dir),
            "tau": tau,
            "seed": seed,
            "expected_commit": "a" * 40,
            "expected_source_sha256": pin["source"]["aggregate_sha256"],
            "output": output,
        },
        "randomness": {
            "seed": seed,
            "python_random_seed": seed,
            "torch_manual_seed": seed,
            "torch_cuda_manual_seed_all": seed,
            "torch_deterministic_algorithms_enabled": False,
        },
        "provenance": pin,
        "provenance_after_run": deepcopy(pin),
        "model": model_identity(model_dir),
        "model_after_run_matches": True,
        "environment": fake_environment(),
        "engine": {
            "initialization_ms": 1.0,
            "config": {
                "max_num_batched_tokens": tau,
                "max_num_seqs": min(512, tau),
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.8,
                "enforce_eager": False,
                "tensor_parallel_size": 1,
                "disable_python_gc": True,
                "kvcache_block_size": 256,
                "num_kvcache_blocks": 1000,
            },
        },
        "python_gc": {
            "disable_requested": True,
            "enabled_before_engine": True,
            "enabled_after_engine_init": False,
            "enabled_after_engine_exit": True,
        },
        **run,
    }
    return payload, engine


def artifacts_for(tmp_path, tau, max_itls):
    artifacts = []
    for offset, max_itl in enumerate(max_itls):
        seed = 20260826 + offset
        payload, _ = build_payload(tmp_path, seed, tau=tau, max_itl_ms=max_itl)
        raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        artifacts.append(LoadedArtifact(
            path=tmp_path / f"tau{tau}-seed{seed}.json",
            raw=raw,
            sha256=hashlib.sha256(raw).hexdigest(),
            payload=payload,
        ))
    return artifacts


def test_cpu_fake_runs_the_exact_protocol_and_retains_raw_metrics(tmp_path):
    payload, engine = build_payload(tmp_path, seed=20260826)
    facts = validate_run_payload(payload, 256)

    assert len(engine.warmup_prompts) == 16
    assert all(len(prompt) == 64 for prompt in engine.warmup_prompts)
    assert [len(prompt) for prompt in engine.added_prompts[:16]] == [64] * 16
    assert [len(prompt) for prompt in engine.added_prompts[16:]] == [2048] * 2
    assert len(payload["requests"]) == 18
    assert len(payload["requests"][0]["metrics"]["engine_itls"]) == 255
    assert payload["summary"]["total_completion_tokens"] == 18 * 256
    assert payload["summary"]["completion_tokens_per_s"] == 18 * 256 / 2.0
    assert payload["summary"]["single_run_latency_certified"] is False
    assert payload["telemetry"]["routing"]["per_step_routes_observed"] is False
    assert facts["interactive_max_itl_ms"] == pytest.approx(5.0)


def test_tau256_requires_all_five_strict_passes(tmp_path):
    passing = artifacts_for(tmp_path, 256, [5.0, 6.0, 7.0, 8.0, 9.999])
    certified = aggregate_artifacts(passing, 256, fake_pin(), ["aggregate"])
    assert certified["policy"]["latency_certified"] is True
    assert certified["policy"]["classification"] == (
        "latency_certified_current_self_pinned"
    )
    assert certified["policy"]["runs_meeting_slo"] == 5

    failing = artifacts_for(tmp_path, 256, [5.0, 6.0, 7.0, 8.0, 10.0])
    rejected = aggregate_artifacts(failing, 256, fake_pin(), ["aggregate"])
    assert rejected["policy"]["latency_certified"] is False
    assert rejected["policy"]["classification"] == "latency_not_certified"
    assert rejected["policy"]["runs_meeting_slo"] == 4


def test_tau512_and_phase_diagnostics_never_certify(tmp_path):
    artifacts = artifacts_for(tmp_path, 512, [5.0] * 5)
    aggregate = aggregate_artifacts(artifacts, 512, fake_pin(), ["aggregate"])
    assert aggregate["policy"]["all_five_runs_meet_slo"] is True
    assert aggregate["policy"]["latency_certified"] is False
    assert aggregate["policy"]["tau512_can_certify"] is False
    assert aggregate["policy"]["classification"] == (
        "throughput_ttft_only_not_latency_certified"
    )

    phase = deepcopy(artifacts[0].payload)
    phase["kind"] = "chunked_prefill_tail_step_diagnostic"
    with pytest.raises(ValueError, match="not a full-completion run"):
        validate_run_payload(phase, 512)


def test_aggregation_rejects_nonfresh_or_mismatched_evidence(tmp_path):
    artifacts = artifacts_for(tmp_path, 256, [5.0] * 5)
    duplicate_seed = deepcopy(artifacts)
    duplicate_seed[-1] = LoadedArtifact(
        path=duplicate_seed[-1].path,
        raw=duplicate_seed[-1].raw + b" ",
        sha256=hashlib.sha256(duplicate_seed[-1].raw + b" ").hexdigest(),
        payload=deepcopy(duplicate_seed[0].payload),
    )
    with pytest.raises(ValueError, match="distinct fresh seeds"):
        aggregate_artifacts(duplicate_seed, 256, fake_pin(), ["aggregate"])

    different_pin = fake_pin(source_marker="different")
    artifacts[-1].payload["provenance"]["source"] = deepcopy(
        different_pin["source"]
    )
    artifacts[-1].payload["provenance_after_run"]["source"] = deepcopy(
        different_pin["source"]
    )
    artifacts[-1].payload["arguments"]["expected_source_sha256"] = (
        different_pin["source"]["aggregate_sha256"]
    )
    source_option = artifacts[-1].payload["argv"].index(
        "--expected-source-sha256"
    )
    artifacts[-1].payload["argv"][source_option + 1] = (
        different_pin["source"]["aggregate_sha256"]
    )
    with pytest.raises(ValueError, match="current clean HEAD"):
        aggregate_artifacts(artifacts, 256, fake_pin(), ["aggregate"])


def test_aggregation_requires_the_current_git_tree_pin(tmp_path):
    artifacts = artifacts_for(tmp_path, 256, [5.0] * 5)
    for artifact in artifacts:
        artifact.payload["provenance"]["git"]["tree"] = "d" * 40
        artifact.payload["provenance_after_run"]["git"]["tree"] = "d" * 40

    with pytest.raises(ValueError, match="current clean HEAD"):
        aggregate_artifacts(artifacts, 256, fake_pin(), ["aggregate"])


def test_readonly_loading_and_archive_are_no_overwrite(tmp_path):
    artifacts = artifacts_for(tmp_path, 256, [5.0] * 5)
    loaded = []
    for artifact in artifacts:
        artifact.path.write_bytes(artifact.raw)
        artifact.path.chmod(0o444)
        loaded.append(load_artifact(artifact.path))
    aggregate = aggregate_artifacts(loaded, 256, fake_pin(), ["aggregate"])
    archive = tmp_path / "archive"
    write_archive(archive, aggregate, loaded)

    marker_path = tmp_path / "archive.COMPLETE"
    assert marker_path.is_file()
    assert len(list((archive / "runs").glob("*.json"))) == 5
    assert (archive.stat().st_mode & 0o222) == 0
    assert all((path.stat().st_mode & 0o222) == 0 for path in archive.rglob("*"))
    manifest = json.loads((archive / "manifest.json").read_text())
    for row in manifest["files"]:
        retained = archive / row["path"]
        raw = retained.read_bytes()
        assert len(raw) == row["bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    marker = json.loads(marker_path.read_text())
    aggregate_raw = (archive / "aggregate.json").read_bytes()
    assert marker["aggregate_sha256"] == hashlib.sha256(aggregate_raw).hexdigest()
    assert (marker_path.stat().st_mode & 0o222) == 0
    with pytest.raises(FileExistsError):
        write_archive(archive, aggregate, loaded)


def test_archive_failure_does_not_publish_complete_marker(tmp_path, monkeypatch):
    artifacts = artifacts_for(tmp_path, 256, [5.0] * 5)
    archive = tmp_path / "broken-archive"
    original_chmod = Path.chmod

    def fail_target_chmod(path, mode, *args, **kwargs):
        if path == archive:
            raise OSError("injected chmod failure")
        return original_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", fail_target_chmod)
    with pytest.raises(OSError, match="injected chmod failure"):
        write_archive(archive, {"policy": {}}, artifacts)
    assert not (tmp_path / "broken-archive.COMPLETE").exists()


def test_loader_rejects_mutable_and_duplicate_key_json(tmp_path):
    mutable = tmp_path / "mutable.json"
    mutable.write_text("{}\n")
    with pytest.raises(ValueError, match="immutable/read-only"):
        load_artifact(mutable)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"kind": 1, "kind": 2}\n')
    duplicate.chmod(0o444)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_artifact(duplicate)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(argv=["x"]), "direct full-completion"),
        (
            lambda payload: payload["randomness"].update(
                torch_deterministic_algorithms_enabled=True
            ),
            "deterministic-algorithms",
        ),
        (lambda payload: payload["telemetry"].update(memory={}), "memory telemetry"),
        (
            lambda payload: payload["requests"][0]["metrics"].update(
                engine_e2e=999.0,
                submission_to_engine_finish=999.0,
            ),
            "engine_e2e",
        ),
    ],
)
def test_validator_rejects_incomplete_or_inconsistent_evidence(
    tmp_path, mutate, message
):
    payload, _ = build_payload(tmp_path, seed=20260826)
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        validate_run_payload(payload, 256)


def test_run_output_must_be_outside_model_directory(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    args = build_run_parser().parse_args([
        "--model",
        str(model),
        "--tau",
        "256",
        "--seed",
        "1",
        "--expected-commit",
        "a" * 40,
        "--expected-source-sha256",
        "b" * 64,
        "--output",
        str(model / "run.json"),
    ])
    with pytest.raises(ValueError, match="outside the pinned model"):
        validate_run_args(args)


def test_historical_protocol_pins_fixed_harness_max_num_seqs():
    assert certification_workload(256)["max_num_seqs"] == 256
    assert certification_workload(512)["max_num_seqs"] == 512
