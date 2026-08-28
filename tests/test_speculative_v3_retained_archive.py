import copy
import importlib.util
import io
import os
import pickle
import sys
import warnings
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "speculative_v3"
    / "validate_retained_evidence.py"
)
ARCHIVE = (
    REPO_ROOT
    / "benchmarks"
    / "speculative_v3"
    / "evidence"
    / "2026-08-28-a100-v3-e8e0452"
)

SPEC = importlib.util.spec_from_file_location(
    "validate_speculative_v3_retained_evidence",
    VALIDATOR_PATH,
)
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


def strict_document(relative):
    return VALIDATOR.load_strict_json_bytes(
        (ARCHIVE / relative).read_bytes(),
        relative,
    )


def raw_cache_records(artifact, mode):
    records = []
    specs = (
        ("boundary", ("boundary",), [255, 256, 257], 255),
        (
            "shared-prefix/cold",
            ("shared_prefix", "cold"),
            [257, 258, 259],
            257,
        ),
        (
            "shared-prefix/hit",
            ("shared_prefix", "shared_prefix"),
            [257, 258, 259],
            257,
        ),
    )
    for index, (name, path, positions, catchup) in enumerate(specs):
        records.extend(
            VALIDATOR._validate_cache_cycle(
                artifact,
                mode=mode,
                cycle_name=name,
                path=path,
                positions=positions,
                catchup=catchup,
                offset=index * 3,
            )
        )
    return records


def test_actual_archive_validates_without_cuda_torch_or_model():
    report = VALIDATOR.validate_archive(ARCHIVE)

    assert report["archive_id"] == VALIDATOR.ARCHIVE_ID
    assert report["artifact_count"] == 21
    assert report["route_intervals"] == {"eager": 32, "graph": 32}
    assert report["cache_records"] == {"eager": 9, "graph": 9}
    assert report["cuda_required"] is False
    assert report["model_locality_required"] is False


def test_manifest_is_exact_generated_release_index():
    manifest = strict_document("manifest.json")

    assert manifest == VALIDATOR.expected_manifest()
    assert [row["path"] for row in manifest["artifacts"]] == sorted(
        VALIDATOR.TRUSTED_PAYLOADS
    )
    assert len({row["sha256"] for row in manifest["artifacts"]}) == 21


@pytest.mark.parametrize(
    "payload,match",
    (
        (b'{"value": 1, "value": 2}', "duplicate JSON key"),
        (b'{"value": NaN}', "non-finite JSON number"),
        (b'{"value": Infinity}', "non-finite JSON number"),
        (b"[]", "JSON root must be an object"),
    ),
)
def test_strict_json_rejects_ambiguous_or_nonfinite_inputs(payload, match):
    with pytest.raises(VALIDATOR.EvidenceValidationError, match=match):
        VALIDATOR.load_strict_json_bytes(payload, "hostile.json")


def test_registered_reader_rejects_symlink_and_hardlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    payload = b"registered"
    registered = (len(payload), VALIDATOR._sha256_bytes(payload))
    ordinary = root / "ordinary.bin"
    ordinary.write_bytes(payload)
    assert VALIDATOR._read_registered(root, "ordinary.bin", registered) == payload

    link = root / "link.bin"
    link.symlink_to(ordinary)
    with pytest.raises(VALIDATOR.EvidenceValidationError, match="not regular"):
        VALIDATOR._read_registered(root, "link.bin", registered)

    hardlink = root / "hardlink.bin"
    os.link(ordinary, hardlink)
    with pytest.raises(VALIDATOR.EvidenceValidationError, match="hard-linked"):
        VALIDATOR._read_registered(root, "ordinary.bin", registered)


def test_archive_rejects_unregistered_file_set_before_parsing(tmp_path):
    root = tmp_path / VALIDATOR.ARCHIVE_ID
    root.mkdir()
    (root / "unregistered.txt").write_text("not evidence")

    with pytest.raises(
        VALIDATOR.EvidenceValidationError,
        match="file set differs",
    ):
        VALIDATOR.validate_archive(root)


class _PickleExploit:
    def __reduce__(self):
        return os.system, ("this must never execute",)


def test_sidecar_pickle_scanner_rejects_callable_global_without_execution():
    payload = pickle.dumps(_PickleExploit(), protocol=2)

    with pytest.raises(
        VALIDATOR.EvidenceValidationError,
        match="disallowed sidecar pickle global",
    ):
        VALIDATOR._scan_pickle(payload)


def test_sidecar_zip_rejects_duplicate_member_names():
    stream = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("archive/data.pkl", b"one")
            archive.writestr("archive/data.pkl", b"two")

    with pytest.raises(
        VALIDATOR.EvidenceValidationError,
        match="duplicate names",
    ):
        VALIDATOR._validate_pt_sidecar(
            stream.getvalue(),
            mode="eager",
            fill="zero",
            raw_records=[],
        )


def test_sidecar_is_decoded_model_free_and_bound_per_tensor():
    artifact = strict_document("raw/cache-eager-zero.json")
    raw_records = raw_cache_records(artifact, "eager")
    sidecar = (ARCHIVE / "raw/cache-eager-zero.tensors.pt").read_bytes()

    records = VALIDATOR._validate_pt_sidecar(
        sidecar,
        mode="eager",
        fill="zero",
        raw_records=raw_records,
    )
    assert [row["record_id"] for row in records] == list(
        VALIDATOR.CACHE_RECORD_IDS
    )
    assert len(records[0]["logits"]) == VALIDATOR.VOCAB_SIZE * 2
    assert len(records[0]["probabilities"]) == VALIDATOR.VOCAB_SIZE * 4

    corrupted = copy.deepcopy(raw_records)
    corrupted[0]["logits_sha256"] = "0" * 64
    with pytest.raises(
        VALIDATOR.EvidenceValidationError,
        match="logits hash does not bind sidecar bytes",
    ):
        VALIDATOR._validate_pt_sidecar(
            sidecar,
            mode="eager",
            fill="zero",
            raw_records=corrupted,
        )


def test_route_log_rejects_output_inside_run_bound_interval():
    labels = [f"record={index}" for index in range(VALIDATOR.ROUTE_RECORD_COUNT)]
    valid = "".join(
        f"{VALIDATOR.BEGIN_PREFIX}{label}\n{VALIDATOR.END_PREFIX}{label}\n"
        for label in labels
    ).encode()
    assert VALIDATOR._validate_route_log(valid, labels)["interval_count"] == 32

    hostile = valid.replace(
        f"{VALIDATOR.BEGIN_PREFIX}{labels[0]}\n".encode(),
        f"{VALIDATOR.BEGIN_PREFIX}{labels[0]}\n[__recompiles] hostile\n".encode(),
        1,
    )
    with pytest.raises(
        VALIDATOR.EvidenceValidationError,
        match="unexpected output inside route interval",
    ):
        VALIDATOR._validate_route_log(hostile, labels)


def test_route_semantic_mutation_fails_before_provenance_acceptance():
    artifact = strict_document("raw/route-eager.json")
    artifact["seed"] += 1

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="seed drifted"):
        VALIDATOR._validate_route_artifact(
            artifact,
            mode="eager",
            historical_sources={},
            runtime_blobs={},
        )


def test_cache_semantic_mutation_fails_before_sidecar_acceptance():
    artifact = strict_document("raw/cache-eager-zero.json")
    artifact["configuration"]["max_num_seqs"] = 2

    with pytest.raises(
        VALIDATOR.EvidenceValidationError,
        match="cache configuration drifted",
    ):
        VALIDATOR._validate_cache_artifact(
            artifact,
            mode="eager",
            fill="zero",
            sidecar_payload=b"not used",
            sidecar_registry=(0, "0" * 64),
            historical_sources={},
            runtime_blobs={},
        )


def test_output_semantic_mutation_fails_before_provenance_acceptance():
    artifact = strict_document("raw/output-graph-on.json")
    artifact["draft_phase_calls"]["capture_draft_cudagraph"] = 1

    with pytest.raises(
        VALIDATOR.EvidenceValidationError,
        match="draft phase counts drifted",
    ):
        VALIDATOR._validate_output_artifact(
            artifact,
            mode="graph",
            side="on",
            historical_sources={},
            runtime_blobs={},
        )


def test_historical_git_reads_ignore_hostile_repository_environment(monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/definitely/not/the/repository")
    monkeypatch.setenv("GIT_WORK_TREE", "/also/not/the/repository")
    monkeypatch.setenv("GIT_NAMESPACE", "hostile")

    identities = VALIDATOR._validate_historical_sources(REPO_ROOT)

    assert set(identities) == set(VALIDATOR.HISTORICAL_SOURCE_HASHES)
    assert all(len(row["blob"]) == 40 for row in identities.values())


def test_validator_source_has_no_torch_or_nanovllm_import():
    source = VALIDATOR_PATH.read_text()

    assert "import torch" not in source
    assert "from torch" not in source
    assert "import nanovllm" not in source
    assert "from nanovllm" not in source
