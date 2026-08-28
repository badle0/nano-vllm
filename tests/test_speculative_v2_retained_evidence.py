import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "speculative_v2"
    / "validate_retained_evidence.py"
)
ARCHIVE_ROOT = (
    REPO_ROOT
    / "benchmarks"
    / "speculative_v2"
    / "evidence"
    / "2026-08-28-a100-v2-d87f168"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_speculative_v2_retained_evidence",
    VALIDATOR_PATH,
)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def _copy_archive(tmp_path):
    destination = tmp_path / "archive"
    shutil.copytree(ARCHIVE_ROOT, destination)
    return destination


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, payload):
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _refresh_artifact_row(archive, relative):
    artifact = archive / relative
    manifest_path = archive / "manifest.json"
    manifest = _load(manifest_path)
    row = next(row for row in manifest["artifacts"] if row["path"] == relative)
    payload = artifact.read_bytes()
    row["size_bytes"] = len(payload)
    row["sha256"] = hashlib.sha256(payload).hexdigest()
    _write(manifest_path, manifest)


def _identity(path):
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _trust_manifest_for_semantic_test(monkeypatch, archive):
    """Test-only escape hatch: keep public validation pinned to release bytes."""
    monkeypatch.setattr(
        VALIDATOR,
        "TRUSTED_MANIFEST",
        _identity(archive / "manifest.json"),
    )


def _trust_mutated_artifacts_for_semantic_test(
    monkeypatch,
    archive,
    *relatives,
):
    """Re-register selected fixture bytes so a test reaches semantic checks."""
    trusted = dict(VALIDATOR.TRUSTED_RAW_ARTIFACTS)
    for relative in relatives:
        trusted[relative] = _identity(archive / relative)
    monkeypatch.setattr(VALIDATOR, "TRUSTED_RAW_ARTIFACTS", trusted)
    _trust_manifest_for_semantic_test(monkeypatch, archive)


def _mutate_json_artifact(archive, relative, mutation, *, monkeypatch):
    path = archive / relative
    payload = _load(path)
    mutation(payload)
    _write(path, payload)
    _refresh_artifact_row(archive, relative)
    _trust_mutated_artifacts_for_semantic_test(monkeypatch, archive, relative)


def test_real_archive_validates_from_manifest_or_directory():
    from_manifest = VALIDATOR.validate_archive(ARCHIVE_ROOT / "manifest.json")
    from_directory = VALIDATOR.validate_archive(ARCHIVE_ROOT)

    assert from_manifest == from_directory
    assert from_manifest["artifact_count"] == 12
    assert from_manifest["lifecycle_modes"] == ["eager", "graph"]
    assert set(from_manifest["recovery_phases"]) == set(
        VALIDATOR.RECOVERY_PHASE_MODES
    )
    assert from_manifest["claim_boundary"]["memory_workspace_gpu_certified"] is False


def test_byte_mutation_without_manifest_update_is_rejected(tmp_path):
    archive = _copy_archive(tmp_path)
    artifact = archive / "raw" / "v2-eager.json"
    artifact.write_bytes(artifact.read_bytes() + b" ")

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="size mismatch"):
        VALIDATOR.validate_archive(archive / "manifest.json")


def test_duplicate_artifact_json_key_is_rejected_even_with_new_outer_hash(
    tmp_path,
    monkeypatch,
):
    archive = _copy_archive(tmp_path)
    relative = "raw/v2-eager.json"
    artifact = archive / relative
    text = artifact.read_text(encoding="utf-8")
    text = text.replace(
        "{\n",
        '{\n  "certification_mode": "retained",\n',
        1,
    )
    artifact.write_text(text, encoding="utf-8")
    _refresh_artifact_row(archive, relative)
    _trust_mutated_artifacts_for_semantic_test(
        monkeypatch,
        archive,
        relative,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="duplicate JSON key"):
        VALIDATOR.validate_archive(archive)


def test_nonfinite_artifact_number_is_rejected_even_with_new_outer_hash(
    tmp_path,
    monkeypatch,
):
    archive = _copy_archive(tmp_path)
    relative = "raw/v2-eager.json"
    artifact = archive / relative
    text = artifact.read_text(encoding="utf-8")
    assert '"configured_k": 2' in text
    artifact.write_text(
        text.replace('"configured_k": 2', '"configured_k": NaN', 1),
        encoding="utf-8",
    )
    _refresh_artifact_row(archive, relative)
    _trust_mutated_artifacts_for_semantic_test(
        monkeypatch,
        archive,
        relative,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="non-finite JSON"):
        VALIDATOR.validate_archive(archive)


def test_exact_n_plus_one_semantics_are_not_replaceable_by_a_larger_miss(
    tmp_path,
    monkeypatch,
):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        payload["first_ineligible_blocks_rejected"] = (
            payload["explicit_selected_blocks_passed"] + 2
        )

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match=r"not N\+1"):
        VALIDATOR.validate_archive(archive)


def test_workspace_arithmetic_mutation_is_rejected_after_rehash(
    tmp_path,
    monkeypatch,
):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        payload["speculative_memory_audit"]["workspace_plan"][
            "probability_floor_bytes"
        ] += 4

    _mutate_json_artifact(
        archive,
        "raw/v2-graph.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="planner arithmetic"):
        VALIDATOR.validate_archive(archive)


def test_foreign_endpoint_consumer_is_rejected_after_rehash(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        payload["provenance"]["gpu"]["before"]["foreign_compute_apps"] = [
            {
                "gpu_uuid": "GPU-foreign",
                "pid": 999999,
                "used_memory_mib": 1,
            }
        ]

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="foreign GPU consumers"):
        VALIDATOR.validate_archive(archive)


def test_recovery_phase_relabel_is_rejected_after_rehash(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    _mutate_json_artifact(
        archive,
        "raw/recovery-draft_load.json",
        lambda payload: payload.__setitem__("phase", "draft_warmup"),
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="recovery phase mismatch"):
        VALIDATOR.validate_archive(archive)


def test_recovery_global_cache_ceiling_is_enforced_after_rehash(
    tmp_path,
    monkeypatch,
):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        memory = payload["cuda_memory"]
        memory["after_failure_allocated_bytes"] = (
            memory["baseline_allocated_bytes"]
            + memory["post_execution_global_allocated_ceiling_bytes"]
            + 1
        )
        memory["after_failure_reserved_bytes"] = max(
            memory["after_failure_reserved_bytes"],
            memory["after_failure_allocated_bytes"],
        )

    _mutate_json_artifact(
        archive,
        "raw/recovery-draft_construct.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="allocated-memory ceiling"):
        VALIDATOR.validate_archive(archive)


def test_model_before_after_mutation_is_rejected_after_rehash(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        payload["model_artifacts"]["draft"]["after"]["weight_files"][0][
            "sha256"
        ] = "0" * 64

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="draft model changed"):
        VALIDATOR.validate_archive(archive)


def test_v0_cross_mode_output_mutation_is_rejected_after_rehash(
    tmp_path,
    monkeypatch,
):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        new_token = payload["run"]["outputs"][0]["token_ids"][0] + 1
        payload["run"]["outputs"][0]["token_ids"][0] = new_token
        payload["run"]["comparison_outputs"][0][1][0] = new_token

    _mutate_json_artifact(
        archive,
        "raw/v0-graph.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="registered control"):
        VALIDATOR.validate_archive(archive)


def test_rehashed_payload_and_manifest_cannot_redefine_the_release(tmp_path):
    archive = _copy_archive(tmp_path)
    relative = "raw/v2-eager.json"
    payload = _load(archive / relative)
    payload["configured_k"] = 3
    _write(archive / relative, payload)
    _refresh_artifact_row(archive, relative)

    with pytest.raises(
        VALIDATOR.EvidenceValidationError,
        match="manifest.json differs from the trusted release registry",
    ):
        VALIDATOR.validate_archive(archive)


def test_readme_mutation_is_rejected_by_the_release_registry(tmp_path):
    archive = _copy_archive(tmp_path)
    readme = archive / "README.md"
    readme.write_bytes(readme.read_bytes() + b"\nchanged\n")

    with pytest.raises(
        VALIDATOR.EvidenceValidationError,
        match="README.md differs from the trusted release registry",
    ):
        VALIDATOR.validate_archive(archive)


def test_manifest_mutation_is_rejected_by_the_release_registry(tmp_path):
    archive = _copy_archive(tmp_path)
    manifest = archive / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")

    with pytest.raises(
        VALIDATOR.EvidenceValidationError,
        match="manifest.json differs from the trusted release registry",
    ):
        VALIDATOR.validate_archive(archive)


def test_hard_linked_artifact_is_rejected(tmp_path):
    archive = _copy_archive(tmp_path)
    artifact = archive / "raw" / "v2-eager.json"
    external = tmp_path / "external-v2-eager.json"
    external.write_bytes(artifact.read_bytes())
    artifact.unlink()
    os.link(external, artifact)

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="must not be hard-linked"):
        VALIDATOR.validate_archive(archive)


def test_manifest_integer_fields_do_not_accept_equal_floats(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)
    manifest_path = archive / "manifest.json"
    manifest = _load(manifest_path)
    manifest["protocol"]["configured_k"] = 2.0
    _write(manifest_path, manifest)
    _trust_manifest_for_semantic_test(monkeypatch, archive)

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="must be an integer"):
        VALIDATOR.validate_archive(archive)


@pytest.mark.parametrize("field", ["gpu_certified", "draft_proposals_executed"])
def test_unknown_claim_critical_fields_are_rejected(
    tmp_path,
    monkeypatch,
    field,
):
    archive = _copy_archive(tmp_path)
    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        lambda payload: payload.__setitem__(field, False),
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="claim-critical"):
        VALIDATOR.validate_archive(archive)


def test_null_rng_snapshot_is_rejected_after_rehash(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        payload["rng_snapshots"]["speculation_on_after_construction"] = None

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="must be an object"):
        VALIDATOR.validate_archive(archive)


def test_unequal_rng_snapshots_are_rejected_after_rehash(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        payload["rng_snapshots"]["speculation_on_after_generation"][
            "cpu_sha256"
        ] = "0" * 64

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="post-generation RNG mismatch"):
        VALIDATOR.validate_archive(archive)


def test_tokenizer_fingerprint_is_bound_to_the_pair(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)
    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        lambda payload: payload.__setitem__(
            "speculative_tokenizer_fingerprint",
            "0" * 64,
        ),
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="not tied to the tokenizer pair"):
        VALIDATOR.validate_archive(archive)


@pytest.mark.parametrize(
    "field",
    ["half_config_error", "valid_second_engine_error"],
)
def test_engine_errors_require_exact_exception_types(
    tmp_path,
    monkeypatch,
    field,
):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        payload[field]["type"] = "other.ValueError"

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="diagnostic mismatch"):
        VALIDATOR.validate_archive(archive)


def test_n_plus_one_error_message_is_exact(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        payload["first_ineligible_capacity_error"]["message"] += " extra"

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match=r"N\+1 capacity diagnostic"):
        VALIDATOR.validate_archive(archive)


def test_post_init_headroom_uses_the_recorded_free_memory(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        payload["speculative_memory_audit"]["post_init_free_bytes"] += 1

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="post-init budget headroom"):
        VALIDATOR.validate_archive(archive)


def test_post_init_free_memory_cannot_exceed_device_total(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        audit = payload["speculative_memory_audit"]
        audit["post_init_free_bytes"] = audit["total_memory_bytes"] + 1

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="exceeds total device memory"):
        VALIDATOR.validate_archive(archive)


def test_registered_weight_bytes_are_exact(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        payload["speculative_memory_audit"]["target_weight_bytes"] -= 1

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="physical model-weight bytes"):
        VALIDATOR.validate_archive(archive)


def test_registered_joint_block_geometry_is_exact(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        audit = payload["speculative_memory_audit"]
        selected = audit["selected_num_blocks"]
        audit["target_block_bytes"] += 1
        audit["draft_block_bytes"] -= 1
        audit["target_kv_bytes"] += selected
        audit["draft_kv_bytes"] -= selected

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="KV block geometry"):
        VALIDATOR.validate_archive(archive)


def test_explicit_audit_must_equal_automatic_ceiling(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        audit = payload["explicit_memory_audit"]
        audit["post_init_free_bytes"] += 1
        audit["post_init_budget_headroom_bytes"] += 1
        audit["modeled_runtime_headroom_bytes"] += 1
        payload["explicit_memory_reconciliation"] = (
            VALIDATOR.independent_memory_reconciliation(audit, mode="eager")
        )

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="drifted from the automatic"):
        VALIDATOR.validate_archive(archive)


def test_recovery_requires_a_zero_cuda_allocator_baseline(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        payload["cuda_memory"]["baseline_allocated_bytes"] = 1
        payload["cuda_memory"]["baseline_reserved_bytes"] = 1

    _mutate_json_artifact(
        archive,
        "raw/recovery-draft_load.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="fresh CUDA allocator baseline"):
        VALIDATOR.validate_archive(archive)


def test_recovery_import_origins_require_the_certified_checkout(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        payload["runtime_import_origins"]["llm_class"] = (
            "/evil/workspace/nano-vllm-spec-v2/nanovllm/llm.py"
        )

    _mutate_json_artifact(
        archive,
        "raw/recovery-draft_load.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="certified checkout"):
        VALIDATOR.validate_archive(archive)


def test_lifecycle_repository_identity_is_exact(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        source = payload["provenance"]["source"]
        source["git_before"]["repository"] = "/evil/nano-vllm-spec-v2"
        source["git_after"]["repository"] = "/evil/nano-vllm-spec-v2"

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="source checkout identity"):
        VALIDATOR.validate_archive(archive)


def test_gpu_index_does_not_accept_boolean_zero(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        payload["provenance"]["gpu"]["before"]["index"] = False

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="integer device 0"):
        VALIDATOR.validate_archive(archive)


def test_after_endpoint_rejects_an_added_owned_pid(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        after = payload["provenance"]["gpu"]["after"]
        extra = dict(after["compute_apps"][0])
        extra.update(
            {
                "pid": 999999,
                "pid_raw": "999999",
                "used_memory_mib": 1,
                "used_memory_raw": "1",
            }
        )
        after["compute_apps"].append(extra)
        after["own_gpu_process_ids"].append(999999)

    _mutate_json_artifact(
        archive,
        "raw/v2-eager.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="after endpoint ownership"):
        VALIDATOR.validate_archive(archive)


def test_v0_scheduler_trace_is_the_exact_registered_control(tmp_path, monkeypatch):
    archive = _copy_archive(tmp_path)

    def mutate(payload):
        payload["run"]["scheduler_trace"][0][1][0][0] += 1

    _mutate_json_artifact(
        archive,
        "raw/v0-graph.json",
        mutate,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="scheduler trace differs"):
        VALIDATOR.validate_archive(archive)


def test_extra_regular_file_is_rejected(tmp_path):
    archive = _copy_archive(tmp_path)
    (archive / "unexpected.txt").write_text("not evidence\n", encoding="utf-8")

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="archive file set"):
        VALIDATOR.validate_archive(archive)


def test_missing_archive_readme_is_rejected(tmp_path):
    archive = _copy_archive(tmp_path)
    (archive / "README.md").unlink()

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="missing artifact"):
        VALIDATOR.validate_archive(archive)


def test_ignored_root_basename_is_not_ignored_under_raw(tmp_path):
    archive = _copy_archive(tmp_path)
    (archive / "raw" / "README.md").write_text(
        "not a root archive README\n",
        encoding="utf-8",
    )

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="archive file set"):
        VALIDATOR.validate_archive(archive)


def test_extra_dangling_symlink_is_rejected(tmp_path):
    archive = _copy_archive(tmp_path)
    (archive / "dangling").symlink_to("does-not-exist")

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="unmanifested symlink"):
        VALIDATOR.validate_archive(archive)


def test_symlinked_raw_ancestor_is_rejected(tmp_path):
    archive = _copy_archive(tmp_path)
    moved_raw = tmp_path / "moved-raw"
    (archive / "raw").rename(moved_raw)
    (archive / "raw").symlink_to(moved_raw, target_is_directory=True)

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="contains a symlink"):
        VALIDATOR.validate_archive(archive)


def test_symlinked_manifest_is_rejected(tmp_path):
    archive = _copy_archive(tmp_path)
    manifest = archive / "manifest.json"
    external = tmp_path / "external-manifest.json"
    external.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(external)

    with pytest.raises(VALIDATOR.EvidenceValidationError, match="non-symlink"):
        VALIDATOR.validate_archive(archive)
