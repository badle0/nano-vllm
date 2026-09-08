"""CI fixture setup must be pinned and must not fetch model weights."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / ".github/ci/prepare_tokenizer.py"
SPEC = importlib.util.spec_from_file_location("prepare_ci_tokenizer", SCRIPT)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_ci_download_is_pinned_and_tokenizer_only(tmp_path, monkeypatch):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        (kwargs["local_dir"] / kwargs["filename"]).write_text("{}")

    monkeypatch.setattr(PREPARE, "hf_hub_download", download)
    PREPARE.prepare(tmp_path)
    assert [call["filename"] for call in calls] == [
        "config.json", "tokenizer_config.json", "tokenizer.json"
    ]
    assert all(call["repo_id"] == "Qwen/Qwen3-0.6B" for call in calls)
    assert all(call["revision"] == "c1899de289a04d12100db370d81485cdf75e47ca"
               for call in calls)
    assert all(call["local_dir"] == tmp_path for call in calls)


def test_ci_download_requires_actual_files(tmp_path, monkeypatch):
    monkeypatch.setattr(PREPARE, "hf_hub_download", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="did not produce config.json"):
        PREPARE.prepare(tmp_path)
