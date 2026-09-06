"""Fetch only the pinned Qwen tokenizer fixture, never model weights."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


REPOSITORY = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FILES = ("config.json", "tokenizer_config.json", "tokenizer.json")


def prepare(output):
    output = Path(output)
    for filename in FILES:
        hf_hub_download(
            repo_id=REPOSITORY,
            revision=REVISION,
            filename=filename,
            local_dir=output,
        )
        if not (output / filename).is_file():
            raise RuntimeError(f"Tokenizer download did not produce {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output)
