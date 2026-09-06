"""Construction-time proof that target and draft share one token-ID space."""

import hashlib
import json
from typing import NamedTuple


class TokenizerIdentity(NamedTuple):
    implementation: str
    vocabulary_sha256: str
    added_vocabulary_sha256: str
    special_token_ids_sha256: str
    all_special_ids_sha256: str
    special_tokens_map_sha256: str
    tokenizer_config_sha256: str
    backend_sha256: str


def _canonicalize(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if all(
        hasattr(value, attribute)
        for attribute in (
            "content",
            "single_word",
            "lstrip",
            "rstrip",
            "normalized",
            "special",
        )
    ):
        return {
            "content": value.content,
            "single_word": value.single_word,
            "lstrip": value.lstrip,
            "rstrip": value.rstrip,
            "normalized": value.normalized,
            "special": value.special,
        }
    raise ValueError(
        "tokenizer identity contains an unsupported configuration value: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_token_id(token_id, *, role: str, source: str, vocab_size: int):
    if type(token_id) is not int:
        raise ValueError(
            f"{role} tokenizer {source} contains a non-integer token ID"
        )
    if not 0 <= token_id < vocab_size:
        raise ValueError(
            f"{role} tokenizer {source} contains token ID {token_id} outside "
            f"model vocabulary [0, {vocab_size})"
        )


def _mapping(
    tokenizer,
    method_name: str,
    role: str,
    vocab_size: int,
) -> dict[str, int]:
    method = getattr(tokenizer, method_name, None)
    if not callable(method):
        raise ValueError(
            f"{role} tokenizer does not expose {method_name}()"
        )
    result = method()
    if not isinstance(result, dict):
        raise ValueError(
            f"{role} tokenizer {method_name}() must return a dictionary"
        )
    if any(not isinstance(token, str) for token in result):
        raise ValueError(
            f"{role} tokenizer {method_name}() returned an invalid token map"
        )
    for token_id in result.values():
        _validate_token_id(
            token_id,
            role=role,
            source=f"{method_name}()",
            vocab_size=vocab_size,
        )
    return result


def _special_token_ids(
    tokenizer,
    *,
    role: str,
    vocab_size: int,
) -> dict[str, int | list[int] | None]:
    names = getattr(tokenizer, "SPECIAL_TOKENS_ATTRIBUTES", None)
    if not isinstance(names, (list, tuple)):
        raise ValueError(
            f"{role} tokenizer does not expose SPECIAL_TOKENS_ATTRIBUTES"
        )
    result = {}
    for name in names:
        if not isinstance(name, str):
            raise ValueError(
                f"{role} tokenizer SPECIAL_TOKENS_ATTRIBUTES is invalid"
            )
        id_name = (
            "additional_special_tokens_ids"
            if name == "additional_special_tokens"
            else f"{name}_id"
        )
        value = getattr(tokenizer, id_name, None)
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            for token_id in value:
                _validate_token_id(
                    token_id,
                    role=role,
                    source=id_name,
                    vocab_size=vocab_size,
                )
        elif value is not None:
            _validate_token_id(
                value,
                role=role,
                source=id_name,
                vocab_size=vocab_size,
            )
        result[id_name] = value
    return result


def _required_special_metadata(tokenizer, *, role: str, vocab_size: int):
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    _validate_token_id(
        eos_token_id,
        role=role,
        source="eos_token_id",
        vocab_size=vocab_size,
    )

    all_special_ids = getattr(tokenizer, "all_special_ids", None)
    if not isinstance(all_special_ids, (list, tuple)):
        raise ValueError(f"{role} tokenizer does not expose all_special_ids")
    all_special_ids = list(all_special_ids)
    for token_id in all_special_ids:
        _validate_token_id(
            token_id,
            role=role,
            source="all_special_ids",
            vocab_size=vocab_size,
        )
    if eos_token_id not in all_special_ids:
        raise ValueError(
            f"{role} tokenizer eos_token_id is absent from all_special_ids"
        )

    special_tokens_map = getattr(
        tokenizer,
        "special_tokens_map_extended",
        None,
    )
    if not isinstance(special_tokens_map, dict):
        special_tokens_map = getattr(tokenizer, "special_tokens_map", None)
    if not isinstance(special_tokens_map, dict):
        raise ValueError(
            f"{role} tokenizer does not expose a special_tokens_map"
        )
    return eos_token_id, all_special_ids, special_tokens_map


def _tokenizer_config(tokenizer, role: str):
    config = getattr(tokenizer, "init_kwargs", None)
    if not isinstance(config, dict):
        raise ValueError(f"{role} tokenizer does not expose init_kwargs")
    # These entries identify where equivalent artifacts were loaded from, not
    # tokenization semantics. Target and draft normally live in different dirs.
    location_keys = {
        "added_tokens_file",
        "merges_file",
        "name_or_path",
        "special_tokens_map_file",
        "tokenizer_config_file",
        "tokenizer_file",
        "vocab_file",
    }
    return {
        key: value
        for key, value in config.items()
        if key not in location_keys
    }


def tokenizer_identity(
    tokenizer,
    *,
    role: str,
    vocab_size: int,
) -> TokenizerIdentity:
    """Return stable fingerprints of every token-ID-defining component."""

    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError("model vocab_size must be a positive integer")

    backend = getattr(tokenizer, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if not callable(to_str):
        raise ValueError(
            f"{role} tokenizer must be a fast tokenizer with a serializable backend"
        )
    serialized_backend = to_str()
    if not isinstance(serialized_backend, str):
        raise ValueError(
            f"{role} tokenizer backend serialization must be a string"
        )
    try:
        backend_document = json.loads(serialized_backend)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{role} tokenizer backend serialization is not valid JSON"
        ) from error

    implementation = (
        f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    )
    eos_token_id, all_special_ids, special_tokens_map = (
        _required_special_metadata(
            tokenizer,
            role=role,
            vocab_size=vocab_size,
        )
    )
    special_token_ids = _special_token_ids(
        tokenizer,
        role=role,
        vocab_size=vocab_size,
    )
    special_token_ids["eos_token_id"] = eos_token_id
    return TokenizerIdentity(
        implementation=implementation,
        vocabulary_sha256=_canonical_sha256(
            _mapping(tokenizer, "get_vocab", role, vocab_size)
        ),
        added_vocabulary_sha256=_canonical_sha256(
            _mapping(tokenizer, "get_added_vocab", role, vocab_size)
        ),
        special_token_ids_sha256=_canonical_sha256(special_token_ids),
        all_special_ids_sha256=_canonical_sha256(all_special_ids),
        special_tokens_map_sha256=_canonical_sha256(special_tokens_map),
        tokenizer_config_sha256=_canonical_sha256(
            _tokenizer_config(tokenizer, role)
        ),
        backend_sha256=_canonical_sha256(backend_document),
    )


def require_same_token_id_space(
    target_tokenizer,
    draft_tokenizer,
    *,
    vocab_size: int,
) -> str:
    """Validate exact target/draft tokenizer identity and return its fingerprint."""

    target = tokenizer_identity(
        target_tokenizer,
        role="target",
        vocab_size=vocab_size,
    )
    draft = tokenizer_identity(
        draft_tokenizer,
        role="draft",
        vocab_size=vocab_size,
    )
    labels = {
        "implementation": "implementation class",
        "vocabulary_sha256": "complete vocabulary",
        "added_vocabulary_sha256": "added-token mapping",
        "special_token_ids_sha256": "special-token IDs",
        "all_special_ids_sha256": "complete special-token ID list",
        "special_tokens_map_sha256": "special-token map",
        "tokenizer_config_sha256": "tokenizer configuration",
        "backend_sha256": "normalizer/pre-tokenizer/backend configuration",
    }
    for field_name in TokenizerIdentity._fields:
        if getattr(target, field_name) != getattr(draft, field_name):
            raise ValueError(
                "target and draft tokenizers do not define the same token-ID "
                f"space: {labels[field_name]} differs"
            )
    return _canonical_sha256(target._asdict())
