import hashlib
import os
import torch
from torch import nn
from safetensors import safe_open


class ModelWeightLoadError(RuntimeError):
    """The supported safetensors artifacts did not initialize the model."""


def require_exact_weight_shape(
    destination: torch.Tensor,
    loaded_weight: torch.Tensor,
) -> None:
    """Reject ``copy_`` broadcasting after any tensor-parallel sharding.

    PyTorch's ``copy_`` accepts broadcast-compatible inputs. That behavior is
    useful for ordinary tensor programs but unsafe for checkpoint loading: a
    malformed one-row tensor can otherwise appear to initialize a multi-row
    parameter successfully. Callers must pass the final local destination view
    and the final local checkpoint shard so this check remains correct for TP.
    """

    destination_shape = tuple(destination.shape)
    loaded_shape = tuple(loaded_weight.shape)
    if loaded_shape != destination_shape:
        raise ValueError(
            "checkpoint tensor shape does not match its local destination: "
            f"loaded={loaded_shape}, destination={destination_shape}"
        )


def require_exact_global_weight_shape(
    local_destination: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    shard_dim: int,
    num_shards: int,
) -> None:
    """Validate full checkpoint geometry before selecting a local TP shard."""

    if type(num_shards) is not int or num_shards < 1:
        raise ValueError("num_shards must be a positive integer")
    ndim = local_destination.ndim
    if type(shard_dim) is not int or not -ndim <= shard_dim < ndim:
        raise ValueError("shard_dim is outside the destination rank")
    shard_dim %= ndim
    expected_global_shape = list(local_destination.shape)
    expected_global_shape[shard_dim] *= num_shards
    expected_global_shape = tuple(expected_global_shape)
    loaded_shape = tuple(loaded_weight.shape)
    if loaded_shape != expected_global_shape:
        raise ValueError(
            "checkpoint tensor shape does not match its expected global "
            "source geometry: "
            f"loaded={loaded_shape}, expected_global={expected_global_shape}, "
            f"local_destination={tuple(local_destination.shape)}, "
            f"shard_dim={shard_dim}, num_shards={num_shards}"
        )


def _parameter_view_key(parameter: nn.Parameter):
    storage = parameter.untyped_storage()
    return (
        storage,
        parameter.storage_offset(),
        tuple(parameter.shape),
        tuple(parameter.stride()),
        parameter.dtype,
    )


def _tensor_fingerprint(tensor: torch.Tensor):
    tensor = tensor.detach().cpu().contiguous()
    byte_view = tensor.reshape(-1).view(torch.uint8).numpy()
    return (
        str(tensor.dtype),
        tuple(tensor.shape),
        hashlib.sha256(memoryview(byte_view)).hexdigest(),
    )


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    require_exact_weight_shape(param.data, loaded_weight)
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for source_name, (packed_name, _) in packed_modules_mapping.items():
        if "." in source_name or "." in packed_name:
            raise ModelWeightLoadError(
                "packed module mappings must use exact single path segments; "
                f"got {source_name!r} -> {packed_name!r}"
            )

    inverse_packed_mapping = {}
    for source_name, (packed_name, shard_id) in packed_modules_mapping.items():
        shards = inverse_packed_mapping.setdefault(packed_name, {})
        if shard_id in shards:
            raise ModelWeightLoadError(
                f"duplicate packed shard id {shard_id!r} for {packed_name!r}"
            )
        shards[shard_id] = source_name

    named_parameters = dict(model.named_parameters(
        remove_duplicate=False
    ))
    parameter_views = {
        name: _parameter_view_key(parameter)
        for name, parameter in named_parameters.items()
    }
    view_names = {}
    packed_requirements = {}
    for parameter_name, view in parameter_views.items():
        view_names.setdefault(view, []).append(parameter_name)
        segments = parameter_name.split(".")
        packed_destinations = [
            packed_name
            for packed_name in inverse_packed_mapping
            if packed_name in segments
        ]
        if len(packed_destinations) > 1:
            raise ModelWeightLoadError(
                "parameter name matches multiple packed destinations: "
                f"{parameter_name!r} -> {packed_destinations!r}"
            )
        if packed_destinations:
            packed_name = packed_destinations[0]
            packed_requirements[parameter_name] = set(
                inverse_packed_mapping[packed_name]
            )
    aliased_views = {
        view for view, names in view_names.items() if len(names) > 1
    }

    try:
        files = sorted(
            entry.path
            for entry in os.scandir(path)
            if entry.name.endswith(".safetensors") and entry.is_file()
        )
    except OSError as error:
        raise ModelWeightLoadError(
            f"cannot inspect model directory {path!r}: {error}"
        ) from error
    if not files:
        raise ModelWeightLoadError(
            f"model directory contains no supported safetensors weights: {path!r}"
        )

    seen_source_names = set()
    direct_loaded_names = set()
    direct_view_fingerprints = {}
    loaded_packed_shards = {}
    for file in files:
        try:
            safe_file = safe_open(file, "pt", "cpu")
        except (MemoryError, torch.cuda.OutOfMemoryError):
            raise
        except Exception as error:
            raise ModelWeightLoadError(
                f"cannot open safetensors weight file {file!r}"
            ) from error
        with safe_file as f:
            try:
                weight_names = f.keys()
            except (MemoryError, torch.cuda.OutOfMemoryError):
                raise
            except Exception as error:
                raise ModelWeightLoadError(
                    f"cannot enumerate safetensors weight file {file!r}"
                ) from error
            for weight_name in weight_names:
                if weight_name in seen_source_names:
                    raise ModelWeightLoadError(
                        f"duplicate checkpoint tensor name: {weight_name!r}"
                    )
                seen_source_names.add(weight_name)
                segments = weight_name.split(".")
                source_matches = [
                    (index, source_name)
                    for index, segment in enumerate(segments)
                    for source_name in packed_modules_mapping
                    if segment == source_name
                ]
                if len(source_matches) > 1:
                    raise ModelWeightLoadError(
                        "checkpoint tensor matches multiple packed sources: "
                        f"{weight_name!r} -> {source_matches!r}"
                    )
                if source_matches:
                    index, source_name = source_matches[0]
                    packed_name, shard_id = packed_modules_mapping[source_name]
                    destination_segments = list(segments)
                    destination_segments[index] = packed_name
                    parameter_name = ".".join(destination_segments)
                    if parameter_name not in named_parameters:
                        raise ModelWeightLoadError(
                            "checkpoint packed tensor has no destination "
                            f"parameter: {weight_name!r} -> {parameter_name!r}"
                        )
                    observed = loaded_packed_shards.setdefault(
                        parameter_name,
                        set(),
                    )
                    if shard_id in observed:
                        raise ModelWeightLoadError(
                            "duplicate packed shard for destination parameter: "
                            f"{parameter_name!r}:{shard_id!r}"
                        )
                    parameter = named_parameters[parameter_name]
                    weight_loader = getattr(parameter, "weight_loader", None)
                    if weight_loader is None:
                        raise ModelWeightLoadError(
                            "packed destination has no shard-aware weight loader: "
                            f"{parameter_name!r}"
                        )
                    try:
                        weight_loader(
                            parameter,
                            f.get_tensor(weight_name),
                            shard_id,
                        )
                    except (MemoryError, torch.cuda.OutOfMemoryError):
                        raise
                    except Exception as error:
                        raise ModelWeightLoadError(
                            "failed loading packed checkpoint tensor "
                            f"{weight_name!r} into {parameter_name!r}"
                        ) from error
                    observed.add(shard_id)
                elif weight_name in packed_requirements:
                    raise ModelWeightLoadError(
                        "direct already-packed checkpoint tensors are not "
                        f"supported: {weight_name!r}"
                    )
                else:
                    if weight_name not in named_parameters:
                        raise ModelWeightLoadError(
                            "checkpoint tensor has no destination parameter: "
                            f"{weight_name!r}"
                        )
                    parameter = named_parameters[weight_name]
                    try:
                        loaded_weight = f.get_tensor(weight_name)
                    except (MemoryError, torch.cuda.OutOfMemoryError):
                        raise
                    except Exception as error:
                        raise ModelWeightLoadError(
                            f"cannot read checkpoint tensor {weight_name!r}"
                        ) from error
                    view = parameter_views[weight_name]
                    if view in aliased_views:
                        fingerprint = _tensor_fingerprint(loaded_weight)
                        previous = direct_view_fingerprints.get(view)
                        if previous is not None and previous != fingerprint:
                            raise ModelWeightLoadError(
                                "tied checkpoint aliases contain different "
                                f"tensor values: {view_names[view]!r}"
                            )
                        direct_view_fingerprints[view] = fingerprint
                    weight_loader = getattr(
                        parameter,
                        "weight_loader",
                        default_weight_loader,
                    )
                    try:
                        weight_loader(parameter, loaded_weight)
                    except (MemoryError, torch.cuda.OutOfMemoryError):
                        raise
                    except Exception as error:
                        raise ModelWeightLoadError(
                            "failed loading checkpoint tensor "
                            f"{weight_name!r}"
                        ) from error
                    direct_loaded_names.add(weight_name)

    fully_covered_names = set(direct_loaded_names)
    missing_packed = []
    for parameter_name, required_shards in packed_requirements.items():
        observed = loaded_packed_shards.get(parameter_name, set())
        missing = required_shards - observed
        if missing:
            missing_packed.extend(
                f"{parameter_name}:{shard_id!r}"
                for shard_id in missing
            )
        else:
            fully_covered_names.add(parameter_name)

    missing_views = [
        sorted(names)
        for names in view_names.values()
        if not any(name in fully_covered_names for name in names)
    ]
    if missing_views or missing_packed:
        raise ModelWeightLoadError(
            "safetensors weights did not initialize every exact parameter "
            f"view and packed shard; missing_views={sorted(missing_views)}, "
            f"missing_packed_shards={sorted(missing_packed)}"
        )
