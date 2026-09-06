from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import nanovllm.utils.loader as loader_module
from nanovllm.layers.embed_head import VocabParallelEmbedding
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.utils.loader import ModelWeightLoadError, load_model


class PackedFixture(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self):
        super().__init__()
        self.plain = nn.Parameter(torch.full((2,), float("nan")))
        self.qkv_proj = nn.Parameter(torch.full((6,), float("nan")))

        def load_packed(parameter, loaded_weight, shard_id):
            index = {"q": 0, "k": 1, "v": 2}[shard_id]
            parameter.data[index * 2 : (index + 1) * 2].copy_(
                loaded_weight
            )

        self.qkv_proj.weight_loader = load_packed


def write_weights(directory: Path, **weights):
    save_file(weights, str(directory / "model.safetensors"))


def test_load_model_proves_plain_storage_and_every_packed_shard(tmp_path):
    model = PackedFixture()
    write_weights(
        tmp_path,
        plain=torch.tensor([1.0, 2.0]),
        q_proj=torch.tensor([3.0, 4.0]),
        k_proj=torch.tensor([5.0, 6.0]),
        v_proj=torch.tensor([7.0, 8.0]),
    )

    load_model(model, str(tmp_path))

    assert torch.equal(model.plain, torch.tensor([1.0, 2.0]))
    assert torch.equal(
        model.qkv_proj,
        torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )


def test_load_model_supports_integer_gate_up_shard_ids(tmp_path):
    class GateUpFixture(nn.Module):
        packed_modules_mapping = {
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1),
        }

        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.full((4,), float("nan")))

            def load_packed(parameter, loaded_weight, shard_id):
                parameter.data[shard_id * 2 : (shard_id + 1) * 2].copy_(
                    loaded_weight
                )

            self.gate_up_proj.weight_loader = load_packed

    write_weights(
        tmp_path,
        gate_proj=torch.tensor([1.0, 2.0]),
        up_proj=torch.tensor([3.0, 4.0]),
    )
    model = GateUpFixture()

    load_model(model, str(tmp_path))

    assert torch.equal(model.gate_up_proj, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_load_model_requires_and_loads_qkv_weight_and_bias_shards(tmp_path):
    class QKVBiasFixture(nn.Module):
        packed_modules_mapping = PackedFixture.packed_modules_mapping

        def __init__(self):
            super().__init__()
            self.attn = nn.Module()
            self.attn.qkv_proj = nn.Linear(1, 6, bias=True)

            def load_packed(parameter, loaded_weight, shard_id):
                index = {"q": 0, "k": 1, "v": 2}[shard_id]
                destination = parameter.data[index * 2 : (index + 1) * 2]
                destination.copy_(loaded_weight)

            self.attn.qkv_proj.weight.weight_loader = load_packed
            self.attn.qkv_proj.bias.weight_loader = load_packed

    weights = {}
    for index, source in enumerate(("q_proj", "k_proj", "v_proj"), 1):
        weights[f"attn.{source}.weight"] = torch.full((2, 1), float(index))
        weights[f"attn.{source}.bias"] = torch.full((2,), float(index + 3))
    write_weights(tmp_path, **weights)
    model = QKVBiasFixture()

    load_model(model, str(tmp_path))

    assert torch.equal(
        model.attn.qkv_proj.weight,
        torch.tensor([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]]),
    )
    assert torch.equal(
        model.attn.qkv_proj.bias,
        torch.tensor([4.0, 4.0, 5.0, 5.0, 6.0, 6.0]),
    )


def test_load_model_rejects_plain_broadcast_shape(tmp_path):
    class PlainFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.full((2, 3), float("nan")))

    model = PlainFixture()
    write_weights(tmp_path, weight=torch.tensor([[1.0, 2.0, 3.0]]))

    with pytest.raises(ModelWeightLoadError, match="failed loading") as error:
        load_model(model, str(tmp_path))

    assert isinstance(error.value.__cause__, ValueError)
    assert "shape does not match" in str(error.value.__cause__)
    assert bool(torch.isnan(model.weight).all())


def test_load_model_rejects_packed_broadcast_shape(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    class OfficialQKVFixture(nn.Module):
        packed_modules_mapping = PackedFixture.packed_modules_mapping

        def __init__(self):
            super().__init__()
            self.qkv_proj = QKVParallelLinear(
                hidden_size=3,
                head_size=1,
                total_num_heads=2,
                total_num_kv_heads=2,
                bias=False,
            )

    model = OfficialQKVFixture()
    write_weights(
        tmp_path,
        **{
            "q_proj.weight": torch.tensor([[1.0, 2.0, 3.0]]),
            "k_proj.weight": torch.ones(2, 3),
            "v_proj.weight": torch.ones(2, 3),
        },
    )

    with pytest.raises(
        ModelWeightLoadError,
        match="failed loading packed checkpoint tensor",
    ) as error:
        load_model(model, str(tmp_path))

    assert isinstance(error.value.__cause__, ValueError)
    assert "shape does not match" in str(error.value.__cause__)


def test_every_official_weight_loader_rejects_broadcast_shape(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    direct_cases = (
        (ReplicatedLinear(3, 2, bias=False), torch.ones(1, 3)),
        (ColumnParallelLinear(3, 2, bias=False), torch.ones(2, 1)),
        (RowParallelLinear(3, 2, bias=False), torch.ones(1, 3)),
        (VocabParallelEmbedding(2, 3), torch.ones(2, 1)),
    )
    for module, loaded_weight in direct_cases:
        with pytest.raises(ValueError, match="shape does not match"):
            module.weight.weight_loader(module.weight, loaded_weight)

    merged = MergedColumnParallelLinear(3, [2, 2], bias=False)
    with pytest.raises(ValueError, match="shape does not match"):
        merged.weight.weight_loader(
            merged.weight,
            torch.ones(1, 3),
            0,
        )

    qkv = QKVParallelLinear(
        hidden_size=3,
        head_size=1,
        total_num_heads=2,
        total_num_kv_heads=2,
        bias=False,
    )
    with pytest.raises(ValueError, match="shape does not match"):
        qkv.weight.weight_loader(
            qkv.weight,
            torch.ones(1, 3),
            "q",
        )


def test_sharded_weight_loaders_reject_oversized_global_source(monkeypatch):
    # Rank 1 is the dangerous case for uneven ``chunk``: without a source-level
    # check it can receive the expected local rows while rank 0 alone sees the
    # oversized remainder. ``narrow`` silently discards the extra tail on both
    # ranks for Column/Row/Vocab loaders.
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    direct_cases = (
        (ColumnParallelLinear(3, 4, bias=False), torch.ones(5, 3)),
        (RowParallelLinear(4, 3, bias=False), torch.ones(3, 5)),
        (VocabParallelEmbedding(4, 3), torch.ones(5, 3)),
    )
    for module, loaded_weight in direct_cases:
        with pytest.raises(ValueError, match="expected global source"):
            module.weight.weight_loader(module.weight, loaded_weight)

    merged = MergedColumnParallelLinear(3, [4, 4], bias=False)
    with pytest.raises(ValueError, match="expected global source"):
        merged.weight.weight_loader(
            merged.weight,
            torch.ones(5, 3),
            0,
        )

    qkv = QKVParallelLinear(
        hidden_size=3,
        head_size=1,
        total_num_heads=4,
        total_num_kv_heads=2,
        bias=False,
    )
    with pytest.raises(ValueError, match="expected global source"):
        qkv.weight.weight_loader(
            qkv.weight,
            torch.ones(5, 3),
            "q",
        )


def test_sharded_weight_loaders_accept_exact_global_geometry(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    column = ColumnParallelLinear(3, 4, bias=False)
    column_source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    column.weight.weight_loader(column.weight, column_source)
    assert torch.equal(column.weight, column_source[2:4])

    row = RowParallelLinear(4, 3, bias=False)
    row_source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    row.weight.weight_loader(row.weight, row_source)
    assert torch.equal(row.weight, row_source[:, 2:4])

    embedding = VocabParallelEmbedding(4, 3)
    embedding_source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    embedding.weight.weight_loader(embedding.weight, embedding_source)
    assert torch.equal(embedding.weight, embedding_source[2:4])

    merged = MergedColumnParallelLinear(3, [4, 4], bias=False)
    merged_source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    merged.weight.weight_loader(merged.weight, merged_source, 0)
    assert torch.equal(merged.weight[:2], merged_source[2:4])

    qkv = QKVParallelLinear(
        hidden_size=3,
        head_size=1,
        total_num_heads=4,
        total_num_kv_heads=2,
        bias=False,
    )
    qkv_source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    qkv.weight.weight_loader(qkv.weight, qkv_source, "q")
    assert torch.equal(qkv.weight[:2], qkv_source[2:4])


def test_load_model_rejects_missing_plain_storage(tmp_path):
    model = PackedFixture()
    write_weights(
        tmp_path,
        q_proj=torch.tensor([3.0, 4.0]),
        k_proj=torch.tensor([5.0, 6.0]),
        v_proj=torch.tensor([7.0, 8.0]),
    )

    with pytest.raises(ModelWeightLoadError, match="missing_views=.*plain"):
        load_model(model, str(tmp_path))


def test_load_model_rejects_missing_packed_shard(tmp_path):
    model = PackedFixture()
    write_weights(
        tmp_path,
        plain=torch.tensor([1.0, 2.0]),
        q_proj=torch.tensor([3.0, 4.0]),
        k_proj=torch.tensor([5.0, 6.0]),
    )

    with pytest.raises(
        ModelWeightLoadError,
        match=r"missing_packed_shards=.*qkv_proj.*'v'",
    ):
        load_model(model, str(tmp_path))


def test_load_model_counts_tied_aliases_by_physical_storage(tmp_path):
    class TiedFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Parameter(torch.full((2,), float("nan")))
            self.lm_head = nn.Parameter(self.embed.data)

    model = TiedFixture()
    write_weights(tmp_path, embed=torch.tensor([9.0, 10.0]))

    load_model(model, str(tmp_path))

    assert model.embed.untyped_storage().data_ptr() == (
        model.lm_head.untyped_storage().data_ptr()
    )
    assert torch.equal(model.lm_head, torch.tensor([9.0, 10.0]))


def test_load_model_accepts_both_checkpoint_names_for_tied_aliases(tmp_path):
    class TiedFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Parameter(torch.full((2,), float("nan")))
            self.lm_head = nn.Parameter(self.embed.data)

    model = TiedFixture()
    write_weights(
        tmp_path,
        embed=torch.tensor([9.0, 10.0]),
        lm_head=torch.tensor([9.0, 10.0]),
    )

    load_model(model, str(tmp_path))

    assert torch.equal(model.embed, torch.tensor([9.0, 10.0]))
    assert torch.equal(model.lm_head, torch.tensor([9.0, 10.0]))


def test_load_model_rejects_conflicting_tied_alias_values(tmp_path):
    class TiedFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Parameter(torch.full((2,), float("nan")))
            self.lm_head = nn.Parameter(self.embed.data)

    write_weights(
        tmp_path,
        embed=torch.tensor([9.0, 10.0]),
        lm_head=torch.tensor([9.0, 11.0]),
    )

    with pytest.raises(ModelWeightLoadError, match="different tensor values"):
        load_model(TiedFixture(), str(tmp_path))


def test_load_model_fingerprints_tied_scalar_aliases(tmp_path):
    class TiedScalarFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Parameter(torch.tensor(float("nan")))
            self.right = nn.Parameter(self.left.data)

    write_weights(
        tmp_path,
        left=torch.tensor(9.0),
        right=torch.tensor(9.0),
    )

    model = TiedScalarFixture()
    load_model(model, str(tmp_path))

    assert model.left.item() == 9.0
    assert model.right.item() == 9.0


def test_load_model_rejects_directory_without_supported_weights(tmp_path):
    with pytest.raises(ModelWeightLoadError, match="no supported safetensors"):
        load_model(PackedFixture(), str(tmp_path))


def test_disjoint_views_of_one_storage_do_not_alias_coverage(tmp_path):
    class DisjointFixture(nn.Module):
        def __init__(self):
            super().__init__()
            backing = torch.full((4,), float("nan"))
            self.left = nn.Parameter(backing[:2])
            self.right = nn.Parameter(backing[2:])

    model = DisjointFixture()
    write_weights(tmp_path, left=torch.tensor([1.0, 2.0]))

    with pytest.raises(ModelWeightLoadError, match="missing_views=.*right"):
        load_model(model, str(tmp_path))


def test_packed_source_names_match_exact_path_segments_only(tmp_path):
    class SubstringFixture(nn.Module):
        packed_modules_mapping = PackedFixture.packed_modules_mapping

        def __init__(self):
            super().__init__()
            self.not_q_proj_aux = nn.Parameter(torch.full((2,), float("nan")))

    model = SubstringFixture()
    write_weights(
        tmp_path,
        not_q_proj_aux=torch.tensor([11.0, 12.0]),
    )

    load_model(model, str(tmp_path))

    assert torch.equal(model.not_q_proj_aux, torch.tensor([11.0, 12.0]))


def test_duplicate_checkpoint_tensor_across_files_is_rejected(tmp_path):
    complete = {
        "plain": torch.tensor([1.0, 2.0]),
        "q_proj": torch.tensor([3.0, 4.0]),
        "k_proj": torch.tensor([5.0, 6.0]),
        "v_proj": torch.tensor([7.0, 8.0]),
    }
    save_file(complete, str(tmp_path / "a.safetensors"))
    save_file(
        {"plain": torch.tensor([9.0, 10.0])},
        str(tmp_path / "b.safetensors"),
    )

    with pytest.raises(ModelWeightLoadError, match="duplicate checkpoint"):
        load_model(PackedFixture(), str(tmp_path))


def test_weight_loader_failure_is_typed_and_not_reported_as_coverage(tmp_path):
    class FailingFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.full((2,), float("nan")))

            def fail(parameter, loaded_weight):
                raise ValueError("injected loader failure")

            self.weight.weight_loader = fail

    write_weights(tmp_path, weight=torch.tensor([1.0, 2.0]))

    with pytest.raises(ModelWeightLoadError, match="failed loading") as error:
        load_model(FailingFixture(), str(tmp_path))
    assert isinstance(error.value.__cause__, ValueError)


def test_load_model_types_malformed_safetensors_file(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"not safetensors")

    with pytest.raises(ModelWeightLoadError, match="cannot open safetensors"):
        load_model(PackedFixture(), str(tmp_path))


def test_load_model_preserves_resource_exhaustion(tmp_path):
    class OomFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.full((2,), float("nan")))

            def fail(parameter, loaded_weight):
                raise MemoryError("injected exhaustion")

            self.weight.weight_loader = fail

    write_weights(tmp_path, weight=torch.tensor([1.0, 2.0]))

    with pytest.raises(MemoryError, match="injected exhaustion"):
        load_model(OomFixture(), str(tmp_path))


@pytest.mark.parametrize("phase", ["open", "keys", "get_tensor"])
def test_load_model_preserves_resource_exhaustion_from_safetensors(
    tmp_path,
    monkeypatch,
    phase,
):
    class PlainFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

    class FakeSafeFile:
        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return False

        def keys(self):
            if phase == "keys":
                raise MemoryError("injected exhaustion")
            return ["weight"]

        def get_tensor(self, unused_name):
            raise MemoryError("injected exhaustion")

    (tmp_path / "model.safetensors").write_bytes(b"placeholder")

    def fake_safe_open(*unused_args, **unused_kwargs):
        if phase == "open":
            raise MemoryError("injected exhaustion")
        return FakeSafeFile()

    monkeypatch.setattr(loader_module, "safe_open", fake_safe_open)

    with pytest.raises(MemoryError, match="injected exhaustion"):
        load_model(PlainFixture(), str(tmp_path))


def test_model_path_with_glob_metacharacters_loads_normally(tmp_path):
    model_path = tmp_path / "model[1]"
    model_path.mkdir()
    write_weights(
        model_path,
        plain=torch.tensor([1.0, 2.0]),
        q_proj=torch.tensor([3.0, 4.0]),
        k_proj=torch.tensor([5.0, 6.0]),
        v_proj=torch.tensor([7.0, 8.0]),
    )

    model = PackedFixture()
    load_model(model, str(model_path))

    assert torch.equal(model.plain, torch.tensor([1.0, 2.0]))
