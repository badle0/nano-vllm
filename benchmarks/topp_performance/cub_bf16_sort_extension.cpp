#include <torch/extension.h>

#include <cstdint>

int64_t cub_bf16_sort_workspace_size(int64_t rows, int64_t columns);

void cub_bf16_sort_out(
    const at::Tensor& input,
    const at::Tensor& output,
    const at::Tensor& workspace);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "workspace_size",
      &cub_bf16_sort_workspace_size,
      "CUB segmented BF16 SortKeys workspace size");
  module.def(
      "sort_out",
      &cub_bf16_sort_out,
      "CUB segmented BF16 SortKeys into preallocated output");
}
