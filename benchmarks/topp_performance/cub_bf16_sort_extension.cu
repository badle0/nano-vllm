#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <cstdint>
#include <limits>

namespace {

struct SegmentOffset {
  int stride;
  int begin;

  __host__ __device__ __forceinline__ int operator[](int index) const {
    return stride * (begin + index);
  }
};

size_t query_workspace_size(int rows, int columns) {
  const int64_t items = static_cast<int64_t>(rows) * columns;
  TORCH_CHECK(rows > 0, "rows must be positive");
  TORCH_CHECK(columns > 0, "columns must be positive");
  TORCH_CHECK(
      items < std::numeric_limits<int>::max(),
      "CUB segmented radix sort requires fewer than INT_MAX items");

  size_t workspace_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
      nullptr,
      workspace_bytes,
      static_cast<const __nv_bfloat16*>(nullptr),
      static_cast<__nv_bfloat16*>(nullptr),
      static_cast<int>(items),
      rows,
      SegmentOffset{columns, 0},
      SegmentOffset{columns, 1},
      0,
      16,
      at::cuda::getCurrentCUDAStream()));
  return workspace_bytes;
}

}  // namespace

int64_t cub_bf16_sort_workspace_size(int64_t rows, int64_t columns) {
  TORCH_CHECK(
      rows <= std::numeric_limits<int>::max() &&
          columns <= std::numeric_limits<int>::max(),
      "rows and columns must fit in int32");
  return static_cast<int64_t>(
      query_workspace_size(static_cast<int>(rows), static_cast<int>(columns)));
}

void cub_bf16_sort_out(
    const at::Tensor& input,
    const at::Tensor& output,
    const at::Tensor& workspace) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
  TORCH_CHECK(workspace.scalar_type() == at::kByte, "workspace must be uint8");
  TORCH_CHECK(input.dim() == 2, "input must be two-dimensional");
  TORCH_CHECK(output.sizes() == input.sizes(), "output shape must match input");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
  TORCH_CHECK(
      input.device() == output.device() && input.device() == workspace.device(),
      "input, output, and workspace must share a device");
  TORCH_CHECK(
      input.data_ptr() != output.data_ptr(),
      "CUB input and output buffers must not overlap");

  const auto rows64 = input.size(0);
  const auto columns64 = input.size(1);
  TORCH_CHECK(
      rows64 <= std::numeric_limits<int>::max() &&
          columns64 <= std::numeric_limits<int>::max(),
      "rows and columns must fit in int32");
  const int rows = static_cast<int>(rows64);
  const int columns = static_cast<int>(columns64);
  const int64_t items64 = input.numel();
  TORCH_CHECK(
      items64 < std::numeric_limits<int>::max(),
      "CUB segmented radix sort requires fewer than INT_MAX items");

  c10::cuda::CUDAGuard device_guard(input.device());
  size_t workspace_bytes = static_cast<size_t>(workspace.numel());
  const size_t required_bytes = query_workspace_size(rows, columns);
  TORCH_CHECK(
      workspace_bytes >= required_bytes,
      "workspace is too small: need ",
      required_bytes,
      " bytes, got ",
      workspace_bytes);

  auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(
      input.const_data_ptr<at::BFloat16>());
  auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(
      output.mutable_data_ptr<at::BFloat16>());
  C10_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
      workspace.mutable_data_ptr<uint8_t>(),
      workspace_bytes,
      input_ptr,
      output_ptr,
      static_cast<int>(items64),
      rows,
      SegmentOffset{columns, 0},
      SegmentOffset{columns, 1},
      0,
      16,
      at::cuda::getCurrentCUDAStream()));
}
