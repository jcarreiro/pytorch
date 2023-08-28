#include <ATen/ArrayRef.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <torch/library.h>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

using namespace api::utils;

#ifdef USE_VULKAN_FP16_INFERENCE
constexpr float kEpsilon = 0.0009765625;
#else
constexpr float kEpsilon = 1.1920928955078125e-07;
#endif

Tensor& uniform_(
    Tensor& self,
    double from,
    double to,
    c10::optional<at::Generator> /* not implemented */) {
  TORCH_CHECK(
      self.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self);

  const struct Block final {
    uvec3 extents;
    float from;
    float to;
  } block{v_self.extents(), static_cast<float>(from), static_cast<float>(to)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      // shader_descriptor,
      VK_KERNEL(uniform_),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self;
}

Tensor rand_like(
    Tensor const& self,
    c10::optional<c10::ScalarType> scale_type,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format) {
  // Returns a tensor with the same size as input that is filled with random
  // numbers from a uniform distribution on the interval [0,1)
  return self.uniform_(0.0, 1.0 - kEpsilon).clone();
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::uniform_"), TORCH_FN(uniform_));
  m.impl(TORCH_SELECTIVE_NAME("aten::rand_like"), TORCH_FN(rand_like));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
