#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void add_bias(const int *input, int *output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = input[index] + 7;
  }
}

static void require_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    std::exit(2);
  }
}

int main() {
  constexpr int count = 257;
  std::vector<int> input(count);
  std::vector<int> output(count, 0);
  long long expected_sum = 0;
  for (int index = 0; index < count; ++index) {
    input[index] = (index * 17) - 31;
    expected_sum += static_cast<long long>(input[index] + 7);
  }

  int *device_input = nullptr;
  int *device_output = nullptr;
  require_cuda(cudaMalloc(&device_input, count * sizeof(int)), "cudaMalloc(input)");
  require_cuda(cudaMalloc(&device_output, count * sizeof(int)), "cudaMalloc(output)");
  require_cuda(
      cudaMemcpy(
          device_input,
          input.data(),
          count * sizeof(int),
          cudaMemcpyHostToDevice),
      "cudaMemcpy(host-to-device)");

  add_bias<<<(count + 63) / 64, 64>>>(device_input, device_output, count);
  require_cuda(cudaGetLastError(), "kernel launch");
  require_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  require_cuda(
      cudaMemcpy(
          output.data(),
          device_output,
          count * sizeof(int),
          cudaMemcpyDeviceToHost),
      "cudaMemcpy(device-to-host)");

  long long observed_sum = 0;
  for (int index = 0; index < count; ++index) {
    const int expected = input[index] + 7;
    if (output[index] != expected) {
      std::fprintf(
          stderr,
          "mismatch index=%d expected=%d observed=%d\n",
          index,
          expected,
          output[index]);
      return 3;
    }
    observed_sum += output[index];
  }

  require_cuda(cudaFree(device_output), "cudaFree(output)");
  require_cuda(cudaFree(device_input), "cudaFree(input)");
  std::printf(
      "CUDA_HIP_SMOKE PASS count=%d expected_sum=%lld observed_sum=%lld\n",
      count,
      expected_sum,
      observed_sum);
  return observed_sum == expected_sum ? 0 : 4;
}
