#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace lemon {
namespace backends {

struct VLLMArgResolution {
    std::vector<std::string> args;
    bool has_memory_budget_arg = false;
    // True when the user/family already supplied --dtype, so backend code
    // should not force its own (e.g. the AWQ float16 default).
    bool has_dtype_arg = false;
    // True when the user asked for eager. The resolver has already removed the flag from
    // `args` — load() re-emits it from the launch policy.
    bool has_enforce_eager = false;
    bool has_quantization_arg = false;
    std::string quantization_arg;
    // Structured rather than a passthrough flag: the `vllm_args` tokenizer strips quotes
    // and would corrupt inline JSON. The backend re-serializes it.
    std::string speculative_config;
};

VLLMArgResolution resolve_vllm_args(const std::string& model_name,
                                    const std::string& checkpoint,
                                    const nlohmann::json& config,
                                    const std::string& user_vllm_args);

// Discrete-HBM datacenter GPUs (AMD Instinct, gfx9xx) get vLLM's native memory budgeting
// and graph capture; APUs and consumer GPUs keep the conservative launch defaults. This
// predicate is the seam a future memory planner replaces.
bool is_discrete_hbm_arch(const std::string& arch);

struct DeviceClassLaunchPolicy {
    bool enforce_eager;     // push --enforce-eager (disables CUDA-graph capture)
    bool force_awq_kernel;  // force the 'awq' kernel (+ float16) for AWQ models
    bool cap_kv_cache;      // push the fixed --kv-cache-memory-bytes cap
};

DeviceClassLaunchPolicy device_class_launch_policy(const std::string& arch,
                                                   bool has_memory_budget_arg,
                                                   bool has_enforce_eager = false);

// vLLM's CPU platform needs torch.compile for PACE's fused-MLP pass, so eager is opt-in, and
// it honors --kv-cache-memory-bytes directly, so the same fixed cap bounds host RAM.
DeviceClassLaunchPolicy cpu_launch_policy(bool has_memory_budget_arg, bool has_enforce_eager);

// Arguments the CPU backend appends to the resolved ones. PACE kernels are BF16-only, so the
// dtype is pinned unless the user chose one. Multimodal checkpoints (e.g. Qwen3.5) otherwise
// profile their vision encoder at startup, which on CPU costs minutes and several GB, so a
// model not registered for vision is served text-only unless the user set that flag.
//
// PACE keeps vLLM's in-process executor for a single worker and leaves OpenMP placement to
// the launcher, but vLLM only binds threads (VLLM_CPU_OMP_THREADS_BIND) under its
// multiprocess executor, while it binds memory to one NUMA node either way. On a host whose
// CPUs span NUMA nodes, the unbound threads straddle nodes and sockets, so `bind_omp_threads`
// requests the multiprocess executor unless the user chose an executor.
std::vector<std::string> cpu_launch_args(const VLLMArgResolution& resolved,
                                         bool model_has_vision,
                                         bool bind_omp_threads);

constexpr uint64_t kKvCacheCapBytes = 4ULL << 30;
constexpr const char* kKvCacheCapArg = "4G";

// The --gpu-memory-utilization for a device holding `free_bytes` of `total_bytes`, or a
// negative value to leave vLLM's own default in place. Both must describe the one device
// vLLM will run on; the definition explains why.
double shared_memory_gpu_utilization(uint64_t free_bytes, uint64_t total_bytes);

} // namespace backends
} // namespace lemon
