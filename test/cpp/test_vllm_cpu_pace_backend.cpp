// The vllm "cpu-pace" backend is gated on the x86_64-avx512bf16 CPU family,
// which /system-info reports in cpu.extra_families. Exercises the real
// SystemInfo::build_recipes_info() support matrix with fixed device inputs.

#include <cstdlib>
#include <iostream>
#include <string>

#include <lemon/config_file.h>
#include <lemon/runtime_config.h>
#include <lemon/system_info.h>

using lemon::json;

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) {
        std::cout << "[ok] " << message << std::endl;
    } else {
        std::cerr << "[FAIL] " << message << std::endl;
        ++failures;
    }
}

class TestSystemInfo final : public lemon::SystemInfo {
public:
    lemon::CPUInfo get_cpu_device() override { return {}; }
    lemon::GPUInfo get_amd_igpu_device() override { return {}; }
    std::vector<lemon::GPUInfo> get_amd_dgpu_devices() override { return {}; }
    std::vector<lemon::GPUInfo> get_nvidia_gpu_devices() override { return {}; }
    lemon::NPUInfo get_npu_device() override { return {}; }
};

json cpu_devices(bool avx512bf16) {
    json cpu = {{"name", "Test CPU"}, {"available", true}, {"family", "x86_64"}};
    if (avx512bf16) {
        cpu["extra_families"] = json::array({"x86_64-avx512bf16"});
    }
    return {{"cpu", cpu}};
}

json cpu_pace_status(bool avx512bf16) {
    lemon::RuntimeConfig config(lemon::ConfigFile::base_defaults());
    lemon::RuntimeConfig* previous = lemon::RuntimeConfig::global();
    lemon::RuntimeConfig::set_global(&config);
    TestSystemInfo system_info;
    const json recipes = system_info.build_recipes_info(cpu_devices(avx512bf16));
    lemon::RuntimeConfig::set_global(previous);

    if (!recipes.contains("vllm") || !recipes["vllm"]["backends"].contains("cpu-pace")) {
        return json::object();
    }
    return recipes["vllm"]["backends"]["cpu-pace"];
}

}  // namespace

int main() {
#ifdef _WIN32
    _putenv("LEMONADE_VLLM_CPU_PACE_BIN=");
#else
    unsetenv("LEMONADE_VLLM_CPU_PACE_BIN");
#endif

    const json with_bf16 = cpu_pace_status(true);
    const json without_bf16 = cpu_pace_status(false);
    check(!with_bf16.empty() && !without_bf16.empty(), "vllm recipe lists the cpu-pace backend");

    check(without_bf16.value("state", "") == "unsupported",
          "cpu-pace is unsupported without AVX-512 BF16");

#ifdef __linux__
    const std::string state = with_bf16.value("state", "");
    check(state == "installable" || state == "installed",
          "cpu-pace is available on Linux with AVX-512 BF16 (state: " + state + ")");
    check(without_bf16.value("message", "").find("AVX-512 BF16") != std::string::npos,
          "unsupported message names the missing CPU feature");
#else
    check(with_bf16.value("state", "") == "unsupported", "cpu-pace is Linux-only");
#endif

    if (failures != 0) {
        std::cerr << "Total failures: " << failures << std::endl;
        return 1;
    }
    std::cout << "All vllm cpu-pace backend tests passed!" << std::endl;
    return 0;
}
