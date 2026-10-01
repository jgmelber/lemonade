#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <istream>
#include <set>
#include <sstream>
#include <system_error>
#include <string>
#include <utility>
#include <vector>

namespace lemon::system_info_detail {

inline const std::set<std::string>& cuda_supported_archs() {
    static const std::set<std::string> archs = {
        "sm_75",   // Turing       (RTX 20, GTX 16, T4, Quadro RTX)
        "sm_80",   // Ampere DC    (A100)
        "sm_86",   // Ampere       (RTX 30, A40, A6000, A4000)
        "sm_89",   // Ada Lovelace (RTX 40, L40, L4)
        "sm_90",   // Hopper       (H100, H200)
        "sm_100",  // Blackwell DC (B100, B200)
        "sm_120",  // Blackwell    (RTX 50)
        "sm_121",  // Blackwell    (GB10 / Thor SoC)
    };
    return archs;
}

// Map a CUDA Compute Capability "MAJOR.MINOR" string to the sm_XX token used
// in llamacpp-cuda release filenames. Keep the parsing behavior identical to
// the server path so this helper is the single implementation under test.
inline std::string compute_cap_to_sm(const std::string& compute_cap) {
    const size_t dot = compute_cap.find('.');
    if (dot == std::string::npos) return "";

    const std::string major = compute_cap.substr(0, dot);
    const std::string minor = compute_cap.substr(dot + 1);
    if (major.empty() || minor.empty()) return "";

    try {
        const int m = std::stoi(major);
        const int n = std::stoi(minor);
        return "sm_" + std::to_string(m * 10 + n);
    } catch (...) {
        return "";
    }
}

// Match a concrete device family against exact or trailing-X family tokens.
inline bool device_matches_constraint(
    const std::string& device_family,
    const std::set<std::string>& allowed_families) {
    if (allowed_families.empty()) {
        return true;
    }
    if (allowed_families.count(device_family) > 0) {
        return true;
    }

    for (const auto& allowed_family : allowed_families) {
        if (allowed_family.size() > 1 && allowed_family.back() == 'X') {
            const std::string prefix = allowed_family.substr(0, allowed_family.size() - 1);
            if (device_family.compare(0, prefix.size(), prefix) == 0) {
                return true;
            }
        }
    }
    return false;
}

// Infer CUDA architecture from an NVIDIA marketing name when compute_cap is
// unavailable. Only architectures with published llamacpp-cuda binaries are
// represented here.
inline std::string identify_cuda_arch_from_name(const std::string& device_name) {
    std::string name = device_name;
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    static const std::vector<std::string> nvidia_ids = {
        "nvidia", "geforce", "rtx", "gtx", "quadro", "tesla", "titan",
        "a100", "a40", "a30", "a10", "h100", "h200", "b100", "b200",
        "l40", "gb10",
    };

    const bool is_nvidia = std::any_of(
        nvidia_ids.begin(), nvidia_ids.end(), [&](const std::string& id) {
            return name.find(id) != std::string::npos;
        });
    if (!is_nvidia) {
        return "";
    }

    // Data-center Blackwell must win before the generic "blackwell" keyword.
    if (name.find("b100") != std::string::npos ||
        name.find("b200") != std::string::npos) {
        return "sm_100";
    }

    static const std::vector<std::pair<std::string, std::vector<std::string>>> table = {
        {"sm_121", {"gb10"}},
        {"sm_100", {"b100", "b200"}},
        {"sm_120", {"blackwell", "rtx 50", "rtx50", "5090", "5080", "5070", "5060",
                    "rtx pro 6000", "rtx pro 5000", "rtx pro 4500", "rtx pro 4000",
                    "rtx pro 3500", "rtx pro 3000", "rtx pro 2000", "rtx pro 1000"}},
        {"sm_90",  {"h100", "h200"}},
        {"sm_89",  {"rtx 40", "rtx40", "4090", "4080", "4070", "4060", "l40", " l4"}},
        {"sm_80",  {"a100"}},
        {"sm_86",  {"rtx 30", "rtx30", "3090", "3080", "3070", "3060", "3050",
                    "a40", "a30", "a10", "a6000", "a5000", "a4000", "a2000"}},
        {"sm_75",  {"rtx 20", "rtx20", "2080", "2070", "2060",
                    "gtx 16", "gtx16", "1660", "1650",
                    "titan rtx", "quadro rtx", " t4"}},
    };

    for (const auto& [sm, keywords] : table) {
        for (const auto& keyword : keywords) {
            if (name.find(keyword) != std::string::npos) {
                return sm;
            }
        }
    }
    return "";
}

// KFD publishes a GPU's ISA as packed decimal (110501 -> gfx1151), where the minor and
// step fields are hex nibbles. Empty when the value is not a decimal integer; the CPU
// node's 0 yields the unmatchable "gfx000" rather than an error.
inline std::string gfx_target_version_to_arch(const std::string& gfx_target_version) {
    if (gfx_target_version.empty() ||
        !std::all_of(gfx_target_version.begin(), gfx_target_version.end(),
                     [](unsigned char c) { return std::isdigit(c) != 0; })) {
        return "";
    }

    int packed = 0;
    try {
        packed = std::stoi(gfx_target_version);
    } catch (const std::exception&) {
        return "";
    }

    char buf[16];
    std::snprintf(buf, sizeof(buf), "gfx%d%x%x",
                  packed / 10000, (packed / 100) % 100, packed % 100);
    return std::string(buf);
}

// Keeps the ISA visible next to the driver's marketing name, since users match it
// against the arch targets backends are published for. Either half may be missing:
// the marketing name needs a driver that reports one, and the ISA needs a
// recognizable gfx_target_version.
inline std::string gpu_display_name(const std::string& marketing_name,
                                    const std::string& arch) {
    if (marketing_name.empty()) {
        return arch;
    }
    if (arch.empty() || marketing_name == arch) {
        return marketing_name;
    }
    return marketing_name + " (" + arch + ")";
}

// Reads the amdgpu driver's own per-device accounting under `kfd_nodes_dir` (KFD
// topology) and `drm_dir` (/sys/class/drm). Both are world-readable, unlike the libdrm
// probe elsewhere in this file's callers, which needs O_RDWR on a render node and so
// answers wrongly for a service account outside the `render` group.
//
// Reports nothing unless exactly one GPU matches: two of the same ISA leave no way to say
// which one a launch lands on, and guessing would size against the wrong device.
inline bool rocm_device_memory_from_sysfs(const std::filesystem::path& kfd_nodes_dir,
                                          const std::filesystem::path& drm_dir,
                                          const std::string& arch,
                                          uint64_t& free_bytes,
                                          uint64_t& total_bytes) {
    namespace fs = std::filesystem;

    if (arch.empty()) {
        return false;
    }

    std::error_code ec;
    if (!fs::is_directory(kfd_nodes_dir, ec)) {
        return false;
    }

    std::string wanted = arch;
    std::transform(wanted.begin(), wanted.end(), wanted.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    std::vector<std::string> matching_render_minors;
    for (fs::directory_iterator it(kfd_nodes_dir, ec), end; it != end && !ec;
         it.increment(ec)) {
        std::ifstream props(it->path() / "properties");
        if (!props.is_open()) {
            continue;
        }

        std::string line;
        std::string gfx_target_version;
        std::string drm_render_minor;
        while (std::getline(props, line)) {
            const size_t space = line.find(' ');
            if (space == std::string::npos) {
                continue;
            }
            const std::string key = line.substr(0, space);
            std::string value = line.substr(space + 1);
            const size_t end_of_value = value.find_last_not_of(" \t\n\r");
            value = end_of_value == std::string::npos
                ? ""
                : value.substr(0, end_of_value + 1);

            if (key == "gfx_target_version") {
                gfx_target_version = value;
            } else if (key == "drm_render_minor") {
                drm_render_minor = value;
            }
        }

        // drm_render_minor -1 marks a GPU with no render node, which no userspace
        // runtime can open. The CPU node needs no special case: its packed version of 0
        // cannot match a real ISA.
        if (drm_render_minor.empty() || drm_render_minor == "-1") {
            continue;
        }
        if (gfx_target_version_to_arch(gfx_target_version) == wanted) {
            matching_render_minors.push_back(drm_render_minor);
        }
    }

    if (matching_render_minors.size() != 1) {
        return false;
    }

    auto read_u64 = [](const fs::path& path, uint64_t& value) {
        std::ifstream file(path);
        return file.is_open() && static_cast<bool>(file >> value);
    };

    const fs::path device_dir =
        drm_dir / ("renderD" + matching_render_minors.front()) / "device";
    uint64_t used = 0;
    uint64_t total = 0;
    if (!read_u64(device_dir / "mem_info_vram_used", used) ||
        !read_u64(device_dir / "mem_info_vram_total", total) ||
        total == 0 || used > total) {
        return false;
    }

    free_bytes = total - used;
    total_bytes = total;
    return true;
}

// The value of the first "flags" line of /proc/cpuinfo, or "" if none.
inline std::string read_cpuinfo_flags(std::istream& cpuinfo) {
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.compare(0, 5, "flags") != 0) continue;
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        return line.substr(colon + 1);
    }
    return "";
}

// CPU family tokens a support row can require beyond the base architecture
// (x86_64 / arm64), derived from the CPU feature flags.
inline std::vector<std::string> cpu_isa_families(const std::string& flags) {
    std::set<std::string> present;
    std::istringstream iss(flags);
    std::string flag;
    while (iss >> flag) present.insert(flag);

    std::vector<std::string> families;
    if (present.count("avx512f") && present.count("avx512_bf16")) {
        families.push_back("x86_64-avx512bf16");
    }
    return families;
}

// CPU ids in a sysfs cpulist such as "0-15,32,96-111". Malformed entries are skipped.
inline std::set<int> parse_cpu_list(const std::string& list) {
    std::set<int> cpus;
    std::istringstream iss(list);
    std::string entry;
    while (std::getline(iss, entry, ',')) {
        int first = 0;
        int last = 0;
        char dash = 0;
        std::istringstream range(entry);
        if (!(range >> first)) continue;
        if (range >> dash) {
            if (dash != '-' || !(range >> last) || last < first) continue;
        } else {
            last = first;
        }
        for (int cpu = first; cpu <= last; ++cpu) cpus.insert(cpu);
    }
    return cpus;
}

// How many NUMA nodes under `node_dir` (/sys/devices/system/node) hold at least one of
// `allowed_cpus`, or any CPU at all when `allowed_cpus` is empty. CPU-less nodes, such as
// CXL memory expanders, do not count.
inline int numa_nodes_spanned(const std::filesystem::path& node_dir,
                              const std::set<int>& allowed_cpus) {
    namespace fs = std::filesystem;

    int nodes = 0;
    std::error_code ec;
    for (fs::directory_iterator it(node_dir, ec), end; it != end && !ec; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.size() <= 4 || name.compare(0, 4, "node") != 0 ||
            !std::all_of(name.begin() + 4, name.end(),
                         [](unsigned char c) { return std::isdigit(c); })) {
            continue;
        }
        std::ifstream file(it->path() / "cpulist");
        std::string list;
        std::getline(file, list);
        const std::set<int> cpus = parse_cpu_list(list);
        if (allowed_cpus.empty() ? !cpus.empty()
                                 : std::any_of(cpus.begin(), cpus.end(), [&](int cpu) {
                                       return allowed_cpus.count(cpu) > 0;
                                   })) {
            ++nodes;
        }
    }
    return nodes;
}

}  // namespace lemon::system_info_detail
