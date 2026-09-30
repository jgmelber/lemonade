// Coverage for the per-model backend allowlist ("backends" in server_models.json):
// parsing it from ModelInfo extras, and the pure helpers the router and the model
// filter use to pick an allowed backend or reject a disallowed one.

#include "lemon/model_manager.h"

#include <cstdio>
#include <string>
#include <vector>

using lemon::ModelInfo;
using lemon::ModelManager;

static int g_failures = 0;

static void check(const std::string& name, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (!ok) ++g_failures;
}

int main() {
    using Backends = std::vector<std::string>;

    ModelInfo info;
    check("no allowlist parses as empty", info.allowed_backends().empty());
    info.extras["backends"] = nlohmann::json::array({"rocm"});
    check("allowlist parses from extras", info.allowed_backends() == Backends{"rocm"});
    info.extras["backends"] = "rocm";
    check("malformed allowlist is ignored", info.allowed_backends().empty());

    check("empty allowlist permits any backend", ModelManager::backend_allowed({}, "cpu-pace"));
    check("listed backend is permitted", ModelManager::backend_allowed({"rocm"}, "rocm"));
    check("unlisted backend is rejected", !ModelManager::backend_allowed({"rocm"}, "cpu-pace"));
    check("unresolved backend is rejected", !ModelManager::backend_allowed({"rocm"}, ""));

    // supported is in preference order, e.g. a Zen 4 host with a gfx1151 iGPU where
    // cpu-pace is the configured default.
    check("picks the first supported backend the allowlist permits",
          ModelManager::first_allowed_backend({"rocm"}, {"cpu-pace", "rocm"}) == "rocm");
    check("keeps preference order among permitted backends",
          ModelManager::first_allowed_backend({"rocm", "cpu-pace"}, {"cpu-pace", "rocm"}) ==
              "cpu-pace");
    check("no permitted backend supported -> empty",
          ModelManager::first_allowed_backend({"rocm"}, {"cpu-pace"}).empty());
    check("empty allowlist takes the preferred backend",
          ModelManager::first_allowed_backend({}, {"cpu-pace", "rocm"}) == "cpu-pace");

    check("describes one backend",
          ModelManager::describe_allowed_backends("vllm", {"rocm"}) == "the vllm backend rocm");
    check("describes several backends",
          ModelManager::describe_allowed_backends("vllm", {"rocm", "cpu-pace"}) ==
              "the vllm backends rocm, cpu-pace");

    if (g_failures == 0) {
        std::printf("All model backend allowlist tests passed.\n");
    } else {
        std::printf("%d model backend allowlist test(s) failed.\n", g_failures);
    }
    return g_failures == 0 ? 0 : 1;
}
