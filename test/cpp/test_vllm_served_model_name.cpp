// Every request VLLMServer forwards must carry the canonical --served-model-name,
// whatever model id the client used (e.g. a user.* model's bare public alias).

#include <httplib.h>

#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#include "lemon/backends/vllm/vllm_server.h"

namespace {

const std::string kServedName = "user.Alias-Test";

class TestVLLMServer : public lemon::backends::VLLMServer {
public:
    explicit TestVLLMServer(int port) : VLLMServer("error", nullptr, nullptr) {
        port_ = port;
        set_state(lemon::ModelState::READY);
    }

    bool is_backend_alive() const override { return true; }
};

}  // namespace

int main() {
    int failures = 0;
    auto check = [&failures](bool condition, const std::string& name) {
        std::printf("[%s] %s\n", condition ? "PASS" : "FAIL", name.c_str());
        if (!condition) {
            ++failures;
        }
    };

    std::mutex mutex;
    std::map<std::string, std::string> received_models;
    auto record = [&](const httplib::Request& req, httplib::Response& res) {
        const auto body = nlohmann::json::parse(req.body, nullptr, false);
        {
            std::lock_guard<std::mutex> lock(mutex);
            const bool stream = body.value("stream", false);
            received_models[req.path + (stream ? " (stream)" : "")] =
                body.is_object() && body.contains("model") && body["model"].is_string()
                    ? body["model"].get<std::string>()
                    : "<missing>";
        }
        if (body.value("stream", false)) {
            res.set_content("data: [DONE]\n\n", "text/event-stream");
        } else {
            res.set_content(R"({"choices":[]})", "application/json");
        }
    };

    httplib::Server backend;
    backend.Post("/v1/chat/completions", record);
    backend.Post("/v1/completions", record);
    backend.Post("/v1/responses", record);
    const int port = backend.bind_to_any_port("127.0.0.1");
    std::thread backend_thread([&backend]() { backend.listen_after_bind(); });
    backend.wait_until_ready();

    {
        TestVLLMServer server(port);
        server.set_model_metadata(kServedName, "org/checkpoint", lemon::ModelType::LLM,
                                  lemon::DeviceType::DEVICE_GPU, lemon::RecipeOptions());

        const nlohmann::json chat = {
            {"model", "Alias-Test"},
            {"messages", {{{"role", "user"}, {"content", "hi"}}}},
        };
        const nlohmann::json completion = {{"model", "Alias-Test"}, {"prompt", "hi"}};
        const nlohmann::json responses = {{"model", "Alias-Test"}, {"input", "hi"}};

        server.chat_completion(chat);
        server.completion(completion);
        server.responses(responses);

        httplib::DataSink sink;
        sink.write = [](const char*, size_t) { return true; };
        sink.done = []() {};
        sink.is_writable = []() { return true; };

        auto stream = [](nlohmann::json request) {
            request["stream"] = true;
            return request.dump();
        };
        server.forward_streaming_request("/v1/chat/completions", stream(chat), sink);
        server.forward_streaming_request("/v1/completions", stream(completion), sink);
        server.forward_streaming_request("/v1/responses", stream(responses), sink);
    }

    backend.stop();
    backend_thread.join();

    for (const std::string& path :
         {"/v1/chat/completions", "/v1/completions", "/v1/responses",
          "/v1/chat/completions (stream)", "/v1/completions (stream)",
          "/v1/responses (stream)"}) {
        const auto it = received_models.find(path);
        const std::string model = it == received_models.end() ? "<not received>" : it->second;
        check(model == kServedName, path + " forwards model '" + model + "'");
    }

    return failures == 0 ? 0 : 1;
}
