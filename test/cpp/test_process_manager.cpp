#include <lemon/utils/process_manager.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>

#include <sys/wait.h>
#include <unistd.h>

using lemon::utils::ProcessHandle;
using lemon::utils::ProcessManager;

namespace {

int failures = 0;

void check(const char* name, bool condition) {
    std::printf("[%s] %s\n", condition ? "PASS" : "FAIL", name);
    if (!condition) {
        ++failures;
    }
}

ProcessHandle make_handle(pid_t pid) {
    return {nullptr, static_cast<int>(pid)};
}

pid_t spawn_exiting_child(int exit_code) {
    const pid_t pid = fork();
    if (pid == 0) {
        _exit(exit_code);
    }
    return pid;
}

pid_t spawn_running_child() {
    const pid_t pid = fork();
    if (pid == 0) {
        for (;;) {
            pause();
        }
    }
    return pid;
}

bool is_zombie(pid_t pid) {
    std::ifstream stat_file("/proc/" + std::to_string(pid) + "/stat");
    std::string stat_line;
    if (!std::getline(stat_file, stat_line)) {
        return false;
    }

    const auto close_paren = stat_line.rfind(')');
    return close_paren != std::string::npos &&
           close_paren + 2 < stat_line.size() &&
           stat_line[close_paren + 2] == 'Z';
}

bool wait_for_zombie(pid_t pid, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        if (is_zombie(pid)) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);

    return is_zombie(pid);
}

void kill_and_reap(pid_t pid) {
    if (pid <= 0) {
        return;
    }

    ::kill(pid, SIGKILL);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
}

bool wait_until_gone(pid_t pid, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        // An orphan is reparented and reaped by init, so a lingering zombie also counts as gone.
        if ((::kill(pid, 0) != 0 && errno == ESRCH) || is_zombie(pid)) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);

    return false;
}

// Starts `sh -c "sleep 60 & <then>"` through ProcessManager and returns the pid of the
// backgrounded sleep, which stands in for a helper a backend forks (e.g. vLLM's EngineCore).
pid_t start_shell_with_helper(const std::string& then, ProcessHandle& handle) {
    const auto pid_file = std::filesystem::temp_directory_path() /
                          ("lemonade_pm_test_" + std::to_string(getpid()));
    std::filesystem::remove(pid_file);

    handle = ProcessManager::start_process(
        "sh", {"-c", "sleep 60 & echo $! > '" + pid_file.string() + "'; " + then});

    pid_t helper = -1;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (helper <= 0 && std::chrono::steady_clock::now() < deadline) {
        std::ifstream in(pid_file);
        if (!(in >> helper)) {
            helper = -1;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    std::filesystem::remove(pid_file);
    return helper;
}

void check_helper_stopped(const char* name, pid_t helper) {
    const bool gone = wait_until_gone(helper, std::chrono::seconds(5));
    check(name, gone);
    if (!gone) {
        ::kill(helper, SIGKILL);
    }
}

} // namespace

int main() {
    {
        const pid_t child = spawn_exiting_child(42);
        check("fork exited child", child > 0);

        if (child > 0) {
            const bool zombie = wait_for_zombie(child, std::chrono::seconds(5));
            check("child reaches zombie state without reaping", zombie);

            if (zombie) {
                const ProcessHandle handle = make_handle(child);
                check("is_running() returns false for zombie",
                      !ProcessManager::is_running(handle));
                check("reap_process() preserves exit code 42",
                      ProcessManager::reap_process(handle) == 42);
            } else {
                kill_and_reap(child);
            }
        }
    }

    {
        const pid_t child = spawn_running_child();
        check("fork running child", child > 0);

        if (child > 0) {
            const ProcessHandle handle = make_handle(child);
            check("is_running() returns true for running child",
                  ProcessManager::is_running(handle));
            check("reap_process() does not reap running child",
                  ProcessManager::reap_process(handle) == -1);
            kill_and_reap(child);
        }
    }

    {
        ProcessHandle handle{};
        const pid_t helper = start_shell_with_helper("wait", handle);
        check("backend forks a helper", helper > 0);
        if (helper > 0) {
            ProcessManager::stop_process(handle);
            check_helper_stopped("stop_process() stops the backend's helpers", helper);
        } else {
            ProcessManager::kill_process(handle);
        }
    }

    {
        ProcessHandle handle{};
        const pid_t helper = start_shell_with_helper("exit 0", handle);
        check("exited backend leaves a helper", helper > 0);
        if (helper > 0) {
            wait_for_zombie(handle.pid, std::chrono::seconds(5));
            ProcessManager::stop_process(handle);
            check_helper_stopped("stop_process() stops helpers of an exited backend", helper);
        } else {
            ProcessManager::kill_process(handle);
        }
    }

    {
        ProcessHandle handle{};
        const pid_t helper = start_shell_with_helper("wait", handle);
        check("killed backend leaves a helper", helper > 0);
        ::kill(handle.pid, SIGKILL);
        wait_for_zombie(handle.pid, std::chrono::seconds(5));
        ProcessManager::reap_process(handle);
        if (helper > 0) {
            check_helper_stopped("reap_process() stops helpers of a killed backend", helper);
        }
    }

    {
        ProcessHandle handle{};
        const pid_t helper = start_shell_with_helper("wait", handle);
        check("backend forks a helper for kill_process()", helper > 0);
        ProcessManager::kill_process(handle);
        if (helper > 0) {
            check_helper_stopped("kill_process() stops the backend's helpers", helper);
        }
    }

    check("is_running() rejects PID 0",
          !ProcessManager::is_running(make_handle(0)));
    check("is_running() rejects negative PID",
          !ProcessManager::is_running(make_handle(-1)));
    check("is_running() rejects non-existent PID",
          !ProcessManager::is_running(
              make_handle(std::numeric_limits<int>::max())));

    if (failures == 0) {
        std::printf("\nAll process_manager tests passed\n");
        return 0;
    }

    std::printf("\n%d process_manager test(s) failed\n", failures);
    return 1;
}
