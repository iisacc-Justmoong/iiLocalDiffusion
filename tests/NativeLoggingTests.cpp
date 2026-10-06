#include <stable-diffusion.h>
#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include <iostream>

void log_printf(sd_log_level_t, const char *, int, const char *, ...);
thread_local int expectedWorker = -1;
thread_local int expectedStep = -1;
int main() {
    std::atomic_int failures{0};
    sd_set_log_callback([](sd_log_level_t, const char *text, void *data) {
        const std::string line(text);
        const auto expected = std::to_string(expectedWorker) + ':' + std::to_string(expectedStep)
            + ':' + std::string(512, 'a' + expectedWorker);
        const auto start = line.find(" - ");
        if (start == std::string::npos || line.substr(start + 3) != expected + '\n')
            ++*static_cast<std::atomic_int *>(data);
    }, &failures);
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 8; ++worker) workers.emplace_back([worker] {
        for (int step = 0; step < 2000; ++step) {
            expectedWorker = worker; expectedStep = step;
            const auto expected = std::to_string(worker) + ':' + std::to_string(step) + ':' + std::string(512, 'a' + worker);
            log_printf(SD_LOG_INFO, "parallel", 17, "%s", expected.c_str());
        }
    });
    for (auto &worker : workers) worker.join();
    sd_set_log_callback(nullptr, nullptr);
    std::cout << "parallel log corruption=" << failures << '\n';
    return failures ? 1 : 0;
}
