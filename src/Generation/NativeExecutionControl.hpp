#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace iiLocalDiffusion {
// Cooperatively park work at the engine's existing tensor/compute boundaries.
// A GPU call already submitted must finish. This does not grant background GPU
// permission or make an in-memory job survive process termination.
class NativeExecutionControl {
public:
    using Clock = std::chrono::steady_clock;
    void setPaused(bool paused) {
        {
            const std::lock_guard lock(m_mutex);
            if (m_paused == paused) return;
            const auto now = Clock::now();
            if (paused) m_pauseStarted = now;
            else m_pausedTime += now - m_pauseStarted;
            m_paused = paused;
        }
        m_resumed.notify_all();
    }
    bool isPaused() const { return m_paused.load(); }
    bool isWaiting() const { return m_waiters.load() > 0; }
    bool waitUntilRunnable(const std::atomic_bool &cancelled) {
        if (!m_paused || cancelled) return !cancelled;
        std::unique_lock lock(m_mutex);
        ++m_waiters;
        while (m_paused && !cancelled)
            m_resumed.wait_for(lock, std::chrono::milliseconds(20));
        --m_waiters;
        return !cancelled;
    }
    Clock::duration pausedDuration() const {
        const std::lock_guard lock(m_mutex);
        return m_pausedTime + (m_paused ? Clock::now() - m_pauseStarted : Clock::duration::zero());
    }
private:
    mutable std::mutex m_mutex;
    std::condition_variable m_resumed;
    std::atomic_bool m_paused{false};
    std::atomic_int m_waiters{0};
    Clock::time_point m_pauseStarted;
    Clock::duration m_pausedTime{};
};
}
