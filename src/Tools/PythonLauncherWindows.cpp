#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::wstring wide(const char *text) {
    const int size = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    if (!size) return {};
    std::wstring value(size, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text, -1, value.data(), size);
    value.resize(size - 1);
    return value;
}
std::wstring quote(const std::wstring &value) {
    std::wstring result = L"\"";
    unsigned slashes = 0;
    for (const wchar_t c : value) {
        if (c == L'\\') { ++slashes; continue; }
        result.append(c == L'\"' ? 2 * slashes + 1 : slashes, L'\\');
        slashes = 0;
        result += c;
    }
    result.append(2 * slashes, L'\\');
    return result + L'\"';
}
}

int wmain(int argc, wchar_t **argv) {
    std::wstring module(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, module.data(), DWORD(module.size()));
    if (!length || length == module.size()) return 127;
    module.resize(length);
    const auto script = std::filesystem::path(module).parent_path() / wide(IILD_SCRIPT_FILENAME);
    const auto override = _wgetenv(L"IILD_PYTHON_EXECUTABLE");
    const std::wstring python = override && *override ? override : wide(IILD_DEFAULT_PYTHON);
    std::wstring command = quote(python) + L" " + quote(script.native());
    for (int index = 1; index < argc; ++index) command += L" " + quote(argv[index]);
    STARTUPINFOW startup{}; startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION process{};
    const HANDLE job = CreateJobObjectW(nullptr, nullptr);
    struct CloseJob { HANDLE value; ~CloseJob() { if (value) CloseHandle(value); } } closeJob{job};
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!job || !SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) return 127;
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED, nullptr, nullptr, &startup, &process)) {
        std::cerr << "Cannot start iiLocalDiffusion Python launcher (Windows error " << GetLastError()
                  << "). Set IILD_PYTHON_EXECUTABLE to a Python 3.10+ executable.\n";
        return 127;
    }
    if (!AssignProcessToJobObject(job, process.hProcess) || ResumeThread(process.hThread) == DWORD(-1)) {
        TerminateProcess(process.hProcess, 127);
        CloseHandle(process.hThread); CloseHandle(process.hProcess);
        return 127;
    }
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hProcess);
    return static_cast<int>(code);
}
