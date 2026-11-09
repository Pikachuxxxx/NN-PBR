#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <filesystem>
#include <string>

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR arguments, int) {
    wchar_t path[32768];
    const DWORD count = GetModuleFileNameW(nullptr, path, 32768);
    if (!count || count == 32768) return 1;
    const auto folder = std::filesystem::path(path).parent_path();
    const auto runtime = folder / L"runtime";
    const auto python = runtime / L"pythonw.exe";
    SetEnvironmentVariableW(L"PYTHONHOME", runtime.c_str());
    SetEnvironmentVariableW(L"PYTHONPATH", nullptr);
    SetEnvironmentVariableW(L"PYTHONNOUSERSITE", L"1");
    std::wstring command = L"\"" + python.wstring() + L"\" -m nnpbr.studio ";
    command += *arguments ? arguments : L"--export-dir models/hammered-copper-512/export";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(python.c_str(), command.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW, nullptr, folder.c_str(), &startup, &process)) {
        MessageBoxW(nullptr, L"Could not start the bundled runtime. Keep the runtime directory beside this executable.",
                    L"NN-PBR Studio", MB_ICONERROR);
        return 1;
    }
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hProcess);
    return static_cast<int>(code);
}
