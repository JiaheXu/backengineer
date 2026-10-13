#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

// SDK 5.5.0 only: private decoder located from the .model2 magic check and
// the caller reporting "v2 model decrypt/inflate failed". No GPU inference.
int main(int argc, char **argv) {
    if (argc != 3) { std::cerr << "Usage: extract_model INPUT OUTPUT\n"; return 2; }
    void *lib = dlopen("/usr/local/zed/lib/libsl_ai.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::cerr << dlerror() << '\n'; return 1; }
    Dl_info info{};
    if (!dladdr(dlsym(lib, "loadDisparityInferenceModule"), &info)) return 1;
    auto *base = static_cast<unsigned char *>(info.dli_fbase);
    // Fail closed if this build's function signature bytes have changed.
    if (base[0x191ab0] != 0xf3 || base[0x191ab4] != 0x55 ||
        base[0x191ae2] != 0x81 || base[0x191ae4] != 0x89) return 1;
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) return 1;
    std::vector<unsigned char> data{std::istreambuf_iterator<char>(input), {}};
    std::vector<unsigned char> output;
    using Decoder = int (*)(const unsigned char *, size_t, std::vector<unsigned char> *, void *);
    int rc = reinterpret_cast<Decoder>(base + 0x191ab0)(data.data(), data.size(), &output, nullptr);
    std::cerr << "decoder rc=" << rc << ", bytes=" << output.size() << '\n';
    if (rc || output.empty()) return 1;
    std::ofstream file(argv[2], std::ios::binary | std::ios::out | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(output.data()), output.size());
    return file ? 0 : 1;
}
