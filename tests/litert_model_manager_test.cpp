#include "core/litert_model_manager.hpp"

#include <openssl/sha.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::string sha256(const std::string &value)
{
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), digest);
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (const auto byte : digest) {
        result += hex[byte >> 4];
        result += hex[byte & 0x0f];
    }
    return result;
}

}  // namespace

int main()
{
    const auto &catalog = lll::litert_model_catalog();
    assert(catalog.size() >= 10);
    for (const auto &spec : catalog) {
        assert(!spec.id.empty());
        assert(spec.url.rfind("https://", 0) == 0);
        assert(spec.filename.ends_with(".tflite"));
        assert(spec.sha256.size() == 64);
    }

    const auto root = std::filesystem::temp_directory_path() / "litert-model-manager-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    const auto source = root / "source.tflite";
    {
        std::ofstream output(source, std::ios::binary);
        output << "test model";
    }

    lll::LiteRtModelManager manager(root / "models");
    const lll::LiteRtModelSpec valid{
        "test", "Test", "file://" + source.string(), sha256("test model"), "test.tflite"};
    std::string error;
    const auto installed = manager.install(valid, &error);
    assert(installed);
    assert(std::filesystem::is_regular_file(*installed));
    assert(!std::filesystem::exists(installed->string() + ".part"));

    const auto mismatch = valid;
    const lll::LiteRtModelSpec invalid{
        mismatch.id, mismatch.display_name, mismatch.url, std::string(64, '0'), mismatch.filename};
    assert(!manager.install(invalid, &error));
    assert(error.find("mismatch") != std::string::npos);
    assert(std::filesystem::is_regular_file(*installed));

    assert(!manager.install({"test", "Test", valid.url, {}, "bad.tflite"}, &error));
    assert(error.find("SHA-256") != std::string::npos);
    std::filesystem::remove_all(root);
    return 0;
}
