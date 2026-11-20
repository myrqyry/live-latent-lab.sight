#include "litert_model_manager.hpp"

#include <openssl/sha.h>

#include <array>
#include <cctype>
#include <fstream>
#include <sstream>
#include <system_error>

namespace lll {
namespace {

std::string shell_quote(const std::string &value)
{
    std::string quoted = "'";
    for (const char character : value) {
        if (character == '\'') {
            quoted += "'\\''";
        } else {
            quoted += character;
        }
    }
    return quoted + "'";
}

bool copy_file_url(const std::string &url,
                   const std::filesystem::path &destination,
                   std::string *error)
{
    constexpr std::string_view prefix = "file://";
    const auto source = url.starts_with(prefix)
                            ? std::filesystem::path(url.substr(prefix.size()))
                            : std::filesystem::path(url);
    std::error_code ec;
    std::filesystem::copy_file(source, destination,
                               std::filesystem::copy_options::overwrite_existing,
                               ec);
    if (ec) {
        if (error) {
            *error = "failed to copy model: " + ec.message();
        }
        return false;
    }
    return true;
}

bool download_with_curl(const std::string &url,
                        const std::filesystem::path &destination,
                        std::string *error)
{
    const std::string command = "curl --fail --location --silent --show-error "
                                "--retry 2 --output " +
                                shell_quote(destination.string()) + " " +
                                shell_quote(url);
    if (std::system(command.c_str()) == 0) {
        return true;
    }
    if (error) {
        *error = "curl failed while downloading model";
    }
    return false;
}

bool default_download(const std::string &url,
                      const std::filesystem::path &destination,
                      std::string *error)
{
    if (url.starts_with("file://") || url.find("://") == std::string::npos) {
        return copy_file_url(url, destination, error);
    }
    if (url.starts_with("https://") || url.starts_with("http://")) {
        return download_with_curl(url, destination, error);
    }
    if (error) {
        *error = "unsupported model URL scheme";
    }
    return false;
}

std::string lowercase(std::string value)
{
    for (char &character : value) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    return value;
}

std::string sha256_file(const std::filesystem::path &path, std::string *error)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        if (error) {
            *error = "failed to open downloaded model for verification";
        }
        return {};
    }

    SHA256_CTX context;
    SHA256_Init(&context);
    std::array<char, 64 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) {
            SHA256_Update(&context, buffer.data(), static_cast<size_t>(count));
        }
    }
    if (!input.eof()) {
        if (error) {
            *error = "failed while reading downloaded model";
        }
        return {};
    }

    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256_Final(digest.data(), &context);
    std::ostringstream result;
    result.setf(std::ios::hex, std::ios::basefield);
    result.fill('0');
    for (const auto byte : digest) {
        result.width(2);
        result << static_cast<unsigned int>(byte);
    }
    return result.str();
}

}  // namespace

LiteRtModelManager::LiteRtModelManager(std::filesystem::path model_dir,
                                       Downloader downloader)
    : model_dir_(std::move(model_dir)),
      downloader_(downloader ? std::move(downloader) : default_download)
{
}

std::optional<std::filesystem::path> LiteRtModelManager::install(
    const LiteRtModelSpec &spec, std::string *error)
{
    if (spec.id.empty() || spec.filename.empty() || spec.url.empty()) {
        if (error) {
            *error = "model id, filename, and URL are required";
        }
        return std::nullopt;
    }
    if (spec.sha256.empty()) {
        if (error) {
            *error = "SHA-256 is required before installing a model";
        }
        return std::nullopt;
    }
    if (std::filesystem::path(spec.filename).filename() != spec.filename) {
        if (error) {
            *error = "model filename must not contain a directory";
        }
        return std::nullopt;
    }

    std::error_code ec;
    std::filesystem::create_directories(model_dir_, ec);
    if (ec) {
        if (error) {
            *error = "failed to create model directory: " + ec.message();
        }
        return std::nullopt;
    }

    const auto final_path = model_dir_ / spec.filename;
    const auto temporary_path = final_path.string() + ".part";
    std::filesystem::remove(temporary_path, ec);

    std::string download_error;
    if (!downloader_(spec.url, temporary_path, &download_error)) {
        std::filesystem::remove(temporary_path, ec);
        if (error) {
            *error = download_error;
        }
        return std::nullopt;
    }

    std::string hash_error;
    const auto actual_hash = sha256_file(temporary_path, &hash_error);
    if (!hash_error.empty() || lowercase(actual_hash) != lowercase(spec.sha256)) {
        std::filesystem::remove(temporary_path, ec);
        if (error) {
            *error = hash_error.empty() ? "model SHA-256 mismatch" : hash_error;
        }
        return std::nullopt;
    }

    std::filesystem::rename(temporary_path, final_path, ec);
    if (ec) {
        std::filesystem::remove(temporary_path, ec);
        if (error) {
            *error = "failed to atomically install model: " + ec.message();
        }
        return std::nullopt;
    }
    return final_path;
}

std::optional<std::filesystem::path> LiteRtModelManager::resolve(
    const std::string &model_id) const
{
    const auto *spec = find_litert_model(model_id);
    if (!spec) {
        return std::nullopt;
    }
    const auto path = model_dir_ / spec->filename;
    if (!std::filesystem::is_regular_file(path)) {
        return std::nullopt;
    }
    return path;
}

}  // namespace lll
