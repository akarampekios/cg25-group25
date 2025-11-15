#include <fstream>
#include <iostream>
#include <stdexcept>
#include <filesystem>

#include "FileUtils.hpp"

std::vector<char> FileUtils::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::in | std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }

    std::streamsize const fileSize = static_cast<std::streamsize>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}
