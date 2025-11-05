#pragma once

#include <filesystem>

namespace FileUtils {
std::vector<char> readFile(const std::string& filename);
}
