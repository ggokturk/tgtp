#pragma once
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace bench {
inline std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> result;
    std::stringstream input(text);
    std::string token;
    while (std::getline(input, token, ',')) {
        if (token.empty()) throw std::invalid_argument("empty CSV item");
        result.push_back(token);
    }
    if (result.empty() || text.back()==',') throw std::invalid_argument("empty CSV list/item");
    return result;
}
inline double number(const std::string& text) {
    size_t used=0;
    const double value=std::stod(text,&used);
    if (used!=text.size() || !std::isfinite(value)) throw std::invalid_argument("invalid number: "+text);
    return value;
}
inline uint32_t positive(const std::string& text) {
    const double value=number(text);
    if(value<1 || value>INT32_MAX || value!=std::floor(value)) throw std::invalid_argument("invalid positive integer: "+text);
    return static_cast<uint32_t>(value);
}
}
