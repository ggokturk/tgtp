#include "aesc.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <folder> <graph> <epsilon> <omega> <gamma> <outfile>\n";
        return 1;
    }
    try {
        AESCConfig config{};
        config.strFolder = argv[1];
        config.strGraph = argv[2];
        auto number = [](const char* arg) {
            std::size_t used;
            double value = std::stod(arg, &used);
            if (arg[used] != '\0' || !std::isfinite(value))
                throw std::invalid_argument("invalid numeric argument");
            return value;
        };
        config.epsilon = number(argv[3]);
        double omega = number(argv[4]), gamma = number(argv[5]);
        if (!(config.epsilon > 0 && config.epsilon < 1) ||
            omega < 2 || omega > INT_MAX || omega != std::floor(omega) ||
            gamma < 1 || gamma > INT_MAX || gamma != std::floor(gamma))
            throw std::invalid_argument("require 0 < epsilon < 1, integer omega >= 2, integer gamma >= 1");
        config.omega = static_cast<int>(omega);
        config.gamma = static_cast<int>(gamma);
        auto directory = std::filesystem::path(config.strFolder) / config.strGraph;
        AESCGraph graph((directory / "graph.txt").string(),
            (directory / ("sorted_eigens_" + std::to_string(config.omega) + ".txt")).string(), config.omega);
        config.delta = 1.0 / graph.n;
        std::ofstream out(argv[6], std::ios::binary);
        if (!out) throw std::runtime_error("cannot open output file");
        AESC algo(graph, config);
        std::unique_ptr<float[]> scores(algo.tgtp());
        out.write(reinterpret_cast<const char*>(scores.get()), graph.m * sizeof(float));
        out.close();
        if (!out) throw std::runtime_error("failed to write output file");
        graph.free();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tgtp: " << error.what() << '\n';
        return 1;
    }
}
