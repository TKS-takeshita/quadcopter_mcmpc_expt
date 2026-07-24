#pragma once

#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "quadcopter_mcmpc_position/const_params.hpp"

namespace qc_mcmpc
{
struct WaypointConfig
{
    std::array<std::array<float, 3>, _SQUARE_WAYPOINTS> points{};
    int count = 0;
    float threshold = 0.05f;
};

inline WaypointConfig default_waypoint_config()
{
    WaypointConfig config;
    config.count = square_waypoint_count;
    config.threshold = square_waypoint_threshold;
    for (int i = 0; i < config.count; i++) {
        for (int axis = 0; axis < 3; axis++) {
            config.points[i][axis] = CONST_PARAM_FLOAT::square_waypoints[i][axis];
        }
    }
    return config;
}

inline WaypointConfig load_waypoint_csv(const std::string& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open waypoint file: " + path);
    }
    WaypointConfig config;
    config.threshold = square_waypoint_threshold;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        line_number++;
        if (line.empty() || line[0] == '#') continue;
        for (char& ch : line) if (ch == ',') ch = ' ';
        std::istringstream stream(line);
        std::string first;
        if (!(stream >> first)) continue;
        if (first == "threshold") {
            if (!(stream >> config.threshold) || config.threshold <= 0.0f) {
                throw std::runtime_error("invalid threshold at line " + std::to_string(line_number));
            }
            continue;
        }
        if (first == "x") continue;
        if (config.count >= _SQUARE_WAYPOINTS) {
            throw std::runtime_error("too many waypoints; maximum is " + std::to_string(_SQUARE_WAYPOINTS));
        }
        try {
            config.points[config.count][0] = std::stof(first);
        } catch (...) {
            throw std::runtime_error("invalid x at line " + std::to_string(line_number));
        }
        if (!(stream >> config.points[config.count][1] >> config.points[config.count][2])) {
            throw std::runtime_error("expected x,y,z at line " + std::to_string(line_number));
        }
        config.count++;
    }
    if (config.count < 2) {
        throw std::runtime_error("waypoint file must contain at least two points");
    }
    return config;
}
}
