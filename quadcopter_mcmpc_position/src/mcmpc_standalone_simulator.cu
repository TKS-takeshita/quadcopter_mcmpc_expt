#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "quadcopter_mcmpc_position/mcmpc_controller.cuh"
#include "quadcopter_mcmpc_position/const_params.hpp"
#include "quadcopter_mcmpc_position/waypoint_config.hpp"

namespace
{
constexpr int kStateQ0 = 0;
constexpr int kStateWx = 4;
constexpr int kStateX = 7;
constexpr int kStateVx = 10;
constexpr int kSquareStartWaypointIndex = 1;

struct Options
{
    int steps = 300;
    std::string output =
        "visualize_mcmpc/simulator/mcmpc_sim_output.json";
    std::string waypoints;
    float waypoint_threshold = -1.0f;
    bool prediction_wall_collision = true;
    bool truth_wall_collision = true;
};

qc_mcmpc::WaypointConfig waypoint_config;

void check_cuda(cudaError_t err, const char* label)
{
    if (err != cudaSuccess) {
        std::cerr << label << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

Options parse_args(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            options.steps = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            options.output = argv[++i];
        } else if (std::strcmp(argv[i], "--waypoints") == 0 && i + 1 < argc) {
            options.waypoints = argv[++i];
        } else if (std::strcmp(argv[i], "--waypoint-threshold") == 0 && i + 1 < argc) {
            options.waypoint_threshold = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--wall-collision") == 0 && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "on") {
                options.prediction_wall_collision = true;
            } else if (value == "off") {
                options.prediction_wall_collision = false;
            } else {
                std::cerr << "--wall-collision must be on or off\n";
                std::exit(1);
            }
        } else if (std::strcmp(argv[i], "--truth-wall-collision") == 0 && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "on") {
                options.truth_wall_collision = true;
            } else if (value == "off") {
                options.truth_wall_collision = false;
            } else {
                std::cerr << "--truth-wall-collision must be on or off\n";
                std::exit(1);
            }
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout
                << "Usage: mcmpc_standalone_simulator "
                << "[--steps N] [--output path.json] "
                << "[--waypoints path.csv] [--waypoint-threshold metres] "
                << "[--wall-collision on|off] [--truth-wall-collision on|off]\n"
                << "  --wall-collision controls the MPC prediction model; truth defaults to on.\n";
            std::exit(0);
        }
    }
    return options;
}

float wrap_pi(float angle)
{
    while (angle > static_cast<float>(M_PI)) angle -= 2.0f * static_cast<float>(M_PI);
    while (angle < -static_cast<float>(M_PI)) angle += 2.0f * static_cast<float>(M_PI);
    return angle;
}

void set_target_yaw(float yaw)
{
    const float target_yaw = wrap_pi(yaw);
    const float half_yaw = 0.5f * target_yaw;
    qc_mcmpc::target_host.e0 = std::cos(half_yaw);
    qc_mcmpc::target_host.e1 = 0.0f;
    qc_mcmpc::target_host.e2 = 0.0f;
    qc_mcmpc::target_host.e3 = std::sin(half_yaw);
}

void set_square_waypoint(int index, float sim_time)
{
    if (index < kSquareStartWaypointIndex) {
        index = kSquareStartWaypointIndex;
    }
    if (index >= waypoint_config.count) {
        index = waypoint_config.count - 1;
    }
    square_waypoint_index = index;
    square_waypoint_change_time = sim_time;
    qc_mcmpc::target_host.x =
        waypoint_config.points[square_waypoint_index][0];
    qc_mcmpc::target_host.y =
        waypoint_config.points[square_waypoint_index][1];
    qc_mcmpc::target_host.z =
        waypoint_config.points[square_waypoint_index][2];
    qc_mcmpc::target_host.xp = 0.0f;
    qc_mcmpc::target_host.yp = 0.0f;
    qc_mcmpc::target_host.zp = 0.0f;
    qc_mcmpc::target_host.wx = 0.0f;
    qc_mcmpc::target_host.wy = 0.0f;
    qc_mcmpc::target_host.wz = 0.0f;
    set_target_yaw(0.0f);
    qc_mcmpc::update_target_state_device();
    check_cuda(
        cudaMemcpyToSymbol(
            qc_mcmpc::square_waypoint_index_device,
            &square_waypoint_index,
            sizeof(int)),
        "copy square_waypoint_index_device");
    check_cuda(
        cudaMemcpyToSymbol(
            qc_mcmpc::square_waypoint_change_time_device,
            &square_waypoint_change_time,
            sizeof(float)),
        "copy square_waypoint_change_time_device");
    check_cuda(
        cudaMemcpyToSymbol(qc_mcmpc::mcmpc_log_device, &mcmpc_log, sizeof(float)),
        "copy mcmpc_log_device");
}

void reset_model_context()
{
    float zeros3[3] = {0.0f, 0.0f, 0.0f};
    float zeros4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int motor_valid = 0;
    int takeoff_state = CONST_PARAM_FLOAT::TAKEOFF_STATE_FLIGHT;
    int landed = 0;
    int ground_contact = 0;
    int maybe_landed = 0;
    float takeoff_tilt_limit = CONST_PARAM_FLOAT::MPC_TILT_MAX;
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_velocity_device, zeros3, sizeof(zeros3)), "copy prev_velocity_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_acceleration_device, zeros3, sizeof(zeros3)), "copy prev_acceleration_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::vel_int_device, zeros3, sizeof(zeros3)), "copy vel_int_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::rate_int_device, zeros3, sizeof(zeros3)), "copy rate_int_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_angular_velocity_device, zeros3, sizeof(zeros3)), "copy prev_angular_velocity_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_angular_acceleration_device, zeros3, sizeof(zeros3)), "copy prev_angular_acceleration_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_motor_speed_device, zeros4, sizeof(zeros4)), "copy prev_motor_speed_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_motor_speed_valid_device, &motor_valid, sizeof(int)), "copy prev_motor_speed_valid_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::acceleration_bias_device, zeros3, sizeof(zeros3)), "copy acceleration_bias_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::takeoff_state_device, &takeoff_state, sizeof(int)), "copy takeoff_state_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::takeoff_tilt_limit_device, &takeoff_tilt_limit, sizeof(float)), "copy takeoff_tilt_limit_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::landed_device, &landed, sizeof(int)), "copy landed_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::ground_contact_device, &ground_contact, sizeof(int)), "copy ground_contact_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::maybe_landed_device, &maybe_landed, sizeof(int)), "copy maybe_landed_device");
}

void copy_next_context_from_deterministic_step(
    const float previous_state[_N_OF_ODES])
{
    float vel_int[3];
    float prev_acc[3];
    float rate_int[3];
    float prev_angular_acc[3];
    float motor_speed[4];
    float prev_velocity[3] = {
        previous_state[kStateVx],
        previous_state[kStateVx + 1],
        previous_state[kStateVx + 2],
    };
    float prev_angular_velocity[3] = {
        previous_state[kStateWx],
        previous_state[kStateWx + 1],
        previous_state[kStateWx + 2],
    };
    int motor_valid = 1;
    check_cuda(cudaMemcpyFromSymbol(vel_int, qc_mcmpc::deterministic_sim_vel_int_device, sizeof(vel_int)), "read deterministic vel_int");
    check_cuda(cudaMemcpyFromSymbol(prev_acc, qc_mcmpc::deterministic_sim_prev_acceleration_device, sizeof(prev_acc)), "read deterministic prev_acc");
    check_cuda(cudaMemcpyFromSymbol(rate_int, qc_mcmpc::deterministic_sim_rate_int_device, sizeof(rate_int)), "read deterministic rate_int");
    check_cuda(cudaMemcpyFromSymbol(prev_angular_acc, qc_mcmpc::deterministic_sim_prev_angular_acceleration_device, sizeof(prev_angular_acc)), "read deterministic prev_angular_acc");
    check_cuda(cudaMemcpyFromSymbol(motor_speed, qc_mcmpc::deterministic_sim_motor_speed_device, sizeof(motor_speed)), "read deterministic motor_speed");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_velocity_device, prev_velocity, sizeof(prev_velocity)), "copy next prev_velocity_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_acceleration_device, prev_acc, sizeof(prev_acc)), "copy next prev_acceleration_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::vel_int_device, vel_int, sizeof(vel_int)), "copy next vel_int_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::rate_int_device, rate_int, sizeof(rate_int)), "copy next rate_int_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_angular_velocity_device, prev_angular_velocity, sizeof(prev_angular_velocity)), "copy next prev_angular_velocity_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_angular_acceleration_device, prev_angular_acc, sizeof(prev_angular_acc)), "copy next prev_angular_acceleration_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_motor_speed_device, motor_speed, sizeof(motor_speed)), "copy next prev_motor_speed_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_motor_speed_valid_device, &motor_valid, sizeof(int)), "copy next prev_motor_speed_valid_device");
}

void write_json_state(
    std::ofstream& out,
    int step,
    float time,
    const float state[_N_OF_ODES],
    double input_x,
    double input_y,
    double input_z,
    double input_yaw,
    float cost,
    bool collision,
    const float contact_point[3],
    float collision_impulse,
    float pre_contact_normal_velocity,
    const float prediction[_DEVICE_CONST_HORIZON][_N_OF_ODES],
    const qc_mcmpc::input_array& input_prediction)
{
    if (step > 0) {
        out << ",\n";
    }
    out << "    {"
        << "\"t\":" << time
        << ",\"x\":" << state[kStateX]
        << ",\"y\":" << state[kStateX + 1]
        << ",\"z\":" << state[kStateX + 2]
        << ",\"vx\":" << state[kStateVx]
        << ",\"vy\":" << state[kStateVx + 1]
        << ",\"vz\":" << state[kStateVx + 2]
        << ",\"q0\":" << state[kStateQ0]
        << ",\"q1\":" << state[kStateQ0 + 1]
        << ",\"q2\":" << state[kStateQ0 + 2]
        << ",\"q3\":" << state[kStateQ0 + 3]
        << ",\"wx\":" << state[kStateWx]
        << ",\"wy\":" << state[kStateWx + 1]
        << ",\"wz\":" << state[kStateWx + 2]
        << ",\"ux\":" << input_x
        << ",\"uy\":" << input_y
        << ",\"uz\":" << input_z
        << ",\"uyaw\":" << input_yaw
        << ",\"target_x\":" << qc_mcmpc::target_host.x
        << ",\"target_y\":" << qc_mcmpc::target_host.y
        << ",\"target_z\":" << qc_mcmpc::target_host.z
        << ",\"waypoint_index\":" << square_waypoint_index
        << ",\"cost\":" << cost
        << ",\"collision\":" << (collision ? "true" : "false")
        << ",\"contact_point\":[" << contact_point[0] << ","
        << contact_point[1] << "," << contact_point[2] << "]"
        << ",\"collision_impulse\":" << collision_impulse
        << ",\"pre_contact_normal_velocity\":" << pre_contact_normal_velocity
        << ",\"prediction\":[";
    for (int h = 0; h < _DEVICE_CONST_HORIZON; h++) {
        if (h > 0) {
            out << ",";
        }
        out << "[";
        for (int state_index = 0; state_index < _N_OF_ODES; state_index++) {
            if (state_index > 0) {
                out << ",";
            }
            out << prediction[h][state_index];
        }
        out << "]";
    }
    out << "],\"input_prediction\":[";
    for (int h = 0; h < _DEVICE_CONST_HORIZON; h++) {
        if (h > 0) {
            out << ",";
        }
        out << "["
            << input_prediction.decoupled_position[h][x] << ","
            << input_prediction.decoupled_position[h][y] << ","
            << input_prediction.decoupled_position[h][z] << ","
            << input_prediction.decoupled_position[h][yaw] << "]";
    }
    out << "]}";
}
}

int main(int argc, char** argv)
{
    const Options options = parse_args(argc, argv);
    qc_mcmpc::mcmpc_controller& controller =
        qc_mcmpc::mcmpc_controller::get_instance();
    try {
        waypoint_config = options.waypoints.empty()
            ? qc_mcmpc::default_waypoint_config()
            : qc_mcmpc::load_waypoint_csv(options.waypoints);
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    if (options.waypoint_threshold > 0.0f) {
        waypoint_config.threshold = options.waypoint_threshold;
    }
    square_waypoint_count = waypoint_config.count;
    square_waypoint_threshold = waypoint_config.threshold;
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::square_waypoints_device, waypoint_config.points.data(), sizeof(waypoint_config.points)), "copy runtime waypoints");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::square_waypoint_count_device, &square_waypoint_count, sizeof(int)), "copy runtime waypoint count");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::square_waypoint_threshold_device, &square_waypoint_threshold, sizeof(float)), "copy runtime waypoint threshold");
#ifdef PREDICTABLE_COLLISION_WITH_WALL
    const int prediction_wall_collision_enabled =
        options.prediction_wall_collision ? 1 : 0;
    const int truth_wall_collision_enabled = options.truth_wall_collision ? 1 : 0;
    check_cuda(cudaMemcpyToSymbol(
        qc_mcmpc::prediction_wall_collision_enabled_device,
        &prediction_wall_collision_enabled,
        sizeof(prediction_wall_collision_enabled)), "copy prediction wall collision mode");
    check_cuda(cudaMemcpyToSymbol(
        qc_mcmpc::truth_wall_collision_enabled_device,
        &truth_wall_collision_enabled,
        sizeof(truth_wall_collision_enabled)), "copy truth wall collision mode");
#endif
    reset_model_context();

    float state[_N_OF_ODES] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f,
        0.0f, 0.0f, 0.0f,
    };
    mcmpc_log = 0.0f;
    set_square_waypoint(kSquareStartWaypointIndex, mcmpc_log);

    const std::filesystem::path output_path(options.output);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream out(options.output);
    if (!out) {
        std::cerr << "failed to open output: " << options.output << std::endl;
        return 1;
    }
    out << "{\n"
        << "  \"dt\":" << CONST_PARAM_FLOAT::CONTROL_PERIOD << ",\n"
        << "  \"horizon\":" << _DEVICE_CONST_HORIZON << ",\n"
        << "  \"wall_collision_enabled\":"
        << (options.prediction_wall_collision ? "true" : "false") << ",\n"
        << "  \"prediction_wall_collision_enabled\":"
        << (options.prediction_wall_collision ? "true" : "false") << ",\n"
        << "  \"truth_wall_collision_enabled\":"
        << (options.truth_wall_collision ? "true" : "false") << ",\n"
        << "  \"waypoint_threshold\":" << waypoint_config.threshold << ",\n"
        << "  \"waypoints\":[\n";
    for (int i = 0; i < waypoint_config.count; i++) {
        if (i > 0) {
            out << ",\n";
        }
        out << "    {\"index\":" << i
            << ",\"x\":" << waypoint_config.points[i][0]
            << ",\"y\":" << waypoint_config.points[i][1]
            << ",\"z\":" << waypoint_config.points[i][2]
            << "}";
    }
    out << "\n  ],\n"
        << "  \"samples\":[\n";

    for (int step = 0; step < options.steps; step++) {
        const float dx = qc_mcmpc::target_host.x - state[kStateX];
        const float dy = qc_mcmpc::target_host.y - state[kStateX + 1];
        const float threshold = waypoint_config.threshold;
        if (dx * dx + dy * dy < threshold * threshold &&
            square_waypoint_index + 1 < waypoint_config.count) {
            set_square_waypoint(square_waypoint_index + 1, mcmpc_log);
        }

        check_cuda(
            cudaMemcpyToSymbol(qc_mcmpc::mcmpc_log_device, &mcmpc_log, sizeof(float)),
            "copy mcmpc_log_device");

        double input_x = 0.0;
        double input_y = 0.0;
        double input_z = 0.0;
        double input_yaw = 0.0;
        const float cost = controller.calc_optimal_input(
            state,
            input_x,
            input_y,
            input_z,
            input_yaw);

        qc_mcmpc::input_array best_input;
        controller.copy_best_input_array(best_input);
        qc_mcmpc::simulate_best_input_trajectory_kernel<<<1, 1>>>(best_input);
        check_cuda(cudaGetLastError(), "launch simulate_best_input_trajectory_kernel");
        check_cuda(cudaDeviceSynchronize(), "sync simulate_best_input_trajectory_kernel");

        const float previous_state[_N_OF_ODES] = {
            state[0], state[1], state[2], state[3],
            state[4], state[5], state[6],
            state[7], state[8], state[9],
            state[10], state[11], state[12],
        };
        float prediction[_DEVICE_CONST_HORIZON][_N_OF_ODES];
        check_cuda(
            cudaMemcpyFromSymbol(
                prediction,
                qc_mcmpc::deterministic_sim_trajectory_device,
                sizeof(prediction)),
            "read deterministic prediction");
        std::copy(prediction[0], prediction[0] + _N_OF_ODES, state);
        copy_next_context_from_deterministic_step(previous_state);

        int collision_flag = 0;
        float contact_point[3] = {0.0f, 0.0f, 0.0f};
        float collision_impulse = 0.0f;
        float pre_contact_normal_velocity = 0.0f;
#ifdef PREDICTABLE_COLLISION_WITH_WALL
        check_cuda(cudaMemcpyFromSymbol(
            &collision_flag, qc_mcmpc::deterministic_sim_collision_device,
            sizeof(collision_flag)), "read deterministic collision flag");
        check_cuda(cudaMemcpyFromSymbol(
            contact_point, qc_mcmpc::deterministic_sim_contact_point_device,
            sizeof(contact_point)), "read deterministic contact point");
        check_cuda(cudaMemcpyFromSymbol(
            &collision_impulse, qc_mcmpc::deterministic_sim_impulse_device,
            sizeof(collision_impulse)), "read deterministic collision impulse");
        check_cuda(cudaMemcpyFromSymbol(
            &pre_contact_normal_velocity,
            qc_mcmpc::deterministic_sim_pre_contact_normal_velocity_device,
            sizeof(pre_contact_normal_velocity)),
            "read deterministic pre-contact normal velocity");
#endif
        if (collision_flag && square_waypoint_index + 1 < waypoint_config.count) {
            set_square_waypoint(square_waypoint_index + 1, mcmpc_log);
        }

        write_json_state(
            out,
            step,
            mcmpc_log + CONST_PARAM_FLOAT::CONTROL_PERIOD,
            state,
            input_x,
            input_y,
            input_z,
            input_yaw,
            cost,
            collision_flag != 0,
            contact_point,
            collision_impulse,
            pre_contact_normal_velocity,
            prediction,
            best_input);

        mcmpc_log += CONST_PARAM_FLOAT::CONTROL_PERIOD;
    }

    out << "\n  ]\n}\n";
    std::cout << "wrote " << options.output << std::endl;
    return 0;
}
