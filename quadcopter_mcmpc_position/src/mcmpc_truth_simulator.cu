#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <px4_msgs/msg/actuator_motors.hpp>
#include <px4_msgs/msg/hover_thrust_estimate.hpp>
#include <px4_msgs/msg/rate_ctrl_status.hpp>
#include <px4_msgs/msg/takeoff_status.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_angular_velocity.hpp>
#include <px4_msgs/msg/vehicle_attitude_setpoint.hpp>
#include <px4_msgs/msg/vehicle_land_detected.hpp>
#include <px4_msgs/msg/vehicle_local_position_setpoint.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_rates_setpoint.hpp>
#include <px4_msgs/msg/vehicle_torque_setpoint.hpp>

#include "quadcopter_mcmpc_position/const_params.hpp"
#include "quadcopter_mcmpc_position/mcmpc_controller.cuh"

namespace
{
constexpr int kStateQ0 = 0;
constexpr int kStateWx = 4;
constexpr int kStateX = 7;
constexpr int kStateVx = 10;

struct ModelOutput
{
    float vel_sp[3] = {};
    float acc_sp[3] = {};
    float att_sp[4] = {};
    float thrust_sp[3] = {};
    float omega_sp[3] = {};
    float torque_sp[3] = {};
    float vel_int[3] = {};
    float rate_int[3] = {};
    float prev_acc[3] = {};
    float prev_angular_acc[3] = {};
    float motor_speed[4] = {};
    float motor_setpoint[4] = {};
};

void check_cuda(cudaError_t err, const char* label)
{
    if (err != cudaSuccess) {
        std::cerr << label << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

uint64_t timestamp_us(const rclcpp::Clock::SharedPtr& clock)
{
    return static_cast<uint64_t>(clock->now().nanoseconds() / 1000);
}

void copy_symbol_context(
    const float previous_state[_N_OF_ODES],
    const ModelOutput& output)
{
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
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_velocity_device, prev_velocity, sizeof(prev_velocity)), "copy prev_velocity_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_acceleration_device, output.prev_acc, sizeof(output.prev_acc)), "copy prev_acceleration_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::vel_int_device, output.vel_int, sizeof(output.vel_int)), "copy vel_int_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::rate_int_device, output.rate_int, sizeof(output.rate_int)), "copy rate_int_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_angular_velocity_device, prev_angular_velocity, sizeof(prev_angular_velocity)), "copy prev_angular_velocity_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_angular_acceleration_device, output.prev_angular_acc, sizeof(output.prev_angular_acc)), "copy prev_angular_acceleration_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_motor_speed_device, output.motor_speed, sizeof(output.motor_speed)), "copy prev_motor_speed_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_motor_speed_valid_device, &motor_valid, sizeof(int)), "copy prev_motor_speed_valid_device");
}

void reset_symbol_context()
{
    float zeros3[3] = {};
    float zeros4[4] = {};
    int zero = 0;
    int takeoff_state = CONST_PARAM_FLOAT::TAKEOFF_STATE_FLIGHT;
    float tilt_limit = CONST_PARAM_FLOAT::MPC_TILT_MAX;
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_velocity_device, zeros3, sizeof(zeros3)), "copy prev_velocity_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_acceleration_device, zeros3, sizeof(zeros3)), "copy prev_acceleration_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::vel_int_device, zeros3, sizeof(zeros3)), "copy vel_int_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::rate_int_device, zeros3, sizeof(zeros3)), "copy rate_int_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_angular_velocity_device, zeros3, sizeof(zeros3)), "copy prev_angular_velocity_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_angular_acceleration_device, zeros3, sizeof(zeros3)), "copy prev_angular_acceleration_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_motor_speed_device, zeros4, sizeof(zeros4)), "copy prev_motor_speed_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::prev_motor_speed_valid_device, &zero, sizeof(int)), "copy prev_motor_speed_valid_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::acceleration_bias_device, zeros3, sizeof(zeros3)), "copy acceleration_bias_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::takeoff_state_device, &takeoff_state, sizeof(int)), "copy takeoff_state_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::takeoff_tilt_limit_device, &tilt_limit, sizeof(float)), "copy takeoff_tilt_limit_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::landed_device, &zero, sizeof(int)), "copy landed_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::ground_contact_device, &zero, sizeof(int)), "copy ground_contact_device");
    check_cuda(cudaMemcpyToSymbol(qc_mcmpc::maybe_landed_device, &zero, sizeof(int)), "copy maybe_landed_device");
}

ModelOutput read_model_output()
{
    ModelOutput output;
    check_cuda(cudaMemcpyFromSymbol(output.vel_sp, qc_mcmpc::deterministic_sim_vel_setpoint_device, sizeof(output.vel_sp)), "read vel_sp");
    check_cuda(cudaMemcpyFromSymbol(output.acc_sp, qc_mcmpc::deterministic_sim_acc_setpoint_device, sizeof(output.acc_sp)), "read acc_sp");
    check_cuda(cudaMemcpyFromSymbol(output.att_sp, qc_mcmpc::deterministic_sim_att_setpoint_device, sizeof(output.att_sp)), "read att_sp");
    check_cuda(cudaMemcpyFromSymbol(output.thrust_sp, qc_mcmpc::deterministic_sim_thrust_setpoint_device, sizeof(output.thrust_sp)), "read thrust_sp");
    check_cuda(cudaMemcpyFromSymbol(output.omega_sp, qc_mcmpc::deterministic_sim_omega_setpoint_device, sizeof(output.omega_sp)), "read omega_sp");
    check_cuda(cudaMemcpyFromSymbol(output.torque_sp, qc_mcmpc::deterministic_sim_torque_setpoint_device, sizeof(output.torque_sp)), "read torque_sp");
    check_cuda(cudaMemcpyFromSymbol(output.vel_int, qc_mcmpc::deterministic_sim_vel_int_device, sizeof(output.vel_int)), "read vel_int");
    check_cuda(cudaMemcpyFromSymbol(output.rate_int, qc_mcmpc::deterministic_sim_rate_int_device, sizeof(output.rate_int)), "read rate_int");
    check_cuda(cudaMemcpyFromSymbol(output.prev_acc, qc_mcmpc::deterministic_sim_prev_acceleration_device, sizeof(output.prev_acc)), "read prev_acc");
    check_cuda(cudaMemcpyFromSymbol(output.prev_angular_acc, qc_mcmpc::deterministic_sim_prev_angular_acceleration_device, sizeof(output.prev_angular_acc)), "read prev_angular_acc");
    check_cuda(cudaMemcpyFromSymbol(output.motor_speed, qc_mcmpc::deterministic_sim_motor_speed_device, sizeof(output.motor_speed)), "read motor_speed");
    check_cuda(cudaMemcpyFromSymbol(output.motor_setpoint, qc_mcmpc::deterministic_sim_motor_setpoint_device, sizeof(output.motor_setpoint)), "read motor_setpoint");
    return output;
}
}

class MCMPCTruthSimulator : public rclcpp::Node
{
public:
    MCMPCTruthSimulator()
        : Node("mcmpc_truth_simulator")
    {
        qc_mcmpc::mcmpc_controller::get_instance();
        reset_symbol_context();
        latest_command_.position = {0.0f, 0.0f, 0.0f};
        latest_command_.yaw = 0.0f;

        const auto qos = rclcpp::QoS(10).best_effort();
        trajectory_sub_ = create_subscription<px4_msgs::msg::TrajectorySetpoint>(
            "/fmu/in/trajectory_setpoint",
            qos,
            [this](const px4_msgs::msg::TrajectorySetpoint::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(command_mutex_);
                latest_command_ = *msg;
                has_command_ = true;
            });

        odom_pub_ = create_publisher<px4_msgs::msg::VehicleOdometry>("/fmu/out/vehicle_odometry", qos);
        local_sp_pub_ = create_publisher<px4_msgs::msg::VehicleLocalPositionSetpoint>("/fmu/out/vehicle_local_position_setpoint", qos);
        att_sp_pub_ = create_publisher<px4_msgs::msg::VehicleAttitudeSetpoint>("/fmu/out/vehicle_attitude_setpoint_v1", qos);
        rate_sp_pub_ = create_publisher<px4_msgs::msg::VehicleRatesSetpoint>("/fmu/out/vehicle_rates_setpoint", qos);
        hover_pub_ = create_publisher<px4_msgs::msg::HoverThrustEstimate>("/fmu/out/hover_thrust_estimate", qos);
        takeoff_pub_ = create_publisher<px4_msgs::msg::TakeoffStatus>("/fmu/out/takeoff_status", qos);
        land_pub_ = create_publisher<px4_msgs::msg::VehicleLandDetected>("/fmu/out/vehicle_land_detected", qos);
        angular_pub_ = create_publisher<px4_msgs::msg::VehicleAngularVelocity>("/fmu/out/vehicle_angular_velocity", qos);
        rate_status_pub_ = create_publisher<px4_msgs::msg::RateCtrlStatus>("/fmu/out/rate_ctrl_status", qos);
        torque_pub_ = create_publisher<px4_msgs::msg::VehicleTorqueSetpoint>("/fmu/out/vehicle_torque_setpoint", qos);
        actuator_pub_ = create_publisher<px4_msgs::msg::ActuatorMotors>("/fmu/out/actuator_motors", qos);

        timer_ = create_wall_timer(
            std::chrono::duration<double>(CONST_PARAM_FLOAT::CONTROL_PERIOD),
            std::bind(&MCMPCTruthSimulator::step, this));
    }

private:
    void step()
    {
        px4_msgs::msg::TrajectorySetpoint command;
        {
            std::lock_guard<std::mutex> lock(command_mutex_);
            command = latest_command_;
        }

        const float input_x = finite_or(command.position[0], state_[kStateX]);
        const float input_y = finite_or(command.position[1], state_[kStateX + 1]);
        const float input_z = finite_or(command.position[2], state_[kStateX + 2]);
        const float input_yaw = finite_or(command.yaw, 0.0f);

        qc_mcmpc::input_array input;
        for (int h = 0; h < _DEVICE_CONST_HORIZON; h++) {
            input.decoupled_position[h][x] = input_x;
            input.decoupled_position[h][y] = input_y;
            input.decoupled_position[h][z] = input_z;
            input.decoupled_position[h][yaw] = input_yaw;
        }
        input.cost = 0.0f;

        check_cuda(cudaMemcpyToSymbol(qc_mcmpc::var_and_z_i_device, state_, sizeof(state_)), "copy var_and_z_i_device");
        qc_mcmpc::simulate_best_input_trajectory_kernel<<<1, 1>>>(input);
        check_cuda(cudaGetLastError(), "launch simulate_best_input_trajectory_kernel");
        check_cuda(cudaDeviceSynchronize(), "sync simulate_best_input_trajectory_kernel");

        float previous_state[_N_OF_ODES];
        std::copy(state_, state_ + _N_OF_ODES, previous_state);
        check_cuda(cudaMemcpyFromSymbol(state_, qc_mcmpc::deterministic_sim_trajectory_device, sizeof(state_)), "read deterministic state");
        const ModelOutput output = read_model_output();
        copy_symbol_context(previous_state, output);

        publish_truth(command, output);
        sim_time_ += CONST_PARAM_FLOAT::CONTROL_PERIOD;
    }

    static float finite_or(float value, float fallback)
    {
        return std::isfinite(value) ? value : fallback;
    }

    void publish_truth(
        const px4_msgs::msg::TrajectorySetpoint& command,
        const ModelOutput& output)
    {
        const uint64_t now_us = timestamp_us(get_clock());

        px4_msgs::msg::VehicleOdometry odom;
        odom.timestamp = now_us;
        odom.q = {state_[kStateQ0], state_[kStateQ0 + 1], state_[kStateQ0 + 2], state_[kStateQ0 + 3]};
        odom.angular_velocity = {state_[kStateWx], state_[kStateWx + 1], state_[kStateWx + 2]};
        odom.position = {state_[kStateX], state_[kStateX + 1], state_[kStateX + 2]};
        odom.velocity = {state_[kStateVx], state_[kStateVx + 1], state_[kStateVx + 2]};
        odom_pub_->publish(odom);

        px4_msgs::msg::VehicleLocalPositionSetpoint local_sp;
        local_sp.timestamp = now_us;
        local_sp.x = finite_or(command.position[0], state_[kStateX]);
        local_sp.y = finite_or(command.position[1], state_[kStateX + 1]);
        local_sp.z = finite_or(command.position[2], state_[kStateX + 2]);
        local_sp.vx = output.vel_sp[0];
        local_sp.vy = output.vel_sp[1];
        local_sp.vz = output.vel_sp[2];
        local_sp.acceleration = {output.acc_sp[0], output.acc_sp[1], output.acc_sp[2]};
        local_sp.yaw = finite_or(command.yaw, 0.0f);
        local_sp_pub_->publish(local_sp);

        px4_msgs::msg::VehicleAttitudeSetpoint att_sp;
        att_sp.timestamp = now_us;
        att_sp.q_d = {output.att_sp[0], output.att_sp[1], output.att_sp[2], output.att_sp[3]};
        att_sp.thrust_body = {output.thrust_sp[0], output.thrust_sp[1], output.thrust_sp[2]};
        att_sp_pub_->publish(att_sp);

        px4_msgs::msg::VehicleRatesSetpoint rate_sp;
        rate_sp.timestamp = now_us;
        rate_sp.roll = output.omega_sp[0];
        rate_sp.pitch = output.omega_sp[1];
        rate_sp.yaw = output.omega_sp[2];
        rate_sp.thrust_body = {output.thrust_sp[0], output.thrust_sp[1], output.thrust_sp[2]};
        rate_sp_pub_->publish(rate_sp);

        px4_msgs::msg::HoverThrustEstimate hover;
        hover.timestamp = now_us;
        hover.hover_thrust = CONST_PARAM_FLOAT::MPC_THR_HOVER;
        hover.valid = true;
        hover_pub_->publish(hover);

        px4_msgs::msg::TakeoffStatus takeoff;
        takeoff.timestamp = now_us;
        takeoff.takeoff_state = CONST_PARAM_FLOAT::TAKEOFF_STATE_FLIGHT;
        takeoff.tilt_limit = CONST_PARAM_FLOAT::MPC_TILT_MAX;
        takeoff_pub_->publish(takeoff);

        px4_msgs::msg::VehicleLandDetected land;
        land.timestamp = now_us;
        land.landed = false;
        land.ground_contact = false;
        land.maybe_landed = false;
        land_pub_->publish(land);

        px4_msgs::msg::VehicleAngularVelocity angular;
        angular.timestamp = now_us;
        angular.xyz = {state_[kStateWx], state_[kStateWx + 1], state_[kStateWx + 2]};
        angular.xyz_derivative = {
            output.prev_angular_acc[0],
            output.prev_angular_acc[1],
            output.prev_angular_acc[2]};
        angular_pub_->publish(angular);

        px4_msgs::msg::RateCtrlStatus rate_status;
        rate_status.timestamp = now_us;
        rate_status.rollspeed_integ = output.rate_int[0];
        rate_status.pitchspeed_integ = output.rate_int[1];
        rate_status.yawspeed_integ = output.rate_int[2];
        rate_status_pub_->publish(rate_status);

        px4_msgs::msg::VehicleTorqueSetpoint torque;
        torque.timestamp = now_us;
        torque.xyz = {output.torque_sp[0], output.torque_sp[1], output.torque_sp[2]};
        torque_pub_->publish(torque);

        px4_msgs::msg::ActuatorMotors actuator;
        actuator.timestamp = now_us;
        for (int i = 0; i < px4_msgs::msg::ActuatorMotors::NUM_CONTROLS; i++) {
            actuator.control[i] = std::numeric_limits<float>::quiet_NaN();
        }
        for (int i = 0; i < 4; i++) {
            actuator.control[i] = output.motor_setpoint[i];
        }
        actuator_pub_->publish(actuator);
    }

    px4_msgs::msg::TrajectorySetpoint latest_command_{};
    bool has_command_ = false;
    std::mutex command_mutex_;
    rclcpp::Subscription<px4_msgs::msg::TrajectorySetpoint>::SharedPtr trajectory_sub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleLocalPositionSetpoint>::SharedPtr local_sp_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleAttitudeSetpoint>::SharedPtr att_sp_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleRatesSetpoint>::SharedPtr rate_sp_pub_;
    rclcpp::Publisher<px4_msgs::msg::HoverThrustEstimate>::SharedPtr hover_pub_;
    rclcpp::Publisher<px4_msgs::msg::TakeoffStatus>::SharedPtr takeoff_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleLandDetected>::SharedPtr land_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleAngularVelocity>::SharedPtr angular_pub_;
    rclcpp::Publisher<px4_msgs::msg::RateCtrlStatus>::SharedPtr rate_status_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleTorqueSetpoint>::SharedPtr torque_pub_;
    rclcpp::Publisher<px4_msgs::msg::ActuatorMotors>::SharedPtr actuator_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    float state_[_N_OF_ODES] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    float sim_time_ = 0.0f;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MCMPCTruthSimulator>());
    rclcpp::shutdown();
    return 0;
}
