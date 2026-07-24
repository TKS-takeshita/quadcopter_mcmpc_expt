#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_attitude_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_rates_setpoint.hpp>
#include <px4_msgs/msg/vehicle_local_position_setpoint.hpp>
#include <px4_msgs/msg/hover_thrust_estimate.hpp>
#include <px4_msgs/msg/takeoff_status.hpp>
#include <px4_msgs/msg/vehicle_land_detected.hpp>
#include <px4_msgs/msg/vehicle_angular_velocity.hpp>
#include <px4_msgs/msg/rate_ctrl_status.hpp>
#include <px4_msgs/msg/vehicle_torque_setpoint.hpp>
#include <px4_msgs/msg/actuator_motors.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <memory>
#include <mutex>
// csv保存用
#include <fstream>
#include <filesystem>
#include <sstream>
#include <sensor_msgs/msg/joy.hpp>
#include <rclcpp/qos.hpp>

static bool has_traj_sp = false;
static bool has_att_sp = false;
static bool has_rate_sp = false;
static bool has_hover_thrust = false;
static bool has_takeoff_status = false;
static bool has_land_detected = false;
static bool has_angular_velocity = false;
static bool has_rate_ctrl_status = false;
static bool has_torque_sp = false;
static bool has_actuator_motors = false;

static px4_msgs::msg::VehicleLocalPositionSetpoint latest_traj_sp;
static px4_msgs::msg::VehicleAttitudeSetpoint latest_att_sp;
static px4_msgs::msg::VehicleRatesSetpoint latest_rate_sp;
static px4_msgs::msg::HoverThrustEstimate latest_hover_thrust;
static px4_msgs::msg::TakeoffStatus latest_takeoff_status;
static px4_msgs::msg::VehicleLandDetected latest_land_detected;
static px4_msgs::msg::VehicleAngularVelocity latest_angular_velocity;
static px4_msgs::msg::RateCtrlStatus latest_rate_ctrl_status;
static px4_msgs::msg::VehicleTorqueSetpoint latest_torque_sp;
static px4_msgs::msg::ActuatorMotors latest_actuator_motors;
static std::mutex sp_mutex;
static float motor_speed_host_for_model[4] = {0.0f, 0.0f, 0.0f, 0.0f};
static int motor_speed_valid_host_for_model = 0;

#include "quadcopter_mcmpc_position/const_params.hpp"
#include "quadcopter_mcmpc_position/mcmpc_controller.cuh"
#include "quadcopter_mcmpc_position/waypoint_config.hpp"

using namespace qc_mcmpc;

static qc_mcmpc::WaypointConfig runtime_waypoint_config;

static constexpr bool USE_REALTIME_ACCELERATION_BIAS_OBSERVER = true;
static constexpr bool USE_REALTIME_VELOCITY_DELTA_BIAS_OBSERVER = true;
static constexpr bool USE_REALTIME_Z_POSITION_VELOCITY_BIAS_CORRECTION = true;
static constexpr float VELOCITY_DELTA_BIAS_ALPHA = 0.10f;
static constexpr float ACCELERATION_BIAS_LIMIT[3] = {3.0f, 3.0f, 2.0f};
static constexpr float ACCELERATION_BIAS_DEADBAND[3] = {0.02f, 0.02f, 0.01f};
static float acceleration_bias_host_for_model[3] = {0.0f, 0.0f, 0.0f};
static float nominal_acc_host_for_model[3] = {0.0f, 0.0f, 0.0f};
static float measured_acc_host_for_model[3] = {0.0f, 0.0f, 0.0f};
static float prev_log_velocity_host_for_bias[3] = {0.0f, 0.0f, 0.0f};
static bool has_prev_log_velocity_host_for_bias = false;

enum class ControlMode {
    PX4_POSITION,
    MCMPC_ATTITUDE
};
static std::ofstream csv;

static ControlMode control_mode = ControlMode::PX4_POSITION;

static bool offboard_enabled = false;
static bool arm_request = false;
static bool disarm_request = false;
static bool kill_request = false;
static bool takeoff_requested = false;
static bool mcmpc_running = false;

static float move_dist = 1.0f;
static float rotate_angle = M_PI/4.0f;
static constexpr float TAKEOFF_X = 0.0f;
static constexpr float TAKEOFF_Y = 0.0f;
static constexpr float TAKEOFF_Z = -1.0f;
static constexpr int SQUARE_START_WAYPOINT_INDEX = 1;

static uint64_t offboard_setpoint_counter = 0;
float cost = 0.0f;
float sum_cost = 0.0f;

float vel_setpoint[3];
float acc_setpoint[3];
float thrust_setpoint[3];
float att_setpoint[4];
float omega_setpoint[3];
static float prev_angular_velocity_host_for_model[3] = {0.0f, 0.0f, 0.0f};
static bool has_prev_angular_velocity_host_for_model = false;

// 現在状態
static float current_x = 0.0f;
static float current_y = 0.0f;
static float current_yaw = 0.0f;
float target_yaw = 0.0f;

float wrap_pi(float yaw)
{
    return std::atan2(std::sin(yaw), std::cos(yaw));
}

float arctan2f(float y, float x){
    const float c1 = M_PI / 4.0f;
    const float c2 = 3.0f * c1;

    float abs_y = fabsf(y) + 1e-6f;
    float r, angle;

    if (x >= 0.0f) {
        r = (x - abs_y) / (x + abs_y);
        angle = c1 - c1 * r;
    } else {
        r = (x + abs_y) / (abs_y - x);
        angle = c2 - c1 * r;
    }

    return (y < 0.0f) ? -angle : angle;
}
void sin_cosf(float x, float* s, float* c){
    // wrap [-pi, pi]
    if (x > M_PI) x -= 2.0f * M_PI;
    if (x < -M_PI) x += 2.0f * M_PI;
    float x2 = x * x;
    // 5次近似
    *s = x * (1.0f - x2 / 6.0f + x2 * x2 / 120.0f);
    *c = 1.0f - x2 / 2.0f + x2 * x2 / 24.0f;
}

static float motor_speed_ref_from_setpoint_host(float actuator)
{
    actuator = fminf(fmaxf(actuator, 0.0f), 1.0f);
    float ref = 1528.43944677f * actuator - 406.61258522f * actuator * actuator;
    return fminf(fmaxf(ref, 0.0f), CONST_PARAM_FLOAT::MAX_ROT_VELOCITY);
}

static void compute_nominal_acceleration_from_motor_speed_host(
    const float q[4],
    const float velocity[3],
    const float motor_speed[4],
    float acc_ned[3])
{
    float force_body_z = 0.0f;
    for (int m = 0; m < 4; m++) {
        float speed = fminf(fmaxf(motor_speed[m], 0.0f), CONST_PARAM_FLOAT::MAX_ROT_VELOCITY);
        float rotor_force = CONST_PARAM_FLOAT::MOTOR_THRUST_CONSTANT * speed * speed;
        force_body_z -= rotor_force;
    }

    float q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
    float r02 = 2.0f * (q1 * q3 + q0 * q2);
    float r12 = 2.0f * (q2 * q3 - q0 * q1);
    float r22 = 1.0f - 2.0f * (q1 * q1 + q2 * q2);

    acc_ned[0] = r02 * force_body_z / CONST_PARAM_FLOAT::MASS_OF_MACHINE;
    acc_ned[1] = r12 * force_body_z / CONST_PARAM_FLOAT::MASS_OF_MACHINE;
    acc_ned[2] = r22 * force_body_z / CONST_PARAM_FLOAT::MASS_OF_MACHINE + CONST_PARAM_FLOAT::A_OF_GRAVITY;
}

static void update_acceleration_bias_observer_host(
    const float q[4],
    const float velocity[3],
    bool flying,
    bool landed_or_maybe_landed,
    bool ground_contact)
{
    if (!USE_REALTIME_ACCELERATION_BIAS_OBSERVER ||
        !USE_REALTIME_VELOCITY_DELTA_BIAS_OBSERVER ||
        !motor_speed_valid_host_for_model) {
        for (int axis = 0; axis < 3; axis++) {
            prev_log_velocity_host_for_bias[axis] = velocity[axis];
        }
        has_prev_log_velocity_host_for_bias = true;
        return;
    }

    bool velocity_finite =
        std::isfinite(velocity[0]) &&
        std::isfinite(velocity[1]) &&
        std::isfinite(velocity[2]);
    if (!velocity_finite || !flying || landed_or_maybe_landed || ground_contact) {
        if (velocity_finite) {
            for (int axis = 0; axis < 3; axis++) {
                prev_log_velocity_host_for_bias[axis] = velocity[axis];
            }
            has_prev_log_velocity_host_for_bias = true;
        }
        return;
    }

    compute_nominal_acceleration_from_motor_speed_host(
        q,
        velocity,
        motor_speed_host_for_model,
        nominal_acc_host_for_model);

    if (has_prev_log_velocity_host_for_bias) {
        for (int axis = 0; axis < 3; axis++) {
            measured_acc_host_for_model[axis] =
                (velocity[axis] - prev_log_velocity_host_for_bias[axis]) /
                CONST_PARAM_FLOAT::CONTROL_PERIOD;

            bool axis_enabled = (axis < 2) || USE_REALTIME_Z_POSITION_VELOCITY_BIAS_CORRECTION;
            if (!axis_enabled ||
                !std::isfinite(measured_acc_host_for_model[axis]) ||
                !std::isfinite(nominal_acc_host_for_model[axis])) {
                continue;
            }

            float residual = measured_acc_host_for_model[axis] - nominal_acc_host_for_model[axis];
            if (fabsf(residual) < ACCELERATION_BIAS_DEADBAND[axis]) {
                residual = 0.0f;
            }
            acceleration_bias_host_for_model[axis] += VELOCITY_DELTA_BIAS_ALPHA * residual;
            acceleration_bias_host_for_model[axis] = fminf(
                fmaxf(acceleration_bias_host_for_model[axis], -ACCELERATION_BIAS_LIMIT[axis]),
                ACCELERATION_BIAS_LIMIT[axis]);
        }
    }

    for (int axis = 0; axis < 3; axis++) {
        prev_log_velocity_host_for_bias[axis] = velocity[axis];
    }
    has_prev_log_velocity_host_for_bias = true;
}

void set_target_yaw(float yaw)
{
    target_yaw = wrap_pi(yaw);
    float half_yaw = 0.5f * target_yaw;
    target_host.e0 = std::cos(half_yaw);
    target_host.e1 = 0.0f;
    target_host.e2 = 0.0f;
    target_host.e3 = std::sin(half_yaw);
}

void reset_integrator_state()
{
    vel_int_host[0] = 0.0f;
    vel_int_host[1] = 0.0f;
    vel_int_host[2] = 0.0f;
    prev_velocity_host[0] = 0.0f;
    prev_velocity_host[1] = 0.0f;
    prev_velocity_host[2] = 0.0f;
    prev_acceleration_host[0] = 0.0f;
    prev_acceleration_host[1] = 0.0f;
    prev_acceleration_host[2] = 0.0f;
    for (int i = 0; i < 4; i++) {
        motor_speed_host_for_model[i] = 0.0f;
    }
    motor_speed_valid_host_for_model = 0;
    for (int i = 0; i < 3; i++) {
        acceleration_bias_host_for_model[i] = 0.0f;
        nominal_acc_host_for_model[i] = 0.0f;
        measured_acc_host_for_model[i] = 0.0f;
        prev_log_velocity_host_for_bias[i] = 0.0f;
    }
    has_prev_log_velocity_host_for_bias = false;

    cudaMemcpyToSymbol(qc_mcmpc::prev_velocity_device, prev_velocity_host, 3*sizeof(float));
    cudaMemcpyToSymbol(qc_mcmpc::prev_acceleration_device, prev_acceleration_host, 3*sizeof(float));
    cudaMemcpyToSymbol(qc_mcmpc::vel_int_device, vel_int_host, 3*sizeof(float));
    cudaMemcpyToSymbol(qc_mcmpc::prev_motor_speed_device, motor_speed_host_for_model, 4*sizeof(float));
    cudaMemcpyToSymbol(qc_mcmpc::prev_motor_speed_valid_device, &motor_speed_valid_host_for_model, sizeof(int));
    cudaMemcpyToSymbol(qc_mcmpc::acceleration_bias_device, acceleration_bias_host_for_model, 3*sizeof(float));
    has_prev_angular_velocity_host_for_model = false;
}

void set_square_waypoint(int index)
{
    if (index < SQUARE_START_WAYPOINT_INDEX) {
        index = SQUARE_START_WAYPOINT_INDEX;
    }
    if (index >= runtime_waypoint_config.count) {
        index = runtime_waypoint_config.count - 1;
    }
    square_waypoint_index = index;
    square_waypoint_change_time = mcmpc_log;
    target_host.x = runtime_waypoint_config.points[square_waypoint_index][0];
    target_host.y = runtime_waypoint_config.points[square_waypoint_index][1];
    target_host.z = runtime_waypoint_config.points[square_waypoint_index][2];
    set_target_yaw(0.0f);
    update_target_state_device();
    cudaMemcpyToSymbol(qc_mcmpc::square_waypoint_index_device, &square_waypoint_index, sizeof(int));
    cudaMemcpyToSymbol(qc_mcmpc::square_waypoint_change_time_device, &square_waypoint_change_time, sizeof(float));
    cudaMemcpyToSymbol(qc_mcmpc::mcmpc_log_device, &mcmpc_log, sizeof(float));
    RCLCPP_INFO(rclcpp::get_logger("mcmpc"),
        "square waypoint %d/%d: x=%f y=%f z=%f",
        square_waypoint_index + 1,
        runtime_waypoint_config.count,
        target_host.x,
        target_host.y,
        target_host.z);
}

void start_mcmpc_square()
{
    mcmpc_running = true;
    control_mode = ControlMode::MCMPC_ATTITUDE;
    mcmpc_log = 0.0f;
    reset_integrator_state();
    set_square_waypoint(SQUARE_START_WAYPOINT_INDEX);
    RCLCPP_INFO(rclcpp::get_logger("mcmpc"), "A -> MCMPC square start");
}

void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    int a_button            = msg->buttons.size() > 0   ? msg->buttons[0] : 0;
    static int pre_a        = false;
    int b_button            = msg->buttons.size() > 1   ? msg->buttons[1] : 0;
    static int pre_b        = false;
    int x_button            = msg->buttons.size() > 2   ? msg->buttons[2] : 0;
    static int pre_x        = false;
    int y_button            = msg->buttons.size() > 3   ? msg->buttons[3] : 0;
    static int pre_y        = false;
    int lb_button 			= msg->buttons.size() > 4 	? msg->buttons[4] : 0;
    static int pre_lb       = false;
    int rb_button 			= msg->buttons.size() > 5	? msg->buttons[5] : 0;
    static int pre_rb       = false;
    int back_button 		= msg->buttons.size() > 6	? msg->buttons[6] : 0;
    static int pre_back     = false;
    int start_button 		= msg->buttons.size() > 7 	? msg->buttons[7] : 0;
    static int pre_start    = false;
    int power_button 		= msg->buttons.size() > 8 	? msg->buttons[8] : 0;
    static int pre_power    = false;
    int stick_left_button 	= msg->buttons.size() > 9 	? msg->buttons[9] : 0;
    static int pre_stick_left = false;
    int stick_right_button 	= msg->buttons.size() > 10 	? msg->buttons[10]: 0;
    static int pre_stick_right = false;

    if (a_button && !pre_a) {
        if (!takeoff_requested) {
            arm_request = true;
            takeoff_requested = true;
            mcmpc_running = false;
            control_mode = ControlMode::PX4_POSITION;
            target_host.x = TAKEOFF_X;
            target_host.y = TAKEOFF_Y;
            target_host.z = TAKEOFF_Z;
            set_target_yaw(current_yaw);
            reset_integrator_state();
            RCLCPP_INFO(rclcpp::get_logger("mcmpc"), "A -> ARM/OFFBOARD + PX4 position takeoff to (0, 0, -1)");
        } else if (!mcmpc_running) {
            start_mcmpc_square();
        } else {
            RCLCPP_INFO(rclcpp::get_logger("mcmpc"), "A ignored: MCMPC is already running");
        }
        pre_a = true;
    }
    else if(!a_button)
        pre_a = false;

    if (x_button & !pre_x) {
        pre_x = true;
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"), "X unused: press A again to start MCMPC");
    }
    else if(!x_button)
        pre_x = false;

    if (y_button && !pre_y) {
        mcmpc_running = false;
        control_mode = ControlMode::PX4_POSITION;
        target_host.x = current_x;
        target_host.y = current_y;
        target_host.z = 0.0f;
        set_target_yaw(current_yaw);
        reset_integrator_state();
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"), "%f", mcmpc_log);
        pre_y = true;
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"), "Y -> MCMPC stop, PX4 position landing to z=0.0");
    }
    else if(!y_button){
        pre_y = false;
    }

    bool target_changed = false;

    if (lb_button && !pre_lb) {
        target_host.x += move_dist;
        target_changed = true;
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"),
            "LB -> target x=%f y=%f z=%f",
            target_host.x, target_host.y, target_host.z);
    }
    pre_lb = lb_button;

    if (rb_button && !pre_rb) {
        target_host.x -= move_dist;
        target_changed = true;
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"),
            "RB -> target x=%f y=%f z=%f",
            target_host.x, target_host.y, target_host.z);
    }
    pre_rb = rb_button;

    if (back_button && !pre_back) {
        target_host.y += move_dist;
        target_changed = true;
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"),
            "Back -> target x=%f y=%f z=%f",
            target_host.x, target_host.y, target_host.z);
    }
    pre_back = back_button;

    if (start_button && !pre_start) {
        target_host.y -= move_dist;
        target_changed = true;
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"),
            "Start -> target x=%f y=%f z=%f",
            target_host.x, target_host.y, target_host.z);
    }
    pre_start = start_button;

    if (stick_left_button && !pre_stick_left) {
        set_target_yaw(current_yaw + rotate_angle);
        target_changed = true;
    }
    pre_stick_left = stick_left_button;

    if (stick_right_button && !pre_stick_right) {
        set_target_yaw(current_yaw - rotate_angle);
        target_changed = true;
    }
    pre_stick_right = stick_right_button;

    if(power_button && !pre_power) {
        kill_request = true;
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"), "Power -> KILL");
    }
    pre_power = power_button;

    if (target_changed) {
        update_target_state_device();
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"),
            "target copied: x=%f y=%f z=%f yaw=%f",
            target_host.x, target_host.y, target_host.z, target_yaw);
    }
}

void publish_vehicle_command(
    const rclcpp::Node::SharedPtr &node,
    const rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr &pub,
    uint16_t command, float param1 = 0.0f, float param2 = 0.0f)
{
    px4_msgs::msg::VehicleCommand msg{};
    msg.param1 = param1;
    msg.param2 = param2;
    msg.command = command;
    msg.target_system = 1;
    msg.target_component = 1;
    msg.source_system = 1;
    msg.source_component = 1;
    msg.from_external = true;
    msg.timestamp = node->get_clock()->now().nanoseconds() / 1000;
    pub->publish(msg);
}

void trajectory_setpoint_callback(
    const px4_msgs::msg::VehicleLocalPositionSetpoint::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(sp_mutex);
    latest_traj_sp = *msg;
    has_traj_sp = true;
}

void attitude_setpoint_callback(
    const px4_msgs::msg::VehicleAttitudeSetpoint::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(sp_mutex);
    latest_att_sp = *msg;
    has_att_sp = true;
}

void rates_setpoint_callback(
    const px4_msgs::msg::VehicleRatesSetpoint::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(sp_mutex);
    latest_rate_sp = *msg;
    has_rate_sp = true;
}

void hover_thrust_estimate_callback(
    const px4_msgs::msg::HoverThrustEstimate::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(sp_mutex);
    latest_hover_thrust = *msg;
    has_hover_thrust = true;
}

void takeoff_status_callback(
    const px4_msgs::msg::TakeoffStatus::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(sp_mutex);
    latest_takeoff_status = *msg;
    has_takeoff_status = true;
}

void vehicle_land_detected_callback(
    const px4_msgs::msg::VehicleLandDetected::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(sp_mutex);
    latest_land_detected = *msg;
    has_land_detected = true;
}

void angular_velocity_callback(
    const px4_msgs::msg::VehicleAngularVelocity::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(sp_mutex);
    latest_angular_velocity = *msg;
    has_angular_velocity = true;
}

void rate_ctrl_status_callback(
    const px4_msgs::msg::RateCtrlStatus::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(sp_mutex);
    latest_rate_ctrl_status = *msg;
    has_rate_ctrl_status = true;
}

void torque_setpoint_callback(
    const px4_msgs::msg::VehicleTorqueSetpoint::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(sp_mutex);
    latest_torque_sp = *msg;
    has_torque_sp = true;
}

void actuator_motors_callback(
    const px4_msgs::msg::ActuatorMotors::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(sp_mutex);
    latest_actuator_motors = *msg;
    has_actuator_motors = true;
}

namespace quad_sim_base
{
    void do_simulation(float var_array_to_integrate[]);

    // 最新のオドメトリデータ
    static px4_msgs::msg::VehicleOdometry::SharedPtr latest_odom;
    static std::mutex odom_mutex;

    float var_array_to_integrate[_N_OF_ODES];//現在の状態
    double input1, input2, input3, input4;

    // 現在状態からMCMPCを回し、そのときの最適入力を得る
    void do_simulation(float var_array_to_integrate[]){
        float var_and_z_i_temp_float[_N_OF_ODES];

        for(int i=0; i < _N_OF_ODES; i++)
            var_and_z_i_temp_float[i] = var_array_to_integrate[i];
        // コンストラクタを実行するため、get_instance()
        qc_mcmpc::mcmpc_controller::get_instance();//mcmpc_controllerのインスタンス生成

        cost = (mcmpc_controller::get_instance()).calc_optimal_input(var_and_z_i_temp_float, input1, input2, input3, input4);
        sum_cost += cost;
    }
}

void odometry_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(quad_sim_base::odom_mutex);
    
    quad_sim_base::latest_odom = msg;

    current_x = msg->position[0];
    current_y = msg->position[1];
    double qw = msg->q[0];
    double qx = msg->q[1];
    double qy = msg->q[2];
    double qz = msg->q[3];

    double siny = 2.0 * (qw*qz + qx*qy);
    double cosy = 1.0 - 2.0 * (qy*qy + qz*qz);
    current_yaw = std::atan2(siny, cosy);
}


int main(int argc, char *argv[])
{
    // quad_sim_base::init_values();
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("quadcopter_mcmpc");
    node->declare_parameter<std::string>("waypoints_file", "");
    node->declare_parameter<double>("waypoint_threshold", -1.0);
    try {
        const std::string waypoint_file = node->get_parameter("waypoints_file").as_string();
        runtime_waypoint_config = waypoint_file.empty()
            ? qc_mcmpc::default_waypoint_config()
            : qc_mcmpc::load_waypoint_csv(waypoint_file);
    } catch (const std::exception& error) {
        RCLCPP_FATAL(node->get_logger(), "%s", error.what());
        rclcpp::shutdown();
        return 1;
    }
    const double threshold_override = node->get_parameter("waypoint_threshold").as_double();
    if (threshold_override > 0.0) {
        runtime_waypoint_config.threshold = static_cast<float>(threshold_override);
    }
    square_waypoint_count = runtime_waypoint_config.count;
    square_waypoint_threshold = runtime_waypoint_config.threshold;
    qc_mcmpc::mcmpc_controller::get_instance();
    cudaMemcpyToSymbol(qc_mcmpc::square_waypoints_device, runtime_waypoint_config.points.data(), sizeof(runtime_waypoint_config.points));
    cudaMemcpyToSymbol(qc_mcmpc::square_waypoint_count_device, &square_waypoint_count, sizeof(int));
    cudaMemcpyToSymbol(qc_mcmpc::square_waypoint_threshold_device, &square_waypoint_threshold, sizeof(float));
    RCLCPP_INFO(node->get_logger(), "loaded %d runtime waypoints, threshold=%.3f m", square_waypoint_count, square_waypoint_threshold);
    auto qos  = rclcpp::QoS(10).reliability(rclcpp::ReliabilityPolicy::BestEffort);
    auto attitude_thrust_pub = node->create_publisher<px4_msgs::msg::VehicleAttitudeSetpoint>("/fmu/in/vehicle_attitude_setpoint", qos);
    auto rate_thrust_pub = node->create_publisher<px4_msgs::msg::VehicleRatesSetpoint>("/fmu/in/vehicle_rates_setpoint", qos);
    auto offboard_ctrl_pub = node->create_publisher<px4_msgs::msg::OffboardControlMode>("/fmu/in/offboard_control_mode", qos);
    auto subscription      = node->create_subscription<px4_msgs::msg::VehicleOdometry>("/fmu/out/vehicle_odometry", qos, odometry_callback);
    auto traj_pub = node->create_publisher<px4_msgs::msg::TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
    auto cmd_pub = node->create_publisher<px4_msgs::msg::VehicleCommand>("/fmu/in/vehicle_command", 10);
    auto joy_sub = node->create_subscription<sensor_msgs::msg::Joy>("/joy", 10, joy_callback);
    auto traj_sp_sub = node->create_subscription<px4_msgs::msg::VehicleLocalPositionSetpoint>("/fmu/out/vehicle_local_position_setpoint", qos,trajectory_setpoint_callback);
    auto att_sp_sub = node->create_subscription<px4_msgs::msg::VehicleAttitudeSetpoint>("/fmu/out/vehicle_attitude_setpoint_v1",qos,attitude_setpoint_callback);
    auto rate_sp_sub = node->create_subscription<px4_msgs::msg::VehicleRatesSetpoint>("/fmu/out/vehicle_rates_setpoint",qos,rates_setpoint_callback);
    auto hover_thrust_sub = node->create_subscription<px4_msgs::msg::HoverThrustEstimate>("/fmu/out/hover_thrust_estimate",qos,hover_thrust_estimate_callback);
    auto takeoff_status_sub = node->create_subscription<px4_msgs::msg::TakeoffStatus>("/fmu/out/takeoff_status",qos,takeoff_status_callback);
    auto land_detected_sub = node->create_subscription<px4_msgs::msg::VehicleLandDetected>("/fmu/out/vehicle_land_detected",qos,vehicle_land_detected_callback);
    auto angular_velocity_sub = node->create_subscription<px4_msgs::msg::VehicleAngularVelocity>("/fmu/out/vehicle_angular_velocity",qos,angular_velocity_callback);
    auto rate_ctrl_status_sub = node->create_subscription<px4_msgs::msg::RateCtrlStatus>("/fmu/out/rate_ctrl_status",qos,rate_ctrl_status_callback);
    auto torque_sp_sub = node->create_subscription<px4_msgs::msg::VehicleTorqueSetpoint>("/fmu/out/vehicle_torque_setpoint",qos,torque_setpoint_callback);
    auto actuator_motors_sub = node->create_subscription<px4_msgs::msg::ActuatorMotors>("/fmu/out/actuator_motors",qos,actuator_motors_callback);
    csv.open(("/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_expt/quadcopter_mcmpc_position/csv/mcmpc_log_" + ([](){auto n=std::chrono::system_clock::now();std::time_t t=std::chrono::system_clock::to_time_t(n);std::tm tm;localtime_r(&t,&tm);std::ostringstream s;s<<std::put_time(&tm,"%Y%m%d_%H%M%S");return s.str();})() + ".csv"), std::ios::out);
    // csv.open(("/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_position/csv/mcmpc_log_" + ([](){auto n=std::chrono::system_clock::now();std::time_t t=std::chrono::system_clock::to_time_t(n);std::tm tm;localtime_r(&t,&tm);std::ostringstream s;s<<std::put_time(&tm,"%Y%m%d_%H%M%S");return s.str();})() + ".csv"), std::ios::out);
    const char* names[_N_OF_ODES] = {
        "e0","e1","e2","e3",
        "wx","wy","wz",
        "x","y","z",
        "vx","vy","vz"
    };
    csv << "t";
    // 現在状態
    for (int j = 0; j < _N_OF_ODES; j++) {
        csv << ",cur_" << names[j];
    }
    // 入力
    for (int h = 0; h < _DEVICE_CONST_HORIZON; h++) {
        csv << ",u" << h << "_x,u" << h << "_y,u" << h << "_z,u" << h << "_yaw";
    }
    // 目標位置
    csv << ",target_x,target_y,target_z";

    csv << ",px4_sp_vx,px4_sp_vy,px4_sp_vz";
    csv << ",px4_sp_acc_x,px4_sp_acc_y,px4_sp_acc_z";
    csv << ",px4_sp_q0,px4_sp_q1,px4_sp_q2,px4_sp_q3";
    csv << ",px4_sp_roll_rate,px4_sp_pitch_rate,px4_sp_yaw_rate";
    csv << ",px4_sp_thrust_x,px4_sp_thrust_y,px4_sp_thrust_z";
    csv << ",hover_thrust,hover_thrust_valid";
    csv << ",takeoff_state,takeoff_tilt_limit";
    csv << ",landed,ground_contact,maybe_landed";
    csv << ",rollspeed_integ,pitchspeed_integ,yawspeed_integ";
    csv << ",angular_accel_x,angular_accel_y,angular_accel_z";
    csv << ",torque_sp_x,torque_sp_y,torque_sp_z";
    for (int i = 0; i < px4_msgs::msg::ActuatorMotors::NUM_CONTROLS; i++) {
        csv << ",actuator_motor_" << i;
    }
    csv << ",model_vel_int_x,model_vel_int_y,model_vel_int_z";
    for (int i = 0; i < 4; i++) {
        csv << ",model_motor_speed_" << i;
    }
    csv << ",model_acc_bias_x,model_acc_bias_y,model_acc_bias_z";
    csv << ",model_acc_nominal_x,model_acc_nominal_y,model_acc_nominal_z";
    csv << ",model_acc_measured_x,model_acc_measured_y,model_acc_measured_z";

    csv << "\n";
    rclcpp::Rate rate(50);
    
    while (rclcpp::ok()) {

        rclcpp::spin_some(node);
        std::lock_guard<std::mutex> lock(quad_sim_base::odom_mutex);
        if (!quad_sim_base::latest_odom) {
            rate.sleep();
            continue;
        }

        // 相対座標系に変換
        auto msg = quad_sim_base::latest_odom;
        // 状態更新
        quad_sim_base::var_array_to_integrate[0]  = msg->q[0];//qw
        quad_sim_base::var_array_to_integrate[1]  = msg->q[1];//qx
        quad_sim_base::var_array_to_integrate[2]  = msg->q[2];//qy
        quad_sim_base::var_array_to_integrate[3]  = msg->q[3];//qz
        quad_sim_base::var_array_to_integrate[4]  = msg->angular_velocity[0];//wx
        quad_sim_base::var_array_to_integrate[5]  = msg->angular_velocity[1];//wy
        quad_sim_base::var_array_to_integrate[6]  = msg->angular_velocity[2];//wz
        quad_sim_base::var_array_to_integrate[7]  = msg->position[0];//x
        quad_sim_base::var_array_to_integrate[8]  = msg->position[1];//y
        quad_sim_base::var_array_to_integrate[9]  = msg->position[2];//z
        quad_sim_base::var_array_to_integrate[10] = msg->velocity[0];//vx
        quad_sim_base::var_array_to_integrate[11] = msg->velocity[1];//vy
        quad_sim_base::var_array_to_integrate[12] = msg->velocity[2];//vz

        float angular_accel_for_model[3] = {0.0f, 0.0f, 0.0f};
        int takeoff_state_for_model = CONST_PARAM_FLOAT::TAKEOFF_STATE_FLIGHT;
        int landed_for_model = 0;
        int ground_contact_for_model = 0;
        int maybe_landed_for_model = 0;
        {
            std::lock_guard<std::mutex> sp_lock(sp_mutex);
            float hover_thrust_for_model = CONST_PARAM_FLOAT::MPC_THR_HOVER;
            if (has_hover_thrust && latest_hover_thrust.valid && std::isfinite(latest_hover_thrust.hover_thrust) && latest_hover_thrust.hover_thrust > 1.0e-6f) {
                hover_thrust_for_model = latest_hover_thrust.hover_thrust;
            }
            takeoff_state_for_model = has_takeoff_status
                ? static_cast<int>(latest_takeoff_status.takeoff_state)
                : CONST_PARAM_FLOAT::TAKEOFF_STATE_FLIGHT;

            float takeoff_tilt_limit_for_model = has_takeoff_status
                ? latest_takeoff_status.tilt_limit
                : CONST_PARAM_FLOAT::MPC_TILT_MAX;
            landed_for_model = (has_land_detected && latest_land_detected.landed) ? 1 : 0;
            ground_contact_for_model = (has_land_detected && latest_land_detected.ground_contact) ? 1 : 0;
            maybe_landed_for_model = (has_land_detected && latest_land_detected.maybe_landed) ? 1 : 0;

            cudaMemcpyToSymbol(qc_mcmpc::mpc_thr_hover, &hover_thrust_for_model, sizeof(float));
            cudaMemcpyToSymbol(qc_mcmpc::takeoff_state_device, &takeoff_state_for_model, sizeof(int));
            cudaMemcpyToSymbol(qc_mcmpc::takeoff_tilt_limit_device, &takeoff_tilt_limit_for_model, sizeof(float));
            cudaMemcpyToSymbol(qc_mcmpc::landed_device, &landed_for_model, sizeof(int));
            cudaMemcpyToSymbol(qc_mcmpc::ground_contact_device, &ground_contact_for_model, sizeof(int));
            cudaMemcpyToSymbol(qc_mcmpc::maybe_landed_device, &maybe_landed_for_model, sizeof(int));
            float rate_int_for_model[3] = {0.0f, 0.0f, 0.0f};
            if (has_rate_ctrl_status) {
                rate_int_for_model[0] = latest_rate_ctrl_status.rollspeed_integ;
                rate_int_for_model[1] = latest_rate_ctrl_status.pitchspeed_integ;
                rate_int_for_model[2] = latest_rate_ctrl_status.yawspeed_integ;
            }
            if (has_angular_velocity) {
                angular_accel_for_model[0] = latest_angular_velocity.xyz_derivative[0];
                angular_accel_for_model[1] = latest_angular_velocity.xyz_derivative[1];
                angular_accel_for_model[2] = latest_angular_velocity.xyz_derivative[2];
            }
            cudaMemcpyToSymbol(qc_mcmpc::rate_int_device, rate_int_for_model, 3*sizeof(float));
            cudaMemcpyToSymbol(qc_mcmpc::prev_angular_acceleration_device, angular_accel_for_model, 3*sizeof(float));
            if (!has_prev_angular_velocity_host_for_model) {
                prev_angular_velocity_host_for_model[0] = quad_sim_base::var_array_to_integrate[4];
                prev_angular_velocity_host_for_model[1] = quad_sim_base::var_array_to_integrate[5];
                prev_angular_velocity_host_for_model[2] = quad_sim_base::var_array_to_integrate[6];
                has_prev_angular_velocity_host_for_model = true;
            }
            cudaMemcpyToSymbol(qc_mcmpc::prev_angular_velocity_device, prev_angular_velocity_host_for_model, 3*sizeof(float));
        }

        float vel_ref[3];
        float acc_now[3];
        vel_ref[0] = CONST_PARAM_FLOAT::MPC_XY_P * (target_host.x - quad_sim_base::var_array_to_integrate[7]);
        vel_ref[1] = CONST_PARAM_FLOAT::MPC_XY_P * (target_host.y - quad_sim_base::var_array_to_integrate[8]);
        vel_ref[2] = CONST_PARAM_FLOAT::MPC_Z_P  * (target_host.z - quad_sim_base::var_array_to_integrate[9]);
        for(int i=0;i<3;i++){
            float vel_dot = (quad_sim_base::var_array_to_integrate[i+10] - prev_velocity_host[i]) / CONST_PARAM_FLOAT::CONTROL_PERIOD;
            float vel_dot_alpha = CONST_PARAM_FLOAT::CONTROL_PERIOD /
                (CONST_PARAM_FLOAT::CONTROL_PERIOD + 1.0f / (2.0f * M_PI * CONST_PARAM_FLOAT::MPC_VELD_LP));
            acc_now[i] = prev_acceleration_host[i] + vel_dot_alpha * (vel_dot - prev_acceleration_host[i]);
        }
        acc_setpoint[0]= CONST_PARAM_FLOAT::MPC_XY_VEL_P_ACC*(vel_ref[0]-quad_sim_base::var_array_to_integrate[10]);
        acc_setpoint[1]= CONST_PARAM_FLOAT::MPC_XY_VEL_P_ACC*(vel_ref[1]-quad_sim_base::var_array_to_integrate[11]);
        acc_setpoint[2]= CONST_PARAM_FLOAT::MPC_Z_VEL_P_ACC* (vel_ref[2]-quad_sim_base::var_array_to_integrate[12]);

        {
            std::lock_guard<std::mutex> sp_lock(sp_mutex);
            if (has_traj_sp &&
                std::isfinite(latest_traj_sp.vx) &&
                std::isfinite(latest_traj_sp.vy) &&
                std::isfinite(latest_traj_sp.vz) &&
                std::isfinite(latest_traj_sp.acceleration[0]) &&
                std::isfinite(latest_traj_sp.acceleration[1]) &&
                std::isfinite(latest_traj_sp.acceleration[2]) &&
                std::fabs(latest_traj_sp.acceleration[2]) < 50.0f) {
                float local_sp_vel[3] = {
                    latest_traj_sp.vx,
                    latest_traj_sp.vy,
                    latest_traj_sp.vz,
                };
                float local_sp_acc[3] = {
                    latest_traj_sp.acceleration[0],
                    latest_traj_sp.acceleration[1],
                    latest_traj_sp.acceleration[2],
                };
                float current_vel[3] = {
                    quad_sim_base::var_array_to_integrate[10],
                    quad_sim_base::var_array_to_integrate[11],
                    quad_sim_base::var_array_to_integrate[12],
                };
                float vel_error_at_now[3] = {
                    local_sp_vel[0] - current_vel[0],
                    local_sp_vel[1] - current_vel[1],
                    local_sp_vel[2] - current_vel[2],
                };
                vel_int_host[0] = local_sp_acc[0] - CONST_PARAM_FLOAT::MPC_XY_VEL_P_ACC * vel_error_at_now[0]
                    + CONST_PARAM_FLOAT::MPC_XY_VEL_D_ACC * acc_now[0];
                vel_int_host[1] = local_sp_acc[1] - CONST_PARAM_FLOAT::MPC_XY_VEL_P_ACC * vel_error_at_now[1]
                    + CONST_PARAM_FLOAT::MPC_XY_VEL_D_ACC * acc_now[1];
                vel_int_host[2] = local_sp_acc[2] - CONST_PARAM_FLOAT::MPC_Z_VEL_P_ACC * vel_error_at_now[2]
                    + CONST_PARAM_FLOAT::MPC_Z_VEL_D_ACC * acc_now[2];
            } else {
                vel_int_host[0] += (vel_ref[0] - quad_sim_base::var_array_to_integrate[10]) * CONST_PARAM_FLOAT::MPC_XY_VEL_I_ACC * CONST_PARAM_FLOAT::CONTROL_PERIOD;
                vel_int_host[1] += (vel_ref[1] - quad_sim_base::var_array_to_integrate[11]) * CONST_PARAM_FLOAT::MPC_XY_VEL_I_ACC * CONST_PARAM_FLOAT::CONTROL_PERIOD;
                vel_int_host[2] += (vel_ref[2] - quad_sim_base::var_array_to_integrate[12]) * CONST_PARAM_FLOAT::MPC_Z_VEL_I_ACC * CONST_PARAM_FLOAT::CONTROL_PERIOD;
            }

            int was_motor_speed_valid = motor_speed_valid_host_for_model;
            motor_speed_valid_host_for_model = 0;

            if (has_actuator_motors) {
                bool all_finite = true;

                for (int i = 0; i < 4; i++) {
                    float actuator = latest_actuator_motors.control[i];
                    if (!std::isfinite(actuator)) {
                        all_finite = false;
                        break;
                    }

                    float motor_speed_ref = motor_speed_ref_from_setpoint_host(actuator);

                    if (!was_motor_speed_valid) {
                        motor_speed_host_for_model[i] = motor_speed_ref;
                    } else {
                        float prev_speed = fminf(
                            fmaxf(motor_speed_host_for_model[i], 0.0f),
                            CONST_PARAM_FLOAT::MAX_ROT_VELOCITY
                        );

                        float tau = motor_speed_ref > prev_speed
                            ? CONST_PARAM_FLOAT::MOTOR_TIME_CONSTANT_UP
                            : CONST_PARAM_FLOAT::MOTOR_TIME_CONSTANT_DOWN;

                        float alpha = std::exp(
                            -CONST_PARAM_FLOAT::CONTROL_PERIOD / fmaxf(tau, 1.0e-9f)
                        );

                        motor_speed_host_for_model[i] =
                            alpha * prev_speed + (1.0f - alpha) * motor_speed_ref;
                    }
                }
                motor_speed_valid_host_for_model = all_finite ? 1 : 0;
            }
        }

        {
            float q_for_bias[4] = {
                quad_sim_base::var_array_to_integrate[0],
                quad_sim_base::var_array_to_integrate[1],
                quad_sim_base::var_array_to_integrate[2],
                quad_sim_base::var_array_to_integrate[3],
            };
            float velocity_for_bias[3] = {
                quad_sim_base::var_array_to_integrate[10],
                quad_sim_base::var_array_to_integrate[11],
                quad_sim_base::var_array_to_integrate[12],
            };
            bool flying_for_bias = takeoff_state_for_model >= CONST_PARAM_FLOAT::TAKEOFF_STATE_FLIGHT;
            bool landed_or_maybe_landed_for_bias = landed_for_model || maybe_landed_for_model;
            update_acceleration_bias_observer_host(
                q_for_bias,
                velocity_for_bias,
                flying_for_bias,
                landed_or_maybe_landed_for_bias,
                ground_contact_for_model != 0);
        }

        vel_int_host[2] = fminf(fmaxf(vel_int_host[2], -CONST_PARAM_FLOAT::A_OF_GRAVITY), CONST_PARAM_FLOAT::A_OF_GRAVITY);

        cudaMemcpyToSymbol(qc_mcmpc::vel_int_device, vel_int_host, 3*sizeof(float));
        cudaMemcpyToSymbol(qc_mcmpc::prev_motor_speed_device, motor_speed_host_for_model, 4*sizeof(float));
        cudaMemcpyToSymbol(qc_mcmpc::prev_motor_speed_valid_device, &motor_speed_valid_host_for_model, sizeof(int));
        cudaMemcpyToSymbol(qc_mcmpc::acceleration_bias_device, acceleration_bias_host_for_model, 3*sizeof(float));

        if (mcmpc_running) {
            float dx = target_host.x - quad_sim_base::var_array_to_integrate[7];
            float dy = target_host.y - quad_sim_base::var_array_to_integrate[8];
            float waypoint_error_sq = dx * dx + dy * dy;
            float square_waypoint_threshold_sq =
                square_waypoint_threshold * square_waypoint_threshold;
            if ((mcmpc_log - square_waypoint_change_time) >= _SQUARE_WAYPOINT_HOLD_SEC &&
                waypoint_error_sq < square_waypoint_threshold_sq &&
                square_waypoint_index + 1 < square_waypoint_count) {
                set_square_waypoint(square_waypoint_index + 1);
            }

            // MPC計算はMCMPC中だけ実行する
            cudaMemcpyToSymbol(qc_mcmpc::square_waypoint_index_device, &square_waypoint_index, sizeof(int));
            cudaMemcpyToSymbol(qc_mcmpc::square_waypoint_change_time_device, &square_waypoint_change_time, sizeof(float));
            cudaMemcpyToSymbol(qc_mcmpc::mcmpc_log_device, &mcmpc_log, sizeof(float));
            quad_sim_base::do_simulation(quad_sim_base::var_array_to_integrate);
        }
        
        px4_msgs::msg::OffboardControlMode ocm{};
        ocm.timestamp = node->get_clock()->now().nanoseconds() / 1000;
        ocm.position = false;
        ocm.velocity = false;
        ocm.acceleration = false;
        ocm.attitude = false;
        ocm.body_rate = false;
        if (control_mode == ControlMode::PX4_POSITION) {
            ocm.position = true;
        } else {
            ocm.position = true;
        }
        offboard_ctrl_pub->publish(ocm);

        if (!offboard_enabled) {
            if (offboard_setpoint_counter < 20)
                offboard_setpoint_counter++;
            if (arm_request && offboard_setpoint_counter >= 10) {
                publish_vehicle_command(node, cmd_pub, px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);
                publish_vehicle_command(node, cmd_pub, px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);
                offboard_enabled = true;
                arm_request = false;
            }
        }
        else if(disarm_request){
            publish_vehicle_command(node, cmd_pub, px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0);
            offboard_enabled = false;
            arm_request = false;
            disarm_request = false;
            RCLCPP_WARN(rclcpp::get_logger("mcmpc"), "Disarm command sent");
        }

        if(kill_request){
            publish_vehicle_command(node, cmd_pub, px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0f, 21196.0f);   // ← 強制停止
            kill_request = false;
            RCLCPP_ERROR(rclcpp::get_logger("mcmpc"), "!!! KILL ACTIVATED !!!");
        }
        // ======================
        // PX4位置制御
        // ======================

        if (control_mode == ControlMode::PX4_POSITION) {
            px4_msgs::msg::TrajectorySetpoint sp{};
            sp.timestamp = node->get_clock()->now().nanoseconds() / 1000;
            sp.position = {target_host.x, target_host.y, target_host.z};
            sp.yaw = target_yaw;
            traj_pub->publish(sp);
        }

        // ======================
        // MCMPC位置制御
        // ======================
        if (control_mode == ControlMode::MCMPC_ATTITUDE && mcmpc_running) {
            px4_msgs::msg::TrajectorySetpoint sp{};
            sp.timestamp = node->get_clock()->now().nanoseconds() / 1000;
            sp.position = {(float)quad_sim_base::input1, (float)quad_sim_base::input2, (float)quad_sim_base::input3};
            sp.yaw = (float)quad_sim_base::input4;
            traj_pub->publish(sp);
        }
        qc_mcmpc::input_array best_input;
        if (mcmpc_running) {
            qc_mcmpc::mcmpc_controller::get_instance().copy_best_input_array(best_input);
        }
        for(int i=0;i<3;i++){
            prev_velocity_host[i] = quad_sim_base::var_array_to_integrate[i+10];
            prev_acceleration_host[i] = acc_now[i];
            prev_angular_velocity_host_for_model[i] = quad_sim_base::var_array_to_integrate[i+4];
        }

        cudaMemcpyToSymbol(qc_mcmpc::prev_velocity_device, prev_velocity_host, 3*sizeof(float));
        cudaMemcpyToSymbol(qc_mcmpc::prev_acceleration_device, acc_now,3*sizeof(float));
        if (mcmpc_running) {
            csv << mcmpc_log;
            // 現在状態（左）
            for(int i=0;i<_N_OF_ODES;i++){
                csv << "," << quad_sim_base::var_array_to_integrate[i];
            }

            for (int h = 0; h < _DEVICE_CONST_HORIZON; h++) {
                csv << "," << best_input.decoupled_position[h][x]
                    << "," << best_input.decoupled_position[h][y]
                    << "," << best_input.decoupled_position[h][z]
                    << "," << best_input.decoupled_position[h][yaw];
            }

            csv << "," << target_host.x << "," << target_host.y << "," << target_host.z;

            {
                std::lock_guard<std::mutex> sp_lock(sp_mutex);

                if (has_traj_sp) {
                    csv << "," << latest_traj_sp.vx
                        << "," << latest_traj_sp.vy
                        << "," << latest_traj_sp.vz
                        << "," << latest_traj_sp.acceleration[0]
                        << "," << latest_traj_sp.acceleration[1]
                        << "," << latest_traj_sp.acceleration[2];
                } else {
                    csv << ",nan,nan,nan,nan,nan,nan";
                }

                if (has_att_sp) {
                    csv << "," << latest_att_sp.q_d[0]
                        << "," << latest_att_sp.q_d[1]
                        << "," << latest_att_sp.q_d[2]
                        << "," << latest_att_sp.q_d[3];
                } else {
                    csv << ",nan,nan,nan,nan";
                }

                if (has_rate_sp) {
                    csv << "," << latest_rate_sp.roll
                        << "," << latest_rate_sp.pitch
                        << "," << latest_rate_sp.yaw
                        << "," << latest_rate_sp.thrust_body[0]
                        << "," << latest_rate_sp.thrust_body[1]
                        << "," << latest_rate_sp.thrust_body[2];
                } else {
                    csv << ",nan,nan,nan,nan,nan,nan";
                }

                if (has_hover_thrust) {
                    csv << "," << latest_hover_thrust.hover_thrust
                        << "," << static_cast<int>(latest_hover_thrust.valid);
                } else {
                    csv << ",nan,nan";
                }

                if (has_takeoff_status) {
                    csv << "," << static_cast<int>(latest_takeoff_status.takeoff_state)
                        << "," << latest_takeoff_status.tilt_limit;
                } else {
                    csv << ",nan,nan";
                }

                if (has_land_detected) {
                    csv << "," << static_cast<int>(latest_land_detected.landed)
                        << "," << static_cast<int>(latest_land_detected.ground_contact)
                        << "," << static_cast<int>(latest_land_detected.maybe_landed);
                } else {
                    csv << ",nan,nan,nan";
                }

                if (has_rate_ctrl_status) {
                    csv << "," << latest_rate_ctrl_status.rollspeed_integ
                        << "," << latest_rate_ctrl_status.pitchspeed_integ
                        << "," << latest_rate_ctrl_status.yawspeed_integ;
                } else {
                    csv << ",nan,nan,nan";
                }

                if (has_angular_velocity) {
                    csv << "," << latest_angular_velocity.xyz_derivative[0]
                        << "," << latest_angular_velocity.xyz_derivative[1]
                        << "," << latest_angular_velocity.xyz_derivative[2];
                } else {
                    csv << ",nan,nan,nan";
                }

                if (has_torque_sp) {
                    csv << "," << latest_torque_sp.xyz[0]
                        << "," << latest_torque_sp.xyz[1]
                        << "," << latest_torque_sp.xyz[2];
                } else {
                    csv << ",nan,nan,nan";
                }

                if (has_actuator_motors) {
                    for (int i = 0; i < px4_msgs::msg::ActuatorMotors::NUM_CONTROLS; i++) {
                        csv << "," << latest_actuator_motors.control[i];
                    }
                } else {
                    for (int i = 0; i < px4_msgs::msg::ActuatorMotors::NUM_CONTROLS; i++) {
                        csv << ",nan";
                    }
                }

                csv << "," << vel_int_host[0]
                    << "," << vel_int_host[1]
                    << "," << vel_int_host[2];
                for (int i = 0; i < 4; i++) {
                    if (motor_speed_valid_host_for_model) {
                        csv << "," << motor_speed_host_for_model[i];
                    } else {
                        csv << ",nan";
                    }
                }
                csv << "," << acceleration_bias_host_for_model[0]
                    << "," << acceleration_bias_host_for_model[1]
                    << "," << acceleration_bias_host_for_model[2]
                    << "," << nominal_acc_host_for_model[0]
                    << "," << nominal_acc_host_for_model[1]
                    << "," << nominal_acc_host_for_model[2]
                    << "," << measured_acc_host_for_model[0]
                    << "," << measured_acc_host_for_model[1]
                    << "," << measured_acc_host_for_model[2];
            }

            csv << "\n";
            mcmpc_log +=0.02;
        }
        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
