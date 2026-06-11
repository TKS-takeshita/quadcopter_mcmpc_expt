#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_attitude_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_rates_setpoint.hpp>
#include <px4_msgs/msg/vehicle_local_position_setpoint.hpp>

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

static px4_msgs::msg::VehicleLocalPositionSetpoint latest_traj_sp;
static px4_msgs::msg::VehicleAttitudeSetpoint latest_att_sp;
static px4_msgs::msg::VehicleRatesSetpoint latest_rate_sp;
static std::mutex sp_mutex;

#include "quadcopter_mcmpc_position/const_params.hpp"
#include "quadcopter_mcmpc_position/mcmpc_controller.cuh"

using namespace qc_mcmpc;

#if defined(UNPREDICTABLE_COLLISION_WITH_WALL) || defined(PREDICTABLE_COLLISION_WITH_WALL)
#include <Eigen/Dense>
#endif

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
static constexpr float SQUARE_Z = -1.0f;
static constexpr float SQUARE_WAYPOINT_THRESHOLD = 0.15f;
static constexpr float SQUARE_WAYPOINT_HOLD_SEC = 1.0f;

static uint64_t offboard_setpoint_counter = 0;
float cost = 0.0f;
float sum_cost = 0.0f;

float vel_setpoint[3];
float acc_setpoint[3];
float thrust_setpoint[3];
float att_setpoint[4];
float omega_setpoint[3];

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
    cudaMemcpyToSymbol(qc_mcmpc::prev_velocity_device, prev_velocity_host, 3*sizeof(float));
    cudaMemcpyToSymbol(qc_mcmpc::prev_acceleration_device, prev_acceleration_host, 3*sizeof(float));
    cudaMemcpyToSymbol(qc_mcmpc::vel_int_device, vel_int_host, 3*sizeof(float));
}

void set_square_waypoint(int index)
{
    square_waypoint_index = index;
    square_waypoint_change_time = mcmpc_log;
    target_host.x = CONST_PARAM_FLOAT::square_waypoints[square_waypoint_index][0];
    target_host.y = CONST_PARAM_FLOAT::square_waypoints[square_waypoint_index][1];
    target_host.z = SQUARE_Z;
    set_target_yaw(0.0f);
    update_target_state_device();
    RCLCPP_INFO(rclcpp::get_logger("mcmpc"),
        "square waypoint %d/%d: x=%f y=%f z=%f",
        square_waypoint_index + 1,
        _SQUARE_WAYPOINTS,
        target_host.x,
        target_host.y,
        target_host.z);
}

void start_mcmpc_square()
{
    mcmpc_running = true;
    control_mode = ControlMode::MCMPC_ATTITUDE;
    reset_integrator_state();
    set_square_waypoint(0);
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
        float var_and_z_i_temp_float[_N_OF_ODES + 1];

        // time measurement
        struct timespec start_time;
        struct timespec finish_time;
        std::vector<long> process_time_buff;//時間測定しているが毎回消えるので後で修正

        for(int i=0; i < _N_OF_ODES; i++)
            var_and_z_i_temp_float[i] = var_array_to_integrate[i];
        var_and_z_i_temp_float[_N_OF_ODES] = 0.0f;

        // コンストラクタを実行するため、get_instance()
        qc_mcmpc::mcmpc_controller::get_instance();//mcmpc_controllerのインスタンス生成

        // time measurement
        clock_gettime(CLOCK_REALTIME, &start_time);
        cost = (mcmpc_controller::get_instance()).calc_optimal_input(var_and_z_i_temp_float, input1, input2, input3, input4);
        sum_cost += cost;

        // time measurement
        clock_gettime(CLOCK_REALTIME, &finish_time);

        int sec = (int)(finish_time.tv_sec - start_time.tv_sec);
        long ns = (long)(finish_time.tv_nsec - start_time.tv_nsec);

        process_time_buff.push_back(((long)(sec * 1e9)) + ns);
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
    float var_p_save[_DEVICE_CONST_HORIZON+1][_N_OF_ODES+1];
    for(int i= 0; i<_DEVICE_CONST_HORIZON+1; i++) {
        for(int j=0; j<_N_OF_ODES+1; j++){
            var_p_save[i][j] = 0.0f;
        }
    }
    // csv.open(("/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_expt/quadcopter_mcmpc_position/csv/mcmpc_log_" + ([](){auto n=std::chrono::system_clock::now();std::time_t t=std::chrono::system_clock::to_time_t(n);std::tm tm;localtime_r(&t,&tm);std::ostringstream s;s<<std::put_time(&tm,"%Y%m%d_%H%M%S");return s.str();})() + ".csv"), std::ios::out);
    csv.open(("/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_position/csv/mcmpc_log_" + ([](){auto n=std::chrono::system_clock::now();std::time_t t=std::chrono::system_clock::to_time_t(n);std::tm tm;localtime_r(&t,&tm);std::ostringstream s;s<<std::put_time(&tm,"%Y%m%d_%H%M%S");return s.str();})() + ".csv"), std::ios::out);
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

        float vel_ref[3];
        float acc_now[3];
        vel_ref[0] = CONST_PARAM_FLOAT::MPC_XY_P * (target_host.x - quad_sim_base::var_array_to_integrate[7]);
        vel_ref[1] = CONST_PARAM_FLOAT::MPC_XY_P * (target_host.y - quad_sim_base::var_array_to_integrate[8]);
        vel_ref[2] = CONST_PARAM_FLOAT::MPC_Z_P  * (target_host.z - quad_sim_base::var_array_to_integrate[9]);
        for(int i=0;i<3;i++){
            float vel_dot = (quad_sim_base::var_array_to_integrate[i+10] - prev_velocity_host[i]) / CONST_PARAM_FLOAT::CONTROL_PERIOD;
            acc_now[i] = prev_acceleration_host[i] + CONST_PARAM_FLOAT::LPF * (vel_dot - prev_acceleration_host[i]);
        }
        acc_setpoint[0]= CONST_PARAM_FLOAT::MPC_XY_VEL_P_ACC*(vel_ref[0]-quad_sim_base::var_array_to_integrate[10]);
        acc_setpoint[1]= CONST_PARAM_FLOAT::MPC_XY_VEL_P_ACC*(vel_ref[1]-quad_sim_base::var_array_to_integrate[11]);
        acc_setpoint[2]= CONST_PARAM_FLOAT::MPC_Z_VEL_P_ACC* (vel_ref[2]-quad_sim_base::var_array_to_integrate[12]);

        if (mcmpc_running) {
            float dx = target_host.x - quad_sim_base::var_array_to_integrate[7];
            float dy = target_host.y - quad_sim_base::var_array_to_integrate[8];
            float waypoint_error = std::sqrt(dx * dx + dy * dy);
            if ((mcmpc_log - square_waypoint_change_time) >= SQUARE_WAYPOINT_HOLD_SEC &&
                waypoint_error < SQUARE_WAYPOINT_THRESHOLD &&
                square_waypoint_index + 1 < _SQUARE_WAYPOINTS) {
                set_square_waypoint(square_waypoint_index + 1);
            }

            // MPC計算はMCMPC中だけ実行する
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
        }
        vel_int_host[0] += (vel_ref[0] - quad_sim_base::var_array_to_integrate[10]) * CONST_PARAM_FLOAT::CONTROL_PERIOD;
        vel_int_host[1] += (vel_ref[1] - quad_sim_base::var_array_to_integrate[11]) * CONST_PARAM_FLOAT::CONTROL_PERIOD;
        vel_int_host[2] += (vel_ref[2] - quad_sim_base::var_array_to_integrate[12]) * CONST_PARAM_FLOAT::CONTROL_PERIOD;
        vel_int_host[2] =  fminf(fmaxf(vel_int_host[2], -CONST_PARAM_FLOAT::A_OF_GRAVITY), CONST_PARAM_FLOAT::A_OF_GRAVITY);
        cudaMemcpyToSymbol(qc_mcmpc::prev_velocity_device, prev_velocity_host, 3*sizeof(float));
        cudaMemcpyToSymbol(qc_mcmpc::prev_acceleration_device, acc_now,3*sizeof(float));
        cudaMemcpyToSymbol(qc_mcmpc::vel_int_device, vel_int_host, 3*sizeof(float));
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
            }

            csv << "\n";
            mcmpc_log +=0.02;
        }
        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
