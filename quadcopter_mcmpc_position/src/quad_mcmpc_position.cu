#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_attitude_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_rates_setpoint.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <memory>
// csv保存用
#include <fstream>
#include <filesystem>
#include <sstream>
#include <sensor_msgs/msg/joy.hpp>
#include <rclcpp/qos.hpp>

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

static float move_dist = 1.0f;
static float rotate_angle = M_PI/4.0f;

static uint64_t offboard_setpoint_counter = 0;
float mcmpc_log = 0.0f;
float cost = 0.0f;
float sum_cost = 0.0f;

// 現在状態
static float current_x = 0.0f;
static float current_y = 0.0f;
// static float current_z = 0.0f;
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

    if (a_button & !pre_a) {
        arm_request = true;
        control_mode = ControlMode::PX4_POSITION;
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"), "request ARM +  OFFBOARD");
        target_host.x = current_x;
        target_host.y = current_y;
        target_host.z = -1.2f;
        target_yaw = current_yaw;
        pre_a = true;
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"), "A -> ARM + takeoff");
    }
    else if(!a_button)
        pre_a = false;

    if (x_button & !pre_x) {
        control_mode = ControlMode::MCMPC_ATTITUDE;
        vel_int_host[0] = 0.0f;
        vel_int_host[1] = 0.0f;
        vel_int_host[2] = 0.0f;
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"), "%f", mcmpc_log);
        pre_x = true;
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"), "X -> MCMPC mode");
    }
    else if(!x_button)
        pre_x = false;

    if (y_button & !pre_y) {
        control_mode = ControlMode::PX4_POSITION;
        target_host.x = current_x;
        target_host.y = current_y;
        target_host.z = 0.0f;
        target_yaw = current_yaw;
        pre_y = true;
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"), "Y -> LAND");
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
        target_yaw = wrap_pi(current_yaw + rotate_angle);
        float half_yaw = 0.5f * target_yaw;
        target_host.e0 = std::cos(half_yaw);
        target_host.e1 = 0.0f;
        target_host.e2 = 0.0f;
        target_host.e3 = std::sin(half_yaw);
        target_changed = true;
    }
    pre_stick_left = stick_left_button;

    if (stick_right_button && !pre_stick_right) {
        target_yaw = wrap_pi(current_yaw - rotate_angle);
        float half_yaw = 0.5f * target_yaw;
        target_host.e0 = std::cos(half_yaw);
        target_host.e1 = 0.0f;
        target_host.e2 = 0.0f;
        target_host.e3 = std::sin(half_yaw);
        target_changed = true;
    }
    pre_stick_right = stick_right_button;

    if (target_changed) {
        qc_mcmpc::update_target_state_device();
        RCLCPP_INFO(rclcpp::get_logger("mcmpc"),
            "target copied to device: x=%f y=%f z=%f",
            target_host.x, target_host.y, target_host.z);
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

void verification_simulation_one_step(float var_and_z_i_device[], float var_p_save[_DEVICE_CONST_HORIZON + 1][_N_OF_ODES+1], float rps_cw1, float rps_cw2, float rps_ccw1, float rps_ccw2);

namespace quad_sim_base
{
    void do_simulation(float var_array_to_integrate[]);
    void output_screen(); //画面出力用関数

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

    void output_screen(){
        static int cnt = 0;
        float e0 = quad_sim_base::var_array_to_integrate[0]; // scalar (w)
        float e1 = quad_sim_base::var_array_to_integrate[1]; // x
        float e2 = quad_sim_base::var_array_to_integrate[2]; // y
        float e3 = quad_sim_base::var_array_to_integrate[3]; // z
        float roll  = atan2(2.0 * (e0 * e1 + e2 * e3), 1.0 - 2.0 * (e1 * e1 + e2 * e2));
        float sin_pitch = 2.0 * (e0 * e2 - e3 * e1);
        float pitch = asin(fmaxf(-1.0, fminf(1.0, sin_pitch))); // 範囲外エラー防止
        float yaw   = atan2(2.0 * (e0 * e3 + e1 * e2), 1.0 - 2.0 * (e2 * e2 + e3 * e3));
        float roll_deg  = roll  * (180.0 / M_PI);
        float pitch_deg = pitch * (180.0 / M_PI);
        float yaw_deg   = yaw   * (180.0 / M_PI);
        // printf("%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n",
        //        cnt*0.02f,
        //        quad_sim_base::var_array_to_integrate[0],
        //        quad_sim_base::var_array_to_integrate[1],
        //        quad_sim_base::var_array_to_integrate[2],
        //        quad_sim_base::var_array_to_integrate[3],
        //        quad_sim_base::var_array_to_integrate[4] * (180.0 / M_PI),
        //        quad_sim_base::var_array_to_integrate[5] * (180.0 / M_PI),
        //        quad_sim_base::var_array_to_integrate[6] * (180.0 / M_PI),
        //        quad_sim_base::var_array_to_integrate[7],
        //        quad_sim_base::var_array_to_integrate[8],
        //        quad_sim_base::var_array_to_integrate[9],
        //        quad_sim_base::var_array_to_integrate[10],
        //        quad_sim_base::var_array_to_integrate[11],
        //        quad_sim_base::var_array_to_integrate[12],
        //        roll_deg, pitch_deg, yaw_deg,
        //        quad_sim_base::input1, quad_sim_base::input1/CONST_PARAM::MAX_THRUST,
        //        quad_sim_base::input2, quad_sim_base::input3, quad_sim_base::input4
        //        );
    }
}

void odometry_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(quad_sim_base::odom_mutex);
    
    quad_sim_base::latest_odom = msg;

    current_x = msg->position[0];
    current_y = msg->position[1];
    // current_z = msg->position[2];

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
    float var_p_save[_DEVICE_CONST_HORIZON+1][_N_OF_ODES+1];
    for(int i= 0; i<_DEVICE_CONST_HORIZON+1; i++) {
        for(int j=0; j<_N_OF_ODES+1; j++){
            var_p_save[i][j] = 0.0f;
        }
    }
    csv.open(("/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_expt/quadcopter_mcmpc_position/csv/mcmpc_log_" + ([](){auto n=std::chrono::system_clock::now();std::time_t t=std::chrono::system_clock::to_time_t(n);std::tm tm;localtime_r(&t,&tm);std::ostringstream s;s<<std::put_time(&tm,"%Y%m%d_%H%M%S");return s.str();})() + ".csv"), std::ios::out);
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
    csv << ",input_1,input_2,input_3,input_4";
    // 目標位置
    csv << ",target_x,target_y,target_z";
    // horizon予測（1ステップ先から）
    for (int h = 1; h <= _DEVICE_CONST_HORIZON; h++) {
        csv << ",t_pred_" << h;

        for (int j = 0; j < _N_OF_ODES; j++) {
            csv << "," << h << names[j];  // ← 1e0,1e1,...2e0,...
        }
    }
    csv << ",cost";
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
        float a_ref[3];
        a_ref[0] = CONST_PARAM_FLOAT::MPC_XY_VEL_P_ACC*(vel_ref[0]-quad_sim_base::var_array_to_integrate[10])+CONST_PARAM_FLOAT::MPC_XY_VEL_I_ACC*vel_int_host[0]-CONST_PARAM_FLOAT::MPC_XY_VEL_D_ACC*acc_now[0];
        a_ref[1] = CONST_PARAM_FLOAT::MPC_XY_VEL_P_ACC*(vel_ref[1]-quad_sim_base::var_array_to_integrate[11])+CONST_PARAM_FLOAT::MPC_XY_VEL_I_ACC*vel_int_host[1]-CONST_PARAM_FLOAT::MPC_XY_VEL_D_ACC*acc_now[1];
        a_ref[2] = CONST_PARAM_FLOAT::MPC_Z_VEL_P_ACC* (vel_ref[2]-quad_sim_base::var_array_to_integrate[12])+CONST_PARAM_FLOAT::MPC_Z_VEL_I_ACC*vel_int_host[2] -CONST_PARAM_FLOAT::MPC_Z_VEL_D_ACC *acc_now[2];

        // MPC計算
        quad_sim_base::do_simulation(quad_sim_base::var_array_to_integrate);
        
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
            // ocm.body_rate = true;
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
        if (control_mode == ControlMode::MCMPC_ATTITUDE) {
            px4_msgs::msg::TrajectorySetpoint sp{};
            sp.timestamp = node->get_clock()->now().nanoseconds() / 1000;
            sp.position = {(float)quad_sim_base::input1, (float)quad_sim_base::input2, (float)quad_sim_base::input3};
            sp.yaw = (float)quad_sim_base::input4;
            traj_pub->publish(sp);
        }
        // csv保存用1stepあとの状態をシミュレーション
        verification_simulation_one_step(quad_sim_base::var_array_to_integrate, var_p_save, quad_sim_base::input1, quad_sim_base::input2, quad_sim_base::input3, quad_sim_base::input4);
        for(int i=0;i<3;i++){
            prev_velocity_host[i] = quad_sim_base::var_array_to_integrate[i+10];
            prev_acceleration_host[i] = acc_now[i];
        }
        vel_int_host[0] += (vel_ref[0] - quad_sim_base::var_array_to_integrate[10] - CONST_PARAM_FLOAT::ARW_GAIN*(a_ref[0]-acc_now[0])) * CONST_PARAM_FLOAT::CONTROL_PERIOD;
        vel_int_host[1] += (vel_ref[1] - quad_sim_base::var_array_to_integrate[11] - CONST_PARAM_FLOAT::ARW_GAIN*(a_ref[1]-acc_now[1])) * CONST_PARAM_FLOAT::CONTROL_PERIOD;
        vel_int_host[2] += (vel_ref[2] - quad_sim_base::var_array_to_integrate[12]) * CONST_PARAM_FLOAT::CONTROL_PERIOD;
        vel_int_host[2] =  fminf(fmaxf(vel_int_host[2], -CONST_PARAM_FLOAT::A_OF_GRAVITY), CONST_PARAM_FLOAT::A_OF_GRAVITY);
        cudaMemcpyToSymbol(qc_mcmpc::prev_velocity_device, prev_velocity_host, 3*sizeof(float));
        cudaMemcpyToSymbol(qc_mcmpc::prev_acceleration_device, acc_now,3*sizeof(float));
        cudaMemcpyToSymbol(qc_mcmpc::vel_int_device, vel_int_host, 3*sizeof(float));
        csv << mcmpc_log;
        // 現在状態（左）
        for(int i=0;i<_N_OF_ODES;i++){
            csv << "," << var_p_save[0][i];
        }
        // 入力
        csv << "," << quad_sim_base::input1 << "," << quad_sim_base::input2 << "," << quad_sim_base::input3 << "," << quad_sim_base::input4;
        // 目標位置
        csv << "," << target_host.x << "," << target_host.y << "," << target_host.z;
        // 予測状態（右)
        for(int i=1;i<_DEVICE_CONST_HORIZON+1;i++){
            // 予測時刻
            csv << "," << (mcmpc_log + 0.02*i);
            for(int j=0;j<_N_OF_ODES;j++){
                csv << "," << var_p_save[i][j];
            }
        }
        csv << "," << cost;
        csv << "\n";
        mcmpc_log +=0.02;
        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}

void verification_simulation_one_step(float var_and_z_i_device[], float var_p_save[_DEVICE_CONST_HORIZON + 1][_N_OF_ODES+1], float input1, float input2, float input3, float input4){
    float decoupled_position[4];
    decoupled_position[0] = input1;//x
    decoupled_position[1] = input2;//y
    decoupled_position[2] = input3;//z
    decoupled_position[3] = input4;//yaw

    float control_period_device        = (float)CONST_PARAM::CONTROL_PERIOD;
    float integration_step_size_device = (float)CONST_PARAM::CONTROL_PERIOD / 2.0f;
    float mass_of_machine_device       = (float)CONST_PARAM::MASS_OF_MACHINE;
    float a_of_gravity_device          = (float)CONST_PARAM::A_OF_GRAVITY;
    float max_thrust_device            = (float)CONST_PARAM::MAX_THRUST;
    float mpc_xy_p                     = CONST_PARAM_FLOAT::MPC_XY_P;
    float mpc_z_p                      = CONST_PARAM_FLOAT::MPC_Z_P;
    float mpc_xy_vel_p_acc             = CONST_PARAM_FLOAT::MPC_XY_VEL_P_ACC;
    float mpc_z_vel_p_acc              = CONST_PARAM_FLOAT::MPC_Z_VEL_P_ACC;
    float mpc_xy_vel_i_acc             = CONST_PARAM_FLOAT::MPC_XY_VEL_I_ACC;
    float mpc_z_vel_i_acc              = CONST_PARAM_FLOAT::MPC_Z_VEL_I_ACC;
    float mc_roll_p                    = CONST_PARAM_FLOAT::MC_ROLL_P;
    float mc_pitch_p                   = CONST_PARAM_FLOAT::MC_PITCH_P;
    float mc_yaw_p                     = CONST_PARAM_FLOAT::MC_YAW_P;
    cost =0.0f;
    for(int j=0; j<_N_OF_ODES; j++){
        var_p_save[0][j] = var_and_z_i_device[j];
    }
    for (int i = 1; i < _DEVICE_CONST_HORIZON + 1; i++) {
        for (int j = 0; j < _N_OF_ODES; j++) {
            var_p_save[i][j] = 0.0f;
        }
    }

    // __constant__ メモリから状態量をコピー
    float var_and_z_i_temp[_N_OF_ODES + 1];
    for ( int i = 0; i < _N_OF_ODES + 1; i++ )
        var_and_z_i_temp[i] = var_and_z_i_device[i];
    
    float var_p_temp[_N_OF_ODES];
    float prev_vel[3];
    prev_vel[0] = prev_velocity_host[0];
    prev_vel[1] = prev_velocity_host[1];
    prev_vel[2] = prev_velocity_host[2];
    // 前時刻までの誤差の積分値
    float vel_int[3];
    vel_int[0] = vel_int_host[0];
    vel_int[1] = vel_int_host[1];
    vel_int[2] = vel_int_host[2];
    float prev_acc[3];
    prev_acc[0] = prev_acceleration_host[0];
    prev_acc[1] = prev_acceleration_host[1];
    prev_acc[2] = prev_acceleration_host[2];
    for ( int i = 0; i < _DEVICE_CONST_HORIZON; i++ )
    {
        float x_ref   = decoupled_position[x];
        float y_ref   = decoupled_position[y];
        float z_ref   = decoupled_position[z];
        float yaw_ref = decoupled_position[yaw];
        float inv_mass = 1.0f / mass_of_machine_device;
        float vel_ref[3];
        for ( float t = 0.0f; t < CONST_PARAM_FLOAT::CONTROL_PERIOD - integration_step_size_device / 2; t += integration_step_size_device ){
            // PX4のPIDを素に速度, 加速度, 姿勢, スラスト, 角速度を導出
            for ( int k = 0; k < _N_OF_ODES; k++ ) var_p_temp[k] = var_p_save[i][k];
            /*目標速度*/
           
            vel_ref[0] = mpc_xy_p * (x_ref - var_p_temp[7]);
            vel_ref[1] = mpc_xy_p * (y_ref - var_p_temp[8]);
            vel_ref[2] = mpc_z_p  * (z_ref - var_p_temp[9]);
            /*目標加速度*/
            float vel_dot_x = (var_p_temp[10] - prev_vel[0]) / CONST_PARAM_FLOAT::CONTROL_PERIOD;
            float vel_dot_y = (var_p_temp[11] - prev_vel[1]) / CONST_PARAM_FLOAT::CONTROL_PERIOD;
            float vel_dot_z = (var_p_temp[12] - prev_vel[2]) / CONST_PARAM_FLOAT::CONTROL_PERIOD;
            // 力を考慮する場合 a_ref = a_ref_pid - Fcut/m
            float a_ref[3];
            a_ref[0] = mpc_xy_vel_p_acc*(vel_ref[0]-var_p_temp[10])+mpc_xy_vel_i_acc*vel_int[0]-CONST_PARAM_FLOAT::MPC_XY_VEL_D_ACC*(prev_acc[0]+CONST_PARAM_FLOAT::LPF*(vel_dot_x-prev_acc[0]));
            a_ref[1] = mpc_xy_vel_p_acc*(vel_ref[1]-var_p_temp[11])+mpc_xy_vel_i_acc*vel_int[1]-CONST_PARAM_FLOAT::MPC_XY_VEL_D_ACC*(prev_acc[1]+CONST_PARAM_FLOAT::LPF*(vel_dot_y-prev_acc[1]));
            a_ref[2] = mpc_z_vel_p_acc* (vel_ref[2]-var_p_temp[12])+mpc_z_vel_i_acc* vel_int[2]-CONST_PARAM_FLOAT::MPC_Z_VEL_D_ACC *(prev_acc[2]+CONST_PARAM_FLOAT::LPF*(vel_dot_z-prev_acc[2]));

            // /* e0p */ var_p_save[i+1][0]  = var_p_temp[0] + (-0.5f*var_p_temp[1]*var_p_temp[4] - 0.5f*var_p_temp[2]*var_p_temp[5] - 0.5f*var_p_temp[3]*var_p_temp[6])*integration_step_size_device;
            // /* e1p */ var_p_save[i+1][1]  = var_p_temp[1] + ( 0.5f*var_p_temp[0]*var_p_temp[4] + 0.5f*var_p_temp[2]*var_p_temp[6] - 0.5f*var_p_temp[3]*var_p_temp[5])*integration_step_size_device;
            // /* e2p */ var_p_save[i+1][2]  = var_p_temp[2] + ( 0.5f*var_p_temp[0]*var_p_temp[5] + 0.5f*var_p_temp[3]*var_p_temp[4] - 0.5f*var_p_temp[1]*var_p_temp[6])*integration_step_size_device;
            // /* e3p */ var_p_save[i+1][3]  = var_p_temp[3] + ( 0.5f*var_p_temp[0]*var_p_temp[6] + 0.5f*var_p_temp[1]*var_p_temp[5] - 0.5f*var_p_temp[2]*var_p_temp[4])*integration_step_size_device;
            //         // normalize quaternion
            //         float q_norm_inv = rsqrtf(var_p_save[i+1][0]*var_p_save[i+1][0] +var_p_save[i+1][1]*var_p_save[i+1][1] +var_p_save[i+1][2]*var_p_save[i+1][2] +var_p_save[i+1][3]*var_p_save[i+1][3]);
            //         var_p_save[i+1][0] *= q_norm_inv;
            //         var_p_save[i+1][1] *= q_norm_inv;
            //         var_p_save[i+1][2] *= q_norm_inv;
            //         var_p_save[i+1][3] *= q_norm_inv;
            // /* wxp */ var_p_save[i][4]  = 2.0f*mc_roll_p *sign_w*(var_p_temp[0]*ref_qx-var_p_temp[1]*ref_qw-var_p_temp[2]*ref_qz+var_p_temp[3]*ref_qy);
            // /* wyp */ var_p_save[i][5]  = 2.0f*mc_pitch_p*sign_w*(var_p_temp[0]*ref_qy+var_p_temp[1]*ref_qz-var_p_temp[2]*ref_qw-var_p_temp[3]*ref_qx);
            // /* wzp */ var_p_save[i][6]  = 2.0f*mc_yaw_p*  sign_w*(var_p_temp[0]*ref_qz-var_p_temp[1]*ref_qy+var_p_temp[2]*ref_qx-var_p_temp[3]*ref_qw);

            /* xp  */ var_p_save[i+1][7]  = var_p_temp[7] + var_p_temp[10] * integration_step_size_device;
            /* yp  */ var_p_save[i+1][8]  = var_p_temp[8] + var_p_temp[11] * integration_step_size_device;
            /* zp  */ var_p_save[i+1][9]  = var_p_temp[9] + var_p_temp[12] * integration_step_size_device;

            /* xpp */ var_p_save[i+1][10] = var_p_temp[10] + a_ref[0] * integration_step_size_device;
            /* ypp */ var_p_save[i+1][11] = var_p_temp[11] + a_ref[1] * integration_step_size_device;
            /* zpp */ var_p_save[i+1][12] = var_p_temp[12] + a_ref[2] * integration_step_size_device;
        }
        vel_int[0] += (vel_ref[0]-var_p_save[i][10])*CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;
        vel_int[1] += (vel_ref[1]-var_p_save[i][11])*CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;
        vel_int[2] += (vel_ref[2]-var_p_save[i][12])*CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;
        vel_int[2] = fminf(fmaxf(vel_int[2], -CONST_PARAM_FLOAT::A_OF_GRAVITY), CONST_PARAM_FLOAT::A_OF_GRAVITY);
        // コストの計算
        cost += (_COST_Q_X*(var_p_save[i][7] -target_host.x )*(var_p_save[i][7] -target_host.x) + _COST_Q_Y *(var_p_save[i][8] -target_host.y) *(var_p_save[i][8] -target_host.y) + _COST_Q_Z *(var_p_save[i][9] -target_host.z) *(var_p_save[i][9] -target_host.z)     // x, y, z
                +  _COST_Q_XP*(var_p_save[i][10]-target_host.xp)*(var_p_save[i][10]-target_host.xp)+ _COST_Q_YP*(var_p_save[i][11]-target_host.yp)*(var_p_save[i][11]-target_host.yp)+ _COST_Q_ZP*(var_p_save[i][12]-target_host.zp)*(var_p_save[i][12]-target_host.zp)    // xp, yp, zp
                +  _COST_Q_E1*(var_p_save[i][1] -target_host.e1)*(var_p_save[i][1] -target_host.e1)+ _COST_Q_E2*(var_p_save[i][2] -target_host.e2)*(var_p_save[i][2] -target_host.e2)+ _COST_Q_E3*(var_p_save[i][3] -target_host.e3)*(var_p_save[i][3] -target_host.e3)         // e1, e2, e3
                +  _COST_Q_WX*(var_p_save[i][4] -target_host.wx)*(var_p_save[i][4] -target_host.wx)+ _COST_Q_WY*(var_p_save[i][5] -target_host.wy)*(var_p_save[i][5] -target_host.wy)+ _COST_Q_WZ*(var_p_save[i][6] -target_host.wz)*(var_p_save[i][6] -target_host.wz)          // wx, wy, wz                                                                                            // z_i
                +  _COST_R_X*(decoupled_position[x]-qc_mcmpc::target_host.x)*(decoupled_position[x]-qc_mcmpc::target_host.x)
                +  _COST_R_Y*(decoupled_position[y]-qc_mcmpc::target_host.y)*(decoupled_position[y]-qc_mcmpc::target_host.y)
                +  _COST_R_Z*(decoupled_position[z]-qc_mcmpc::target_host.z)*(decoupled_position[z]-qc_mcmpc::target_host.z)
                +  _COST_R_YAW*decoupled_position[yaw]*decoupled_position[yaw]
        );
    }
    sum_cost += cost;
}

