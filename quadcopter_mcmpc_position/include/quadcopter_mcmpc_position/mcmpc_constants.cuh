#pragma once
#include "quadcopter_mcmpc_position/const_params.hpp"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

namespace qc_mcmpc{
#ifdef MCMPC_WITH_FORCE_STATE
    struct target_state_t
    {
        float e0;
        float e1;
        float e2;
        float e3;
        float wx;
        float wy;
        float wz;
        float x;
        float y;
        float z;
        float xp;
        float yp;
        float zp;
        float fx;
        float fy;
        float fz;
    };
#else
    struct target_state_t
    {
        float e0;
        float e1;
        float e2;
        float e3;
        float wx;
        float wy;
        float wz;
        float x;
        float y;
        float z;
        float xp;
        float yp;
        float zp;
    };
#endif
    // 入力列
    class input_array
    {
    public:
        float decoupled_position[_DEVICE_CONST_HORIZON][4];
        float cost;

        __device__ void generate_input_array(curandState &state);
        __device__ void do_simulation(int sample_id);
    };

    // __constant__ GPUメモリ上に展開する定数　(初期化時のみ変更)
    extern __constant__ target_state_t target_state_device;

    extern __constant__ float u_g_device;
    extern __constant__ float u_upper_lim_device;
    extern __constant__ float u_lower_lim_device;

    extern __constant__ float i_xx_device;
    extern __constant__ float i_yy_device;
    extern __constant__ float i_zz_device;
    extern __constant__ float rotor_distance_device;
    extern __constant__ float max_rps_pow_device;
    extern __constant__ float max_thrust_device;
    extern __constant__ float mass_of_machine_device;
    extern __constant__ float torque_rate_device;
    extern __constant__ float a_of_gravity_device;

    extern __constant__ float mpc_xy_p;
    extern __constant__ float mpc_z_p;
    extern __constant__ float mpc_xy_vel_p_acc;
    extern __constant__ float mpc_z_vel_p_acc;
    extern __constant__ float mpc_xy_vel_i_acc;
    extern __constant__ float mpc_z_vel_i_acc;
    extern __constant__ float mpc_xy_vel_d_acc;
    extern __constant__ float mpc_z_vel_d_acc;
    extern __constant__ float mpc_xy_vel_max;
    extern __constant__ float mpc_z_vel_max_up;
    extern __constant__ float mpc_z_vel_max_down;
    extern __constant__ float mpc_thr_min;
    extern __constant__ float mpc_thr_max;
    extern __constant__ float mpc_thr_xy_margin;
    extern __constant__ float arw_gain;
    extern __constant__ float mc_roll_p;
    extern __constant__ float mc_pitch_p;
    extern __constant__ float mc_yaw_p;
    extern __constant__ float mc_yaw_weight;
    extern __constant__ float mc_rollrate_p;
    extern __constant__ float mc_pitchrate_p;
    extern __constant__ float mc_yawrate_p;
    extern __constant__ float mc_rollrate_d;
    extern __constant__ float mc_pitchrate_d;
    extern __constant__ float mc_yawrate_d;
    extern __constant__ float mc_rollrate_i;
    extern __constant__ float mc_pitchrate_i;
    extern __constant__ float mc_yawrate_i;
    extern __constant__ float lpf;
    extern __constant__ float mpc_thr_hover;
    extern __constant__ float mpc_veld_lp;

    extern __constant__ float control_period_device;
    extern __constant__ float integration_step_size_device;

    extern __constant__ float prev_velocity_device[3];
    extern __constant__ float vel_int_device[3];
    extern __constant__ float prev_acceleration_device[3];
    extern __constant__ float prev_angular_velocity_device[3];
    extern __constant__ float prev_angular_acceleration_device[3];
    extern __constant__ float rate_int_device[3];
    extern __constant__ float prev_motor_speed_device[4];
    extern __constant__ int prev_motor_speed_valid_device;

    // __constant__ GPUのconstantメモリ（各制御周期ごとにCPUから更新）
    extern __constant__ float var_and_z_i_device[_N_OF_ODES];
    extern __constant__ input_array average_input_device;
    extern __constant__ float sigma_k_device[4];
    extern __constant__ int square_waypoint_index_device;
    extern __constant__ float square_waypoint_change_time_device;
    extern __constant__ float mcmpc_log_device;
    extern __constant__ float square_waypoints_device[_SQUARE_WAYPOINTS][3];
    extern __constant__ int square_waypoint_count_device;
    extern __constant__ float square_waypoint_threshold_device;
    extern __constant__ int takeoff_state_device;
    extern __constant__ float takeoff_tilt_limit_device;
    extern __constant__ int landed_device;
    extern __constant__ int ground_contact_device;
    extern __constant__ int maybe_landed_device;

    extern __constant__ float motor_input_scaling;
    extern __constant__ float max_rot_velocity;
    extern __constant__ float motor_time_constant_up;
    extern __constant__ float motor_time_constant_down;
    extern __constant__ float motor_thrust_constant;
    extern __constant__ float moment_constant;

    extern __constant__ float rotor_positions[4][3];
    extern __constant__ float rotor_yaw_signs[4];
    extern __constant__ float px4_quad_x_mix[4][4];
    extern __constant__ float px4_quad_x_mix_inv[4][4];
    extern __constant__ float px4_actuator_min[4];
    extern __constant__ float px4_actuator_max[4];
    extern __constant__ float ca_minimum_yaw_margin;

    extern __constant__ float acceleration_bias_device[3];

    extern __constant__ float angular_accel_lp;
    extern __constant__ float mc_rollrate_k;
    extern __constant__ float mc_pitchrate_k;
    extern __constant__ float mc_yawrate_k;
    extern __constant__ float mc_rollrate_ff;
    extern __constant__ float mc_pitchrate_ff;
    extern __constant__ float mc_yawrate_ff;
    extern __constant__ float mc_rr_int_lim;
    extern __constant__ float mc_pr_int_lim;
    extern __constant__ float mc_yr_int_lim;
    extern __constant__ float mc_yaw_tq_cutoff;

    extern __constant__ int   takeoff_state_rampup_device;
    extern __constant__ int   takeoff_state_flight_device;
    extern __constant__ float mpc_tilt_max_device;

    extern __constant__ int motor_command_delay_steps;

    extern __device__ float deterministic_sim_trajectory_device[_DEVICE_CONST_HORIZON][_N_OF_ODES];
    extern __device__ float deterministic_sim_vel_int_device[_DEVICE_CONST_HORIZON][3];
    extern __device__ float deterministic_sim_prev_acceleration_device[_DEVICE_CONST_HORIZON][3];
    extern __device__ float deterministic_sim_rate_int_device[_DEVICE_CONST_HORIZON][3];
    extern __device__ float deterministic_sim_prev_angular_acceleration_device[_DEVICE_CONST_HORIZON][3];
    extern __device__ float deterministic_sim_motor_speed_device[_DEVICE_CONST_HORIZON][4];
    extern __device__ float deterministic_sim_vel_setpoint_device[_DEVICE_CONST_HORIZON][3];
    extern __device__ float deterministic_sim_acc_setpoint_device[_DEVICE_CONST_HORIZON][3];
    extern __device__ float deterministic_sim_att_setpoint_device[_DEVICE_CONST_HORIZON][4];
    extern __device__ float deterministic_sim_thrust_setpoint_device[_DEVICE_CONST_HORIZON][3];
    extern __device__ float deterministic_sim_omega_setpoint_device[_DEVICE_CONST_HORIZON][3];
    extern __device__ float deterministic_sim_torque_setpoint_device[_DEVICE_CONST_HORIZON][3];
    extern __device__ float deterministic_sim_motor_setpoint_device[_DEVICE_CONST_HORIZON][4];
#ifdef PREDICTABLE_COLLISION_WITH_WALL
    extern __device__ int deterministic_sim_collision_device[_DEVICE_CONST_HORIZON];
    extern __device__ float deterministic_sim_contact_point_device[_DEVICE_CONST_HORIZON][3];
    extern __device__ float deterministic_sim_impulse_device[_DEVICE_CONST_HORIZON];
    extern __device__ float deterministic_sim_pre_contact_normal_velocity_device[_DEVICE_CONST_HORIZON];
#endif

    __global__ void simulate_best_input_trajectory_kernel(input_array best_input);

    // These host APIs are implemented in the same CUDA translation unit that
    // owns the device symbols.  Callers linked through mcmpc_core must not use
    // cudaMemcpyToSymbol on those symbols directly: CUDA device symbols are not
    // reliably addressable across a shared-library boundary.
    extern cudaError_t update_target_state_device();
    extern cudaError_t upload_waypoint_config_device(
        const float waypoints[_SQUARE_WAYPOINTS][3],
        int count,
        float threshold);
    extern cudaError_t update_waypoint_progress_device(
        int index,
        float change_time,
        float log_time);
    extern cudaError_t update_mcmpc_log_device(float log_time);

    static float prev_velocity_host[3] = {0.0f, 0.0f, 0.0f};
    static float prev_acceleration_host[3] = {0.0f, 0.0f, 0.0f};
    static float vel_int_host[3] = {0.0f, 0.0f, 0.0f};


    extern target_state_t target_host;

#ifdef PREDICTABLE_COLLISION_WITH_WALL
    extern __constant__ int prediction_wall_collision_enabled_device;
    extern __constant__ int truth_wall_collision_enabled_device;
    extern __constant__ float x_wall_device;
    extern __constant__ float wall_nv_x_device;
    extern __constant__ float wall_nv_y_device;
    extern __constant__ float wall_nv_z_device;
    extern __constant__ float r_of_ring_device;
    extern __constant__ float coeff_of_rest_device;
#endif
}
