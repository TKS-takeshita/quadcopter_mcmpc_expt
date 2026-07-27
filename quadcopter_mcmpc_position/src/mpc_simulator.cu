#include "quadcopter_mcmpc_position/mcmpc_controller.cuh"
#include "quadcopter_mcmpc_position/const_params.hpp"

#include <cmath>

namespace qc_mcmpc
{
    __host__ __device__ static void sin_cosf(float x, float* s, float* c);
    __host__ __device__ static void sin_cosf(float x, float* s, float* c){
            // wrap [-pi, pi]
        if (x > M_PI) x -= 2.0f * M_PI;
        if (x < -M_PI) x += 2.0f * M_PI;
        float x2 = x * x;
        // 5次近似
        *s = x * (1.0f - x2 / 6.0f + x2 * x2 / 120.0f);
        *c = 1.0f - x2 / 2.0f + x2 * x2 / 24.0f;
    }

    __device__ static void quat_mul_device(const float a[4], const float b[4], float out[4])
    {
        out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
        out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
        out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
        out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
    }

    __device__ static void quat_normalize_device(float q[4])
    {
        float n2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
        if (n2 > 1.0e-12f) {
            float inv = rsqrtf(n2);
            q[0] *= inv;
            q[1] *= inv;
            q[2] *= inv;
            q[3] *= inv;
        } else {
            q[0] = 1.0f;
            q[1] = q[2] = q[3] = 0.0f;
        }
    }

    __device__ static void desaturate_motor_outputs_device(
        float motor_raw[4],
        int mix_axis,
        bool increase_only,
        float actuator_max)
    {
        float k_min = 0.0f;
        float k_max = 0.0f;
        for (int m = 0; m < 4; m++) {
            float desat = px4_quad_x_mix[m][mix_axis];
            if (fabsf(desat) < 0.2f) continue;
            if (motor_raw[m] < 0.0f) {
                float k = -motor_raw[m] * __frcp_rn(desat);
                k_min = fminf(k_min, k);
                k_max = fmaxf(k_max, k);
            }
            if (motor_raw[m] > actuator_max) {
                float k = (actuator_max - motor_raw[m]) * __frcp_rn(desat);
                k_min = fminf(k_min, k);
                k_max = fmaxf(k_max, k);
            }
        }
        float gain = k_min + k_max;
        if (increase_only && gain < 0.0f) {
            return;
        }

        for (int m = 0; m < 4; m++) motor_raw[m] += gain * px4_quad_x_mix[m][mix_axis];
        k_min = 0.0f;
        k_max = 0.0f;
        for (int m = 0; m < 4; m++) {
            float desat = px4_quad_x_mix[m][mix_axis];
            if (fabsf(desat) < 0.2f) continue;
            if (motor_raw[m] < 0.0f) {
                float k = -motor_raw[m] * __frcp_rn(desat);
                k_min = fminf(k_min, k);
                k_max = fmaxf(k_max, k);
            }
            if (motor_raw[m] > actuator_max) {
                float k = (actuator_max - motor_raw[m]) * __frcp_rn(desat);
                k_min = fminf(k_min, k);
                k_max = fmaxf(k_max, k);
            }
        }
        for (int m = 0; m < 4; m++) motor_raw[m] += 0.5f * (k_min + k_max) * px4_quad_x_mix[m][mix_axis];
    }

    __device__ static void allocate_px4_quad_x_motor_setpoint_device(const float control_sp[4], float motor_setpoint[4])
    {
        float motor_raw[4];
        for (int m = 0; m < 4; m++) {
            motor_raw[m] = px4_quad_x_mix[m][0] * control_sp[0] + px4_quad_x_mix[m][1] * control_sp[1] + px4_quad_x_mix[m][3] * control_sp[3];
        }

        desaturate_motor_outputs_device(motor_raw, 3, true,  px4_actuator_max[0]); // thrust_z
        desaturate_motor_outputs_device(motor_raw, 0, false, px4_actuator_max[0]); // roll
        desaturate_motor_outputs_device(motor_raw, 1, false, px4_actuator_max[0]); // pitch

        for (int m = 0; m < 4; m++) {
            motor_raw[m] += px4_quad_x_mix[m][2] * control_sp[2]; // yaw
        }

        float yaw_actuator_max =
            px4_actuator_max[0] +
            (px4_actuator_max[0] - px4_actuator_min[0]) * ca_minimum_yaw_margin;

        desaturate_motor_outputs_device(motor_raw, 2, false, yaw_actuator_max);     // yaw
        desaturate_motor_outputs_device(motor_raw, 3, true,  px4_actuator_max[0]);  // thrust_z

        for (int m = 0; m < 4; m++) {
            motor_setpoint[m] = fminf(
                fmaxf(motor_raw[m], px4_actuator_min[m]),
                px4_actuator_max[m]
            );
        }
    }

    __device__ static void allocate_px4_quad_x_device(const float control_sp[4],float motor_setpoint[4],float unallocated_control[4])
    {
        allocate_px4_quad_x_motor_setpoint_device(control_sp, motor_setpoint);

        for (int axis = 0; axis < 4; axis++) {
            float allocated = 0.0f;
            for (int m = 0; m < 4; m++) {
                allocated += px4_quad_x_mix_inv[axis][m] * motor_setpoint[m];
            }
            unallocated_control[axis] = control_sp[axis] - allocated;
        }
    }

    __device__ static float motor_speed_ref_from_setpoint_device(float motor_setpoint)
    {
        float u = fminf(fmaxf(motor_setpoint, 0.0f), 1.0f);
        float ref = 1528.43944677f * u - 406.61258522f * u * u;
        return fminf(fmaxf(ref, 0.0f), max_rot_velocity);
    }

    __device__ static void torque_from_motor_setpoint_no_lag_device(const float motor_setpoint[4], float torque_body[3])
    {
        torque_body[0] = 0.0f;
        torque_body[1] = 0.0f;
        torque_body[2] = 0.0f;
        for (int m = 0; m < 4; m++) {
            float motor_speed = motor_speed_ref_from_setpoint_device(motor_setpoint[m]);
            float force = motor_thrust_constant * motor_speed * motor_speed;
            torque_body[0] += rotor_positions[m][1] * (-force);
            torque_body[1] += -rotor_positions[m][0] * (-force);
            torque_body[2] += rotor_yaw_signs[m] * moment_constant * motor_speed * motor_speed;
        }
    }

    __device__ static void estimate_rate_int_bias_torque_device(const float rate_int[3], float thrust_z_setpoint, float bias_torque[3])
    {
        float trim_control[4] = {rate_int[0], rate_int[1], rate_int[2], thrust_z_setpoint};
        float trim_motor[4];
        float trim_torque[3];
        allocate_px4_quad_x_motor_setpoint_device(trim_control, trim_motor);
        torque_from_motor_setpoint_no_lag_device(trim_motor, trim_torque);
        bias_torque[0] = trim_torque[0];
        bias_torque[1] = trim_torque[1];
        bias_torque[2] = trim_torque[2];
    }

#ifdef PREDICTABLE_COLLISION_WITH_WALL
    __device__ static bool apply_guard_wall_collision(
        float state[_N_OF_ODES], float omega_body[3],
        float contact_point_world[3], float& impulse_out,
        float& pre_contact_normal_velocity_out, bool collision_enabled)
    {
        contact_point_world[0] = contact_point_world[1] = contact_point_world[2] = 0.0f;
        impulse_out = 0.0f;
        pre_contact_normal_velocity_out = 0.0f;
        if (!collision_enabled) {
            return false;
        }
        float normal_world[3] = {wall_nv_x_device, wall_nv_y_device, wall_nv_z_device};
        const float normal_norm_squared =
            normal_world[0] * normal_world[0] +
            normal_world[1] * normal_world[1] +
            normal_world[2] * normal_world[2];
        if (normal_norm_squared < 1.0e-12f) {
            return false;
        }
        const float normal_inv_norm = rsqrtf(normal_norm_squared);
        normal_world[0] *= normal_inv_norm;
        normal_world[1] *= normal_inv_norm;
        normal_world[2] *= normal_inv_norm;

        const float q0 = state[0], q1 = state[1], q2 = state[2], q3 = state[3];
        const float rotation[3][3] = {
            {1.0f - 2.0f*(q2*q2 + q3*q3), 2.0f*(q1*q2 - q0*q3), 2.0f*(q1*q3 + q0*q2)},
            {2.0f*(q1*q2 + q0*q3), 1.0f - 2.0f*(q1*q1 + q3*q3), 2.0f*(q2*q3 - q0*q1)},
            {2.0f*(q1*q3 - q0*q2), 2.0f*(q2*q3 + q0*q1), 1.0f - 2.0f*(q1*q1 + q2*q2)},
        };
        float normal_body[3] = {
            rotation[0][0]*normal_world[0] + rotation[1][0]*normal_world[1] + rotation[2][0]*normal_world[2],
            rotation[0][1]*normal_world[0] + rotation[1][1]*normal_world[1] + rotation[2][1]*normal_world[2],
            rotation[0][2]*normal_world[0] + rotation[1][2]*normal_world[1] + rotation[2][2]*normal_world[2],
        };

        float contact_r_body[3] = {0.0f, 0.0f, 0.0f};
        float contact_r_world[3] = {0.0f, 0.0f, 0.0f};
        float minimum_distance = INFINITY;
        const float desired_guard_angle = atan2f(-normal_body[1], -normal_body[0]);
        const float half_guard_angle = 0.5f * _GUARD_ARC_ANGLE_RAD;
        for (int motor = 0; motor < 4; motor++) {
            const float arm_angle = atan2f(rotor_positions[motor][1], rotor_positions[motor][0]);
            float angle_error = atan2f(
                sinf(desired_guard_angle - arm_angle),
                cosf(desired_guard_angle - arm_angle));
            angle_error = fminf(fmaxf(angle_error, -half_guard_angle), half_guard_angle);
            const float guard_angle = arm_angle + angle_error;
            const float candidate_r_body[3] = {
                rotor_positions[motor][0] + r_of_ring_device * cosf(guard_angle),
                rotor_positions[motor][1] + r_of_ring_device * sinf(guard_angle),
                rotor_positions[motor][2],
            };
            const float candidate_r_world[3] = {
                rotation[0][0]*candidate_r_body[0] + rotation[0][1]*candidate_r_body[1] + rotation[0][2]*candidate_r_body[2],
                rotation[1][0]*candidate_r_body[0] + rotation[1][1]*candidate_r_body[1] + rotation[1][2]*candidate_r_body[2],
                rotation[2][0]*candidate_r_body[0] + rotation[2][1]*candidate_r_body[1] + rotation[2][2]*candidate_r_body[2],
            };
            const float signed_distance =
                normal_world[0] * (state[7] + candidate_r_world[0] - x_wall_device) +
                normal_world[1] * (state[8] + candidate_r_world[1]) +
                normal_world[2] * (state[9] + candidate_r_world[2]);
            if (signed_distance < minimum_distance) {
                minimum_distance = signed_distance;
                contact_r_body[0] = candidate_r_body[0];
                contact_r_body[1] = candidate_r_body[1];
                contact_r_body[2] = candidate_r_body[2];
                contact_r_world[0] = candidate_r_world[0];
                contact_r_world[1] = candidate_r_world[1];
                contact_r_world[2] = candidate_r_world[2];
            }
        }

        if (minimum_distance > 0.0f) {
            return false;
        }

        // Remove discrete-time penetration before applying the normal impulse.
        state[7] -= minimum_distance * normal_world[0];
        state[8] -= minimum_distance * normal_world[1];
        state[9] -= minimum_distance * normal_world[2];
        contact_point_world[0] = state[7] + contact_r_world[0];
        contact_point_world[1] = state[8] + contact_r_world[1];
        contact_point_world[2] = state[9] + contact_r_world[2];

        const float omega_cross_r[3] = {
            omega_body[1]*contact_r_body[2] - omega_body[2]*contact_r_body[1],
            omega_body[2]*contact_r_body[0] - omega_body[0]*contact_r_body[2],
            omega_body[0]*contact_r_body[1] - omega_body[1]*contact_r_body[0],
        };
        const float contact_normal_velocity =
            state[10]*normal_world[0] + state[11]*normal_world[1] + state[12]*normal_world[2] +
            omega_cross_r[0]*normal_body[0] + omega_cross_r[1]*normal_body[1] + omega_cross_r[2]*normal_body[2];
        pre_contact_normal_velocity_out = contact_normal_velocity;
        if (contact_normal_velocity >= 0.0f) {
            return true;
        }

        const float r_cross_n[3] = {
            contact_r_body[1]*normal_body[2] - contact_r_body[2]*normal_body[1],
            contact_r_body[2]*normal_body[0] - contact_r_body[0]*normal_body[2],
            contact_r_body[0]*normal_body[1] - contact_r_body[1]*normal_body[0],
        };
        const float inertia_inverse_r_cross_n[3] = {
            r_cross_n[0] * __frcp_rn(i_xx_device),
            r_cross_n[1] * __frcp_rn(i_yy_device),
            r_cross_n[2] * __frcp_rn(i_zz_device),
        };
        const float inertia_term_cross_r[3] = {
            inertia_inverse_r_cross_n[1]*contact_r_body[2] - inertia_inverse_r_cross_n[2]*contact_r_body[1],
            inertia_inverse_r_cross_n[2]*contact_r_body[0] - inertia_inverse_r_cross_n[0]*contact_r_body[2],
            inertia_inverse_r_cross_n[0]*contact_r_body[1] - inertia_inverse_r_cross_n[1]*contact_r_body[0],
        };
        const float effective_mass_inverse = __frcp_rn(mass_of_machine_device) +
            inertia_term_cross_r[0]*normal_body[0] +
            inertia_term_cross_r[1]*normal_body[1] +
            inertia_term_cross_r[2]*normal_body[2];
        if (effective_mass_inverse <= 1.0e-9f) {
            return true;
        }
        const float impulse =
            -(1.0f + coeff_of_rest_device) * contact_normal_velocity *
            __frcp_rn(effective_mass_inverse);
        impulse_out = impulse;
        const float impulse_over_mass = impulse * __frcp_rn(mass_of_machine_device);
        state[10] += impulse_over_mass * normal_world[0];
        state[11] += impulse_over_mass * normal_world[1];
        state[12] += impulse_over_mass * normal_world[2];
        omega_body[0] += impulse * inertia_inverse_r_cross_n[0];
        omega_body[1] += impulse * inertia_inverse_r_cross_n[1];
        omega_body[2] += impulse * inertia_inverse_r_cross_n[2];
        state[4] = omega_body[0];
        state[5] = omega_body[1];
        state[6] = omega_body[2];
        return true;
    }
#endif

    __device__ void input_array::generate_input_array(curandState &state)
    {
#ifdef USE_INPUT_INTERPOLATION
        // 補間を利用した入力列の生成
        int t_randp[_DEVICE_CONST_N_OF_RANDP];
        for ( int i = 0; i < _DEVICE_CONST_N_OF_RANDP; i++ )
        {
            t_randp[i] = (_DEVICE_CONST_HORIZON / (_DEVICE_CONST_N_OF_RANDP - 1)) * i;    // n_of_randp = 6, horizon 150 -> t_randp[i] = 0, 30, 60, 90, 120, 149
            if ( t_randp[i] == _DEVICE_CONST_HORIZON )
                t_randp[i]--;

            // 代表点のみ，乱数で生成
            for ( int j = 0; j < 4; j++ )
                decoupled_position[t_randp[i]][j] = average_input_device.decoupled_position[t_randp[i]][j] + curand_normal( &state ) * sigma_k_device[j];
        }

        // 他の点をラグランジュ補間で生成
        int randp_ctr = 0;
        for ( int i = 1; i < _DEVICE_CONST_HORIZON; i++ )
        {
            // 乱数で生成済みの時刻はスキップ
            bool continue_flag = false;
            for ( int j = 0; j < _DEVICE_CONST_N_OF_RANDP; j++ )
                if ( i == t_randp[j] )
                    continue_flag = true;

            if ( continue_flag )
            {
                if ( randp_ctr + _DEVICE_CONST_ORDER_OF_INTERPOLATION + 1 < _DEVICE_CONST_N_OF_RANDP )
                    randp_ctr++;
                continue;
            }

            // ラグランジュ補間
            for ( int j = 0; j < 4; j++ )
            {
                float v_temp = 0.0f;

                for ( int i1 = 0; i1 < _DEVICE_CONST_ORDER_OF_INTERPOLATION + 1; i1++ )
                {
                    float ln = 1.0f, ld = 1.0f;
                    for ( int i2 = 0; i2 < _DEVICE_CONST_ORDER_OF_INTERPOLATION + 1; i2++ )
                    {
                        if ( i2 != i1 )
                        {
                            ln *= (float)(i - t_randp[randp_ctr + i2]);
                            ld *= (float)(t_randp[randp_ctr + i1] - t_randp[randp_ctr + i2]);
                        }
                    }
                    v_temp += decoupled_position[t_randp[randp_ctr + i1]][j] * ln / ld;
                }
                decoupled_position[i][j] = v_temp;
            }
        }
#else
        // 入力列の生成（補間による次元削減なし）
        for ( int i = 0; i < _DEVICE_CONST_HORIZON; i++ )
            for ( int j = 0; j < 4; j++ ){
                decoupled_position[i][j] = average_input_device.decoupled_position[i][j] + curand_normal( &state ) * sigma_k_device[j];
            }
#endif
    }

    __device__ void input_array::do_simulation(int sample_id)
    {
        cost = 0.0f;

        // __constant__ メモリから状態量をコピー
        float var_and_z_i_temp[_N_OF_ODES];
        for ( int i = 0; i < _N_OF_ODES; i++ )
            var_and_z_i_temp[i] = var_and_z_i_device[i];
        
        // 衝突予測用変数
#ifdef PREDICTABLE_COLLISION_WITH_WALL
        bool col_constraint_flag = false;
#endif
        float prev_vel[3];
        prev_vel[0] = prev_velocity_device[0];
        prev_vel[1] = prev_velocity_device[1];
        prev_vel[2] = prev_velocity_device[2];
        // 前時刻までの誤差の積分値
        float vel_int[3];
        vel_int[0] = vel_int_device[0];
        vel_int[1] = vel_int_device[1];
        vel_int[2] = vel_int_device[2];
        float prev_acc[3];
        prev_acc[0] = prev_acceleration_device[0];
        prev_acc[1] = prev_acceleration_device[1];
        prev_acc[2] = prev_acceleration_device[2];
        float prev_omega[3];
        prev_omega[0] = prev_angular_velocity_device[0];
        prev_omega[1] = prev_angular_velocity_device[1];
        prev_omega[2] = prev_angular_velocity_device[2];
        float prev_omega_dot[3];
        prev_omega_dot[0] = prev_angular_acceleration_device[0];
        prev_omega_dot[1] = prev_angular_acceleration_device[1];
        prev_omega_dot[2] = prev_angular_acceleration_device[2];
        float rate_int[3];
        rate_int[0] = rate_int_device[0];
        rate_int[1] = rate_int_device[1];
        rate_int[2] = rate_int_device[2];
        bool saturation_positive[3] = {false, false, false};
        bool saturation_negative[3] = {false, false, false};
        float yaw_torque_lpf_state = 0.0f;
        bool yaw_torque_lpf_initialized = false;
        float motor_speed_state[4] = {
            prev_motor_speed_device[0],
            prev_motor_speed_device[1],
            prev_motor_speed_device[2],
            prev_motor_speed_device[3],
        };
        bool motor_speed_initialized = (prev_motor_speed_valid_device != 0);
        float vel_setpoint[3];
        float acc_setpoint[3];
        float att_setpoint[4];
        float thrust_setpoint[3];
        float omega_setpoint[3];

        // Project the current position onto the active path segment.  The scalar
        // progress on that segment is carried forward as a prediction state.
        int ref_segment_index = square_waypoint_index_device - 1;
        ref_segment_index = max(0, min(ref_segment_index, square_waypoint_count_device - 2));
        const float initial_segment_x =
            square_waypoints_device[ref_segment_index + 1][0] -
            square_waypoints_device[ref_segment_index][0];
        const float initial_segment_y =
            square_waypoints_device[ref_segment_index + 1][1] -
            square_waypoints_device[ref_segment_index][1];
        const float initial_segment_z =
            square_waypoints_device[ref_segment_index + 1][2] -
            square_waypoints_device[ref_segment_index][2];
        const float initial_segment_length_sq =
            initial_segment_x * initial_segment_x +
            initial_segment_y * initial_segment_y +
            initial_segment_z * initial_segment_z;
        float segment_x = initial_segment_x;
        float segment_y = initial_segment_y;
        float segment_z = initial_segment_z;
        float segment_length = initial_segment_length_sq > 1.0e-12f
            ? sqrtf(initial_segment_length_sq) : 0.0f;
        float segment_inv_length = initial_segment_length_sq > 1.0e-12f
            ? rsqrtf(initial_segment_length_sq) : 0.0f;
        float ref_segment_progress = 0.0f;
        if (initial_segment_length_sq > 1.0e-12f) {
            ref_segment_progress = fminf(fmaxf((
                (var_and_z_i_temp[7] - square_waypoints_device[ref_segment_index][0]) * initial_segment_x +
                (var_and_z_i_temp[8] - square_waypoints_device[ref_segment_index][1]) * initial_segment_y +
                (var_and_z_i_temp[9] - square_waypoints_device[ref_segment_index][2]) * initial_segment_z
            ) * segment_inv_length, 0.0f), segment_length);
        }
#ifdef PREDICTABLE_COLLISION_WITH_WALL
        int predicted_mission_index = square_waypoint_index_device;
        bool predicted_post_collision_phase = false;
        float post_collision_elapsed = 0.0f;
        float post_collision_origin[3] = {0.0f, 0.0f, 0.0f};
        float post_collision_delta[3] = {0.0f, 0.0f, 0.0f};
        float post_collision_distance = 0.0f;
        float post_collision_inv_distance = 0.0f;
#endif

        // ホライズン内で変化しない制御・モデル係数はサンプルごとに一度だけ計算する。
        const bool flying = takeoff_state_device >= takeoff_state_flight_device;
        const bool landed_or_maybe_landed = landed_device || maybe_landed_device;
        const bool flying_but_ground_contact = flying && (ground_contact_device || maybe_landed_device);
        const bool no_thrust = (takeoff_state_device < takeoff_state_rampup_device) || flying_but_ground_contact;
        const float thrust_min = flying ? mpc_thr_min : 0.0f;

        float tilt_limit = takeoff_tilt_limit_device;
        if (!isfinite(tilt_limit) || tilt_limit <= 0.0f) {
            tilt_limit = mpc_tilt_max_device;
        }
        const float sin_tilt_limit = sinf(tilt_limit);
        const float cos_tilt_limit = cosf(tilt_limit);

        float hover_thrust = mpc_thr_hover;
        if (!isfinite(hover_thrust) || hover_thrust < 1.0e-6f) {
            hover_thrust = 0.60f;
        }
        const float hover_over_gravity = hover_thrust * __frcp_rn(a_of_gravity_device);
        const float gravity_over_hover = a_of_gravity_device * __frcp_rn(hover_thrust);
        const float thrust_max_squared = mpc_thr_max * mpc_thr_max;
        const float inv_control_period = __frcp_rn(control_period_device);
        const float inv_mass = __frcp_rn(mass_of_machine_device);
        const float inv_i_xx = __frcp_rn(i_xx_device);
        const float inv_i_yy = __frcp_rn(i_yy_device);
        const float inv_i_zz = __frcp_rn(i_zz_device);

        const float vel_dot_alpha = (mpc_veld_lp > 1.0e-6f)
            ? control_period_device * __frcp_rn(control_period_device +
                __frcp_rn(2.0f * M_PI * mpc_veld_lp))
            : 1.0f;
        const float omega_dot_lpf_alpha = (angular_accel_lp > 1.0e-6f)
            ? control_period_device * __frcp_rn(control_period_device +
                __frcp_rn(2.0f * M_PI * angular_accel_lp))
            : 1.0f;
        const float yaw_alpha = (mc_yaw_tq_cutoff > 1.0e-6f)
            ? control_period_device * __frcp_rn(control_period_device +
                __frcp_rn(2.0f * M_PI * mc_yaw_tq_cutoff))
            : 1.0f;
        const float yaw_gain = (mc_yaw_weight > 1.0e-4f)
            ? mc_yaw_p * __frcp_rn(mc_yaw_weight) : mc_yaw_p;
        const float rate_i_gain[3] = {
            mc_rollrate_k * mc_rollrate_i,
            mc_pitchrate_k * mc_pitchrate_i,
            mc_yawrate_k * mc_yawrate_i,
        };
        const float rate_int_lim[3] = {mc_rr_int_lim, mc_pr_int_lim, mc_yr_int_lim};
        const float motor_alpha_up = expf(
            -control_period_device * __frcp_rn(fmaxf(motor_time_constant_up, 1.0e-9f)));
        const float motor_alpha_down = expf(
            -control_period_device * __frcp_rn(fmaxf(motor_time_constant_down, 1.0e-9f)));

        // 内部で位置/ヨー入力からsetpointを計算し，同定済み離散モデルで状態遷移する
        for ( int i = 0; i < _DEVICE_CONST_HORIZON; i++ )
        {
            // PX4のPIDを素に速度, 加速度, 姿勢, スラスト, 角速度を導出
            float x_ref   = decoupled_position[i][x];
            float y_ref   = decoupled_position[i][y];
            float z_ref   = decoupled_position[i][z];
            float yaw_ref = decoupled_position[i][yaw];
            vel_setpoint[0] = mpc_xy_p * (x_ref - var_and_z_i_temp[7]);
            vel_setpoint[1] = mpc_xy_p * (y_ref - var_and_z_i_temp[8]);
            vel_setpoint[2] = mpc_z_p  * (z_ref - var_and_z_i_temp[9]);

            const float vel_xy_norm_squared =
                vel_setpoint[0]*vel_setpoint[0] + vel_setpoint[1]*vel_setpoint[1];
            if (vel_xy_norm_squared > mpc_xy_vel_max * mpc_xy_vel_max &&
                vel_xy_norm_squared > 1.0e-16f) {
                const float vel_scale = mpc_xy_vel_max * rsqrtf(vel_xy_norm_squared);
                vel_setpoint[0] *= vel_scale;
                vel_setpoint[1] *= vel_scale;
            }
            vel_setpoint[2] = fminf(fmaxf(vel_setpoint[2], -mpc_z_vel_max_up), mpc_z_vel_max_down);

            float vel_error[3];
            vel_error[0] = vel_setpoint[0] - var_and_z_i_temp[10];
            vel_error[1] = vel_setpoint[1] - var_and_z_i_temp[11];
            vel_error[2] = vel_setpoint[2] - var_and_z_i_temp[12];

            float vel_dot_x = (var_and_z_i_temp[10] - prev_vel[0]) * inv_control_period;
            float vel_dot_y = (var_and_z_i_temp[11] - prev_vel[1]) * inv_control_period;
            float vel_dot_z = (var_and_z_i_temp[12] - prev_vel[2]) * inv_control_period;

            float vel_dot_lpf[3];
            vel_dot_lpf[0] = prev_acc[0] + vel_dot_alpha * (vel_dot_x - prev_acc[0]);
            vel_dot_lpf[1] = prev_acc[1] + vel_dot_alpha * (vel_dot_y - prev_acc[1]);
            vel_dot_lpf[2] = prev_acc[2] + vel_dot_alpha * (vel_dot_z - prev_acc[2]);

            acc_setpoint[0] = mpc_xy_vel_p_acc * vel_error[0] + vel_int[0] - mpc_xy_vel_d_acc * vel_dot_lpf[0];
            acc_setpoint[1] = mpc_xy_vel_p_acc * vel_error[1] + vel_int[1] - mpc_xy_vel_d_acc * vel_dot_lpf[1];
            acc_setpoint[2] = mpc_z_vel_p_acc  * vel_error[2] + vel_int[2] - mpc_z_vel_d_acc  * vel_dot_lpf[2];

            float body_z[3];
            body_z[0] = -acc_setpoint[0];
            body_z[1] = -acc_setpoint[1];
            body_z[2] =  a_of_gravity_device - acc_setpoint[2];

            float bz_norm_inv = rsqrtf(body_z[0]*body_z[0]+body_z[1]*body_z[1]+body_z[2]*body_z[2]);
            body_z[0] *= bz_norm_inv;
            body_z[1] *= bz_norm_inv;
            body_z[2] *= bz_norm_inv;

            // acosf(body_z[2]) > tilt_limit と同値の単調比較。
            if (body_z[2] < cos_tilt_limit) {
                float rejection[3];
                rejection[0] = body_z[0];
                rejection[1] = body_z[1];
                rejection[2] = 0.0f;
                const float rejection_norm_squared =
                    rejection[0]*rejection[0] + rejection[1]*rejection[1];
                if (rejection_norm_squared < 1.0e-16f) {
                    rejection[0] = 1.0f;
                    rejection[1] = 0.0f;
                } else {
                    const float rejection_inv_norm = rsqrtf(rejection_norm_squared);
                    rejection[0] *= rejection_inv_norm;
                    rejection[1] *= rejection_inv_norm;
                }
                body_z[0] = sin_tilt_limit * rejection[0];
                body_z[1] = sin_tilt_limit * rejection[1];
                body_z[2] = cos_tilt_limit;
            }

            if (no_thrust) {
                acc_setpoint[0] = 0.0f;
                acc_setpoint[1] = 0.0f;
                acc_setpoint[2] = 100.0f;
            }

            float thrust_ned_z = acc_setpoint[2] * hover_over_gravity - hover_thrust;
            float cos_ned_body = body_z[2];
            if (fabsf(cos_ned_body) < 1.0e-6f) cos_ned_body = 1.0e-6f;

            float collective_thrust = fminf(thrust_ned_z * __frcp_rn(cos_ned_body), -thrust_min);
            thrust_setpoint[0] = body_z[0] * collective_thrust;
            thrust_setpoint[1] = body_z[1] * collective_thrust;
            thrust_setpoint[2] = body_z[2] * collective_thrust;
            if (no_thrust) {
                thrust_setpoint[0] = 0.0f;
                thrust_setpoint[1] = 0.0f;
                thrust_setpoint[2] = 0.0f;
            }

            // mcmpc_viewer と同じ thrust saturation
            const float thrust_sp_xy_norm_squared =
                thrust_setpoint[0]*thrust_setpoint[0] + thrust_setpoint[1]*thrust_setpoint[1];
            float thrust_sp_xy_norm = sqrtf(thrust_sp_xy_norm_squared);
            float allocated_horizontal_thrust = fminf(thrust_sp_xy_norm, mpc_thr_xy_margin);
            float thrust_z_max_squared = thrust_max_squared - allocated_horizontal_thrust * allocated_horizontal_thrust;
            thrust_setpoint[2] = fmaxf(thrust_setpoint[2], -sqrtf(fmaxf(0.0f, thrust_z_max_squared)));

            float thrust_max_xy_squared = thrust_max_squared - thrust_setpoint[2] * thrust_setpoint[2];
            float thrust_max_xy = sqrtf(fmaxf(0.0f, thrust_max_xy_squared));
            if (thrust_sp_xy_norm > thrust_max_xy && thrust_sp_xy_norm > 1.0e-8f) {
                const float thrust_xy_scale = thrust_max_xy * rsqrtf(thrust_sp_xy_norm_squared);
                thrust_setpoint[0] *= thrust_xy_scale;
                thrust_setpoint[1] *= thrust_xy_scale;
            }

            float acc_sp_xy_produced[2];
            acc_sp_xy_produced[0] = thrust_setpoint[0] * gravity_over_hover;
            acc_sp_xy_produced[1] = thrust_setpoint[1] * gravity_over_hover;

            float vel_error_for_int[3];
            vel_error_for_int[0] = vel_error[0];
            vel_error_for_int[1] = vel_error[1];
            vel_error_for_int[2] = vel_error[2];
            if ((thrust_setpoint[2] >= -thrust_min && vel_error_for_int[2] >= 0.0f) ||
                (thrust_setpoint[2] <= -mpc_thr_max && vel_error_for_int[2] <= 0.0f)) {
                vel_error_for_int[2] = 0.0f;
            }
            if (acc_setpoint[0] * acc_setpoint[0] + acc_setpoint[1] * acc_setpoint[1] >
                acc_sp_xy_produced[0] * acc_sp_xy_produced[0] + acc_sp_xy_produced[1] * acc_sp_xy_produced[1]) {
                float xy_arw_gain = arw_gain;
                vel_error_for_int[0] -= xy_arw_gain * (acc_setpoint[0] - acc_sp_xy_produced[0]);
                vel_error_for_int[1] -= xy_arw_gain * (acc_setpoint[1] - acc_sp_xy_produced[1]);
            }

            float att_body_z[3];
            att_body_z[0] = -thrust_setpoint[0];
            att_body_z[1] = -thrust_setpoint[1];
            att_body_z[2] = -thrust_setpoint[2];
            const float att_bz_norm_squared =
                att_body_z[0]*att_body_z[0]+att_body_z[1]*att_body_z[1]+att_body_z[2]*att_body_z[2];
            if (att_bz_norm_squared < 1.0e-16f) {
                att_body_z[0] = 0.0f;
                att_body_z[1] = 0.0f;
                att_body_z[2] = 1.0f;
            } else {
                const float att_bz_inv_norm = rsqrtf(att_bz_norm_squared);
                att_body_z[0] *= att_bz_inv_norm;
                att_body_z[1] *= att_bz_inv_norm;
                att_body_z[2] *= att_bz_inv_norm;
            }

            float sy, cy;
            sin_cosf(yaw_ref, &sy, &cy);
            float y_c[3]  = {-sy, cy, 0.0f};
            float body_x[3];
            float body_y[3];

            body_x[0] = y_c[1] * att_body_z[2] - y_c[2] * att_body_z[1];
            body_x[1] = y_c[2] * att_body_z[0] - y_c[0] * att_body_z[2];
            body_x[2] = y_c[0] * att_body_z[1] - y_c[1] * att_body_z[0];

            if (att_body_z[2] < 0.0f) {
                body_x[0] = -body_x[0];
                body_x[1] = -body_x[1];
                body_x[2] = -body_x[2];
            }
            if (fabsf(att_body_z[2]) < 1.0e-6f) {
                body_x[0] = 0.0f;
                body_x[1] = 0.0f;
                body_x[2] = 1.0f;
            }

            float bx_norm_inv = rsqrtf(body_x[0]*body_x[0]+body_x[1]*body_x[1]+body_x[2]*body_x[2]);
            body_x[0] *= bx_norm_inv;
            body_x[1] *= bx_norm_inv;
            body_x[2] *= bx_norm_inv;

            body_y[0] = att_body_z[1]*body_x[2] - att_body_z[2]*body_x[1];
            body_y[1] = att_body_z[2]*body_x[0] - att_body_z[0]*body_x[2];
            body_y[2] = att_body_z[0]*body_x[1] - att_body_z[1]*body_x[0];

            float r00 = body_x[0], r01 = body_y[0], r02 = att_body_z[0];
            float r10 = body_x[1], r11 = body_y[1], r12 = att_body_z[1];
            float r20 = body_x[2], r21 = body_y[2], r22 = att_body_z[2];
            float tr = r00 + r11 + r22;
            if (tr > 0.0f) {
                float s = sqrtf(tr + 1.0f) * 2.0f;
                const float inv_s = __frcp_rn(s);
                att_setpoint[0] = 0.25f * s;
                att_setpoint[1] = (r21 - r12) * inv_s;
                att_setpoint[2] = (r02 - r20) * inv_s;
                att_setpoint[3] = (r10 - r01) * inv_s;
            } else if (r00 > r11 && r00 > r22) {
                float s = sqrtf(1.0f + r00 - r11 - r22) * 2.0f;
                const float inv_s = __frcp_rn(s);
                att_setpoint[0] = (r21 - r12) * inv_s;
                att_setpoint[1] = 0.25f * s;
                att_setpoint[2] = (r01 + r10) * inv_s;
                att_setpoint[3] = (r02 + r20) * inv_s;
            } else if (r11 > r22) {
                float s = sqrtf(1.0f + r11 - r00 - r22) * 2.0f;
                const float inv_s = __frcp_rn(s);
                att_setpoint[0] = (r02 - r20) * inv_s;
                att_setpoint[1] = (r01 + r10) * inv_s;
                att_setpoint[2] = 0.25f * s;
                att_setpoint[3] = (r12 + r21) * inv_s;
            } else {
                float s = sqrtf(1.0f + r22 - r00 - r11) * 2.0f;
                const float inv_s = __frcp_rn(s);
                att_setpoint[0] = (r10 - r01) * inv_s;
                att_setpoint[1] = (r02 + r20) * inv_s;
                att_setpoint[2] = (r12 + r21) * inv_s;
                att_setpoint[3] = 0.25f * s;
            }
            float qsp_norm_inv = rsqrtf(att_setpoint[0]*att_setpoint[0] + att_setpoint[1]*att_setpoint[1] + att_setpoint[2]*att_setpoint[2] + att_setpoint[3]*att_setpoint[3]);
            att_setpoint[0] *= qsp_norm_inv;
            att_setpoint[1] *= qsp_norm_inv;
            att_setpoint[2] *= qsp_norm_inv;
            att_setpoint[3] *= qsp_norm_inv;

            float q_prev_for_att[4] = {
                var_and_z_i_temp[0],
                var_and_z_i_temp[1],
                var_and_z_i_temp[2],
                var_and_z_i_temp[3],
            };
            float e_z[3] = {
                2.0f * (q_prev_for_att[1]*q_prev_for_att[3] + q_prev_for_att[0]*q_prev_for_att[2]),
                2.0f * (q_prev_for_att[2]*q_prev_for_att[3] - q_prev_for_att[0]*q_prev_for_att[1]),
                1.0f - 2.0f * (q_prev_for_att[1]*q_prev_for_att[1] + q_prev_for_att[2]*q_prev_for_att[2])
            };
            float e_z_d[3] = {
                2.0f * (att_setpoint[1]*att_setpoint[3] + att_setpoint[0]*att_setpoint[2]),
                2.0f * (att_setpoint[2]*att_setpoint[3] - att_setpoint[0]*att_setpoint[1]),
                1.0f - 2.0f * (att_setpoint[1]*att_setpoint[1] + att_setpoint[2]*att_setpoint[2])
            };
            float tilt_axis[3] = {
                e_z[1]*e_z_d[2] - e_z[2]*e_z_d[1],
                e_z[2]*e_z_d[0] - e_z[0]*e_z_d[2],
                e_z[0]*e_z_d[1] - e_z[1]*e_z_d[0]
            };
            const float tilt_axis_norm_squared =
                tilt_axis[0]*tilt_axis[0] + tilt_axis[1]*tilt_axis[1] + tilt_axis[2]*tilt_axis[2];
            float tilt_axis_norm = sqrtf(tilt_axis_norm_squared);
            float tilt_dot = fminf(fmaxf(e_z[0]*e_z_d[0] + e_z[1]*e_z_d[1] + e_z[2]*e_z_d[2], -1.0f), 1.0f);
            float qd_red[4];
            if (tilt_axis_norm < 1.0e-8f) {
                if (tilt_dot > 0.0f) {
                    qd_red[0] = 1.0f; qd_red[1] = 0.0f; qd_red[2] = 0.0f; qd_red[3] = 0.0f;
                } else {
                    qd_red[0] = att_setpoint[0]; qd_red[1] = att_setpoint[1]; qd_red[2] = att_setpoint[2]; qd_red[3] = att_setpoint[3];
                }
            } else {
                const float tilt_axis_inv_norm = rsqrtf(tilt_axis_norm_squared);
                tilt_axis[0] *= tilt_axis_inv_norm;
                tilt_axis[1] *= tilt_axis_inv_norm;
                tilt_axis[2] *= tilt_axis_inv_norm;
                float tilt_angle_red = atan2f(tilt_axis_norm, tilt_dot);
                qd_red[0] = cosf(0.5f * tilt_angle_red);
                qd_red[1] = tilt_axis[0] * sinf(0.5f * tilt_angle_red);
                qd_red[2] = tilt_axis[1] * sinf(0.5f * tilt_angle_red);
                qd_red[3] = tilt_axis[2] * sinf(0.5f * tilt_angle_red);
                quat_normalize_device(qd_red);
                if (fabsf(qd_red[1]) > 1.0f - 1.0e-5f || fabsf(qd_red[2]) > 1.0f - 1.0e-5f) {
                    qd_red[0] = att_setpoint[0]; qd_red[1] = att_setpoint[1]; qd_red[2] = att_setpoint[2]; qd_red[3] = att_setpoint[3];
                } else {
                    float qd_red_tmp[4];
                    quat_mul_device(qd_red, q_prev_for_att, qd_red_tmp);
                    qd_red[0] = qd_red_tmp[0]; qd_red[1] = qd_red_tmp[1]; qd_red[2] = qd_red_tmp[2]; qd_red[3] = qd_red_tmp[3];
                    quat_normalize_device(qd_red);
                }
            }

            float qd_red_inv[4] = {qd_red[0], -qd_red[1], -qd_red[2], -qd_red[3]};
            float qd_dyaw[4];
            quat_mul_device(qd_red_inv, att_setpoint, qd_dyaw);
            quat_normalize_device(qd_dyaw);
            if (qd_dyaw[0] < 0.0f) {
                qd_dyaw[0] = -qd_dyaw[0];
                qd_dyaw[1] = -qd_dyaw[1];
                qd_dyaw[2] = -qd_dyaw[2];
                qd_dyaw[3] = -qd_dyaw[3];
            }
            qd_dyaw[0] = fminf(fmaxf(qd_dyaw[0], -1.0f), 1.0f);
            qd_dyaw[3] = fminf(fmaxf(qd_dyaw[3], -1.0f), 1.0f);
            float q_yaw_weighted[4] = {
                cosf(mc_yaw_weight * acosf(qd_dyaw[0])),
                0.0f,
                0.0f,
                sinf(mc_yaw_weight * asinf(qd_dyaw[3]))
            };
            float qd_weighted[4];
            quat_mul_device(qd_red, q_yaw_weighted, qd_weighted);
            quat_normalize_device(qd_weighted);
            float q_prev_inv[4] = {q_prev_for_att[0], -q_prev_for_att[1], -q_prev_for_att[2], -q_prev_for_att[3]};
            float qe[4];
            quat_mul_device(q_prev_inv, qd_weighted, qe);
            quat_normalize_device(qe);
            if (qe[0] < 0.0f) {
                qe[0] = -qe[0];
                qe[1] = -qe[1];
                qe[2] = -qe[2];
                qe[3] = -qe[3];
            }
            omega_setpoint[0] = 2.0f * qe[1] * mc_roll_p;
            omega_setpoint[1] = 2.0f * qe[2] * mc_pitch_p;
            omega_setpoint[2] = 2.0f * qe[3] * yaw_gain;
            omega_setpoint[0] = fminf(fmaxf(omega_setpoint[0], -3.8397244f), 3.8397244f);
            omega_setpoint[1] = fminf(fmaxf(omega_setpoint[1], -3.8397244f), 3.8397244f);
            omega_setpoint[2] = fminf(fmaxf(omega_setpoint[2], -3.4906585f), 3.4906585f);

            float q_prev[4] = {
                var_and_z_i_temp[0],
                var_and_z_i_temp[1],
                var_and_z_i_temp[2],
                var_and_z_i_temp[3],
            };
            float v_prev[3] = {
                var_and_z_i_temp[10],
                var_and_z_i_temp[11],
                var_and_z_i_temp[12],
            };
            float w_prev[3] = {
                var_and_z_i_temp[4],
                var_and_z_i_temp[5],
                var_and_z_i_temp[6],
            };
            float raw_omega_dot[3] = {
                (w_prev[0] - prev_omega[0]) * inv_control_period,
                (w_prev[1] - prev_omega[1]) * inv_control_period,
                (w_prev[2] - prev_omega[2]) * inv_control_period,
            };

            float omega_dot[3] = {
                prev_omega_dot[0] + omega_dot_lpf_alpha * (raw_omega_dot[0] - prev_omega_dot[0]),
                prev_omega_dot[1] + omega_dot_lpf_alpha * (raw_omega_dot[1] - prev_omega_dot[1]),
                prev_omega_dot[2] + omega_dot_lpf_alpha * (raw_omega_dot[2] - prev_omega_dot[2]),
            };

            float rate_error[3] = {
                omega_setpoint[0] - w_prev[0],
                omega_setpoint[1] - w_prev[1],
                omega_setpoint[2] - w_prev[2],
            };
            float torque_setpoint[3] = {
                mc_rollrate_k  * mc_rollrate_p  * rate_error[0] + rate_int[0] - mc_rollrate_k  * mc_rollrate_d  * omega_dot[0] + mc_rollrate_ff  * omega_setpoint[0],
                mc_pitchrate_k * mc_pitchrate_p * rate_error[1] + rate_int[1] - mc_pitchrate_k * mc_pitchrate_d * omega_dot[1] + mc_pitchrate_ff * omega_setpoint[1],
                mc_yawrate_k   * mc_yawrate_p   * rate_error[2] + rate_int[2] - mc_yawrate_k   * mc_yawrate_d   * omega_dot[2] + mc_yawrate_ff   * omega_setpoint[2],
            };

            if (mc_yaw_tq_cutoff > 1.0e-6f) {
                if (!yaw_torque_lpf_initialized) {
                    yaw_torque_lpf_state = torque_setpoint[2];
                    yaw_torque_lpf_initialized = true;
                } else {
                    yaw_torque_lpf_state += yaw_alpha * (torque_setpoint[2] - yaw_torque_lpf_state);
                    torque_setpoint[2] = yaw_torque_lpf_state;
                }
            } else {
                yaw_torque_lpf_state = torque_setpoint[2];
                yaw_torque_lpf_initialized = true;
            }

            if (!landed_or_maybe_landed) {
                for (int axis = 0; axis < 3; axis++) {
                    float err_for_int = rate_error[axis];
                    if (saturation_positive[axis]) err_for_int = fminf(err_for_int, 0.0f);
                    if (saturation_negative[axis]) err_for_int = fmaxf(err_for_int, 0.0f);
                    float i_factor = err_for_int * 0.143239450f;
                    i_factor = fmaxf(0.0f, 1.0f - i_factor * i_factor);
                    rate_int[axis] += i_factor * rate_i_gain[axis] * err_for_int * control_period_device;
                    rate_int[axis] = fminf(fmaxf(rate_int[axis], -rate_int_lim[axis]), rate_int_lim[axis]);
                }
            }

            float control_sp[4] = {torque_setpoint[0], torque_setpoint[1], torque_setpoint[2], thrust_setpoint[2]};
            float motor_setpoint[4];
            float unallocated_control[4];
            allocate_px4_quad_x_device(control_sp, motor_setpoint, unallocated_control);
            for (int axis = 0; axis < 3; axis++) {
                float unallocated = unallocated_control[axis];
                saturation_positive[axis] = unallocated > 1.1920929e-7f;
                saturation_negative[axis] = unallocated < -1.1920929e-7f;
            }

            float motor_speed[4];
            for (int m = 0; m < 4; m++) {
                float motor_speed_ref = motor_speed_ref_from_setpoint_device(motor_setpoint[m]);
                if (!motor_speed_initialized) {
                    motor_speed[m] = motor_speed_ref;
                } else {
                    float prev_speed = fminf(fmaxf(motor_speed_state[m], 0.0f), max_rot_velocity);
                    float motor_alpha = (motor_speed_ref > prev_speed)
                        ? motor_alpha_up
                        : motor_alpha_down;
                    motor_speed[m] = motor_alpha * prev_speed + (1.0f - motor_alpha) * motor_speed_ref;
                }
                motor_speed_state[m] = motor_speed[m];
            }
            motor_speed_initialized = true;

            float rotor_forces[4];
            for (int m = 0; m < 4; m++) {
                rotor_forces[m] = motor_thrust_constant * motor_speed[m] * motor_speed[m];
            }
            float force_body_z = -(rotor_forces[0] + rotor_forces[1] + rotor_forces[2] + rotor_forces[3]);
            float torque_body[3] = {0.0f, 0.0f, 0.0f};
            for (int m = 0; m < 4; m++) {
                torque_body[0] += rotor_positions[m][1] * (-rotor_forces[m]);
                torque_body[1] += -rotor_positions[m][0] * (-rotor_forces[m]);
                torque_body[2] += rotor_yaw_signs[m] * moment_constant * motor_speed[m] * motor_speed[m];
            }
            float bias_torque[3];
            estimate_rate_int_bias_torque_device(rate_int, thrust_setpoint[2], bias_torque);
            torque_body[0] -= bias_torque[0];
            torque_body[1] -= bias_torque[1];
            torque_body[2] -= bias_torque[2];
            float q0 = q_prev[0], q1 = q_prev[1], q2 = q_prev[2], q3 = q_prev[3];
            float r02_force = 2.0f * (q1*q3 + q0*q2);
            float r12_force = 2.0f * (q2*q3 - q0*q1);
            float r22_force = 1.0f - 2.0f * (q1*q1 + q2*q2);
            float acc_for_dynamics[3] = {
                r02_force * force_body_z * inv_mass,
                r12_force * force_body_z * inv_mass,
                r22_force * force_body_z * inv_mass + a_of_gravity_device,
            };

            acc_for_dynamics[0] += acceleration_bias_device[0];
            acc_for_dynamics[1] += acceleration_bias_device[1];
            acc_for_dynamics[2] += acceleration_bias_device[2];

            float inertia_omega[3] = {
                i_xx_device * w_prev[0],
                i_yy_device * w_prev[1],
                i_zz_device * w_prev[2],
            };
            float cross_w_iw[3] = {
                w_prev[1]*inertia_omega[2] - w_prev[2]*inertia_omega[1],
                w_prev[2]*inertia_omega[0] - w_prev[0]*inertia_omega[2],
                w_prev[0]*inertia_omega[1] - w_prev[1]*inertia_omega[0],
            };
            float omega_dot_phys[3] = {
                (torque_body[0] - cross_w_iw[0]) * inv_i_xx,
                (torque_body[1] - cross_w_iw[1]) * inv_i_yy,
                (torque_body[2] - cross_w_iw[2]) * inv_i_zz,
            };
            float omega_dot_for_dynamics[3] = {omega_dot_phys[0], omega_dot_phys[1], omega_dot_phys[2]};

            float w_next[3] = {
                w_prev[0] + omega_dot_for_dynamics[0] * control_period_device,
                w_prev[1] + omega_dot_for_dynamics[1] * control_period_device,
                w_prev[2] + omega_dot_for_dynamics[2] * control_period_device,
            };

            var_and_z_i_temp[10] = v_prev[0] + acc_for_dynamics[0] * control_period_device;
            var_and_z_i_temp[11] = v_prev[1] + acc_for_dynamics[1] * control_period_device;
            var_and_z_i_temp[12] = v_prev[2] + acc_for_dynamics[2] * control_period_device;

            var_and_z_i_temp[7] += var_and_z_i_temp[10] * control_period_device;
            var_and_z_i_temp[8] += var_and_z_i_temp[11] * control_period_device;
            var_and_z_i_temp[9] += var_and_z_i_temp[12] * control_period_device;

            float q_dot[4];
            q_dot[0] = -0.5f * (q_prev[1] * w_next[0] + q_prev[2] * w_next[1] + q_prev[3] * w_next[2]);
            q_dot[1] =  0.5f * (q_prev[0] * w_next[0] + q_prev[2] * w_next[2] - q_prev[3] * w_next[1]);
            q_dot[2] =  0.5f * (q_prev[0] * w_next[1] - q_prev[1] * w_next[2] + q_prev[3] * w_next[0]);
            q_dot[3] =  0.5f * (q_prev[0] * w_next[2] + q_prev[1] * w_next[1] - q_prev[2] * w_next[0]);

            var_and_z_i_temp[0] = q_prev[0] + q_dot[0] * control_period_device;
            var_and_z_i_temp[1] = q_prev[1] + q_dot[1] * control_period_device;
            var_and_z_i_temp[2] = q_prev[2] + q_dot[2] * control_period_device;
            var_and_z_i_temp[3] = q_prev[3] + q_dot[3] * control_period_device;

            float q_norm_inv = rsqrtf(
                var_and_z_i_temp[0] * var_and_z_i_temp[0] +
                var_and_z_i_temp[1] * var_and_z_i_temp[1] +
                var_and_z_i_temp[2] * var_and_z_i_temp[2] +
                var_and_z_i_temp[3] * var_and_z_i_temp[3]
            );
            var_and_z_i_temp[0] *= q_norm_inv;
            var_and_z_i_temp[1] *= q_norm_inv;
            var_and_z_i_temp[2] *= q_norm_inv;
            var_and_z_i_temp[3] *= q_norm_inv;

            var_and_z_i_temp[4] = w_next[0];
            var_and_z_i_temp[5] = w_next[1];
            var_and_z_i_temp[6] = w_next[2];

            // Apply the same guard/plane impact transition to sampled rollouts and
            // to the deterministic trajectory used as simulator truth.
#ifdef PREDICTABLE_COLLISION_WITH_WALL
            float collision_contact_point[3];
            float collision_impulse = 0.0f;
            float pre_contact_normal_velocity = 0.0f;
            const bool collision_event = apply_guard_wall_collision(
                var_and_z_i_temp, w_next, collision_contact_point,
                collision_impulse, pre_contact_normal_velocity,
                sample_id < 0 ? truth_wall_collision_enabled_device != 0
                              : prediction_wall_collision_enabled_device != 0);
            col_constraint_flag = col_constraint_flag || collision_event;
            if (collision_event && !predicted_post_collision_phase &&
                predicted_mission_index + 1 < square_waypoint_count_device) {
                predicted_mission_index++;
                predicted_post_collision_phase = true;
                post_collision_elapsed = 0.0f;
                post_collision_origin[0] = var_and_z_i_temp[7];
                post_collision_origin[1] = var_and_z_i_temp[8];
                post_collision_origin[2] = var_and_z_i_temp[9];
                post_collision_delta[0] =
                    square_waypoints_device[predicted_mission_index][0] - post_collision_origin[0];
                post_collision_delta[1] =
                    square_waypoints_device[predicted_mission_index][1] - post_collision_origin[1];
                post_collision_delta[2] =
                    square_waypoints_device[predicted_mission_index][2] - post_collision_origin[2];
                const float post_collision_distance_squared =
                    post_collision_delta[0] * post_collision_delta[0] +
                    post_collision_delta[1] * post_collision_delta[1] +
                    post_collision_delta[2] * post_collision_delta[2];
                if (post_collision_distance_squared > 1.0e-12f) {
                    post_collision_inv_distance = rsqrtf(post_collision_distance_squared);
                    post_collision_distance =
                        post_collision_distance_squared * post_collision_inv_distance;
                }
            }
            if (sample_id < 0) {
                deterministic_sim_collision_device[i] = collision_event ? 1 : 0;
                deterministic_sim_impulse_device[i] = collision_impulse;
                deterministic_sim_pre_contact_normal_velocity_device[i] =
                    pre_contact_normal_velocity;
                for (int axis = 0; axis < 3; axis++) {
                    deterministic_sim_contact_point_device[i][axis] =
                        collision_contact_point[axis];
                }
            }
#endif
            ref_segment_progress += _WAYPOINT_CRUISE_SPEED * control_period_device;
            while (ref_segment_index < square_waypoint_count_device - 1) {
                if (segment_length > 1.0e-6f && ref_segment_progress <= segment_length) {
                    break;
                }
                ref_segment_progress = fmaxf(ref_segment_progress - segment_length, 0.0f);
                if (ref_segment_index >= square_waypoint_count_device - 2) {
                    ref_segment_progress = segment_length;
                    break;
                }
                ref_segment_index++;
                segment_x = square_waypoints_device[ref_segment_index + 1][0] -
                    square_waypoints_device[ref_segment_index][0];
                segment_y = square_waypoints_device[ref_segment_index + 1][1] -
                    square_waypoints_device[ref_segment_index][1];
                segment_z = square_waypoints_device[ref_segment_index + 1][2] -
                    square_waypoints_device[ref_segment_index][2];
                const float segment_length_squared =
                    segment_x * segment_x + segment_y * segment_y + segment_z * segment_z;
                segment_inv_length = segment_length_squared > 1.0e-12f
                    ? rsqrtf(segment_length_squared) : 0.0f;
                segment_length = segment_length_squared * segment_inv_length;
            }

            float cost_target_x = square_waypoints_device[ref_segment_index][0];
            float cost_target_y = square_waypoints_device[ref_segment_index][1];
            float cost_target_z = square_waypoints_device[ref_segment_index][2];
            float waypoint_velocity_ref[3] = {0.0f, 0.0f, 0.0f};
            if (segment_length > 1.0e-6f) {
                const float path_fraction = fminf(ref_segment_progress * segment_inv_length, 1.0f);
                cost_target_x += path_fraction * segment_x;
                cost_target_y += path_fraction * segment_y;
                cost_target_z += path_fraction * segment_z;
                const bool at_path_end =
                    ref_segment_index == square_waypoint_count_device - 2 && path_fraction >= 1.0f;
                if (!at_path_end) {
                    const float velocity_scale = _WAYPOINT_CRUISE_SPEED * segment_inv_length;
                    waypoint_velocity_ref[0] = segment_x * velocity_scale;
                    waypoint_velocity_ref[1] = segment_y * velocity_scale;
                    waypoint_velocity_ref[2] = segment_z * velocity_scale;
                }
            }

#ifdef PREDICTABLE_COLLISION_WITH_WALL
            // After an impact, every rollout immediately pursues the exit waypoint.
            if (predicted_post_collision_phase &&
                predicted_mission_index < square_waypoint_count_device) {
                post_collision_elapsed += control_period_device;
                if (post_collision_distance > 1.0e-6f) {
                    const float exit_progress = fminf(
                        _WAYPOINT_CRUISE_SPEED * post_collision_elapsed,
                        post_collision_distance);
                    const float exit_fraction = exit_progress * post_collision_inv_distance;
                    cost_target_x = post_collision_origin[0] + exit_fraction * post_collision_delta[0];
                    cost_target_y = post_collision_origin[1] + exit_fraction * post_collision_delta[1];
                    cost_target_z = post_collision_origin[2] + exit_fraction * post_collision_delta[2];
                    const float exit_speed = exit_progress < post_collision_distance
                        ? _WAYPOINT_CRUISE_SPEED : 0.0f;
                    waypoint_velocity_ref[0] = exit_speed * post_collision_delta[0] * post_collision_inv_distance;
                    waypoint_velocity_ref[1] = exit_speed * post_collision_delta[1] * post_collision_inv_distance;
                    waypoint_velocity_ref[2] = exit_speed * post_collision_delta[2] * post_collision_inv_distance;
                }
            }

            float collision_mission_cost = 0.0f;
            if (collision_event) {
                const int approach_index = max(predicted_mission_index - 1, 0);
                const float contact_y_error = collision_contact_point[1] -
                    square_waypoints_device[approach_index][1];
                const float contact_z_error = collision_contact_point[2] -
                    square_waypoints_device[approach_index][2];
                const float contact_x_error = collision_contact_point[0] - x_wall_device;
                const float contact_velocity_error =
                    pre_contact_normal_velocity + _CONTACT_APPROACH_SPEED;
                collision_mission_cost +=
                    _COST_CONTACT_POSITION * (
                        contact_x_error*contact_x_error +
                        contact_y_error*contact_y_error +
                        contact_z_error*contact_z_error) +
                    _COST_CONTACT_VELOCITY *
                        contact_velocity_error * contact_velocity_error;
            }
            if (predicted_post_collision_phase) {
                const float velocity_norm_squared =
                    var_and_z_i_temp[10]*var_and_z_i_temp[10] +
                    var_and_z_i_temp[11]*var_and_z_i_temp[11] +
                    var_and_z_i_temp[12]*var_and_z_i_temp[12];
                const float desired_norm_squared =
                    waypoint_velocity_ref[0]*waypoint_velocity_ref[0] +
                    waypoint_velocity_ref[1]*waypoint_velocity_ref[1] +
                    waypoint_velocity_ref[2]*waypoint_velocity_ref[2];
                if (velocity_norm_squared > 1.0e-10f && desired_norm_squared > 1.0e-10f) {
                    const float velocity_inv_norm = rsqrtf(velocity_norm_squared);
                    const float desired_inv_norm = rsqrtf(desired_norm_squared);
                    const float direction_error_x =
                        var_and_z_i_temp[10]*velocity_inv_norm - waypoint_velocity_ref[0]*desired_inv_norm;
                    const float direction_error_y =
                        var_and_z_i_temp[11]*velocity_inv_norm - waypoint_velocity_ref[1]*desired_inv_norm;
                    const float direction_error_z =
                        var_and_z_i_temp[12]*velocity_inv_norm - waypoint_velocity_ref[2]*desired_inv_norm;
                    collision_mission_cost += _COST_EXIT_DIRECTION * (
                        direction_error_x*direction_error_x +
                        direction_error_y*direction_error_y +
                        direction_error_z*direction_error_z);
                }
            }
#else
            const float collision_mission_cost = 0.0f;
#endif

            float input_delta_cost = 0.0f;
            if (i > 0) {
                float du_x = decoupled_position[i][x] - decoupled_position[i - 1][x];
                float du_y = decoupled_position[i][y] - decoupled_position[i - 1][y];
                float du_z = decoupled_position[i][z] - decoupled_position[i - 1][z];
                float du_yaw = decoupled_position[i][yaw] - decoupled_position[i - 1][yaw];
                input_delta_cost =
                    _COST_DU_X * du_x * du_x +
                    _COST_DU_Y * du_y * du_y +
                    _COST_DU_Z * du_z * du_z +
                    _COST_DU_YAW * du_yaw * du_yaw;
            }

            // コストの計算
            cost += (_COST_Q_X*(var_and_z_i_temp[7] -cost_target_x )*(var_and_z_i_temp[7] -cost_target_x) + _COST_Q_Y *(var_and_z_i_temp[8] -cost_target_y) *(var_and_z_i_temp[8] -cost_target_y) + _COST_Q_Z *(var_and_z_i_temp[9] -cost_target_z) *(var_and_z_i_temp[9] -cost_target_z)     // x, y, z
                 +  _COST_Q_XP*(var_and_z_i_temp[10]-waypoint_velocity_ref[0])*(var_and_z_i_temp[10]-waypoint_velocity_ref[0])+ _COST_Q_YP*(var_and_z_i_temp[11]-waypoint_velocity_ref[1])*(var_and_z_i_temp[11]-waypoint_velocity_ref[1])+ _COST_Q_ZP*(var_and_z_i_temp[12]-waypoint_velocity_ref[2])*(var_and_z_i_temp[12]-waypoint_velocity_ref[2])    // xp, yp, zp
                 +  _COST_Q_E1*(var_and_z_i_temp[1] -target_state_device.e1)*(var_and_z_i_temp[1] -target_state_device.e1)+ _COST_Q_E2*(var_and_z_i_temp[2] -target_state_device.e2)*(var_and_z_i_temp[2] -target_state_device.e2)+ _COST_Q_E3*(var_and_z_i_temp[3] -target_state_device.e3)*(var_and_z_i_temp[3] -target_state_device.e3)         // e1, e2, e3
                 +  _COST_Q_WX*(var_and_z_i_temp[4] -target_state_device.wx)*(var_and_z_i_temp[4] -target_state_device.wx)+ _COST_Q_WY*(var_and_z_i_temp[5] -target_state_device.wy)*(var_and_z_i_temp[5] -target_state_device.wy)+ _COST_Q_WZ*(var_and_z_i_temp[6] -target_state_device.wz)*(var_and_z_i_temp[6] -target_state_device.wz)          // wx, wy, wz
                 +  _COST_R_X*(decoupled_position[i][x]-cost_target_x)*(decoupled_position[i][x]-cost_target_x)
                 +  _COST_R_Y*(decoupled_position[i][y]-cost_target_y)*(decoupled_position[i][y]-cost_target_y)
                 +  _COST_R_Z*(decoupled_position[i][z]-cost_target_z)*(decoupled_position[i][z]-cost_target_z)
                 +  _COST_R_YAW*(decoupled_position[i][yaw])*(decoupled_position[i][yaw])
                 +  input_delta_cost
                 +  collision_mission_cost
            );

            if (i == _DEVICE_CONST_HORIZON - 1) {
                float terminal_pos_x = var_and_z_i_temp[7] - cost_target_x;
                float terminal_pos_y = var_and_z_i_temp[8] - cost_target_y;
                float terminal_pos_z = var_and_z_i_temp[9] - cost_target_z;
                float terminal_vel_x = var_and_z_i_temp[10] - waypoint_velocity_ref[0];
                float terminal_vel_y = var_and_z_i_temp[11] - waypoint_velocity_ref[1];
                float terminal_vel_z = var_and_z_i_temp[12] - waypoint_velocity_ref[2];
                cost +=
                    _COST_TERMINAL_X * terminal_pos_x * terminal_pos_x +
                    _COST_TERMINAL_Y * terminal_pos_y * terminal_pos_y +
                    _COST_TERMINAL_Z * terminal_pos_z * terminal_pos_z +
                    _COST_TERMINAL_VX * terminal_vel_x * terminal_vel_x +
                    _COST_TERMINAL_VY * terminal_vel_y * terminal_vel_y +
                    _COST_TERMINAL_VZ * terminal_vel_z * terminal_vel_z;
            }

            // prev_acc は加速度setpointではなく，速度微分LPF状態として次ステップへ引き継ぐ
            prev_acc[0] = vel_dot_lpf[0];
            prev_acc[1] = vel_dot_lpf[1];
            prev_acc[2] = vel_dot_lpf[2];
            prev_omega[0] = w_prev[0];
            prev_omega[1] = w_prev[1];
            prev_omega[2] = w_prev[2];
            prev_omega_dot[0] = omega_dot[0];
            prev_omega_dot[1] = omega_dot[1];
            prev_omega_dot[2] = omega_dot[2];

            vel_int[0] += vel_error_for_int[0] * mpc_xy_vel_i_acc * control_period_device;
            vel_int[1] += vel_error_for_int[1] * mpc_xy_vel_i_acc * control_period_device;
            vel_int[2] += vel_error_for_int[2] * mpc_z_vel_i_acc  * control_period_device;
            vel_int[2] = fminf(fmaxf(vel_int[2], -a_of_gravity_device), a_of_gravity_device);

            prev_vel[0] = var_and_z_i_temp[10];
            prev_vel[1] = var_and_z_i_temp[11];
            prev_vel[2] = var_and_z_i_temp[12];

            if (sample_id < 0) {
                for (int state_index = 0; state_index < _N_OF_ODES; state_index++) {
                    deterministic_sim_trajectory_device[i][state_index] =
                        var_and_z_i_temp[state_index];
                }
                for (int axis = 0; axis < 3; axis++) {
                    deterministic_sim_vel_int_device[i][axis] = vel_int[axis];
                    deterministic_sim_prev_acceleration_device[i][axis] = prev_acc[axis];
                    deterministic_sim_rate_int_device[i][axis] = rate_int[axis];
                    deterministic_sim_prev_angular_acceleration_device[i][axis] =
                        prev_omega_dot[axis];
                    deterministic_sim_vel_setpoint_device[i][axis] = vel_setpoint[axis];
                    deterministic_sim_acc_setpoint_device[i][axis] = acc_setpoint[axis];
                    deterministic_sim_thrust_setpoint_device[i][axis] = thrust_setpoint[axis];
                    deterministic_sim_omega_setpoint_device[i][axis] = omega_setpoint[axis];
                    deterministic_sim_torque_setpoint_device[i][axis] = torque_setpoint[axis];
                }
                for (int quat_index = 0; quat_index < 4; quat_index++) {
                    deterministic_sim_att_setpoint_device[i][quat_index] =
                        att_setpoint[quat_index];
                }
                for (int motor = 0; motor < 4; motor++) {
                    deterministic_sim_motor_speed_device[i][motor] = motor_speed_state[motor];
                    deterministic_sim_motor_setpoint_device[i][motor] = motor_setpoint[motor];
                }
            }
        }
        // 衝突に対して制約を与えたい場合はここに記述
#ifdef PREDICTABLE_COLLISION_WITH_WALL
        // if ( col_constraint_flag )  cost += 5;
#endif
    }

}
