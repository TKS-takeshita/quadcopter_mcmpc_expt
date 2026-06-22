#include "quadcopter_mcmpc_position/mcmpc_controller.cuh"
#include "quadcopter_mcmpc_position/const_params.hpp"

#include <cmath>

namespace qc_mcmpc
{
    __device__ static float dot_vec( float v1_x, float v1_y, float v1_z, float v2_x, float v2_y, float v2_z );
    __device__ static void cross_vec( float v1_x, float v1_y, float v1_z, float v2_x, float v2_y, float v2_z, float& v_ans_x, float& v_ans_y, float& v_ans_z );
    __device__ static void inverse_3x3( float matrix_src[3][3], float ans[3][3] );
    __host__ __device__ static float arctan2f(float y, float x);
    __host__ __device__ static void sin_cosf(float x, float* s, float* c);

#ifdef PREDICTABLE_COLLISION_WITH_WALL

    __device__ static void calculate_deviations( float var_and_z_i[], float var_p[], float input_x, float input_y, float input_z, float input_yaw, bool& col_flag, float v_plus[], float w_plus[] );
#else
    __device__ static void calculate_deviations( float var_and_z_i[], float var_p[], float input_x, float input_y, float input_z, float input_yaw );

#endif
    static float dot_vec_cpu(float v1_x, float v1_y, float v1_z, float v2_x, float v2_y, float v2_z){
		return v1_x * v2_x + v1_y * v2_y + v1_z * v2_z;
	}

	static void cross_vec_cpu(float v1_x, float v1_y, float v1_z, float v2_x, float v2_y, float v2_z, float& v_ans_x, float& v_ans_y, float& v_ans_z){
		v_ans_x = v1_y * v2_z - v1_z * v2_y;
		v_ans_y = v1_z * v2_x - v1_x * v2_z;
		v_ans_z = v1_x * v2_y - v1_y * v2_x;
	}

	static void inverse_3x3__cpu(float matrix_src[3][3], float ans[3][3]){
		float det  = matrix_src[0][0]*matrix_src[1][1]*matrix_src[2][2];
              det += matrix_src[1][0]*matrix_src[2][1]*matrix_src[0][2];
              det += matrix_src[2][0]*matrix_src[0][1]*matrix_src[1][2];
              det -= matrix_src[2][0]*matrix_src[1][1]*matrix_src[0][2];
              det -= matrix_src[1][0]*matrix_src[0][1]*matrix_src[2][2];
              det -= matrix_src[0][0]*matrix_src[2][1]*matrix_src[1][2];
        
        ans[0][0] =  ( matrix_src[1][1]*matrix_src[2][2] - matrix_src[1][2]*matrix_src[2][1] ) / det;
        ans[0][1] = -( matrix_src[1][0]*matrix_src[2][2] - matrix_src[1][2]*matrix_src[2][0] ) / det;
        ans[0][2] =  ( matrix_src[1][0]*matrix_src[2][1] - matrix_src[1][1]*matrix_src[2][0] ) / det;
        ans[1][0] = -( matrix_src[0][1]*matrix_src[2][2] - matrix_src[0][2]*matrix_src[2][1] ) / det;
        ans[1][1] =  ( matrix_src[0][0]*matrix_src[2][2] - matrix_src[0][2]*matrix_src[2][0] ) / det;
        ans[1][2] = -( matrix_src[0][0]*matrix_src[2][1] - matrix_src[0][1]*matrix_src[2][0] ) / det;
        ans[2][0] =  ( matrix_src[0][1]*matrix_src[1][2] - matrix_src[0][2]*matrix_src[1][1] ) / det;
        ans[2][1] = -( matrix_src[0][0]*matrix_src[1][2] - matrix_src[0][2]*matrix_src[1][0] ) / det;
        ans[2][2] =  ( matrix_src[0][0]*matrix_src[1][1] - matrix_src[0][1]*matrix_src[1][0] ) / det;
	}

    __host__ __device__ static float arctan2f(float y, float x){
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
    __host__ __device__ static void sin_cosf(float x, float* s, float* c){
            // wrap [-pi, pi]
        if (x > M_PI) x -= 2.0f * M_PI;
        if (x < -M_PI) x += 2.0f * M_PI;
        float x2 = x * x;
        // 5次近似
        *s = x * (1.0f - x2 / 6.0f + x2 * x2 / 120.0f);
        *c = 1.0f - x2 / 2.0f + x2 * x2 / 24.0f;
    }

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
        float var_and_z_i_temp[_N_OF_ODES + 1];
        for ( int i = 0; i < _N_OF_ODES + 1; i++ )
            var_and_z_i_temp[i] = var_and_z_i_device[i];
        
        // 衝突予測用変数
#ifdef PREDICTABLE_COLLISION_WITH_WALL
        bool col_flag = false;
        float v_plus[3], w_plus[3];
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
        float vel_setpoint[3];
        float acc_setpoint[3];
        float att_setpoint[4];
        float thrust_setpoint[3];
        float omega_setpoint[3];
        const int rate_delay_steps = 3;
        const int acc_delay_steps = 3;
        float rate_delay_buffer[rate_delay_steps][3];
        float acc_delay_buffer[acc_delay_steps][3];
        for (int d = 0; d < rate_delay_steps; d++) {
            rate_delay_buffer[d][0] = var_and_z_i_temp[4];
            rate_delay_buffer[d][1] = var_and_z_i_temp[5];
            rate_delay_buffer[d][2] = var_and_z_i_temp[6];
        }
        for (int d = 0; d < acc_delay_steps; d++) {
            acc_delay_buffer[d][0] = 0.0f;
            acc_delay_buffer[d][1] = 0.0f;
            acc_delay_buffer[d][2] = 0.0f;
        }

        int pred_wp_index = square_waypoint_index_device;
        if (pred_wp_index < 1) {
            pred_wp_index = 1;
        }
        if (pred_wp_index >= _SQUARE_WAYPOINTS) {
            pred_wp_index = _SQUARE_WAYPOINTS - 1;
        }
        float pred_wp_change_time = square_waypoint_change_time_device;
        float pred_target_x = target_state_device.x;
        float pred_target_y = target_state_device.y;
        float pred_target_z = target_state_device.z;
        for (int wp = pred_wp_index; wp < _SQUARE_WAYPOINTS; wp++) {
            if (fabsf(square_waypoints_device[wp][0] - pred_target_x) < 1.0e-4f &&
                fabsf(square_waypoints_device[wp][1] - pred_target_y) < 1.0e-4f) {
                pred_wp_index = wp;
                break;
            }
        }

        // 内部で位置/ヨー入力からsetpointを計算し，同定済み離散モデルで状態遷移する
        for ( int i = 0; i < _DEVICE_CONST_HORIZON; i++ )
        {
            // PX4のPIDを素に速度, 加速度, 姿勢, スラスト, 角速度を導出
            float x_ref   = decoupled_position[i][x];
            float y_ref   = decoupled_position[i][y];
            float z_ref   = decoupled_position[i][z];
            float yaw_ref = decoupled_position[i][yaw];
            bool flying = takeoff_state_device >= 5;
            bool flying_but_ground_contact = flying && (ground_contact_device || maybe_landed_device);
            bool no_thrust = (takeoff_state_device < 4) || flying_but_ground_contact;
            float thrust_min = flying ? 0.1f : 0.0f;
            float tilt_limit = takeoff_tilt_limit_device;
            if (!isfinite(tilt_limit) || tilt_limit <= 0.0f) {
                tilt_limit = 0.78539816339f;
            }
            float hover_thrust = mpc_thr_hover;
            if (!isfinite(hover_thrust) || hover_thrust < 1.0e-6f) {
                hover_thrust = 0.60f;
            }

            vel_setpoint[0] = mpc_xy_p * (x_ref - var_and_z_i_temp[7]);
            vel_setpoint[1] = mpc_xy_p * (y_ref - var_and_z_i_temp[8]);
            vel_setpoint[2] = mpc_z_p  * (z_ref - var_and_z_i_temp[9]);

            const float mpc_xy_vel_max = 12.0f;
            const float mpc_z_vel_max_up = 3.0f;
            const float mpc_z_vel_max_down = 1.0f;
            float vel_xy_norm = sqrtf(vel_setpoint[0]*vel_setpoint[0] + vel_setpoint[1]*vel_setpoint[1]);
            if (vel_xy_norm > mpc_xy_vel_max && vel_xy_norm > 1.0e-8f) {
                vel_setpoint[0] = vel_setpoint[0] / vel_xy_norm * mpc_xy_vel_max;
                vel_setpoint[1] = vel_setpoint[1] / vel_xy_norm * mpc_xy_vel_max;
            }
            vel_setpoint[2] = fminf(fmaxf(vel_setpoint[2], -mpc_z_vel_max_up), mpc_z_vel_max_down);

            float vel_error[3];
            vel_error[0] = vel_setpoint[0] - var_and_z_i_temp[10];
            vel_error[1] = vel_setpoint[1] - var_and_z_i_temp[11];
            vel_error[2] = vel_setpoint[2] - var_and_z_i_temp[12];

            float vel_dot_x = (var_and_z_i_temp[10] - prev_vel[0]) / control_period_device;
            float vel_dot_y = (var_and_z_i_temp[11] - prev_vel[1]) / control_period_device;
            float vel_dot_z = (var_and_z_i_temp[12] - prev_vel[2]) / control_period_device;

            float vel_dot_alpha;
            if (mpc_veld_lp > 1.0e-6f) {
                vel_dot_alpha = control_period_device / (control_period_device + 1.0f / (2.0f * M_PI * mpc_veld_lp));
            } else {
                vel_dot_alpha = 1.0f;
            }

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

            float dot_z = fminf(fmaxf(body_z[2], -1.0f), 1.0f);
            float tilt_angle = acosf(dot_z);
            if (tilt_angle > tilt_limit) {
                float rejection[3];
                rejection[0] = body_z[0];
                rejection[1] = body_z[1];
                rejection[2] = 0.0f;
                float rejection_norm = sqrtf(rejection[0]*rejection[0] + rejection[1]*rejection[1]);
                if (rejection_norm < 1.0e-8f) {
                    rejection[0] = 1.0f;
                    rejection[1] = 0.0f;
                    rejection_norm = 1.0f;
                }
                rejection[0] /= rejection_norm;
                rejection[1] /= rejection_norm;
                body_z[0] = sinf(tilt_limit) * rejection[0];
                body_z[1] = sinf(tilt_limit) * rejection[1];
                body_z[2] = cosf(tilt_limit);
            }

            if (no_thrust) {
                acc_setpoint[0] = 0.0f;
                acc_setpoint[1] = 0.0f;
                acc_setpoint[2] = 100.0f;
            }

            float thrust_ned_z = acc_setpoint[2] * (hover_thrust / a_of_gravity_device) - hover_thrust;
            float cos_ned_body = body_z[2];
            if (fabsf(cos_ned_body) < 1.0e-6f) cos_ned_body = 1.0e-6f;

            const float mpc_thr_max = 0.9f;
            const float mpc_thr_xy_margin = 0.3f;

            float collective_thrust = fminf(thrust_ned_z / cos_ned_body, -thrust_min);
            thrust_setpoint[0] = body_z[0] * collective_thrust;
            thrust_setpoint[1] = body_z[1] * collective_thrust;
            thrust_setpoint[2] = body_z[2] * collective_thrust;
            if (no_thrust) {
                thrust_setpoint[0] = 0.0f;
                thrust_setpoint[1] = 0.0f;
                thrust_setpoint[2] = 0.0f;
            }

            // mcmpc_viewer と同じ thrust saturation
            float thrust_sp_xy_norm = sqrtf(thrust_setpoint[0]*thrust_setpoint[0] + thrust_setpoint[1]*thrust_setpoint[1]);
            float thrust_max_squared = mpc_thr_max * mpc_thr_max;
            float allocated_horizontal_thrust = fminf(thrust_sp_xy_norm, mpc_thr_xy_margin);
            float thrust_z_max_squared = thrust_max_squared - allocated_horizontal_thrust * allocated_horizontal_thrust;
            thrust_setpoint[2] = fmaxf(thrust_setpoint[2], -sqrtf(fmaxf(0.0f, thrust_z_max_squared)));

            float thrust_max_xy_squared = thrust_max_squared - thrust_setpoint[2] * thrust_setpoint[2];
            float thrust_max_xy = sqrtf(fmaxf(0.0f, thrust_max_xy_squared));
            if (thrust_sp_xy_norm > thrust_max_xy && thrust_sp_xy_norm > 1.0e-8f) {
                thrust_setpoint[0] = thrust_setpoint[0] / thrust_sp_xy_norm * thrust_max_xy;
                thrust_setpoint[1] = thrust_setpoint[1] / thrust_sp_xy_norm * thrust_max_xy;
            }

            float acc_sp_xy_produced[2];
            acc_sp_xy_produced[0] = thrust_setpoint[0] * (a_of_gravity_device / hover_thrust);
            acc_sp_xy_produced[1] = thrust_setpoint[1] * (a_of_gravity_device / hover_thrust);

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
                float xy_arw_gain = 2.0f / mpc_xy_vel_p_acc;
                vel_error_for_int[0] -= xy_arw_gain * (acc_setpoint[0] - acc_sp_xy_produced[0]);
                vel_error_for_int[1] -= xy_arw_gain * (acc_setpoint[1] - acc_sp_xy_produced[1]);
            }

            float att_body_z[3];
            att_body_z[0] = -thrust_setpoint[0];
            att_body_z[1] = -thrust_setpoint[1];
            att_body_z[2] = -thrust_setpoint[2];
            float att_bz_norm = sqrtf(att_body_z[0]*att_body_z[0]+att_body_z[1]*att_body_z[1]+att_body_z[2]*att_body_z[2]);
            if (att_bz_norm < 1.0e-8f) {
                att_body_z[0] = 0.0f;
                att_body_z[1] = 0.0f;
                att_body_z[2] = 1.0f;
                att_bz_norm = 1.0f;
            }
            att_body_z[0] /= att_bz_norm;
            att_body_z[1] /= att_bz_norm;
            att_body_z[2] /= att_bz_norm;

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
                att_setpoint[0] = 0.25f * s;
                att_setpoint[1] = (r21 - r12) / s;
                att_setpoint[2] = (r02 - r20) / s;
                att_setpoint[3] = (r10 - r01) / s;
            } else if (r00 > r11 && r00 > r22) {
                float s = sqrtf(1.0f + r00 - r11 - r22) * 2.0f;
                att_setpoint[0] = (r21 - r12) / s;
                att_setpoint[1] = 0.25f * s;
                att_setpoint[2] = (r01 + r10) / s;
                att_setpoint[3] = (r02 + r20) / s;
            } else if (r11 > r22) {
                float s = sqrtf(1.0f + r11 - r00 - r22) * 2.0f;
                att_setpoint[0] = (r02 - r20) / s;
                att_setpoint[1] = (r01 + r10) / s;
                att_setpoint[2] = 0.25f * s;
                att_setpoint[3] = (r12 + r21) / s;
            } else {
                float s = sqrtf(1.0f + r22 - r00 - r11) * 2.0f;
                att_setpoint[0] = (r10 - r01) / s;
                att_setpoint[1] = (r02 + r20) / s;
                att_setpoint[2] = (r12 + r21) / s;
                att_setpoint[3] = 0.25f * s;
            }
            float qsp_norm_inv = rsqrtf(att_setpoint[0]*att_setpoint[0] + att_setpoint[1]*att_setpoint[1] + att_setpoint[2]*att_setpoint[2] + att_setpoint[3]*att_setpoint[3]);
            att_setpoint[0] *= qsp_norm_inv;
            att_setpoint[1] *= qsp_norm_inv;
            att_setpoint[2] *= qsp_norm_inv;
            att_setpoint[3] *= qsp_norm_inv;

            float qe0 =  var_and_z_i_temp[0]*att_setpoint[0] + var_and_z_i_temp[1]*att_setpoint[1] + var_and_z_i_temp[2]*att_setpoint[2] + var_and_z_i_temp[3]*att_setpoint[3];
            float sgn = (qe0 >= 0.0f) ? 1.0f : -1.0f;

            omega_setpoint[0]  = 2.0f*mc_roll_p *sgn * (var_and_z_i_temp[0]*att_setpoint[1]-var_and_z_i_temp[1]*att_setpoint[0]-var_and_z_i_temp[2]*att_setpoint[3]+var_and_z_i_temp[3]*att_setpoint[2]);
            omega_setpoint[1]  = 2.0f*mc_pitch_p*sgn * (var_and_z_i_temp[0]*att_setpoint[2]+var_and_z_i_temp[1]*att_setpoint[3]-var_and_z_i_temp[2]*att_setpoint[0]-var_and_z_i_temp[3]*att_setpoint[1]);
            omega_setpoint[2]  = 2.0f*mc_yaw_p  *sgn * (var_and_z_i_temp[0]*att_setpoint[3]-var_and_z_i_temp[1]*att_setpoint[2]+var_and_z_i_temp[2]*att_setpoint[1]-var_and_z_i_temp[3]*att_setpoint[0]);

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
            float acc_for_dynamics[3] = {
                acc_sp_xy_produced[0],
                acc_sp_xy_produced[1],
                acc_setpoint[2],
            };
            float omega_for_dynamics[3] = {
                omega_setpoint[0],
                omega_setpoint[1],
                omega_setpoint[2],
            };
            if (i == 0) {
                for (int d = 0; d < acc_delay_steps; d++) {
                    acc_delay_buffer[d][0] = acc_for_dynamics[0];
                    acc_delay_buffer[d][1] = acc_for_dynamics[1];
                    acc_delay_buffer[d][2] = acc_for_dynamics[2];
                }
            }
            for (int axis = 0; axis < 3; axis++) {
                acc_for_dynamics[axis] = acc_delay_buffer[0][axis];
                omega_for_dynamics[axis] = rate_delay_buffer[0][axis];
            }
            for (int d = 0; d < acc_delay_steps - 1; d++) {
                acc_delay_buffer[d][0] = acc_delay_buffer[d + 1][0];
                acc_delay_buffer[d][1] = acc_delay_buffer[d + 1][1];
                acc_delay_buffer[d][2] = acc_delay_buffer[d + 1][2];
            }
            acc_delay_buffer[acc_delay_steps - 1][0] = acc_sp_xy_produced[0];
            acc_delay_buffer[acc_delay_steps - 1][1] = acc_sp_xy_produced[1];
            acc_delay_buffer[acc_delay_steps - 1][2] = acc_setpoint[2];
            for (int d = 0; d < rate_delay_steps - 1; d++) {
                rate_delay_buffer[d][0] = rate_delay_buffer[d + 1][0];
                rate_delay_buffer[d][1] = rate_delay_buffer[d + 1][1];
                rate_delay_buffer[d][2] = rate_delay_buffer[d + 1][2];
            }
            rate_delay_buffer[rate_delay_steps - 1][0] = omega_setpoint[0];
            rate_delay_buffer[rate_delay_steps - 1][1] = omega_setpoint[1];
            rate_delay_buffer[rate_delay_steps - 1][2] = omega_setpoint[2];

            var_and_z_i_temp[7] += v_prev[0] * control_period_device;
            var_and_z_i_temp[8] += v_prev[1] * control_period_device;
            var_and_z_i_temp[9] += v_prev[2] * control_period_device;

            var_and_z_i_temp[10] = v_prev[0] + acc_for_dynamics[0] * control_period_device;
            var_and_z_i_temp[11] = v_prev[1] + acc_for_dynamics[1] * control_period_device;
            var_and_z_i_temp[12] = v_prev[2] + acc_for_dynamics[2] * control_period_device;

            float q_dot[4];
            q_dot[0] = -0.5f * (q_prev[1] * w_prev[0] + q_prev[2] * w_prev[1] + q_prev[3] * w_prev[2]);
            q_dot[1] =  0.5f * (q_prev[0] * w_prev[0] + q_prev[2] * w_prev[2] - q_prev[3] * w_prev[1]);
            q_dot[2] =  0.5f * (q_prev[0] * w_prev[1] - q_prev[1] * w_prev[2] + q_prev[3] * w_prev[0]);
            q_dot[3] =  0.5f * (q_prev[0] * w_prev[2] + q_prev[1] * w_prev[1] - q_prev[2] * w_prev[0]);

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

            var_and_z_i_temp[4] = omega_for_dynamics[0];
            var_and_z_i_temp[5] = omega_for_dynamics[1];
            var_and_z_i_temp[6] = omega_for_dynamics[2];

            var_and_z_i_temp[_N_OF_ODES] += var_and_z_i_temp[9] * control_period_device;

            // 衝突予測がONのとき，速度と角速度を上書きする
#ifdef PREDICTABLE_COLLISION_WITH_WALL
            if ( col_flag )
            {
                col_constraint_flag = true;

                var_and_z_i_temp[10] = v_plus[0];
                var_and_z_i_temp[11] = v_plus[1];
                var_and_z_i_temp[12] = v_plus[2];

                // 角速度は機体座標系に変換してから代入
                var_and_z_i_temp[4] = 2.0f*w_plus[1]*(var_and_z_i_temp[0]*var_and_z_i_temp[3]+var_and_z_i_temp[1]*var_and_z_i_temp[2]) + w_plus[0]*(-1.0f+2.0f*var_and_z_i_temp[0]*var_and_z_i_temp[0]+2.0f*var_and_z_i_temp[1]*var_and_z_i_temp[1]) - 2.0f*w_plus[2]*(var_and_z_i_temp[0]*var_and_z_i_temp[2]-var_and_z_i_temp[1]*var_and_z_i_temp[3]);
                var_and_z_i_temp[5] = 2.0f*w_plus[2]*(var_and_z_i_temp[0]*var_and_z_i_temp[1]+var_and_z_i_temp[2]*var_and_z_i_temp[3]) + w_plus[1]*(-1.0f+2.0f*var_and_z_i_temp[0]*var_and_z_i_temp[0]+2.0f*var_and_z_i_temp[2]*var_and_z_i_temp[2]) - 2.0f*w_plus[0]*(var_and_z_i_temp[0]*var_and_z_i_temp[3]-var_and_z_i_temp[1]*var_and_z_i_temp[2]);
                var_and_z_i_temp[6] = 2.0f*w_plus[0]*(var_and_z_i_temp[0]*var_and_z_i_temp[2]+var_and_z_i_temp[1]*var_and_z_i_temp[3]) + w_plus[2]*(-1.0f+2.0f*var_and_z_i_temp[0]*var_and_z_i_temp[0]+2.0f*var_and_z_i_temp[3]*var_and_z_i_temp[3]) - 2.0f*w_plus[1]*(var_and_z_i_temp[0]*var_and_z_i_temp[1]-var_and_z_i_temp[2]*var_and_z_i_temp[3]);
            }
#endif
            float pred_time = mcmpc_log_device + (i + 1) * control_period_device;
            float dx = pred_target_x - var_and_z_i_temp[7];
            float dy = pred_target_y - var_and_z_i_temp[8];
            float waypoint_error = sqrtf(dx * dx + dy * dy);

            if ((pred_time - pred_wp_change_time) >= _SQUARE_WAYPOINT_HOLD_SEC &&
                waypoint_error < _SQUARE_WAYPOINT_THRESHOLD &&
                pred_wp_index + 1 < _SQUARE_WAYPOINTS) {
                pred_wp_index++;
                pred_wp_change_time = pred_time;

                pred_target_x = square_waypoints_device[pred_wp_index][0];
                pred_target_y = square_waypoints_device[pred_wp_index][1];
                pred_target_z = square_waypoints_device[pred_wp_index][2];
                if (sample_id == 0) {
                    printf(
                        "[mcmpc gpu] pred target advanced: h=%d t=%.3f wp=%d target=(%.3f, %.3f, %.3f) pos=(%.3f, %.3f, %.3f) err=%.3f\n",
                        i + 1,
                        pred_time,
                        pred_wp_index,
                        pred_target_x,
                        pred_target_y,
                        pred_target_z,
                        var_and_z_i_temp[7],
                        var_and_z_i_temp[8],
                        var_and_z_i_temp[9],
                        waypoint_error
                    );
                }
            }

            // コストの計算
            cost += (_COST_Q_X*(var_and_z_i_temp[7] -pred_target_x )*(var_and_z_i_temp[7] -pred_target_x) + _COST_Q_Y *(var_and_z_i_temp[8] -pred_target_y) *(var_and_z_i_temp[8] -pred_target_y) + _COST_Q_Z *(var_and_z_i_temp[9] -pred_target_z) *(var_and_z_i_temp[9] -pred_target_z)     // x, y, z
                 +  _COST_Q_XP*(var_and_z_i_temp[10]-target_state_device.xp)*(var_and_z_i_temp[10]-target_state_device.xp)+ _COST_Q_YP*(var_and_z_i_temp[11]-target_state_device.yp)*(var_and_z_i_temp[11]-target_state_device.yp)+ _COST_Q_ZP*(var_and_z_i_temp[12]-target_state_device.zp)*(var_and_z_i_temp[12]-target_state_device.zp)    // xp, yp, zp
                 +  _COST_Q_E1*(var_and_z_i_temp[1] -target_state_device.e1)*(var_and_z_i_temp[1] -target_state_device.e1)+ _COST_Q_E2*(var_and_z_i_temp[2] -target_state_device.e2)*(var_and_z_i_temp[2] -target_state_device.e2)+ _COST_Q_E3*(var_and_z_i_temp[3] -target_state_device.e3)*(var_and_z_i_temp[3] -target_state_device.e3)         // e1, e2, e3
                 +  _COST_Q_WX*(var_and_z_i_temp[4] -target_state_device.wx)*(var_and_z_i_temp[4] -target_state_device.wx)+ _COST_Q_WY*(var_and_z_i_temp[5] -target_state_device.wy)*(var_and_z_i_temp[5] -target_state_device.wy)+ _COST_Q_WZ*(var_and_z_i_temp[6] -target_state_device.wz)*(var_and_z_i_temp[6] -target_state_device.wz)          // wx, wy, wz
                 +  _COST_Q_ZI*var_and_z_i_temp[_N_OF_ODES]*var_and_z_i_temp[_N_OF_ODES]                                                                                                  // z_i
                 +  _COST_R_X*(decoupled_position[i][x]-pred_target_x)*(decoupled_position[i][x]-pred_target_x)
                 +  _COST_R_Y*(decoupled_position[i][y]-pred_target_y)*(decoupled_position[i][y]-pred_target_y)
                 +  _COST_R_Z*(decoupled_position[i][z]-pred_target_z)*(decoupled_position[i][z]-pred_target_z)
                 +  _COST_R_YAW*(decoupled_position[i][yaw])*(decoupled_position[i][yaw])
            );

            // prev_acc は加速度setpointではなく，速度微分LPF状態として次ステップへ引き継ぐ
            prev_acc[0] = vel_dot_lpf[0];
            prev_acc[1] = vel_dot_lpf[1];
            prev_acc[2] = vel_dot_lpf[2];

            vel_int[0] += vel_error_for_int[0] * mpc_xy_vel_i_acc * control_period_device;
            vel_int[1] += vel_error_for_int[1] * mpc_xy_vel_i_acc * control_period_device;
            vel_int[2] += vel_error_for_int[2] * mpc_z_vel_i_acc  * control_period_device;
            vel_int[2] = fminf(fmaxf(vel_int[2], -a_of_gravity_device), a_of_gravity_device);

            prev_vel[0] = var_and_z_i_temp[10];
            prev_vel[1] = var_and_z_i_temp[11];
            prev_vel[2] = var_and_z_i_temp[12];
        }
        // 衝突に対して制約を与えたい場合はここに記述
#ifdef PREDICTABLE_COLLISION_WITH_WALL
        // if ( col_constraint_flag )  cost += 5;
#endif
    }

}
