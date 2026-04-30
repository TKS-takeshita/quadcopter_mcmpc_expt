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
            for ( int j = 0; j < 4; j++ )
                decoupled_position[i][j] = average_input_device.decoupled_position[i][j] + curand_normal( &state ) * sigma_k_device[j];
#endif
    }

    __device__ void input_array::do_simulation()
    {
        cost = 0.0f;

        // __constant__ メモリから状態量をコピー
        float var_and_z_i_temp[_N_OF_ODES + 1];
        for ( int i = 0; i < _N_OF_ODES + 1; i++ )
            var_and_z_i_temp[i] = var_and_z_i_device[i];
        
        float var_p_temp[_N_OF_ODES];

        // 衝突予測用変数
#ifdef PREDICTABLE_COLLISION_WITH_WALL
        bool  col_flag = false;
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
        // オイラー積分とコスト評価
        for ( int i = 0; i < _DEVICE_CONST_HORIZON; i++ )
        {
            // PX4のPIDを素に速度, 加速度, 姿勢, スラスト, 角速度を導出
            float x_ref   = decoupled_position[i][x];
            float y_ref   = decoupled_position[i][y];
            float z_ref   = decoupled_position[i][z];
            float yaw_ref = decoupled_position[i][yaw];
            float inv_mass = 1.0f / mass_of_machine_device;
            float vel_ref[3];
            float a_ref[3];
            for ( float t = 0.0f; t < control_period_device - integration_step_size_device / 2; t += integration_step_size_device )
            {              
                for ( int k = 0; k < _N_OF_ODES; k++ ) var_p_temp[k] = var_and_z_i_temp[k];
                /*目標速度*/
                
                vel_ref[0] = mpc_xy_p * (x_ref - var_p_temp[7]);
                vel_ref[1] = mpc_xy_p * (y_ref - var_p_temp[8]);
                vel_ref[2] = mpc_z_p  * (z_ref - var_p_temp[9]);
                /*目標加速度*/
                float vel_dot_x = (var_p_temp[10] - prev_vel[0]) / control_period_device;
                float vel_dot_y = (var_p_temp[11] - prev_vel[1]) / control_period_device;
                float vel_dot_z = (var_p_temp[12] - prev_vel[2]) / control_period_device;
                // 力を考慮する場合 a_ref = a_ref_pid - Fcut/m
                a_ref[0] = mpc_xy_vel_p_acc*(vel_ref[0]-var_p_temp[10])+mpc_xy_vel_i_acc*vel_int[0]-mpc_xy_vel_d_acc*(prev_acc[0]+lpf*(vel_dot_x-prev_acc[0]));
                a_ref[1] = mpc_xy_vel_p_acc*(vel_ref[1]-var_p_temp[11])+mpc_xy_vel_i_acc*vel_int[1]-mpc_xy_vel_d_acc*(prev_acc[1]+lpf*(vel_dot_y-prev_acc[1]));
                a_ref[2] = mpc_z_vel_p_acc* (vel_ref[2]-var_p_temp[12])+mpc_z_vel_i_acc* vel_int[2]-mpc_z_vel_d_acc *(prev_acc[2]+lpf*(vel_dot_z-prev_acc[2]));
                // PIDカスケード用修正
                
                // /* e0p */ var_and_z_i_temp[0]  += (-0.5f*var_p_temp[1]*var_p_temp[4] - 0.5f*var_p_temp[2]*var_p_temp[5] - 0.5f*var_p_temp[3]*var_p_temp[6])*integration_step_size_device;
                // /* e1p */ var_and_z_i_temp[1]  += ( 0.5f*var_p_temp[0]*var_p_temp[4] + 0.5f*var_p_temp[2]*var_p_temp[6] - 0.5f*var_p_temp[3]*var_p_temp[5])*integration_step_size_device;
                // /* e2p */ var_and_z_i_temp[2]  += ( 0.5f*var_p_temp[0]*var_p_temp[5] + 0.5f*var_p_temp[3]*var_p_temp[4] - 0.5f*var_p_temp[1]*var_p_temp[6])*integration_step_size_device;
                // /* e3p */ var_and_z_i_temp[3]  += ( 0.5f*var_p_temp[0]*var_p_temp[6] + 0.5f*var_p_temp[1]*var_p_temp[5] - 0.5f*var_p_temp[2]*var_p_temp[4])*integration_step_size_device;
                // // normalize quaternion
                // float q_norm_inv = rsqrtf(var_and_z_i_temp[0]*var_and_z_i_temp[0] +var_and_z_i_temp[1]*var_and_z_i_temp[1] +var_and_z_i_temp[2]*var_and_z_i_temp[2] +var_and_z_i_temp[3]*var_and_z_i_temp[3]);
                // var_and_z_i_temp[0] *= q_norm_inv;
                // var_and_z_i_temp[1] *= q_norm_inv;
                // var_and_z_i_temp[2] *= q_norm_inv;
                // var_and_z_i_temp[3] *= q_norm_inv;

                /* xp  */ var_and_z_i_temp[7]  +=  var_p_temp[10] * integration_step_size_device;
                /* yp  */ var_and_z_i_temp[8]  +=  var_p_temp[11] * integration_step_size_device;
                /* zp  */ var_and_z_i_temp[9]  +=  var_p_temp[12] * integration_step_size_device;

                /* xpp */ var_and_z_i_temp[10] += a_ref[0] * integration_step_size_device;
                /* ypp */ var_and_z_i_temp[11] += a_ref[1] * integration_step_size_device;
                /* zpp */ var_and_z_i_temp[12] += a_ref[2] * integration_step_size_device;

                /* z_i */ var_and_z_i_temp[_N_OF_ODES] += var_and_z_i_temp[9] * integration_step_size_device;

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
            }
            // コストの計算
            cost += (_COST_Q_X*(var_and_z_i_temp[7] -target_state_device.x )*(var_and_z_i_temp[7] -target_state_device.x) + _COST_Q_Y *(var_and_z_i_temp[8] -target_state_device.y) *(var_and_z_i_temp[8] -target_state_device.y) + _COST_Q_Z *(var_and_z_i_temp[9] -target_state_device.z) *(var_and_z_i_temp[9] -target_state_device.z)     // x, y, z
                 +  _COST_Q_XP*(var_and_z_i_temp[10]-target_state_device.xp)*(var_and_z_i_temp[10]-target_state_device.xp)+ _COST_Q_YP*(var_and_z_i_temp[11]-target_state_device.yp)*(var_and_z_i_temp[11]-target_state_device.yp)+ _COST_Q_ZP*(var_and_z_i_temp[12]-target_state_device.zp)*(var_and_z_i_temp[12]-target_state_device.zp)    // xp, yp, zp
                 +  _COST_Q_E1*(var_and_z_i_temp[1] -target_state_device.e1)*(var_and_z_i_temp[1] -target_state_device.e1)+ _COST_Q_E2*(var_and_z_i_temp[2] -target_state_device.e2)*(var_and_z_i_temp[2] -target_state_device.e2)+ _COST_Q_E3*(var_and_z_i_temp[3] -target_state_device.e3)*(var_and_z_i_temp[3] -target_state_device.e3)         // e1, e2, e3
                 +  _COST_Q_WX*(var_and_z_i_temp[4] -target_state_device.wx)*(var_and_z_i_temp[4] -target_state_device.wx)+ _COST_Q_WY*(var_and_z_i_temp[5] -target_state_device.wy)*(var_and_z_i_temp[5] -target_state_device.wy)+ _COST_Q_WZ*(var_and_z_i_temp[6] -target_state_device.wz)*(var_and_z_i_temp[6] -target_state_device.wz)          // wx, wy, wz
                 +  _COST_Q_ZI*var_and_z_i_temp[_N_OF_ODES]*var_and_z_i_temp[_N_OF_ODES]                                                                                                  // z_i
                 +  _COST_R_X*(decoupled_position[i][x]-target_state_device.x)*(decoupled_position[i][x]-target_state_device.x)
                 +  _COST_R_Y*(decoupled_position[i][y]-target_state_device.y)*(decoupled_position[i][y]-target_state_device.y)
                 +  _COST_R_Z*(decoupled_position[i][z]-target_state_device.z)*(decoupled_position[i][z]-target_state_device.z)
                 +  _COST_R_YAW*(decoupled_position[i][yaw])*(decoupled_position[i][yaw])
            );
            
            vel_int[0] += (vel_ref[0]-var_and_z_i_temp[10])*control_period_device;
            vel_int[1] += (vel_ref[1]-var_and_z_i_temp[11])*control_period_device;
            vel_int[2] += (vel_ref[2]-var_and_z_i_temp[12])*control_period_device;
            vel_int[2] = fminf(fmaxf(vel_int[2], -a_of_gravity_device), a_of_gravity_device);
        }
        // 衝突に対して制約を与えたい場合はここに記述
#ifdef PREDICTABLE_COLLISION_WITH_WALL
        // if ( col_constraint_flag )  cost += 5;
#endif
    }

}