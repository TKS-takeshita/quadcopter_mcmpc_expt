#include <cmath>
#include <iostream>

#include "mpc_simulator.cu"

namespace qc_mcmpc
{
    // 実体を定義（externを外す）
    __constant__ target_state_t target_state_device;

    __constant__ float u_g_device;
	__constant__ float u_upper_lim_device;
	__constant__ float u_lower_lim_device;

	__constant__ float i_xx_device;
	__constant__ float i_yy_device;
	__constant__ float i_zz_device;
	__constant__ float rotor_distance_device;
	__constant__ float max_rps_pow_device;
	__constant__ float max_thrust_device;
	__constant__ float mass_of_machine_device;
	__constant__ float torque_rate_device;
	__constant__ float a_of_gravity_device;

    __constant__ float mpc_xy_p;
    __constant__ float mpc_z_p;
    __constant__ float mpc_xy_vel_p_acc;
    __constant__ float mpc_z_vel_p_acc;
    __constant__ float mpc_xy_vel_i_acc;
    __constant__ float mpc_z_vel_i_acc;
    __constant__ float mpc_xy_vel_d_acc;
    __constant__ float mpc_z_vel_d_acc;
    __constant__ float mpc_xy_vel_max;
    __constant__ float mpc_z_vel_max_up;
    __constant__ float mpc_z_vel_max_down;
    __constant__ float mpc_thr_min;
    __constant__ float mpc_thr_max;
    __constant__ float mpc_thr_xy_margin;
    __constant__ float arw_gain;
    __constant__ float mc_roll_p;
    __constant__ float mc_pitch_p;
    __constant__ float mc_yaw_p;
    __constant__ float mc_yaw_weight;
    __constant__ float lpf;
    __constant__ float mpc_thr_hover;
    __constant__ float mpc_veld_lp;
    __constant__ float mc_rollrate_p;
    __constant__ float mc_pitchrate_p;
    __constant__ float mc_yawrate_p;
    __constant__ float mc_rollrate_d;
    __constant__ float mc_pitchrate_d;
    __constant__ float mc_yawrate_d;
    __constant__ float mc_rollrate_i;
    __constant__ float mc_pitchrate_i;
    __constant__ float mc_yawrate_i;

    __constant__ float prev_velocity_device[3];
    __constant__ float vel_int_device[3];
    __constant__ float prev_acceleration_device[3];
    __constant__ float prev_angular_velocity_device[3];
    __constant__ float prev_angular_acceleration_device[3];
    __constant__ float rate_int_device[3];
    __constant__ float prev_motor_speed_device[4];
    __constant__ int   prev_motor_speed_valid_device;

    __constant__ float motor_input_scaling;
    __constant__ float max_rot_velocity;
    __constant__ float motor_time_constant_up;
    __constant__ float motor_time_constant_down;
    __constant__ float motor_thrust_constant;
    __constant__ float moment_constant;

    __constant__ float rotor_positions[4][3];
    __constant__ float rotor_yaw_signs[4];
    __constant__ float px4_quad_x_mix[4][4];
    __constant__ float px4_quad_x_mix_inv[4][4];
    __constant__ float px4_actuator_min[4];
    __constant__ float px4_actuator_max[4];

    __constant__ float angular_accel_lp;
    __constant__ float mc_rollrate_k;
    __constant__ float mc_pitchrate_k;
    __constant__ float mc_yawrate_k;
    __constant__ float mc_rollrate_ff;
    __constant__ float mc_pitchrate_ff;
    __constant__ float mc_yawrate_ff;
    __constant__ float mc_rr_int_lim;
    __constant__ float mc_pr_int_lim;
    __constant__ float mc_yr_int_lim;
    __constant__ float mc_yaw_tq_cutoff;

    __constant__ float ca_minimum_yaw_margin;
    __constant__ float acceleration_bias_device[3];
    __constant__ int   motor_command_delay_steps;

    __constant__ int takeoff_state_rampup_device;
    __constant__ int takeoff_state_flight_device;

    target_state_t target_host;

#ifdef PREDICTABLE_COLLISION_WITH_WALL
	__constant__ float x_wall_device;
	__constant__ float wall_nv_x_device;
	__constant__ float wall_nv_y_device;
	__constant__ float wall_nv_z_device;
	__constant__ float r_of_ring_device;
	__constant__ float coeff_of_rest_device;
#endif

	__constant__ float control_period_device;
	__constant__ float integration_step_size_device;

    __constant__ float var_and_z_i_device[_N_OF_ODES + 1];
    __constant__ input_array average_input_device;
    __constant__ float sigma_k_device[4];
    __constant__ int square_waypoint_index_device;
    __constant__ float square_waypoint_change_time_device;
    __constant__ float mcmpc_log_device;
    __constant__ float square_waypoints_device[_SQUARE_WAYPOINTS][3];
    __constant__ int takeoff_state_device;
    __constant__ float takeoff_tilt_limit_sin_device;
    __constant__ float takeoff_tilt_limit_cos_device;
    __constant__ int landed_device;
    __constant__ int ground_contact_device;
    __constant__ int maybe_landed_device;

    __global__ static void init_curand_seed(curandState *state_array, int seed);
	__global__ static void generate_input_samples_and_calc_costs(curandState *state, input_array* input_array_sample_device, float* cost_vec);
    __global__ static void select_elite_indices_and_lambda(
        const float* cost_vec,
        int* elite_indices,
        int* selected_flags,
        float* lambda_out,
        int n_samples,
        int n_elite);
    __global__ static void calc_weighted_average_input_on_device(
        const input_array* input_samples,
        const int* elite_indices,
        const float* lambda_in,
        input_array* best_input,
        int n_elite);

	// コスト再計算用関数
	static void input_constraint_cpu(float& rps_z, float& rps_wx, float& rps_wy, float& rps_ws);

    void update_target_state_device();

#ifdef PREDICTABLE_COLLISION_WITH_WALL
	static float dot_vec_cpu(float v1_x, float v1_y, float v1_z, float v2_x, float v2_y, float v2_z);
	static void cross_vec_cpu(float v1_x, float v1_y, float v1_z, float v2_x, float v2_y, float v2_z, float& v_ans_x, float& v_ans_y, float& v_ans_z);
	static void inverse_3x3__cpu(float matrix_src[3][3], float ans[3][3]);
	static void calculate_deviations_cpu(float var_and_z_i[], float var_p[], float rps_cw1, float rps_cw2, float rps_ccw1, float rps_ccw2, bool& col_flag, float v_plus[], float w_plus[]);
#else
	static void calculate_deviations_cpu(float var_and_z_i[], float var_p[], float rps_cw1, float rps_cw2, float rps_ccw1, float rps_ccw2);
#endif

// コンストラクタ
	mcmpc_controller::mcmpc_controller()
	{
		// 目標状態の設定

        target_host = {
            CONST_PARAM_FLOAT::INIT_TARGET_E0,
            CONST_PARAM_FLOAT::INIT_TARGET_E1,
            CONST_PARAM_FLOAT::INIT_TARGET_E2,
            CONST_PARAM_FLOAT::INIT_TARGET_E3,
            CONST_PARAM_FLOAT::INIT_TARGET_WX,
            CONST_PARAM_FLOAT::INIT_TARGET_WY,
            CONST_PARAM_FLOAT::INIT_TARGET_WZ,
            CONST_PARAM_FLOAT::INIT_TARGET_X,
            CONST_PARAM_FLOAT::INIT_TARGET_Y,
            CONST_PARAM_FLOAT::INIT_TARGET_Z,
            CONST_PARAM_FLOAT::INIT_TARGET_XP,
            CONST_PARAM_FLOAT::INIT_TARGET_YP,
            CONST_PARAM_FLOAT::INIT_TARGET_ZP
        };

		// 分散の設定　（変数なのは分散固定化が暫定的措置であるため）
		for(int i=0; i<4; i++)
			sigma_k[i] = CONST_PARAM_FLOAT::SIGMA_CONST[i];

        // PIDカスケード用修正
		for(int i = 0; i < _DEVICE_CONST_HORIZON; i++){
			best_input_array.decoupled_position[i][x] = CONST_PARAM_FLOAT::INIT_TARGET_X;
            best_input_array.decoupled_position[i][y] = CONST_PARAM_FLOAT::INIT_TARGET_Y;
            best_input_array.decoupled_position[i][z] = CONST_PARAM_FLOAT::INIT_TARGET_Z;
            best_input_array.decoupled_position[i][yaw] = atan2f(
                                                            2.0f * (CONST_PARAM_FLOAT::INIT_TARGET_E0 * CONST_PARAM_FLOAT::INIT_TARGET_E3
                                                                + CONST_PARAM_FLOAT::INIT_TARGET_E1 * CONST_PARAM_FLOAT::INIT_TARGET_E2),
                                                            1.0f - 2.0f * (CONST_PARAM_FLOAT::INIT_TARGET_E2 * CONST_PARAM_FLOAT::INIT_TARGET_E2
                                                                        + CONST_PARAM_FLOAT::INIT_TARGET_E3 * CONST_PARAM_FLOAT::INIT_TARGET_E3)
                                                        );
        }
		// curandの乱数シード設定
		cudaMalloc(&curand_state_array, CONST_PARAM::N_OF_SAMPLES * sizeof(curandState));
		init_curand_seed<<< _DEVICE_CONST_N_OF_BLOCK, _DEVICE_CONST_THREAD_PER_BLOCK >>>(curand_state_array, (unsigned)time(NULL));

		// device_vector を生成
		thrust::device_vector<input_array> input_vec_dev_temp(CONST_PARAM::N_OF_SAMPLES);
		input_array_device_vec = input_vec_dev_temp;

		thrust::device_vector<float> cost_vec_dev_temp( CONST_PARAM::N_OF_SAMPLES );
		cost_device_vec_for_sorting = cost_vec_dev_temp;

        thrust::device_vector<int> elite_indices_vec_dev_temp(CONST_PARAM::N_OF_THE_USING_BEST);
        elite_indices_device_vec = elite_indices_vec_dev_temp;

        thrust::device_vector<int> elite_selected_flags_vec_dev_temp(CONST_PARAM::N_OF_SAMPLES);
        elite_selected_flags_device_vec = elite_selected_flags_vec_dev_temp;

        thrust::device_vector<float> elite_lambda_vec_dev_temp(1);
        elite_lambda_device_vec = elite_lambda_vec_dev_temp;

        thrust::device_vector<input_array> best_input_vec_dev_temp(1);
        best_input_array_device_vec = best_input_vec_dev_temp;

		//  __constant__ メモリに定数をコピー
		cudaMemcpyToSymbol(target_state_device, &target_host, sizeof(target_state_t));

		cudaMemcpyToSymbol( sigma_k_device, sigma_k, 4 * sizeof( float ) );

        cudaMemcpyToSymbol( u_g_device, &CONST_PARAM_FLOAT::U_G, sizeof( float ) );
        cudaMemcpyToSymbol( u_upper_lim_device, &CONST_PARAM_FLOAT::U_UPPER_LIM, sizeof( float ) );
        cudaMemcpyToSymbol( u_lower_lim_device, &CONST_PARAM_FLOAT::U_LOWER_LIM, sizeof( float ) );

        cudaMemcpyToSymbol( i_xx_device, &CONST_PARAM_FLOAT::I_XX, sizeof( float ) );
        cudaMemcpyToSymbol( i_yy_device, &CONST_PARAM_FLOAT::I_YY, sizeof( float ) );
        cudaMemcpyToSymbol( i_zz_device, &CONST_PARAM_FLOAT::I_ZZ, sizeof( float ) );

        cudaMemcpyToSymbol( rotor_distance_device,  &CONST_PARAM_FLOAT::ROTOR_DISTANCE,  sizeof( float ) );
        cudaMemcpyToSymbol( max_rps_pow_device,     &CONST_PARAM_FLOAT::MAX_RPS_POW,     sizeof( float ) );
        cudaMemcpyToSymbol( max_thrust_device,      &CONST_PARAM_FLOAT::MAX_THRUST,      sizeof( float ) );
        cudaMemcpyToSymbol( mass_of_machine_device, &CONST_PARAM_FLOAT::MASS_OF_MACHINE, sizeof( float ) );
        cudaMemcpyToSymbol( torque_rate_device,     &CONST_PARAM_FLOAT::TORQUE_RATE,     sizeof( float ) );
        cudaMemcpyToSymbol( a_of_gravity_device,    &CONST_PARAM_FLOAT::A_OF_GRAVITY,    sizeof( float ) );

        cudaMemcpyToSymbol( mpc_xy_p,               &CONST_PARAM_FLOAT::MPC_XY_P,            sizeof( float ) );
        cudaMemcpyToSymbol( mpc_z_p,                &CONST_PARAM_FLOAT::MPC_Z_P,             sizeof( float ) );
        cudaMemcpyToSymbol( mpc_xy_vel_p_acc,       &CONST_PARAM_FLOAT::MPC_XY_VEL_P_ACC,    sizeof( float ) );
        cudaMemcpyToSymbol( mpc_z_vel_p_acc,        &CONST_PARAM_FLOAT::MPC_Z_VEL_P_ACC,     sizeof( float ) );
        cudaMemcpyToSymbol( mpc_xy_vel_i_acc,       &CONST_PARAM_FLOAT::MPC_XY_VEL_I_ACC,    sizeof( float ) );
        cudaMemcpyToSymbol( mpc_z_vel_i_acc,        &CONST_PARAM_FLOAT::MPC_Z_VEL_I_ACC,     sizeof( float ) );
        cudaMemcpyToSymbol( mpc_xy_vel_d_acc,       &CONST_PARAM_FLOAT::MPC_XY_VEL_D_ACC,    sizeof( float ) );
        cudaMemcpyToSymbol( mpc_z_vel_d_acc,        &CONST_PARAM_FLOAT::MPC_Z_VEL_D_ACC,     sizeof( float ) );
        cudaMemcpyToSymbol( mpc_xy_vel_max,         &CONST_PARAM_FLOAT::MPC_XY_VEL_MAX,      sizeof( float ) );
        cudaMemcpyToSymbol( mpc_z_vel_max_up,       &CONST_PARAM_FLOAT::MPC_Z_VEL_MAX_UP,    sizeof( float ) );
        cudaMemcpyToSymbol( mpc_z_vel_max_down,     &CONST_PARAM_FLOAT::MPC_Z_VEL_MAX_DOWN,  sizeof( float ) );
        cudaMemcpyToSymbol( mpc_thr_min,            &CONST_PARAM_FLOAT::MPC_THR_MIN,         sizeof( float ) );
        cudaMemcpyToSymbol( mpc_thr_max,            &CONST_PARAM_FLOAT::MPC_THR_MAX,         sizeof( float ) );
        cudaMemcpyToSymbol( mpc_thr_xy_margin,      &CONST_PARAM_FLOAT::MPC_THR_XY_MARGIN,   sizeof( float ) );
        cudaMemcpyToSymbol( arw_gain,               &CONST_PARAM_FLOAT::ARW_GAIN,            sizeof( float ) );
        cudaMemcpyToSymbol( mc_roll_p,              &CONST_PARAM_FLOAT::MC_ROLL_P,           sizeof( float ) );
        cudaMemcpyToSymbol( mc_pitch_p,             &CONST_PARAM_FLOAT::MC_PITCH_P,          sizeof( float ) );
        cudaMemcpyToSymbol( mc_yaw_p,               &CONST_PARAM_FLOAT::MC_YAW_P,            sizeof( float ) );
        cudaMemcpyToSymbol( mc_yaw_weight,          &CONST_PARAM_FLOAT::MC_YAW_WEIGHT,       sizeof( float ) );
        cudaMemcpyToSymbol( lpf,                    &CONST_PARAM_FLOAT::LPF,                 sizeof( float ) );
        cudaMemcpyToSymbol( mpc_thr_hover,          &CONST_PARAM_FLOAT::MPC_THR_HOVER,       sizeof( float ) );
        cudaMemcpyToSymbol( mpc_veld_lp,            &CONST_PARAM_FLOAT::MPC_VELD_LP,         sizeof( float ) );
        cudaMemcpyToSymbol( mc_rollrate_p,          &CONST_PARAM_FLOAT::MC_ROLLRATE_P,       sizeof( float ) );
        cudaMemcpyToSymbol( mc_pitchrate_p,         &CONST_PARAM_FLOAT::MC_PITCHRATE_P,      sizeof( float ) );
        cudaMemcpyToSymbol( mc_yawrate_p,           &CONST_PARAM_FLOAT::MC_YAWRATE_P,        sizeof( float ) );
        cudaMemcpyToSymbol( mc_rollrate_d,          &CONST_PARAM_FLOAT::MC_ROLLRATE_D,       sizeof( float ) );
        cudaMemcpyToSymbol( mc_pitchrate_d,         &CONST_PARAM_FLOAT::MC_PITCHRATE_D,      sizeof( float ) );
        cudaMemcpyToSymbol( mc_yawrate_d,           &CONST_PARAM_FLOAT::MC_YAWRATE_D,        sizeof( float ) );    
        cudaMemcpyToSymbol( mc_rollrate_i,          &CONST_PARAM_FLOAT::MC_ROLLRATE_I,       sizeof( float ) );
        cudaMemcpyToSymbol( mc_pitchrate_i,         &CONST_PARAM_FLOAT::MC_PITCHRATE_I,      sizeof( float ) );
        cudaMemcpyToSymbol( mc_yawrate_i,           &CONST_PARAM_FLOAT::MC_YAWRATE_I,        sizeof( float ) );

        cudaMemcpyToSymbol( control_period_device,        &CONST_PARAM_FLOAT::CONTROL_PERIOD,        sizeof( float ) );
        cudaMemcpyToSymbol( integration_step_size_device, &CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE, sizeof( float ) );

        cudaMemcpyToSymbol( motor_input_scaling,          &CONST_PARAM_FLOAT::MOTOR_INPUT_SCALING,      sizeof(float));
        cudaMemcpyToSymbol( max_rot_velocity,             &CONST_PARAM_FLOAT::MAX_ROT_VELOCITY,         sizeof(float));
        cudaMemcpyToSymbol( motor_time_constant_up,       &CONST_PARAM_FLOAT::MOTOR_TIME_CONSTANT_UP,   sizeof(float));
        cudaMemcpyToSymbol( motor_time_constant_down,     &CONST_PARAM_FLOAT::MOTOR_TIME_CONSTANT_DOWN, sizeof(float));
        cudaMemcpyToSymbol( motor_thrust_constant,        &CONST_PARAM_FLOAT::MOTOR_THRUST_CONSTANT,    sizeof(float));
        cudaMemcpyToSymbol( moment_constant,              &CONST_PARAM_FLOAT::MOMENT_CONSTANT,          sizeof(float));

        cudaMemcpyToSymbol( rotor_positions,              CONST_PARAM_FLOAT::ROTOR_POSITIONS,            sizeof(CONST_PARAM_FLOAT::ROTOR_POSITIONS));
        cudaMemcpyToSymbol( rotor_yaw_signs,              CONST_PARAM_FLOAT::ROTOR_YAW_SIGNS,            sizeof(CONST_PARAM_FLOAT::ROTOR_YAW_SIGNS));
        cudaMemcpyToSymbol( px4_quad_x_mix,               CONST_PARAM_FLOAT::PX4_QUAD_X_MIX,             sizeof(CONST_PARAM_FLOAT::PX4_QUAD_X_MIX));
        cudaMemcpyToSymbol( px4_quad_x_mix_inv,           CONST_PARAM_FLOAT::PX4_QUAD_X_MIX_INV,         sizeof(CONST_PARAM_FLOAT::PX4_QUAD_X_MIX_INV));
        cudaMemcpyToSymbol( px4_actuator_min,             CONST_PARAM_FLOAT::PX4_ACTUATOR_MIN,           sizeof(CONST_PARAM_FLOAT::PX4_ACTUATOR_MIN));
        cudaMemcpyToSymbol( px4_actuator_max,             CONST_PARAM_FLOAT::PX4_ACTUATOR_MAX,           sizeof(CONST_PARAM_FLOAT::PX4_ACTUATOR_MAX));

        cudaMemcpyToSymbol( angular_accel_lp,            &CONST_PARAM_FLOAT::ANGULAR_ACCEL_LP,           sizeof(float));
        cudaMemcpyToSymbol( mc_rollrate_k,               &CONST_PARAM_FLOAT::MC_ROLLRATE_K,              sizeof(float));
        cudaMemcpyToSymbol( mc_pitchrate_k,              &CONST_PARAM_FLOAT::MC_PITCHRATE_K,             sizeof(float));
        cudaMemcpyToSymbol( mc_yawrate_k,                &CONST_PARAM_FLOAT::MC_YAWRATE_K,               sizeof(float));
        cudaMemcpyToSymbol( mc_rollrate_ff,              &CONST_PARAM_FLOAT::MC_ROLLRATE_FF,             sizeof(float));
        cudaMemcpyToSymbol( mc_pitchrate_ff,             &CONST_PARAM_FLOAT::MC_PITCHRATE_FF,            sizeof(float));
        cudaMemcpyToSymbol( mc_yawrate_ff,               &CONST_PARAM_FLOAT::MC_YAWRATE_FF,              sizeof(float));
        cudaMemcpyToSymbol( mc_rr_int_lim,               &CONST_PARAM_FLOAT::MC_RR_INT_LIM,              sizeof(float));
        cudaMemcpyToSymbol( mc_pr_int_lim,               &CONST_PARAM_FLOAT::MC_PR_INT_LIM,              sizeof(float));
        cudaMemcpyToSymbol( mc_yr_int_lim,               &CONST_PARAM_FLOAT::MC_YR_INT_LIM,              sizeof(float));
        cudaMemcpyToSymbol( mc_yaw_tq_cutoff,            &CONST_PARAM_FLOAT::MC_YAW_TQ_CUTOFF,           sizeof(float));

        cudaMemcpyToSymbol( ca_minimum_yaw_margin,        &CONST_PARAM_FLOAT::CA_MINIMUM_YAW_MARGIN,     sizeof(float));
        {
            float acceleration_bias_init[3] = {0.0f, 0.0f, 0.0f};
            cudaMemcpyToSymbol(acceleration_bias_device, acceleration_bias_init, sizeof(acceleration_bias_init));
        }
        cudaMemcpyToSymbol( motor_command_delay_steps,    &CONST_PARAM_FLOAT::MOTOR_COMMAND_DELAY_STEPS, sizeof(int));

        cudaMemcpyToSymbol(takeoff_state_rampup_device, &CONST_PARAM_FLOAT::TAKEOFF_STATE_RAMPUP, sizeof(int));
        cudaMemcpyToSymbol(takeoff_state_flight_device, &CONST_PARAM_FLOAT::TAKEOFF_STATE_FLIGHT, sizeof(int));
        cudaMemcpyToSymbol( square_waypoints_device,    CONST_PARAM_FLOAT::square_waypoints,            sizeof(CONST_PARAM_FLOAT::square_waypoints) );

        {
            int takeoff_state_init = CONST_PARAM_FLOAT::TAKEOFF_STATE_FLIGHT;
            float takeoff_tilt_limit_init = CONST_PARAM_FLOAT::MPC_TILT_MAX;
            float takeoff_tilt_limit_sin_init = std::sin(takeoff_tilt_limit_init);
            float takeoff_tilt_limit_cos_init = std::cos(takeoff_tilt_limit_init);
            int landed_init = 0;
            int ground_contact_init = 0;
            int maybe_landed_init = 0;
            cudaMemcpyToSymbol(takeoff_state_device, &takeoff_state_init, sizeof(int));
            cudaMemcpyToSymbol(takeoff_tilt_limit_sin_device, &takeoff_tilt_limit_sin_init, sizeof(float));
            cudaMemcpyToSymbol(takeoff_tilt_limit_cos_device, &takeoff_tilt_limit_cos_init, sizeof(float));
            cudaMemcpyToSymbol(landed_device, &landed_init, sizeof(int));
            cudaMemcpyToSymbol(ground_contact_device, &ground_contact_init, sizeof(int));
            cudaMemcpyToSymbol(maybe_landed_device, &maybe_landed_init, sizeof(int));
        }

#ifdef PREDICTABLE_COLLISION_WITH_WALL
        cudaMemcpyToSymbol( x_wall_device,        &CONST_PARAM_FLOAT::X_WALL,               sizeof( float ) );
        cudaMemcpyToSymbol( wall_nv_x_device,     &CONST_PARAM_FLOAT::WALL_NORMAL_VECTOR_X, sizeof( float ) );
        cudaMemcpyToSymbol( wall_nv_y_device,     &CONST_PARAM_FLOAT::WALL_NORMAL_VECTOR_Y, sizeof( float ) );
        cudaMemcpyToSymbol( wall_nv_z_device,     &CONST_PARAM_FLOAT::WALL_NORMAL_VECTOR_Z, sizeof( float ) );
        cudaMemcpyToSymbol( r_of_ring_device,     &CONST_PARAM_FLOAT::R_OF_RING,            sizeof( float ) );
        cudaMemcpyToSymbol( coeff_of_rest_device, &CONST_PARAM_FLOAT::COEFF_OF_REST,        sizeof( float ) );
#endif
	}

    // デストラクタ
	mcmpc_controller::~mcmpc_controller()
	{
		cudaFree(curand_state_array);
	}

    // 乱数シードの初期化
	__global__ static void init_curand_seed( curandState* state_array, int seed )
    {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        curand_init( seed, id, 0, &state_array[id] );
    }


    // GPU上で入力列を生成, シミュレーションしコストを求める
	__global__ static void generate_input_samples_and_calc_costs(curandState* state, input_array* input_array_sample_device, float* cost_vec){
		int id = blockDim.x * blockIdx.x + threadIdx.x;

		input_array_sample_device[id].generate_input_array(state[id]);
		input_array_sample_device[id].do_simulation(id);

		// sort用に, float配列にコストを同順でコピー
		cost_vec[id] = input_array_sample_device[id].cost;
	}

    __global__ static void select_elite_indices_and_lambda(
        const float* cost_vec,
        int* elite_indices,
        int* selected_flags,
        float* lambda_out,
        int n_samples,
        int n_elite)
    {
        __shared__ float block_best_cost[_DEVICE_CONST_THREAD_PER_BLOCK];
        __shared__ int block_best_index[_DEVICE_CONST_THREAD_PER_BLOCK];
        __shared__ float lambda_sum;

        int tid = threadIdx.x;
        for (int sample = tid; sample < n_samples; sample += blockDim.x) {
            selected_flags[sample] = 0;
        }
        if (tid == 0) {
            lambda_sum = 0.0f;
        }
        __syncthreads();

        for (int elite = 0; elite < n_elite; elite++) {
            float local_best_cost = 1.0e30f;
            int local_best_index = -1;
            for (int sample = tid; sample < n_samples; sample += blockDim.x) {
                if (selected_flags[sample] != 0) {
                    continue;
                }
                float sample_cost = cost_vec[sample];
                if (sample_cost < local_best_cost ||
                    (sample_cost == local_best_cost && sample < local_best_index)) {
                    local_best_cost = sample_cost;
                    local_best_index = sample;
                }
            }
            block_best_cost[tid] = local_best_cost;
            block_best_index[tid] = local_best_index;
            __syncthreads();

            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    float other_cost = block_best_cost[tid + stride];
                    int other_index = block_best_index[tid + stride];
                    if (other_cost < block_best_cost[tid] ||
                        (other_cost == block_best_cost[tid] &&
                         other_index >= 0 &&
                         (block_best_index[tid] < 0 || other_index < block_best_index[tid]))) {
                        block_best_cost[tid] = other_cost;
                        block_best_index[tid] = other_index;
                    }
                }
                __syncthreads();
            }

            if (tid == 0) {
                int best_index = block_best_index[0];
                if (best_index >= 0) {
                    elite_indices[elite] = best_index;
                    selected_flags[best_index] = 1;
                    lambda_sum += block_best_cost[0];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            lambda_out[0] = lambda_sum / (float)n_elite;
        }
    }

    __global__ static void calc_weighted_average_input_on_device(
        const input_array* input_samples,
        const int* elite_indices,
        const float* lambda_in,
        input_array* best_input,
        int n_elite)
    {
        __shared__ float elite_weights[_DEVICE_CONST_THREAD_PER_BLOCK];
        __shared__ float sum_of_weight_shared[_DEVICE_CONST_THREAD_PER_BLOCK];
        __shared__ float sum_of_weight;

        int tid = threadIdx.x;
        float lambda = lambda_in[0];
        float local_weight = 0.0f;
        if (tid < n_elite) {
            int sample_index = elite_indices[tid];
            local_weight = expf(-input_samples[sample_index].cost / lambda);
            elite_weights[tid] = local_weight;
        }
        sum_of_weight_shared[tid] = local_weight;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sum_of_weight_shared[tid] += sum_of_weight_shared[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            sum_of_weight = sum_of_weight_shared[0];
            best_input[0].cost = input_samples[elite_indices[0]].cost;
        }
        __syncthreads();

        int n_elements = _DEVICE_CONST_HORIZON * 4;
        for (int element = tid; element < n_elements; element += blockDim.x) {
            int horizon = element / 4;
            int axis = element - horizon * 4;
            float weighted_value = 0.0f;
            for (int elite = 0; elite < n_elite; elite++) {
                int sample_index = elite_indices[elite];
                weighted_value +=
                    input_samples[sample_index].decoupled_position[horizon][axis] *
                    elite_weights[elite];
            }
            best_input[0].decoupled_position[horizon][axis] =
                weighted_value / sum_of_weight;
        }
    }

    static void input_constraint_cpu(float& rps_cw1, float& rps_cw2, float& rps_ccw1, float& rps_ccw2){
		// rps_cw1
		if(rps_cw1 > CONST_PARAM_FLOAT::U_UPPER_LIM) rps_cw1 = CONST_PARAM_FLOAT::U_UPPER_LIM;
		if(rps_cw1 < CONST_PARAM_FLOAT::U_LOWER_LIM) rps_cw1 = CONST_PARAM_FLOAT::U_LOWER_LIM;
		// rps_cw2
		if(rps_cw2 > CONST_PARAM_FLOAT::U_UPPER_LIM) rps_cw2 = CONST_PARAM_FLOAT::U_UPPER_LIM;
		if(rps_cw2 < CONST_PARAM_FLOAT::U_LOWER_LIM) rps_cw2 = CONST_PARAM_FLOAT::U_LOWER_LIM;
		// rps_ccw1
		if(rps_ccw1 > CONST_PARAM_FLOAT::U_UPPER_LIM) rps_ccw1 = CONST_PARAM_FLOAT::U_UPPER_LIM;
		if(rps_ccw1 < CONST_PARAM_FLOAT::U_LOWER_LIM) rps_ccw1 = CONST_PARAM_FLOAT::U_LOWER_LIM;
		// rps_ccw2
		if(rps_ccw2 > CONST_PARAM_FLOAT::U_UPPER_LIM) rps_ccw2 = CONST_PARAM_FLOAT::U_UPPER_LIM;
		if(rps_ccw2 < CONST_PARAM_FLOAT::U_LOWER_LIM) rps_ccw2 = CONST_PARAM_FLOAT::U_LOWER_LIM;
	}

    void update_target_state_device()
    {
        cudaError_t err = cudaMemcpyToSymbol(
            target_state_device,
            &target_host,
            sizeof(target_state_t)
        );

        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol target_state_device failed: "
                    << cudaGetErrorString(err) << std::endl;
        }
    }

    float mcmpc_controller::calc_weighted_average_and_min_cost(float var_and_z_i[])
	{
        select_elite_indices_and_lambda<<<1, _DEVICE_CONST_THREAD_PER_BLOCK>>>(
            thrust::raw_pointer_cast(cost_device_vec_for_sorting.data()),
            thrust::raw_pointer_cast(elite_indices_device_vec.data()),
            thrust::raw_pointer_cast(elite_selected_flags_device_vec.data()),
            thrust::raw_pointer_cast(elite_lambda_device_vec.data()),
            CONST_PARAM::N_OF_SAMPLES,
            CONST_PARAM::N_OF_THE_USING_BEST);

        calc_weighted_average_input_on_device<<<1, _DEVICE_CONST_THREAD_PER_BLOCK>>>(
            thrust::raw_pointer_cast(input_array_device_vec.data()),
            thrust::raw_pointer_cast(elite_indices_device_vec.data()),
            thrust::raw_pointer_cast(elite_lambda_device_vec.data()),
            thrust::raw_pointer_cast(best_input_array_device_vec.data()),
            CONST_PARAM::N_OF_THE_USING_BEST);

        cudaMemcpy(
            &best_input_array,
            thrust::raw_pointer_cast(best_input_array_device_vec.data()),
            sizeof(input_array),
            cudaMemcpyDeviceToHost);
		return best_input_array.cost;
	}

    // 最初に get_instance() が呼ばれたときに唯一のインスタンスを生成し，以降はそれを保持（シングルトン）
    mcmpc_controller& mcmpc_controller::get_instance()
    {
        static mcmpc_controller instance;   // インスタンスの生成
        return instance;
    }

    float mcmpc_controller::calc_optimal_input(float var_and_z_i[], double &optimal_input1, double &optimal_input2, double &optimal_input3, double &optimal_input4 )
    {
		float min_cost = std::numeric_limits<float>::infinity();

		// 変数のコピー
        cudaMemcpyToSymbol( var_and_z_i_device, var_and_z_i, (_N_OF_ODES +1) * sizeof( float ) );

		// インプットシフト
        for ( int i = 0; i < _DEVICE_CONST_HORIZON - 1; i++ )
            for ( int j = 0; j < 4; j++ )
                best_input_array.decoupled_position[i][j] = best_input_array.decoupled_position[i + 1][j];

		// 準最適制御入力の計算 同一周期内反復
        for ( int k = 0; k < CONST_PARAM::ITERATION_TIMES; k++ )
        {
            float sigma_k_iter[4];
            float scale = powf(0.5f, k);
            for (int j = 0; j < 4; j++) {
                sigma_k_iter[j] = sigma_k[j] * scale;
            }
            cudaMemcpyToSymbol(sigma_k_device, sigma_k_iter, 4 * sizeof(float));
            
            cudaMemcpyToSymbol( average_input_device, &best_input_array, sizeof( input_array ) );
            
            generate_input_samples_and_calc_costs<<< _DEVICE_CONST_N_OF_BLOCK, _DEVICE_CONST_THREAD_PER_BLOCK >>>( curand_state_array, thrust::raw_pointer_cast( input_array_device_vec.data() ), thrust::raw_pointer_cast( cost_device_vec_for_sorting.data() ) );

            cudaDeviceSynchronize();

            min_cost = calc_weighted_average_and_min_cost( var_and_z_i );
        }

		// PIDカスケード用修正　入力：スラスト＋姿勢
        optimal_input1 = (double)(best_input_array.decoupled_position[0][x]);
        optimal_input2 = (double)(best_input_array.decoupled_position[0][y]);
        optimal_input3 = (double)(best_input_array.decoupled_position[0][z]);
        optimal_input4 = (double)(best_input_array.decoupled_position[0][yaw]);

		return min_cost;
	}

    // グラフ出力用に計算済みの準最適入力をコピー
	void mcmpc_controller::copy_best_input_array(input_array &dst)
	{
		for ( int i = 0; i < _DEVICE_CONST_HORIZON; i++ )
            for ( int j = 0; j < 4; j++ )
                dst.decoupled_position[i][j] = best_input_array.decoupled_position[i][j];
	}

}
