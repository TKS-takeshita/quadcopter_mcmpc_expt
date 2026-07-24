#include <iostream>
#include <cmath>

#include "mpc_simulator.cu"

namespace qc_mcmpc
{
    static constexpr int TOPK_BLOCK_SIZE = 1024;
    static constexpr int TOPK_SIZE = 100;
    static constexpr int MCMPC_SAMPLE_COUNT =
        _DEVICE_CONST_THREAD_PER_BLOCK * _DEVICE_CONST_N_OF_BLOCK;
    static constexpr int TOPK_NUM_BLOCKS =
        (MCMPC_SAMPLE_COUNT + TOPK_BLOCK_SIZE - 1) / TOPK_BLOCK_SIZE;
    static constexpr int TOPK_CANDIDATE_COUNT = TOPK_NUM_BLOCKS * TOPK_SIZE;

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
    __constant__ float mpc_tilt_max_device;

    target_state_t target_host;

#ifdef PREDICTABLE_COLLISION_WITH_WALL
	__constant__ int prediction_wall_collision_enabled_device;
	__constant__ int truth_wall_collision_enabled_device;
	__constant__ float x_wall_device;
	__constant__ float wall_nv_x_device;
	__constant__ float wall_nv_y_device;
	__constant__ float wall_nv_z_device;
	__constant__ float r_of_ring_device;
	__constant__ float coeff_of_rest_device;
#endif

    __constant__ float control_period_device;
	__constant__ float integration_step_size_device;

    __device__ float deterministic_sim_trajectory_device[_DEVICE_CONST_HORIZON][_N_OF_ODES];
    __device__ float deterministic_sim_vel_int_device[_DEVICE_CONST_HORIZON][3];
    __device__ float deterministic_sim_prev_acceleration_device[_DEVICE_CONST_HORIZON][3];
    __device__ float deterministic_sim_rate_int_device[_DEVICE_CONST_HORIZON][3];
    __device__ float deterministic_sim_prev_angular_acceleration_device[_DEVICE_CONST_HORIZON][3];
    __device__ float deterministic_sim_motor_speed_device[_DEVICE_CONST_HORIZON][4];
    __device__ float deterministic_sim_vel_setpoint_device[_DEVICE_CONST_HORIZON][3];
    __device__ float deterministic_sim_acc_setpoint_device[_DEVICE_CONST_HORIZON][3];
    __device__ float deterministic_sim_att_setpoint_device[_DEVICE_CONST_HORIZON][4];
    __device__ float deterministic_sim_thrust_setpoint_device[_DEVICE_CONST_HORIZON][3];
    __device__ float deterministic_sim_omega_setpoint_device[_DEVICE_CONST_HORIZON][3];
    __device__ float deterministic_sim_torque_setpoint_device[_DEVICE_CONST_HORIZON][3];
    __device__ float deterministic_sim_motor_setpoint_device[_DEVICE_CONST_HORIZON][4];
#ifdef PREDICTABLE_COLLISION_WITH_WALL
    __device__ int deterministic_sim_collision_device[_DEVICE_CONST_HORIZON];
    __device__ float deterministic_sim_contact_point_device[_DEVICE_CONST_HORIZON][3];
    __device__ float deterministic_sim_impulse_device[_DEVICE_CONST_HORIZON];
    __device__ float deterministic_sim_pre_contact_normal_velocity_device[_DEVICE_CONST_HORIZON];
#endif

    __constant__ float var_and_z_i_device[_N_OF_ODES];
    __constant__ input_array average_input_device;
    __constant__ float sigma_k_device[4];
    __constant__ int square_waypoint_index_device;
    __constant__ float square_waypoint_change_time_device;
    __constant__ float mcmpc_log_device;
    __constant__ float square_waypoints_device[_SQUARE_WAYPOINTS][3];
    __constant__ int square_waypoint_count_device;
    __constant__ float square_waypoint_threshold_device;
    __constant__ int takeoff_state_device;
    __constant__ float takeoff_tilt_limit_device;
    __constant__ int landed_device;
    __constant__ int ground_contact_device;
    __constant__ int maybe_landed_device;

    __global__ static void init_curand_seed(curandState *state_array, int seed);
	__global__ static void generate_input_samples_and_calc_costs(curandState *state, input_array* input_array_sample_device, float* cost_vec);
    __global__ static void select_block_topk(
        const float* costs, int count, float* candidate_costs, int* candidate_indices);
    __global__ static void select_final_topk(
        const float* candidate_costs, const int* candidate_indices, int count, int* top_indices);
    __global__ static void weighted_average_topk(
        const input_array* samples, const float* costs, const int* top_indices, input_array* best_input);

    __global__ void simulate_best_input_trajectory_kernel(input_array best_input)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            best_input.do_simulation(-1);
        }
    }

    void update_target_state_device();

#ifdef PREDICTABLE_COLLISION_WITH_WALL
	static float dot_vec_cpu(float v1_x, float v1_y, float v1_z, float v2_x, float v2_y, float v2_z);
	static void cross_vec_cpu(float v1_x, float v1_y, float v1_z, float v2_x, float v2_y, float v2_z, float& v_ans_x, float& v_ans_y, float& v_ans_z);
	static void inverse_3x3__cpu(float matrix_src[3][3], float ans[3][3]);
#endif

// コンストラクタ
	mcmpc_controller::mcmpc_controller()
	{
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


        topk_candidate_cost_device_vec.resize(TOPK_CANDIDATE_COUNT);
        topk_candidate_index_device_vec.resize(TOPK_CANDIDATE_COUNT);
        topk_index_device_vec.resize(TOPK_SIZE);
        best_input_device_vec.resize(1);

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
        cudaMemcpyToSymbol(mpc_tilt_max_device, &CONST_PARAM_FLOAT::MPC_TILT_MAX, sizeof(float));

        cudaMemcpyToSymbol( square_waypoints_device,    CONST_PARAM_FLOAT::square_waypoints,            sizeof(CONST_PARAM_FLOAT::square_waypoints) );
        cudaMemcpyToSymbol(square_waypoint_count_device, &square_waypoint_count, sizeof(int));
        cudaMemcpyToSymbol(square_waypoint_threshold_device, &square_waypoint_threshold, sizeof(float));

        {
            int takeoff_state_init = CONST_PARAM_FLOAT::TAKEOFF_STATE_FLIGHT;
            float takeoff_tilt_limit_init = CONST_PARAM_FLOAT::MPC_TILT_MAX;
            int landed_init = 0;
            int ground_contact_init = 0;
            int maybe_landed_init = 0;
            cudaMemcpyToSymbol(takeoff_state_device, &takeoff_state_init, sizeof(int));
            cudaMemcpyToSymbol(takeoff_tilt_limit_device, &takeoff_tilt_limit_init, sizeof(float));
            cudaMemcpyToSymbol(landed_device, &landed_init, sizeof(int));
            cudaMemcpyToSymbol(ground_contact_device, &ground_contact_init, sizeof(int));
            cudaMemcpyToSymbol(maybe_landed_device, &maybe_landed_init, sizeof(int));
        }

#ifdef PREDICTABLE_COLLISION_WITH_WALL
        const int wall_collision_enabled = 1;
        cudaMemcpyToSymbol( prediction_wall_collision_enabled_device, &wall_collision_enabled, sizeof( int ) );
        cudaMemcpyToSymbol( truth_wall_collision_enabled_device, &wall_collision_enabled,      sizeof( int ) );
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

    __device__ static bool topk_pair_greater(float lhs_cost, int lhs_index, float rhs_cost, int rhs_index)
    {
        return (lhs_cost > rhs_cost) ||
            ((lhs_cost == rhs_cost) && (lhs_index > rhs_index));
    }

    __global__ static void select_block_topk(
        const float* costs,
        int count,
        float* candidate_costs,
        int* candidate_indices)
    {
        __shared__ float shared_cost[TOPK_BLOCK_SIZE];
        __shared__ int shared_index[TOPK_BLOCK_SIZE];

        const int tid = threadIdx.x;
        const int global_index = blockIdx.x * TOPK_BLOCK_SIZE + tid;
        float cost = (global_index < count) ? costs[global_index] : INFINITY;
        if (!isfinite(cost)) {
            cost = INFINITY;
        }
        shared_cost[tid] = cost;
        shared_index[tid] = (global_index < count) ? global_index : 0x7fffffff;
        __syncthreads();

        // 1024件をコスト、同値時はインデックスで昇順に並べる。
        for (int size = 2; size <= TOPK_BLOCK_SIZE; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                const int other = tid ^ stride;
                if (other > tid) {
                    const bool ascending = (tid & size) == 0;
                    const bool greater = topk_pair_greater(
                        shared_cost[tid], shared_index[tid],
                        shared_cost[other], shared_index[other]);
                    if (greater == ascending) {
                        const float tmp_cost = shared_cost[tid];
                        const int tmp_index = shared_index[tid];
                        shared_cost[tid] = shared_cost[other];
                        shared_index[tid] = shared_index[other];
                        shared_cost[other] = tmp_cost;
                        shared_index[other] = tmp_index;
                    }
                }
                __syncthreads();
            }
        }

        if (tid < TOPK_SIZE) {
            const int output = blockIdx.x * TOPK_SIZE + tid;
            candidate_costs[output] = shared_cost[tid];
            candidate_indices[output] = shared_index[tid];
        }
    }

    __global__ static void select_final_topk(
        const float* candidate_costs,
        const int* candidate_indices,
        int count,
        int* top_indices)
    {
        __shared__ float shared_cost[TOPK_BLOCK_SIZE];
        __shared__ int shared_index[TOPK_BLOCK_SIZE];

        const int tid = threadIdx.x;
        float cost = (tid < count) ? candidate_costs[tid] : INFINITY;
        if (!isfinite(cost)) {
            cost = INFINITY;
        }
        shared_cost[tid] = cost;
        shared_index[tid] = (tid < count) ? candidate_indices[tid] : 0x7fffffff;
        __syncthreads();

        for (int size = 2; size <= TOPK_BLOCK_SIZE; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                const int other = tid ^ stride;
                if (other > tid) {
                    const bool ascending = (tid & size) == 0;
                    const bool greater = topk_pair_greater(
                        shared_cost[tid], shared_index[tid],
                        shared_cost[other], shared_index[other]);
                    if (greater == ascending) {
                        const float tmp_cost = shared_cost[tid];
                        const int tmp_index = shared_index[tid];
                        shared_cost[tid] = shared_cost[other];
                        shared_index[tid] = shared_index[other];
                        shared_cost[other] = tmp_cost;
                        shared_index[other] = tmp_index;
                    }
                }
                __syncthreads();
            }
        }

        if (tid < TOPK_SIZE) {
            top_indices[tid] = shared_index[tid];
        }
    }

    __global__ static void weighted_average_topk(
        const input_array* samples,
        const float* costs,
        const int* top_indices,
        input_array* best_input)
    {
        __shared__ float reduction[128];
        __shared__ float weights[TOPK_SIZE];
        const int tid = threadIdx.x;

        float cost = 0.0f;
        if (tid < TOPK_SIZE) {
            cost = costs[top_indices[tid]];
        }
        reduction[tid] = cost;
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }
        const float lambda = reduction[0] / static_cast<float>(TOPK_SIZE);

        const float weight = (tid < TOPK_SIZE)
            ? expf(-cost / lambda)
            : 0.0f;
        if (tid < TOPK_SIZE) {
            weights[tid] = weight;
        }
        reduction[tid] = weight;
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }
        const float weight_sum = reduction[0];

        constexpr int input_count = _DEVICE_CONST_HORIZON * 4;
        for (int element = tid; element < input_count; element += blockDim.x) {
            float weighted_sum = 0.0f;
            const int horizon_index = element / 4;
            const int input_index = element % 4;
            for (int elite = 0; elite < TOPK_SIZE; elite++) {
                weighted_sum += samples[top_indices[elite]].decoupled_position[horizon_index][input_index]
                    * weights[elite];
            }
            best_input->decoupled_position[horizon_index][input_index] = weighted_sum / weight_sum;
        }
        if (tid == 0) {
            best_input->cost = 0.0f;
        }
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

    float mcmpc_controller::calc_weighted_average_and_min_cost()
	{
        static_assert(TOPK_SIZE == 100, "TOPK_SIZE must match N_OF_THE_USING_BEST");
        select_block_topk<<<TOPK_NUM_BLOCKS, TOPK_BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(cost_device_vec_for_sorting.data()),
            CONST_PARAM::N_OF_SAMPLES,
            thrust::raw_pointer_cast(topk_candidate_cost_device_vec.data()),
            thrust::raw_pointer_cast(topk_candidate_index_device_vec.data()));
        select_final_topk<<<1, TOPK_BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(topk_candidate_cost_device_vec.data()),
            thrust::raw_pointer_cast(topk_candidate_index_device_vec.data()),
            TOPK_CANDIDATE_COUNT,
            thrust::raw_pointer_cast(topk_index_device_vec.data()));
        weighted_average_topk<<<1, 128>>>(
            thrust::raw_pointer_cast(input_array_device_vec.data()),
            thrust::raw_pointer_cast(cost_device_vec_for_sorting.data()),
            thrust::raw_pointer_cast(topk_index_device_vec.data()),
            thrust::raw_pointer_cast(best_input_device_vec.data()));

        // CPUへ戻すのは平均後の1入力列のみ。elite 100入力列はGPU内で完結する。
        cudaMemcpy(
            &best_input_array,
            thrust::raw_pointer_cast(best_input_device_vec.data()),
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
        cudaMemcpyToSymbol( var_and_z_i_device, var_and_z_i, _N_OF_ODES * sizeof( float ) );

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

            min_cost = calc_weighted_average_and_min_cost();
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
