#include <iostream>
#include <thrust/sequence.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

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
    __constant__ float arw_gain;
    __constant__ float mc_roll_p;
    __constant__ float mc_pitch_p;
    __constant__ float mc_yaw_p;
    __constant__ float mc_yaw_weight;
    __constant__ float lpf;
    __constant__ float mpc_thr_hover;
    __constant__ float mpc_vel_lp;
    __constant__ float mpc_veld_lp;
    __constant__ float mc_rollrate_p;
    __constant__ float mc_pitchrate_p;
    __constant__ float mc_yawrate_p;
    __constant__ float mc_rollrate_d;
    __constant__ float mc_pitchrate_d;
    __constant__ float mc_yawrate_d;

    __constant__ float prev_velocity_device[3];
    __constant__ float vel_int_device[3];
    __constant__ float prev_acceleration_device[3];
    __constant__ float prev_angular_velocity_device[3];

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

	__global__ static void init_curand_seed(curandState *state_array, int seed);
	__global__ static void generate_input_samples_and_calc_costs(curandState *state, input_array* input_array_sample_device, float* cost_vec);
	__global__ static void extract_elite_sample(input_array* src, input_array* dst, int* elite_indices);

	// コスト再計算用関数
	static void input_constraint_cpu(float& rps_z, float& rps_wx, float& rps_wy, float& rps_ws);

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
		target_state_t init_target{};
		init_target.e0 = CONST_PARAM_FLOAT::INIT_TARGET_E0;
		init_target.e1 = CONST_PARAM_FLOAT::INIT_TARGET_E1;
		init_target.e2 = CONST_PARAM_FLOAT::INIT_TARGET_E2;
		init_target.e3 = CONST_PARAM_FLOAT::INIT_TARGET_E3;
		init_target.wx = CONST_PARAM_FLOAT::INIT_TARGET_WX;
		init_target.wy = CONST_PARAM_FLOAT::INIT_TARGET_WY;
		init_target.wz = CONST_PARAM_FLOAT::INIT_TARGET_WZ;
		init_target.x = CONST_PARAM_FLOAT::INIT_TARGET_X;
		init_target.y = CONST_PARAM_FLOAT::INIT_TARGET_Y;
		init_target.z = CONST_PARAM_FLOAT::INIT_TARGET_Z;
		init_target.xp = CONST_PARAM_FLOAT::INIT_TARGET_ZP;
		init_target.yp = CONST_PARAM_FLOAT::INIT_TARGET_YP;
		init_target.zp = CONST_PARAM_FLOAT::INIT_TARGET_ZP;

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

		thrust::device_vector<input_array> input_vec_dev_elite_temp(CONST_PARAM::N_OF_THE_USING_BEST);
		input_array_device_vec_elite = input_vec_dev_elite_temp;

		thrust::device_vector<int> indices_vec_dev_temp(CONST_PARAM::N_OF_SAMPLES);
		indices_device_vec = indices_vec_dev_temp;

		thrust::device_vector<float> cost_vec_dev_temp( CONST_PARAM::N_OF_SAMPLES );
		cost_device_vec_for_sorting = cost_vec_dev_temp;

		// host_vectorを生成
		thrust::host_vector<input_array> input_vec_host_elite_temp( CONST_PARAM::N_OF_THE_USING_BEST );
		input_array_host_vec_elite = input_vec_host_elite_temp;

		//  __constant__ メモリに定数をコピー
		cudaMemcpyToSymbol(target_state_device, &init_target, sizeof(target_state_t));

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
        cudaMemcpyToSymbol( arw_gain,               &CONST_PARAM_FLOAT::ARW_GAIN,            sizeof( float ) );
        cudaMemcpyToSymbol( mc_roll_p,              &CONST_PARAM_FLOAT::MC_ROLL_P,           sizeof( float ) );
        cudaMemcpyToSymbol( mc_pitch_p,             &CONST_PARAM_FLOAT::MC_PITCH_P,          sizeof( float ) );
        cudaMemcpyToSymbol( mc_yaw_p,               &CONST_PARAM_FLOAT::MC_YAW_P,            sizeof( float ) );
        cudaMemcpyToSymbol( mc_yaw_weight,          &CONST_PARAM_FLOAT::MC_YAW_WEIGHT,       sizeof( float ) );
        cudaMemcpyToSymbol( lpf,                    &CONST_PARAM_FLOAT::LPF,                 sizeof( float ) );
        cudaMemcpyToSymbol( mpc_thr_hover,          &CONST_PARAM_FLOAT::MPC_THR_HOVER,       sizeof( float ) );
        cudaMemcpyToSymbol( mpc_vel_lp,             &CONST_PARAM_FLOAT::MPC_VEL_LP,          sizeof( float ) );
        cudaMemcpyToSymbol( mpc_veld_lp,            &CONST_PARAM_FLOAT::MPC_VELD_LP,         sizeof( float ) );
        cudaMemcpyToSymbol( mc_rollrate_p,          &CONST_PARAM_FLOAT::MC_ROLLRATE_P,       sizeof( float ) );
        cudaMemcpyToSymbol( mc_pitchrate_p,         &CONST_PARAM_FLOAT::MC_PITCHRATE_P,      sizeof( float ) );
        cudaMemcpyToSymbol( mc_yawrate_p,           &CONST_PARAM_FLOAT::MC_YAWRATE_P,        sizeof( float ) );
        cudaMemcpyToSymbol( mc_rollrate_d,          &CONST_PARAM_FLOAT::MC_ROLLRATE_D,       sizeof( float ) );
        cudaMemcpyToSymbol( mc_pitchrate_d,         &CONST_PARAM_FLOAT::MC_PITCHRATE_D,      sizeof( float ) );
        cudaMemcpyToSymbol( mc_yawrate_d,           &CONST_PARAM_FLOAT::MC_YAWRATE_D,        sizeof( float ) );    

        cudaMemcpyToSymbol( control_period_device,        &CONST_PARAM_FLOAT::CONTROL_PERIOD,        sizeof( float ) );
        cudaMemcpyToSymbol( integration_step_size_device, &CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE, sizeof( float ) );

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
		input_array_sample_device[id].do_simulation();

		// sort用に, float配列にコストを同順でコピー
		cost_vec[id] = input_array_sample_device[id].cost;
	}

    // エリートサンプルだけの入力列 devivce_vectorをGPUで生成 (ホストへの大量転送, ホストからのランダムアクセスを防ぐ)
	__global__ static void extract_elite_sample(input_array* src, input_array* dst, int* elite_indices){
		int id = blockDim.x * blockIdx.x + threadIdx.x;//blockDim.x = 1, threadIdx.x = 0

		dst[id].cost = src[elite_indices[id]].cost;
		for(int i=0; i< _DEVICE_CONST_HORIZON; i++){
			for(int j=0; j<4; j++){
				dst[id].decoupled_position[i][j] = src[elite_indices[id]].decoupled_position[i][j];
			}
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

    float mcmpc_controller::calc_weighted_average_and_min_cost(float var_and_z_i[])
	{
		// 0, 1, 2, ... となる昇順インデックスを生成
        thrust::sequence( indices_device_vec.begin(), indices_device_vec.end() );

		// cost_device_vec_for_sortingを昇順にインデックスをソート
        thrust::sort_by_key( cost_device_vec_for_sorting.begin(), cost_device_vec_for_sorting.end(), indices_device_vec.begin() );

		// input_array_device_vec 中からエリートサンプルのみをホストにコピー，input_array_host_vec_elite はソート済みの配列となる
        extract_elite_sample<<< CONST_PARAM::N_OF_THE_USING_BEST, 1 >>>( thrust::raw_pointer_cast( input_array_device_vec.data() ), thrust::raw_pointer_cast( input_array_device_vec_elite.data() ), thrust::raw_pointer_cast( indices_device_vec.data() ) );
        input_array_host_vec_elite = input_array_device_vec_elite;

		// コストを正規化するためのλを計算（expが0にならないための処理）
        float lambda = 0.0f;

        for ( int n = 0; n < CONST_PARAM::N_OF_THE_USING_BEST; n++ )
            lambda += input_array_host_vec_elite[n].cost;
        lambda /= (float)CONST_PARAM::N_OF_THE_USING_BEST;

		// 加重平均を計算
        float weight_temp;
        float sum_of_weight = 0.0f;

        for ( int i = 0; i < _DEVICE_CONST_HORIZON; i++ ){
            for ( int j = 0; j < 4; j++ )
                best_input_array.decoupled_position[i][j] = 0.0f;
        }
        for ( int n = 0; n < CONST_PARAM::N_OF_THE_USING_BEST; n++ )
        {
            weight_temp = exp( -input_array_host_vec_elite[n].cost / lambda );
            sum_of_weight += weight_temp;

            for ( int i = 0; i < _DEVICE_CONST_HORIZON; i++ )
                for ( int j = 0; j < 4; j++ )
                    best_input_array.decoupled_position[i][j] += input_array_host_vec_elite[n].decoupled_position[i][j] * weight_temp;
        }

        for ( int i = 0; i < _DEVICE_CONST_HORIZON; i++ )
            for ( int j = 0; j < 4; j++ )
                best_input_array.decoupled_position[i][j] /= sum_of_weight;

		//
        // 加重平均された入力列に対してコスト関数を再計算
        //
        best_input_array.cost = 0.0f;

        float var_and_z_i_temp[_N_OF_ODES + 1];
        for ( int i = 0; i < _N_OF_ODES + 1; i++ )
            var_and_z_i_temp[i] = var_and_z_i[i];

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
			// PIDカスケード用修正
            float x_ref   = best_input_array.decoupled_position[i][x];
            float y_ref   = best_input_array.decoupled_position[i][y];
            float z_ref   = best_input_array.decoupled_position[i][z];
            float yaw_ref = best_input_array.decoupled_position[i][yaw];
            float vel_ref[3];
            for ( float t = 0.0f; t < CONST_PARAM_FLOAT::CONTROL_PERIOD - CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE  / 2; t += CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE )
            {
				for ( int k = 0; k < _N_OF_ODES; k++ ) var_p_temp[k] = var_and_z_i_temp[k];
                /*目標速度*/
                vel_ref[0] = CONST_PARAM_FLOAT::MPC_XY_P * (x_ref - var_p_temp[7]);
                vel_ref[1] = CONST_PARAM_FLOAT::MPC_XY_P * (y_ref - var_p_temp[8]);
                vel_ref[2] = CONST_PARAM_FLOAT::MPC_Z_P  * (z_ref - var_p_temp[9]);
                /*目標加速度*/
                float vel_dot_x = (var_p_temp[10] - prev_vel[0]) / CONST_PARAM_FLOAT::CONTROL_PERIOD;
                float vel_dot_y = (var_p_temp[11] - prev_vel[1]) / CONST_PARAM_FLOAT::CONTROL_PERIOD;
                float vel_dot_z = (var_p_temp[12] - prev_vel[2]) / CONST_PARAM_FLOAT::CONTROL_PERIOD;
                // 力を考慮する場合 a_ref = a_ref_pid - Fcut/m
                float a_ref[3];
                a_ref[0] = CONST_PARAM_FLOAT::MPC_XY_VEL_P_ACC*(vel_ref[0]-var_p_temp[10])+CONST_PARAM_FLOAT::MPC_XY_VEL_I_ACC*vel_int[0]-CONST_PARAM_FLOAT::MPC_XY_VEL_D_ACC*(prev_acc[0]+CONST_PARAM_FLOAT::LPF*(vel_dot_x-prev_acc[0]));
                a_ref[1] = CONST_PARAM_FLOAT::MPC_XY_VEL_P_ACC*(vel_ref[1]-var_p_temp[11])+CONST_PARAM_FLOAT::MPC_XY_VEL_I_ACC*vel_int[1]-CONST_PARAM_FLOAT::MPC_XY_VEL_D_ACC*(prev_acc[1]+CONST_PARAM_FLOAT::LPF*(vel_dot_y-prev_acc[1]));
                a_ref[2] = CONST_PARAM_FLOAT::MPC_Z_VEL_P_ACC* (vel_ref[2]-var_p_temp[12])+CONST_PARAM_FLOAT::MPC_Z_VEL_I_ACC* vel_int[2]-CONST_PARAM_FLOAT::MPC_Z_VEL_D_ACC *(prev_acc[2]+CONST_PARAM_FLOAT::LPF*(vel_dot_z-prev_acc[2]));
                float inv_mass = 1.0f / CONST_PARAM_FLOAT::MASS_OF_MACHINE;
                // /* e0p */ var_and_z_i_temp[0]  += (-0.5f*var_p_temp[1]*var_p_temp[4] - 0.5f*var_p_temp[2]*var_p_temp[5] - 0.5f*var_p_temp[3]*var_p_temp[6])*CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;
                // /* e1p */ var_and_z_i_temp[1]  += ( 0.5f*var_p_temp[0]*var_p_temp[4] + 0.5f*var_p_temp[2]*var_p_temp[6] - 0.5f*var_p_temp[3]*var_p_temp[5])*CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;
                // /* e2p */ var_and_z_i_temp[2]  += ( 0.5f*var_p_temp[0]*var_p_temp[5] + 0.5f*var_p_temp[3]*var_p_temp[4] - 0.5f*var_p_temp[1]*var_p_temp[6])*CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;
                // /* e3p */ var_and_z_i_temp[3]  += ( 0.5f*var_p_temp[0]*var_p_temp[6] + 0.5f*var_p_temp[1]*var_p_temp[5] - 0.5f*var_p_temp[2]*var_p_temp[4])*CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;
                // // normalize quaternion
                // float q_norm_inv = rsqrtf(var_and_z_i_temp[0]*var_and_z_i_temp[0] +var_and_z_i_temp[1]*var_and_z_i_temp[1] +var_and_z_i_temp[2]*var_and_z_i_temp[2] +var_and_z_i_temp[3]*var_and_z_i_temp[3]);
                // var_and_z_i_temp[0] *= q_norm_inv;
                // var_and_z_i_temp[1] *= q_norm_inv;
                // var_and_z_i_temp[2] *= q_norm_inv;
                // var_and_z_i_temp[3] *= q_norm_inv;
                // /* wxp */ var_and_z_i_temp[4]   = ref_roll;
                // /* wyp */ var_and_z_i_temp[5]   = ref_pitch;
                // /* wzp */ var_and_z_i_temp[6]   = ref_yaw;

                /* xp  */ var_and_z_i_temp[7]  +=  var_p_temp[10] * CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;
                /* yp  */ var_and_z_i_temp[8]  +=  var_p_temp[11] * CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;
                /* zp  */ var_and_z_i_temp[9]  +=  var_p_temp[12] * CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;

                /* xpp */ var_and_z_i_temp[10] += a_ref[0] * CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;
                /* ypp */ var_and_z_i_temp[11] += a_ref[1] * CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;
                /* zpp */ var_and_z_i_temp[12] += a_ref[2] * CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;

                /* z_i */ var_and_z_i_temp[_N_OF_ODES] += var_and_z_i_temp[9] * CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;

                
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
			// コストの計算（0.5かけるのはACADOに合わせるため）
            best_input_array.cost += (_COST_Q_X*(var_and_z_i_temp[7] - target_host.x )*(var_and_z_i_temp[7] -target_host.x) + _COST_Q_Y *(var_and_z_i_temp[8] -target_host.y) *(var_and_z_i_temp[8] -target_host.y) + _COST_Q_Z *(var_and_z_i_temp[9] -target_host.z) *(var_and_z_i_temp[9] -target_host.z)     // x, y, z
                 +  _COST_Q_XP*(var_and_z_i_temp[10]-target_host.xp)*(var_and_z_i_temp[10]-target_host.xp)+ _COST_Q_YP*(var_and_z_i_temp[11]-target_host.yp)*(var_and_z_i_temp[11]-target_host.yp)+ _COST_Q_ZP*(var_and_z_i_temp[12]-target_host.zp)*(var_and_z_i_temp[12]-target_host.zp)    // xp, yp, zp
                 +  _COST_Q_E1*(var_and_z_i_temp[1] -target_host.e1)*(var_and_z_i_temp[1] -target_host.e1)+ _COST_Q_E2*(var_and_z_i_temp[2] -target_host.e2)*(var_and_z_i_temp[2] -target_host.e2)+ _COST_Q_E3*(var_and_z_i_temp[3] -target_host.e3)*(var_and_z_i_temp[3] -target_host.e3)         // e1, e2, e3
                 +  _COST_Q_WX*(var_and_z_i_temp[4] -target_host.wx)*(var_and_z_i_temp[4] -target_host.wx)+ _COST_Q_WY*(var_and_z_i_temp[5] -target_host.wy)*(var_and_z_i_temp[5] -target_host.wy)+ _COST_Q_WZ*(var_and_z_i_temp[6] -target_host.wz)*(var_and_z_i_temp[6] -target_host.wz)          // wx, wy, wz
                 +  _COST_Q_ZI*var_and_z_i_temp[_N_OF_ODES]*var_and_z_i_temp[_N_OF_ODES]                                                                                                  // z_i
                 +  _COST_R_X*(best_input_array.decoupled_position[i][x])*(best_input_array.decoupled_position[i][x])
                 +  _COST_R_Y*best_input_array.decoupled_position[i][y]*best_input_array.decoupled_position[i][y]
                 +  _COST_R_Z*best_input_array.decoupled_position[i][z]*best_input_array.decoupled_position[i][z]
                 +  _COST_R_YAW*best_input_array.decoupled_position[i][yaw]*best_input_array.decoupled_position[i][yaw]
            );
            
            vel_int[0] += (vel_ref[0]-var_and_z_i_temp[10])*CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;
            vel_int[1] += (vel_ref[1]-var_and_z_i_temp[11])*CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;
            vel_int[2] += (vel_ref[2]-var_and_z_i_temp[12])*CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE ;
            vel_int[2] = fminf(fmaxf(vel_int[2], -CONST_PARAM_FLOAT::A_OF_GRAVITY), CONST_PARAM_FLOAT::A_OF_GRAVITY);
        }
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

		// 準最適制御入力の計算
        for ( int k = 0; k < CONST_PARAM::ITERATION_TIMES; k++ )
        {
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


