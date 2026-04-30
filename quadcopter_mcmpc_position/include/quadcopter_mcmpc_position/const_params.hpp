#pragma once

#include <string>

//#define UNPREDICTABLE_IMPULSE
//#define UNPREDICTABLE_COLLISION_WITH_WALL
//#define PREDICTABLE_COLLISION_WITH_WALL
//#define MCMPC_WITH_FORCE_STATE

#define SIMULATION

#ifdef MCMPC_WITH_FORCE_STATE
    #define _N_OF_ODES          16 //[q, av, x, v, F]
#else
    #define _N_OF_ODES          13 // number of states [quaternion(4), angular velocity(3),position(3), velocity(3)]
#endif
#define _PI_FROAT		3.141592653589793f
#define _RAD_TO_DEG		180.0f / _PI_FROAT
#define _INV_SQRT_2		0.7071067811865475f // 1/sqrt(2)

// const for device (GPU)
#define _DEVICE_CONST_HORIZON 			50 // horizon
#define _DEVICE_CONST_THREAD_PER_BLOCK 	128 //_DEVICE_CONST_THREAD_PER_BLOCK * N_OF_BLOCK = N_OF_SAMPLES
#define _DEVICE_CONST_N_OF_BLOCK 		64

// cost for MPC
#define _COST_Q_X 		36.0f
#define _COST_Q_Y 		36.0f
#define _COST_Q_Z 		100.0f
#define _COST_Q_XP 		3.0f
#define _COST_Q_YP 		3.0f
#define _COST_Q_ZP 		5.0f
#define _COST_Q_E1 		0.01f
#define _COST_Q_E2 		0.01f
#define _COST_Q_E3 		0.001f
#define _COST_Q_WX 		10.0f
#define _COST_Q_WY 		10.0f
#define _COST_Q_WZ 		9.0f
#define _COST_Q_ZI 		0.0f
#define _COST_R_X 		0.0001f
#define _COST_R_Y 	    0.001f
#define _COST_R_Z 	    0.001f
#define _COST_R_YAW 	0.001f
#ifdef MCMPC_WITH_FORCE_STATE
    #define _COST_Q_FX  1.0f
    #define _COST_Q_FY  1.0f
    #define _COST_Q_FZ  1.0f
#endif

// const for host(CPU)

struct CONST_PARAM
{
    static const double U_G;//rps for hover
    static const double U_DIFF_LIM;
    static const double MAX_RPS;

    static const double A_OF_GRAVITY;
    static const double I_XX;
    static const double I_YY;
	static const double I_ZZ;
	static const double ROTOR_DISTANCE;
	static const double MAX_RPS_POW; // max rps^2 of the propeller
	static const double MAX_THRUST;
	static const double MASS_OF_MACHINE;
	static const double TORQUE_RATE;
    static const double LPF;

    static const double INIT_U_THRUST;

    static const double INIT_TARGET_E0;
	static const double INIT_TARGET_E1;
	static const double INIT_TARGET_E2;
	static const double INIT_TARGET_E3;
	static const double INIT_TARGET_WX;
	static const double INIT_TARGET_WY;
	static const double INIT_TARGET_WZ;
	static const double INIT_TARGET_X;
	static const double INIT_TARGET_Y;
	static const double INIT_TARGET_Z;
	static const double INIT_TARGET_XP;
	static const double INIT_TARGET_YP;
	static const double INIT_TARGET_ZP;

    static const double CONTROL_PERIOD;
	static const int 	N_OF_SAMPLES;
	static const int 	N_OF_THE_USING_BEST;
	static const int 	ITERATION_TIMES; // iteration times for MPC optimization

	static const double U_UPPER_LIM; // upper limit of input
	static const double U_LOWER_LIM; // lower limit of input

	static const std::string FILE_HEADER[11];

    // QGC setting
    static const double MPC_XY_P;
    static const double MPC_Z_P;
    static const double MPC_XY_VEL_P_ACC;
    static const double MPC_XY_VEL_I_ACC;
    static const double MPC_XY_VEL_D_ACC;
    static const double MPC_Z_VEL_P_ACC;
    static const double MPC_Z_VEL_I_ACC;
    static const double MPC_Z_VEL_D_ACC;
    static const double MC_YAW_WEIGHT;
    static const double MC_ROLL_P;
    static const double MC_PITCH_P;
    static const double MC_YAW_P;
    static const double MPC_THR_HOVER;
    static const double MPC_VEL_LP;
    static const double MPC_VELD_LP;
    static const double MC_ROLLRATE_P;
    static const double MC_PITCHRATE_P;
    static const double MC_YAWRATE_P;
    static const double MC_ROLLRATE_D;
    static const double MC_PITCHRATE_D;
    static const double MC_YAWRATE_D;
    static const double MC_ROLLRATE_I;
    static const double MC_PITCHRATE_I;
    static const double MC_YAWRATE_I;
    static const double CA_ROTOR0_KM;
    static const double CA_ROTOR1_KM;
    static const double CA_ROTOR2_KM;
    static const double CA_ROTOR3_KM;
    static const double CA_ROTOR_CT[4];

    static const double ARW_GAIN;
    
#ifdef UNPREDICTABLE_IMPULSE
	static const double IMPULSE_TIME;
	static const double IMPULSE_XP;
	static const double IMPLUSE_WY;
#endif

#if defined(UNPREDICTABLE_COLLISION_WITH_WALL) || defined(PREDICTABLE_COLLISION_WITH_WALL)
	static const double X_WALL;
	static const double WALL_NORMAL_VECTOR_X;
	static const double WALL_NORMAL_VECTOR_Y;
	static const double WALL_NORMAL_VECTOR_Z;
	static const double R_OF_RING;
	static const double COEFF_OF_REST;
#endif

#ifdef MCMPC_WITH_FORCE_STATE
    static const double INIT_TARGET_FX;
    static const double INIT_TARGET_FY;
    static const double INIT_TARGET_FZ;
#endif
};

struct CONST_PARAM_FLOAT
{
    static const float U_G;
    static const float U_UPPER_LIM;
    static const float U_LOWER_LIM;

    static const float A_OF_GRAVITY;
	static const float I_XX;
    static const float I_YY;
    static const float I_ZZ;
    static const float ROTOR_DISTANCE;
    static const float MAX_RPS_POW;
    static const float MAX_THRUST;
    static const float MASS_OF_MACHINE;
    static const float TORQUE_RATE;
    static const float LPF;

	static const float INIT_TARGET_E0;
	static const float INIT_TARGET_E1;
	static const float INIT_TARGET_E2;
	static const float INIT_TARGET_E3;
	static const float INIT_TARGET_WX;
	static const float INIT_TARGET_WY;
	static const float INIT_TARGET_WZ;
	static const float INIT_TARGET_X;
	static const float INIT_TARGET_Y;
	static const float INIT_TARGET_Z;
	static const float INIT_TARGET_XP;
	static const float INIT_TARGET_YP;
	static const float INIT_TARGET_ZP;

	static const float CONTROL_PERIOD;
    static const float INTEGRATION_STEP_SIZE;

    static const float SIGMA_CONST[4];

    // QGC setting
    static const float MPC_XY_P;
    static const float MPC_Z_P;
    static const float MPC_XY_VEL_P_ACC;
    static const float MPC_XY_VEL_I_ACC;
    static const float MPC_XY_VEL_D_ACC;
    static const float MPC_Z_VEL_P_ACC;
    static const float MPC_Z_VEL_I_ACC;
    static const float MPC_Z_VEL_D_ACC;
    static const float MC_YAW_WEIGHT;
    static const float MC_ROLL_P;
    static const float MC_PITCH_P;
    static const float MC_YAW_P;
    static const float MPC_THR_HOVER;
    static const float MPC_VEL_LP;
    static const float MPC_VELD_LP;
    static const float MC_ROLLRATE_P;
    static const float MC_PITCHRATE_P;
    static const float MC_YAWRATE_P;
    static const float MC_ROLLRATE_D;
    static const float MC_PITCHRATE_D;
    static const float MC_YAWRATE_D;
    static const float MC_ROLLRATE_I;
    static const float MC_PITCHRATE_I;
    static const float MC_YAWRATE_I;
    static const float CA_ROTOR_KM[4];
    static const float CA_ROTOR_CT[4];

    static const float ARW_GAIN;
    
#ifdef PREDICTABLE_COLLISION_WITH_WALL
    static const float X_WALL;
    static const float WALL_NORMAL_VECTOR_X;
    static const float WALL_NORMAL_VECTOR_Y;
    static const float WALL_NORMAL_VECTOR_Z;
    static const float R_OF_RING;
    static const float COEFF_OF_REST;
#endif
#ifdef MCMPC_WITH_FORCE_STATE
    static const float INIT_TARGET_FX;
    static const float INIT_TARGET_FY;
    static const float INIT_TARGET_FZ;
#endif
};
