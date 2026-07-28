#pragma once

#include <string>
#include <cmath>

//#define UNPREDICTABLE_IMPULSE
//#define UNPREDICTABLE_COLLISION_WITH_WALL
#ifndef MCMPC_DISABLE_WALL_MODEL
#define PREDICTABLE_COLLISION_WITH_WALL
#endif
//#define MCMPC_WITH_FORCE_STATE

//#define SIMULATION

#ifdef MCMPC_WITH_FORCE_STATE
    #define _N_OF_ODES          16 //[q, av, x, v, F]
#else
    #define _N_OF_ODES          13 // number of states [quaternion(4), angular velocity(3),position(3), velocity(3)]
#endif
#define _PI_FROAT		3.141592653589793f
#define _RAD_TO_DEG		180.0f / _PI_FROAT
#define _INV_SQRT_2		0.7071067811865475f // 1/sqrt(2)

// const for device (GPU)
#define _DEVICE_CONST_HORIZON 			70 // horizon
#define _DEVICE_CONST_THREAD_PER_BLOCK 	128 //_DEVICE_CONST_THREAD_PER_BLOCK * N_OF_BLOCK = N_OF_SAMPLES
#define _DEVICE_CONST_N_OF_BLOCK 		64

#define _SQUARE_WAYPOINTS               32 // runtime-configurable maximum
#define _SQUARE_WAYPOINT_THRESHOLD      0.05f // fallback default
#define _SQUARE_WAYPOINT_HOLD_SEC       0.0f
#define _WAYPOINT_CRUISE_SPEED          0.50f // m/s, zero only at the final waypoint
#define _GUARD_ARC_ANGLE_RAD              (92.1f * _PI_FROAT / 180.0f)

// cost for MPC
#define _COST_Q_X 		4.0f
#define _COST_Q_Y 		4.0f
#define _COST_Q_Z 		6.0f
#define _COST_Q_XP 		3.0f
#define _COST_Q_YP 		3.0f
#define _COST_Q_ZP 		5.0f
#define _COST_Q_E1 		0.0f
#define _COST_Q_E2 		0.0f
#define _COST_Q_E3 		0.0f
#define _COST_Q_WX 		0.0f
#define _COST_Q_WY 		0.0f
#define _COST_Q_WZ 		0.0f
#define _COST_Q_ZI 		0.0f
#define _COST_R_X 		0.25f
#define _COST_R_Y 	    0.25f
#define _COST_R_Z 	    0.50f
#define _COST_R_YAW 	0.05f
#define _COST_TERMINAL_X 	120.0f
#define _COST_TERMINAL_Y 	120.0f
#define _COST_TERMINAL_Z 	140.0f
#define _COST_TERMINAL_VX 	40.0f
#define _COST_TERMINAL_VY 	40.0f
#define _COST_TERMINAL_VZ 	50.0f
// A zero move penalty lets the independently sampled horizon points chatter.
// Keep it large enough to make the position-setpoint sequence executable by
// the cascaded PX4 controller, while leaving the stage tracking term dominant.
#define _COST_DU_X 		2.0f
#define _COST_DU_Y 		2.0f
#define _COST_DU_Z 		4.0f
#define _COST_DU_YAW 	0.25f
// The stage reference moves along the waypoint path.  Anchor the last
// position-setpoint explicitly to the active waypoint as well as anchoring the
// predicted vehicle state there.
#define _COST_TERMINAL_U_X 	40.0f
#define _COST_TERMINAL_U_Y 	40.0f
#define _COST_TERMINAL_U_Z 	60.0f
#define _COST_TERMINAL_U_YAW 2.0f
#define _COST_CONTACT_POSITION  1.0f
#define _COST_CONTACT_VELOCITY  0.5f
#define _COST_EXIT_DIRECTION    2.0f
#define _CONTACT_APPROACH_SPEED 0.30f
#ifdef MCMPC_WITH_FORCE_STATE
    #define _COST_Q_FX  1.0f
    #define _COST_Q_FY  1.0f
    #define _COST_Q_FZ  1.0f
#endif

// const for host(CPU)

extern int square_waypoint_index;
extern float square_waypoint_change_time;
extern float mcmpc_log;
extern int square_waypoint_count;
extern float square_waypoint_threshold;

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

    // motor dynamics
    static const double MOTOR_INPUT_SCALING;
    static const double MAX_ROT_VELOCITY;
    static const double MOTOR_TIME_CONSTANT_UP;
    static const double MOTOR_TIME_CONSTANT_DOWN;
    static const double MOTOR_THRUST_CONSTANT_SDF;
    static const double MOTOR_THRUST_SCALE;
    static const double MOTOR_THRUST_CONSTANT;
    static const double MOMENT_CONSTANT;

    // rotor / allocator
    static const double ROTOR_POSITIONS[4][3];
    static const double ROTOR_YAW_SIGNS[4];
    static const double PX4_QUAD_X_MIX[4][4];
    static const double PX4_QUAD_X_MIX_INV[4][4];
    static const double PX4_ACTUATOR_MIN[4];
    static const double PX4_ACTUATOR_MAX[4];
    static const double CA_MINIMUM_YAW_MARGIN;

    // vehicle dynamics
    static const double LINEAR_VELOCITY_DAMPING[3];
    static const double ANGULAR_VELOCITY_DAMPING[3];
    static const double BODY_TORQUE_SCALE[3];

    // QGC setting
    // position /thrust control
    static const double MPC_TILT_MAX;
    static const double MPC_XY_P;
    static const double MPC_Z_P;
    static const double MPC_XY_VEL_P_ACC;
    static const double MPC_XY_VEL_I_ACC;
    static const double MPC_XY_VEL_D_ACC;
    static const double MPC_Z_VEL_P_ACC;
    static const double MPC_Z_VEL_I_ACC;
    static const double MPC_Z_VEL_D_ACC;
    static const double MPC_XY_VEL_MAX;
    static const double MPC_Z_VEL_MAX_UP;
    static const double MPC_Z_VEL_MAX_DOWN;
    static const double MPC_THR_HOVER;
    static const double MPC_THR_MIN;
    static const double MPC_THR_MAX;
    static const double MPC_THR_XY_MARGIN;
    static const double MPC_VELD_LP;
    // attitude / rate control
    static const double MC_ROLL_P;
    static const double MC_PITCH_P;
    static const double MC_YAW_P;
    static const double MC_YAW_WEIGHT;
    static const double MC_ROLLRATE_MAX;
    static const double MC_PITCHRATE_MAX;
    static const double MC_YAWRATE_MAX;
    static const double ANGULAR_ACCEL_LP;
    static const double MC_ROLLRATE_P;
    static const double MC_PITCHRATE_P;
    static const double MC_YAWRATE_P;
    static const double MC_ROLLRATE_K;
    static const double MC_PITCHRATE_K;
    static const double MC_YAWRATE_K;
    static const double MC_ROLLRATE_D;
    static const double MC_PITCHRATE_D;
    static const double MC_YAWRATE_D;
    static const double MC_ROLLRATE_I;
    static const double MC_PITCHRATE_I;
    static const double MC_YAWRATE_I;
    static const double MC_ROLLRATE_FF;
    static const double MC_PITCHRATE_FF;
    static const double MC_YAWRATE_FF;
    static const double MC_RR_INT_LIM;
    static const double MC_PR_INT_LIM;
    static const double MC_YR_INT_LIM;
    static const double MC_YAW_TQ_CUTOFF;

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

    static const bool MPC_ACC_DECOUPLE;
    static const int MOTOR_COMMAND_DELAY_STEPS;
    static const int TAKEOFF_STATE_RAMPUP;
    static const int TAKEOFF_STATE_FLIGHT;
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

    // motor model
    static const float MOTOR_INPUT_SCALING;
    static const float MAX_ROT_VELOCITY;
    static const float MOTOR_TIME_CONSTANT_UP;
    static const float MOTOR_TIME_CONSTANT_DOWN;
    static const float MOTOR_THRUST_CONSTANT_SDF;
    static const float MOTOR_THRUST_SCALE;
    static const float MOTOR_THRUST_CONSTANT;
    static const float MOMENT_CONSTANT;

    // rotor / allocator
    static const float ROTOR_POSITIONS[4][3];
    static const float ROTOR_YAW_SIGNS[4];
    static const float PX4_QUAD_X_MIX[4][4];
    static const float PX4_QUAD_X_MIX_INV[4][4];
    static const float PX4_ACTUATOR_MIN[4];
    static const float PX4_ACTUATOR_MAX[4];
    static const float CA_MINIMUM_YAW_MARGIN;

    // vehicle dynamics
    static const float LINEAR_VELOCITY_DAMPING[3];
    static const float ANGULAR_VELOCITY_DAMPING[3];
    static const float BODY_TORQUE_SCALE[3];

    // QGC setting
    // position / thrust control
    static const float MPC_TILT_MAX;
    static const float MPC_XY_P;
    static const float MPC_Z_P;
    static const float MPC_XY_VEL_P_ACC;
    static const float MPC_XY_VEL_I_ACC;
    static const float MPC_XY_VEL_D_ACC;
    static const float MPC_Z_VEL_P_ACC;
    static const float MPC_Z_VEL_I_ACC;
    static const float MPC_Z_VEL_D_ACC;
    static const float MPC_XY_VEL_MAX;
    static const float MPC_Z_VEL_MAX_UP;
    static const float MPC_Z_VEL_MAX_DOWN;
    static const float MPC_THR_HOVER;
    static const float MPC_THR_MIN;
    static const float MPC_THR_MAX;
    static const float MPC_THR_XY_MARGIN;
    static const float MPC_VELD_LP;

    // attitude / rate control
    static const float MC_ROLL_P;
    static const float MC_PITCH_P;
    static const float MC_YAW_P;
    static const float MC_YAW_WEIGHT;
    static const float MC_ROLLRATE_MAX;
    static const float MC_PITCHRATE_MAX;
    static const float MC_YAWRATE_MAX;
    static const float ANGULAR_ACCEL_LP;
    static const float MC_ROLLRATE_P;
    static const float MC_PITCHRATE_P;
    static const float MC_YAWRATE_P;
    static const float MC_ROLLRATE_K;
    static const float MC_PITCHRATE_K;
    static const float MC_YAWRATE_K;
    static const float MC_ROLLRATE_D;
    static const float MC_PITCHRATE_D;
    static const float MC_YAWRATE_D;
    static const float MC_ROLLRATE_I;
    static const float MC_PITCHRATE_I;
    static const float MC_YAWRATE_I;
    static const float MC_ROLLRATE_FF;
    static const float MC_PITCHRATE_FF;
    static const float MC_YAWRATE_FF;
    static const float MC_RR_INT_LIM;
    static const float MC_PR_INT_LIM;
    static const float MC_YR_INT_LIM;
    static const float MC_YAW_TQ_CUTOFF;

    static const float ARW_GAIN;

    static const float square_waypoints[_SQUARE_WAYPOINTS][3];

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

    // model switches
    static const bool MPC_ACC_DECOUPLE;
    static const int MOTOR_COMMAND_DELAY_STEPS;

    // takeoff state
    static const int TAKEOFF_STATE_RAMPUP;
    static const int TAKEOFF_STATE_FLIGHT;
};
