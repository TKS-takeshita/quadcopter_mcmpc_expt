#include "quadcopter_mcmpc_position/const_params.hpp"

// double for host programs

const double CONST_PARAM::U_G        = 113.0;//rps for hover
const double CONST_PARAM::U_DIFF_LIM = 60.0;
const double CONST_PARAM::MAX_RPS    = CONST_PARAM::U_G + CONST_PARAM::U_DIFF_LIM; // max rps of the propeller

const double CONST_PARAM::A_OF_GRAVITY    = 9.80665;    // m/s^2
const double CONST_PARAM::I_XX            = 0.04418; //kg・m^2　from CAD
const double CONST_PARAM::I_YY            = 0.04226;
const double CONST_PARAM::I_ZZ            = 0.05619;
const double CONST_PARAM::ROTOR_DISTANCE  = 0.50; //m 向かい合うロータ間の距離
const double CONST_PARAM::MASS_OF_MACHINE = 2.35; // kg
const double CONST_PARAM::MAX_RPS_POW     = CONST_PARAM::MAX_RPS * CONST_PARAM::MAX_RPS;

const double CONST_PARAM::MAX_THRUST      = 53.6;      // N
const double CONST_PARAM::TORQUE_RATE     = 0.19 / CONST_PARAM::MAX_RPS / CONST_PARAM::MAX_RPS;      // N・m

const double CONST_PARAM::INIT_U_THRUST   = 0.0;

const double CONST_PARAM::INIT_TARGET_E0  = 1.0;
const double CONST_PARAM::INIT_TARGET_E1  = 0.0;
const double CONST_PARAM::INIT_TARGET_E2  = 0.0;
const double CONST_PARAM::INIT_TARGET_E3  = 0.0;
const double CONST_PARAM::INIT_TARGET_WX  = 0.0;
const double CONST_PARAM::INIT_TARGET_WY  = 0.0;
const double CONST_PARAM::INIT_TARGET_WZ  = 0.0;
const double CONST_PARAM::INIT_TARGET_X   = 1.0;
const double CONST_PARAM::INIT_TARGET_Y   = 0.0;
const double CONST_PARAM::INIT_TARGET_Z   = -1.2;
const double CONST_PARAM::INIT_TARGET_XP  = 0.0;
const double CONST_PARAM::INIT_TARGET_YP  = 0.0;
const double CONST_PARAM::INIT_TARGET_ZP  = 0.0;

const double CONST_PARAM::CONTROL_PERIOD      = 0.05;   // s
const int    CONST_PARAM::N_OF_SAMPLES        = _DEVICE_CONST_THREAD_PER_BLOCK * _DEVICE_CONST_N_OF_BLOCK;
const int    CONST_PARAM::N_OF_THE_USING_BEST = 10;//100
const int    CONST_PARAM::ITERATION_TIMES     = 2;
const double CONST_PARAM::U_UPPER_LIM         = CONST_PARAM::U_G + CONST_PARAM::U_DIFF_LIM; // no units
const double CONST_PARAM::U_LOWER_LIM         = CONST_PARAM::U_G - CONST_PARAM::U_DIFF_LIM; // no units


#ifdef SIMULATION
    const double CONST_PARAM::MPC_XY_P         = 0.30;
    const double CONST_PARAM::MPC_Z_P          = 1.00;
    const double CONST_PARAM::MPC_XY_VEL_P_ACC = 1.80;
    const double CONST_PARAM::MPC_XY_VEL_I_ACC = 0.40;
    const double CONST_PARAM::MPC_XY_VEL_D_ACC = 0.20;
    const double CONST_PARAM::MPC_Z_VEL_P_ACC  = 4.00;
    const double CONST_PARAM::MPC_Z_VEL_I_ACC  = 2.00;
    const double CONST_PARAM::MPC_Z_VEL_D_ACC  = 0.00;
    const double CONST_PARAM::MC_YAW_WEIGHT    = 0.40;
    const double CONST_PARAM::MC_ROLL_P        = 4.00;
    const double CONST_PARAM::MC_PITCH_P       = 4.00;
    const double CONST_PARAM::MC_YAW_P         = 2.80;
    const double CONST_PARAM::MPC_THR_HOVER    = 0.43 * CONST_PARAM::MAX_THRUST;// [N]
    const double CONST_PARAM::MPC_VEL_LP       = 0.0;// velocity derivative low pass cutoff frequency[Hz]
    const double CONST_PARAM::MPC_VELD_LP      = 5.0;// velocity derivative low pass cutoff frequency[Hz]
    const double CONST_PARAM::MC_ROLLRATE_P    = 0.15;
    const double CONST_PARAM::MC_PITCHRATE_P   = 0.15;
    const double CONST_PARAM::MC_YAWRATE_P     = 0.20;
    const double CONST_PARAM::MC_ROLLRATE_D    = 0.0030;
    const double CONST_PARAM::MC_PITCHRATE_D   = 0.0030;
    const double CONST_PARAM::MC_YAWRATE_D     = 0.00;
    const double CONST_PARAM::MC_ROLLRATE_I    = 0.00;
    const double CONST_PARAM::MC_PITCHRATE_I   = 0.00;
    const double CONST_PARAM::MC_YAWRATE_I     = 0.00;
    const double CONST_PARAM::CA_ROTOR_KM[4]     = {0.05, 0.05, -0.05, -0.05}; // N・m・s^2
    const double CONST_PARAM::CA_ROTOR_CT[4]     = {6.5, 6.5, 6.5, 6.5}; // N・s^2
#else
    const double CONST_PARAM::MPC_XY_P         = 0.95;
    const double CONST_PARAM::MPC_Z_P          = 1.00;
    const double CONST_PARAM::MPC_XY_VEL_P_ACC = 1.80;
    const double CONST_PARAM::MPC_XY_VEL_I_ACC = 0.40;
    const double CONST_PARAM::MPC_XY_VEL_D_ACC = 0.20;
    const double CONST_PARAM::MPC_Z_VEL_P_ACC  = 4.00;
    const double CONST_PARAM::MPC_Z_VEL_I_ACC  = 2.00;
    const double CONST_PARAM::MPC_Z_VEL_D_ACC  = 0.00;
    const double CONST_PARAM::MC_YAW_WEIGHT    = 0.40;
    const double CONST_PARAM::MC_ROLL_P        = 6.50;
    const double CONST_PARAM::MC_PITCH_P       = 6.50;
    const double CONST_PARAM::MC_YAW_P         = 2.80;
    const double CONST_PARAM::MPC_THR_HOVER    = 0.65 *  CONST_PARAM::MAX_THRUST;// [N]
    const double CONST_PARAM::MPC_VEL_LP       = 0.0;// velocity derivative low pass cutoff frequency[Hz]
    const double CONST_PARAM::MPC_VELD_LP      = 5.0;// velocity derivative low pass cutoff frequency[Hz]
    const double CONST_PARAM::MC_ROLLRATE_P    = 0.15;
    const double CONST_PARAM::MC_PITCHRATE_P   = 0.15;
    const double CONST_PARAM::MC_YAWRATE_P     = 0.20;
    const double CONST_PARAM::MC_ROLLRATE_D    = 0.0030;
    const double CONST_PARAM::MC_PITCHRATE_D   = 0.0030;
    const double CONST_PARAM::MC_YAWRATE_D     = 0.00;
    const double CONST_PARAM::MC_ROLLRATE_I    = 0.00;
    const double CONST_PARAM::MC_PITCHRATE_I   = 0.00;
    const double CONST_PARAM::MC_YAWRATE_I     = 0.00;
    const double CONST_PARAM::CA_ROTOR_KM[4]     = {0.05, 0.05, -0.05, -0.05}; // N・m・s^2
    const double CONST_PARAM::CA_ROTOR_CT[4]     = {6.5, 6.5, 6.5, 6.5}; // N・s^2
#endif

const double CONST_PARAM::LPF              = (2*M_PI*CONST_PARAM::MPC_VELD_LP)/(1+2*M_PI*CONST_PARAM::MPC_VELD_LP);
const double CONST_PARAM::ARW_GAIN         = 2.0/MPC_XY_VEL_P_ACC;

#ifdef UNPREDICTABLE_IMPULSE
    const double CONST_PARAM::IMPULSE_TIME = 3.0;               // s
    const double CONST_PARAM::IMPULSE_XP   = 1.0;               // m/s
    const double CONST_PARAM::IMPLUSE_WY   = 2 * 3.14159265359; // ras/s
#endif

#if defined(UNPREDICTABLE_COLLISION_WITH_WALL) || defined(PREDICTABLE_COLLISION_WITH_WALL)
    const double CONST_PARAM::X_WALL               = -1.0;  // m
    const double CONST_PARAM::WALL_NORMAL_VECTOR_X = 1.0;   // no units
    const double CONST_PARAM::WALL_NORMAL_VECTOR_Y = 0.0;   // no units
    const double CONST_PARAM::WALL_NORMAL_VECTOR_Z = 0.0;   // no units
    const double CONST_PARAM::R_OF_RING            = 0.135; // m
    const double CONST_PARAM::COEFF_OF_REST        = 0.5;   // coefficient of restitution
#endif

const std::string CONST_PARAM::FILE_HEADER[11] = {"% FILE: quadcopter_mcmpc.1\n%\n"
                                                  "%       t           RpsCW1         RpsCW2         RpsCCW1        RpsCCW2\n"
                                                  "%     (sec)        (NoUnits)      (NoUnits)      (NoUnits)      (NoUnits)\n\n",
                                                  "% FILE: quadcopter_mcmpc.2\n%\n"
                                                  "%       t           theta_x        theta_y        theta_z\n"
                                                  "%     (sec)        (degrees)      (degrees)      (degrees)\n\n",
                                                  "% FILE: quadcopter_mcmpc.3\n%\n"
                                                  "%       t              x              y              z\n"
                                                  "%     (sec)           (m)            (m)            (m)\n\n",
                                                  "% FILE: quadcopter_mcmpc.4\n%\n"
                                                  "%       t        P_EARTHo_DRONEcm[1] P_EARTHo_DRONEcm[2] P_EARTHo_DRONEcm[3] EARTH_DRONE[1,1] EARTH_DRONE[1,2] EARTH_DRONE[1,3] EARTH_DRONE[2,1] EARTH_DRONE[2,2] EARTH_DRONE[2,3] EARTH_DRONE[3,1] EARTH_DRONE[3,2] EARTH_DRONE[3,3]\n"
                                                  "%   (second)           (meter)             (meter)             (meter)          (NoUnits)        (NoUnits)        (NoUnits)        (NoUnits)        (NoUnits)        (NoUnits)        (NoUnits)        (NoUnits)        (NoUnits)\n\n",
                                                  "% FILE: quadcopter_mcmpc.5\n%\n"
                                                  "%       t        P_EARTHo_RoterCCW1[1] P_EARTHo_RoterCCW1[2] P_EARTHo_RoterCCW1[3]\n"
                                                  "%   (second)            (meter)               (meter)               (meter)\n\n",
                                                  "% FILE: quadcopter_mcmpc.6\n%\n"
                                                  "%       t        P_EARTHo_RoterCCW2[1] P_EARTHo_RoterCCW2[2] P_EARTHo_RoterCCW2[3]\n"
                                                  "%   (second)            (meter)               (meter)               (meter)\n\n",
                                                  "% FILE: quadcopter_mcmpc.7\n%\n"
                                                  "%       t       P_EARTHo_RoterCW1[1] P_EARTHo_RoterCW1[2] P_EARTHo_RoterCW1[3]\n"
                                                  "%   (second)           (meter)              (meter)              (meter)\n\n",
                                                  "% FILE: quadcopter_mcmpc.8\n%\n"
                                                  "%       t       P_EARTHo_RoterCW2[1] P_EARTHo_RoterCW2[2] P_EARTHo_RoterCW2[3]\n"
                                                  "%   (second)           (meter)              (meter)              (meter)\n\n",
                                                  "% FILE: quadcopter_mcmpc.9\n%\n"
                                                  "%       t               Cost\n"
                                                  "%   (second)            ( - )\n\n",
                                                  "% FILE: quadcopter_mcmpc.10\n%\n"
                                                  "%       t             x_p            y_p            z_p\n"
                                                  "%     (sec)          (m/s)          (m/s)          (m/s)\n\n",
                                                  "% FILE: quadcopter_mcmpc.11\n%\n"
                                                  "%       t             w_x            w_y            w_z\n"
                                                  "%     (sec)         (rad/s)        (rad/s)        (rad/s)\n\n"
                                                };

#ifdef MCMPC_WITH_FORCE_STATE
    const double CONST_PARAM::INIT_TARGET_FX  = 0.0;
    const double CONST_PARAM::INIT_TARGET_FY  = 0.0;
    const double CONST_PARAM::INIT_TARGET_FZ  = 0.0;
#endif

// float for host programs
const float CONST_PARAM_FLOAT::U_G             = (float)CONST_PARAM::U_G;
const float CONST_PARAM_FLOAT::U_UPPER_LIM     = (float)CONST_PARAM::U_UPPER_LIM;
const float CONST_PARAM_FLOAT::U_LOWER_LIM     = (float)CONST_PARAM::U_LOWER_LIM;

const float CONST_PARAM_FLOAT::I_XX            = (float)CONST_PARAM::I_XX;
const float CONST_PARAM_FLOAT::I_YY            = (float)CONST_PARAM::I_YY;
const float CONST_PARAM_FLOAT::I_ZZ            = (float)CONST_PARAM::I_ZZ;
const float CONST_PARAM_FLOAT::ROTOR_DISTANCE  = (float)CONST_PARAM::ROTOR_DISTANCE;
const float CONST_PARAM_FLOAT::MAX_RPS_POW     = (float)CONST_PARAM::MAX_RPS_POW;
const float CONST_PARAM_FLOAT::MAX_THRUST      = (float)CONST_PARAM::MAX_THRUST;
const float CONST_PARAM_FLOAT::MASS_OF_MACHINE = (float)CONST_PARAM::MASS_OF_MACHINE;
const float CONST_PARAM_FLOAT::TORQUE_RATE     = (float)CONST_PARAM::TORQUE_RATE;
const float CONST_PARAM_FLOAT::A_OF_GRAVITY    = (float)CONST_PARAM::A_OF_GRAVITY;

const float CONST_PARAM_FLOAT::INIT_TARGET_E0  = (float)CONST_PARAM::INIT_TARGET_E0;
const float CONST_PARAM_FLOAT::INIT_TARGET_E1  = (float)CONST_PARAM::INIT_TARGET_E1;
const float CONST_PARAM_FLOAT::INIT_TARGET_E2  = (float)CONST_PARAM::INIT_TARGET_E2;
const float CONST_PARAM_FLOAT::INIT_TARGET_E3  = (float)CONST_PARAM::INIT_TARGET_E3;
const float CONST_PARAM_FLOAT::INIT_TARGET_WX  = (float)CONST_PARAM::INIT_TARGET_WX;
const float CONST_PARAM_FLOAT::INIT_TARGET_WY  = (float)CONST_PARAM::INIT_TARGET_WY;
const float CONST_PARAM_FLOAT::INIT_TARGET_WZ  = (float)CONST_PARAM::INIT_TARGET_WZ;
const float CONST_PARAM_FLOAT::INIT_TARGET_X   = (float)CONST_PARAM::INIT_TARGET_X;
const float CONST_PARAM_FLOAT::INIT_TARGET_Y   = (float)CONST_PARAM::INIT_TARGET_Y;
const float CONST_PARAM_FLOAT::INIT_TARGET_Z   = (float)CONST_PARAM::INIT_TARGET_Z;
const float CONST_PARAM_FLOAT::INIT_TARGET_XP  = (float)CONST_PARAM::INIT_TARGET_XP;
const float CONST_PARAM_FLOAT::INIT_TARGET_YP  = (float)CONST_PARAM::INIT_TARGET_YP;
const float CONST_PARAM_FLOAT::INIT_TARGET_ZP  = (float)CONST_PARAM::INIT_TARGET_ZP;

const float CONST_PARAM_FLOAT::MPC_XY_P         = (float)CONST_PARAM::MPC_XY_P;
const float CONST_PARAM_FLOAT::MPC_Z_P          = (float)CONST_PARAM::MPC_Z_P;
const float CONST_PARAM_FLOAT::MPC_XY_VEL_P_ACC = (float)CONST_PARAM::MPC_XY_VEL_P_ACC;
const float CONST_PARAM_FLOAT::MPC_XY_VEL_I_ACC = (float)CONST_PARAM::MPC_XY_VEL_I_ACC;
const float CONST_PARAM_FLOAT::MPC_XY_VEL_D_ACC = (float)CONST_PARAM::MPC_XY_VEL_D_ACC;
const float CONST_PARAM_FLOAT::MPC_Z_VEL_P_ACC  = (float)CONST_PARAM::MPC_Z_VEL_P_ACC;
const float CONST_PARAM_FLOAT::MPC_Z_VEL_I_ACC  = (float)CONST_PARAM::MPC_Z_VEL_I_ACC;
const float CONST_PARAM_FLOAT::MPC_Z_VEL_D_ACC  = (float)CONST_PARAM::MPC_Z_VEL_D_ACC;
const float CONST_PARAM_FLOAT::MC_YAW_WEIGHT    = (float)CONST_PARAM::MC_YAW_WEIGHT;
const float CONST_PARAM_FLOAT::MC_ROLL_P        = (float)CONST_PARAM::MC_ROLL_P;
const float CONST_PARAM_FLOAT::MC_PITCH_P       = (float)CONST_PARAM::MC_PITCH_P;
const float CONST_PARAM_FLOAT::MC_YAW_P         = (float)CONST_PARAM::MC_YAW_P;
const float CONST_PARAM_FLOAT::MPC_THR_HOVER    = (float)CONST_PARAM::MPC_THR_HOVER;// rate [%]
const float CONST_PARAM_FLOAT::MPC_VEL_LP       = (float)CONST_PARAM::MPC_VEL_LP;// velocity derivative low pass cutoff frequency[Hz]
const float CONST_PARAM_FLOAT::MPC_VELD_LP      = (float)CONST_PARAM::MPC_VELD_LP;// velocity derivative low pass cutoff frequency[Hz]
const float CONST_PARAM_FLOAT::MC_ROLLRATE_P    = (float)CONST_PARAM::MC_ROLLRATE_P;
const float CONST_PARAM_FLOAT::MC_PITCHRATE_P   = (float)CONST_PARAM::MC_PITCHRATE_P;
const float CONST_PARAM_FLOAT::MC_YAWRATE_P     = (float)CONST_PARAM::MC_YAWRATE_P;
const float CONST_PARAM_FLOAT::MC_ROLLRATE_D    = (float)CONST_PARAM::MC_ROLLRATE_D;
const float CONST_PARAM_FLOAT::MC_PITCHRATE_D   = (float)CONST_PARAM::MC_PITCHRATE_D;
const float CONST_PARAM_FLOAT::MC_YAWRATE_D     = (float)CONST_PARAM::MC_YAWRATE_D;
const float CONST_PARAM_FLOAT::MC_ROLLRATE_I    = (float)CONST_PARAM::MC_ROLLRATE_I;
const float CONST_PARAM_FLOAT::MC_PITCHRATE_I   = (float)CONST_PARAM::MC_PITCHRATE_I;
const float CONST_PARAM_FLOAT::MC_YAWRATE_I     = (float)CONST_PARAM::MC_YAWRATE_I;
const float CONST_PARAM_FLOAT::CA_ROTOR_KM[4]   = {(float)CONST_PARAM::CA_ROTOR_KM[0], (float)CONST_PARAM::CA_ROTOR_KM[1], (float)CONST_PARAM::CA_ROTOR_KM[2], (float)CONST_PARAM::CA_ROTOR_KM[3]}; // N・m・s^2
const float CONST_PARAM_FLOAT::CA_ROTOR_CT[4]   = {(float)CONST_PARAM::CA_ROTOR_CT[0], (float)CONST_PARAM::CA_ROTOR_CT[1], (float)CONST_PARAM::CA_ROTOR_CT[2], (float)CONST_PARAM::CA_ROTOR_CT[3]}; // N・s^2

const float CONST_PARAM_FLOAT::LPF              = (float)CONST_PARAM::LPF;
const float CONST_PARAM_FLOAT::ARW_GAIN         = (float)CONST_PARAM::ARW_GAIN;

#ifdef PREDICTABLE_COLLISION_WITH_WALL
const float CONST_PARAM_FLOAT::X_WALL               = (float)CONST_PARAM::X_WALL;
const float CONST_PARAM_FLOAT::WALL_NORMAL_VECTOR_X = (float)CONST_PARAM::WALL_NORMAL_VECTOR_X;
const float CONST_PARAM_FLOAT::WALL_NORMAL_VECTOR_Y = (float)CONST_PARAM::WALL_NORMAL_VECTOR_Y;
const float CONST_PARAM_FLOAT::WALL_NORMAL_VECTOR_Z = (float)CONST_PARAM::WALL_NORMAL_VECTOR_Z;
const float CONST_PARAM_FLOAT::R_OF_RING            = (float)CONST_PARAM::R_OF_RING;
const float CONST_PARAM_FLOAT::COEFF_OF_REST        = (float)CONST_PARAM::COEFF_OF_REST;
#endif

const float CONST_PARAM_FLOAT::CONTROL_PERIOD        = (float)CONST_PARAM::CONTROL_PERIOD;
const float CONST_PARAM_FLOAT::INTEGRATION_STEP_SIZE = (float)CONST_PARAM::CONTROL_PERIOD / 2.0f;   // Set it to the 1/N value of CONTROL_PERIOD 

const float CONST_PARAM_FLOAT::SIGMA_CONST[4] = {0.01f, 0.01f, 0.01f, 0.001f};

#ifdef MCMPC_WITH_FORCE_STATE
    const float CONST_PARAM_FLOAT::INIT_TARGET_FX  = (float)CONST_PARAM::INIT_TARGET_FX;
    const float CONST_PARAM_FLOAT::INIT_TARGET_FY  = (float)CONST_PARAM::INIT_TARGET_FY;
    const float CONST_PARAM_FLOAT::INIT_TARGET_FZ  = (float)CONST_PARAM::INIT_TARGET_FZ;
#endif