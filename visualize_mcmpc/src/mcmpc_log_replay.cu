#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "quadcopter_mcmpc_position/mcmpc_constants.cuh"
#include "quadcopter_mcmpc_position/mcmpc_controller.cuh"

namespace {
constexpr int NS = _N_OF_ODES, NH = _DEVICE_CONST_HORIZON;
struct Table { std::unordered_map<std::string,size_t> col; std::vector<std::vector<float>> row; };
struct Options { std::string csv, output="mcmpc_log_replay.json"; int start=0,count=-1,stride=1; };
[[noreturn]] void fail(const std::string& s){ std::cerr<<"mcmpc_log_replay: "<<s<<'\n'; std::exit(1); }
void ck(cudaError_t e,const char* s){ if(e!=cudaSuccess) fail(std::string(s)+": "+cudaGetErrorString(e)); }
std::vector<std::string> split(const std::string& s){ std::vector<std::string> v; std::stringstream x(s); std::string f; while(std::getline(x,f,','))v.push_back(f); return v; }
Table read_csv(const std::string& p){
  std::ifstream in(p); if(!in) fail("cannot open "+p); std::string line; if(!std::getline(in,line))fail("empty CSV");
  Table t; auto h=split(line); for(size_t i=0;i<h.size();++i)t.col[h[i]]=i;
  while(std::getline(in,line)){ if(line.empty())continue; auto f=split(line); std::vector<float> r(h.size(),NAN);
    for(size_t i=0;i<std::min(f.size(),r.size());++i){char* e=nullptr;r[i]=std::strtof(f[i].c_str(),&e);if(e==f[i].c_str())r[i]=NAN;} t.row.push_back(std::move(r)); }
  return t;
}
float get(const Table&t,const std::vector<float>&r,const std::string&n,float d=0){auto i=t.col.find(n);if(i==t.col.end()||!std::isfinite(r[i->second]))return d;return r[i->second];}
int geti(const Table&t,const std::vector<float>&r,const std::string&n,int d){return(int)std::lround(get(t,r,n,(float)d));}
Options args(int ac,char**av){Options o;for(int i=1;i<ac;++i){std::string a=av[i];auto next=[&](){if(++i>=ac)fail("value required after "+a);return std::string(av[i]);};
  if(a=="--csv")o.csv=next();else if(a=="--output")o.output=next();else if(a=="--start")o.start=std::max(0,std::stoi(next()));else if(a=="--count")o.count=std::stoi(next());else if(a=="--stride")o.stride=std::max(1,std::stoi(next()));else if(a=="--help"){std::cout<<"mcmpc_log_replay --csv LOG [--output JSON] [--start ROW] [--count N] [--stride N]\n";std::exit(0);}else fail("unknown option "+a);}
  if(o.csv.empty())fail("--csv is required");return o;}
void required(const Table&t){for(auto n:{"t","cur_e0","cur_e1","cur_e2","cur_e3","cur_wx","cur_wy","cur_wz","cur_x","cur_y","cur_z","cur_vx","cur_vy","cur_vz","u0_x","u69_yaw"})if(!t.col.count(n))fail(std::string("missing column ")+n);}

void context(const Table&t,size_t k,const float s[NS]){
  const auto&r=t.row[k];const auto&p=t.row[k?k-1:k];
  // quad_mcmpc_position refreshes prev_velocity/prev_angular_velocity from
  // the current measured state immediately before the model rollout.
  float pv[3]={s[10],s[11],s[12]};
  // Match quad_mcmpc_position: at the start of cycle k, prev_acceleration
  // is the measured acc_now from cycle k-1 (not the model nominal accel).
  float pa[3]={get(t,p,"model_acc_measured_x",get(t,p,"model_acc_nominal_x")),get(t,p,"model_acc_measured_y",get(t,p,"model_acc_nominal_y")),get(t,p,"model_acc_measured_z",get(t,p,"model_acc_nominal_z"))};
  // Prefer the exact replay snapshot recorded by the vehicle logger.
  // Older logs contain an invalid snapshot (large sentinel values); reject it.
  float rax=get(t,r,"replay_prev_acc_x",NAN), ray=get(t,r,"replay_prev_acc_y",NAN), raz=get(t,r,"replay_prev_acc_z",NAN);
  if(std::isfinite(rax)&&std::isfinite(ray)&&std::isfinite(raz)&&fabsf(rax)<100.0f&&fabsf(ray)<100.0f&&fabsf(raz)<100.0f){pa[0]=rax;pa[1]=ray;pa[2]=raz;}
  float pw[3]={s[4],s[5],s[6]};
  // The flight loop writes angular_accel_for_model immediately before the
  // next model call; use the current row, not the previous row.
  float aa[3]={get(t,r,"angular_accel_x"),get(t,r,"angular_accel_y"),get(t,r,"angular_accel_z")};
  if(k==0) aa[0]=aa[1]=aa[2]=0.0f;
  float vi[3]={get(t,r,"model_vel_int_x"),get(t,r,"model_vel_int_y"),get(t,r,"model_vel_int_z")};
  float ri[3]={get(t,r,"rollspeed_integ"),get(t,r,"pitchspeed_integ"),get(t,r,"yawspeed_integ")};
  float ab[3]={get(t,r,"model_acc_bias_x"),get(t,r,"model_acc_bias_y"),get(t,r,"model_acc_bias_z")};
  float ms[4];int mv=1;for(int i=0;i<4;++i){auto n="model_motor_speed_"+std::to_string(i);auto c=t.col.find(n);ms[i]=c==t.col.end()?NAN:r[c->second];if(!std::isfinite(ms[i])){ms[i]=0;mv=0;}}
  float hover=get(t,r,"hover_thrust",CONST_PARAM_FLOAT::MPC_THR_HOVER),tilt=get(t,r,"takeoff_tilt_limit",CONST_PARAM_FLOAT::MPC_TILT_MAX);
  int takeoff=geti(t,r,"takeoff_state",CONST_PARAM_FLOAT::TAKEOFF_STATE_FLIGHT),landed=geti(t,r,"landed",0),ground=geti(t,r,"ground_contact",0),maybe=geti(t,r,"maybe_landed",0);
  // The simulator's controller uses target_state_device on every rollout.
  // It must be synchronized to the target recorded at this log instant;
  // leaving the constructor's initial target here produces an unrelated
  // trajectory even when the logged input sequence is correct.
  qc_mcmpc::target_state_t target{};
  target.e0=get(t,r,"target_state_e0",1.0f); target.e1=get(t,r,"target_state_e1",0.0f); target.e2=get(t,r,"target_state_e2",0.0f); target.e3=get(t,r,"target_state_e3",0.0f);
  target.wx=get(t,r,"target_state_wx",0.0f); target.wy=get(t,r,"target_state_wy",0.0f); target.wz=get(t,r,"target_state_wz",0.0f);
  target.x=get(t,r,"target_state_x",get(t,r,"target_x",s[7])); target.y=get(t,r,"target_state_y",get(t,r,"target_y",s[8])); target.z=get(t,r,"target_state_z",get(t,r,"target_z",s[9]));
  target.xp=get(t,r,"target_state_vx",0.0f); target.yp=get(t,r,"target_state_vy",0.0f); target.zp=get(t,r,"target_state_vz",0.0f);
  ck(cudaMemcpyToSymbol(qc_mcmpc::target_state_device,&target,sizeof(target)),"target state");
  // Reproduce the runtime square-path constants used by quad_mcmpc_position.
  // The CSV target identifies the active waypoint; selecting it also makes
  // the path preview in mpc_simulator.cu deterministic.
  static bool path_initialized=false;
  if(!path_initialized){
    float wp[_SQUARE_WAYPOINTS][3] = {};
    int n=0; std::ifstream wi("../quadcopter_mcmpc_position/config/waypoints.csv");
    if(!wi) wi.open("/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_expt/quadcopter_mcmpc_position/config/waypoints.csv");
    std::string line; if(wi){ while(n<_SQUARE_WAYPOINTS && std::getline(wi,line)){ std::stringstream ss(line); std::string a,b,c; if(!std::getline(ss,a,',')||!std::getline(ss,b,',')||!std::getline(ss,c,',')) continue; try { size_t ea=0,eb=0,ec=0; float ax=std::stof(a,&ea), by=std::stof(b,&eb), cz=std::stof(c,&ec); if(ea==0||eb==0||ec==0) continue; wp[n][0]=ax;wp[n][1]=by;wp[n][2]=cz; ++n; } catch(const std::exception&) { continue; } }}
    if(n<2){n=4; float d[4][3]={{0,0,-1},{1.2f,0,-1},{0.0f,0.7f,-1},{0,0,-1}}; std::copy(&d[0][0],&d[0][0]+12,&wp[0][0]);}
    float threshold=0.05f; int index=1;
    ck(cudaMemcpyToSymbol(qc_mcmpc::square_waypoints_device,wp,sizeof(wp)),"waypoints");
    ck(cudaMemcpyToSymbol(qc_mcmpc::square_waypoint_count_device,&n,sizeof(n)),"waypoint count");
    ck(cudaMemcpyToSymbol(qc_mcmpc::square_waypoint_threshold_device,&threshold,sizeof(threshold)),"waypoint threshold");
    ck(cudaMemcpyToSymbol(qc_mcmpc::square_waypoint_index_device,&index,sizeof(index)),"waypoint index");
    path_initialized=true;
  }
  int logged_wp=geti(t,r,"waypoint_index",1);
  float logged_change=get(t,r,"waypoint_change_time",0.0f);
  ck(cudaMemcpyToSymbol(qc_mcmpc::square_waypoint_index_device,&logged_wp,sizeof(logged_wp)),"logged waypoint index");
  ck(cudaMemcpyToSymbol(qc_mcmpc::square_waypoint_change_time_device,&logged_change,sizeof(logged_change)),"logged waypoint time");
  ck(cudaMemcpyToSymbol(qc_mcmpc::var_and_z_i_device,s,NS*sizeof(float)),"state");
  ck(cudaMemcpyToSymbol(qc_mcmpc::prev_velocity_device,pv,sizeof(pv)),"previous velocity");
  ck(cudaMemcpyToSymbol(qc_mcmpc::prev_acceleration_device,pa,sizeof(pa)),"previous acceleration");
  ck(cudaMemcpyToSymbol(qc_mcmpc::prev_angular_velocity_device,pw,sizeof(pw)),"previous angular velocity");
  ck(cudaMemcpyToSymbol(qc_mcmpc::prev_angular_acceleration_device,aa,sizeof(aa)),"angular acceleration");
  ck(cudaMemcpyToSymbol(qc_mcmpc::vel_int_device,vi,sizeof(vi)),"velocity integral");
  ck(cudaMemcpyToSymbol(qc_mcmpc::rate_int_device,ri,sizeof(ri)),"rate integral");
  ck(cudaMemcpyToSymbol(qc_mcmpc::prev_motor_speed_device,ms,sizeof(ms)),"motor speed");
  ck(cudaMemcpyToSymbol(qc_mcmpc::prev_motor_speed_valid_device,&mv,sizeof(mv)),"motor valid");
  ck(cudaMemcpyToSymbol(qc_mcmpc::acceleration_bias_device,ab,sizeof(ab)),"acceleration bias");
  ck(cudaMemcpyToSymbol(qc_mcmpc::mpc_thr_hover,&hover,sizeof(hover)),"hover thrust");
  ck(cudaMemcpyToSymbol(qc_mcmpc::takeoff_state_device,&takeoff,sizeof(takeoff)),"takeoff state");
  ck(cudaMemcpyToSymbol(qc_mcmpc::takeoff_tilt_limit_device,&tilt,sizeof(tilt)),"tilt limit");
  ck(cudaMemcpyToSymbol(qc_mcmpc::landed_device,&landed,sizeof(landed)),"landed");
  ck(cudaMemcpyToSymbol(qc_mcmpc::ground_contact_device,&ground,sizeof(ground)),"ground contact");
  ck(cudaMemcpyToSymbol(qc_mcmpc::maybe_landed_device,&maybe,sizeof(maybe)),"maybe landed");
}
void num(std::ostream&o,float x){if(std::isfinite(x))o<<x;else o<<"null";}
void sample(std::ostream&o,const Table&t,size_t k,const float s[NS],const qc_mcmpc::input_array&u,const float pred[NH][NS],bool first){
  if(!first)o<<",\n";const auto&r=t.row[k];static const char*n[NS]={"q0","q1","q2","q3","wx","wy","wz","x","y","z","vx","vy","vz"};
  o<<"    {\"source_row\":"<<k<<",\"t\":";num(o,get(t,r,"t"));for(int i=0;i<NS;++i)o<<",\""<<n[i]<<"\":"<<s[i];
  o<<",\"ux\":"<<u.decoupled_position[0][x]<<",\"uy\":"<<u.decoupled_position[0][y]<<",\"uz\":"<<u.decoupled_position[0][z]<<",\"uyaw\":"<<u.decoupled_position[0][yaw];
  o<<",\"target_x\":";num(o,get(t,r,"target_x",s[7]));o<<",\"target_y\":";num(o,get(t,r,"target_y",s[8]));o<<",\"target_z\":";num(o,get(t,r,"target_z",s[9]));
  o<<",\"prediction\":[";for(int h=0;h<NH;++h){if(h)o<<',';o<<'[';for(int j=0;j<NS;++j){if(j)o<<',';o<<pred[h][j];}o<<']';}
  o<<"],\"input_prediction\":[";for(int h=0;h<NH;++h){if(h)o<<',';o<<'['<<u.decoupled_position[h][x]<<','<<u.decoupled_position[h][y]<<','<<u.decoupled_position[h][z]<<','<<u.decoupled_position[h][yaw]<<']';}o<<"]}";
}
}
int main(int ac,char**av){
  auto o=args(ac,av);auto t=read_csv(o.csv);required(t);if(o.start>=(int)t.row.size())fail("--start is beyond CSV");
  (void)qc_mcmpc::mcmpc_controller::get_instance();std::ofstream out(o.output);if(!out)fail("cannot create "+o.output);out<<std::setprecision(9);
  out<<"{\n  \"format\":\"mcmpc-log-cuda-replay-v1\",\n  \"source_csv\":\""<<o.csv<<"\",\n  \"model\":\"actual mpc_simulator.cu deterministic rollout\",\n  \"dt\":"<<CONST_PARAM_FLOAT::CONTROL_PERIOD<<",\n  \"horizon\":"<<NH<<",\n  \"limitations\":[\"row 0 previous-cycle derivative state is unavailable in the CSV\"],\n  \"waypoints\":[],\n  \"samples\":[\n";
  size_t end=o.count<0?t.row.size():std::min(t.row.size(),(size_t)(o.start+o.count));int written=0;const char*c[NS]={"cur_e0","cur_e1","cur_e2","cur_e3","cur_wx","cur_wy","cur_wz","cur_x","cur_y","cur_z","cur_vx","cur_vy","cur_vz"};
  for(size_t k=o.start;k<end;k+=o.stride){float s[NS];for(int i=0;i<NS;++i)s[i]=get(t,t.row[k],c[i]);
    context(t,k,s);
    // Replay the optimized input sequence recorded by MCMPC (u0..u69).
    // No random sampling or re-optimization is performed here.
    qc_mcmpc::input_array u{};
    for(int h=0;h<NH;++h){auto p="u"+std::to_string(h);u.decoupled_position[h][x]=get(t,t.row[k],p+"_x");u.decoupled_position[h][y]=get(t,t.row[k],p+"_y");u.decoupled_position[h][z]=get(t,t.row[k],p+"_z");u.decoupled_position[h][yaw]=get(t,t.row[k],p+"_yaw");}
    qc_mcmpc::simulate_best_input_trajectory_kernel<<<1,1>>>(u);ck(cudaGetLastError(),"launch rollout");ck(cudaDeviceSynchronize(),"sync rollout");float pred[NH][NS];ck(cudaMemcpyFromSymbol(pred,qc_mcmpc::deterministic_sim_trajectory_device,sizeof(pred)),"read prediction");
    float tq[NH][3], ms[NH][4], mm[NH][4], pa[NH][3], aa[NH][3], rsp[NH][3];
    ck(cudaMemcpyFromSymbol(tq,qc_mcmpc::deterministic_sim_torque_setpoint_device,sizeof(tq)),"read torque");
    ck(cudaMemcpyFromSymbol(mm,qc_mcmpc::deterministic_sim_motor_setpoint_device,sizeof(mm)),"read motor setpoint");
    ck(cudaMemcpyFromSymbol(ms,qc_mcmpc::deterministic_sim_motor_speed_device,sizeof(ms)),"read motor speed");
    ck(cudaMemcpyFromSymbol(pa,qc_mcmpc::deterministic_sim_prev_acceleration_device,sizeof(pa)),"read prev acc");
    ck(cudaMemcpyFromSymbol(aa,qc_mcmpc::deterministic_sim_prev_angular_acceleration_device,sizeof(aa)),"read prev angular acc");
    ck(cudaMemcpyFromSymbol(rsp,qc_mcmpc::deterministic_sim_omega_setpoint_device,sizeof(rsp)),"read rate setpoint");
    if(k==o.start){std::cerr<<"input_prev_acc="<<get(t,t.row[k],"model_acc_measured_x")<<","<<get(t,t.row[k],"model_acc_measured_y")<<","<<get(t,t.row[k],"model_acc_measured_z")<<" input_prev_ang_acc="<<get(t,t.row[k],"angular_accel_x")<<","<<get(t,t.row[k],"angular_accel_y")<<","<<get(t,t.row[k],"angular_accel_z")<<" deterministic_prev_acc="<<pa[0][0]<<","<<pa[0][1]<<","<<pa[0][2]<<" deterministic_prev_ang_acc="<<aa[0][0]<<","<<aa[0][1]<<","<<aa[0][2]<<" omega_setpoint="<<rsp[0][0]<<","<<rsp[0][1]<<","<<rsp[0][2]<<"\n";}
    if(k==o.start){float vi[3],ri[3],po[3],pao[3]; qc_mcmpc::target_state_t ts; ck(cudaMemcpyFromSymbol(vi,qc_mcmpc::vel_int_device,sizeof(vi)),"read vel int"); ck(cudaMemcpyFromSymbol(ri,qc_mcmpc::rate_int_device,sizeof(ri)),"read rate int"); ck(cudaMemcpyFromSymbol(po,qc_mcmpc::prev_angular_velocity_device,sizeof(po)),"read prev omega"); ck(cudaMemcpyFromSymbol(pao,qc_mcmpc::prev_angular_acceleration_device,sizeof(pao)),"read prev omega dot"); ck(cudaMemcpyFromSymbol(&ts,qc_mcmpc::target_state_device,sizeof(ts)),"read target"); std::cerr<<"CUDA h0 vel_int="<<vi[0]<<","<<vi[1]<<","<<vi[2]<<" rate_int="<<ri[0]<<","<<ri[1]<<","<<ri[2]<<" prev_omega="<<po[0]<<","<<po[1]<<","<<po[2]<<" target_q="<<ts.e0<<","<<ts.e1<<","<<ts.e2<<","<<ts.e3<<" u0="<<u.decoupled_position[0][x]<<","<<u.decoupled_position[0][y]<<","<<u.decoupled_position[0][z]<<","<<u.decoupled_position[0][yaw]<<" rate_sp="<<aa[0][0]<<","<<aa[0][1]<<","<<aa[0][2]<<"\n";}
    sample(out,t,k,s,u,pred,written==0);++written;}
  out<<"\n  ]\n}\n";std::cout<<"wrote "<<written<<" predictions to "<<o.output<<'\n';
}
