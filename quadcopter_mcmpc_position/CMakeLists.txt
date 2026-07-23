cmake_minimum_required(VERSION 3.8)
project(quadcopter_mcmpc_position LANGUAGES C CXX CUDA)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# PCなら61, Jetson Orin Nanoなら87
set(CMAKE_CUDA_ARCHITECTURES 61 87)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(CUDA REQUIRED)

include_directories(include)

add_library(mcmpc_core SHARED
  src/mcmpc_controller.cu
  src/const_params.cu
)

set_target_properties(mcmpc_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(mcmpc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(mcmpc_core
  ${CUDA_LIBRARIES}
  curand
)

ament_target_dependencies(mcmpc_core
  rclcpp
  px4_msgs
  sensor_msgs
  std_msgs
)

add_executable(quad_mcmpc_node
  src/quad_mcmpc_position.cu
)

add_executable(mcmpc_standalone_simulator
  src/mcmpc_standalone_simulator.cu
)

add_executable(mcmpc_truth_simulator
  src/mcmpc_truth_simulator.cu
)

add_executable(mcmpc_sim_joy_starter
  src/mcmpc_sim_joy_starter.cpp
)

set_target_properties(quad_mcmpc_node PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(mcmpc_standalone_simulator PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(mcmpc_truth_simulator PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(quad_mcmpc_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(mcmpc_standalone_simulator PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(mcmpc_truth_simulator PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(quad_mcmpc_node
  mcmpc_core
)

target_link_libraries(mcmpc_standalone_simulator
  mcmpc_core
)

target_link_libraries(mcmpc_truth_simulator
  mcmpc_core
)

ament_target_dependencies(mcmpc_sim_joy_starter
  rclcpp
  sensor_msgs
)

ament_target_dependencies(quad_mcmpc_node
  rclcpp
  px4_msgs
  sensor_msgs
  std_msgs
)

ament_target_dependencies(mcmpc_truth_simulator
  rclcpp
  px4_msgs
  sensor_msgs
  std_msgs
)

install(TARGETS
  quad_mcmpc_node
  mcmpc_standalone_simulator
  mcmpc_truth_simulator
  mcmpc_sim_joy_starter
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
  mcmpc_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
