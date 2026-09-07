cmake_minimum_required(VERSION 3.8)
project(visualize_mcmpc LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61 87)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(CUDA REQUIRED)

# Compile the controller's model sources in place: the flight package remains
# untouched and replay cannot silently diverge into a second dynamics model.
set(MCMPC_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../quadcopter_mcmpc_position")
add_executable(mcmpc_log_replay
  src/mcmpc_log_replay.cu
  ${MCMPC_PACKAGE_DIR}/src/mcmpc_controller.cu
  ${MCMPC_PACKAGE_DIR}/src/const_params.cu)
target_compile_definitions(mcmpc_log_replay PRIVATE MCMPC_DISABLE_WALL_MODEL)
target_compile_options(mcmpc_log_replay PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=128>)
target_include_directories(mcmpc_log_replay PRIVATE ${MCMPC_PACKAGE_DIR}/include)
set_target_properties(mcmpc_log_replay PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
ament_target_dependencies(mcmpc_log_replay rclcpp px4_msgs sensor_msgs std_msgs)

install(PROGRAMS
  scripts/mcmpc_viewer.py
  scripts/verify_mcmpc_sim.py
  DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS mcmpc_log_replay DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY simulator DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
