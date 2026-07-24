from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    waypoint_file = os.path.join(
        get_package_share_directory("quadcopter_mcmpc_position"),
        "config", "waypoints.csv")
    truth_simulator = Node(
        package="quadcopter_mcmpc_position",
        executable="mcmpc_truth_simulator",
        name="mcmpc_truth_simulator",
        output="screen",
    )

    controller = Node(
        package="quadcopter_mcmpc_position",
        executable="quad_mcmpc_node",
        name="quadcopter_mcmpc",
        output="screen",
        parameters=[{"waypoints_file": waypoint_file}],
    )

    starter = Node(
        package="quadcopter_mcmpc_position",
        executable="mcmpc_sim_joy_starter",
        name="mcmpc_sim_joy_starter",
        output="screen",
    )

    return LaunchDescription([
        truth_simulator,
        controller,
        starter,
    ])
