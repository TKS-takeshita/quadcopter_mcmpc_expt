from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
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
