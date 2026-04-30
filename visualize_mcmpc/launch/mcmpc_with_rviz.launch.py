from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    logdir_arg = DeclareLaunchArgument('logdir', default_value='/home/ros2/quadcopter_mcmpc_for_cuda/outputs', description='Path to logs dir')
    rate_arg  = DeclareLaunchArgument('rate',  default_value='1.0', description='Playback speed')

    pos_arg   = DeclareLaunchArgument('pos_file', default_value='/home/ros2/quadcopter_mcmpc_for_cuda/outputs/quadcopter_mcmpc.3', description='Path to position file (.3)')
    att_arg   = DeclareLaunchArgument('att_file', default_value='/home/ros2/quadcopter_mcmpc_for_cuda/outputs/quadcopter_mcmpc.2', description='Path to attitude file (.2)')

    pkg_share = get_package_share_directory('visualize_mcmpc')
    rviz_cfg  = os.path.join(pkg_share, 'rviz', 'mcmpc.rviz')

    logdir = LaunchConfiguration('logdir')
    rate   = LaunchConfiguration('rate')
    posf   = LaunchConfiguration('pos_file')
    attf   = LaunchConfiguration('att_file')

    player = Node(
        package='visualize_mcmpc',
        executable='mcmpc_replay.py',
        name='mcmpc_replay',
        output='screen',
        arguments=[
            '--logdir', logdir,
            '--rate',   rate,
            '--pos_file', posf,
            '--att_file', attf,
        ],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_cfg]
    )

    return LaunchDescription([logdir_arg, rate_arg, pos_arg, att_arg, player, rviz])
