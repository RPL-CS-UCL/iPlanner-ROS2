from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare 
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_name = 'iplanner_ros2'
    
    # Locate configuration file
    config_file = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'vehicle_sim.yaml'
    )

    return LaunchDescription([
        Node(
            package='iplanner_ros2',
            executable='iplanner_node',
            name='iplanner_node',
            output='screen',
            parameters=[config_file, {'use_sim_time': True}]
        ),
        IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare('iplanner_path_follower'),
                'launch',
                'path_follower.launch.py'
            ]),
            launch_arguments={
                'odomTopic':'/state_estimation',
                'commandTopic':'/cmd_vel'
            }.items()
        )
    ])
