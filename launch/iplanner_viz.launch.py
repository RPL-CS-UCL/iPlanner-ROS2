from launch import LaunchDescription
from launch_ros.actions import Node
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
            executable='iplanner_viz',
            name='iplanner_viz',
            output='screen',
            parameters=[config_file, {'use_sim_time': True}],
            remappings=[
                # Remap if necessary, but config file already has correct topic names
                # ('/rgbd_camera/depth/image', '/camera/depth/image_raw') 
            ]
        )
    ])
