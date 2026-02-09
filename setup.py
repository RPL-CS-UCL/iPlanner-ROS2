from setuptools import find_packages, setup

package_name = 'iplanner_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/iplanner.launch.py', 'launch/iplanner_viz.launch.py']),
        ('share/' + package_name + '/config', ['config/vehicle_sim.yaml']),
        ('share/' + package_name + '/models', ['iplanner_ros2/models/plannernet.pt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jlcucumber',
    maintainer_email='jlcucumber@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'iplanner_node = iplanner_ros2.iplanner_node:main',
        ],
    },
)
