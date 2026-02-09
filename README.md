# iPlanner ROS 2

This is a ROS 2 port of the [iPlanner](https://github.com/leggedrobotics/iPlanner#) package, integrated into the Autonomous Exploration Development Environment.

## Overview

iPlanner is a deep learning-based path planner that generates local paths from depth images. This package migrates the original ROS 1 functionality to ROS 2 Humble.

## Dependencies

- **ROS 2 Humble**
- **Python 3.10+**
- **PyTorch** (tested with 2.5.1+cu124)
- **Torchvision**
- **NumPy** (<2.0)
- **OpenCV**
- **cv_bridge**

## Installation

1.  Clone specific branch (if applicable) or copy to your `src` folder.
    > **Note**: This repository uses Git LFS for large model files. Ensure you have `git-lfs` installed:
    > ```bash
    > sudo apt-get install git-lfs
    > git lfs install
    > ```

2.  Install dependencies:
    ```bash
    rosdep install --from-paths src --ignore-src -r -y
    ```
3.  Build the package:
    ```bash
    colcon build --symlink-install --packages-select iplanner_ros2
    ```

## Usage

### 1. Start Simulation
First, launch the `vehicle_simulator` with the desired environment:
```bash
ros2 launch vehicle_simulator system_garage.launch
```

### 2. Launch iPlanner
In a separate terminal, source the workspace and launch the planner:
```bash
source install/setup.bash
ros2 launch iplanner_ros2 iplanner.launch.py
```

### 3. Send Goals
You can check `iplanner_node` status via:
```bash
ros2 topic echo /ip_planner_status
```
Send a goal using Rviz "2D Nav Goal" or publish to `/way_point` (PointStamped).

## Topics

-   **Subscribed**:
    -   `/camera/depth/image_raw` (sensor_msgs/Image): Depth image input
    -   `/way_point` (geometry_msgs/PointStamped): Goal point
    -   `/joy` (sensor_msgs/Joy): Joystick input (optional)

-   **Published**:
    -   `/path` (nav_msgs/Path): Generated path
    -   `/path_fear` (nav_msgs/Path): Alternative path when "fear" (obstacle) is detected
    -   `/ip_planner_status` (std_msgs/Int16): Planning status (0: Planning, 1: Success, -1: Fail)

## Known Issues / Notes

-   **Visualization**: The original `iplanner_viz` node has not been ported. Use Rviz to visualize the `/path` and `/camera/depth/image_raw` topics.
-   **Model File**: The pre-trained model `plannernet.pt` is loaded from `src/iplanner_ros2/iplanner_ros2/models/`.

## Original Citation

If you use this work, please cite the [original paper](https://github.com/leggedrobotics/iPlanner#):

```bibtex
@article{yang2023iplanner,
  title={iPlanner: perception-aware path planning for autonomous navigation in complex environments},
  author={Yang, Fan and Zhang, Tingrui and Hutter, Marco},
  journal={arXiv preprint arXiv:2309.02700},
  year={2023}
}
```
