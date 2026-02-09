# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.
# ======================================================================

import os
import PIL
import sys
import torch
import rclpy
from rclpy.node import Node
import rclpy.duration
import time
from std_msgs.msg import Float32, Int16
import numpy as np
from sensor_msgs.msg import Image, Joy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from cv_bridge import CvBridge
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
# Use tf2_geometry_msgs to register geometry_msgs support
import tf2_geometry_msgs

# Add current directory to path so imports work if not installed as package
# Though in ROS 2 ament_python, proper package structure is preferred.
# We assume the layout allows 'iplanner_ros2' imports if structured correctly.
# But original code appended paths.
import ament_index_python.packages

try:
    pack_path = ament_index_python.packages.get_package_share_directory('iplanner_ros2')
except:
    # Fallback if package share is not found (e.g. not installed yet), use current dir
    pack_path = os.path.dirname(os.path.abspath(__file__))

# Adjust source path for importing the algo (assuming iplanner_ros2 package structure)
# The algo files are in the same directory as this script in the new package.
# So we can just import directly if they are in the python path.
from .ip_algo import IPlannerAlgo
from .rosutil import ROSArgparse

class iPlannerNode(Node):
    def __init__(self):
        super().__init__('iplanner_node')
        
        self.bridge = CvBridge()
        
        # Initialize ROSArgparse with self (Node)
        node_name = 'iplanner_node'
        parser = ROSArgparse(node=self, relative=node_name)
        
        # Declare parameters (moved from main)
        # Note: defaults here might be overridden by launch files
        parser.add_argument('main_freq',         type=int,   default=5,                          help="Main frequency of the path planner.")
        parser.add_argument('model_save',        type=str,   default='/models/plannernet.pt',    help="Path to the saved model.")
        # crop_size tuple handling via parameter might need string conversion or array
        # ROS2 params support integer arrays.
        parser.add_argument('crop_size',         type=list, default=[360,640],                  help='Size to crop the image to.') 
        parser.add_argument('uint_type',         type=bool,  default=False,                      help="Determines if the image is in uint type.")
        parser.add_argument('depth_topic',       type=str,   default='/camera/depth/image_raw', help='Topic for depth image.') # Updated default
        parser.add_argument('goal_topic',        type=str,   default='/way_point',               help='Topic for goal waypoints.')
        parser.add_argument('path_topic',        type=str,   default='/path',                    help='Topic for iPlanner path.')
        parser.add_argument('robot_id',          type=str,   default='base',                     help='TF frame ID for the robot.')
        parser.add_argument('world_id',          type=str,   default='odom',                     help='TF frame ID for the world.')
        parser.add_argument('depth_max',         type=float, default=10.0,                       help='Maximum depth distance in the image.')
        parser.add_argument('image_flip',        type=bool,  default=True,                       help='Indicates if the image is flipped.')
        parser.add_argument('conv_dist',         type=float, default=0.5,                        help='Convergence range to the goal.')
        parser.add_argument('is_fear_act',       type=bool,  default=True,                       help='Indicates if fear action is enabled.')
        parser.add_argument('buffer_size',       type=int,   default=10,                         help='Buffer size for fear reaction.')
        parser.add_argument('angular_thred',     type=float, default=0.3,                        help='Angular threshold for turning.')
        parser.add_argument('track_dist',        type=float, default=0.5,                        help='Look-ahead distance for path tracking.')
        parser.add_argument('joyGoal_scale',     type=float, default=0.5,                        help='Scale for joystick goal distance.')
        parser.add_argument('sensor_offset_x',   type=float, default=0.0,                        help='Sensor offset on the X-axis.')
        parser.add_argument('sensor_offset_y',   type=float, default=0.0,                        help='Sensor offset on the Y-axis.')
        
        args = parser.parse_args()
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        candidate_path = base_path + args.model_save
        if os.path.exists(candidate_path):
            args.model_save = candidate_path
        else:
             # Try package share
            try:
                share_path = ament_index_python.packages.get_package_share_directory('iplanner_ros2')
                # Note: models usually need to be installed.
                # If not found, warn.
                args.model_save = os.path.join(share_path, args.model_save.lstrip('/'))
            except:
                pass

        self.config(args)

        # init planner algo class
        self.iplanner_algo = IPlannerAlgo(args=args)
        
        # TF2 Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Wait for tf listener to be ready
        
        self.image_time = self.get_clock().now()
        self.is_goal_init = False
        self.ready_for_planning = False

        # planner status
        self.planner_status = Int16()
        self.planner_status.data = 0
        self.is_goal_processed = False
        self.is_smartjoy = False

        # fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False
        # process time
        self.timer_data = Float32()
        
        self.create_subscription(Image, self.image_topic, self.imageCallback, 10)
        self.create_subscription(PointStamped, self.goal_topic, self.goalCallback, 10)
        self.create_subscription(Joy, "/joy", self.joyCallback, 10)

        timer_topic = '/ip_timer'
        status_topic = '/ip_planner_status'
        
        # planning status topics
        self.timer_pub = self.create_publisher(Float32, timer_topic, 10)
        self.status_pub = self.create_publisher(Int16, status_topic, 10)

        self.path_pub  = self.create_publisher(Path, self.path_topic, 10)
        self.fear_path_pub = self.create_publisher(Path, self.path_topic + "_fear", 10)

        self.get_logger().info("iPlanner Ready.")
        
        # Main Timer Loop
        self.create_timer(1.0/self.main_freq, self.timer_callback)

    def config(self, args):
        self.main_freq   = args.main_freq
        self.model_save  = args.model_save
        self.image_topic = args.depth_topic
        self.goal_topic  = args.goal_topic
        self.path_topic  = args.path_topic
        self.frame_id    = args.robot_id
        self.world_id    = args.world_id
        self.uint_type   = args.uint_type
        self.image_flip  = args.image_flip
        self.conv_dist   = args.conv_dist
        self.depth_max   = args.depth_max
        # fear reaction
        self.is_fear_act = args.is_fear_act
        self.buffer_size = args.buffer_size
        self.ang_thred   = args.angular_thred
        self.track_dist  = args.track_dist
        self.joyGoal_scale = args.joyGoal_scale
        return 

    def timer_callback(self):
        if self.ready_for_planning and self.is_goal_init:
            # main planning starts
            # Need to ensure self.img exists
            if not hasattr(self, 'img'):
                return

            cur_image = self.img.copy()
            start = time.time()

            # Network Planning (Model inference)
            self.preds, self.waypoints, fear_output, _ = self.iplanner_algo.plan(cur_image, self.goal_rb)
            end = time.time()
            self.timer_data.data = (end - start) * 1000
            self.timer_pub.publish(self.timer_data)

            # check goal less than converage range
            if (np.sqrt(self.goal_rb[0][0]**2 + self.goal_rb[0][1]**2) < self.conv_dist) and self.is_goal_processed and (not self.is_smartjoy):
                self.ready_for_planning = False
                self.is_goal_init = False

                # planner status -> Success
                if self.planner_status.data == 0:
                    self.planner_status.data = 1
                    self.status_pub.publish(self.planner_status)

                self.get_logger().info("Goal Arrived")
            
            # Reconstruct fear tensor on the correct device if needed
            self.fear = torch.tensor([[0.0]], device=fear_output.device)
            if self.is_fear_act:
                self.fear = fear_output
                is_track_ahead = self.isForwardTraking(self.waypoints)
                self.fearPathDetection(self.fear, is_track_ahead)
                if self.is_fear_reaction:
                    # logwarn_throttle replacement - manual implementation or just log for now
                    # self.get_logger().warn("current path prediction is invaild.") # Throttle logic omitted for brevity
                    pass
                    
                    # planner status -> Fails
                    if self.planner_status.data == 0:
                        self.planner_status.data = -1
                        self.status_pub.publish(self.planner_status)
            self.pubPath(self.waypoints, self.is_goal_init)

    def pubPath(self, waypoints, is_goal_init=True):
        path = Path()
        fear_path = Path()
        if is_goal_init:
            for p in waypoints.squeeze(0):
                pose = PoseStamped()
                pose.pose.position.x = float(p[0])
                pose.pose.position.y = float(p[1])
                pose.pose.position.z = float(p[2])
                path.poses.append(pose)
        
        # header
        path.header.frame_id = self.frame_id
        fear_path.header.frame_id = self.frame_id
        path.header.stamp = self.image_time.to_msg() # Convert Time object to msg
        fear_path.header.stamp = self.image_time.to_msg()
        
        # publish fear path
        if self.is_fear_reaction:
            fear_path.poses = list(path.poses) # Copy
            path.poses = path.poses[:1]
        # publish path
        self.fear_path_pub.publish(fear_path)
        self.path_pub.publish(path)
        return

    def fearPathDetection(self, fear, is_forward):
        if fear > 0.5 and is_forward:
            if not self.is_fear_reaction:
                self.fear_buffter = self.fear_buffter + 1
        elif self.is_fear_reaction:
            self.fear_buffter = self.fear_buffter - 1
        if self.fear_buffter > self.buffer_size:
            self.is_fear_reaction = True
        elif self.fear_buffter <= 0:
            self.is_fear_reaction = False
        return None

    def isForwardTraking(self, waypoints):
        xhead = np.array([1.0, 0])
        phead = None
        for p in waypoints.squeeze(0):
            if torch.norm(p[0:2]).item() > self.track_dist:
                phead = np.array([p[0].item(), p[1].item()])
                phead /= np.linalg.norm(phead)
                break
        if phead is None or phead.dot(xhead) > 1.0 - self.ang_thred:
            return True
        return False

    def joyCallback(self, joy_msg):
        if joy_msg.buttons[4] > 0.9:
            self.get_logger().info("Switch to Smart Joystick mode ...")
            self.is_smartjoy = True
            # reset fear reaction
            self.fear_buffter = 0
            self.is_fear_reaction = False
        if self.is_smartjoy:
            if np.sqrt(joy_msg.axes[3]**2 + joy_msg.axes[4]**2) < 1e-3:
                # reset fear reaction
                self.fear_buffter = 0
                self.is_fear_reaction = False
                self.ready_for_planning = False
                self.is_goal_init = False
            else:
                joy_goal = PointStamped()
                joy_goal.header.frame_id = self.frame_id
                joy_goal.point.x = joy_msg.axes[4] * self.joyGoal_scale
                joy_goal.point.y = joy_msg.axes[3] * self.joyGoal_scale
                joy_goal.point.z = 0.0
                joy_goal.header.stamp = self.get_clock().now().to_msg()
                self.goal_pose = joy_goal
                self.is_goal_init = True
                self.is_goal_processed = False
        return

    def goalCallback(self, msg):
        self.get_logger().info("Recevied a new goal")
        self.goal_pose = msg
        self.is_smartjoy = False
        self.is_goal_init = True
        self.is_goal_processed = False
        # reset fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False
        # reste planner status
        self.planner_status.data = 0
        return

    def imageCallback(self, msg):
        self.image_time = rclpy.time.Time.from_msg(msg.header.stamp)
        # If ros_numpy is not available, cv_bridge is the alternative.
        try:
             frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
             self.get_logger().error(f'cv_bridge exception: {e}')
             return
        frame[~np.isfinite(frame)] = 0
        if self.uint_type:
            frame = frame / 1000.0
        frame[frame > self.depth_max] = 0.0
        
        if self.image_flip:
            frame = PIL.Image.fromarray(frame)
            self.img = np.array(frame.transpose(PIL.Image.ROTATE_180))
        else:
            self.img = frame

        if self.is_goal_init:
            goal_robot_frame = self.goal_pose
            if not self.goal_pose.header.frame_id == self.frame_id:
                try:
                    # In ROS 2, transform is different.
                    # We need to look up transform from goal_frame to robot_frame
                    # and then transform the point.
                    
                    # We use buffer.transform(msg, target_frame)
                    # Note: We need tf2_geometry_msgs imported for this to work on PointStamped.
                    
                    # Check for latest common time
                    # In tf2_ros, we often just ask for the transform at the time of the message
                    # or latest if time is 0.
                    
                    # Timeout is needed.
                    timeout = rclpy.duration.Duration(seconds=0.1)
                    
                    # Look up latest transform from map (or goal frame) to robot
                    # We use Time() (sim time 0) to get the latest available transform
                    # This assumes the goal is static in the goal frame (e.g. map)
                    
                    trans = self.tf_buffer.lookup_transform(
                        self.frame_id,
                        self.goal_pose.header.frame_id,
                        rclpy.time.Time()
                    )
                    
                    # Apply transform
                    goal_robot_frame = tf2_geometry_msgs.do_transform_point(self.goal_pose, trans)
                    
                except TransformException as ex:
                    # Log throttling could be useful here
                    self.get_logger().error(f'Fail to transfer the goal into base frame: {ex}')
                    return
            
            # goal_robot_frame is PointStamped
            goal_robot_frame = torch.tensor([goal_robot_frame.point.x, goal_robot_frame.point.y, goal_robot_frame.point.z], dtype=torch.float32)[None, ...]
            self.goal_rb = goal_robot_frame
        else:
            return
        self.ready_for_planning = True
        self.is_goal_processed  = True
        return

def main(args=None):
    rclpy.init(args=args)
    node = iPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
