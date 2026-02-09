# ==============================================================================
# Copyright <2019> <Chen Wang [https://chenwang.site], Carnegie Mellon University>
# Refer to: https://github.com/wang-chen/interestingness_ros/blob/master/script/rosutil.py
# ==============================================================================

import os
import rclpy
from rclpy.node import Node
import torch
import numpy as np

class ROSArgparse():
    def __init__(self, node, relative=None):
        self.node = node
        self.relative = relative

    def add_argument(self, name, default, type=None, help=None):
        # In ROS 2, we declare parameters.
        # Check if parameter is already declared to avoid errors/warnings or just declare it
        if not self.node.has_parameter(name):
             self.node.declare_parameter(name, default)
        
        # Get parameter
        try:
            value = self.node.get_parameter(name).value
            self.node.get_logger().info(f'Get param {name}: {value}')
        except Exception as e:
            self.node.get_logger().warn(f'Couldn\'t find param: {name}, Using default: {default}')
            value = default

        # Clean up variable name
        variable = name[name.rfind('/')+1:].replace('-','_')
        
        # Set attribute on self
        setattr(self, variable, value)

    def parse_args(self):
        return self


def msg_to_torch(data, shape=np.array([-1])):
    return torch.from_numpy(data).view(shape.tolist())


def torch_to_msg(tensor):
    return [tensor.view(-1).cpu().numpy(), tensor.shape]
