#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription

yolo_params = os.path.join(
    get_package_share_directory('yolo_inference'),
    'config/yolo_inference_params.yaml',
)


def generate_launch_description() -> LaunchDescription:
    """Generates a launch description for the yolo_inference node.

    This function creates a ROS 2 launch description that includes the
    yolo_inference node. The node is configured to use the
    parameters specified in the 'param_yolo_inference.yaml' file.

    Returns:
        LaunchDescription: A ROS 2 launch description containing the
        yolo_inference node.

    """
    yolo_inference_node = Node(
        package='yolo_inference',
        executable='yolo_inference_node.py',
        name='yolo_inference',
        namespace='yolo',
        output='screen',
        parameters=[yolo_params],
    )

    return LaunchDescription([yolo_inference_node])
