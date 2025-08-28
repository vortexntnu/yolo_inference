#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

ALLOWED_DEVICES = ['cpu', '0']


def validate_device(device: str):
    if device not in ALLOWED_DEVICES:
        raise RuntimeError(
            f"Invalid device '{device}'. Choose one of: {', '.join(ALLOWED_DEVICES)}"
        )


def launch_setup(context, *args, **kwargs):
    device = LaunchConfiguration('device').perform(context)
    validate_device(device)

    yolo_params = os.path.join(
        get_package_share_directory('yolo_inference'),
        'config/yolo_inference_params.yaml',
    )

    yolo_node = Node(
        package='yolo_inference',
        executable='yolo_inference_node.py',
        name='yolo_inference',
        namespace='yolo',
        output='screen',
        parameters=[
            yolo_params,
            {'device': device},
        ],
    )

    return [yolo_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'device',
                default_value='0',
                description='Device to run YOLO inference on (\'0\' for GPU or \'cpu\')',
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
