"""Launch the motionforge planner node with its parameter file."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory("motionforge_ros"), "config", "params.yaml"
    )
    return LaunchDescription(
        [
            Node(
                package="motionforge_ros",
                executable="planner_node",
                name="motionforge_planner",
                output="screen",
                parameters=[params],
            )
        ]
    )
