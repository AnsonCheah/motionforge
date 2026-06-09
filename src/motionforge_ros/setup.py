import os
from glob import glob

from setuptools import find_packages, setup

package_name = "motionforge_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anson",
    maintainer_email="anson08035@gmail.com",
    description="Thin ROS2 node layer wrapping the motionforge GPU planning pipeline.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "planner_node = motionforge_ros.planner_node:main",
        ],
    },
)
