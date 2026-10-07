#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    limo_share = get_package_share_directory("limo_car")
    gazebo_share = get_package_share_directory("gazebo_ros")
    pp_share = get_package_share_directory("pure_pursuit")
    world = os.path.join(limo_share, "worlds", "roboracer", "ifac_roboracer.world")
    pp_params = os.path.join(pp_share, "config", "params.yaml")

    xacro_file = os.path.join(THIS_DIR, "wrapper_izero.xacro")
    robot_description_config = xacro.process_file(xacro_file)
    params = {"robot_description": robot_description_config.toxml(), "use_sim_time": True}

    spawn_x = LaunchConfiguration("spawn_x")
    spawn_y = LaunchConfiguration("spawn_y")
    spawn_yaw = LaunchConfiguration("spawn_yaw")

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        arguments=["--ros-args", "--log-level", "error"],
        output="screen",
        parameters=[params],
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_share, "launch", "gazebo.launch.py")),
        launch_arguments={"world": world, "verbose": "false", "gui": "false"}.items(),
    )

    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "limo",
            "-x", spawn_x, "-y", spawn_y, "-z", "0.15", "-Y", spawn_yaw,
            "-timeout", "60",
            "--ros-args", "--log-level", "error",
        ],
        output="screen",
    )

    adapter = Node(
        package="limo_car",
        executable="ackermann_twist_adapter.py",
        output="screen",
    )

    pp_support = ExecuteProcess(
        cmd=["python3", os.path.join(THIS_DIR, "pp_support.py"),
             "--ros-args", "-p", "use_sim_time:=true"],
        output="screen",
    )

    pure_pursuit_node = Node(
        package="pure_pursuit",
        executable="pure_pursuit_node",
        name="pure_pursuit_node",
        parameters=[pp_params, {"use_fused_odometry": False, "use_sim_time": True}],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("spawn_x", default_value="-0.2"),
        DeclareLaunchArgument("spawn_y", default_value="0.80"),
        DeclareLaunchArgument("spawn_yaw", default_value="0.3491"),
        pure_pursuit_node,
        TimerAction(period=8.0, actions=[pp_support]),
        TimerAction(period=16.0, actions=[robot_state_publisher, gazebo]),
        TimerAction(period=31.0, actions=[spawn, adapter]),
    ])
